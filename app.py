
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
from datetime import timedelta

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Aedes Alert",
    page_icon="🦟",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

FEATURES = [
    "rain_lag_1", "rain_lag_2",
    "rain_lag_3", "rain_lag_4",
    "rain_roll_2", "rain_roll_3",
    "rain_roll_4",
    "temp_roll_2", "temp_roll_4",
    "humidity_roll_2", "humidity_roll_4",
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "month", "week_of_year"
]

# =========================
# LOAD DATA AND MODELS
# =========================

@st.cache_data
def load_data():
    path = DATA_DIR / "ward_weather_features.csv"
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    return data


@st.cache_data
def load_raw_data():
    path = DATA_DIR / "hackathon_preli_dataset.xlsx"
    raw = pd.read_excel(path)
    raw["date"] = pd.to_datetime(raw["date"])
    return raw


@st.cache_resource
def load_models():
    return {
        2: joblib.load(MODEL_DIR / "dengue_2w.pkl"),
        3: joblib.load(MODEL_DIR / "dengue_3w.pkl"),
        4: joblib.load(MODEL_DIR / "dengue_4w.pkl")
    }


try:
    df = load_data()
    raw_df = load_raw_data()
    models = load_models()

    missing = [f for f in FEATURES if f not in df.columns]

    if missing:
        st.error(f"Missing model features: {missing}")
        st.stop()

except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()


# =========================
# HEADER
# =========================

st.title("🦟 Aedes Alert")
st.subheader("AI-Powered Dengue Risk Prediction and Early Warning System")





# Ward list used by the future prediction form
wards = sorted(df["ward_id"].unique())


# =====================================================
# FUTURE DENGUE PREDICTION USING LATEST 7-DAY WEATHER
# =====================================================

st.divider()
st.header("🌦️ Future Dengue Risk Prediction")

st.write(
    "Enter the latest weeks weather information. "
    
)

today = pd.Timestamp.now().normalize()

# Align the model reference date to the next Monday.
# The latest 7-day input is treated as the most recently completed week.
days_until_next_monday = 7 - today.weekday()
model_date = today + timedelta(days=days_until_next_monday)

period_start = today - timedelta(days=6)
period_end = today

st.info(
    f"Weather input period: {period_start.date()} to {period_end.date()}  |  "
    f"Date: {model_date.date()}"
)


# =========================
# WEATHER INPUT FORM
# =========================

with st.form("weekly_weather_input_form"):

    st.subheader("Prediction Information")

    selected_future_ward = st.selectbox(
        "Select Ward",
        wards,
        key="weekly_future_ward"
    )

    st.subheader("Latest Weather Information")

    col1, col2, col3 = st.columns(3)

    rainfall_7d = col1.number_input(
        "Total Rainfall (mm)",
        min_value=0.0,
        max_value=2000.0,
        value=0.0,
        step=1.0,
        key="latest_7d_rainfall"
    )

    temperature_7d = col2.number_input(
        "Average Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=28.0,
        step=0.1,
        key="latest_7d_temperature"
    )

    humidity_7d = col3.number_input(
        "Average Relative Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0,
        key="latest_7d_humidity"
    )

    submitted = st.form_submit_button(
        "Generate Dengue Risk Prediction",
        use_container_width=True
    )


# =========================
# RISK CLASSIFICATION
# =========================

def classify_future_risk(cases):
    if cases < 3:
        return "Low"
    elif cases < 6:
        return "Medium"
    elif cases < 10:
        return "High"
    return "Critical"


# =========================
# HISTORICAL WEATHER ESTIMATION
# =========================

def get_seasonal_weather_estimate(ward_id, target_week_date):
    """
    Estimate one missing historical week using the same ward's
    historical observations from the matching ISO week (2016-2025).
    """

    weather_columns = [
        "rainfall_mm",
        "avg_temp_c",
        "relative_humidity_pct"
    ]

    history = raw_df.loc[
        raw_df["ward_id"] == ward_id,
        ["date"] + weather_columns
    ].copy()

    history["iso_week"] = (
        history["date"].dt.isocalendar().week.astype(int)
    )

    target_iso_week = int(
        pd.Timestamp(target_week_date).isocalendar().week
    )

    matching_weeks = history[
        history["iso_week"] == target_iso_week
    ]

    if matching_weeks.empty:
        matching_weeks = history

    estimates = matching_weeks[weather_columns].mean()

    if estimates.isna().any():
        raise ValueError(
            f"Could not estimate historical weather for {ward_id}, "
            f"ISO week {target_iso_week}."
        )

    return estimates


# =========================
# SAVE USER INPUT
# =========================

if submitted:

    st.session_state["weekly_weather_input"] = {
        "ward_id": selected_future_ward,
        "observation_date": today,
        "period_start": period_start,
        "period_end": period_end,
        "rainfall_mm": float(rainfall_7d),
        "avg_temp_c": float(temperature_7d),
        "relative_humidity_pct": float(humidity_7d)
    }


# =========================
# PROCESS USER INPUT
# =========================

if "weekly_weather_input" in st.session_state:

    user_weather = st.session_state["weekly_weather_input"]
    prediction_ward = user_weather["ward_id"]

    st.success("Latest 7-day weather input received successfully!")

    # =========================
    # SUBMITTED WEATHER INFORMATION
    # =========================

    st.subheader("📋 Submitted Weather Information")

    input_cols = st.columns(4)

    input_cols[0].metric(
        "Selected Ward",
        prediction_ward
    )

    input_cols[1].metric(
        "7-Day Rainfall",
        f'{user_weather["rainfall_mm"]:.2f} mm'
    )

    input_cols[2].metric(
        "Average Temperature",
        f'{user_weather["avg_temp_c"]:.1f} °C'
    )

    input_cols[3].metric(
        "Relative Humidity",
        f'{user_weather["relative_humidity_pct"]:.1f}%'
    )

    try:

        # =========================
        # WARD VULNERABILITY VALUES
        # =========================

        ward_raw_history = raw_df[
            raw_df["ward_id"] == prediction_ward
        ].sort_values("date")

        if ward_raw_history.empty:
            raise ValueError(
                "No original historical records found for this ward."
            )

        ward_info = ward_raw_history.iloc[-1]

        # =========================
        # WEATHER LAGS
        # =========================

        # Lag 1 uses user-provided weather.
        # Lags 2-4 use historical seasonal estimates.

        lag_weather = {
            1: {
                "rainfall_mm": user_weather["rainfall_mm"],
                "avg_temp_c": user_weather["avg_temp_c"],
                "relative_humidity_pct": user_weather[
                    "relative_humidity_pct"
                ],
                "source": "User-provided latest 7-day observation"
            }
        }

        for lag in [2, 3, 4]:

            historical_week_date = model_date - timedelta(weeks=lag)

            estimate = get_seasonal_weather_estimate(
                prediction_ward,
                historical_week_date
            )

            lag_weather[lag] = {
                "rainfall_mm": float(estimate["rainfall_mm"]),
                "avg_temp_c": float(estimate["avg_temp_c"]),
                "relative_humidity_pct": float(
                    estimate["relative_humidity_pct"]
                ),
                "source": (
                    "Ward-specific historical seasonal average "
                    f"(ISO week {historical_week_date.isocalendar().week})"
                )
            }

        rain_values = [
            lag_weather[i]["rainfall_mm"]
            for i in [1, 2, 3, 4]
        ]

        temp_values = [
            lag_weather[i]["avg_temp_c"]
            for i in [1, 2, 3, 4]
        ]

        humidity_values = [
            lag_weather[i]["relative_humidity_pct"]
            for i in [1, 2, 3, 4]
        ]

        # =========================
        # GENERATE 16 MODEL FEATURES
        # =========================

        future_features = {

            "rain_lag_1": rain_values[0],
            "rain_lag_2": rain_values[1],
            "rain_lag_3": rain_values[2],
            "rain_lag_4": rain_values[3],

            "rain_roll_2": sum(rain_values[:2]),
            "rain_roll_3": sum(rain_values[:3]),
            "rain_roll_4": sum(rain_values[:4]),

            "temp_roll_2": sum(temp_values[:2]) / 2,
            "temp_roll_4": sum(temp_values) / 4,

            "humidity_roll_2": sum(humidity_values[:2]) / 2,
            "humidity_roll_4": sum(humidity_values) / 4,

            "population_density_sqkm": float(
                ward_info["population_density_sqkm"]
            ),

            "water_logging_risk_index": float(
                ward_info["water_logging_risk_index"]
            ),

            "sanitation_score": float(
                ward_info["sanitation_score"]
            ),

            "month": int(model_date.month),

            "week_of_year": int(
                model_date.isocalendar().week
            )
        }

        X_future = pd.DataFrame(
            [future_features],
            columns=FEATURES
        )

        if X_future.isna().any().any():
            raise ValueError(
                "Generated model features contain missing values."
            )

        # =========================
        # WARD VULNERABILITY INFORMATION
        # =========================

        st.divider()
        st.header("📚 Ward Vulnerability Information")

        v1, v2, v3 = st.columns(3)

        v1.metric(
            "Population Density",
            f'{future_features["population_density_sqkm"]:,.0f}'
        )

        v2.metric(
            "Waterlogging Risk",
            f'{future_features["water_logging_risk_index"]:.2f}'
        )

        v3.metric(
            "Sanitation Score",
            f'{future_features["sanitation_score"]:.2f}'
        )

        # =========================
        # GENERATED MODEL FEATURES
        # =========================

        st.divider()
        st.header("🧮 Generated Model Features")

        feature_table = pd.DataFrame([
            {
                "Feature": f"rain_lag_{lag}",
                "Value": future_features[f"rain_lag_{lag}"],
                "Data Source": lag_weather[lag]["source"]
            }
            for lag in [1, 2, 3, 4]
        ])

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True
        )

        with st.expander("View all 16 model input features"):

            st.dataframe(
                pd.DataFrame([future_features])[FEATURES].T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )

        st.caption(
            "Weather for lags 2-4 is estimated from historical observations "
            "for the same ward and matching ISO weeks. These are estimates, "
            "not actual current weather observations."
        )


        # =========================
        # MAIN WEATHER DRIVERS
        # =========================

        st.divider()
        st.header("🌦️ Main Weather Drivers")

        st.write(
            "The following chart shows which input features "
            "the Random Forest model relies on most."
        )

        # Use the 3-week model for overall feature importance
        importance = models[3].feature_importances_

        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importance
        })

        # Group features into meaningful categories
        def get_driver_group(feature):
            if feature.startswith("rain_"):
                return "Rainfall"
            elif feature.startswith("temp_"):
                return "Temperature"
            elif feature.startswith("humidity_"):
                return "Humidity"
            elif feature == "water_logging_risk_index":
                return "Waterlogging Risk"
            elif feature == "population_density_sqkm":
                return "Population Density"
            elif feature == "sanitation_score":
                return "Sanitation"
            else:
                return "Seasonal Factors"

        importance_df["Driver"] = (
            importance_df["Feature"].apply(get_driver_group)
        )

        driver_summary = (
            importance_df.groupby("Driver")["Importance"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        driver_summary["Importance (%)"] = (
            driver_summary["Importance"] * 100
        ).round(2)

        fig_drivers = px.bar(
            driver_summary,
            x="Importance (%)",
            y="Driver",
            orientation="h",
            title="Random Forest Feature Importance",
            text="Importance (%)"
        )

        st.plotly_chart(
            fig_drivers,
            use_container_width=True
        )

        st.caption(
            "Feature importance represents the model's overall reliance "
            "on input features. It does not establish causation or explain "
            "the exact reason for an individual prediction."
        )


        # =========================
        # FUTURE DENGUE RISK FORECAST
        # =========================

        st.divider()
        st.header("🔮 Future Dengue Risk Forecast")

        results = []

        for weeks_ahead in [2, 3, 4]:

            predicted_cases = max(
                0.0,
                float(
                    models[weeks_ahead].predict(X_future)[0]
                )
            )

            forecast_date = model_date + timedelta(
                weeks=weeks_ahead
            )

            results.append({
                "Forecast Horizon": f"{weeks_ahead} Weeks Ahead",
                "Forecast Week": forecast_date.date(),
                "Predicted Dengue Cases": round(
                    predicted_cases, 2
                ),
                "Risk Level": classify_future_risk(
                    predicted_cases
                )
            })

        result_df = pd.DataFrame(results)

        st.write(
            f"Selected Ward: **{prediction_ward}**"
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        result_cols = st.columns(3)

        for index, result in enumerate(results):

            result_cols[index].metric(
                f'{result["Forecast Horizon"]} Risk',
                result["Risk Level"],
                f'{result["Predicted Dengue Cases"]:.2f} predicted cases'
            )

        # =========================
        # DOWNLOAD FUTURE RISK REPORT
        # =========================

        st.download_button(
            "Download Future Risk Report",
            result_df.to_csv(index=False),
            file_name=f"dengue_future_risk_{prediction_ward}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")


# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "AedesAlert BD | Machine Learning Research Prototype"
)

st.caption(
    "Predictions are experimental decision-support estimates, "
    "not official public health alerts."
)
