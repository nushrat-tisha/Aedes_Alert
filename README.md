# Aedes_Alert

## AI Assistants and External API Disclosure

### 1. AI Assistants Used

The following AI assistant was used during the development
of ADIS Alert:

- **ChatGPT (OpenAI):** Used for Python coding assistance,
  debugging, implementation guidance, problem-solving,
  and technical documentation.

All AI-assisted suggestions incorporated into the project
are subject to review and understanding by the development
team. Team members are responsible for explaining the
code, algorithms, and implementation decisions.

### 2. External APIs Used

The current application does not depend on external
weather or AI inference APIs.

| External API | Usage |
|---|---|
| Weather API | Not used |
| OpenAI API | Not used |
| Other external APIs | None |

### 3. Data Sources and Prediction

- Historical weather information is obtained from local datasets.
- Latest weather observations are entered manually by users.
- Three pre-trained Random Forest models are loaded locally
  using Joblib.
- Dengue risk predictions are generated locally without
  external AI inference services.

### 4. Cost Control

- No paid weather API calls.
- No external LLM token consumption.
- Local model inference using serialized Joblib models.
- Streamlit caching to reduce repeated data and model loading.
AI-Powered Dengue Risk Prediction and Early Warning System
