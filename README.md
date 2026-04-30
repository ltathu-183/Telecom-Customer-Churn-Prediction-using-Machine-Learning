# Telco Churn AI System

Production-ready AI system for predicting customer churn in telecommunications.

## Overview

This project demonstrates a complete end-to-end machine learning system for customer churn prediction, from data exploration to production deployment.

### Key Features

- **Advanced ML Models**: LightGBM, XGBoost, Neural Networks (PyTorch)
- **Time Series Forecasting**: ARIMA, Prophet for churn trend prediction
- **Model Explainability**: SHAP for interpretability
- **Production API**: FastAPI for real-time predictions
- **Interactive Dashboard**: Streamlit for business users
- **MLOps Pipeline**: Airflow for automated workflows
- **Business Impact**: ROI analysis and retention strategies

## Project Structure

```
Telecom-Customer-Churn-Prediction-using-Machine-Learning/
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_eda_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_comparison.ipynb
│   ├── 04_model_explainability_shap.ipynb
│   └── 05_business_impact_analysis.ipynb
├── src/                   # Production code
│   ├── api/              # FastAPI application
│   │   └── main.py
│   ├── data/             # Data preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # ML models
│   ├── services/         # Business logic
│   ├── monitoring/       # Model monitoring
│   ├── foundations/      # Foundation utilities
│   └── path.py           # Path utilities
├── models/               # Trained models
├── dashboard/            # Streamlit dashboard
│   └── app.py
├── data/                 # Data files
│   └── processed/        # Processed datasets
├── dags/                 # Airflow DAGs
├── docker/               # Docker configuration
├── reports/              # Analysis reports
└── README.md             # This file
```

## Quick Start

### Prerequisites

- Python 3.9+
- pip
- (Optional) Docker & Docker Compose

### Installation

```bash
# Clone repository
git clone https://github.com/yourname/telco-churn-ai-system.git
cd telco-churn-ai-system

# Install dependencies
pip install -r requirements.txt
```

## How to Run

> ** Important:** The API requires model files to be generated first. Run the notebooks in order (especially notebook 03) to create the model files before starting the API server.

### 1. Run Jupyter Notebooks (Data Analysis & Model Training)

```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

Then open notebooks in order:
1. `01_eda_exploratory_data_analysis.ipynb` - Explore the data
2. `02_feature_engineering.ipynb` - Feature engineering
3. `03_model_training_comparison.ipynb` - Train and compare models
4. `04_model_explainability_shap.ipynb` - Model interpretability
5. `05_business_impact_analysis.ipynb` - Business ROI analysis

### 2. Run FastAPI Server

```bash
# Navigate to project root and run
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Or with Python directly
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- API Docs: http://localhost:8000/docs
- API Endpoint: http://localhost:8000/predict (POST)
- Health Check: http://localhost:8000/health

> **Note:** If you see "Error loading models" in the logs, run notebook `03_model_training_comparison.ipynb` first to generate the required model files (`best_model_*.pkl`, `feature_scaler.pkl`, `categorical_encoder.pkl`).

**Example API Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST001",
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1574.2,
    "contract": "Two year",
    "payment_method": "Credit card",
    "internet_service": "Fiber optic",
    "senior_citizen": 0,
    "partner": 1,
    "dependents": 0,
    "phone_service": 1,
    "multiple_lines": 0,
    "online_security": 1,
    "online_backup": 0,
    "device_protection": 1,
    "tech_support": 0,
    "streaming_tv": 1,
    "streaming_movies": 0,
    "paperless_billing": 1
  }'
```

### 3. Run Streamlit Dashboard

```bash
# Navigate to project root and run
streamlit run dashboard/app.py
```

Dashboard will be available at:
- URL: http://localhost:8501

### 4. Run with Docker (Full Stack)

```bash
# Navigate to docker folder
cd docker

# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d
```

Services will be available at:
- FastAPI: http://localhost:8000
- Airflow: http://localhost:8080
- Grafana: http://localhost:3000 (login: admin/admin123)
- PostgreSQL: localhost:5432

### 5. Run Airflow DAGs Only

```bash
# Install Airflow (if not using Docker)
pip install apache-airflow

# Initialize Airflow database
airflow db init

# Start Airflow webserver
airflow webserver --port 8080

# Start Airflow scheduler (in another terminal)
airflow scheduler
```

DAG file location: `dags/churn_prediction_pipeline.py`

## Data

Processed datasets are located in `data/processed/`:
- `clean_telco.csv` - Cleaned raw data
- `features_scaled.csv` - Scaled features for modeling
- `target.csv` - Target variable (churn labels)

## Models

Trained models are stored in `models/`:
- `best_nn_model.pth` - Best Neural Network model (PyTorch)
- `neural_network.pth` - Neural Network checkpoint

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch prediction |
| `/model/info` | GET | Model information |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best_model_logistic_regression.pkl` | Path to model file |
| `SCALER_PATH` | `models/feature_scaler.pkl` | Path to scaler file |
| `ENCODER_PATH` | `models/categorical_encoder.pkl` | Path to encoder file |
| `DATABASE_URL` | - | PostgreSQL connection string |

## License

This project is licensed under the MIT License - see the LICENSE file for details.


