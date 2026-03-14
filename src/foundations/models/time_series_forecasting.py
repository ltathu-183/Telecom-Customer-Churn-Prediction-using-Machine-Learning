# src/models/time_series_forecasting.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class CustomerTimeSeriesAnalyzer:
    """Time series forecasting for customer metrics"""
    
    def __init__(self):
        self.models = {}
    
    def prepare_time_series_data(self, df: pd.DataFrame, metric: str = 'monthly_charges'):
        """Prepare time series data from customer data"""
        # Aggregate by time period (e.g., monthly)
        df_ts = df.copy()
        df_ts['date'] = pd.to_datetime(df_ts.index)
        df_ts['month'] = df_ts['date'].dt.to_period('M')
        
        # Aggregate metrics
        ts_data = df_ts.groupby('month').agg({
            metric: 'mean',
            'customer_id': 'count'
        }).rename(columns={'customer_id': 'customer_count'})
        
        ts_data.index = ts_data.index.to_timestamp()
        ts_data = ts_data.sort_index()
        
        return ts_data
    
    def train_arima(self, series: pd.Series, order=(1, 1, 1)):
        """Train ARIMA model"""
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        
        self.models['arima'] = model_fit
        
        return model_fit
    
    def train_sarimax(self, series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Train SARIMAX model with seasonality"""
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        
        self.models['sarimax'] = model_fit
        
        return model_fit
    
    def train_prophet(self, df: pd.DataFrame, target_col: str):
        """Train Facebook Prophet model"""
        # Prepare data for Prophet
        prophet_df = df.reset_index()
        prophet_df = prophet_df.rename(columns={'index': 'ds', target_col: 'y'})
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_df)
        self.models['prophet'] = model
        
        return model
    
    def forecast(self, model_name: str, periods: int = 12):
        """Generate forecasts"""
        model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if model_name == 'arima' or model_name == 'sarimax':
            forecast = model.forecast(steps=periods)
            return forecast
        
        elif model_name == 'prophet':
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def evaluate_forecast(self, actual: pd.Series, predicted: pd.Series):
        """Evaluate forecast accuracy"""
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }


class ChurnTrendAnalyzer:
    """Analyze churn trends over time"""
    
    @staticmethod
    def calculate_churn_rate_over_time(df: pd.DataFrame, time_col: str = 'signup_date'):
        """Calculate churn rate by time period"""
        df_time = df.copy()
        df_time[time_col] = pd.to_datetime(df_time[time_col])
        df_time['period'] = df_time[time_col].dt.to_period('M')
        
        churn_by_period = df_time.groupby('period').agg({
            'churn': ['count', 'sum'],
            'tenure': 'mean'
        })
        
        churn_by_period.columns = ['total_customers', 'churned', 'avg_tenure']
        churn_by_period['churn_rate'] = churn_by_period['churned'] / churn_by_period['total_customers']
        
        return churn_by_period
    
    @staticmethod
    def detect_churn_anomalies(df: pd.DataFrame, threshold=2.0):
        """Detect anomalous churn periods"""
        churn_trend = ChurnTrendAnalyzer.calculate_churn_rate_over_time(df)
        
        # Calculate rolling statistics
        churn_trend['rolling_mean'] = churn_trend['churn_rate'].rolling(3).mean()
        churn_trend['rolling_std'] = churn_trend['churn_rate'].rolling(3).std()
        
        # Detect anomalies
        churn_trend['z_score'] = (
            churn_trend['churn_rate'] - churn_trend['rolling_mean']
        ) / churn_trend['rolling_std']
        
        churn_trend['is_anomaly'] = abs(churn_trend['z_score']) > threshold
        
        return churn_trend[churn_trend['is_anomaly']]
    
    @staticmethod
    def cohort_analysis(df: pd.DataFrame, cohort_col='signup_month', metric='churn'):
        """Perform cohort analysis"""
        df_cohort = df.copy()
        df_cohort[cohort_col] = pd.to_datetime(df_cohort[cohort_col])
        df_cohort['cohort'] = df_cohort[cohort_col].dt.to_period('M')
        
        # Calculate retention by cohort
        cohort_data = df_cohort.groupby(['cohort', 'tenure']).agg({
            metric: 'mean'
        }).unstack()
        
        return cohort_data