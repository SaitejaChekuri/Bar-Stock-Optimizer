import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

class Forecaster:
    """
    Class responsible for forecasting future demand using time series models.
    """
    
    def __init__(self):
        self.models = {}
        self.forecast_results = {}
        
    def train_test_split(self, time_series_data, test_size=0.2):
        """
        Split the time series data into training and testing sets
        
        Parameters:
            time_series_data: DataFrame with time series data
            test_size: Proportion of data to use for testing
            
        Returns:
            train_data, test_data: Split DataFrames
        """
        if time_series_data.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        # Determine the split point
        n = len(time_series_data)
        split_idx = int(n * (1 - test_size))
        
        # Split the data
        train_data = time_series_data.iloc[:split_idx].copy()
        test_data = time_series_data.iloc[split_idx:].copy()
        
        return train_data, test_data
    
    def train_models(self, time_series_data, forecast_horizon=7):
        """
        Train forecasting models for each bar and product combination
        
        Parameters:
            time_series_data: DataFrame with time series data
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dictionary with trained models
        """
        if time_series_data.empty:
            return {}
            
        # Get unique bar-product combinations
        bar_products = time_series_data[['Bar Name', 'Alcohol Type', 'Brand Name']].drop_duplicates()
        
        # Initialize models dictionary
        models = {}
        
        # For each bar-product combination
        for _, row in bar_products.iterrows():
            bar_name = row['Bar Name']
            alcohol_type = row['Alcohol Type']
            brand_name = row['Brand Name']
            
            # Filter data for this combination
            product_data = time_series_data[
                (time_series_data['Bar Name'] == bar_name) & 
                (time_series_data['Alcohol Type'] == alcohol_type) & 
                (time_series_data['Brand Name'] == brand_name)
            ].sort_values('Date')
            
            # Set index to date for time series analysis
            product_data = product_data.set_index('Date')
            consumption_series = product_data['Consumed (ml)']
            
            # Skip if not enough data
            if len(consumption_series) < 10:
                continue
            
            # Define key for model storage
            model_key = (bar_name, alcohol_type, brand_name)
            
            try:
                # Try different models and select the best one
                
                # 1. Exponential Smoothing (Holt-Winters)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    hw_model = ExponentialSmoothing(
                        consumption_series, 
                        trend='add',
                        seasonal='add', 
                        seasonal_periods=7  # Weekly seasonality
                    ).fit()
                
                # 2. ARIMA model (simplistic approach without extensive order tuning)
                try:
                    arima_model = ARIMA(consumption_series, order=(1,1,1)).fit()
                except:
                    arima_model = None
                
                # 3. Simple moving average
                ma_model = consumption_series.rolling(window=3).mean()
                
                # Store models
                models[model_key] = {
                    'exponential_smoothing': hw_model,
                    'arima': arima_model,
                    'moving_average': ma_model,
                    'data': consumption_series
                }
                
            except Exception as e:
                # If modeling fails, skip this combination
                print(f"Error training models for {model_key}: {str(e)}")
                continue
        
        self.models = models
        return models
    
    def evaluate_models(self, train_data, test_data):
        """
        Evaluate model performance on test data
        
        Parameters:
            train_data: Training data
            test_data: Test data for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if train_data.empty or test_data.empty or not self.models:
            return {}
        
        evaluation_results = {}
        
        # For each product model
        for model_key, model_dict in self.models.items():
            bar_name, alcohol_type, brand_name = model_key
            
            # Get test data for this product
            product_test = test_data[
                (test_data['Bar Name'] == bar_name) & 
                (test_data['Alcohol Type'] == alcohol_type) & 
                (test_data['Brand Name'] == brand_name)
            ].set_index('Date')['Consumed (ml)']
            
            if product_test.empty:
                continue
                
            model_evals = {}
            
            # Evaluate each model type
            if 'exponential_smoothing' in model_dict:
                hw_model = model_dict['exponential_smoothing']
                hw_forecast = hw_model.forecast(len(product_test))
                
                # Align forecast dates with test dates
                hw_forecast.index = product_test.index
                
                # Calculate error metrics
                mse = mean_squared_error(product_test, hw_forecast)
                mae = mean_absolute_error(product_test, hw_forecast)
                
                model_evals['exponential_smoothing'] = {
                    'MSE': mse,
                    'MAE': mae,
                    'forecast': hw_forecast
                }
            
            if 'arima' in model_dict and model_dict['arima'] is not None:
                arima_model = model_dict['arima']
                arima_forecast = arima_model.forecast(len(product_test))
                
                # Align forecast dates with test dates
                arima_forecast.index = product_test.index
                
                # Calculate error metrics
                mse = mean_squared_error(product_test, arima_forecast)
                mae = mean_absolute_error(product_test, arima_forecast)
                
                model_evals['arima'] = {
                    'MSE': mse,
                    'MAE': mae,
                    'forecast': arima_forecast
                }
            
            # Select best model based on MAE
            best_model = min(
                [m for m in model_evals.keys() if model_evals[m]['MAE'] > 0],
                key=lambda m: model_evals[m]['MAE'],
                default=None
            )
            
            if best_model:
                model_evals['best_model'] = best_model
                
            evaluation_results[model_key] = model_evals
        
        return evaluation_results
    
    def generate_forecasts(self, time_series_data, forecast_horizon=7):
        """
        Generate forecasts for the specified horizon
        
        Parameters:
            time_series_data: Complete time series data
            forecast_horizon: Number of days to forecast
            
        Returns:
            DataFrame with forecasts
        """
        if time_series_data.empty or not self.models:
            return pd.DataFrame()
        
        forecast_dfs = []
        
        # Get the latest date in the data
        latest_date = time_series_data['Date'].max()
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=latest_date + timedelta(days=1), 
            periods=forecast_horizon, 
            freq='D'
        )
        
        # For each product model
        for model_key, model_dict in self.models.items():
            bar_name, alcohol_type, brand_name = model_key
            
            # Choose the model to use for forecasting (default to exponential smoothing)
            if 'exponential_smoothing' in model_dict:
                forecast_model = model_dict['exponential_smoothing']
                model_type = 'exponential_smoothing'
            elif 'arima' in model_dict and model_dict['arima'] is not None:
                forecast_model = model_dict['arima']
                model_type = 'arima'
            else:
                continue  # Skip if no suitable model
                
            try:
                # Generate forecast
                forecast_values = forecast_model.forecast(forecast_horizon)
                
                # Ensure forecast values are non-negative
                forecast_values = np.maximum(forecast_values, 0)
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Bar Name': bar_name,
                    'Alcohol Type': alcohol_type,
                    'Brand Name': brand_name,
                    'Forecast (ml)': forecast_values,
                    'Model': model_type
                })
                
                forecast_dfs.append(forecast_df)
                
            except Exception as e:
                print(f"Error generating forecast for {model_key}: {str(e)}")
                continue
                
        # Combine all forecasts
        if forecast_dfs:
            all_forecasts = pd.concat(forecast_dfs, ignore_index=True)
            self.forecast_results = all_forecasts
            return all_forecasts
        else:
            return pd.DataFrame()
            
    def get_seasonal_patterns(self, time_series_data):
        """
        Analyze seasonal patterns in the consumption data
        
        Parameters:
            time_series_data: DataFrame with time series data
            
        Returns:
            Dictionary with seasonal patterns for each product
        """
        if time_series_data.empty:
            return {}
            
        seasonal_patterns = {}
        
        # Get unique bar-product combinations
        bar_products = time_series_data[['Bar Name', 'Alcohol Type', 'Brand Name']].drop_duplicates()
        
        # For each combination
        for _, row in bar_products.iterrows():
            bar_name = row['Bar Name']
            alcohol_type = row['Alcohol Type']
            brand_name = row['Brand Name']
            
            # Filter data for this combination
            product_data = time_series_data[
                (time_series_data['Bar Name'] == bar_name) & 
                (time_series_data['Alcohol Type'] == alcohol_type) & 
                (time_series_data['Brand Name'] == brand_name)
            ].sort_values('Date')
            
            # Skip if not enough data
            if len(product_data) < 14:  # Need at least 2 weeks of data
                continue
                
            # Set index to date
            product_data = product_data.set_index('Date')
            consumption_series = product_data['Consumed (ml)']
            
            # Calculate daily averages
            daily_avg = consumption_series.groupby(consumption_series.index.dayofweek).mean()
            
            # Find peak consumption day
            peak_day = daily_avg.idxmax()
            
            # Store seasonal pattern
            seasonal_patterns[(bar_name, alcohol_type, brand_name)] = {
                'daily_average': daily_avg.to_dict(),
                'peak_day': peak_day,
                'weekly_pattern': daily_avg.to_dict()
            }
            
        return seasonal_patterns
