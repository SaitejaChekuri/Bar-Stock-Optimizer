import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    """
    Class responsible for loading, cleaning, and preprocessing inventory data.
    """
    
    def __init__(self):
        pass
    
    def load_data(self, file_path):
        """
        Load data from CSV file
        
        Parameters:
            file_path: Path to the CSV data file
            
        Returns:
            Pandas DataFrame with the inventory data
        """
        df = pd.read_csv(file_path)
        return df
        
    def preprocess_data(self, df):
        """
        Clean and preprocess the inventory data
        
        Parameters:
            df: Raw DataFrame with inventory data
            
        Returns:
            Preprocessed DataFrame
        """
        # Check if dataframe is empty
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Convert 'Date Time Served' to datetime
        df_processed['Date Time Served'] = pd.to_datetime(df_processed['Date Time Served'], errors='coerce')
        
        # Extract date component
        df_processed['Date'] = df_processed['Date Time Served'].dt.date
        
        # Handle missing values
        numeric_columns = ['Opening Balance (ml)', 'Purchase (ml)', 'Consumed (ml)', 'Closing Balance (ml)']
        for col in numeric_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            # Fill any remaining NaNs with 0
            df_processed[col] = df_processed[col].fillna(0)
            
        # Fix any inconsistencies in the data
        # If closing balance doesn't equal opening + purchase - consumed, adjust it
        df_processed['Expected Closing'] = df_processed['Opening Balance (ml)'] + df_processed['Purchase (ml)'] - df_processed['Consumed (ml)']
        df_processed['Balance Error'] = df_processed['Closing Balance (ml)'] - df_processed['Expected Closing']
        
        # Create time-based features
        df_processed['DayOfWeek'] = df_processed['Date Time Served'].dt.dayofweek
        df_processed['Month'] = df_processed['Date Time Served'].dt.month
        df_processed['WeekOfYear'] = df_processed['Date Time Served'].dt.isocalendar().week
        df_processed['IsWeekend'] = df_processed['DayOfWeek'].isin([5, 6]).astype(int)  # 5=Saturday, 6=Sunday
            
        return df_processed
    
    def aggregate_daily_consumption(self, df):
        """
        Aggregate consumption data at daily level per bar and product
        
        Parameters:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with daily aggregated consumption
        """
        # Check if dataframe is empty
        if df.empty:
            return df
            
        # Group by date, bar, alcohol type, and brand
        daily_consumption = df.groupby(['Date', 'Bar Name', 'Alcohol Type', 'Brand Name'])['Consumed (ml)'].sum().reset_index()
        
        # Convert Date back to datetime for time series processing
        daily_consumption['Date'] = pd.to_datetime(daily_consumption['Date'])
        
        return daily_consumption
    
    def create_time_series_dataset(self, daily_consumption):
        """
        Transform the data into a format suitable for time series forecasting
        
        Parameters:
            daily_consumption: DataFrame with daily aggregated consumption
            
        Returns:
            DataFrame structured for time series analysis with filled dates
        """
        # Check if dataframe is empty
        if daily_consumption.empty:
            return daily_consumption
            
        # Get unique combinations of bar, alcohol type, and brand
        product_bars = daily_consumption[['Bar Name', 'Alcohol Type', 'Brand Name']].drop_duplicates()
        
        # Get min and max dates
        min_date = daily_consumption['Date'].min()
        max_date = daily_consumption['Date'].max()
        
        # Create complete date range
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create a complete dataset with all combinations of dates and products
        all_combinations = []
        for _, row in product_bars.iterrows():
            for date in date_range:
                all_combinations.append({
                    'Date': date,
                    'Bar Name': row['Bar Name'],
                    'Alcohol Type': row['Alcohol Type'],
                    'Brand Name': row['Brand Name']
                })
        
        # Create complete dataset
        complete_dataset = pd.DataFrame(all_combinations)
        
        # Merge with actual consumption data
        merged_data = pd.merge(
            complete_dataset,
            daily_consumption,
            on=['Date', 'Bar Name', 'Alcohol Type', 'Brand Name'],
            how='left'
        )
        
        # Fill missing consumption values with 0
        merged_data['Consumed (ml)'] = merged_data['Consumed (ml)'].fillna(0)
        
        return merged_data
    
    def get_inventory_summary(self, df):
        """
        Generate inventory summary statistics
        
        Parameters:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with inventory summary statistics
        """
        # Check if dataframe is empty
        if df.empty:
            return df
            
        # Create summary by bar and product
        summary = df.groupby(['Bar Name', 'Alcohol Type', 'Brand Name']).agg({
            'Consumed (ml)': ['sum', 'mean', 'std', 'max'],
            'Opening Balance (ml)': ['mean', 'min', 'max'],
            'Purchase (ml)': ['sum', 'mean', 'count'],
            'Closing Balance (ml)': ['mean', 'min', 'max']
        }).reset_index()
        
        # Flatten the column hierarchy
        summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
        
        # Calculate turnover rate (total consumption / average inventory)
        summary['Turnover_Rate'] = summary['Consumed (ml)_sum'] / summary['Opening Balance (ml)_mean']
        
        # Calculate days of supply (avg inventory / avg daily consumption)
        purchase_count = summary['Purchase (ml)_count']
        purchase_count = purchase_count.replace(0, 1)  # Avoid division by zero
        summary['Avg_Daily_Consumption'] = summary['Consumed (ml)_sum'] / purchase_count
        summary['Days_Of_Supply'] = summary['Opening Balance (ml)_mean'] / summary['Avg_Daily_Consumption']
        summary['Days_Of_Supply'] = summary['Days_Of_Supply'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return summary
