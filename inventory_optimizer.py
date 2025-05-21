import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class InventoryOptimizer:
    """
    Class responsible for optimizing inventory levels based on forecasts.
    """
    
    def __init__(self):
        self.recommendations = {}
        
    def calculate_par_levels(self, forecast_data, historical_data, lead_time=2, safety_factor=1.5):
        """
        Calculate recommended par levels for each product at each bar
        
        Parameters:
            forecast_data: DataFrame with demand forecasts
            historical_data: DataFrame with historical data
            lead_time: Replenishment lead time in days
            safety_factor: Safety stock multiplier
            
        Returns:
            DataFrame with recommended par levels
        """
        if forecast_data.empty:
            return pd.DataFrame()
            
        # Group forecasts by bar and product, then sum forecast values for the lead time period
        lead_time_demand = forecast_data.groupby(['Bar Name', 'Alcohol Type', 'Brand Name'])['Forecast (ml)'].sum().reset_index()
        lead_time_demand.rename(columns={'Forecast (ml)': 'Lead Time Demand (ml)'}, inplace=True)
        
        # Calculate historical variability for safety stock calculation
        if not historical_data.empty:
            # Group historical data by bar and product
            historical_stats = historical_data.groupby(['Bar Name', 'Alcohol Type', 'Brand Name'])['Consumed (ml)'].agg(['std', 'mean']).reset_index()
            
            # Merge with lead time demand
            par_levels = pd.merge(
                lead_time_demand,
                historical_stats,
                on=['Bar Name', 'Alcohol Type', 'Brand Name'],
                how='left'
            )
            
            # Fill missing values
            par_levels['std'] = par_levels['std'].fillna(par_levels['Lead Time Demand (ml)'] * 0.2)  # Assume 20% variability if no history
            par_levels['mean'] = par_levels['mean'].fillna(0)
        else:
            # If no historical data, just use forecast with assumed variability
            par_levels = lead_time_demand.copy()
            par_levels['std'] = par_levels['Lead Time Demand (ml)'] * 0.2  # Assume 20% variability
            par_levels['mean'] = 0
        
        # Calculate safety stock based on variability and safety factor
        par_levels['Safety Stock (ml)'] = par_levels['std'] * safety_factor
        
        # Calculate par level (lead time demand + safety stock)
        par_levels['Par Level (ml)'] = par_levels['Lead Time Demand (ml)'] + par_levels['Safety Stock (ml)']
        
        # Round up to nearest 100ml for practical purposes
        par_levels['Par Level (ml)'] = np.ceil(par_levels['Par Level (ml)'] / 100) * 100
        
        # Calculate min and max levels
        par_levels['Min Level (ml)'] = par_levels['Safety Stock (ml)']
        par_levels['Max Level (ml)'] = par_levels['Par Level (ml)'] * 1.5  # Max is 50% more than par
        
        # Calculate reorder point
        par_levels['Reorder Point (ml)'] = par_levels['Lead Time Demand (ml)'] + (par_levels['Safety Stock (ml)'] / 2)
        
        # Add average daily consumption
        forecast_daily_avg = forecast_data.groupby(['Bar Name', 'Alcohol Type', 'Brand Name'])['Forecast (ml)'].mean().reset_index()
        forecast_daily_avg.rename(columns={'Forecast (ml)': 'Avg Daily Consumption (ml)'}, inplace=True)
        
        par_levels = pd.merge(
            par_levels,
            forecast_daily_avg,
            on=['Bar Name', 'Alcohol Type', 'Brand Name'],
            how='left'
        )
        
        # Calculate days of supply
        par_levels['Days of Supply'] = par_levels['Par Level (ml)'] / par_levels['Avg Daily Consumption (ml)']
        par_levels['Days of Supply'] = par_levels['Days of Supply'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Store recommendations
        self.recommendations = par_levels
        
        return par_levels
    
    def calculate_orders(self, par_levels, current_inventory):
        """
        Calculate recommended order quantities based on par levels and current inventory
        
        Parameters:
            par_levels: DataFrame with par level recommendations
            current_inventory: DataFrame with current inventory levels
            
        Returns:
            DataFrame with recommended order quantities
        """
        if par_levels.empty or current_inventory.empty:
            return pd.DataFrame()
            
        # Merge par levels with current inventory
        orders = pd.merge(
            par_levels,
            current_inventory[['Bar Name', 'Alcohol Type', 'Brand Name', 'Closing Balance (ml)']],
            on=['Bar Name', 'Alcohol Type', 'Brand Name'],
            how='left'
        )
        
        # Fill missing current inventory with 0
        orders['Closing Balance (ml)'] = orders['Closing Balance (ml)'].fillna(0)
        
        # Calculate order quantity (par level - current inventory)
        orders['Order Quantity (ml)'] = orders['Par Level (ml)'] - orders['Closing Balance (ml)']
        
        # Order only if below reorder point
        orders['Below Reorder Point'] = orders['Closing Balance (ml)'] < orders['Reorder Point (ml)']
        orders.loc[~orders['Below Reorder Point'], 'Order Quantity (ml)'] = 0
        
        # Ensure non-negative order quantities
        orders['Order Quantity (ml)'] = np.maximum(orders['Order Quantity (ml)'], 0)
        
        # Round to nearest 100ml
        orders['Order Quantity (ml)'] = np.ceil(orders['Order Quantity (ml)'] / 100) * 100
        
        return orders
    
    def optimize_inventory_costs(self, historical_data, forecast_data, holding_cost_rate=0.2, stockout_cost_multiplier=3):
        """
        Optimize inventory levels considering holding costs and stockout costs
        
        Parameters:
            historical_data: DataFrame with historical data
            forecast_data: DataFrame with demand forecasts
            holding_cost_rate: Annual holding cost as a fraction of item value
            stockout_cost_multiplier: Multiplier for stockout cost vs holding cost
            
        Returns:
            DataFrame with cost-optimized inventory levels
        """
        if historical_data.empty or forecast_data.empty:
            return pd.DataFrame()
            
        # Calculate initial par levels
        initial_par_levels = self.calculate_par_levels(forecast_data, historical_data)
        
        if initial_par_levels.empty:
            return pd.DataFrame()
            
        # Assign item values based on alcohol type (simplified)
        alcohol_values = {
            'Vodka': 30,
            'Rum': 25,
            'Whiskey': 35,
            'Beer': 5,
            'Wine': 20
        }
        
        # Create copy for optimization
        optimized_levels = initial_par_levels.copy()
        
        # Assign item values
        optimized_levels['Item Value per ml'] = optimized_levels['Alcohol Type'].map(
            lambda x: alcohol_values.get(x, 10) / 1000  # Convert to value per ml
        )
        
        # Calculate daily holding cost
        optimized_levels['Daily Holding Cost per ml'] = optimized_levels['Item Value per ml'] * (holding_cost_rate / 365)
        
        # Calculate stockout cost per ml
        optimized_levels['Stockout Cost per ml'] = optimized_levels['Item Value per ml'] * stockout_cost_multiplier
        
        # Optimize par levels
        for idx, row in optimized_levels.iterrows():
            avg_demand = row['Avg Daily Consumption (ml)']
            daily_holding_cost = row['Daily Holding Cost per ml']
            stockout_cost = row['Stockout Cost per ml']
            
            if avg_demand <= 0 or daily_holding_cost <= 0:
                continue
                
            # Economic Order Quantity calculation (simplified)
            service_level = stockout_cost / (stockout_cost + daily_holding_cost)
            
            # Adjust safety factor based on service level
            # For normal distribution, service level of 0.9 corresponds to z-score of 1.28
            if service_level > 0.99:
                safety_factor = 3.0
            elif service_level > 0.95:
                safety_factor = 2.0
            elif service_level > 0.9:
                safety_factor = 1.5
            elif service_level > 0.8:
                safety_factor = 1.0
            else:
                safety_factor = 0.8
                
            # Recalculate safety stock with optimized safety factor
            optimized_levels.at[idx, 'Safety Stock (ml)'] = row['std'] * safety_factor
            
            # Recalculate par level
            optimized_levels.at[idx, 'Par Level (ml)'] = row['Lead Time Demand (ml)'] + optimized_levels.at[idx, 'Safety Stock (ml)']
            
            # Round up to nearest 100ml
            optimized_levels.at[idx, 'Par Level (ml)'] = np.ceil(optimized_levels.at[idx, 'Par Level (ml)'] / 100) * 100
            
            # Recalculate min and max levels
            optimized_levels.at[idx, 'Min Level (ml)'] = optimized_levels.at[idx, 'Safety Stock (ml)']
            optimized_levels.at[idx, 'Max Level (ml)'] = optimized_levels.at[idx, 'Par Level (ml)'] * 1.5
            
            # Recalculate reorder point
            optimized_levels.at[idx, 'Reorder Point (ml)'] = row['Lead Time Demand (ml)'] + (optimized_levels.at[idx, 'Safety Stock (ml)'] / 2)
            
        return optimized_levels
