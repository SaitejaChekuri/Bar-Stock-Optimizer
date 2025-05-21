import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class InventorySimulator:
    """
    Class responsible for simulating inventory movements based on forecast and recommendations.
    """
    
    def __init__(self):
        self.simulation_results = None
        
    def run_simulation(self, forecasts, par_levels, current_inventory, simulation_days=30, lead_time=2):
        """
        Run inventory simulation based on forecasts and par levels
        
        Parameters:
            forecasts: DataFrame with demand forecasts
            par_levels: DataFrame with par level recommendations
            current_inventory: DataFrame with current inventory levels
            simulation_days: Number of days to simulate
            lead_time: Replenishment lead time in days
            
        Returns:
            DataFrame with simulation results
        """
        if forecasts.empty or par_levels.empty or current_inventory.empty:
            return pd.DataFrame()
            
        # Get unique bar-product combinations
        product_bars = current_inventory[['Bar Name', 'Alcohol Type', 'Brand Name', 'Closing Balance (ml)']].drop_duplicates()
        
        # Get the latest date in the data
        latest_date = datetime.now().date()
        if 'Date' in forecasts.columns:
            latest_date = forecasts['Date'].min() - timedelta(days=1)
            
        # Initialize simulation results
        sim_results = []
        
        # For each bar-product combination
        for _, row in product_bars.iterrows():
            bar_name = row['Bar Name']
            alcohol_type = row['Alcohol Type']
            brand_name = row['Brand Name']
            
            # Get par level for this product
            product_par = par_levels[
                (par_levels['Bar Name'] == bar_name) & 
                (par_levels['Alcohol Type'] == alcohol_type) & 
                (par_levels['Brand Name'] == brand_name)
            ]
            
            if product_par.empty:
                continue
                
            par_level = product_par['Par Level (ml)'].iloc[0]
            reorder_point = product_par['Reorder Point (ml)'].iloc[0]
            
            # Get forecast for this product
            product_forecast = forecasts[
                (forecasts['Bar Name'] == bar_name) & 
                (forecasts['Alcohol Type'] == alcohol_type) & 
                (forecasts['Brand Name'] == brand_name)
            ]
            
            # If no forecast, use average from par levels
            if product_forecast.empty and 'Avg Daily Consumption (ml)' in product_par.columns:
                avg_daily_consumption = product_par['Avg Daily Consumption (ml)'].iloc[0]
                # Create synthetic forecast using average
                product_forecast = pd.DataFrame({
                    'Date': [latest_date + timedelta(days=i) for i in range(1, simulation_days + 1)],
                    'Forecast (ml)': [avg_daily_consumption] * simulation_days
                })
            elif product_forecast.empty:
                continue
                
            # Initialize inventory for simulation
            inventory_level = row['Closing Balance (ml)']
            
            # Track pending orders
            pending_orders = []
            
            # Simulate each day
            for day in range(simulation_days):
                sim_date = latest_date + timedelta(days=day + 1)
                
                # Find forecast for this day
                day_forecast = product_forecast[product_forecast['Date'] == sim_date]
                
                # Use forecast value or average if not available
                if not day_forecast.empty:
                    day_demand = day_forecast['Forecast (ml)'].iloc[0]
                else:
                    # Use last value or average from par levels
                    if 'Avg Daily Consumption (ml)' in product_par.columns:
                        day_demand = product_par['Avg Daily Consumption (ml)'].iloc[0]
                    else:
                        day_demand = 0
                
                # Add random variation to demand (±20%)
                day_demand = max(0, day_demand * np.random.uniform(0.8, 1.2))
                
                # Check for order arrivals
                order_received = 0
                new_pending_orders = []
                for order_date, order_qty in pending_orders:
                    if (sim_date - order_date).days >= lead_time:
                        order_received += order_qty
                    else:
                        new_pending_orders.append((order_date, order_qty))
                        
                pending_orders = new_pending_orders
                
                # Update inventory with received orders
                inventory_level += order_received
                
                # Fulfill demand
                fulfilled_demand = min(inventory_level, day_demand)
                stockout = max(0, day_demand - inventory_level)
                inventory_level -= fulfilled_demand
                
                # Check if order needs to be placed
                order_placed = 0
                if inventory_level <= reorder_point:
                    order_placed = par_level - inventory_level
                    pending_orders.append((sim_date, order_placed))
                
                # Record simulation results
                sim_results.append({
                    'Date': sim_date,
                    'Bar Name': bar_name,
                    'Alcohol Type': alcohol_type,
                    'Brand Name': brand_name,
                    'Starting Inventory (ml)': inventory_level + fulfilled_demand,
                    'Demand (ml)': day_demand,
                    'Fulfilled (ml)': fulfilled_demand,
                    'Stockout (ml)': stockout,
                    'Orders Received (ml)': order_received,
                    'Orders Placed (ml)': order_placed,
                    'Ending Inventory (ml)': inventory_level,
                    'Par Level (ml)': par_level,
                    'Reorder Point (ml)': reorder_point
                })
        
        # Convert results to DataFrame
        simulation_df = pd.DataFrame(sim_results)
        self.simulation_results = simulation_df
        
        return simulation_df
    
    def calculate_simulation_metrics(self, simulation_results):
        """
        Calculate performance metrics from simulation results
        
        Parameters:
            simulation_results: DataFrame with simulation results
            
        Returns:
            DataFrame with performance metrics
        """
        if simulation_results.empty:
            return pd.DataFrame()
            
        # Group by product and bar
        metrics = simulation_results.groupby(['Bar Name', 'Alcohol Type', 'Brand Name']).agg({
            'Demand (ml)': 'sum',
            'Fulfilled (ml)': 'sum',
            'Stockout (ml)': 'sum',
            'Orders Placed (ml)': 'sum',
            'Orders Received (ml)': 'sum',
            'Ending Inventory (ml)': 'mean'
        }).reset_index()
        
        # Calculate service level (fulfilled demand / total demand)
        metrics['Service Level'] = metrics['Fulfilled (ml)'] / metrics['Demand (ml)']
        metrics['Service Level'] = metrics['Service Level'].fillna(1)  # If no demand, service level is 100%
        
        # Calculate inventory turnover (demand / average inventory)
        metrics['Inventory Turnover'] = metrics['Demand (ml)'] / metrics['Ending Inventory (ml)']
        metrics['Inventory Turnover'] = metrics['Inventory Turnover'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate stockout rate (days with stockout / total days)
        stockout_days = simulation_results[simulation_results['Stockout (ml)'] > 0].groupby(['Bar Name', 'Alcohol Type', 'Brand Name']).size().reset_index(name='Stockout Days')
        total_days = simulation_results.groupby(['Bar Name', 'Alcohol Type', 'Brand Name']).size().reset_index(name='Total Days')
        
        metrics = pd.merge(metrics, stockout_days, on=['Bar Name', 'Alcohol Type', 'Brand Name'], how='left')
        metrics = pd.merge(metrics, total_days, on=['Bar Name', 'Alcohol Type', 'Brand Name'], how='left')
        
        metrics['Stockout Days'] = metrics['Stockout Days'].fillna(0)
        metrics['Stockout Rate'] = metrics['Stockout Days'] / metrics['Total Days']
        
        # Calculate average order size
        order_counts = simulation_results[simulation_results['Orders Placed (ml)'] > 0].groupby(['Bar Name', 'Alcohol Type', 'Brand Name']).size().reset_index(name='Order Count')
        
        metrics = pd.merge(metrics, order_counts, on=['Bar Name', 'Alcohol Type', 'Brand Name'], how='left')
        metrics['Order Count'] = metrics['Order Count'].fillna(0)
        metrics['Average Order Size (ml)'] = metrics.apply(
            lambda x: x['Orders Placed (ml)'] / x['Order Count'] if x['Order Count'] > 0 else 0, 
            axis=1
        )
        
        return metrics
