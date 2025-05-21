import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from data_processor import DataProcessor
from forecasting import Forecaster
from inventory_optimizer import InventoryOptimizer
from simulation import InventorySimulator
from visualization import VisualizationManager

def main():
    """
    Main function to run the end-to-end hotel bar inventory optimization pipeline
    """
    print("Starting Hotel Bar Inventory Optimization System...")
    
    # Initialize components
    data_processor = DataProcessor()
    forecaster = Forecaster()
    optimizer = InventoryOptimizer()
    simulator = InventorySimulator()
    visualizer = VisualizationManager()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    try:
        # Try to load the data from the CSV file
        raw_data = data_processor.load_data('hotel_bar_data.csv')
        print(f"Data loaded successfully with {len(raw_data)} records.")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Preprocess the data
    processed_data = data_processor.preprocess_data(raw_data)
    print("Data preprocessing complete.")
    
    # Create time series dataset
    daily_consumption = data_processor.aggregate_daily_consumption(processed_data)
    time_series_data = data_processor.create_time_series_dataset(daily_consumption)
    print("Time series dataset created.")
    
    # Generate inventory summary
    inventory_summary = data_processor.get_inventory_summary(processed_data)
    print("Inventory summary generated.")
    
    # Forecasting
    print("\nGenerating demand forecasts...")
    forecast_horizon = 14  # 2 weeks
    
    # Split data for model evaluation
    train_data, test_data = forecaster.train_test_split(time_series_data)
    
    # Train forecasting models
    models = forecaster.train_models(train_data, forecast_horizon=forecast_horizon)
    print(f"Trained {len(models)} forecasting models.")
    
    # Evaluate models
    evaluation = forecaster.evaluate_models(train_data, test_data)
    
    # Generate forecasts
    forecast_results = forecaster.generate_forecasts(time_series_data, forecast_horizon=forecast_horizon)
    print(f"Generated forecasts for {len(forecast_results) if isinstance(forecast_results, pd.DataFrame) else 0} product-day combinations.")
    
    # Inventory optimization
    print("\nOptimizing inventory levels...")
    
    # Calculate par levels
    par_levels = optimizer.calculate_par_levels(forecast_results, time_series_data)
    print(f"Generated par level recommendations for {len(par_levels) if isinstance(par_levels, pd.DataFrame) else 0} products.")
    
    # Get current inventory for order calculation
    current_inventory = processed_data.groupby(['Bar Name', 'Alcohol Type', 'Brand Name'])['Closing Balance (ml)'].last().reset_index()
    
    # Calculate order recommendations
    order_recommendations = optimizer.calculate_orders(par_levels, current_inventory)
    print(f"Generated order recommendations for {len(order_recommendations) if isinstance(order_recommendations, pd.DataFrame) else 0} products.")
    
    # Optimize inventory costs
    optimized_levels = optimizer.optimize_inventory_costs(time_series_data, forecast_results)
    print("Cost optimization complete.")
    
    # Simulation
    print("\nRunning inventory simulation...")
    simulation_days = 30
    
    # Run simulation
    simulation_results = simulator.run_simulation(
        forecast_results, 
        par_levels, 
        current_inventory, 
        simulation_days=simulation_days
    )
    print(f"Simulation complete for {len(simulation_results) if isinstance(simulation_results, pd.DataFrame) else 0} days.")
    
    # Calculate performance metrics
    simulation_metrics = simulator.calculate_simulation_metrics(simulation_results)
    
    # Display key metrics
    if not simulation_metrics.empty:
        print("\nSimulation Performance Summary:")
        print(f"Average Service Level: {simulation_metrics['Service Level'].mean():.2%}")
        print(f"Average Stockout Rate: {simulation_metrics['Stockout Rate'].mean():.2%}")
        print(f"Average Inventory Turnover: {simulation_metrics['Inventory Turnover'].mean():.2f} times per period")
    
    print("\nHotel Bar Inventory Optimization process completed successfully.")

if __name__ == "__main__":
    main()
