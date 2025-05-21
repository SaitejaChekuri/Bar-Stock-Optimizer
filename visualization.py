import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class VisualizationManager:
    """
    Class responsible for creating visualizations for the inventory analysis.
    """
    
    def __init__(self):
        pass
    
    def plot_consumption_trends(self, time_series_data, bar_name=None, product_type=None):
        """
        Create consumption trend visualization
        
        Parameters:
            time_series_data: DataFrame with time series data
            bar_name: Optional filter for specific bar
            product_type: Optional filter for specific product type
            
        Returns:
            Plotly figure
        """
        if time_series_data.empty:
            return go.Figure()
            
        # Apply filters if provided
        filtered_data = time_series_data.copy()
        if bar_name:
            filtered_data = filtered_data[filtered_data['Bar Name'] == bar_name]
        if product_type:
            filtered_data = filtered_data[filtered_data['Alcohol Type'] == product_type]
            
        if filtered_data.empty:
            return go.Figure()
            
        # Aggregate consumption by date and alcohol type
        agg_data = filtered_data.groupby(['Date', 'Alcohol Type'])['Consumed (ml)'].sum().reset_index()
        
        # Create the line plot
        fig = px.line(
            agg_data, 
            x='Date', 
            y='Consumed (ml)', 
            color='Alcohol Type',
            title='Consumption Trends Over Time',
            labels={'Consumed (ml)': 'Consumption (ml)', 'Date': 'Date'},
            template='plotly_white'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Consumption (ml)',
            legend_title='Alcohol Type',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_inventory_levels(self, time_series_data, bar_name=None, product_type=None):
        """
        Create inventory levels visualization
        
        Parameters:
            time_series_data: DataFrame with time series data
            bar_name: Optional filter for specific bar
            product_type: Optional filter for specific product type
            
        Returns:
            Plotly figure
        """
        if time_series_data.empty:
            return go.Figure()
            
        # Apply filters if provided
        filtered_data = time_series_data.copy()
        if bar_name:
            filtered_data = filtered_data[filtered_data['Bar Name'] == bar_name]
        if product_type:
            filtered_data = filtered_data[filtered_data['Alcohol Type'] == product_type]
            
        if filtered_data.empty:
            return go.Figure()
            
        # Create a figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Aggregated data for plotting
        opening_balance = filtered_data.groupby('Date')['Opening Balance (ml)'].mean().reset_index()
        closing_balance = filtered_data.groupby('Date')['Closing Balance (ml)'].mean().reset_index()
        consumption = filtered_data.groupby('Date')['Consumed (ml)'].sum().reset_index()
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=opening_balance['Date'], 
                y=opening_balance['Opening Balance (ml)'],
                name="Opening Balance",
                line=dict(color='blue')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=closing_balance['Date'], 
                y=closing_balance['Closing Balance (ml)'],
                name="Closing Balance",
                line=dict(color='green')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Bar(
                x=consumption['Date'], 
                y=consumption['Consumed (ml)'],
                name="Consumption",
                marker_color='rgba(246, 78, 139, 0.6)'
            ),
            secondary_y=True,
        )
        
        # Set titles
        fig.update_layout(
            title_text="Inventory Levels and Consumption Over Time",
            template='plotly_white'
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Date")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Inventory Level (ml)", secondary_y=False)
        fig.update_yaxes(title_text="Consumption (ml)", secondary_y=True)
        
        return fig
    
    def plot_forecast_vs_actual(self, historical_data, forecast_data, bar_name=None, product_type=None, brand_name=None):
        """
        Create visualization comparing forecast to actual consumption
        
        Parameters:
            historical_data: DataFrame with historical data
            forecast_data: DataFrame with forecast data
            bar_name: Optional filter for specific bar
            product_type: Optional filter for specific product type
            brand_name: Optional filter for specific brand
            
        Returns:
            Plotly figure
        """
        if historical_data.empty or forecast_data.empty:
            return go.Figure()
            
        # Apply filters
        hist_filtered = historical_data.copy()
        fore_filtered = forecast_data.copy()
        
        if bar_name:
            hist_filtered = hist_filtered[hist_filtered['Bar Name'] == bar_name]
            fore_filtered = fore_filtered[fore_filtered['Bar Name'] == bar_name]
            
        if product_type:
            hist_filtered = hist_filtered[hist_filtered['Alcohol Type'] == product_type]
            fore_filtered = fore_filtered[fore_filtered['Alcohol Type'] == product_type]
            
        if brand_name:
            hist_filtered = hist_filtered[hist_filtered['Brand Name'] == brand_name]
            fore_filtered = fore_filtered[fore_filtered['Brand Name'] == brand_name]
            
        if hist_filtered.empty or fore_filtered.empty:
            return go.Figure()
            
        # Create the figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=hist_filtered['Date'],
                y=hist_filtered['Consumed (ml)'],
                mode='lines+markers',
                name='Actual Consumption',
                line=dict(color='blue')
            )
        )
        
        # Add forecast data
        fig.add_trace(
            go.Scatter(
                x=fore_filtered['Date'],
                y=fore_filtered['Forecast (ml)'],
                mode='lines+markers',
                name='Forecasted Consumption',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Forecast vs. Actual Consumption",
            xaxis_title="Date",
            yaxis_title="Consumption (ml)",
            legend_title="Data Type",
            template='plotly_white',
            hovermode="x unified"
        )
        
        return fig
    
    def plot_simulation_results(self, simulation_results, bar_name=None, product_type=None, brand_name=None):
        """
        Create visualization of simulation results
        
        Parameters:
            simulation_results: DataFrame with simulation results
            bar_name: Optional filter for specific bar
            product_type: Optional filter for specific product type
            brand_name: Optional filter for specific brand
            
        Returns:
            Plotly figure
        """
        if simulation_results.empty:
            return go.Figure()
            
        # Apply filters
        filtered_sim = simulation_results.copy()
        
        if bar_name:
            filtered_sim = filtered_sim[filtered_sim['Bar Name'] == bar_name]
            
        if product_type:
            filtered_sim = filtered_sim[filtered_sim['Alcohol Type'] == product_type]
            
        if brand_name:
            filtered_sim = filtered_sim[filtered_sim['Brand Name'] == brand_name]
            
        if filtered_sim.empty:
            return go.Figure()
            
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add inventory level
        fig.add_trace(
            go.Scatter(
                x=filtered_sim['Date'],
                y=filtered_sim['Ending Inventory (ml)'],
                mode='lines',
                name='Inventory Level',
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        # Add reorder point line
        fig.add_trace(
            go.Scatter(
                x=filtered_sim['Date'],
                y=filtered_sim['Reorder Point (ml)'],
                mode='lines',
                name='Reorder Point',
                line=dict(color='red', dash='dash')
            ),
            secondary_y=False
        )
        
        # Add par level line
        fig.add_trace(
            go.Scatter(
                x=filtered_sim['Date'],
                y=filtered_sim['Par Level (ml)'],
                mode='lines',
                name='Par Level',
                line=dict(color='green', dash='dash')
            ),
            secondary_y=False
        )
        
        # Add demand as bars
        fig.add_trace(
            go.Bar(
                x=filtered_sim['Date'],
                y=filtered_sim['Demand (ml)'],
                name='Demand',
                marker_color='rgba(246, 78, 139, 0.6)'
            ),
            secondary_y=True
        )
        
        # Add order placements as markers
        order_placed = filtered_sim[filtered_sim['Orders Placed (ml)'] > 0]
        if not order_placed.empty:
            fig.add_trace(
                go.Scatter(
                    x=order_placed['Date'],
                    y=order_placed['Ending Inventory (ml)'],
                    mode='markers',
                    name='Order Placed',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='orange'
                    )
                ),
                secondary_y=False
            )
        
        # Add stockouts as markers
        stockouts = filtered_sim[filtered_sim['Stockout (ml)'] > 0]
        if not stockouts.empty:
            fig.add_trace(
                go.Scatter(
                    x=stockouts['Date'],
                    y=stockouts['Ending Inventory (ml)'],
                    mode='markers',
                    name='Stockout',
                    marker=dict(
                        symbol='x',
                        size=12,
                        color='black'
                    )
                ),
                secondary_y=False
            )
            
        # Update layout
        fig.update_layout(
            title="Inventory Simulation Results",
            template='plotly_white',
            hovermode="x unified"
        )
        
        # Set axes titles
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Inventory Level (ml)", secondary_y=False)
        fig.update_yaxes(title_text="Demand (ml)", secondary_y=True)
        
        return fig
    
    def plot_performance_metrics(self, simulation_metrics, metric='Service Level'):
        """
        Create bar chart of performance metrics by bar and alcohol type
        
        Parameters:
            simulation_metrics: DataFrame with performance metrics
            metric: Metric to visualize (Service Level, Inventory Turnover, etc.)
            
        Returns:
            Plotly figure
        """
        if simulation_metrics.empty or metric not in simulation_metrics.columns:
            return go.Figure()
            
        # Group by bar and alcohol type
        grouped_metrics = simulation_metrics.groupby(['Bar Name', 'Alcohol Type'])[metric].mean().reset_index()
        
        # Create the bar chart
        fig = px.bar(
            grouped_metrics,
            x='Bar Name',
            y=metric,
            color='Alcohol Type',
            barmode='group',
            title=f'{metric} by Bar and Alcohol Type',
            template='plotly_white'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Bar Name',
            yaxis_title=metric,
            legend_title='Alcohol Type'
        )
        
        return fig
    
    def plot_par_level_recommendations(self, par_levels, bar_name=None):
        """
        Create visualization of par level recommendations
        
        Parameters:
            par_levels: DataFrame with par level recommendations
            bar_name: Optional filter for specific bar
            
        Returns:
            Plotly figure
        """
        if par_levels.empty:
            return go.Figure()
            
        # Apply filter if provided
        filtered_pars = par_levels.copy()
        if bar_name:
            filtered_pars = filtered_pars[filtered_pars['Bar Name'] == bar_name]
            
        if filtered_pars.empty:
            return go.Figure()
        
        # Group by bar and alcohol type
        grouped_pars = filtered_pars.groupby(['Bar Name', 'Alcohol Type']).agg({
            'Par Level (ml)': 'sum',
            'Avg Daily Consumption (ml)': 'sum',
            'Days of Supply': 'mean'
        }).reset_index()
        
        # Create a bubble chart
        fig = px.scatter(
            grouped_pars,
            x='Avg Daily Consumption (ml)',
            y='Par Level (ml)',
            size='Days of Supply',
            color='Alcohol Type',
            facet_col='Bar Name',
            hover_name='Alcohol Type',
            title='Par Level Recommendations by Bar and Alcohol Type',
            labels={
                'Avg Daily Consumption (ml)': 'Average Daily Consumption (ml)',
                'Par Level (ml)': 'Recommended Par Level (ml)',
                'Days of Supply': 'Days of Supply'
            },
            template='plotly_white'
        )
        
        # Update layout
        fig.update_layout(
            legend_title='Alcohol Type'
        )
        
        return fig
