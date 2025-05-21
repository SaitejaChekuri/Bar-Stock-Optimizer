# Hotel Bar Inventory Optimization System

## Created by Saiteja Chekuri

This project implements a complete end-to-end solution for optimizing inventory management in hotel bars. The system analyzes historical consumption data, forecasts future demand, recommends optimal inventory levels, and simulates the performance of these recommendations.

## Business Problem

Hotel bars face significant inventory management challenges, including:

- **Frequent stockouts** of popular items leading to lost sales and reduced guest satisfaction
- **Excess inventory** of slow-moving products increasing holding costs and capital tied up in inventory
- **Inefficient ordering patterns** resulting in emergency orders and higher operational costs
- **Inconsistent inventory levels** across different locations and seasons

These challenges directly impact the hotel's revenue, operational efficiency, and guest experience. This solution addresses these challenges through data-driven demand forecasting, optimal par level calculation, and inventory simulation.

## Features

- **Data Preprocessing:** Cleans and transforms raw inventory data into a format suitable for analysis
- **Time Series Forecasting:** Predicts future demand for each product at each bar location
- **Inventory Optimization:** Calculates optimal par levels and safety stock based on forecasts
- **Performance Simulation:** Tests inventory recommendations in a simulated environment
- **Recommendations Dashboard:** Provides actionable insights for inventory improvements

## Technical Approach

The system leverages multiple forecasting methods:

1. **Exponential Smoothing (Holt-Winters)** - Primary forecasting method chosen for its ability to handle seasonal patterns
2. **ARIMA (Autoregressive Integrated Moving Average)** - Used as a secondary model for more complex time series
3. **Moving Average** - Used as a simple baseline model

For inventory optimization, the system:
- Calculates par levels based on lead time demand and safety stock
- Optimizes cost trade-offs between holding costs and stockout costs
- Recommends reorder points and order quantities

## Files in this Repository

- `bar_inventory_optimization.ipynb`: Main Jupyter notebook with end-to-end solution
- `data_processor.py`: Module for data loading and preprocessing
- `forecasting.py`: Module containing forecasting models
- `inventory_optimizer.py`: Module for inventory optimization
- `simulation.py`: Module for simulating inventory performance
- `visualization.py`: Module for creating visualizations
- `hotel_bar_data.csv`: Sample dataset of hotel bar inventory movements

## Results

The inventory optimization system achieved:
- **Service Level:** Average service level of 95.3%, exceeding target of 95%
- **Stockout Rate:** Average stockout rate reduced to 4.7%
- **Inventory Reduction:** Estimated 18% reduction in average inventory levels
- **Cost Savings:** Projected 15-20% reduction in inventory holding costs

## How to Use

1. Clone this repository
2. Install required dependencies: pandas, numpy, matplotlib, seaborn, plotly, statsmodels, scikit-learn
3. Open the Jupyter notebook `bar_inventory_optimization.ipynb`
4. Run the cells to explore the complete solution

## Future Improvements

- Incorporate machine learning models (LSTM, Prophet) for products with complex patterns
- Add external factors like hotel occupancy and events to forecasting
- Implement dynamic lead times based on supplier performance
- Develop a web dashboard for real-time monitoring and recommendations
