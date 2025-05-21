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

## Output screens
![Screenshot 2025-05-21 200625](https://github.com/user-attachments/assets/ba6e6df5-dd39-44c5-a5ef-27338c04451d)
![Screenshot 2025-05-21 200951](https://github.com/user-attachments/assets/1ea817cc-6e58-42ff-92f4-b793d912daa8)
![Screenshot 2025-05-21 201109](https://github.com/user-attachments/assets/b64ffe07-61c8-47a4-981f-0f98b223742f)
![Screenshot 2025-05-21 201125](https://github.com/user-attachments/assets/2dd2e4a2-b854-4bae-8bf3-e29f19db7eb2)
![Screenshot 2025-05-21 201136](https://github.com/user-attachments/assets/2ce64301-2ddf-4150-8646-107664a9fdac)
![Screenshot 2025-05-21 201424](https://github.com/user-attachments/assets/d46dae11-dc91-4352-8c38-dd6b4be1967f)
![Screenshot 2025-05-21 201441](https://github.com/user-attachments/assets/6a4311db-2fad-46b0-ad73-387829666de4)
![Screenshot 2025-05-21 201448](https://github.com/user-attachments/assets/b91f5288-d460-4de4-bfe5-6867ef7f54a3)
![Screenshot 2025-05-21 201503](https://github.com/user-attachments/assets/c149a6bf-010c-4733-b840-baacd41f24d5)
![Screenshot 2025-05-21 201511](https://github.com/user-attachments/assets/2a3aa325-8813-45a2-a0c3-17ea2af0a196)
![Screenshot 2025-05-21 201521](https://github.com/user-attachments/assets/745a6eae-d297-43b0-89db-054b5cc12f27)
![Screenshot 2025-05-21 201527](https://github.com/user-attachments/assets/84b78222-9c2d-4747-a9bb-1baa6fb84df0)
![Screenshot 2025-05-21 201543](https://github.com/user-attachments/assets/865fc2bc-2824-4d3d-bea6-99bd835a8a45)
![Screenshot 2025-05-21 201552](https://github.com/user-attachments/assets/5cd51dee-09d8-479a-a111-748ebbd0bd00)
![Screenshot 2025-05-21 201602](https://github.com/user-attachments/assets/99231ffb-5777-486c-ab9c-b84a76ec058d)
![Screenshot 2025-05-21 201608](https://github.com/user-attachments/assets/e97fd784-6cd9-4f90-9ea3-42a209a66924)
![Screenshot 2025-05-21 201621](https://github.com/user-attachments/assets/bd47986f-1b5e-48de-802a-a1cb005e8c05)
![Screenshot 2025-05-21 201629](https://github.com/user-attachments/assets/48f01cbb-eaeb-4157-b0d2-41a0b78b476e)
![Screenshot 2025-05-21 201637](https://github.com/user-attachments/assets/fae16528-898a-47ee-80bd-14db58063bba)
![Screenshot 2025-05-21 201652](https://github.com/user-attachments/assets/5298b7c4-3fb6-4a5f-a9b8-64381fd56071)
![Screenshot 2025-05-21 201710](https://github.com/user-attachments/assets/250f52a5-ebbc-4ad0-a937-9912d18a5c76)
![Screenshot 2025-05-21 201705](https://github.com/user-attachments/assets/7563beb8-577d-41f0-b9d0-23a9a7bc3cff)
