# EV Presence and Consumption Forecasting

This project provides a comprehensive analysis and forecasting of Electric Vehicle (EV) charging behavior using a large dataset from a public parking lot. The primary goal is to predict both the number of EVs present (presence) and the total energy consumed (consumption) on an hourly basis.

## Project Overview

The project follows a complete data science workflow:
1.  **Exploratory Data Analysis (EDA)**: To understand the underlying patterns, seasonality, and correlations in the data.
2.  **Data Transformation**: Converting discrete charging session data into a regular hourly time series suitable for forecasting models.
3.  **Modeling**: Implementing and comparing various forecasting models, from simple benchmarks to advanced deep learning models.
4.  **Probabilistic Forecasting**: Generating prediction intervals to quantify uncertainty and aid in risk management.
5.  **Future Prediction**: Creating actionable 24-hour ahead forecasts for operational planning.

## Dataset

The analysis is based on a dataset containing over 55,000 unique charging sessions from a public parking lot in Utrecht, Netherlands.

-   **Source**: [4TU.ResearchData](https://data.4tu.nl/datasets/80ef3824-3f5d-4e45-8794-3b8791efbd13/1)
-   **Content**: The dataset includes information such as start and end times of charging, total energy consumed, and EV brand.

## Key Findings

-   **Strong Seasonality**: The data exhibits strong daily (24-hour) and weekly seasonal patterns, with charging demand concentrated during weekday working hours.
-   **Model Performance**: The advanced deep learning model, **Chronos 2**, demonstrated superior accuracy compared to traditional benchmark models (Naive, Mean, Seasonal Naive) and ARIMA.
-   **Effective Data Transformation**: A custom function successfully converted session-based data into an hourly time series while conserving the total energy, ensuring the validity of the analysis.
-   **Value of Probabilistic Forecasts**: By generating prediction intervals, the model provides a range of potential outcomes, which is crucial for operators managing grid load and resource allocation under uncertainty.

## File Structure

```
.
├── data/
│   └── ev_dataset.csv      # The dataset file
├── .gitignore              # Gitignore
├── functions.py            # Python script with helper functions for models and analysis
├── notebook.ipynb          # Jupyter Notebook with the complete analysis
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Axeeh/EV_PresenceAndConsumption.git
    cd EV_PresenceAndConsumption
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook notebook.ipynb
    ```

## Dependencies

All the required Python libraries are listed in the `requirements.txt` file. Key libraries include:
-   `pandas` & `numpy` for data manipulation
-   `matplotlib` & `seaborn` for visualization
-   `statsmodels` for statistical analysis
-   `chronos-forecasting` for the Chronos 2 model
-   `scikit-learn` for metrics
-   `jupyter` for running the notebook

## Authors

-   Alessio Carnevale
-   Manuel Cattoni
