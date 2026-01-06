# EV presence and consumption

EV penetration is steadily increasing. EVs’ electric batteries can be used to provide electrical grid support. This dataset collects 55,000 unique charging sessions from a large public parking lot in Utrecht. Forecasts derived from this dataset can inform:

- Distribution grid operation and reinforcement planning
- Charging infrastructure sizing and utilization forecasting
- Energy community peak-shaving and flexibility market participation
- Policy and planning studies on EV adoption and charging behavior

## Available data:

### EV

- **EV_id_x**: EV identifier pseudonym, based on the registered charging ID.

### Measurement

- **start_datetime**: Time of charger connection for this charging session in `YYYY-MM-DDThh:mm:ss` format. Times are according to local time, UTC+01:00 during winter and UTC+02:00 during summer.
- **end_datetime**: Time of charger disconnection for this charging session in `YYYY-MM-DDThh:mm:ss` format. Times are according to local time, UTC+01:00 during winter and UTC+02:00 during summer.
- **total_energy**: Total energy the EV charged during the charging session, in kWh.

### Charging infrastructure

- **rail**: Rail or group to which the charger that the EV connected to belongs.
- **evse_id**: Charging station identifier.
- **channel**: `1` or `2`, socket of the (potentially double-socketed) charging station the EV is connected to.

### Survey data

This data is parse and mostly not available

- **capacity_kwh**: Battery capacity, in kWh. Based on survey response 1\*.
- **commute_km_range_min**: Work–home commute (single trip), in km. Lower bound of the applicable range. Based on survey response 2\*.
- **commute_km_range_max**: Work–home commute (single trip), in km. Upper bound of the applicable range. Based on survey response 2\*.
- **EV_brand_selfreported**: Answer provided to the question of EV brand. Direct input from survey.
- **EV_model_selfreported**: Answer provided to the question of EV model. Direct input from survey.
- **capacity_kWh_selfreported**: Answer provided to the question of EV battery capacity. Direct input from survey.
- **ownership**: Ownership type. One of`'Gekocht'`, `'Privaat geleased'`, `'Door werkgever geleased'`, `'Anders, namelijk'`
(bought, privately leased, leased by employer, other). Multiple-choice input from survey.

A more detailed description is available in the readme file you can find [here](https://data.4tu.nl/datasets/80ef3824-3f5d-4e45-8794-3b8791efbd13/1).

## Challenge:

Retrieve the time-series for presence and consumption curves for 

- the overall fleet
- location (indicated by the `rail` column)

Forecast 24 hours ahead at hourly resolution. 

**Provide a probabilistic forecast for the overall fleet**

## Dataset:

**ev_dataset.csv**



# General information

For the final challenge you’re request to form **groups of 2-3 persons** and hand in a time series analysis and forecasting pipeline.

1. Chose a dataset among the proposed one
2. For the chosen dataset, forecast the target through the following steps:
    1. EXPLORATORY ANALYSIS. E.g. data plotting, correlation plots, ACF, etc..
    2. MODEL BUILDING. E.g. check stationarity, transform, choose model & fit, check residuals
    3. CV and model SELECTION. 
    4. Presentation of results

Keep your code:

- **Clean and well-organized**
- **Efficient** – avoid extremely long runtimes; (if it takes 1 hour to run assume it is not usable, alternatively, warn the user and explain very well why it should take so long)
- **Commented** – your code should be easy to understand without guessing.
- **Short -** Chatgpt implementations tend to be long, I would like to correct **your** code ****

### Deadline

- *You can hand in a notebook or (extensively) commented **running** code, by **Jan 19 2026***

### Hard filter

- **Your best model performing worse than naive persistent is sufficient condition for failing the exam**