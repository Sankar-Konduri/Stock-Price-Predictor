# Stock-Price-Predictor

A modular C++17 command-line application that forecasts stock closing prices from historical CSV data. It runs several independent statistical models, combines them into an **ensemble forecast**, and reports how much the models agree with each other.

> This is a learning/engineering project. Forecasts are statistical extrapolations of past prices, **not investment advice to anyone**.

## Done by
K.Sankar - 092
P.Daman- 771
N.Sathwik Raghava - 859



## Features

- Loads and validates historical OHLCV data from CSV (malformed rows are reported, never silently replaced with zero).
- Three independent models behind one abstract interface: Simple Moving Average (SMA), Weighted Moving Average (WMA) and Linear Regression.
- Ensemble forecast with equal or custom weights, per-day disagreement diagnostics, and safe handling of failing models.
- Menu-driven CLI: load data, view summary statistics, configure models, generate a forecast, inspect individual model predictions.
- Library code for walk-forward backtesting (MAE, RMSE, MAPE) and leak-free adaptive weighting. See [Evaluation library](#evaluation-library-not-in-the-menu).

## Building

Requirements: a C++17 compiler (MinGW-w64 GCC 8+ on Windows), CMake 3.16+, and `mingw32-make`.

```
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
ctest --test-dir build --output-on-failure
```

(`ctest --test-dir` needs CMake 3.20+; on older versions run `ctest --output-on-failure` from inside `build/`.)

## Running

Run from the project root so the default data path resolves:

```
.\build\stock_predictor.exe
```

Main menu:

```
 1. Load historical stock data
 2. View data and summary statistics
 3. Configure model parameters
 4. Generate ensemble forecast
 5. Display individual model predictions
 6. Exit
```

- **Load** accepts any CSV path; pressing Enter uses `data/historical_prices.csv`. If a load fails, previously loaded data is kept.
- **Configure** sets the SMA window, WMA window, Linear Regression lookback (0 = all history), forecast horizon (1-365 trading days), and ensemble weights (`equal` or three custom numbers, which are normalized).
- **Generate forecast** prints, in this order: the combined forecast, individual model predictions, model weights, forecast changes vs the last close, and diagnostics.

Forecast days are labelled `+1`, `+2`, ... (trading days after the last observation). No calendar dates are projected.

## CSV format

The first non-blank line is a header. Required columns (case-insensitive, any order): `Date`, `Open`, `High`, `Low`, `Close`, `Volume`. Extra columns, such as Yahoo Finance's `Adj Close`, are ignored.

```
Date,Open,High,Low,Close,Volume
2024-01-02,100.00,101.50,99.20,100.80,1200000
```

| Field | Rule |
|---|---|
| Date | Real calendar date as `YYYY-MM-DD` (leap years handled). Other formats are rejected. |
| Open/High/Low/Close | Finite and greater than 0, with `low <= min(open, close)` and `high >= max(open, close)`. |
| Volume | Whole number, 0 or greater. |

Other behaviour: blank lines are ignored; CRLF line endings and a UTF-8 BOM are handled; double quotes may wrap a field. Rows with a wrong field count, missing or non-numeric values, NaN/infinity, or invalid OHLC relationships are **skipped and reported with their line number**. Rows out of date order are sorted. Duplicate dates keep the first row and are reported. An input with no valid rows is an error.

> `data/historical_prices.csv` ships with **synthetic sample data** (40 rows) for development. It is not real market data. Replace it with a real export for meaningful forecasts.

## Models

All models implement `PredictionModel::predict(closingPrices, horizon)`, which returns exactly `horizon` values; element *i* is the forecast for trading day *t+1+i*.

| Model | One-step forecast | Multi-day strategy |
|---|---|---|
| **SMA(window)** | Mean of the last `window` closes. | *Recursive*: each forecast is appended to the window and the oldest value dropped. This produces a smoothed level and does **not** learn a trend. |
| **WMA(window)** | Weighted mean of the last `window` closes with weights 1..window (newest highest), normalized to sum to 1. | Recursive, same as SMA. |
| **LinearRegression(lookback)** | Least-squares line fitted to the last `lookback` closes (0 = all history), evaluated at the next index. | *Direct extrapolation*: one line is fitted and evaluated at indices n, n+1, ... This is the only model that projects a trend; forecasts are not clamped, so a steep downtrend can extrapolate below zero. |

Each model validates its input (horizon at least 1, enough history, finite prices) and throws a descriptive exception otherwise.

## Ensemble

`Ensemble` owns a list of `PredictionModel` objects and only talks to the base-class interface; it contains none of the model arithmetic.

- **Weights** are stored normalized and default to equal. Custom weights must be finite and non-negative with a positive sum.
- **Failures:** a model that throws, returns the wrong number of values, or returns NaN/infinity is excluded and listed in the warnings. Remaining weights are renormalized for that forecast. If nothing usable remains, the forecast fails with a clear message.
- **Diagnostics** per forecast day: spread (max - min across contributing models) and a weighted standard deviation around the combined forecast. The report also counts how many models forecast up, down or flat on the final day.

## Evaluation library (not in the menu)

The library also contains evaluation code with its own unit tests. It is not reachable from the CLI.

- `ModelEvaluator`: walk-forward backtesting of an ensemble and each of its models, with MAE, RMSE and MAPE (percent; points with a zero actual value are excluded and counted). Models and ensemble are scored on identical forecast origins.
- `AdaptiveEvaluator` + `inverseErrorWeights`: compares equal-weight, fixed-validation and rolling-adaptive inverse-RMSE ensembles on the same test origins.

**Leakage rule:** at forecast origin *t* a model sees only `closes[0..t)`. A forecast may influence adaptive weights at time *t* only if all prices it is scored against are already known (`origin + horizon <= t`).

Example use from C++:

```cpp
spp::Ensemble ensemble;
ensemble.addModel(std::make_unique<spp::SMA>(5));
ensemble.addModel(std::make_unique<spp::LinearRegression>());

spp::BacktestConfig cfg;
cfg.horizon = 3;
auto report = spp::ModelEvaluator().backtest(ensemble, closes, cfg);
std::cout << spp::ModelEvaluator::toText(report);
```

An ensemble is not automatically more accurate than its parts; the report states plainly whether its error is lower than the best individual model's.

## OOP design

- **Abstraction:** `PredictionModel` is an abstract class defining the forecasting interface.
- **Inheritance:** `SMA`, `WMA` and `LinearRegression` derive from it.
- **Runtime polymorphism:** `Ensemble` holds `std::unique_ptr<PredictionModel>` and calls `predict()` and `clone()` through virtual functions; the virtual destructor makes deletion through the base pointer safe.
- **Encapsulation:** each model keeps its configuration, validation and arithmetic private; callers use only the interface and the model's setters.
- **Composition:** `Ensemble` owns its models; `Application` coordinates `DataLoader`, `Ensemble` and the user's settings.

## Project layout

```
Stock-Price-Predictor/
├── CMakeLists.txt
├── data/historical_prices.csv
├── include/
│   ├── app/Application.hpp
│   ├── data/{StockRecord,DataLoader}.hpp
│   ├── ensemble/{Ensemble,Weighting}.hpp
│   ├── evaluation/{ModelEvaluator,AdaptiveEvaluator}.hpp
│   └── models/{PredictionModel,SMA,WMA,LinearRegression}.hpp
├── src/
│   ├── main.cpp
│   ├── app/Application.cpp
│   ├── data/DataLoader.cpp
│   ├── ensemble/{Ensemble,Weighting}.cpp
│   ├── evaluation/{ModelEvaluator,AdaptiveEvaluator}.cpp
│   ├── models/PredictionModel.cpp        (shared input validation)
│   ├── SMA/SMA.cpp
│   ├── WMA/WMA.cpp
│   └── LinearRegression/LinearRegression.cpp
└── tests/
    ├── test_helpers.hpp                  (tiny dependency-free test helpers)
    └── test_{SMA,WMA,LinearRegression,Ensemble,DataLoader,
              ModelEvaluator,Weighting,AdaptiveEvaluator,Application}.cpp
```

Everything except `main.cpp` and the tests is built into the static library `spp_core`; the tests and the executable link against it. Tests are registered with CTest and run from the project root.

## Limitations

- Moving averages and a straight-line fit are simple baselines. They cannot anticipate news, regime changes or turning points, and multi-day forecasts get less reliable as the horizon grows.
- Forecasts are for closing prices only; volume, open, high and low are loaded and summarized but not used by the models.
- Forecast days are trading-day offsets, not calendar dates.
- Backtesting and adaptive weighting exist only as library code (see above).