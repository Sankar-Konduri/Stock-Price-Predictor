#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "ensemble/Ensemble.hpp"

namespace spp {

/// Error statistics for a set of (actual, predicted) pairs.
struct ErrorMetrics {
    std::size_t count = 0;         // number of pairs
    double mae = 0.0;              // mean absolute error
    double rmse = 0.0;             // root mean squared error
    std::optional<double> mape;    // mean absolute percentage error, in percent;
                                   // empty if every actual value was zero
    std::size_t mapeExcluded = 0;  // pairs skipped for MAPE because actual == 0
};

/// Settings for a walk-forward backtest.
///
/// A "forecast origin" is the number of observations available when a forecast
/// is made: origin t means the forecast sees closes[0..t) and is scored against
/// closes[t..t+horizon).
struct BacktestConfig {
    std::size_t horizon = 1;      // days forecast at each origin (>= 1)
    std::size_t firstOrigin = 0;  // 0 = automatic (ensemble.minimumHistory())
    std::size_t lastOrigin = 0;   // 0 = automatic (observations - horizon)
    std::size_t step = 1;         // distance between consecutive origins (>= 1)
};

/// Evaluation of one forecaster (an individual model or the ensemble).
struct ModelEvaluation {
    std::string name;
    std::string description;
    ErrorMetrics overall;              // pooled over all origins and forecast steps
    std::vector<ErrorMetrics> byStep;  // byStep[0] = next-day errors, size == horizon
};

/// One line of the ranked comparison table.
struct ComparisonRow {
    std::string name;
    ErrorMetrics overall;
    std::size_t rank = 0;  // 1 = lowest RMSE
    bool isEnsemble = false;
};

/// Does the ensemble beat its best component? (Negative change = lower error.)
struct EnsembleComparison {
    std::string bestIndividual;             // individual model with the lowest RMSE
    double bestRmse = 0.0;
    double ensembleRmse = 0.0;
    double bestMae = 0.0;
    double ensembleMae = 0.0;
    std::optional<double> rmseChangePct;    // empty if bestRmse == 0
    std::optional<double> maeChangePct;     // empty if bestMae == 0
    bool ensembleBeatsBest = false;         // ensemble RMSE strictly lower
};

/// Structured result of ModelEvaluator::backtest().
struct BacktestReport {
    BacktestConfig config;  // resolved settings (firstOrigin / lastOrigin filled in)
    std::size_t observations = 0;
    std::size_t originsEvaluated = 0;
    std::size_t originsSkipped = 0;  // origins where some model failed

    std::vector<ModelEvaluation> models;  // registration order
    ModelEvaluation ensemble;

    std::vector<ComparisonRow> ranking;   // models + ensemble, best RMSE first
    EnsembleComparison ensembleVsBest;
};

/**
 * Walk-forward backtesting and error metrics.
 *
 * Leakage guarantee: at origin t, the ensemble is given only closes[0..t).
 * Actual values are read after the forecast has been produced.
 */
class ModelEvaluator {
public:
    /// MAE, RMSE and MAPE for paired actual/predicted values.
    /// @throws InvalidInputError on size mismatch, empty input, or non-finite values.
    static ErrorMetrics computeMetrics(const std::vector<double>& actual,
                                       const std::vector<double>& predicted);

    /// Runs a walk-forward backtest of `ensemble` and each of its models.
    /// @throws InvalidInputError     for invalid configuration or non-finite prices
    /// @throws InsufficientDataError if the history is too short for any origin
    BacktestReport backtest(const Ensemble& ensemble,
                            const std::vector<double>& closingPrices,
                            const BacktestConfig& config = BacktestConfig{}) const;

    /// Human-readable report (ranked table, ensemble verdict, per-step errors).
    static std::string toText(const BacktestReport& report);
};

}  // namespace spp