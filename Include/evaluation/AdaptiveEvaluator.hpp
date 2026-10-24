#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "ensemble/Ensemble.hpp"
#include "evaluation/ModelEvaluator.hpp"

namespace spp {

/// Settings for comparing equal-weight and adaptive ensembles.
struct AdaptiveConfig {
    std::size_t horizon = 1;           // days forecast at each origin (>= 1)
    std::size_t step = 1;              // distance between consecutive origins (>= 1)
    std::size_t validationWindow = 20; // past scored forecasts used to set weights (>= 1)
    std::size_t firstTestOrigin = 0;   // 0 = earliest origin with a full validation window
    std::size_t lastTestOrigin = 0;    // 0 = observations - horizon
    double minRelativeError = 0.01;    // floor for inverseErrorWeights, in (0, 1]
};

/// An adaptive scheme measured against the equal-weight ensemble.
/// Percent changes are negative when the adaptive scheme has LOWER error.
struct WeightingComparison {
    std::optional<double> rmseChangePct;  // empty if equal-weight RMSE is 0
    std::optional<double> maeChangePct;   // empty if equal-weight MAE is 0
    bool lowerRmse = false;               // RMSE strictly below equal-weight's
    std::size_t wins = 0;                 // test origins with lower absolute error
    std::size_t losses = 0;               // test origins with higher absolute error
    std::size_t ties = 0;
};

struct AdaptiveReport {
    AdaptiveConfig config;  // resolved (firstTestOrigin / lastTestOrigin filled in)
    std::size_t observations = 0;
    std::size_t originsSkipped = 0;       // origins where some model failed
    std::size_t testOrigins = 0;
    std::size_t validationFirstOrigin = 0;  // oldest forecast origin behind the fixed weights
    std::size_t validationLastOrigin = 0;   // newest one (always before the first test origin)

    std::vector<ModelEvaluation> models;  // individual models, on the test origins
    ModelEvaluation equalWeight;
    ModelEvaluation fixedValidation;
    ModelEvaluation rollingAdaptive;

    std::vector<double> fixedWeights;        // registration order
    std::vector<double> rollingMeanWeights;  // average over test origins
    std::vector<double> rollingLastWeights;  // at the final test origin

    std::vector<ComparisonRow> ranking;      // models + 3 ensembles, lowest RMSE first
    WeightingComparison fixedVsEqual;
    WeightingComparison rollingVsEqual;
};

/**
 * Compares equal-weight, fixed-validation and rolling-adaptive ensembles.
 *
 * Every model forecasts once per origin; the three ensembles are built from
 * those same forecasts and scored on identical test origins.
 *
 * Leakage rule: a forecast made at origin o (horizon h) may influence weights
 * at time t only if o + h <= t, i.e. all the prices it is scored against are
 * already known at t. Weights never see any test-period outcome in advance.
 */
class AdaptiveEvaluator {
public:
    /// @throws InvalidInputError     for invalid configuration or non-finite prices
    /// @throws InsufficientDataError if there is not enough history for the
    ///         requested validation window plus at least one test origin
    AdaptiveReport run(const Ensemble& ensemble,
                       const std::vector<double>& closingPrices,
                       const AdaptiveConfig& config = AdaptiveConfig{}) const;

    /// Human-readable comparison (ranking, weights, adaptive-vs-equal verdicts).
    static std::string toText(const AdaptiveReport& report);
};

}  // namespace spp