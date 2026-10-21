#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/PredictionModel.hpp"

namespace spp {

/// Thrown when an ensemble forecast cannot be produced
/// (no models registered, or no usable model remained).
class EnsembleError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// One registered model's contribution to an ensemble forecast.
struct ModelForecast {
    std::string name;                // e.g. "SMA"
    std::string description;         // e.g. "SMA(window=5)"
    bool succeeded = false;
    std::vector<double> forecast;    // empty if the model failed
    double weight = 0.0;             // effective normalized weight (0 if failed)
    std::string error;               // reason, if the model failed
};

/// Structured result of Ensemble::forecast().
struct EnsembleResult {
    std::size_t horizon = 0;
    double lastClose = 0.0;                // last observed closing price

    std::vector<ModelForecast> models;     // one entry per registered model, in
                                           // registration order (failed ones too)
    std::vector<double> combined;          // weighted forecast, size == horizon

    // Disagreement diagnostics, one value per forecast step. Computed over
    // models that succeeded AND have a positive weight.
    std::vector<double> spread;            // max - min across those models
    std::vector<double> stdDev;            // weighted std dev around `combined`
    std::size_t modelsUsed = 0;            // number of such models

    std::vector<std::string> warnings;     // excluded models, renormalization, etc.
};

/**
 * Combines any number of PredictionModel objects into one forecast.
 *
 * The ensemble only talks to the PredictionModel interface (runtime
 * polymorphism); it contains no model-specific calculations.
 *
 * Weights are stored normalized (sum to 1) and default to equal weights.
 * Adding a model resets all weights to equal. If some models fail at forecast
 * time, the weights of the remaining models are renormalized for that forecast
 * (the stored weights are not changed).
 *
 * Copying an Ensemble deep-copies its models via PredictionModel::clone().
 */
class Ensemble {
public:
    Ensemble() = default;
    ~Ensemble() = default;

    Ensemble(const Ensemble& other);
    Ensemble& operator=(const Ensemble& other);
    Ensemble(Ensemble&&) noexcept = default;
    Ensemble& operator=(Ensemble&&) noexcept = default;

    /// Takes ownership of `model` and resets weights to equal.
    /// @throws InvalidInputError if `model` is null.
    void addModel(std::unique_ptr<PredictionModel> model);

    std::size_t modelCount() const { return models_.size(); }

    /// Read-only access to a registered model.
    /// @throws std::out_of_range if index is invalid.
    const PredictionModel& model(std::size_t index) const;

    /// Largest minimumHistory() among registered models (0 if none): the
    /// history length needed for every model to participate.
    std::size_t minimumHistory() const;

    /// Sets weights, one per model in registration order. Values must be
    /// finite and >= 0 with a positive sum; they are normalized to sum to 1.
    /// @throws InvalidInputError on size mismatch or invalid values.
    void setWeights(const std::vector<double>& weights);

    void setEqualWeights();

    /// Current normalized weights, in registration order.
    const std::vector<double>& weights() const { return weights_; }

    /// Runs every registered model and combines the results.
    /// @param closingPrices chronological, oldest first
    /// @param horizon       number of future days (>= 1)
    /// @throws InvalidInputError     if horizon < 1
    /// @throws InsufficientDataError if closingPrices is empty
    /// @throws EnsembleError         if no models are registered or none usable
    EnsembleResult forecast(const std::vector<double>& closingPrices,
                            std::size_t horizon) const;

private:
    std::vector<std::unique_ptr<PredictionModel>> models_;
    std::vector<double> weights_;  // parallel to models_, normalized
};

}  // namespace spp