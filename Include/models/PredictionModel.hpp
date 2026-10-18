#pragma once

#include <cstddef>// Provides fundamental type definitions and macros from the C standard library 
#include <memory>//Contains utilities for dynamic memory management, including smart pointers
#include <stdexcept>//Defines standard runtime and logic exception classes
#include <string>
#include <vector>

namespace spp {

/// Thrown when input (prices, horizon, configuration) is invalid.
class InvalidInputError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/// Thrown when the price history is too short for the requested model.
class InsufficientDataError : public InvalidInputError {
public:
    using InvalidInputError::InvalidInputError;
};

/**
 * Abstract interface for all forecasting models.
 *
 * Contract for predict():
 *  - `closingPrices` is in chronological order (oldest first, newest last).
 *  - `horizon` is the number of future trading days to forecast (>= 1).
 *  - Returns exactly `horizon` values; element i is the forecast for day t+1+i,
 *    where t is the last observation.
 *  - Throws InvalidInputError / InsufficientDataError on bad input.
 *  - Never modifies the model (predict is const): models are stateless
 *    between calls, which keeps backtesting free of hidden data leakage.
 *
 * Each concrete model must document how it produces multi-day forecasts.
 */
class PredictionModel {
public:
    virtual ~PredictionModel() = default;  // required for safe delete via base pointer

    /// Short model name, e.g. "SMA".
    virtual std::string name() const = 0;

    /// Name plus configuration, e.g. "SMA(window=5)". Defaults to name().
    virtual std::string describe() const { return name(); }

    /// Smallest number of closing prices this model needs to forecast.
    virtual std::size_t minimumHistory() const = 0;

    /// Forecast `horizon` future closing prices.
    virtual std::vector<double> predict(const std::vector<double>& closingPrices,
                                        std::size_t horizon) const = 0;

    /// Polymorphic copy ("virtual constructor"). Needed because a base-class
    /// object cannot be copied through a base pointer without slicing.
    virtual std::unique_ptr<PredictionModel> clone() const = 0;

protected:
    // Protected so only derived classes can copy (used by clone()),
    // and nobody can accidentally slice-copy a model through the base type.
    PredictionModel() = default;
    PredictionModel(const PredictionModel&) = default;
    PredictionModel& operator=(const PredictionModel&) = default;

    /**
     * Shared input validation for derived classes. Checks:
     *  - horizon >= 1
     *  - closingPrices.size() >= minimumHistory()
     *  - every price is finite (no NaN / infinity)
     * Throws InvalidInputError or InsufficientDataError with a descriptive message.
     */
    void validateInputs(const std::vector<double>& closingPrices,
                        std::size_t horizon) const;
};

}  // namespace spp