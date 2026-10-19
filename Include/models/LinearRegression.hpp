#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "models/PredictionModel.hpp"

namespace spp {

/// Result of a least-squares line fit y = intercept + slope * x.
/// x is the 0-based index within the fitted window (0 = oldest point used).
struct LinearFit {
    double slope = 0.0;
    double intercept = 0.0;
};

/**
 * Ordinary least-squares linear trend.
 *
 * Fits a line to the last `lookback` closing prices (or the whole history if
 * lookback == 0), using the day index 0..n-1 as the x-axis.
 *
 * Multi-step forecast (direct extrapolation): the line is fitted ONCE and
 * evaluated at x = n, n+1, ..., n+horizon-1. Unlike the moving averages, this
 * does project a trend. It also assumes the trend continues unchanged, and
 * forecasts are NOT clamped, so a steep downtrend can extrapolate below zero.
 *
 * Degenerate input: a constant price series gives slope 0 exactly. The fit
 * uses mean-centered sums, which is numerically stable for large prices.
 */
class LinearRegression final : public PredictionModel {
public:
    /// @param lookback number of most recent prices to fit; 0 = use all history.
    /// @throws InvalidInputError if lookback == 1 (a line needs 2+ points).
    explicit LinearRegression(std::size_t lookback = 0);

    std::string name() const override;
    std::string describe() const override;
    std::size_t minimumHistory() const override;

    std::vector<double> predict(const std::vector<double>& closingPrices,
                                std::size_t horizon) const override;

    std::unique_ptr<PredictionModel> clone() const override;

    std::size_t lookback() const;

    /// @throws InvalidInputError if lookback == 1.
    void setLookback(std::size_t lookback);

    /// Least-squares fit of `y` against indices 0..n-1.
    /// @throws InsufficientDataError if fewer than 2 points.
    /// @throws InvalidInputError if any value is not finite.
    static LinearFit fit(const std::vector<double>& y);

private:
    std::size_t lookback_;
};

}  // namespace spp