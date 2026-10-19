#include "models/LinearRegression.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>

namespace spp {

namespace {
void requireValidLookback(std::size_t lookback) {
    if (lookback == 1) {
        throw InvalidInputError(
            "LinearRegression: lookback must be 0 (use all history) or at least 2.");
    }
}
}  // namespace

LinearRegression::LinearRegression(std::size_t lookback) : lookback_(lookback) {
    requireValidLookback(lookback);
}

std::string LinearRegression::name() const { return "LinearRegression"; }

std::string LinearRegression::describe() const {
    const std::string lb = (lookback_ == 0) ? "all" : std::to_string(lookback_);
    return "LinearRegression(lookback=" + lb + ")";
}

std::size_t LinearRegression::minimumHistory() const {
    return (lookback_ == 0) ? 2 : lookback_;
}

std::size_t LinearRegression::lookback() const { return lookback_; }

void LinearRegression::setLookback(std::size_t lookback) {
    requireValidLookback(lookback);
    lookback_ = lookback;
}

LinearFit LinearRegression::fit(const std::vector<double>& y) {
    const std::size_t n = y.size();
    if (n < 2) {
        throw InsufficientDataError(
            "LinearRegression: at least 2 points are needed to fit a line.");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(y[i])) {
            throw InvalidInputError("LinearRegression: value at index " +
                                    std::to_string(i) + " is not finite.");
        }
    }

    const double nd = static_cast<double>(n);
    const double xMean = (nd - 1.0) / 2.0;  // mean of 0..n-1
    const double yMean = std::accumulate(y.begin(), y.end(), 0.0) / nd;

    double sxx = 0.0;  // sum of (x - xMean)^2, always > 0 when n >= 2
    double sxy = 0.0;  // sum of (x - xMean)(y - yMean)
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = static_cast<double>(i) - xMean;
        sxx += dx * dx;
        sxy += dx * (y[i] - yMean);
    }

    LinearFit result;
    result.slope = sxy / sxx;
    result.intercept = yMean - result.slope * xMean;
    return result;
}

std::vector<double> LinearRegression::predict(
    const std::vector<double>& closingPrices, std::size_t horizon) const {
    validateInputs(closingPrices, horizon);

    // Select the fitting window: the last `lookback_` prices, or everything.
    const std::size_t n = (lookback_ == 0) ? closingPrices.size() : lookback_;
    const std::vector<double> window(
        closingPrices.end() - static_cast<std::ptrdiff_t>(n), closingPrices.end());

    const LinearFit line = fit(window);

    std::vector<double> forecast;
    forecast.reserve(horizon);
    for (std::size_t step = 0; step < horizon; ++step) {
        // Next unseen index is n, then n+1, ...
        const double x = static_cast<double>(n + step);
        forecast.push_back(line.intercept + line.slope * x);
    }
    return forecast;
}

std::unique_ptr<PredictionModel> LinearRegression::clone() const {
    return std::make_unique<LinearRegression>(*this);
}

}  // namespace spp