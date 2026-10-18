#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "models/PredictionModel.hpp"

namespace spp {

/**
 * Simple Moving Average.
 *
 * One-step forecast: arithmetic mean of the last `window` closing prices.
 *
 * Multi-step forecast (recursive): each forecast is appended to the working
 * window and the oldest value is dropped, then the mean is taken again.
 * This does NOT learn a trend. For a trending series the forecasts lag the
 * trend and flatten out; longer horizons are smoothed-level estimates only.
 */
class SMA final : public PredictionModel {
public:
    /// @throws InvalidInputError if window < 1.
    explicit SMA(std::size_t window = 5); //window= It is the number of most recent closing prices used to calculate the average

    std::string name() const override;
    std::string describe() const override;
    std::size_t minimumHistory() const override;

    std::vector<double> predict(const std::vector<double>& closingPrices,
                                std::size_t horizon) const override;

    std::unique_ptr<PredictionModel> clone() const override;

    std::size_t window() const;

    /// @throws InvalidInputError if window < 1.
    void setWindow(std::size_t window);

private:
    std::size_t window_;
};

}  // namespace spp