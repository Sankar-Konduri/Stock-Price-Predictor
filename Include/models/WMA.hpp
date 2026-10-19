#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "models/PredictionModel.hpp"

namespace spp {

/**
 * Weighted Moving Average with linearly increasing weights.
 *
 * Within a window of size n, the oldest observation gets weight 1, the next
 * gets 2, ..., and the newest gets n. Weights are normalized by n(n+1)/2 so
 * they sum to exactly 1 (the forecast stays on the same price scale).
 *
 * Multi-step forecast (recursive): identical strategy to SMA. Each forecast
 * is appended to the window and the oldest value dropped. This does NOT learn
 * a trend; it only reacts to recent prices more strongly than SMA does.
 */
class WMA final : public PredictionModel {
public:
    /// @throws InvalidInputError if window < 1.
    explicit WMA(std::size_t window = 5);

    std::string name() const override;
    std::string describe() const override;
    std::size_t minimumHistory() const override;

    std::vector<double> predict(const std::vector<double>& closingPrices,
                                std::size_t horizon) const override;

    std::unique_ptr<PredictionModel> clone() const override;

    std::size_t window() const;

    /// @throws InvalidInputError if window < 1.
    void setWindow(std::size_t window);

    /// Normalized weights, oldest first (weights().back() is the newest's).
    std::vector<double> weights() const;

private:
    std::size_t window_;
};

}  // namespace spp