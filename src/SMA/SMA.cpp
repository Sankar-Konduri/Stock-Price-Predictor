#include "models/SMA.hpp"

#include <cstddef>
#include <deque>
#include <numeric>

namespace spp {

namespace {
void requireValidWindow(std::size_t window) {
    if (window < 1) {
        throw InvalidInputError("SMA: window size must be at least 1.");
    }
}
}  // namespace

SMA::SMA(std::size_t window) : window_(window) {
    requireValidWindow(window);
}

std::string SMA::name() const { return "SMA"; }

std::string SMA::describe() const {
    return "SMA(window=" + std::to_string(window_) + ")";
}

std::size_t SMA::minimumHistory() const { return window_; }

std::size_t SMA::window() const { return window_; }

void SMA::setWindow(std::size_t window) {
    requireValidWindow(window);
    window_ = window;
}

std::vector<double> SMA::predict(const std::vector<double>& closingPrices,
                                 std::size_t horizon) const {
    validateInputs(closingPrices, horizon);

    // Working window holding the most recent `window_` values (oldest first).
    std::deque<double> buffer(
        closingPrices.end() - static_cast<std::ptrdiff_t>(window_),
        closingPrices.end());

    std::vector<double> forecast;
    forecast.reserve(horizon);

    for (std::size_t step = 0; step < horizon; ++step) {
        const double sum = std::accumulate(buffer.begin(), buffer.end(), 0.0);
        const double mean = sum / static_cast<double>(window_);
        forecast.push_back(mean);

        // Recursive step: feed the forecast back in as if it were observed.
        buffer.pop_front();
        buffer.push_back(mean);
    }
    return forecast;
}

std::unique_ptr<PredictionModel> SMA::clone() const {
    return std::make_unique<SMA>(*this);
}

}  // namespace spp