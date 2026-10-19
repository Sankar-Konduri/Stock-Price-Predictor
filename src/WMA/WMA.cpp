#include "models/WMA.hpp"

#include <cstddef>
#include <deque>

namespace spp {

namespace {
void requireValidWindow(std::size_t window) {
    if (window < 1) {
        throw InvalidInputError("WMA: window size must be at least 1.");
    }
}
}  // namespace

WMA::WMA(std::size_t window) : window_(window) {
    requireValidWindow(window);
}

std::string WMA::name() const { return "WMA"; }

std::string WMA::describe() const {
    return "WMA(window=" + std::to_string(window_) + ")";
}

std::size_t WMA::minimumHistory() const { return window_; }

std::size_t WMA::window() const { return window_; }

void WMA::setWindow(std::size_t window) {
    requireValidWindow(window);
    window_ = window;
}

std::vector<double> WMA::weights() const {
    const double n = static_cast<double>(window_);
    const double denominator = n * (n + 1.0) / 2.0;  // 1 + 2 + ... + n

    std::vector<double> w(window_);
    for (std::size_t i = 0; i < window_; ++i) {
        w[i] = static_cast<double>(i + 1) / denominator;
    }
    return w;
}

std::vector<double> WMA::predict(const std::vector<double>& closingPrices,
                                 std::size_t horizon) const {
    validateInputs(closingPrices, horizon);

    const std::vector<double> w = weights();

    // Working window (oldest first); order matters here, unlike SMA.
    std::deque<double> buffer(
        closingPrices.end() - static_cast<std::ptrdiff_t>(window_),
        closingPrices.end());

    std::vector<double> forecast;
    forecast.reserve(horizon);

    for (std::size_t step = 0; step < horizon; ++step) {
        double weighted = 0.0;
        for (std::size_t i = 0; i < window_; ++i) {
            weighted += w[i] * buffer[i];
        }
        forecast.push_back(weighted);

        buffer.pop_front();
        buffer.push_back(weighted);
    }
    return forecast;
}

std::unique_ptr<PredictionModel> WMA::clone() const {
    return std::make_unique<WMA>(*this);
}

}  // namespace spp