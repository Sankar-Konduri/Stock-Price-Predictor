#include "ensemble/Weighting.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "models/PredictionModel.hpp"

namespace spp {

std::vector<double> inverseErrorWeights(const std::vector<double>& errors,
                                        double minRelativeError) {
    if (errors.empty()) {
        throw InvalidInputError("inverseErrorWeights: no errors were provided.");
    }
    if (!(minRelativeError > 0.0 && minRelativeError <= 1.0)) {
        throw InvalidInputError("inverseErrorWeights: minRelativeError must be in (0, 1].");
    }

    double largest = 0.0;
    for (std::size_t i = 0; i < errors.size(); ++i) {
        if (!std::isfinite(errors[i]) || errors[i] < 0.0) {
            throw InvalidInputError("inverseErrorWeights: error at index " +
                                    std::to_string(i) + " must be finite and non-negative.");
        }
        largest = std::max(largest, errors[i]);
    }

    const double n = static_cast<double>(errors.size());
    if (largest == 0.0) {
        return std::vector<double>(errors.size(), 1.0 / n);  // nothing to distinguish models
    }

    const double floorValue = largest * minRelativeError;
    std::vector<double> weights(errors.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < errors.size(); ++i) {
        weights[i] = 1.0 / std::max(errors[i], floorValue);
        sum += weights[i];
    }
    for (double& w : weights) {
        w /= sum;
    }
    return weights;
}

}  // namespace spp