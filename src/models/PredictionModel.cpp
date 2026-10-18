#include "models/PredictionModel.hpp"

#include <cmath>
#include <string>

//Mostly just for validation of data wrt to prediction models

namespace spp {

void PredictionModel::validateInputs(const std::vector<double>& closingPrices,
                                     std::size_t horizon) const {
    if (horizon < 1) {
        throw InvalidInputError(name() + ": forecast horizon must be at least 1.");
    }

    const std::size_t needed = minimumHistory();
    if (closingPrices.size() < needed) {
        throw InsufficientDataError(
            name() + ": needs at least " + std::to_string(needed) +
            " closing prices, but only " + std::to_string(closingPrices.size()) +
            " were provided.");
    }

    for (std::size_t i = 0; i < closingPrices.size(); ++i) {
        if (!std::isfinite(closingPrices[i])) {
            throw InvalidInputError(
                name() + ": closing price at index " + std::to_string(i) +
                " is not a finite number.");
        }
    }
}

}  // namespace spp