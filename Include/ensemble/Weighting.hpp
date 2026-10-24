#pragma once

#include <vector>

namespace spp {

/**
 * Inverse-error weights: each model's weight is proportional to 1 / error,
 * so a model with half the error gets twice the weight. Returned weights are
 * normalized (sum to 1), in the same order as `errors`.
 *
 * To stop a model with near-zero error from taking all the weight, each error
 * is first raised to at least `minRelativeError * (largest error)`. That caps
 * the ratio between any two weights at 1 / minRelativeError (100:1 by default)
 * and makes the result independent of the price scale. If every error is
 * exactly zero, equal weights are returned.
 *
 * @param errors            one non-negative, finite error per model (e.g. RMSE)
 * @param minRelativeError  in (0, 1]
 * @throws InvalidInputError for empty input, invalid errors, or invalid floor
 */
std::vector<double> inverseErrorWeights(const std::vector<double>& errors,
                                        double minRelativeError = 0.01);

}  // namespace spp