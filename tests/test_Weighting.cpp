#include <limits>
#include <numeric>
#include <vector>

#include "ensemble/Weighting.hpp"
#include "models/PredictionModel.hpp"
#include "test_helpers.hpp"

using namespace spp;
using namespace spp_test;

int main() {
    // Half the error -> twice the weight; normalized.
    auto w = inverseErrorWeights({1.0, 2.0});
    check(approx(w[0], 2.0 / 3.0) && approx(w[1], 1.0 / 3.0), "weights proportional to 1/error");
    check(approx(std::accumulate(w.begin(), w.end(), 0.0), 1.0), "weights sum to 1");

    // Equal errors -> equal weights.
    auto eq = inverseErrorWeights({3.0, 3.0, 3.0});
    check(approx(eq[0], 1.0 / 3.0) && approx(eq[2], 1.0 / 3.0), "equal errors give equal weights");

    // A zero error is floored at 1% of the largest error: ratio capped at 100:1.
    auto floored = inverseErrorWeights({0.0, 1.0}, 0.01);
    check(approx(floored[0], 100.0 / 101.0) && approx(floored[1], 1.0 / 101.0),
          "near-zero error is floored (100:1 cap)");

    // The cap depends on relative, not absolute, error (scale-free).
    auto scaled = inverseErrorWeights({0.0, 500.0}, 0.01);
    check(approx(scaled[0], 100.0 / 101.0), "floor is relative to the largest error");

    // All errors zero -> nothing to distinguish models.
    auto zeros = inverseErrorWeights({0.0, 0.0});
    check(approx(zeros[0], 0.5) && approx(zeros[1], 0.5), "all-zero errors give equal weights");

    check(approx(inverseErrorWeights({7.0})[0], 1.0), "single model gets weight 1");

    // Validation.
    expectThrows<InvalidInputError>([] { inverseErrorWeights({}); }, "empty input rejected");
    expectThrows<InvalidInputError>([] { inverseErrorWeights({1.0, -1.0}); }, "negative error rejected");
    expectThrows<InvalidInputError>(
        [] { inverseErrorWeights({1.0, std::numeric_limits<double>::quiet_NaN()}); },
        "NaN error rejected");
    expectThrows<InvalidInputError>([] { inverseErrorWeights({1.0}, 0.0); }, "floor 0 rejected");
    expectThrows<InvalidInputError>([] { inverseErrorWeights({1.0}, 1.5); }, "floor above 1 rejected");

    return finish();
}