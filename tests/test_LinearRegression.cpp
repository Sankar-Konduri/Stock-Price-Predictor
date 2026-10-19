#include <limits>
#include <vector>

#include "models/LinearRegression.hpp"
#include "test_helpers.hpp"

using namespace spp;
using namespace spp_test;

int main() {
    // Perfect line y = 1 + 2x: fit recovers it, forecast continues it.
    LinearRegression lr;
    auto line = LinearRegression::fit({1, 3, 5, 7, 9});
    check(approx(line.slope, 2.0) && approx(line.intercept, 1.0), "exact line recovered");

    auto f = lr.predict({1, 3, 5, 7, 9}, 2);
    check(f.size() == 2 && approx(f[0], 11.0) && approx(f[1], 13.0),
          "forecast extrapolates at indices n, n+1");

    // Noisy data with known least-squares answer: y = {2,1,4,3}
    // slope = 0.6, intercept = 1.6, next value at x=4 -> 4.0.
    auto noisy = LinearRegression::fit({2, 1, 4, 3});
    check(approx(noisy.slope, 0.6) && approx(noisy.intercept, 1.6),
          "least-squares slope/intercept match hand calculation");
    check(approx(lr.predict({2, 1, 4, 3}, 1)[0], 4.0), "noisy forecast matches");

    // Degenerate: constant series has slope exactly 0.
    auto flat = LinearRegression::fit({4, 4, 4});
    check(flat.slope == 0.0 && approx(flat.intercept, 4.0), "constant series -> zero slope");

    // Large price level should not hurt numerical stability.
    auto big = LinearRegression::fit({1e9, 1e9 + 1, 1e9 + 2, 1e9 + 3});
    check(approx(big.slope, 1.0, 1e-6), "stable on large price levels");

    // Two points define a line exactly.
    check(approx(lr.predict({10, 12}, 1)[0], 14.0), "two points extrapolate exactly");

    // Lookback uses only the most recent points: fit on {1,2,3}, forecast 4, 5.
    LinearRegression windowed(3);
    auto wf = windowed.predict({100, 100, 100, 1, 2, 3}, 2);
    check(approx(wf[0], 4.0) && approx(wf[1], 5.0), "lookback restricts the fitting window");

    // Metadata and cloning.
    check(lr.name() == "LinearRegression" && lr.describe() == "LinearRegression(lookback=all)",
          "name/describe (all history)");
    check(windowed.describe() == "LinearRegression(lookback=3)", "describe (lookback)");
    check(lr.minimumHistory() == 2 && windowed.minimumHistory() == 3, "minimumHistory");
    auto copy = windowed.clone();
    windowed.setLookback(10);
    check(copy->describe() == "LinearRegression(lookback=3)", "clone is independent");

    // Validation.
    expectThrows<InvalidInputError>([] { LinearRegression bad(1); }, "lookback 1 rejected");
    expectThrows<InsufficientDataError>([&] { lr.predict({5}, 1); },
                                        "single price rejected");
    expectThrows<InsufficientDataError>([] { LinearRegression(5).predict({1, 2, 3}, 1); },
                                        "history shorter than lookback rejected");
    expectThrows<InvalidInputError>([&] { lr.predict({1, 2, 3}, 0); }, "horizon 0 rejected");
    expectThrows<InvalidInputError>(
        [&] { lr.predict({1, std::numeric_limits<double>::infinity(), 3}, 1); },
        "infinite price rejected");

    return finish();
}
/*
to test:
g++ -std=c++17 -Wall -Wextra -Iinclude -Itests tests/test_LinearRegression.cpp src/LinearRegression/LinearRegression.cpp src/models/PredictionModel.cpp -o test_LinearRegression.exe
and then: 

.\test_LinearRegression.exe
*/