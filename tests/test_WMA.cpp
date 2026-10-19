#include <numeric>
#include <vector>

#include "models/WMA.hpp"
#include "test_helpers.hpp"

using namespace spp;
using namespace spp_test;

int main() {
    // Weights for window 3 are 1/6, 2/6, 3/6 (oldest -> newest), summing to 1.
    WMA wma(3);
    auto w = wma.weights();
    check(w.size() == 3 && approx(w[0], 1.0 / 6.0) && approx(w[1], 2.0 / 6.0) &&
              approx(w[2], 3.0 / 6.0),
          "weights are 1:2:3 normalized");
    check(approx(std::accumulate(w.begin(), w.end(), 0.0), 1.0), "weights sum to 1");
    check(approx(WMA(1).weights()[0], 1.0), "window=1 has single weight 1");

    // (10*1 + 20*2 + 30*3) / 6 = 140/6.
    auto f = wma.predict({10, 20, 30}, 2);
    check(approx(f[0], 140.0 / 6.0), "one-step forecast matches hand calculation");
    // Window becomes {20, 30, 140/6}: (20 + 60 + 3*140/6)/6 = 150/6 = 25.
    check(approx(f[1], 25.0), "recursive second step matches hand calculation");

    // Recent prices matter more: {1,1,1,1,10} gives 60/15 = 4.0 (an SMA(5) gives 2.8).
    auto recent = WMA(5).predict({1, 1, 1, 1, 10}, 1);
    check(approx(recent[0], 4.0), "most recent price has the largest influence");

    // Uses only the last `window` prices.
    auto tail = WMA(3).predict({999, 999, 10, 20, 30}, 1);
    check(approx(tail[0], 140.0 / 6.0), "older prices outside the window are ignored");

    // Constant series stays constant (confirms normalization).
    auto flat = WMA(4).predict({3.0, 3.0, 3.0, 3.0}, 3);
    bool allFlat = true;
    for (double v : flat) allFlat = allFlat && approx(v, 3.0);
    check(allFlat, "constant series gives constant forecast");

    // Metadata and cloning.
    check(wma.name() == "WMA" && wma.describe() == "WMA(window=3)", "name/describe");
    auto copy = wma.clone();
    wma.setWindow(8);
    check(copy->describe() == "WMA(window=3)", "clone is independent of original");

    // Validation.
    expectThrows<InvalidInputError>([] { WMA bad(0); }, "window 0 rejected");
    expectThrows<InsufficientDataError>([] { WMA(5).predict({1, 2}, 1); },
                                        "history shorter than window rejected");
    expectThrows<InvalidInputError>([] { WMA(2).predict({1, 2, 3}, 0); },
                                    "horizon 0 rejected");

    return finish();
}

/*
to run the test:
g++ -std=c++17 -Wall -Wextra -Iinclude -Itests tests/test_WMA.cpp src/WMA/WMA.cpp src/models/PredictionModel.cpp -o test_WMA.exe
and then: 

.\test_WMA.exe
*/