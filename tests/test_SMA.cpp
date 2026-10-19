#include <vector>

#include "models/SMA.hpp"
#include "test_helpers.hpp"

using namespace spp;
using namespace spp_test;

int main() {
    // Basic one-step forecast: mean of last 3 of {1..6} = (4+5+6)/3 = 5.
    SMA sma(3);
    auto f = sma.predict({1, 2, 3, 4, 5, 6}, 1);
    check(f.size() == 1 && approx(f[0], 5.0), "one-step forecast is mean of last window");

    // Recursive multi-step: forecasts are fed back into the window.
    auto f3 = sma.predict({1, 2, 3, 4, 5, 6}, 3);
    const double s1 = 5.0;
    const double s2 = (5.0 + 6.0 + s1) / 3.0;
    const double s3 = (6.0 + s1 + s2) / 3.0;
    check(f3.size() == 3, "multi-step returns `horizon` values");
    check(approx(f3[0], s1) && approx(f3[1], s2) && approx(f3[2], s3),
          "multi-step recursion matches hand calculation");

    // Window 1 repeats the last price.
    auto naive = SMA(1).predict({7, 8, 9}, 2);
    check(approx(naive[0], 9.0) && approx(naive[1], 9.0), "window=1 repeats last price");

    // Constant series stays constant.
    auto flat = SMA(4).predict({2.5, 2.5, 2.5, 2.5, 2.5}, 4);
    bool allFlat = true;
    for (double v : flat) allFlat = allFlat && approx(v, 2.5);
    check(allFlat, "constant series gives constant forecast");

    // Predict must not alter the input (const-correctness sanity).
    std::vector<double> input{1, 2, 3, 4};
    SMA(2).predict(input, 2);
    check(input == std::vector<double>({1, 2, 3, 4}), "input unchanged by predict");

    // Metadata and cloning.
    check(sma.name() == "SMA" && sma.describe() == "SMA(window=3)", "name/describe");
    check(sma.minimumHistory() == 3, "minimumHistory equals window");
    auto copy = sma.clone();
    sma.setWindow(10);
    check(copy->describe() == "SMA(window=3)", "clone is independent of original");

    // Validation.
    expectThrows<InvalidInputError>([] { SMA bad(0); }, "window 0 rejected");
    expectThrows<InvalidInputError>([&] { sma.setWindow(0); }, "setWindow(0) rejected");
    expectThrows<InsufficientDataError>([] { SMA(5).predict({1, 2, 3}, 1); },
                                        "history shorter than window rejected");
    expectThrows<InvalidInputError>([] { SMA(2).predict({1, 2, 3}, 0); },
                                    "horizon 0 rejected");

    return finish();
}

/*
TO TEST, in root cpy the terminal cmd:
g++ -std=c++17 -Wall -Wextra -Iinclude -Itests tests/test_SMA.cpp src/SMA/SMA.cpp src/models/PredictionModel.cpp -o test_SMA.exe
and then:
.\test_SMA.exe
*/