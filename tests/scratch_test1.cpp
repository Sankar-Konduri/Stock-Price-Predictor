//TEST REDUNDANT AFTER THE STAGE 1 
/*
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "data/StockRecord.hpp"
#include "models/PredictionModel.hpp"

// A throwaway model that repeats the last price, to exercise the interface.
class LastValueModel final : public spp::PredictionModel {
public:
    std::string name() const override { return "LastValue"; }
    std::size_t minimumHistory() const override { return 2; }

    std::vector<double> predict(const std::vector<double>& prices,
                                std::size_t horizon) const override {
        validateInputs(prices, horizon);
        return std::vector<double>(horizon, prices.back());
    }

    std::unique_ptr<spp::PredictionModel> clone() const override {
        return std::make_unique<LastValueModel>(*this);
    }
};

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
        if (!ok) ++failures;
    };

    // Polymorphic use through a base-class smart pointer.
    std::unique_ptr<spp::PredictionModel> model = std::make_unique<LastValueModel>();
    auto forecast = model->predict({10.0, 11.0, 12.0}, 3);
    check(forecast.size() == 3 && forecast[0] == 12.0, "polymorphic predict");

    // clone() produces an independent working copy.
    auto copy = model->clone();
    check(copy->describe() == "LastValue", "clone + describe default");

    // Validation behavior.
    try { model->predict({1.0, 2.0}, 0); check(false, "horizon 0 rejected"); }
    catch (const spp::InvalidInputError&) { check(true, "horizon 0 rejected"); }

    try { model->predict({1.0}, 1); check(false, "short history rejected"); }
    catch (const spp::InsufficientDataError&) { check(true, "short history rejected"); }

    try {
        model->predict({1.0, std::numeric_limits<double>::quiet_NaN()}, 1);
        check(false, "NaN rejected");
    } catch (const spp::InvalidInputError&) { check(true, "NaN rejected"); }

    // StockRecord ordering.
    spp::StockRecord a{"2024-01-02", 1, 2, 0.5, 1.5, 100};
    spp::StockRecord b{"2024-01-03", 1, 2, 0.5, 1.5, 100};
    check(a < b && !(b < a), "StockRecord chronological ordering");

    std::cout << (failures == 0 ? "All checks passed.\n" : "Some checks failed.\n");
    return failures;
}

/*
 to run this:
-> g++ -std=c++17 -Wall -Wextra -Iinclude tests/scratch_test1.cpp src/models/PredictionModel.cpp -o scratch_test1.exe
and then 
.\scratch_test1.exe
*/
