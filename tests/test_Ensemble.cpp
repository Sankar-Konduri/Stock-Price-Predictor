#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ensemble/Ensemble.hpp"
#include "models/LinearRegression.hpp"
#include "models/SMA.hpp"
#include "models/WMA.hpp"
#include "test_helpers.hpp"

using namespace spp;
using namespace spp_test;

namespace {

// Test doubles: tiny models with exactly predictable behavior.
class ConstantModel final : public PredictionModel {
public:
    explicit ConstantModel(double value) : value_(value) {}
    std::string name() const override { return "Const"; }
    std::string describe() const override { return "Const(" + std::to_string(value_) + ")"; }
    std::size_t minimumHistory() const override { return 1; }
    std::vector<double> predict(const std::vector<double>& prices,
                                std::size_t horizon) const override {
        validateInputs(prices, horizon);
        return std::vector<double>(horizon, value_);
    }
    std::unique_ptr<PredictionModel> clone() const override {
        return std::make_unique<ConstantModel>(*this);
    }
private:
    double value_;
};

class ThrowingModel final : public PredictionModel {
public:
    std::string name() const override { return "Throwing"; }
    std::size_t minimumHistory() const override { return 1; }
    std::vector<double> predict(const std::vector<double>&, std::size_t) const override {
        throw InvalidInputError("simulated model failure");
    }
    std::unique_ptr<PredictionModel> clone() const override {
        return std::make_unique<ThrowingModel>(*this);
    }
};

class WrongSizeModel final : public PredictionModel {
public:
    std::string name() const override { return "WrongSize"; }
    std::size_t minimumHistory() const override { return 1; }
    std::vector<double> predict(const std::vector<double>&, std::size_t horizon) const override {
        return std::vector<double>(horizon + 1, 1.0);
    }
    std::unique_ptr<PredictionModel> clone() const override {
        return std::make_unique<WrongSizeModel>(*this);
    }
};

class NaNModel final : public PredictionModel {
public:
    std::string name() const override { return "NaN"; }
    std::size_t minimumHistory() const override { return 1; }
    std::vector<double> predict(const std::vector<double>&, std::size_t horizon) const override {
        return std::vector<double>(horizon, std::numeric_limits<double>::quiet_NaN());
    }
    std::unique_ptr<PredictionModel> clone() const override {
        return std::make_unique<NaNModel>(*this);
    }
};

Ensemble twoConstants(double a, double b) {
    Ensemble e;
    e.addModel(std::make_unique<ConstantModel>(a));
    e.addModel(std::make_unique<ConstantModel>(b));
    return e;
}

}  // namespace

int main() {
    const std::vector<double> prices{1, 2, 3, 4, 5};

    // ---- Equal weights ----
    {
        Ensemble e = twoConstants(10.0, 20.0);
        auto r = e.forecast(prices, 3);
        check(r.horizon == 3 && r.combined.size() == 3, "result sizes match horizon");
        check(approx(r.combined[0], 15.0) && approx(r.combined[2], 15.0),
              "equal-weight combined forecast is the mean");
        check(approx(r.spread[0], 10.0), "spread = max - min");
        check(approx(r.stdDev[0], 5.0), "std dev of two equally weighted models");
        check(r.modelsUsed == 2 && r.warnings.empty(), "both models used, no warnings");
        check(approx(r.models[0].weight, 0.5) && approx(r.models[1].weight, 0.5),
              "effective weights are 0.5 / 0.5");
        check(approx(r.lastClose, 5.0), "lastClose recorded");
    }

    // ---- Custom weights ----
    {
        Ensemble e = twoConstants(10.0, 20.0);
        e.setWeights({3.0, 1.0});
        check(approx(e.weights()[0], 0.75) && approx(e.weights()[1], 0.25),
              "weights are normalized");
        auto r = e.forecast(prices, 1);
        check(approx(r.combined[0], 12.5), "weighted combination (10*0.75 + 20*0.25)");
        check(approx(r.stdDev[0], std::sqrt(18.75)), "weighted std dev matches hand calculation");

        // Zero weight: model still reported, but excluded from diagnostics.
        e.setWeights({1.0, 0.0});
        auto z = e.forecast(prices, 1);
        check(approx(z.combined[0], 10.0) && z.modelsUsed == 1 && approx(z.spread[0], 0.0),
              "zero-weight model ignored in combination and diagnostics");
        check(z.models[1].succeeded && approx(z.models[1].forecast[0], 20.0),
              "zero-weight model's forecast is still preserved");
    }

    // ---- Weight validation ----
    {
        Ensemble e = twoConstants(1.0, 2.0);
        expectThrows<InvalidInputError>([&] { e.setWeights({1.0}); }, "wrong weight count rejected");
        expectThrows<InvalidInputError>([&] { e.setWeights({1.0, -1.0}); }, "negative weight rejected");
        expectThrows<InvalidInputError>(
            [&] { e.setWeights({1.0, std::numeric_limits<double>::quiet_NaN()}); },
            "NaN weight rejected");
        expectThrows<InvalidInputError>([&] { e.setWeights({0.0, 0.0}); }, "all-zero weights rejected");
        check(approx(e.weights()[0], 0.5), "failed setWeights leaves weights unchanged");

        e.setWeights({3.0, 1.0});
        e.addModel(std::make_unique<ConstantModel>(3.0));
        check(e.weights().size() == 3 && approx(e.weights()[0], 1.0 / 3.0),
              "adding a model resets to equal weights");
    }

    // ---- Failing / misbehaving models are excluded safely ----
    {
        Ensemble e;
        e.addModel(std::make_unique<ConstantModel>(10.0));
        e.addModel(std::make_unique<ThrowingModel>());
        e.addModel(std::make_unique<ConstantModel>(20.0));
        auto r = e.forecast(prices, 2);
        check(r.models.size() == 3 && !r.models[1].succeeded && !r.models[1].error.empty(),
              "throwing model recorded as failed with its message");
        check(approx(r.combined[0], 15.0) && r.modelsUsed == 2,
              "combined forecast uses remaining models");
        check(approx(r.models[0].weight, 0.5) && approx(r.models[1].weight, 0.0),
              "weights renormalized over successful models");
        check(!r.warnings.empty(), "warnings explain the exclusion");
        check(approx(e.weights()[0], 1.0 / 3.0), "stored weights unchanged by a failure");
    }
    {
        Ensemble e;
        e.addModel(std::make_unique<ConstantModel>(10.0));
        e.addModel(std::make_unique<WrongSizeModel>());
        e.addModel(std::make_unique<NaNModel>());
        auto r = e.forecast(prices, 2);
        check(!r.models[1].succeeded && !r.models[2].succeeded && r.modelsUsed == 1,
              "wrong-size and NaN outputs treated as failures");
        check(approx(r.combined[0], 10.0), "surviving model carries the forecast");
    }
    {
        Ensemble e;
        e.addModel(std::make_unique<ThrowingModel>());
        e.addModel(std::make_unique<NaNModel>());
        expectThrows<EnsembleError>([&] { e.forecast(prices, 1); }, "all models failing throws");

        Ensemble f = twoConstants(1.0, 2.0);
        f.addModel(std::make_unique<ThrowingModel>());
        f.setWeights({0.0, 0.0, 1.0});
        expectThrows<EnsembleError>([&] { f.forecast(prices, 1); },
                                    "only positively weighted model failing throws");
    }

    // ---- Real models through the shared interface ----
    {
        Ensemble e;
        e.addModel(std::make_unique<SMA>(3));
        e.addModel(std::make_unique<WMA>(3));
        e.addModel(std::make_unique<LinearRegression>());
        auto r = e.forecast({1, 3, 5, 7, 9, 11}, 2);

        check(r.models[0].name == "SMA" && r.models[1].description == "WMA(window=3)" &&
                  r.models[2].name == "LinearRegression",
              "model names/descriptions preserved in registration order");
        check(approx(r.models[0].forecast[0], 9.0) && approx(r.models[1].forecast[0], 58.0 / 6.0) &&
                  approx(r.models[2].forecast[0], 13.0),
              "individual forecasts match the individual models");
        check(approx(r.combined[0], (9.0 + 58.0 / 6.0 + 13.0) / 3.0),
              "combined = equal-weight mean of SMA, WMA, regression");
        check(r.spread[0] > 0.0 && r.stdDev[0] > 0.0, "models disagree on a trending series");
        check(e.minimumHistory() == 3, "ensemble minimumHistory is the max of its models");
    }
    {
        // A model with too little history is excluded; the rest still forecast.
        Ensemble e;
        e.addModel(std::make_unique<SMA>(3));
        e.addModel(std::make_unique<SMA>(50));
        auto r = e.forecast({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 1);
        check(r.models[0].succeeded && !r.models[1].succeeded && r.modelsUsed == 1,
              "insufficient history excludes only that model");
        check(e.minimumHistory() == 50, "minimumHistory reports the most demanding model");
    }

    // ---- Argument validation ----
    {
        Ensemble empty;
        expectThrows<EnsembleError>([&] { empty.forecast(prices, 1); }, "no models registered rejected");

        Ensemble e = twoConstants(1.0, 2.0);
        expectThrows<InvalidInputError>([&] { e.forecast(prices, 0); }, "horizon 0 rejected");
        expectThrows<InsufficientDataError>([&] { e.forecast({}, 1); }, "empty price history rejected");
        expectThrows<InvalidInputError>([&] { e.addModel(nullptr); }, "null model rejected");
        expectThrows<std::out_of_range>([&] { e.model(5); }, "model(index) bounds-checked");
        check(e.model(0).name() == "Const", "model(index) gives read-only access");
    }

    // ---- Copy and move semantics ----
    {
        Ensemble a = twoConstants(10.0, 20.0);
        a.setWeights({3.0, 1.0});

        Ensemble b(a);                       // copy constructor (deep copy via clone)
        a.addModel(std::make_unique<ConstantModel>(30.0));
        check(b.modelCount() == 2 && a.modelCount() == 3, "copy is independent of the original");
        check(approx(b.forecast(prices, 1).combined[0], 12.5), "copy keeps models and weights");

        Ensemble c;
        c = b;                               // copy assignment
        check(c.modelCount() == 2 && approx(c.weights()[0], 0.75), "copy assignment works");

        Ensemble d(std::move(c));            // move constructor
        check(d.modelCount() == 2, "move transfers the models");
    }

    return finish();
}