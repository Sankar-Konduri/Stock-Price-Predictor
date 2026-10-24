#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "evaluation/AdaptiveEvaluator.hpp"
#include "models/LinearRegression.hpp"
#include "models/SMA.hpp"
#include "models/WMA.hpp"
#include "test_helpers.hpp"

using namespace spp;
using namespace spp_test;

namespace {

/// Fails whenever the history length is even.
class FlakyModel final : public PredictionModel {
public:
    std::string name() const override { return "Flaky"; }
    std::size_t minimumHistory() const override { return 1; }
    std::vector<double> predict(const std::vector<double>& prices,
                                std::size_t horizon) const override {
        validateInputs(prices, horizon);
        if (prices.size() % 2 == 0) throw InvalidInputError("flaky failure");
        return std::vector<double>(horizon, prices.back());
    }
    std::unique_ptr<PredictionModel> clone() const override {
        return std::make_unique<FlakyModel>(*this);
    }
};

// Trend plus two oscillations, so the models make different, non-zero errors.
std::vector<double> wavySeries(std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        v[i] = 100.0 + 0.5 * x + 3.0 * std::sin(0.7 * x) + 2.0 * std::sin(0.23 * x);
    }
    return v;
}

Ensemble makeEnsemble() {
    Ensemble e;
    e.addModel(std::make_unique<SMA>(3));
    e.addModel(std::make_unique<WMA>(3));
    e.addModel(std::make_unique<LinearRegression>());
    return e;
}

double sum(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0); }

}  // namespace

int main() {
    AdaptiveEvaluator evaluator;
    const Ensemble ensemble = makeEnsemble();
    const auto prices = wavySeries(60);

    // ---- Structure: auto-resolved test period (horizon 1) ----
    {
        AdaptiveConfig cfg;
        cfg.validationWindow = 10;
        auto r = evaluator.run(ensemble, prices, cfg);

        // Grid origins 3..59. First test origin t needs 10 forecasts scorable
        // (origin <= t-1), so t = 13; the validation forecasts are origins 3..12.
        check(r.config.firstTestOrigin == 13 && r.config.lastTestOrigin == 59,
              "test range resolved automatically");
        check(r.testOrigins == 47 && r.originsSkipped == 0, "47 test origins");
        check(r.validationFirstOrigin == 3 && r.validationLastOrigin == 12,
              "fixed weights come from origins strictly before the test period");
        check(r.models.size() == 3 && r.ranking.size() == 6, "3 models + 3 ensembles ranked");

        check(approx(sum(r.fixedWeights), 1.0) && approx(sum(r.rollingMeanWeights), 1.0) &&
                  approx(sum(r.rollingLastWeights), 1.0),
              "all weight vectors sum to 1");
        const std::size_t expectedCount = r.testOrigins * r.config.horizon;
        check(r.equalWeight.overall.count == expectedCount &&
                  r.fixedValidation.overall.count == expectedCount &&
                  r.rollingAdaptive.overall.count == expectedCount &&
                  r.models[0].overall.count == expectedCount,
              "every forecaster scored on identical forecasts");

        check(r.fixedVsEqual.wins + r.fixedVsEqual.losses + r.fixedVsEqual.ties == r.testOrigins &&
                  r.rollingVsEqual.wins + r.rollingVsEqual.losses + r.rollingVsEqual.ties ==
                      r.testOrigins,
              "win/loss/tie counts cover every test origin");

        bool sorted = true;
        for (std::size_t i = 1; i < r.ranking.size(); ++i) {
            sorted = sorted && r.ranking[i - 1].overall.rmse <= r.ranking[i].overall.rmse;
        }
        check(sorted, "ranking sorted by RMSE");

        // Cross-check against the Stage 5 evaluator on the same origins.
        BacktestConfig bt;
        bt.firstOrigin = 13;
        bt.lastOrigin = 59;
        auto plain = ModelEvaluator().backtest(ensemble, prices, bt);
        check(approx(plain.ensemble.overall.rmse, r.equalWeight.overall.rmse, 1e-9),
              "equal-weight result matches ModelEvaluator on the same origins");
        check(approx(plain.models[2].overall.rmse, r.models[2].overall.rmse, 1e-9),
              "individual model result matches ModelEvaluator");

        const std::string text = AdaptiveEvaluator::toText(r);
        check(text.find("Rolling-adaptive") != std::string::npos &&
                  text.find("Fixed-validation") != std::string::npos &&
                  text.find("Equal-weight") != std::string::npos,
              "text report mentions all three ensembles");
    }

    // ---- Explicit test start ----
    {
        AdaptiveConfig cfg;
        cfg.validationWindow = 10;
        cfg.firstTestOrigin = 30;
        auto r = evaluator.run(ensemble, prices, cfg);
        check(r.config.firstTestOrigin == 30 && r.testOrigins == 30, "explicit firstTestOrigin honoured");
        check(r.validationFirstOrigin == 20 && r.validationLastOrigin == 29,
              "validation window is the 10 origins just before the test start");
    }

    // ---- Multi-step horizon: forecasts only count once all their outcomes are known ----
    {
        AdaptiveConfig cfg;
        cfg.horizon = 3;
        cfg.validationWindow = 5;
        auto r = evaluator.run(ensemble, prices, cfg);
        // Scorable at t means origin <= t-3; first test origin is 10, validation origins 3..7.
        check(r.config.firstTestOrigin == 10, "multi-step: first test origin accounts for horizon");
        check(r.validationFirstOrigin == 3 && r.validationLastOrigin == 7,
              "multi-step: validation forecasts end at (first test origin - horizon)");
        check(r.equalWeight.byStep.size() == 3, "multi-step: per-step metrics kept");
    }

    // ---- No leakage: changing prices at/after the last test origin cannot change weights ----
    {
        AdaptiveConfig cfg;
        cfg.validationWindow = 10;
        cfg.lastTestOrigin = 30;

        auto a = wavySeries(40);
        auto b = a;
        for (std::size_t i = 30; i < b.size(); ++i) b[i] = b[i] * 5.0 + 50.0;  // alter the "future"

        auto ra = evaluator.run(ensemble, a, cfg);
        auto rb = evaluator.run(ensemble, b, cfg);

        bool sameWeights = true;
        for (std::size_t m = 0; m < ra.fixedWeights.size(); ++m) {
            sameWeights = sameWeights && approx(ra.fixedWeights[m], rb.fixedWeights[m]) &&
                          approx(ra.rollingLastWeights[m], rb.rollingLastWeights[m]) &&
                          approx(ra.rollingMeanWeights[m], rb.rollingMeanWeights[m]);
        }
        check(sameWeights, "weights are unaffected by prices from the forecast period itself");
        check(!approx(ra.rollingAdaptive.overall.rmse, rb.rollingAdaptive.overall.rmse),
              "...while the scores do change (the altered prices were really used for scoring)");
    }

    // ---- Failing models: origins skipped consistently ----
    {
        Ensemble flaky;
        flaky.addModel(std::make_unique<FlakyModel>());
        flaky.addModel(std::make_unique<SMA>(3));
        AdaptiveConfig cfg;
        cfg.validationWindow = 5;
        auto r = evaluator.run(flaky, wavySeries(40), cfg);
        check(r.originsSkipped == 18, "origins with a failing model are skipped (even histories)");
        check(r.config.firstTestOrigin == 13, "validation window counts only usable origins");
    }

    // ---- Validation ----
    {
        AdaptiveConfig cfg;
        cfg.validationWindow = 0;
        expectThrows<InvalidInputError>([&] { evaluator.run(ensemble, prices, cfg); },
                                        "validationWindow 0 rejected");

        cfg = AdaptiveConfig{};
        cfg.horizon = 0;
        expectThrows<InvalidInputError>([&] { evaluator.run(ensemble, prices, cfg); }, "horizon 0 rejected");

        cfg = AdaptiveConfig{};
        cfg.step = 0;
        expectThrows<InvalidInputError>([&] { evaluator.run(ensemble, prices, cfg); }, "step 0 rejected");

        cfg = AdaptiveConfig{};
        cfg.minRelativeError = 0.0;
        expectThrows<InvalidInputError>([&] { evaluator.run(ensemble, prices, cfg); },
                                        "minRelativeError 0 rejected");

        cfg = AdaptiveConfig{};
        cfg.validationWindow = 10;
        cfg.lastTestOrigin = 60;
        expectThrows<InvalidInputError>([&] { evaluator.run(ensemble, prices, cfg); },
                                        "lastTestOrigin without future data rejected");

        cfg = AdaptiveConfig{};
        cfg.validationWindow = 10;
        cfg.firstTestOrigin = 40;
        cfg.lastTestOrigin = 30;
        expectThrows<InvalidInputError>([&] { evaluator.run(ensemble, prices, cfg); },
                                        "firstTestOrigin after lastTestOrigin rejected");

        cfg = AdaptiveConfig{};
        cfg.validationWindow = 10;
        cfg.firstTestOrigin = 5;  // not enough scored forecasts before origin 5
        expectThrows<InsufficientDataError>([&] { evaluator.run(ensemble, prices, cfg); },
                                            "test start without a full validation window rejected");

        cfg = AdaptiveConfig{};  // default window of 20 on 8 prices
        expectThrows<InsufficientDataError>([&] { evaluator.run(ensemble, wavySeries(8), cfg); },
                                            "history too short for the validation window rejected");

        expectThrows<InvalidInputError>([&] { evaluator.run(Ensemble{}, prices); }, "empty ensemble rejected");
        expectThrows<InvalidInputError>(
            [&] {
                auto bad = prices;
                bad[10] = std::nan("");
                evaluator.run(ensemble, bad);
            },
            "non-finite price rejected");
    }

    return finish();
}