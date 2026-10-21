#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "evaluation/ModelEvaluator.hpp"
#include "models/LinearRegression.hpp"
#include "models/SMA.hpp"
#include "models/WMA.hpp"
#include "test_helpers.hpp"

using namespace spp;
using namespace spp_test;

namespace {

/// Records what history each forecast is allowed to see, and predicts "last value".
struct HistoryLog {
    std::vector<std::size_t> sizes;
    std::vector<double> lastValues;
};

class SpyModel final : public PredictionModel {
public:
    explicit SpyModel(HistoryLog* log) : log_(log) {}
    std::string name() const override { return "Spy"; }
    std::size_t minimumHistory() const override { return 3; }
    std::vector<double> predict(const std::vector<double>& prices,
                                std::size_t horizon) const override {
        validateInputs(prices, horizon);
        log_->sizes.push_back(prices.size());
        log_->lastValues.push_back(prices.back());
        return std::vector<double>(horizon, prices.back());
    }
    std::unique_ptr<PredictionModel> clone() const override {
        return std::make_unique<SpyModel>(*this);
    }
private:
    HistoryLog* log_;
};

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

class AlwaysFailsModel final : public PredictionModel {
public:
    std::string name() const override { return "AlwaysFails"; }
    std::size_t minimumHistory() const override { return 1; }
    std::vector<double> predict(const std::vector<double>&, std::size_t) const override {
        throw InvalidInputError("always fails");
    }
    std::unique_ptr<PredictionModel> clone() const override {
        return std::make_unique<AlwaysFailsModel>(*this);
    }
};

std::vector<double> linearSeries(std::size_t n) {  // 1, 2, ..., n
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<double>(i + 1);
    return v;
}

}  // namespace

int main() {
    ModelEvaluator evaluator;

    // ---- computeMetrics ----
    {
        auto m = ModelEvaluator::computeMetrics({100, 200, 400}, {110, 190, 380});
        check(m.count == 3 && approx(m.mae, 40.0 / 3.0), "MAE matches hand calculation");
        check(approx(m.rmse, std::sqrt(200.0)), "RMSE matches hand calculation");
        check(m.mape && approx(*m.mape, 100.0 * (0.1 + 0.05 + 0.05) / 3.0),
              "MAPE (percent) matches hand calculation");

        auto z = ModelEvaluator::computeMetrics({0, 100}, {5, 110});
        check(approx(z.mae, 7.5) && z.mapeExcluded == 1 && z.mape && approx(*z.mape, 10.0),
              "MAPE excludes zero actuals and counts them");

        auto allZero = ModelEvaluator::computeMetrics({0, 0}, {1, 2});
        check(!allZero.mape && allZero.mapeExcluded == 2, "all-zero actuals give no MAPE");

        expectThrows<InvalidInputError>([] { ModelEvaluator::computeMetrics({1, 2}, {1}); },
                                        "size mismatch rejected");
        expectThrows<InvalidInputError>([] { ModelEvaluator::computeMetrics({}, {}); },
                                        "empty input rejected");
        expectThrows<InvalidInputError>(
            [] { ModelEvaluator::computeMetrics({1, std::nan("")}, {1, 2}); },
            "non-finite value rejected");
    }

    // ---- Perfect forecaster on an exact line ----
    {
        Ensemble e;
        e.addModel(std::make_unique<LinearRegression>());
        BacktestConfig cfg;
        cfg.horizon = 3;
        auto r = evaluator.backtest(e, linearSeries(30), cfg);
        check(r.ensemble.overall.mae < 1e-9 && r.ensemble.overall.rmse < 1e-9,
              "regression on an exact line has ~zero error");
        check(r.models[0].byStep.size() == 3 && r.models[0].byStep[0].count == r.originsEvaluated,
              "byStep has one entry per horizon step");
    }

    // ---- Origin counting ----
    {
        Ensemble e;
        e.addModel(std::make_unique<SMA>(5));
        BacktestConfig cfg;
        cfg.horizon = 2;
        auto r = evaluator.backtest(e, linearSeries(20), cfg);
        check(r.config.firstOrigin == 5 && r.config.lastOrigin == 18,
              "origin range resolved automatically");
        check(r.originsEvaluated == 14 && r.originsSkipped == 0, "14 origins evaluated");
        check(r.ensemble.overall.count == 28, "forecasts compared = origins x horizon");

        cfg.step = 3;  // origins 5, 8, 11, 14, 17
        auto strided = evaluator.backtest(e, linearSeries(20), cfg);
        check(strided.originsEvaluated == 5, "stride skips origins");
    }

    // ---- No data leakage: the model only ever sees history before the origin ----
    {
        HistoryLog log;
        Ensemble e;
        e.addModel(std::make_unique<SpyModel>(&log));
        auto r = evaluator.backtest(e, linearSeries(12));  // origins 3..11

        bool sizesOk = log.sizes.size() == 9;
        for (std::size_t i = 0; sizesOk && i < log.sizes.size(); ++i) {
            sizesOk = (log.sizes[i] == i + 3) && approx(log.lastValues[i], static_cast<double>(i + 3));
        }
        check(sizesOk, "forecast at origin t sees exactly closes[0..t)");
        check(approx(r.ensemble.overall.mae, 1.0) && approx(r.ensemble.overall.rmse, 1.0),
              "naive forecast on a +1/day line has error exactly 1");

        // Restricting to a test period: only origins 8..10 are used.
        HistoryLog log2;
        Ensemble e2;
        e2.addModel(std::make_unique<SpyModel>(&log2));
        BacktestConfig cfg;
        cfg.firstOrigin = 8;
        cfg.lastOrigin = 10;
        auto sub = evaluator.backtest(e2, linearSeries(12), cfg);
        check(sub.originsEvaluated == 3 && log2.sizes == std::vector<std::size_t>({8, 9, 10}),
              "explicit origin range limits the evaluated period");
    }

    // ---- Skipped origins when a model fails ----
    {
        Ensemble e;
        e.addModel(std::make_unique<FlakyModel>());
        e.addModel(std::make_unique<SMA>(3));
        auto r = evaluator.backtest(e, linearSeries(12));  // origins 3..11; even sizes fail
        check(r.originsEvaluated == 5 && r.originsSkipped == 4,
              "origins where any model fails are skipped for all models");

        Ensemble bad;
        bad.addModel(std::make_unique<AlwaysFailsModel>());
        expectThrows<InsufficientDataError>([&] { evaluator.backtest(bad, linearSeries(10)); },
                                            "no usable origin rejected");
    }

    // ---- Ranking and ensemble comparison ----
    {
        Ensemble e;
        e.addModel(std::make_unique<SMA>(5));
        e.addModel(std::make_unique<WMA>(5));
        e.addModel(std::make_unique<LinearRegression>());
        auto r = evaluator.backtest(e, linearSeries(40));

        check(r.models.size() == 3 && r.models[0].name == "SMA" &&
                  r.models[1].description == "WMA(window=5)" && r.ensemble.name == "Ensemble",
              "models reported in registration order, ensemble separately");
        check(r.ranking.size() == 4 && r.ranking[0].name.find("LinearRegression") != std::string::npos,
              "regression ranks first on a perfectly linear series");

        bool sorted = true;
        for (std::size_t i = 0; i < r.ranking.size(); ++i) {
            sorted = sorted && r.ranking[i].rank == i + 1;
            if (i > 0) sorted = sorted && r.ranking[i - 1].overall.rmse <= r.ranking[i].overall.rmse;
        }
        check(sorted, "ranking is sorted by RMSE with ranks 1..N");
        check(r.ensembleVsBest.bestIndividual.find("LinearRegression") != std::string::npos &&
                  !r.ensembleVsBest.ensembleBeatsBest,
              "ensemble correctly reported as NOT beating the perfect model");

        const std::string text = ModelEvaluator::toText(r);
        check(text.find("Ensemble") != std::string::npos && text.find("RMSE") != std::string::npos &&
                  text.find("Rank") != std::string::npos,
              "text report contains the comparison table");
    }

    // ---- Multi-step report formatting ----
    {
        Ensemble e;
        e.addModel(std::make_unique<SMA>(3));
        BacktestConfig cfg;
        cfg.horizon = 3;
        auto r = evaluator.backtest(e, linearSeries(20), cfg);
        check(ModelEvaluator::toText(r).find("by forecast step") != std::string::npos,
              "multi-step report includes per-step table");
    }

    // ---- Configuration validation ----
    {
        Ensemble e;
        e.addModel(std::make_unique<SMA>(5));
        const auto prices = linearSeries(20);

        BacktestConfig cfg;
        cfg.horizon = 0;
        expectThrows<InvalidInputError>([&] { evaluator.backtest(e, prices, cfg); }, "horizon 0 rejected");

        cfg = BacktestConfig{};
        cfg.step = 0;
        expectThrows<InvalidInputError>([&] { evaluator.backtest(e, prices, cfg); }, "step 0 rejected");

        cfg = BacktestConfig{};
        cfg.firstOrigin = 3;  // below SMA(5)'s minimum history
        expectThrows<InvalidInputError>([&] { evaluator.backtest(e, prices, cfg); },
                                        "firstOrigin below minimum history rejected");

        cfg = BacktestConfig{};
        cfg.lastOrigin = 20;  // leaves no actual value to score against
        expectThrows<InvalidInputError>([&] { evaluator.backtest(e, prices, cfg); },
                                        "lastOrigin without future data rejected");

        cfg = BacktestConfig{};
        cfg.firstOrigin = 15;
        cfg.lastOrigin = 10;
        expectThrows<InvalidInputError>([&] { evaluator.backtest(e, prices, cfg); },
                                        "firstOrigin after lastOrigin rejected");

        expectThrows<InsufficientDataError>([&] { evaluator.backtest(e, linearSeries(5)); },
                                            "history too short rejected");
        expectThrows<InvalidInputError>(
            [&] {
                auto bad = prices;
                bad[7] = std::nan("");
                evaluator.backtest(e, bad);
            },
            "non-finite price rejected");
        expectThrows<InvalidInputError>([&] { evaluator.backtest(Ensemble{}, prices); },
                                        "empty ensemble rejected");
    }

    return finish();
}