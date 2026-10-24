#include "evaluation/AdaptiveEvaluator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <sstream>
#include <utility>

#include "ensemble/Weighting.hpp"

namespace spp {

namespace {

using StepSeries = std::vector<std::vector<double>>;  // [forecast step][test origin]

// Every model's forecast at one origin, plus the later actual prices.
// `actual` is stored so the origin can be scored, but weights only ever use
// records that are fully scorable at the time (see scorableCount).
struct OriginRecord {
    std::size_t origin = 0;
    std::vector<double> actual;                   // [step]
    std::vector<std::vector<double>> predicted;   // [model][step]
    std::vector<double> mse;                      // [model] mean squared error over steps
};

ModelEvaluation buildEvaluation(std::string name, std::string description,
                                const StepSeries& actualByStep,
                                const StepSeries& predictedByStep) {
    ModelEvaluation ev;
    ev.name = std::move(name);
    ev.description = std::move(description);

    std::vector<double> allActual;
    std::vector<double> allPredicted;
    for (std::size_t s = 0; s < actualByStep.size(); ++s) {
        ev.byStep.push_back(
            ModelEvaluator::computeMetrics(actualByStep[s], predictedByStep[s]));
        allActual.insert(allActual.end(), actualByStep[s].begin(), actualByStep[s].end());
        allPredicted.insert(allPredicted.end(), predictedByStep[s].begin(),
                            predictedByStep[s].end());
    }
    ev.overall = ModelEvaluator::computeMetrics(allActual, allPredicted);
    return ev;
}

std::optional<double> percentChange(double value, double reference) {
    if (reference > 0.0) {
        return (value - reference) / reference * 100.0;
    }
    return std::nullopt;
}

std::string formatMape(const ErrorMetrics& m) {
    if (!m.mape) return "n/a";
    std::ostringstream os;
    os << std::fixed << std::setprecision(3) << *m.mape;
    return os.str();
}

// Number of records (they are sorted by origin) whose forecasts are fully
// scorable at time t: origin + horizon <= t. These are the only forecasts
// whose outcomes are known when a decision is made at origin t.
std::size_t scorableCount(const std::vector<OriginRecord>& recs, std::size_t h,
                          std::size_t t) {
    if (t < h) return 0;
    const std::size_t latestScorable = t - h;
    const auto it = std::upper_bound(
        recs.begin(), recs.end(), latestScorable,
        [](std::size_t value, const OriginRecord& r) { return value < r.origin; });
    return static_cast<std::size_t>(it - recs.begin());
}

// Weights at time t from the `window` most recent scorable forecasts.
// Caller guarantees scorableCount(recs, h, t) >= window.
std::vector<double> weightsAt(const std::vector<OriginRecord>& recs, std::size_t h,
                              std::size_t t, std::size_t window, std::size_t modelCount,
                              double minRelativeError) {
    const std::size_t end = scorableCount(recs, h, t);
    const std::size_t begin = end - window;

    std::vector<double> rmse(modelCount);
    for (std::size_t m = 0; m < modelCount; ++m) {
        double sum = 0.0;
        for (std::size_t j = begin; j < end; ++j) {
            sum += recs[j].mse[m];
        }
        rmse[m] = std::sqrt(sum / static_cast<double>(window));
    }
    return inverseErrorWeights(rmse, minRelativeError);
}

constexpr double kTieTolerance = 1e-12;

void tally(WeightingComparison& cmp, double schemeError, double equalError) {
    if (schemeError < equalError - kTieTolerance) {
        ++cmp.wins;
    } else if (schemeError > equalError + kTieTolerance) {
        ++cmp.losses;
    } else {
        ++cmp.ties;
    }
}

}  // namespace

AdaptiveReport AdaptiveEvaluator::run(const Ensemble& ensemble,
                                      const std::vector<double>& closes,
                                      const AdaptiveConfig& config) const {
    if (ensemble.modelCount() == 0) {
        throw InvalidInputError("AdaptiveEvaluator: the ensemble has no models.");
    }
    if (config.horizon < 1) {
        throw InvalidInputError("AdaptiveEvaluator: horizon must be at least 1.");
    }
    if (config.step < 1) {
        throw InvalidInputError("AdaptiveEvaluator: step must be at least 1.");
    }
    if (config.validationWindow < 1) {
        throw InvalidInputError("AdaptiveEvaluator: validationWindow must be at least 1.");
    }
    if (!(config.minRelativeError > 0.0 && config.minRelativeError <= 1.0)) {
        throw InvalidInputError("AdaptiveEvaluator: minRelativeError must be in (0, 1].");
    }
    for (std::size_t i = 0; i < closes.size(); ++i) {
        if (!std::isfinite(closes[i])) {
            throw InvalidInputError("AdaptiveEvaluator: closing price at index " +
                                    std::to_string(i) + " is not finite.");
        }
    }

    const std::size_t n = closes.size();
    const std::size_t h = config.horizon;
    const std::size_t window = config.validationWindow;
    const std::size_t modelCount = ensemble.modelCount();
    const std::size_t minOrigin = std::max<std::size_t>(ensemble.minimumHistory(), 1);

    if (n < minOrigin + h) {
        throw InsufficientDataError(
            "AdaptiveEvaluator: need at least " + std::to_string(minOrigin + h) +
            " closing prices, but only " + std::to_string(n) + " were provided.");
    }

    const std::size_t lastGrid = n - h;
    if (config.lastTestOrigin > lastGrid) {
        throw InvalidInputError("AdaptiveEvaluator: lastTestOrigin " +
                                std::to_string(config.lastTestOrigin) +
                                " leaves fewer than " + std::to_string(h) +
                                " actual values to score against (maximum is " +
                                std::to_string(lastGrid) + ").");
    }
    if (config.firstTestOrigin != 0 && config.lastTestOrigin != 0 &&
        config.firstTestOrigin > config.lastTestOrigin) {
        throw InvalidInputError("AdaptiveEvaluator: firstTestOrigin is after lastTestOrigin.");
    }
    const std::size_t gridLast = (config.lastTestOrigin == 0) ? lastGrid : config.lastTestOrigin;

    // ---- 1) Every model's forecast at every origin (computed once) -----------
    // `history` only ever holds closes[0..origin): nothing from the forecast
    // period is visible to the models.
    std::vector<OriginRecord> recs;
    std::size_t skipped = 0;
    std::vector<double> history(closes.begin(),
                                closes.begin() + static_cast<std::ptrdiff_t>(minOrigin));

    for (std::size_t origin = minOrigin;;) {
        while (history.size() < origin) {
            history.push_back(closes[history.size()]);
        }

        bool usable = false;
        EnsembleResult result;
        try {
            result = ensemble.forecast(history, h);
            usable = std::all_of(result.models.begin(), result.models.end(),
                                 [](const ModelForecast& mf) { return mf.succeeded; });
        } catch (const std::exception&) {
            usable = false;
        }

        if (usable) {
            OriginRecord rec;
            rec.origin = origin;
            for (std::size_t s = 0; s < h; ++s) {
                rec.actual.push_back(closes[origin + s]);
            }
            rec.predicted.resize(modelCount);
            rec.mse.assign(modelCount, 0.0);
            for (std::size_t m = 0; m < modelCount; ++m) {
                rec.predicted[m] = result.models[m].forecast;
                double sq = 0.0;
                for (std::size_t s = 0; s < h; ++s) {
                    const double e = rec.predicted[m][s] - rec.actual[s];
                    sq += e * e;
                }
                rec.mse[m] = sq / static_cast<double>(h);
            }
            recs.push_back(std::move(rec));
        } else {
            ++skipped;
        }

        if (gridLast - origin < config.step) break;
        origin += config.step;
    }

    if (recs.empty()) {
        throw InsufficientDataError(
            "AdaptiveEvaluator: no forecast origin produced forecasts from every model.");
    }

    // ---- 2) Resolve the test range -------------------------------------------
    std::size_t firstIdx = recs.size();
    if (config.firstTestOrigin == 0) {
        for (std::size_t k = 0; k < recs.size(); ++k) {
            if (scorableCount(recs, h, recs[k].origin) >= window) {
                firstIdx = k;
                break;
            }
        }
        if (firstIdx == recs.size()) {
            throw InsufficientDataError(
                "AdaptiveEvaluator: not enough history. A validation window of " +
                std::to_string(window) + " scored forecast(s) is needed before the first "
                "test origin, but only " + std::to_string(recs.size()) +
                " usable forecast origin(s) exist.");
        }
    } else {
        for (std::size_t k = 0; k < recs.size(); ++k) {
            if (recs[k].origin >= config.firstTestOrigin) {
                firstIdx = k;
                break;
            }
        }
        if (firstIdx == recs.size()) {
            throw InsufficientDataError(
                "AdaptiveEvaluator: no usable forecast origin at or after firstTestOrigin " +
                std::to_string(config.firstTestOrigin) + ".");
        }
        const std::size_t available = scorableCount(recs, h, recs[firstIdx].origin);
        if (available < window) {
            throw InsufficientDataError(
                "AdaptiveEvaluator: only " + std::to_string(available) +
                " scored forecast(s) exist before test origin " +
                std::to_string(recs[firstIdx].origin) + ", but the validation window needs " +
                std::to_string(window) + ".");
        }
    }

    const std::size_t t0 = recs[firstIdx].origin;

    // ---- 3) Fixed-validation weights: computed once, from the period before t0 -
    const std::vector<double> fixedWeights =
        weightsAt(recs, h, t0, window, modelCount, config.minRelativeError);
    const std::size_t validationEnd = scorableCount(recs, h, t0);

    // ---- 4) Score everything on the test origins -------------------------------
    StepSeries actualS(h);
    std::vector<StepSeries> modelS(modelCount, StepSeries(h));
    StepSeries equalS(h);
    StepSeries fixedS(h);
    StepSeries rollingS(h);

    std::vector<double> rollingSum(modelCount, 0.0);
    std::vector<double> rollingLast(modelCount, 0.0);

    AdaptiveReport report;
    std::size_t testCount = 0;

    for (std::size_t k = firstIdx; k < recs.size(); ++k) {
        const OriginRecord& r = recs[k];
        // Rolling weights use only forecasts already scorable at this origin.
        const std::vector<double> rollingW =
            weightsAt(recs, h, r.origin, window, modelCount, config.minRelativeError);

        double aeEqual = 0.0;
        double aeFixed = 0.0;
        double aeRolling = 0.0;

        for (std::size_t s = 0; s < h; ++s) {
            double equal = 0.0;
            double fixed = 0.0;
            double rolling = 0.0;
            for (std::size_t m = 0; m < modelCount; ++m) {
                const double p = r.predicted[m][s];
                equal += p;
                fixed += fixedWeights[m] * p;
                rolling += rollingW[m] * p;
                modelS[m][s].push_back(p);
            }
            equal /= static_cast<double>(modelCount);

            actualS[s].push_back(r.actual[s]);
            equalS[s].push_back(equal);
            fixedS[s].push_back(fixed);
            rollingS[s].push_back(rolling);

            aeEqual += std::fabs(equal - r.actual[s]);
            aeFixed += std::fabs(fixed - r.actual[s]);
            aeRolling += std::fabs(rolling - r.actual[s]);
        }

        const double hd = static_cast<double>(h);
        tally(report.fixedVsEqual, aeFixed / hd, aeEqual / hd);
        tally(report.rollingVsEqual, aeRolling / hd, aeEqual / hd);

        for (std::size_t m = 0; m < modelCount; ++m) {
            rollingSum[m] += rollingW[m];
        }
        rollingLast = rollingW;
        ++testCount;
    }

    // ---- 5) Assemble the report ----------------------------------------------
    report.config = config;
    report.config.firstTestOrigin = t0;
    report.config.lastTestOrigin = recs.back().origin;
    report.observations = n;
    report.originsSkipped = skipped;
    report.testOrigins = testCount;
    report.validationFirstOrigin = recs[validationEnd - window].origin;
    report.validationLastOrigin = recs[validationEnd - 1].origin;

    for (std::size_t m = 0; m < modelCount; ++m) {
        report.models.push_back(buildEvaluation(ensemble.model(m).name(),
                                                ensemble.model(m).describe(),
                                                actualS, modelS[m]));
    }
    report.equalWeight = buildEvaluation("Equal-weight ensemble", "Equal-weight ensemble",
                                         actualS, equalS);
    report.fixedValidation = buildEvaluation("Fixed-validation ensemble",
                                             "Fixed-validation ensemble", actualS, fixedS);
    report.rollingAdaptive = buildEvaluation(
        "Rolling-adaptive ensemble",
        "Rolling-adaptive ensemble (window=" + std::to_string(window) + ")", actualS,
        rollingS);

    report.fixedWeights = fixedWeights;
    report.rollingLastWeights = rollingLast;
    report.rollingMeanWeights.resize(modelCount);
    for (std::size_t m = 0; m < modelCount; ++m) {
        report.rollingMeanWeights[m] = rollingSum[m] / static_cast<double>(testCount);
    }

    for (const ModelEvaluation& ev : report.models) {
        report.ranking.push_back(ComparisonRow{ev.description, ev.overall, 0, false});
    }
    for (const ModelEvaluation* ev :
         {&report.equalWeight, &report.fixedValidation, &report.rollingAdaptive}) {
        report.ranking.push_back(ComparisonRow{ev->description, ev->overall, 0, true});
    }
    std::stable_sort(report.ranking.begin(), report.ranking.end(),
                     [](const ComparisonRow& a, const ComparisonRow& b) {
                         if (a.overall.rmse != b.overall.rmse) return a.overall.rmse < b.overall.rmse;
                         return a.name < b.name;
                     });
    for (std::size_t i = 0; i < report.ranking.size(); ++i) {
        report.ranking[i].rank = i + 1;
    }

    const ErrorMetrics& eq = report.equalWeight.overall;
    report.fixedVsEqual.rmseChangePct = percentChange(report.fixedValidation.overall.rmse, eq.rmse);
    report.fixedVsEqual.maeChangePct = percentChange(report.fixedValidation.overall.mae, eq.mae);
    report.fixedVsEqual.lowerRmse = report.fixedValidation.overall.rmse < eq.rmse;
    report.rollingVsEqual.rmseChangePct = percentChange(report.rollingAdaptive.overall.rmse, eq.rmse);
    report.rollingVsEqual.maeChangePct = percentChange(report.rollingAdaptive.overall.mae, eq.mae);
    report.rollingVsEqual.lowerRmse = report.rollingAdaptive.overall.rmse < eq.rmse;

    return report;
}

std::string AdaptiveEvaluator::toText(const AdaptiveReport& r) {
    std::ostringstream os;
    os << "Adaptive weighting comparison (walk-forward)\n";
    os << "============================================\n";
    os << "Observations:        " << r.observations << "\n";
    os << "Forecast horizon:    " << r.config.horizon << " day(s)\n";
    os << "Validation window:   " << r.config.validationWindow << " scored forecast origin(s)\n";
    os << "Test origins:        " << r.testOrigins << " (first=" << r.config.firstTestOrigin
       << ", last=" << r.config.lastTestOrigin << ", skipped origins: " << r.originsSkipped
       << ")\n";
    os << "Fixed weights from:  forecasts made at origins " << r.validationFirstOrigin << " to "
       << r.validationLastOrigin << " (all scored before the first test origin)\n\n";

    os << "Ranking (by RMSE on the test origins, lower is better)\n";
    os << std::left << std::setw(6) << "Rank" << std::setw(38) << "Forecaster" << std::right
       << std::setw(11) << "MAE" << std::setw(11) << "RMSE" << std::setw(11) << "MAPE(%)" << "\n";
    for (const ComparisonRow& row : r.ranking) {
        os << std::left << std::setw(6) << row.rank << std::setw(38) << row.name << std::right
           << std::fixed << std::setprecision(4) << std::setw(11) << row.overall.mae
           << std::setw(11) << row.overall.rmse << std::setw(11) << formatMape(row.overall)
           << "\n";
    }

    os << "\nModel weights\n";
    os << std::left << std::setw(38) << "Model" << std::right << std::setw(10) << "Fixed"
       << std::setw(15) << "Rolling mean" << std::setw(15) << "Rolling last" << "\n";
    for (std::size_t m = 0; m < r.models.size(); ++m) {
        os << std::left << std::setw(38) << r.models[m].description << std::right << std::fixed
           << std::setprecision(4) << std::setw(10) << r.fixedWeights[m] << std::setw(15)
           << r.rollingMeanWeights[m] << std::setw(15) << r.rollingLastWeights[m] << "\n";
    }

    auto describe = [&](const std::string& label, const ModelEvaluation& ev,
                        const WeightingComparison& c) {
        os << "  " << label << "\n";
        os << std::fixed << std::setprecision(4) << "    RMSE " << ev.overall.rmse
           << " vs equal-weight " << r.equalWeight.overall.rmse;
        if (c.rmseChangePct) {
            os << std::showpos << std::setprecision(1) << " (" << *c.rmseChangePct << "%)"
               << std::noshowpos << std::setprecision(4);
        }
        os << "\n    MAE  " << ev.overall.mae << " vs equal-weight " << r.equalWeight.overall.mae;
        if (c.maeChangePct) {
            os << std::showpos << std::setprecision(1) << " (" << *c.maeChangePct << "%)"
               << std::noshowpos << std::setprecision(4);
        }
        os << "\n    Lower absolute error than equal-weight on " << c.wins << " of "
           << r.testOrigins << " test origins (" << c.losses << " worse, " << c.ties
           << " tied)\n";
        os << "    Result: " << (c.lowerRmse ? "RMSE is lower than equal-weight.\n"
                                             : "RMSE is NOT lower than equal-weight.\n");
    };

    os << "\nAdaptive vs equal-weight\n";
    describe("Fixed-validation weighting", r.fixedValidation, r.fixedVsEqual);
    describe("Rolling-adaptive weighting", r.rollingAdaptive, r.rollingVsEqual);

    os << "\nNote: weights are inverse-RMSE with a floor. Results come from one historical "
          "sample ("
       << r.testOrigins << " test origins) and do not by themselves prove future accuracy; "
          "check the win/loss counts, not only the averages.\n";
    return os.str();
}

}  // namespace spp