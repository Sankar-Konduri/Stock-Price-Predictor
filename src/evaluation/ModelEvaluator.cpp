#include "evaluation/ModelEvaluator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <sstream>
#include <utility>

namespace spp {

namespace {

// Actual values this close to zero are excluded from MAPE (division by ~0).
constexpr double kZeroActual = 1e-12;

using StepSeries = std::vector<std::vector<double>>;  // [forecast step][origin]

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

}  // namespace

ErrorMetrics ModelEvaluator::computeMetrics(const std::vector<double>& actual,
                                            const std::vector<double>& predicted) {
    if (actual.size() != predicted.size()) {
        throw InvalidInputError("ModelEvaluator: actual and predicted sizes differ (" +
                                std::to_string(actual.size()) + " vs " +
                                std::to_string(predicted.size()) + ").");
    }
    if (actual.empty()) {
        throw InvalidInputError("ModelEvaluator: no values to evaluate.");
    }

    double absSum = 0.0;
    double sqSum = 0.0;
    double pctSum = 0.0;
    std::size_t pctCount = 0;
    std::size_t excluded = 0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(predicted[i])) {
            throw InvalidInputError("ModelEvaluator: non-finite value at index " +
                                    std::to_string(i) + ".");
        }
        const double error = predicted[i] - actual[i];
        absSum += std::fabs(error);
        sqSum += error * error;
        if (std::fabs(actual[i]) < kZeroActual) {
            ++excluded;
        } else {
            pctSum += std::fabs(error / actual[i]);
            ++pctCount;
        }
    }

    const double n = static_cast<double>(actual.size());
    ErrorMetrics m;
    m.count = actual.size();
    m.mae = absSum / n;
    m.rmse = std::sqrt(sqSum / n);
    if (pctCount > 0) {
        m.mape = 100.0 * pctSum / static_cast<double>(pctCount);
    }
    m.mapeExcluded = excluded;
    return m;
}

BacktestReport ModelEvaluator::backtest(const Ensemble& ensemble,
                                        const std::vector<double>& closes,
                                        const BacktestConfig& config) const {
    if (ensemble.modelCount() == 0) {
        throw InvalidInputError("ModelEvaluator: the ensemble has no models.");
    }
    if (config.horizon < 1) {
        throw InvalidInputError("ModelEvaluator: horizon must be at least 1.");
    }
    if (config.step < 1) {
        throw InvalidInputError("ModelEvaluator: step must be at least 1.");
    }
    for (std::size_t i = 0; i < closes.size(); ++i) {
        if (!std::isfinite(closes[i])) {
            throw InvalidInputError("ModelEvaluator: closing price at index " +
                                    std::to_string(i) + " is not finite.");
        }
    }

    const std::size_t n = closes.size();
    const std::size_t h = config.horizon;
    const std::size_t minOrigin = std::max<std::size_t>(ensemble.minimumHistory(), 1);

    if (n < minOrigin + h) {
        throw InsufficientDataError(
            "ModelEvaluator: need at least " + std::to_string(minOrigin + h) +
            " closing prices (" + std::to_string(minOrigin) +
            " for the most demanding model plus " + std::to_string(h) +
            " to score), but only " + std::to_string(n) + " were provided.");
    }

    // Resolve the origin range. Every model must be able to forecast at every origin.
    const std::size_t first = (config.firstOrigin == 0) ? minOrigin : config.firstOrigin;
    const std::size_t last = (config.lastOrigin == 0) ? n - h : config.lastOrigin;
    if (first < minOrigin) {
        throw InvalidInputError(
            "ModelEvaluator: firstOrigin " + std::to_string(first) +
            " is below the ensemble's minimum history (" + std::to_string(minOrigin) +
            "); every model must be able to forecast at every origin.");
    }
    if (last > n - h) {
        throw InvalidInputError("ModelEvaluator: lastOrigin " + std::to_string(last) +
                                " leaves fewer than " + std::to_string(h) +
                                " actual values to score against (maximum is " +
                                std::to_string(n - h) + ").");
    }
    if (first > last) {
        throw InvalidInputError("ModelEvaluator: firstOrigin (" + std::to_string(first) +
                                ") is after lastOrigin (" + std::to_string(last) + ").");
    }

    const std::size_t modelCount = ensemble.modelCount();
    StepSeries actualByStep(h);
    std::vector<StepSeries> modelPredicted(modelCount, StepSeries(h));
    StepSeries ensemblePredicted(h);

    // `history` is the ONLY data handed to the models. It always holds exactly
    // closes[0..origin): nothing from the period being forecast is visible.
    std::vector<double> history(closes.begin(),
                                closes.begin() + static_cast<std::ptrdiff_t>(first));
    std::size_t evaluated = 0;
    std::size_t skipped = 0;

    for (std::size_t origin = first;;) {
        while (history.size() < origin) {
            history.push_back(closes[history.size()]);
        }

        bool usable = false;
        EnsembleResult result;
        try {
            result = ensemble.forecast(history, h);
            usable = std::all_of(result.models.begin(), result.models.end(),
                                 [](const ModelForecast& m) { return m.succeeded; });
        } catch (const std::exception&) {
            usable = false;
        }

        if (usable) {
            // Only now are the true future values read, to score the forecasts.
            for (std::size_t s = 0; s < h; ++s) {
                actualByStep[s].push_back(closes[origin + s]);
                ensemblePredicted[s].push_back(result.combined[s]);
                for (std::size_t m = 0; m < modelCount; ++m) {
                    modelPredicted[m][s].push_back(result.models[m].forecast[s]);
                }
            }
            ++evaluated;
        } else {
            ++skipped;
        }

        if (last - origin < config.step) break;
        origin += config.step;
    }

    if (evaluated == 0) {
        throw InsufficientDataError(
            "ModelEvaluator: no forecast origin produced forecasts from every model (" +
            std::to_string(skipped) + " origin(s) skipped).");
    }

    BacktestReport report;
    report.config = config;
    report.config.firstOrigin = first;
    report.config.lastOrigin = last;
    report.observations = n;
    report.originsEvaluated = evaluated;
    report.originsSkipped = skipped;

    for (std::size_t m = 0; m < modelCount; ++m) {
        report.models.push_back(buildEvaluation(ensemble.model(m).name(),
                                                ensemble.model(m).describe(),
                                                actualByStep, modelPredicted[m]));
    }
    report.ensemble = buildEvaluation(
        "Ensemble", "Ensemble of " + std::to_string(modelCount) + " model(s)",
        actualByStep, ensemblePredicted);

    // Ranked comparison (lowest RMSE first; ties broken by name for stable output).
    for (const ModelEvaluation& ev : report.models) {
        report.ranking.push_back(ComparisonRow{ev.description, ev.overall, 0, false});
    }
    report.ranking.push_back(ComparisonRow{report.ensemble.name, report.ensemble.overall, 0, true});
    std::stable_sort(report.ranking.begin(), report.ranking.end(),
                     [](const ComparisonRow& a, const ComparisonRow& b) {
                         if (a.overall.rmse != b.overall.rmse) return a.overall.rmse < b.overall.rmse;
                         return a.name < b.name;
                     });
    for (std::size_t i = 0; i < report.ranking.size(); ++i) {
        report.ranking[i].rank = i + 1;
    }

    // Does the ensemble actually beat its best component?
    const ModelEvaluation* best = &report.models.front();
    for (const ModelEvaluation& ev : report.models) {
        if (ev.overall.rmse < best->overall.rmse) best = &ev;
    }
    EnsembleComparison& cmp = report.ensembleVsBest;
    cmp.bestIndividual = best->description;
    cmp.bestRmse = best->overall.rmse;
    cmp.ensembleRmse = report.ensemble.overall.rmse;
    cmp.bestMae = best->overall.mae;
    cmp.ensembleMae = report.ensemble.overall.mae;
    cmp.rmseChangePct = percentChange(cmp.ensembleRmse, cmp.bestRmse);
    cmp.maeChangePct = percentChange(cmp.ensembleMae, cmp.bestMae);
    cmp.ensembleBeatsBest = cmp.ensembleRmse < cmp.bestRmse;

    return report;
}

std::string ModelEvaluator::toText(const BacktestReport& r) {
    std::ostringstream os;
    os << "Walk-forward backtest report\n";
    os << "============================\n";
    os << "Observations:       " << r.observations << "\n";
    os << "Forecast horizon:   " << r.config.horizon << " day(s)\n";
    os << "Forecast origins:   " << r.originsEvaluated << " evaluated, " << r.originsSkipped
       << " skipped (first=" << r.config.firstOrigin << ", last=" << r.config.lastOrigin
       << ", step=" << r.config.step << ")\n";
    os << "Forecasts compared: " << r.ensemble.overall.count << " per model\n\n";

    os << "Model comparison (ranked by RMSE, lower is better)\n";
    os << std::left << std::setw(6) << "Rank" << std::setw(34) << "Model" << std::right
       << std::setw(11) << "MAE" << std::setw(11) << "RMSE" << std::setw(11) << "MAPE(%)" << "\n";
    for (const ComparisonRow& row : r.ranking) {
        os << std::left << std::setw(6) << row.rank << std::setw(34) << row.name << std::right
           << std::fixed << std::setprecision(4) << std::setw(11) << row.overall.mae
           << std::setw(11) << row.overall.rmse << std::setw(11) << formatMape(row.overall)
           << "\n";
    }
    if (r.ensemble.overall.mapeExcluded > 0) {
        os << "(MAPE excludes " << r.ensemble.overall.mapeExcluded
           << " point(s) where the actual price was zero.)\n";
    }

    const EnsembleComparison& c = r.ensembleVsBest;
    os << "\nEnsemble vs best individual model (" << c.bestIndividual << ")\n";
    os << std::fixed << std::setprecision(4);
    os << "  RMSE: " << c.ensembleRmse << " vs " << c.bestRmse;
    if (c.rmseChangePct) {
        os << std::showpos << std::setprecision(1) << " (" << *c.rmseChangePct << "%)"
           << std::noshowpos << std::setprecision(4);
    }
    os << "\n  MAE:  " << c.ensembleMae << " vs " << c.bestMae;
    if (c.maeChangePct) {
        os << std::showpos << std::setprecision(1) << " (" << *c.maeChangePct << "%)"
           << std::noshowpos << std::setprecision(4);
    }
    os << "\n";
    os << (c.ensembleBeatsBest
               ? "Result: the ensemble's RMSE is lower than every individual model's.\n"
               : "Result: the ensemble's RMSE is NOT lower than the best individual model's.\n");
    os << "Note: this reflects one historical sample (" << r.originsEvaluated
       << " origins); it does not by itself prove future accuracy.\n";

    if (r.config.horizon > 1) {
        const std::size_t shown = std::min<std::size_t>(r.config.horizon, 10);
        os << "\nMAE by forecast step (+1 = next trading day)\n";
        os << std::left << std::setw(34) << "Model" << std::right;
        for (std::size_t s = 0; s < shown; ++s) {
            os << std::setw(9) << ("+" + std::to_string(s + 1));
        }
        os << "\n" << std::fixed << std::setprecision(4);

        auto printRow = [&](const ModelEvaluation& ev, const std::string& label) {
            os << std::left << std::setw(34) << label << std::right;
            for (std::size_t s = 0; s < shown; ++s) {
                os << std::setw(9) << ev.byStep[s].mae;
            }
            os << "\n";
        };
        for (const ModelEvaluation& ev : r.models) printRow(ev, ev.description);
        printRow(r.ensemble, r.ensemble.name);
        if (r.config.horizon > shown) {
            os << "(only the first " << shown << " steps are shown)\n";
        }
    }
    return os.str();
}

}  // namespace spp