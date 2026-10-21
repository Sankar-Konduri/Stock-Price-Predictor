#include "ensemble/Ensemble.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <utility>

namespace spp {

Ensemble::Ensemble(const Ensemble& other) : weights_(other.weights_) {
    models_.reserve(other.models_.size());
    for (const auto& m : other.models_) {
        models_.push_back(m->clone());  // deep copy through the base interface
    }
}

Ensemble& Ensemble::operator=(const Ensemble& other) {
    if (this != &other) {
        Ensemble copy(other);  // copy first so *this stays intact if cloning throws
        models_.swap(copy.models_);
        weights_.swap(copy.weights_);
    }
    return *this;
}

void Ensemble::addModel(std::unique_ptr<PredictionModel> model) {
    if (!model) {
        throw InvalidInputError("Ensemble: cannot register a null model.");
    }
    models_.push_back(std::move(model));
    setEqualWeights();
}

const PredictionModel& Ensemble::model(std::size_t index) const {
    if (index >= models_.size()) {
        throw std::out_of_range("Ensemble: model index " + std::to_string(index) +
                                " is out of range.");
    }
    return *models_[index];
}

std::size_t Ensemble::minimumHistory() const {
    std::size_t needed = 0;
    for (const auto& m : models_) {
        needed = std::max(needed, m->minimumHistory());
    }
    return needed;
}

void Ensemble::setEqualWeights() {
    if (models_.empty()) {
        weights_.clear();
        return;
    }
    weights_.assign(models_.size(), 1.0 / static_cast<double>(models_.size()));
}

void Ensemble::setWeights(const std::vector<double>& weights) {
    if (weights.size() != models_.size()) {
        throw InvalidInputError("Ensemble: expected " + std::to_string(models_.size()) +
                                " weight(s) but received " +
                                std::to_string(weights.size()) + ".");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (!std::isfinite(weights[i]) || weights[i] < 0.0) {
            throw InvalidInputError("Ensemble: weight at index " + std::to_string(i) +
                                    " must be finite and non-negative.");
        }
        sum += weights[i];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw InvalidInputError("Ensemble: weights must have a positive, finite sum.");
    }

    weights_.resize(weights.size());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights_[i] = weights[i] / sum;
    }
}

EnsembleResult Ensemble::forecast(const std::vector<double>& closingPrices,
                                  std::size_t horizon) const {
    if (models_.empty()) {
        throw EnsembleError("Ensemble: no models are registered.");
    }
    if (horizon < 1) {
        throw InvalidInputError("Ensemble: forecast horizon must be at least 1.");
    }
    if (closingPrices.empty()) {
        throw InsufficientDataError("Ensemble: no closing prices were provided.");
    }

    EnsembleResult result;
    result.horizon = horizon;
    result.lastClose = closingPrices.back();
    result.models.reserve(models_.size());

    // 1) Run every model through the common interface. A failing or
    //    misbehaving model is excluded; it never aborts the ensemble.
    std::vector<std::size_t> usable;  // indices of models that produced valid output
    for (std::size_t i = 0; i < models_.size(); ++i) {
        ModelForecast mf;
        mf.name = models_[i]->name();
        mf.description = models_[i]->describe();
        try {
            std::vector<double> f = models_[i]->predict(closingPrices, horizon);
            if (f.size() != horizon) {
                throw EnsembleError("returned " + std::to_string(f.size()) +
                                    " values instead of " + std::to_string(horizon));
            }
            for (double v : f) {
                if (!std::isfinite(v)) {
                    throw EnsembleError("returned a non-finite forecast value");
                }
            }
            mf.forecast = std::move(f);
            mf.succeeded = true;
            usable.push_back(i);
        } catch (const std::exception& e) {
            mf.error = e.what();
        }
        result.models.push_back(std::move(mf));
    }

    // Summary of failures (used in warnings and error messages).
    std::string failureDetail;
    for (const ModelForecast& mf : result.models) {
        if (!mf.succeeded) {
            if (!failureDetail.empty()) failureDetail += "; ";
            failureDetail += mf.description + ": " + mf.error;
        }
    }

    if (usable.empty()) {
        throw EnsembleError("Ensemble: all " + std::to_string(models_.size()) +
                            " model(s) failed (" + failureDetail + ").");
    }

    // 2) Renormalize weights over the models that actually succeeded.
    double weightSum = 0.0;
    for (std::size_t i : usable) weightSum += weights_[i];
    if (!(weightSum > 0.0)) {
        throw EnsembleError(
            "Ensemble: every model with a non-zero weight failed (" + failureDetail + ").");
    }
    for (std::size_t i : usable) {
        result.models[i].weight = weights_[i] / weightSum;
    }

    for (const ModelForecast& mf : result.models) {
        if (!mf.succeeded) {
            result.warnings.push_back(mf.description + " was excluded: " + mf.error);
        }
    }
    if (usable.size() < models_.size()) {
        result.warnings.push_back("Weights were renormalized over the " +
                                  std::to_string(usable.size()) + " remaining model(s).");
    }

    // 3) Combined forecast and disagreement diagnostics, step by step.
    const double inf = std::numeric_limits<double>::infinity();
    result.combined.assign(horizon, 0.0);
    result.spread.assign(horizon, 0.0);
    result.stdDev.assign(horizon, 0.0);

    for (std::size_t step = 0; step < horizon; ++step) {
        double combined = 0.0;
        double lo = inf;
        double hi = -inf;
        for (std::size_t i : usable) {
            const ModelForecast& mf = result.models[i];
            combined += mf.weight * mf.forecast[step];
            if (mf.weight > 0.0) {
                lo = std::min(lo, mf.forecast[step]);
                hi = std::max(hi, mf.forecast[step]);
            }
        }

        double variance = 0.0;
        for (std::size_t i : usable) {
            const ModelForecast& mf = result.models[i];
            const double diff = mf.forecast[step] - combined;
            variance += mf.weight * diff * diff;
        }

        result.combined[step] = combined;
        result.spread[step] = hi - lo;  // at least one model has weight > 0
        result.stdDev[step] = std::sqrt(variance);
    }

    for (std::size_t i : usable) {
        if (result.models[i].weight > 0.0) ++result.modelsUsed;
    }

    return result;
}

}  // namespace spp