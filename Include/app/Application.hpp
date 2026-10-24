#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include "data/DataLoader.hpp"
#include "data/StockRecord.hpp"
#include "ensemble/Ensemble.hpp"

namespace spp {

/// User-adjustable model and forecast settings.
struct ModelSettings {
    std::size_t smaWindow = 5;
    std::size_t wmaWindow = 5;
    std::size_t regressionLookback = 0;                // 0 = fit on all history
    std::size_t horizon = 5;                           // trading days to forecast
    std::optional<std::vector<double>> customWeights;  // empty = equal weights
};

/**
 * Menu-driven command-line application. It coordinates the other components
 * (DataLoader, Ensemble, ...) and owns the user-visible state: loaded data,
 * settings, and the most recent forecast.
 *
 * Input and output streams are injected so the whole menu can be driven by
 * scripted input in tests.
 */
class Application {
public:
    Application(std::istream& in, std::ostream& out);

    /// Runs the menu loop until the user exits or input ends. Returns 0.
    int run();

private:
    enum class InputStatus { Ok, Invalid, Eof };

    // ---- input helpers ----
    bool readLine(const std::string& prompt, std::string& line);  // false at end of input
    InputStatus promptSize(const std::string& prompt, std::size_t low, std::size_t high,
                           std::size_t& value);

    // ---- menu actions (those that read input return false if input ended) ----
    bool loadData();
    void viewData();
    bool configure();
    void generateForecast();
    void showIndividualPredictions();

    // ---- helpers ----
    bool requireData();
    Ensemble buildEnsemble() const;
    bool computeForecast();
    void invalidateForecast();

    // ---- output ----
    void printMenu() const;
    void printConfigMenu() const;
    void printSummary() const;
    void printRecordsPreview() const;
    void printForecastReport(const EnsembleResult& result) const;
    void printIndividualPredictions(const EnsembleResult& result) const;
    void printWeights(const EnsembleResult& result) const;
    void printChanges(const EnsembleResult& result) const;
    void printDiagnostics(const EnsembleResult& result) const;

    std::istream& in_;
    std::ostream& out_;

    std::vector<StockRecord> records_;
    std::string dataPath_;
    ModelSettings settings_;
    std::optional<EnsembleResult> lastForecast_;
};

}  // namespace spp