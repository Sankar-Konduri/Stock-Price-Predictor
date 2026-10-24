#include "app/Application.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <utility>

#include "models/LinearRegression.hpp"
#include "models/SMA.hpp"
#include "models/WMA.hpp"

namespace spp {

namespace {

const char* const kDefaultDataPath = "data/historical_prices.csv";

// ---- small text helpers ------------------------------------------------------

std::string trim(const std::string& s) {
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string toLower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string fixedStr(double v, int precision) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << v;
    return os.str();
}

std::string signedStr(double v, int precision) {
    std::ostringstream os;
    os << std::showpos << std::fixed << std::setprecision(precision) << v;
    return os.str();
}

std::string padRight(std::string s, std::size_t width) {
    if (s.size() < width) s.append(width - s.size(), ' ');
    return s;
}

std::string padLeft(const std::string& s, std::size_t width) {
    return (s.size() < width) ? std::string(width - s.size(), ' ') + s : s;
}

std::string dayLabel(std::size_t index) { return "+" + std::to_string(index + 1); }

double percentChange(double value, double reference) {
    return (reference != 0.0) ? (value - reference) / reference * 100.0 : 0.0;
}

const char* direction(double forecast, double lastClose) {
    if (forecast > lastClose) return "up";
    if (forecast < lastClose) return "down";
    return "flat";
}

// ---- parsing -----------------------------------------------------------------

bool parseSize(const std::string& text, std::size_t& value) {
    const std::string s = trim(text);
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;  // also rejects "-1"
    }
    errno = 0;
    const unsigned long long v = std::strtoull(s.c_str(), nullptr, 10);
    if (errno == ERANGE || v > std::numeric_limits<std::size_t>::max()) return false;
    value = static_cast<std::size_t>(v);
    return true;
}

bool parseDouble(const std::string& token, double& value) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(v)) {
        return false;
    }
    value = v;
    return true;
}

// ---- data summary ------------------------------------------------------------

struct DataSummary {
    std::size_t count = 0;
    std::string firstDate;
    std::string lastDate;
    double firstClose = 0.0;
    double lastClose = 0.0;
    double minClose = 0.0;
    double maxClose = 0.0;
    std::string minDate;
    std::string maxDate;
    double meanClose = 0.0;
    double stdClose = 0.0;
    double totalReturnPct = 0.0;
    bool hasReturns = false;
    double meanReturnPct = 0.0;  // simple daily returns
    double stdReturnPct = 0.0;
    double meanVolume = 0.0;
};

double sampleStd(const std::vector<double>& v, double mean) {
    if (v.size() < 2) return 0.0;
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    return std::sqrt(sq / static_cast<double>(v.size() - 1));
}

DataSummary summarize(const std::vector<StockRecord>& records) {  // records must be non-empty
    DataSummary s;
    s.count = records.size();
    s.firstDate = records.front().date;
    s.lastDate = records.back().date;
    s.firstClose = records.front().close;
    s.lastClose = records.back().close;
    s.minClose = s.maxClose = records.front().close;
    s.minDate = s.maxDate = records.front().date;

    std::vector<double> closes;
    closes.reserve(records.size());
    double volumeSum = 0.0;
    for (const StockRecord& r : records) {
        closes.push_back(r.close);
        volumeSum += static_cast<double>(r.volume);
        if (r.close < s.minClose) { s.minClose = r.close; s.minDate = r.date; }
        if (r.close > s.maxClose) { s.maxClose = r.close; s.maxDate = r.date; }
    }
    double sum = 0.0;
    for (double c : closes) sum += c;
    s.meanClose = sum / static_cast<double>(closes.size());
    s.stdClose = sampleStd(closes, s.meanClose);
    s.totalReturnPct = percentChange(s.lastClose, s.firstClose);
    s.meanVolume = volumeSum / static_cast<double>(records.size());

    if (closes.size() >= 2) {
        std::vector<double> returns;
        returns.reserve(closes.size() - 1);
        double rsum = 0.0;
        for (std::size_t i = 1; i < closes.size(); ++i) {
            const double r = (closes[i] / closes[i - 1] - 1.0) * 100.0;
            returns.push_back(r);
            rsum += r;
        }
        s.hasReturns = true;
        s.meanReturnPct = rsum / static_cast<double>(returns.size());
        s.stdReturnPct = sampleStd(returns, s.meanReturnPct);
    }
    return s;
}

void printRecordRow(std::ostream& out, const StockRecord& r) {
    out << padRight(r.date, 12) << padLeft(fixedStr(r.open, 2), 10)
        << padLeft(fixedStr(r.high, 2), 10) << padLeft(fixedStr(r.low, 2), 10)
        << padLeft(fixedStr(r.close, 2), 10) << padLeft(std::to_string(r.volume), 14) << "\n";
}

}  // namespace

// ---------------------------------------------------------------------------------
// Construction and main loop
// ---------------------------------------------------------------------------------

Application::Application(std::istream& in, std::ostream& out) : in_(in), out_(out) {}

int Application::run() {
    out_ << "Stock Price Predictor: statistical ensemble forecasting from historical prices\n";

    while (true) {
        printMenu();
        std::string line;
        if (!readLine("Select an option (1-6): ", line)) {
            out_ << "Input ended; exiting.\n";
            return 0;
        }

        std::size_t choice = 0;
        if (!parseSize(line, choice) || choice < 1 || choice > 6) {
            out_ << "Invalid selection. Please enter a number from 1 to 6.\n";
            continue;
        }

        bool inputOpen = true;
        switch (choice) {
            case 1: inputOpen = loadData(); break;
            case 2: viewData(); break;
            case 3: inputOpen = configure(); break;
            case 4: generateForecast(); break;
            case 5: showIndividualPredictions(); break;
            case 6:
                out_ << "Goodbye.\n";
                return 0;
            default: break;
        }
        if (!inputOpen) {
            out_ << "Input ended; exiting.\n";
            return 0;
        }
    }
}

// ---------------------------------------------------------------------------------
// Input helpers
// ---------------------------------------------------------------------------------

bool Application::readLine(const std::string& prompt, std::string& line) {
    out_ << prompt << std::flush;
    if (!std::getline(in_, line)) {
        out_ << "\n";
        return false;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

Application::InputStatus Application::promptSize(const std::string& prompt, std::size_t low,
                                                 std::size_t high, std::size_t& value) {
    std::string line;
    if (!readLine(prompt, line)) return InputStatus::Eof;

    std::size_t parsed = 0;
    if (!parseSize(line, parsed) || parsed < low || parsed > high) {
        out_ << "Please enter a whole number between " << low << " and " << high << ".\n";
        return InputStatus::Invalid;
    }
    value = parsed;
    return InputStatus::Ok;
}

// ---------------------------------------------------------------------------------
// Menu actions
// ---------------------------------------------------------------------------------

bool Application::loadData() {
    std::string path;
    if (!readLine(std::string("CSV file path [") + kDefaultDataPath + "]: ", path)) return false;
    path = trim(path);
    if (path.empty()) path = kDefaultDataPath;

    try {
        const DataLoader loader;
        LoadResult result = loader.loadFromFile(path);

        records_ = std::move(result.records);
        dataPath_ = path;
        invalidateForecast();

        out_ << "Loaded " << records_.size() << " record(s) from '" << path << "' ("
             << records_.front().date << " to " << records_.back().date << ").\n";
        if (result.reordered) {
            out_ << "Note: rows were not in date order and have been sorted chronologically.\n";
        }
        if (!result.issues.empty()) {
            out_ << result.issues.size() << " row(s) were skipped or adjusted:\n";
            const std::size_t shown = std::min<std::size_t>(result.issues.size(), 10);
            for (std::size_t i = 0; i < shown; ++i) {
                out_ << "  line " << result.issues[i].line << ": " << result.issues[i].message << "\n";
            }
            if (result.issues.size() > shown) {
                out_ << "  ... and " << (result.issues.size() - shown) << " more.\n";
            }
        }
    } catch (const std::exception& e) {
        out_ << "Could not load data: " << e.what() << "\n";
        if (!records_.empty()) {
            out_ << "Keeping the previously loaded data (" << dataPath_ << ").\n";
        }
    }
    return true;
}

void Application::viewData() {
    if (!requireData()) return;
    printSummary();
    printRecordsPreview();
}

bool Application::configure() {
    while (true) {
        printConfigMenu();
        std::string line;
        if (!readLine("Select an option (1-6): ", line)) return false;

        std::size_t choice = 0;
        if (!parseSize(line, choice) || choice < 1 || choice > 6) {
            out_ << "Invalid selection. Please enter a number from 1 to 6.\n";
            continue;
        }
        if (choice == 6) return true;

        InputStatus status = InputStatus::Ok;
        std::size_t value = 0;

        switch (choice) {
            case 1:
                status = promptSize("New SMA window (1-1000): ", 1, 1000, value);
                if (status == InputStatus::Ok) {
                    settings_.smaWindow = value;
                    invalidateForecast();
                    out_ << "SMA window set to " << value << ".\n";
                }
                break;
            case 2:
                status = promptSize("New WMA window (1-1000): ", 1, 1000, value);
                if (status == InputStatus::Ok) {
                    settings_.wmaWindow = value;
                    invalidateForecast();
                    out_ << "WMA window set to " << value << ".\n";
                }
                break;
            case 3:
                status = promptSize("New lookback in days (0 = all history, otherwise 2-10000): ",
                                    0, 10000, value);
                if (status == InputStatus::Ok) {
                    if (value == 1) {
                        out_ << "Linear Regression lookback must be 0 (all history) or at least 2.\n";
                    } else {
                        settings_.regressionLookback = value;
                        invalidateForecast();
                        out_ << "Linear Regression lookback set to "
                             << (value == 0 ? std::string("all history") : std::to_string(value))
                             << ".\n";
                    }
                }
                break;
            case 4:
                status = promptSize("New forecast horizon in trading days (1-365): ", 1, 365, value);
                if (status == InputStatus::Ok) {
                    settings_.horizon = value;
                    invalidateForecast();
                    out_ << "Forecast horizon set to " << value << " trading day(s).\n";
                }
                break;
            case 5: {
                std::string text;
                if (!readLine("Enter 3 weights (SMA, WMA, LinearRegression) separated by spaces, "
                              "or 'equal': ", text)) {
                    return false;
                }
                if (toLower(trim(text)) == "equal") {
                    settings_.customWeights.reset();
                    invalidateForecast();
                    out_ << "Ensemble weights set to equal.\n";
                    break;
                }

                std::istringstream tokens(text);
                std::string token;
                std::vector<double> weights;
                bool numeric = true;
                while (tokens >> token) {
                    double w = 0.0;
                    if (!parseDouble(token, w)) {
                        numeric = false;
                        break;
                    }
                    weights.push_back(w);
                }
                if (!numeric || weights.empty()) {
                    out_ << "Weights must be numbers, for example: 1 2 3\n";
                    break;
                }
                try {
                    Ensemble probe = buildEnsemble();  // validates count, sign and sum
                    probe.setWeights(weights);
                    settings_.customWeights = weights;
                    invalidateForecast();
                    out_ << "Ensemble weights set (normalized):";
                    for (double w : probe.weights()) out_ << " " << fixedStr(w, 4);
                    out_ << "\n";
                } catch (const std::exception& e) {
                    out_ << "Invalid weights: " << e.what() << "\n";
                }
                break;
            }
            default: break;
        }
        if (status == InputStatus::Eof) return false;
    }
}

void Application::generateForecast() {
    if (!requireData()) return;
    if (computeForecast()) {
        printForecastReport(*lastForecast_);
    }
}

void Application::showIndividualPredictions() {
    if (!requireData()) return;
    if (!lastForecast_ && !computeForecast()) return;
    printIndividualPredictions(*lastForecast_);
    printWeights(*lastForecast_);
}

// ---------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------

bool Application::requireData() {
    if (records_.empty()) {
        out_ << "No data loaded. Choose option 1 to load a CSV file first.\n";
        return false;
    }
    return true;
}

Ensemble Application::buildEnsemble() const {
    Ensemble ensemble;
    ensemble.addModel(std::make_unique<SMA>(settings_.smaWindow));
    ensemble.addModel(std::make_unique<WMA>(settings_.wmaWindow));
    ensemble.addModel(std::make_unique<LinearRegression>(settings_.regressionLookback));
    if (settings_.customWeights) {
        ensemble.setWeights(*settings_.customWeights);
    }
    return ensemble;
}

bool Application::computeForecast() {
    std::size_t needed = 0;
    try {
        const Ensemble ensemble = buildEnsemble();
        needed = ensemble.minimumHistory();
        const std::vector<double> closes = DataLoader::extractClosingPrices(records_);
        lastForecast_ = ensemble.forecast(closes, settings_.horizon);
        return true;
    } catch (const std::exception& e) {
        lastForecast_.reset();
        out_ << "Could not generate a forecast: " << e.what() << "\n";
        if (needed > 0) {
            out_ << "The loaded data has " << records_.size() << " price(s); the most demanding "
                 << "model needs " << needed << ". Load more data or reduce the window sizes.\n";
        }
        return false;
    }
}

void Application::invalidateForecast() { lastForecast_.reset(); }

// ---------------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------------

void Application::printMenu() const {
    out_ << "\n=== Stock Price Predictor ===\n";
    if (records_.empty()) {
        out_ << "Data: none loaded\n";
    } else {
        out_ << "Data: " << dataPath_ << " (" << records_.size() << " records, "
             << records_.front().date << " to " << records_.back().date << ")\n";
    }
    out_ << " 1. Load historical stock data\n"
         << " 2. View data and summary statistics\n"
         << " 3. Configure model parameters\n"
         << " 4. Generate ensemble forecast\n"
         << " 5. Display individual model predictions\n"
         << " 6. Exit\n";
}

void Application::printConfigMenu() const {
    out_ << "\n--- Model configuration ---\n"
         << " 1. SMA window                  (current: " << settings_.smaWindow << ")\n"
         << " 2. WMA window                  (current: " << settings_.wmaWindow << ")\n"
         << " 3. Linear Regression lookback  (current: "
         << (settings_.regressionLookback == 0 ? std::string("all history")
                                               : std::to_string(settings_.regressionLookback))
         << ")\n"
         << " 4. Forecast horizon            (current: " << settings_.horizon
         << " trading day(s))\n"
         << " 5. Ensemble weights            (current: "
         << (settings_.customWeights ? "custom" : "equal") << ")\n"
         << " 6. Back to main menu\n";
}

void Application::printSummary() const {
    const DataSummary s = summarize(records_);
    out_ << "\n--- Summary statistics ---\n";
    out_ << "Source file:         " << dataPath_ << "\n";
    out_ << "Records:             " << s.count << "\n";
    out_ << "Date range:          " << s.firstDate << " to " << s.lastDate << "\n";
    out_ << "Close first / last:  " << fixedStr(s.firstClose, 2) << " / " << fixedStr(s.lastClose, 2) << "\n";
    out_ << "Close min / max:     " << fixedStr(s.minClose, 2) << " (" << s.minDate << ") / "
         << fixedStr(s.maxClose, 2) << " (" << s.maxDate << ")\n";
    out_ << "Close mean / std:    " << fixedStr(s.meanClose, 2) << " / " << fixedStr(s.stdClose, 2) << "\n";
    out_ << "Total return:        " << signedStr(s.totalReturnPct, 2) << "%\n";
    if (s.hasReturns) {
        out_ << "Daily return mean:   " << signedStr(s.meanReturnPct, 3) << "%   std dev: "
             << fixedStr(s.stdReturnPct, 3) << "%\n";
    }
    out_ << "Average volume:      " << fixedStr(s.meanVolume, 0) << "\n";
}

void Application::printRecordsPreview() const {
    out_ << "\n" << padRight("Date", 12) << padLeft("Open", 10) << padLeft("High", 10)
         << padLeft("Low", 10) << padLeft("Close", 10) << padLeft("Volume", 14) << "\n";

    constexpr std::size_t kEdge = 5;
    if (records_.size() <= 2 * kEdge) {
        for (const StockRecord& r : records_) printRecordRow(out_, r);
        return;
    }
    for (std::size_t i = 0; i < kEdge; ++i) printRecordRow(out_, records_[i]);
    out_ << "  ... (" << (records_.size() - 2 * kEdge) << " more rows) ...\n";
    for (std::size_t i = records_.size() - kEdge; i < records_.size(); ++i) {
        printRecordRow(out_, records_[i]);
    }
}

void Application::printForecastReport(const EnsembleResult& r) const {
    // 1) Combined forecast first.
    out_ << "\n==================== ENSEMBLE FORECAST ====================\n";
    out_ << "Based on " << records_.size() << " observation(s) through " << records_.back().date
         << " (last close " << fixedStr(r.lastClose, 2) << ")\n";
    out_ << "Horizon: " << r.horizon << " trading day(s) after the last date\n\n";

    out_ << padRight("Day", 6) << padLeft("Forecast", 12) << padLeft("Change", 12)
         << padLeft("Change %", 11) << "\n";
    for (std::size_t i = 0; i < r.combined.size(); ++i) {
        const double change = r.combined[i] - r.lastClose;
        out_ << padRight(dayLabel(i), 6) << padLeft(fixedStr(r.combined[i], 2), 12)
             << padLeft(signedStr(change, 2), 12)
             << padLeft(signedStr(percentChange(r.combined[i], r.lastClose), 2) + "%", 11) << "\n";
    }
    out_ << "Models contributing: " << r.modelsUsed << " of " << r.models.size() << "\n";

    // 2) Then, in order: individual predictions, weights, changes, diagnostics.
    printIndividualPredictions(r);
    printWeights(r);
    printChanges(r);
    printDiagnostics(r);
}

void Application::printIndividualPredictions(const EnsembleResult& r) const {
    out_ << "\n--- Individual model predictions ---\n";

    std::vector<std::size_t> widths;
    out_ << padRight("Day", 6);
    for (const ModelForecast& m : r.models) {
        const std::size_t w = std::max<std::size_t>(12, m.description.size() + 2);
        widths.push_back(w);
        out_ << padLeft(m.description, w);
    }
    out_ << "\n";

    for (std::size_t i = 0; i < r.horizon; ++i) {
        out_ << padRight(dayLabel(i), 6);
        for (std::size_t k = 0; k < r.models.size(); ++k) {
            const ModelForecast& m = r.models[k];
            out_ << padLeft(m.succeeded ? fixedStr(m.forecast[i], 2) : std::string("n/a"), widths[k]);
        }
        out_ << "\n";
    }
    for (const ModelForecast& m : r.models) {
        if (!m.succeeded) {
            out_ << "  " << m.description << ": excluded (" << m.error << ")\n";
        }
    }
}

void Application::printWeights(const EnsembleResult& r) const {
    out_ << "\n--- Model weights ---\n";
    out_ << "Scheme: " << (settings_.customWeights ? "custom weights (normalized)" : "equal weights")
         << "\n";
    out_ << padRight("Model", 40) << padLeft("Weight", 10) << "\n";
    for (const ModelForecast& m : r.models) {
        out_ << padRight(m.description, 40) << padLeft(fixedStr(m.weight, 4), 10)
             << (m.succeeded ? "" : "  (excluded)") << "\n";
    }
}

void Application::printChanges(const EnsembleResult& r) const {
    out_ << "\n--- Forecast changes vs last close (" << fixedStr(r.lastClose, 2) << ") ---\n";
    const std::size_t last = r.horizon - 1;
    out_ << padRight("Model", 40) << padLeft("Day +1", 10)
         << padLeft("Day +" + std::to_string(r.horizon), 12) << padLeft("Direction", 11) << "\n";

    for (const ModelForecast& m : r.models) {
        out_ << padRight(m.description, 40);
        if (!m.succeeded) {
            out_ << padLeft("excluded", 10) << "\n";
            continue;
        }
        out_ << padLeft(signedStr(percentChange(m.forecast.front(), r.lastClose), 2) + "%", 10)
             << padLeft(signedStr(percentChange(m.forecast[last], r.lastClose), 2) + "%", 12)
             << padLeft(direction(m.forecast[last], r.lastClose), 11) << "\n";
    }
    out_ << padRight("Ensemble (combined)", 40)
         << padLeft(signedStr(percentChange(r.combined.front(), r.lastClose), 2) + "%", 10)
         << padLeft(signedStr(percentChange(r.combined[last], r.lastClose), 2) + "%", 12)
         << padLeft(direction(r.combined[last], r.lastClose), 11) << "\n";
}

void Application::printDiagnostics(const EnsembleResult& r) const {
    out_ << "\n--- Diagnostics (model disagreement) ---\n";
    out_ << padRight("Day", 6) << padLeft("Spread", 12) << padLeft("Std dev", 12)
         << padLeft("Spread % of forecast", 24) << "\n";
    for (std::size_t i = 0; i < r.horizon; ++i) {
        const double rel = (r.combined[i] != 0.0) ? r.spread[i] / std::fabs(r.combined[i]) * 100.0 : 0.0;
        out_ << padRight(dayLabel(i), 6) << padLeft(fixedStr(r.spread[i], 2), 12)
             << padLeft(fixedStr(r.stdDev[i], 2), 12) << padLeft(fixedStr(rel, 2) + "%", 24) << "\n";
    }

    // Direction agreement on the final day, among models that contribute.
    const std::size_t last = r.horizon - 1;
    std::size_t up = 0;
    std::size_t down = 0;
    std::size_t flat = 0;
    for (const ModelForecast& m : r.models) {
        if (!m.succeeded || m.weight <= 0.0) continue;
        const std::string d = direction(m.forecast[last], r.lastClose);
        if (d == "up") ++up;
        else if (d == "down") ++down;
        else ++flat;
    }
    out_ << "\nDirection on day +" << r.horizon << " vs last close: " << up << " up, " << down
         << " down, " << flat << " flat (of " << r.modelsUsed << " contributing model(s)).\n";
    out_ << (std::max({up, down, flat}) == r.modelsUsed ? "The contributing models agree on direction.\n"
                                                        : "The contributing models DISAGREE on direction.\n");

    if (!r.warnings.empty()) {
        out_ << "\nWarnings:\n";
        for (const std::string& w : r.warnings) out_ << "  - " << w << "\n";
    }

    out_ << "\nNotes:\n";
    out_ << "  - Day labels count trading days after the last observation; no calendar dates are projected.\n";
    if (r.horizon > 1) {
        out_ << "  - SMA and WMA extend multi-day forecasts recursively (a smoothed level, not a learned\n"
                "    trend); Linear Regression extrapolates a single fitted straight line.\n";
    }
    out_ << "  - These are statistical extrapolations of past prices, not investment advice.\n";
}

}  // namespace spp