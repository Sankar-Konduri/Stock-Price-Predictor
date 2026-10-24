#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "app/Application.hpp"
#include "test_helpers.hpp"

using namespace spp;
using namespace spp_test;

namespace {

// Runs the application with scripted input and returns everything it printed.
std::string runApp(const std::string& input, int* exitCode = nullptr) {
    std::istringstream in(input);
    std::ostringstream out;
    Application app(in, out);
    const int code = app.run();
    if (exitCode) *exitCode = code;
    return out.str();
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

std::size_t countOf(const std::string& text, const std::string& needle) {
    std::size_t count = 0;
    for (std::size_t pos = text.find(needle); pos != std::string::npos;
         pos = text.find(needle, pos + needle.size())) {
        ++count;
    }
    return count;
}

void writeFile(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    file << content;
}

const char* const kHeader = "Date,Open,High,Low,Close,Volume\n";

}  // namespace

int main() {
    int code = -1;

    // ---- Basic loop behaviour (main menu: 1-5 features, 6 = Exit) ----
    {
        auto out = runApp("6\n", &code);
        check(code == 0 && contains(out, "Stock Price Predictor") && contains(out, "6. Exit"),
              "menu is shown and option 6 exits");
        check(!contains(out, "backtest") && !contains(out, "Backtest") && !contains(out, "Export"),
              "menu lists only the five features and Exit");

        // 7, 8 and 9 no longer exist, so they are invalid like any other bad entry.
        out = runApp("abc\n0\n7\n\n-1\n6\n", &code);
        check(code == 0 && countOf(out, "Invalid selection") == 5,
              "invalid menu entries (including removed options) are rejected without crashing");

        out = runApp("8\n9\n6\n");
        check(countOf(out, "Invalid selection") == 2, "removed options 8 and 9 are invalid");

        out = runApp("", &code);
        check(code == 0 && contains(out, "Input ended"), "end of input exits cleanly");

        out = runApp("1\n", &code);
        check(code == 0 && contains(out, "Input ended"), "end of input inside a prompt exits cleanly");
    }

    // ---- Requires data ----
    {
        auto out = runApp("2\n4\n5\n6\n");
        check(countOf(out, "No data loaded") == 3, "data-dependent options refuse to run without data");
    }

    // ---- Loading ----
    {
        auto out = runApp("1\nno_such_file.csv\n6\n");
        check(contains(out, "Could not load data"), "missing file reported, no crash");

        out = runApp("1\n\n2\n6\n");  // empty path = default sample file
        check(contains(out, "record(s) from 'data/historical_prices.csv'") &&
                  contains(out, "Summary statistics") && contains(out, "Total return"),
              "default file loads and summary statistics are shown");

        out = runApp("1\n\n1\nno_such_file.csv\n4\n6\n");
        check(contains(out, "Keeping the previously loaded data") && contains(out, "ENSEMBLE FORECAST"),
              "a failed reload keeps the earlier data");
    }

    // ---- Forecast report content and order ----
    {
        auto out = runApp("1\n\n4\n6\n");
        const auto pCombined = out.find("ENSEMBLE FORECAST");
        const auto pIndividual = out.find("Individual model predictions");
        const auto pWeights = out.find("Model weights");
        const auto pChanges = out.find("Forecast changes");
        const auto pDiagnostics = out.find("Diagnostics");
        check(pCombined != std::string::npos && pIndividual != std::string::npos &&
                  pWeights != std::string::npos && pChanges != std::string::npos &&
                  pDiagnostics != std::string::npos,
              "forecast report has all five sections");
        check(pCombined < pIndividual && pIndividual < pWeights && pWeights < pChanges &&
                  pChanges < pDiagnostics,
              "combined forecast comes first, then individual, weights, changes, diagnostics");

        out = runApp("1\n\n5\n6\n");
        check(contains(out, "Individual model predictions") && contains(out, "Model weights") &&
                  !contains(out, "ENSEMBLE FORECAST"),
              "option 5 shows individual predictions and weights");
    }

    // ---- Configuration (inside the config menu, 6 = back; then 6 again exits) ----
    {
        auto out = runApp("3\n1\n10\n4\n3\n6\n6\n");
        check(contains(out, "SMA window set to 10") && contains(out, "horizon set to 3") &&
                  contains(out, "(current: 10)"),
              "valid settings are applied");

        out = runApp("3\n1\n0\n4\nabc\n3\n1\n6\n6\n");
        check(countOf(out, "Please enter a whole number") == 2 &&
                  contains(out, "lookback must be 0"),
              "invalid settings are rejected with explanations");

        out = runApp("3\n5\n1 2 3\n5\n1 2\n5\nfoo\n5\n1 -1 1\n5\nequal\n6\n6\n");
        check(contains(out, "Ensemble weights set (normalized): 0.1667 0.3333 0.5000") &&
                  countOf(out, "Invalid weights") == 2 && contains(out, "must be numbers") &&
                  contains(out, "set to equal"),
              "custom weights are validated and can be reset");

        out = runApp("3\n5\n1 2 3\n6\n1\n\n4\n6\n");
        check(contains(out, "custom weights") && contains(out, "0.1667") && contains(out, "0.5000"),
              "custom weights appear in the forecast report");

        out = runApp("3\n4\n3\n6\n1\n\n4\n6\n");
        check(contains(out, "Horizon: 3 trading day(s)"), "horizon setting reaches the forecast");
    }

    // ---- Too little data ----
    {
        const std::string tmp = "app_test_tmp.csv";

        writeFile(tmp, std::string(kHeader) + "2024-01-02,10,11,9,10.5,1000\n");
        auto out = runApp("1\n" + tmp + "\n4\n6\n", &code);
        check(code == 0 && contains(out, "Could not generate a forecast"),
              "one price: friendly error, no crash");

        writeFile(tmp, std::string(kHeader) + "2024-01-02,10,11,9,10.5,1000\n"
                                              "2024-01-03,10.5,12,10,11.5,1200\n");
        out = runApp("1\n" + tmp + "\n4\n6\n", &code);
        check(code == 0 && contains(out, "ENSEMBLE FORECAST") && contains(out, "was excluded") &&
                  contains(out, "Models contributing: 1 of 3"),
              "two prices: models needing more history are excluded, regression still forecasts");

        std::remove(tmp.c_str());
    }

    return finish();
}