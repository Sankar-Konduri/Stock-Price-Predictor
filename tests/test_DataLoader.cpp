#include <sstream>
#include <string>
#include <vector>

#include "data/DataLoader.hpp"
#include "test_helpers.hpp"

using namespace spp;
using namespace spp_test;

namespace {

LoadResult load(const std::string& text, LoaderOptions options = LoaderOptions{}) {
    std::istringstream in(text);
    return DataLoader(options).loadFromStream(in);
}

const char* const kMessyCsv =
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-02,10,11,9,10.5,1000\n"      // line 2: valid
    "2024-01-03,10,11,9,abc,1000\n"       // line 3: invalid number
    "2024-01-04,10,11,9,10.5\n"           // line 4: missing field
    "2024-01-05,10,11,9,nan,1000\n"       // line 5: non-finite
    "2024-01-08,10,11,9,10.5,-5\n"        // line 6: negative volume
    "2024-01-09,10,8,9,9.5,1000\n"        // line 7: high < low
    "2024-01-10,10,11,9,10.5,1200\n";     // line 8: valid

const char* const kDuplicateCsv =
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-02,10,11,9,10.5,1000\n"
    "2024-01-02,10,12,9,11.5,2000\n"
    "2024-01-03,11,12,10,11.0,1500\n";

}  // namespace

int main() {
    // ---- Date validation ----
    check(DataLoader::isValidIsoDate("2024-02-29"), "leap day 2024-02-29 valid");
    check(!DataLoader::isValidIsoDate("2023-02-29"), "2023-02-29 invalid");
    check(!DataLoader::isValidIsoDate("2100-02-29"), "2100 is not a leap year");
    check(DataLoader::isValidIsoDate("2000-02-29"), "2000 is a leap year");
    check(!DataLoader::isValidIsoDate("2024-04-31"), "April 31 invalid");
    check(!DataLoader::isValidIsoDate("2024-13-01"), "month 13 invalid");
    check(!DataLoader::isValidIsoDate("2024-1-01"), "unpadded month invalid");
    check(!DataLoader::isValidIsoDate("01/02/2024"), "non-ISO format invalid");

    // ---- CSV splitting ----
    auto plain = DataLoader::splitCsvLine("a,b,c");
    check(plain.size() == 3 && plain[2] == "c", "split plain line");
    auto quoted = DataLoader::splitCsvLine("\"x,y\",2");
    check(quoted.size() == 2 && quoted[0] == "x,y", "quoted comma kept in one field");
    auto empty = DataLoader::splitCsvLine("a,,c");
    check(empty.size() == 3 && empty[1].empty(), "empty middle field preserved");
    expectThrows<DataError>([] { DataLoader::splitCsvLine("\"oops,1"); },
                            "unterminated quote rejected");

    // ---- Clean load ----
    auto clean = load(
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,10,11,9,10.5,1000\n"
        "2024-01-03,10.5,12,10,11.5,2000\n"
        "\n"
        "2024-01-04,11.5,12,11,11.0,1500\n");
    check(clean.records.size() == 3 && clean.issues.empty(), "clean load: 3 records, no issues");
    check(!clean.reordered, "already-ordered input not flagged");
    auto closes = DataLoader::extractClosingPrices(clean.records);
    check(closes.size() == 3 && approx(closes[0], 10.5) && approx(closes[2], 11.0),
          "closing prices extracted in order");

    // ---- CRLF, mixed-case header, extra column, out-of-order rows ----
    auto windows = load(
        "date, OPEN ,High,Low,Close,Adj Close,Volume\r\n"
        "2024-01-03,10,11,9,10.5,10.4,1000\r\n"
        "2024-01-02,9,10,8,9.5,9.4,900\r\n");
    check(windows.records.size() == 2, "CRLF + extra column loads");
    check(windows.reordered, "out-of-order input flagged as reordered");
    check(windows.records[0].date == "2024-01-02" && approx(windows.records[0].close, 9.5) &&
              windows.records[0].volume == 900,
          "records sorted chronologically with correct columns");

    // ---- Lenient mode skips bad rows and reports them ----
    auto messy = load(kMessyCsv);
    check(messy.dataRowsRead == 7, "messy: 7 data rows read");
    check(messy.records.size() == 2, "messy: only 2 valid rows kept");
    check(messy.issues.size() == 5, "messy: 5 issues reported");
    check(messy.issues.front().line == 3 && messy.issues.back().line == 7,
          "messy: issues carry correct line numbers, sorted");

    // ---- Strict mode throws on the first problem ----
    LoaderOptions strict;
    strict.strict = true;
    expectThrows<DataError>([&] { load(kMessyCsv, strict); }, "strict mode rejects bad rows");

    // ---- Duplicate dates ----
    auto keepFirst = load(kDuplicateCsv);
    check(keepFirst.records.size() == 2 && approx(keepFirst.records[0].close, 10.5) &&
              keepFirst.issues.size() == 1,
          "duplicates: KeepFirst keeps first row and reports");

    LoaderOptions keepLastOpts;
    keepLastOpts.duplicates = DuplicatePolicy::KeepLast;
    auto keepLast = load(kDuplicateCsv, keepLastOpts);
    check(keepLast.records.size() == 2 && approx(keepLast.records[0].close, 11.5) &&
              keepLast.issues.size() == 1,
          "duplicates: KeepLast keeps last row and reports");
    expectThrows<DataError>([&] { load(kDuplicateCsv, strict); },
                            "strict mode rejects duplicate dates");

    // ---- Structural failures ----
    expectThrows<DataError>([] { load(""); }, "empty input rejected");
    expectThrows<DataError>([] { load("Date,Open,High,Low,Close,Volume\n"); },
                            "header-only input rejected");
    expectThrows<DataError>([] { load("Date,Open,High,Low,Close\n2024-01-02,1,2,1,1\n"); },
                            "missing Volume column rejected");
    expectThrows<DataError>(
        [] { load("Date,Open,High,Low,Close,Close,Volume\n"); },
        "duplicated column rejected");
    expectThrows<DataError>(
        [] { load("Date,Open,High,Low,Close,Volume\n2024-01-02,1,2,1,x,10\n"); },
        "all rows invalid rejected");

    // ---- Record-level validation ----
    ColumnMap cols = DataLoader::parseHeader({"Date", "Open", "High", "Low", "Close", "Volume"});
    StockRecord ok = DataLoader::parseRecord({"2024-01-02", "10", "11", "9", "10.5", "1000"}, cols);
    check(ok.date == "2024-01-02" && ok.volume == 1000, "parseRecord parses a valid row");
    expectThrows<DataError>(
        [&] { DataLoader::parseRecord({"2024-01-02", "10", "11", "9", "10.5", "12.5"}, cols); },
        "fractional volume rejected");
    expectThrows<DataError>(
        [&] { DataLoader::parseRecord({"2024-01-02", "0", "11", "9", "10.5", "5"}, cols); },
        "zero price rejected");
    expectThrows<DataError>(
        [&] { DataLoader::parseRecord({"2024-01-02", "10", "11", "9", "", "5"}, cols); },
        "empty field rejected (not replaced with zero)");
    expectThrows<DataError>(
        [&] { DataLoader::parseRecord({"2024-02-30", "10", "11", "9", "10", "5"}, cols); },
        "impossible date rejected");

    // ---- Dataset validation ----
    std::vector<StockRecord> unordered = {
        {"2024-01-03", 10, 11, 9, 10, 1}, {"2024-01-02", 10, 11, 9, 10, 1}};
    expectThrows<DataError>([&] { DataLoader::validateDataset(unordered); },
                            "unordered dataset rejected");
    expectThrows<DataError>([] { DataLoader::validateDataset({}); }, "empty dataset rejected");

    // ---- Files (run from the project root so the relative path resolves) ----
    expectThrows<DataError>([] { DataLoader().loadFromFile("no_such_file.csv"); },
                            "missing file rejected");
    try {
        auto sample = DataLoader().loadFromFile("data/historical_prices.csv");
        check(!sample.records.empty() && sample.issues.empty(),
              "sample CSV loads cleanly (" + std::to_string(sample.records.size()) + " records)");
        DataLoader::validateDataset(sample.records);
        check(true, "sample CSV passes dataset validation");
    } catch (const DataError& e) {
        check(false, std::string("sample CSV failed to load: ") + e.what());
    }

    return finish();
}