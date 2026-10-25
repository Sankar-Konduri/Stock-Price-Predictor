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

// nasdaq.com download layout: Close/Last column, US dates, $ prices, newest first.
const char* const kNasdaqCsv =
    "Date,Close/Last,Volume,Open,High,Low\n"
    "02/28/2024,$111.60,1480000,$112.00,$112.80,$111.10\n"
    "02/27/2024,$112.00,1600000,$111.20,$112.30,$110.80\n";

bool loadsFine(const std::string& path) {
    try {
        return !DataLoader().loadFromFile(path).records.empty();
    } catch (const DataError&) {
        return false;
    }
}

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
    check(!DataLoader::isValidIsoDate("01/02/2024"), "slash date is not ISO");

    // ---- Date normalization (ISO and US MM/DD/YYYY) ----
    check(DataLoader::normalizeDate("2024-02-29") == std::string("2024-02-29"),
          "ISO date passes through");
    check(DataLoader::normalizeDate("02/29/2024") == std::string("2024-02-29"),
          "MM/DD/YYYY converted to ISO");
    check(DataLoader::normalizeDate("10/23/2026") == std::string("2026-10-23"),
          "US date converted to ISO (month first)");
    check(DataLoader::normalizeDate(" 01/02/2024 ") == std::string("2024-01-02"),
          "surrounding spaces ignored");
    check(!DataLoader::normalizeDate("13/01/2024"), "month 13 rejected (day/month order unsupported)");
    check(!DataLoader::normalizeDate("02/30/2024"), "impossible slash date rejected");
    check(!DataLoader::normalizeDate("2023-02-29"), "impossible ISO date rejected");
    check(!DataLoader::normalizeDate("2024/01/02"), "YYYY/MM/DD rejected");
    check(!DataLoader::normalizeDate("1/2/2024"), "unpadded slash date rejected");
    check(!DataLoader::normalizeDate(""), "empty date rejected");

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

    // ---- nasdaq.com layout ----
    {
        ColumnMap cols = DataLoader::parseHeader({"Date", "Close/Last", "Volume", "Open", "High", "Low"});
        check(cols.date == 0 && cols.close == 1 && cols.volume == 2 && cols.open == 3 &&
                  cols.high == 4 && cols.low == 5 && cols.columnCount == 6,
              "Close/Last header maps to the Close column");

        auto nasdaq = load(kNasdaqCsv);
        check(nasdaq.records.size() == 2 && nasdaq.issues.empty(), "nasdaq.com layout loads cleanly");
        check(nasdaq.reordered, "newest-first input is re-sorted");
        check(nasdaq.records[0].date == "2024-02-27" && nasdaq.records[1].date == "2024-02-28",
              "US dates stored as ISO, oldest first");
        check(approx(nasdaq.records[0].close, 112.00) && approx(nasdaq.records[0].open, 111.20) &&
                  approx(nasdaq.records[0].high, 112.30) && approx(nasdaq.records[0].low, 110.80) &&
                  nasdaq.records[0].volume == 1600000,
              "$ prices and column order (Close/Last, Volume, Open, High, Low) read correctly");

        auto grouped = load(
            "Date,Close/Last,Volume,Open,High,Low\n"
            "10/23/2026,\"$1,234.50\",\"12,345,678\",\"$1,230.00\",\"$1,240.75\",\"$1,229.10\"\n");
        check(grouped.records.size() == 1 && approx(grouped.records[0].close, 1234.50) &&
                  grouped.records[0].volume == 12345678 && approx(grouped.records[0].high, 1240.75),
              "thousands separators accepted in prices and volume");

        expectThrows<DataError>(
            [] { load("Date,Close,Close/Last,Volume,Open,High,Low\n"); },
            "both Close and Close/Last in one header rejected");
    }

    // ---- Numbers: separators must be correctly grouped, never guessed ----
    {
        ColumnMap cols = DataLoader::parseHeader({"Date", "Open", "High", "Low", "Close", "Volume"});
        auto row = [&](const std::string& close, const std::string& volume) {
            return DataLoader::parseRecord({"2024-01-02", "10", "11", "9", close, volume}, cols);
        };
        check(approx(row("$10.5", "5").close, 10.5), "leading $ accepted on prices");
        check(row("10", "1,234").volume == 1234, "grouped volume accepted");
        expectThrows<DataError>([&] { row("10,5", "5"); }, "European decimal '10,5' rejected, not read as 105");
        expectThrows<DataError>([&] { row("1,23.45", "5"); }, "bad grouping '1,23.45' rejected");
        expectThrows<DataError>([&] { row("1,2345", "5"); }, "bad grouping '1,2345' rejected");
        expectThrows<DataError>([&] { row("10,", "5"); }, "trailing comma in number rejected");
        expectThrows<DataError>([&] { row("$", "5"); }, "lone $ rejected");
        expectThrows<DataError>([&] { row("10", "$1000"); }, "$ not allowed in volume");
        expectThrows<DataError>([&] { row("10", "1,234.5"); }, "fractional volume rejected");
    }

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
    StockRecord usDate = DataLoader::parseRecord({"01/02/2024", "10", "11", "9", "10.5", "1000"}, cols);
    check(usDate.date == "2024-01-02", "parseRecord converts a US-format date");
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
    check(loadsFine("data/historical_prices.csv"), "plain path loads");
    check(loadsFine("\"data/historical_prices.csv\""), "double-quoted path loads (Copy as path)");
    check(loadsFine("'data/historical_prices.csv'"), "single-quoted path loads");
    check(loadsFine("  \"data/historical_prices.csv\"  "), "quoted path with surrounding spaces loads");
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