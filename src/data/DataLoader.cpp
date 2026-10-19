#include "data/DataLoader.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <utility>

namespace spp {

namespace {

std::string trim(const std::string& s) {
    std::size_t begin = 0;
    std::size_t end = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(begin, end - begin);
}

std::string toLower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

double parseDouble(const std::string& raw, const std::string& column) {
    const std::string s = trim(raw);
    if (s.empty()) {
        throw DataError(column + " is missing (empty field)");
    }
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') {
        throw DataError(column + ": '" + s + "' is not a valid number");
    }
    if (errno == ERANGE || !std::isfinite(value)) {
        throw DataError(column + ": '" + s + "' is not a finite number");
    }
    return value;
}

std::int64_t parseInt64(const std::string& raw, const std::string& column) {
    const std::string s = trim(raw);
    if (s.empty()) {
        throw DataError(column + " is missing (empty field)");
    }
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') {
        throw DataError(column + ": '" + s + "' is not a valid whole number");
    }
    if (errno == ERANGE) {
        throw DataError(column + ": '" + s + "' is out of range");
    }
    return static_cast<std::int64_t>(value);
}

void requirePositiveFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw DataError(std::string(name) + " price is not finite");
    }
    if (value <= 0.0) {
        throw DataError(std::string(name) + " price must be positive, got " +
                        std::to_string(value));
    }
}

// A parsed row remembers its source line for duplicate/ordering reports.
struct ParsedRow {
    StockRecord record;
    std::size_t line;
};

}  // namespace

DataLoader::DataLoader(LoaderOptions options) : options_(options) {}

bool DataLoader::isValidIsoDate(const std::string& date) {
    if (date.size() != 10 || date[4] != '-' || date[7] != '-') return false;
    for (std::size_t i = 0; i < date.size(); ++i) {
        if (i == 4 || i == 7) continue;
        if (!std::isdigit(static_cast<unsigned char>(date[i]))) return false;
    }
    const int year = std::stoi(date.substr(0, 4));
    const int month = std::stoi(date.substr(5, 2));
    const int day = std::stoi(date.substr(8, 2));
    if (year < 1 || month < 1 || month > 12 || day < 1) return false;

    static const int daysInMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const bool leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    const int maxDay = daysInMonth[month - 1] + ((month == 2 && leap) ? 1 : 0);
    return day <= maxDay;
}

std::vector<std::string> DataLoader::splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool inQuotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (inQuotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    current += '"';  // "" inside quotes is a literal quote
                    ++i;
                } else {
                    inQuotes = false;
                }
            } else {
                current += c;
            }
        } else if (c == '"') {
            inQuotes = true;
        } else if (c == ',') {
            fields.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (inQuotes) {
        throw DataError("unterminated quoted field");
    }
    fields.push_back(current);
    return fields;
}

ColumnMap DataLoader::parseHeader(const std::vector<std::string>& headerFields) {
    static const char* const names[6] = {"date", "open", "high", "low", "close", "volume"};
    std::size_t positions[6] = {0, 0, 0, 0, 0, 0};
    bool found[6] = {false, false, false, false, false, false};

    for (std::size_t i = 0; i < headerFields.size(); ++i) {
        const std::string name = toLower(trim(headerFields[i]));
        for (int k = 0; k < 6; ++k) {
            if (name == names[k]) {
                if (found[k]) {
                    throw DataError(std::string("header contains column '") + names[k] +
                                    "' more than once");
                }
                found[k] = true;
                positions[k] = i;
            }
        }
    }

    std::string missing;
    for (int k = 0; k < 6; ++k) {
        if (!found[k]) {
            if (!missing.empty()) missing += ", ";
            missing += names[k];
        }
    }
    if (!missing.empty()) {
        throw DataError("header is missing required column(s): " + missing);
    }

    ColumnMap map;
    map.date = positions[0];
    map.open = positions[1];
    map.high = positions[2];
    map.low = positions[3];
    map.close = positions[4];
    map.volume = positions[5];
    map.columnCount = headerFields.size();
    return map;
}

void DataLoader::validateRecord(const StockRecord& r) {
    if (!isValidIsoDate(r.date)) {
        throw DataError("date '" + r.date + "' is not a valid YYYY-MM-DD date");
    }
    requirePositiveFinite(r.open, "open");
    requirePositiveFinite(r.high, "high");
    requirePositiveFinite(r.low, "low");
    requirePositiveFinite(r.close, "close");

    if (r.high < r.low) {
        throw DataError("high (" + std::to_string(r.high) + ") is below low (" +
                        std::to_string(r.low) + ")");
    }
    if (r.high < std::max(r.open, r.close)) {
        throw DataError("high is below the open or close price");
    }
    if (r.low > std::min(r.open, r.close)) {
        throw DataError("low is above the open or close price");
    }
    if (r.volume < 0) {
        throw DataError("volume must not be negative, got " + std::to_string(r.volume));
    }
}

StockRecord DataLoader::parseRecord(const std::vector<std::string>& fields,
                                    const ColumnMap& columns) {
    if (fields.size() != columns.columnCount) {
        throw DataError("expected " + std::to_string(columns.columnCount) +
                        " fields but found " + std::to_string(fields.size()));
    }

    StockRecord record;
    record.date = trim(fields[columns.date]);
    if (record.date.empty()) {
        throw DataError("Date is missing (empty field)");
    }
    record.open = parseDouble(fields[columns.open], "Open");
    record.high = parseDouble(fields[columns.high], "High");
    record.low = parseDouble(fields[columns.low], "Low");
    record.close = parseDouble(fields[columns.close], "Close");
    record.volume = parseInt64(fields[columns.volume], "Volume");

    validateRecord(record);
    return record;
}

void DataLoader::validateDataset(const std::vector<StockRecord>& records) {
    if (records.empty()) {
        throw DataError("dataset is empty");
    }
    for (std::size_t i = 0; i < records.size(); ++i) {
        try {
            validateRecord(records[i]);
        } catch (const DataError& e) {
            throw DataError("record " + std::to_string(i) + " (" + records[i].date +
                            "): " + e.what());
        }
        if (i > 0 && records[i].date <= records[i - 1].date) {
            throw DataError("dates are not strictly increasing at record " +
                            std::to_string(i) + " (" + records[i - 1].date + " then " +
                            records[i].date + ")");
        }
    }
}

std::vector<double> DataLoader::extractClosingPrices(
    const std::vector<StockRecord>& records) {
    std::vector<double> closes;
    closes.reserve(records.size());
    for (const StockRecord& r : records) {
        closes.push_back(r.close);
    }
    return closes;
}

LoadResult DataLoader::loadFromFile(const std::string& path) const {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw DataError("cannot open file '" + path + "'");
    }
    return loadFromStream(file);
}

LoadResult DataLoader::loadFromStream(std::istream& in) const {
    LoadResult result;
    std::vector<ParsedRow> rows;
    ColumnMap columns;
    bool headerSeen = false;

    std::string line;
    std::size_t lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;

        // Strip a UTF-8 byte-order mark and a trailing '\r' (CRLF files).
        if (lineNo == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line.erase(0, 3);
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (trim(line).empty()) {
            continue;  // blank lines are ignored
        }

        if (!headerSeen) {
            try {
                columns = parseHeader(splitCsvLine(line));
            } catch (const DataError& e) {
                throw DataError("line " + std::to_string(lineNo) + ": " + e.what());
            }
            headerSeen = true;
            continue;
        }

        ++result.dataRowsRead;
        try {
            rows.push_back(ParsedRow{parseRecord(splitCsvLine(line), columns), lineNo});
        } catch (const DataError& e) {
            if (options_.strict) {
                throw DataError("line " + std::to_string(lineNo) + ": " + e.what());
            }
            result.issues.push_back(LoadIssue{lineNo, e.what()});
        }
    }

    if (in.bad()) {
        throw DataError("read error while loading data");
    }
    if (!headerSeen) {
        throw DataError("input is empty: no header row found");
    }
    if (rows.empty()) {
        throw DataError("no valid data rows found (" + std::to_string(result.dataRowsRead) +
                        " data row(s) read, all rejected)");
    }

    // Order chronologically. stable_sort keeps file order among equal dates,
    // which is what makes KeepFirst / KeepLast well defined.
    const auto byDate = [](const ParsedRow& a, const ParsedRow& b) {
        return a.record.date < b.record.date;
    };
    result.reordered = !std::is_sorted(rows.begin(), rows.end(), byDate);
    std::stable_sort(rows.begin(), rows.end(), byDate);

    // Resolve duplicate dates.
    std::vector<StockRecord> unique;
    std::vector<std::size_t> uniqueLine;
    for (const ParsedRow& row : rows) {
        if (!unique.empty() && unique.back().date == row.record.date) {
            const std::string& date = row.record.date;
            if (options_.strict) {
                throw DataError("line " + std::to_string(row.line) + ": duplicate date " +
                                date + " (first seen on line " +
                                std::to_string(uniqueLine.back()) + ")");
            }
            if (options_.duplicates == DuplicatePolicy::KeepFirst) {
                result.issues.push_back(LoadIssue{
                    row.line, "duplicate date " + date + ": kept line " +
                                  std::to_string(uniqueLine.back()) + ", ignored this row"});
            } else {
                result.issues.push_back(LoadIssue{
                    uniqueLine.back(), "duplicate date " + date + ": replaced by line " +
                                           std::to_string(row.line)});
                unique.back() = row.record;
                uniqueLine.back() = row.line;
            }
            continue;
        }
        unique.push_back(row.record);
        uniqueLine.push_back(row.line);
    }

    std::sort(result.issues.begin(), result.issues.end(),
              [](const LoadIssue& a, const LoadIssue& b) { return a.line < b.line; });

    validateDataset(unique);  // final safety net; should always pass here
    result.records = std::move(unique);
    return result;
}

}  // namespace spp