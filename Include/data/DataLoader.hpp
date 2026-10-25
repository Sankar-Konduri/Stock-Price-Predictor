#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "data/StockRecord.hpp"

namespace spp {

/// Thrown for unreadable files, malformed CSV structure, or invalid data.
class DataError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// A row that was skipped or a duplicate that was resolved while loading.
struct LoadIssue {
    std::size_t line = 0;  // 1-based line number in the input
    std::string message;
};

/// Outcome of a load: clean records plus a report of everything that was dropped.
struct LoadResult {
    std::vector<StockRecord> records;  // valid, strictly increasing by date
    std::vector<LoadIssue> issues;     // skipped rows / resolved duplicates
    std::size_t dataRowsRead = 0;      // non-blank rows after the header
    bool reordered = false;            // true if input was not in date order
};

/// Positions of the required columns within the header row.
struct ColumnMap {
    std::size_t date = 0;
    std::size_t open = 0;
    std::size_t high = 0;
    std::size_t low = 0;
    std::size_t close = 0;
    std::size_t volume = 0;
    std::size_t columnCount = 0;  // total columns in the header
};

enum class DuplicatePolicy {
    KeepFirst,  // keep the row that appears first in the file
    KeepLast    // keep the row that appears last in the file
};

struct LoaderOptions {
    bool strict = false;  // true: any bad row or duplicate throws DataError
    DuplicatePolicy duplicates = DuplicatePolicy::KeepFirst;
};

/**
 * Loads and validates historical price data from CSV.
 *
 * CSV format:
 *   - First non-blank line is a header containing (case-insensitive, any
 *     order): Date, Open, High, Low, Close, Volume. "Close/Last" (the name
 *     used by nasdaq.com downloads) is accepted for Close. Extra columns are
 *     ignored.
 *   - Fields are comma-separated; double quotes may wrap a field.
 *   - Date is YYYY-MM-DD or MM/DD/YYYY (US order) and must be a real calendar
 *     date. Records always store the ISO form YYYY-MM-DD.
 *   - Prices may start with '$'. Prices and volume may use thousands
 *     separators, but only in correct grouping (1,234.50 or 12,345,678).
 *   - Prices are finite and > 0; low <= min(open, close);
 *     high >= max(open, close). Volume is a whole number >= 0.
 *   - Blank lines are ignored. A UTF-8 BOM and CRLF line endings are handled.
 */
class DataLoader {
public:
    explicit DataLoader(LoaderOptions options = LoaderOptions{});

    const LoaderOptions& options() const { return options_; }

    /// `path` may be wrapped in matching single or double quotes and padded
    /// with spaces (as pasted from Windows Explorer's "Copy as path").
    /// @throws DataError if the file cannot be opened or holds no valid data.
    LoadResult loadFromFile(const std::string& path) const;

    /// Same as loadFromFile but reads from any stream (used by tests).
    LoadResult loadFromStream(std::istream& in) const;

    /// Maps header names to column positions.
    /// @throws DataError if a required column is missing or duplicated.
    static ColumnMap parseHeader(const std::vector<std::string>& headerFields);

    /// Parses and validates one data row.
    /// @throws DataError describing the first problem found.
    static StockRecord parseRecord(const std::vector<std::string>& fields,
                                   const ColumnMap& columns);

    /// Checks one record's values (date, prices, OHLC consistency, volume).
    /// @throws DataError describing the first problem found.
    static void validateRecord(const StockRecord& record);

    /// Checks that a dataset is non-empty, every record is valid, and dates
    /// are strictly increasing (so unique and chronological).
    /// @throws DataError describing the first problem found.
    static void validateDataset(const std::vector<StockRecord>& records);

    /// Closing prices in record order, ready for PredictionModel::predict.
    static std::vector<double> extractClosingPrices(
        const std::vector<StockRecord>& records);

    /// True for a real calendar date written as YYYY-MM-DD (leap years handled).
    static bool isValidIsoDate(const std::string& date);

    /// Converts a YYYY-MM-DD or MM/DD/YYYY date to YYYY-MM-DD.
    /// Returns an empty optional if the text is not a real calendar date in
    /// one of those two forms.
    static std::optional<std::string> normalizeDate(const std::string& text);

    /// Splits one CSV line into fields (supports "quoted, fields" and "" escapes).
    /// @throws DataError on an unterminated quote.
    static std::vector<std::string> splitCsvLine(const std::string& line);

private:
    LoaderOptions options_;
};

}  // namespace spp