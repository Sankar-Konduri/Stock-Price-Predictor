#pragma once

#include <cstdint>
#include <string>

namespace spp {

/**
 * One historical trading day.
 *
 * date   : ISO-8601 "YYYY-MM-DD". Lexicographic order == chronological order.
 * open, high, low, close : prices in the quote currency.
 * volume : number of shares traded (non-negative).
 *
 * This is a plain data holder (struct). Validation of the values is the
 * responsibility of DataLoader, which is the only place records are created
 * from external input.
 */
struct StockRecord {
    std::string  date;
    double       open   = 0.0;
    double       high   = 0.0;
    double       low    = 0.0;
    double       close  = 0.0;
    std::int64_t volume = 0;
};

/// Orders records chronologically (valid because dates are ISO-formatted).
inline bool operator<(const StockRecord& a, const StockRecord& b) {
    return a.date < b.date;
}

}  // namespace spp