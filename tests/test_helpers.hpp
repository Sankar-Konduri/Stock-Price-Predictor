#pragma once

//SHARED BY ALL THE FUTURE TESTS

#include <cmath>
#include <iostream>
#include <string>

namespace spp_test {

inline int& failureCount() {
    static int failures = 0;
    return failures;
}

inline void check(bool ok, const std::string& what) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!ok) ++failureCount();
}

inline bool approx(double a, double b, double tolerance = 1e-9) {
    return std::fabs(a - b) <= tolerance;
}

/// Passes only if `fn` throws exactly an exception of type `Ex` (or derived).
template <typename Ex, typename Fn>
void expectThrows(Fn&& fn, const std::string& what) {
    try {
        fn();
        check(false, what + " (no exception thrown)");
    } catch (const Ex&) {
        check(true, what);
    } catch (...) {
        check(false, what + " (wrong exception type)");
    }
}

/// Call at the end of main(); returns the process exit code.
inline int finish() {
    if (failureCount() == 0) {
        std::cout << "All checks passed.\n";
        return 0;
    }
    std::cout << failureCount() << " check(s) FAILED.\n";
    return 1;
}

}  // namespace spp_test