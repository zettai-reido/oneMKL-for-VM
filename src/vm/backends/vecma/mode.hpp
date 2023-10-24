#pragma once

#include <cstdint>

namespace vecma::detail {

enum class Indexing : uint32_t {
    kNotDefined = 0,
    kNormal = 1,
    kCyclic = 2
};

enum class GlobalReport : uint32_t {
    kNotDefined = 0,
    kReport = 1,
    kQuiet = 2
};

struct Mode {
    Indexing indexing;
    GlobalReport global_report;

    constexpr Mode(Indexing _indexing = Indexing::kNotDefined, GlobalReport _global_report = GlobalReport::kNotDefined):
        indexing(_indexing),
        global_report(_global_report) { /* empty */ }

    Mode& operator +=(Mode const& rhs) {
        if (rhs.indexing != Indexing::kNotDefined) { indexing = rhs.indexing; }
        if (rhs.global_report != GlobalReport::kNotDefined) { global_report = rhs.global_report; }
        return *this;
    }
}; // struct Mode


} // namespace vecma::detail

