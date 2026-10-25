#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

namespace cxx {
class Symbol;

class TimeTrace {
 public:
  enum Counter {
    kInstantiationRequests,
    kInstantiations,
    kFunctionBodies,
    kConstraints,
    kConcepts,
    kCacheHits,
    kTypeDependenceVisits,
    kTypeDependenceCacheHits,
    kSpecializationComparisons,
    kPartialSpecializationVisits,
    kSpecializationIndexVisits,
    kCount
  };
  using Counts = std::array<std::uint64_t, kCount>;
  using Clock = std::chrono::steady_clock;

  class Scope {
   public:
    Scope(TimeTrace* trace, std::string name, std::string detail = {});
    Scope(TimeTrace* trace, std::string name, const Symbol* symbol);
    ~Scope();
    Scope(const Scope&) = delete;
    auto operator=(const Scope&) -> Scope& = delete;

    void setDetail(std::string detail) { detail_ = std::move(detail); }

   private:
    TimeTrace* trace_;
    std::string name_;
    std::string detail_;
    Clock::time_point start_;
    Counts counts_{};
  };

  void count(Counter counter, std::uint64_t amount = 1) {
    counts_[counter] += amount;
  }

 private:
  friend struct std::formatter<TimeTrace>;
  auto formatTo(std::format_context::iterator out) const
      -> std::format_context::iterator;
  struct Event {
    std::string name;
    std::string detail;
    std::int64_t start;
    std::int64_t duration;
    Counts counts;
  };
  Clock::time_point start_ = Clock::now();
  Counts counts_{};
  std::vector<Event> events_;
};
}  // namespace cxx

template <>
struct std::formatter<cxx::TimeTrace> {
  [[nodiscard]] constexpr auto parse(std::format_parse_context& context) {
    return context.begin();
  }

  auto format(const cxx::TimeTrace& trace, std::format_context& context) const {
    return trace.formatTo(context.out());
  }
};
