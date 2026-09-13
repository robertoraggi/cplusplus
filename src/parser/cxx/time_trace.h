#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace cxx {
class TimeTrace {
 public:
  enum Counter {
    kInstantiationRequests,
    kInstantiations,
    kFunctionBodies,
    kConstraints,
    kConcepts,
    kCacheHits,
    kCount
  };
  using Counts = std::array<std::uint64_t, kCount>;
  using Clock = std::chrono::steady_clock;

  class Scope {
   public:
    Scope(TimeTrace* trace, std::string name, std::string detail = {});
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

  void count(Counter counter) { ++counts_[counter]; }
  void write(std::ostream& out) const;

 private:
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
