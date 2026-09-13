#include <cxx/time_trace.h>

#include <ostream>
#include <utility>

namespace cxx {
namespace {
void writeString(std::ostream& out, const std::string& text) {
  constexpr auto digits = "0123456789abcdef";
  out << '\"';
  for (unsigned char ch : text) {
    if (ch == '\"' || ch == '\\') {
      out << '\\' << ch;
    } else if (ch < 32) {
      out << "\\u00" << digits[ch >> 4] << digits[ch & 15];
    } else {
      out << ch;
    }
  }
  out << '\"';
}
}  // namespace

TimeTrace::Scope::Scope(TimeTrace* trace, std::string name, std::string detail)
    : trace_(trace), name_(std::move(name)), detail_(std::move(detail)) {
  if (!trace_) return;
  start_ = Clock::now();
  counts_ = trace_->counts_;
}

TimeTrace::Scope::~Scope() {
  if (!trace_) return;
  const auto end = Clock::now();
  for (std::size_t i = 0; i < counts_.size(); ++i)
    counts_[i] = trace_->counts_[i] - counts_[i];
  trace_->events_.push_back(
      {std::move(name_), std::move(detail_),
       std::chrono::duration_cast<std::chrono::microseconds>(start_ -
                                                             trace_->start_)
           .count(),
       std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
           .count(),
       counts_});
}

void TimeTrace::write(std::ostream& out) const {
  constexpr std::array names{"instantiation_requests", "instantiations",
                             "function_bodies",        "constraint_checks",
                             "concept_checks",         "cache_hits"};
  out << "{\"traceEvents\":[";
  bool first = true;
  for (const auto& event : events_) {
    if (!first) out << ',';
    first = false;
    out << "{\"name\":";
    writeString(out, event.name);
    out << ",\"cat\":\"cxx\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":"
        << event.start << ",\"dur\":" << event.duration << ",\"args\":{";
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (i) out << ',';
      writeString(out, names[i]);
      out << ':' << event.counts[i];
    }
    if (!event.detail.empty()) {
      out << ",\"detail\":";
      writeString(out, event.detail);
    }
    out << "}}";
  }
  out << "],\"displayTimeUnit\":\"ms\"}\n";
}
}  // namespace cxx
