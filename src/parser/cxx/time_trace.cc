#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/time_trace.h>

#include <format>
#include <utility>

namespace cxx {
namespace {
void writeString(std::format_context::iterator& out, const std::string& text) {
  constexpr auto digits = "0123456789abcdef";
  out = std::format_to(out, "{}", '\"');
  for (unsigned char ch : text) {
    if (ch == '\"' || ch == '\\') {
      out = std::format_to(out, "{}{}", '\\', static_cast<char>(ch));
    } else if (ch < 32) {
      out = std::format_to(out, "{}{}{}", "\\u00", digits[ch >> 4],
                           digits[ch & 15]);
    } else {
      out = std::format_to(out, "{}", static_cast<char>(ch));
    }
  }
  out = std::format_to(out, "{}", '\"');
}
}  // namespace

TimeTrace::Scope::Scope(TimeTrace* trace, std::string name, std::string detail)
    : trace_(trace), name_(std::move(name)), detail_(std::move(detail)) {
  if (!trace_) return;
  start_ = Clock::now();
  counts_ = trace_->counts_;
}

TimeTrace::Scope::Scope(TimeTrace* trace, std::string name,
                        const Symbol* symbol)
    : Scope(trace, std::move(name)) {
  if (trace && symbol) detail_ = to_string(symbol->name());
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

auto TimeTrace::formatTo(std::format_context::iterator out) const
    -> std::format_context::iterator {
  constexpr std::array names{"instantiation_requests",
                             "instantiations",
                             "function_bodies",
                             "constraint_checks",
                             "concept_checks",
                             "cache_hits",
                             "type_dependence_visits",
                             "type_dependence_cache_hits",
                             "specialization_comparisons",
                             "partial_specialization_visits",
                             "specialization_index_visits"};
  out = std::format_to(out, "{}", "{\"traceEvents\":[");
  bool first = true;
  for (const auto& event : events_) {
    if (!first) out = std::format_to(out, "{}", ',');
    first = false;
    out = std::format_to(out, "{}", "{\"name\":");
    writeString(out, event.name);
    out = std::format_to(
        out, "{}{}{}{}{}",
        ",\"cat\":\"cxx\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":",
        event.start, ",\"dur\":", event.duration, ",\"args\":{");
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (i) out = std::format_to(out, "{}", ',');
      writeString(out, names[i]);
      out = std::format_to(out, "{}{}", ':', event.counts[i]);
    }
    if (!event.detail.empty()) {
      out = std::format_to(out, "{}", ",\"detail\":");
      writeString(out, event.detail);
    }
    out = std::format_to(out, "{}", "}}");
  }
  out = std::format_to(out, "{}", "],\"displayTimeUnit\":\"ms\"}\n");
  return out;
}
}  // namespace cxx
