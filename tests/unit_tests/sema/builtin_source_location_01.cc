// RUN: %cxx -toolchain wasm32 -std=c++26 -verify -fsyntax-only -fvalidate-ast %s
// expected-no-diagnostics

#include <source_location>

auto here() -> std::source_location {
  return std::source_location::current();
}

#line 100 "source-location.cc"
constexpr auto location = std::source_location::current();
static_assert(location.line() == 100);
static_assert(location.column() > 0);
static_assert(__builtin_strcmp(location.file_name(), "source-location.cc") == 0);

constexpr auto current(std::source_location value = std::source_location::current()) {
  return value;
}
#line 200 "source-location-caller.cc"
static_assert(current().line() == 200);
static_assert(current().line() == 201);

struct AggregateLocation {
  std::source_location value = std::source_location::current();
};
#line 300 "source-location-aggregate.cc"
constexpr AggregateLocation aggregate{};
static_assert(aggregate.value.line() == 300);

struct ConstructedLocation {
  std::source_location value = std::source_location::current();
#line 400 "source-location-constructor.cc"
  constexpr ConstructedLocation() {}
};
static_assert(ConstructedLocation{}.value.line() == 400);
