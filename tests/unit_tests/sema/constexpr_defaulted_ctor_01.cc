// clang-format off
// RUN: %cxx -verify %s
// expected-no-diagnostics

struct allocator_arg_t {
  explicit allocator_arg_t() = default;
};

constexpr allocator_arg_t allocator_arg = allocator_arg_t();
