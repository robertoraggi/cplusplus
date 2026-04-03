// RUN: %cxx -verify -fcheck %s

struct WithUnary {
  WithUnary operator-() noexcept;
};

WithUnary wu;
static_assert(noexcept(-wu));

struct WithUnaryThrow {
  WithUnaryThrow operator-();
};

WithUnaryThrow wut;
static_assert(!noexcept(-wut));

// Implicit cast - should propagate
void no_throw() noexcept;
static_assert(noexcept(no_throw()));

// Nested parentheses
static_assert(noexcept((no_throw())));

// Assignment operator - noexcept version
struct WithAssign {
  WithAssign& operator=(const WithAssign&) noexcept;
};
WithAssign a, b;
static_assert(noexcept(a = b));

// Assignment operator - throwing version
struct WithAssignThrow {
  WithAssignThrow& operator=(const WithAssignThrow&);
};
WithAssignThrow c, d;
static_assert(!noexcept(c = d));
