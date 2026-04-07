// RUN: %cxx -verify %s

// Literals and arithmetic - never throw
static_assert(noexcept(1 + 2));
static_assert(noexcept(1.0 * 2.0));
static_assert(noexcept(true));
static_assert(noexcept(42));

// throw-expression - always throws
static_assert(!noexcept(throw 1));

// Non-noexcept function
void may_throw();
static_assert(!noexcept(may_throw()));

// noexcept function
void no_throw() noexcept;
static_assert(noexcept(no_throw()));

// Function pointer
void (*fp)() noexcept;
static_assert(noexcept(fp()));

void (*fq)();
static_assert(!noexcept(fq()));

// new - potentially throwing by default
static_assert(!noexcept(new int));

// Nested noexcept is always a bool constant - never throws
static_assert(noexcept(noexcept(may_throw())));

// Class with noexcept / non-noexcept constructor
struct NoThrowCtor {
  NoThrowCtor() noexcept;
};
struct ThrowCtor {
  ThrowCtor();
};

static_assert(noexcept(NoThrowCtor()));
static_assert(!noexcept(ThrowCtor()));

// Conditional expression
static_assert(!noexcept(true ? may_throw() : no_throw()));
static_assert(noexcept(true ? no_throw() : no_throw()));
