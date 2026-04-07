// RUN: %cxx -verify %s

void f1() noexcept(true);
static_assert(noexcept(f1()));

void f2() noexcept(false);
static_assert(!noexcept(f2()));

constexpr bool kTrue = true;
constexpr bool kFalse = false;

void f3() noexcept(kTrue);
static_assert(noexcept(f3()));

void f4() noexcept(kFalse);
static_assert(!noexcept(f4()));

void may_throw();
void no_throw() noexcept;

void f5() noexcept(noexcept(no_throw()));
static_assert(noexcept(f5()));

void f6() noexcept(noexcept(may_throw()));
static_assert(!noexcept(f6()));
