// clang-format off
// RUN: %cxx -verify -fcheck %s

struct BothDefaulted {
  BothDefaulted& operator=(const BothDefaulted&) = default;
  BothDefaulted& operator=(BothDefaulted&&) = default;
};

static_assert(__is_trivially_copyable(BothDefaulted), "BothDefaulted trivially copyable");

struct CopyDefaultedMoveDeleted {
  CopyDefaultedMoveDeleted& operator=(const CopyDefaultedMoveDeleted&) = default;
  CopyDefaultedMoveDeleted& operator=(CopyDefaultedMoveDeleted&&) = delete;
};

static_assert(__is_trivially_copyable(CopyDefaultedMoveDeleted), "CopyDefaultedMoveDeleted");

struct CopyUserProvided {
  CopyUserProvided& operator=(const CopyUserProvided&) { return *this; }
  CopyUserProvided& operator=(CopyUserProvided&&) = default;
};

// expected-error@+1 {{"CopyUserProvided not trivially copyable"}}
static_assert(__is_trivially_copyable(CopyUserProvided), "CopyUserProvided not trivially copyable");

struct MoveUserProvided {
  MoveUserProvided& operator=(const MoveUserProvided&) = default;
  MoveUserProvided& operator=(MoveUserProvided&&) { return *this; }
};

// expected-error@+1 {{"MoveUserProvided not trivially copyable"}}
static_assert(__is_trivially_copyable(MoveUserProvided), "MoveUserProvided not trivially copyable");

struct BothUserProvided {
  BothUserProvided& operator=(const BothUserProvided&) { return *this; }
  BothUserProvided& operator=(BothUserProvided&&) { return *this; }
};

// expected-error@+1 {{"BothUserProvided not trivially copyable"}}
static_assert(__is_trivially_copyable(BothUserProvided), "BothUserProvided not trivially copyable");

struct BothDeleted {
  BothDeleted& operator=(const BothDeleted&) = delete;
  BothDeleted& operator=(BothDeleted&&) = delete;
};

static_assert(__is_trivially_copyable(BothDeleted), "BothDeleted trivially copyable");

struct BothCtorsDefaulted {
  BothCtorsDefaulted(const BothCtorsDefaulted&) = default;
  BothCtorsDefaulted(BothCtorsDefaulted&&) = default;
};

static_assert(__is_trivially_copyable(BothCtorsDefaulted), "BothCtorsDefaulted");

struct CopyCtorUserProvided {
  CopyCtorUserProvided(const CopyCtorUserProvided&) {}
  CopyCtorUserProvided(CopyCtorUserProvided&&) = default;
};

// expected-error@+1 {{"CopyCtorUserProvided not trivially copyable"}}
static_assert(__is_trivially_copyable(CopyCtorUserProvided), "CopyCtorUserProvided not trivially copyable");

struct OpEqWithExtraOverload {
  OpEqWithExtraOverload& operator=(const OpEqWithExtraOverload&) = default;
  OpEqWithExtraOverload& operator=(int) { return *this; }
};

static_assert(__is_trivially_copyable(OpEqWithExtraOverload), "extra operator= overload");

static_assert(__is_trivially_assignable(BothDefaulted, const BothDefaulted&), "trivially copy assignable");
static_assert(__is_trivially_assignable(BothDefaulted, BothDefaulted&&), "trivially move assignable");
// expected-error@+1 {{"not trivially copy assignable"}}
static_assert(__is_trivially_assignable(BothUserProvided, const BothUserProvided&), "not trivially copy assignable");
// expected-error@+1 {{"not trivially move assignable"}}
static_assert(__is_trivially_assignable(BothUserProvided, BothUserProvided&&), "not trivially move assignable");

static_assert(__is_trivial(BothDefaulted), "BothDefaulted is trivial");
// expected-error@+1 {{"BothUserProvided not trivial"}}
static_assert(__is_trivial(BothUserProvided), "BothUserProvided not trivial");

struct OverloadedVirtual {
  virtual void f(int);
  void f(double);
};

static_assert(__is_polymorphic(OverloadedVirtual), "overloaded virtual is polymorphic");
// expected-error@+1 {{"OverloadedVirtual not trivially copyable"}}
static_assert(__is_trivially_copyable(OverloadedVirtual), "OverloadedVirtual not trivially copyable");

struct NoVirtualOverload {
  void f(int);
  void f(double);
};

static_assert(!__is_polymorphic(NoVirtualOverload), "no virtual overload");
static_assert(__is_trivially_copyable(NoVirtualOverload), "NoVirtualOverload trivially copyable");
