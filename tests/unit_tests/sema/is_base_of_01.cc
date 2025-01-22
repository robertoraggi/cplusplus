// RUN: %cxx -verify -fcheck %s

struct Base;
struct OtherBase;
struct Derived;
struct AlsoDerived;

using BaseAlias = Base;
using OtherBaseAlias = OtherBase;
using DerivedAlias = Derived;
using AlsoDerivedAlias = AlsoDerived;

struct Base {};
struct OtherBase {};
struct Derived : Base, OtherBase {};
struct AlsoDerived : BaseAlias, OtherBaseAlias {};

static_assert(__is_base_of(Base, Derived));
static_assert(__is_base_of(Base, DerivedAlias));
static_assert(__is_base_of(BaseAlias, Derived));
static_assert(__is_base_of(BaseAlias, DerivedAlias));

static_assert(__is_base_of(OtherBase, Derived));
static_assert(__is_base_of(OtherBase, DerivedAlias));
static_assert(__is_base_of(OtherBaseAlias, Derived));
static_assert(__is_base_of(OtherBaseAlias, DerivedAlias));

static_assert(__is_base_of(Base, AlsoDerived));
static_assert(__is_base_of(Base, AlsoDerivedAlias));
static_assert(__is_base_of(BaseAlias, AlsoDerived));
static_assert(__is_base_of(BaseAlias, AlsoDerivedAlias));

static_assert(__is_base_of(const Base, const AlsoDerived));
static_assert(__is_base_of(const Base, const AlsoDerivedAlias));
static_assert(__is_base_of(const BaseAlias, const AlsoDerived));
static_assert(__is_base_of(const BaseAlias, const AlsoDerivedAlias));

static_assert(__is_base_of(Derived, Derived));

Base base;

struct C : decltype(base) {};

static_assert(__is_base_of(Base, C));
