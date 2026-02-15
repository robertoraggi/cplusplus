// clang-format off
// RUN: %cxx -verify -fcheck %s

struct A { int m; };
struct B { B(B const&) {} };
struct C { virtual void foo(); };
struct D { int m; D(D const&) = default; D(int x) : m(x + 1) {} };
struct E { ~E() {} };
struct F { virtual ~F() = default; };
struct G { G(G&&) {} };
struct H { H& operator=(H const&) { return *this; } };
struct I { I& operator=(I const&) = default; };
struct J { J(J const&) = delete; J(J&&) = default; };
struct K : A {};
struct L : B {};
struct M { B b; };
struct N : virtual A {};
struct O { int x; float y; };
union U { int i; float f; };
enum E1 { X, Y };
enum class SE { P, Q };

struct DeletedMoveCtor {
  DeletedMoveCtor(DeletedMoveCtor&&) = delete;
  DeletedMoveCtor& operator=(DeletedMoveCtor const&) = default;
};

struct AllCopyMoveDeleted {
  AllCopyMoveDeleted(AllCopyMoveDeleted const&) = delete;
  AllCopyMoveDeleted(AllCopyMoveDeleted&&) = delete;
};

//
// __is_trivially_copyable: scalar types
//

static_assert(__is_trivially_copyable(int), "int");
static_assert(__is_trivially_copyable(float), "float");
static_assert(__is_trivially_copyable(int*), "pointer");
static_assert(__is_trivially_copyable(E1), "enum");
static_assert(__is_trivially_copyable(SE), "scoped enum");

//
// __is_trivially_copyable: trivially copyable classes
//

static_assert(__is_trivially_copyable(A), "A");
static_assert(__is_trivially_copyable(D), "D");
static_assert(__is_trivially_copyable(I), "I");
static_assert(__is_trivially_copyable(J), "J");
static_assert(__is_trivially_copyable(K), "K");
static_assert(__is_trivially_copyable(O), "O");
static_assert(__is_trivially_copyable(U), "U");
static_assert(__is_trivially_copyable(DeletedMoveCtor), "DeletedMoveCtor");
static_assert(__is_trivially_copyable(AllCopyMoveDeleted), "AllCopyMoveDeleted");

//
// __is_trivially_copyable: cv-qualified and arrays
//

static_assert(__is_trivially_copyable(const A), "const A");
static_assert(__is_trivially_copyable(A[3]), "A[3]");
// expected-error@+1 {{"B[3]"}}
static_assert(__is_trivially_copyable(B[3]), "B[3]");

//
// __is_trivially_copyable: NOT trivially copyable
//

// expected-error@+1 {{"B"}}
static_assert(__is_trivially_copyable(B), "B");
// expected-error@+1 {{"C"}}
static_assert(__is_trivially_copyable(C), "C");
// expected-error@+1 {{"E"}}
static_assert(__is_trivially_copyable(E), "E");
// expected-error@+1 {{"F"}}
static_assert(__is_trivially_copyable(F), "F");
// expected-error@+1 {{"G"}}
static_assert(__is_trivially_copyable(G), "G");
// expected-error@+1 {{"H"}}
static_assert(__is_trivially_copyable(H), "H");
// expected-error@+1 {{"L"}}
static_assert(__is_trivially_copyable(L), "L");
// expected-error@+1 {{"M"}}
static_assert(__is_trivially_copyable(M), "M");
// expected-error@+1 {{"N"}}
static_assert(__is_trivially_copyable(N), "N");
