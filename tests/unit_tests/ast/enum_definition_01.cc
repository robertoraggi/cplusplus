// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

enum Kind {
  k1,
  k2,
  k3,
};

enum class Kind2 {
  k1 = 0,
  k2 = 1,
  k3 = 2,
};

enum class Kind3 : int {
  k1 = 0,
  k2 = 1,
  k3 = 2,
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        enum-specifier
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: Kind
// CHECK-NEXT:          enumerator-list
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: k1
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: k2
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: k3
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        enum-specifier
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: Kind2
// CHECK-NEXT:          enumerator-list
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: k1
// CHECK-NEXT:              expression: int-literal-expression
// CHECK-NEXT:                literal: 0
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: k2
// CHECK-NEXT:              expression: int-literal-expression
// CHECK-NEXT:                literal: 1
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: k3
// CHECK-NEXT:              expression: int-literal-expression
// CHECK-NEXT:                literal: 2
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        enum-specifier
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: Kind3
// CHECK-NEXT:          type-specifier-list
// CHECK-NEXT:            integral-type-specifier
// CHECK-NEXT:              specifier: int
// CHECK-NEXT:          enumerator-list
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: k1
// CHECK-NEXT:              expression: int-literal-expression
// CHECK-NEXT:                literal: 0
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: k2
// CHECK-NEXT:              expression: int-literal-expression
// CHECK-NEXT:                literal: 1
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: k3
// CHECK-NEXT:              expression: int-literal-expression
// CHECK-NEXT:                literal: 2
