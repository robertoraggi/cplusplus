// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

enum class fruit { orange, apple };

struct S {
  using enum fruit;
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        enum-specifier
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: fruit
// CHECK-NEXT:          enumerator-list
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: orange
// CHECK-NEXT:            enumerator
// CHECK-NEXT:              identifier: apple
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: struct
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: S
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            using-enum-declaration
// CHECK-NEXT:              enum-type-specifier: elaborated-type-specifier
// CHECK-NEXT:                class-key: enum
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: fruit
