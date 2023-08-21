// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

enum class fruit { orange, apple };

struct S {
  using enum fruit;
};

//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        enum-specifier
// CHECK-NEXT:          name: simple-name
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
// CHECK-NEXT:          is-final: false
// CHECK-NEXT:          name: simple-name
// CHECK-NEXT:            identifier: S
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            using-enum-declaration
// CHECK-NEXT:              enum-type-specifier: elaborated-type-specifier
// CHECK-NEXT:                class-key: enum
// CHECK-NEXT:                name: simple-name
// CHECK-NEXT:                  identifier: fruit
