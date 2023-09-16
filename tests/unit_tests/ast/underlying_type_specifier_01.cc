// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

enum class fruit { orange, apple };

using I = __underlying_type(fruit);
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
// CHECK-NEXT:    alias-declaration
// CHECK-NEXT:      identifier: I
// CHECK-NEXT:      type-id: type-id
// CHECK-NEXT:        type-specifier-list
// CHECK-NEXT:          underlying-type-specifier
// CHECK-NEXT:            type-id: type-id
// CHECK-NEXT:              type-specifier-list
// CHECK-NEXT:                named-type-specifier
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: fruit
// CHECK-NEXT:              declarator: declarator
// CHECK-NEXT:        declarator: declarator
