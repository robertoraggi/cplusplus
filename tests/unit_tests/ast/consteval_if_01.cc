// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

auto main() -> int {
  if consteval {
  }

  if !consteval {
  }

  if consteval {
  } else {
  }

  if !consteval {
  } else {
  }
}

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        auto-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          declarator-id: id-expression
// CHECK-NEXT:            unqualified-id: name-id
// CHECK-NEXT:              identifier: main
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:            parameters-and-qualifiers: parameters-and-qualifiers
// CHECK-NEXT:            trailing-return-type: trailing-return-type
// CHECK-NEXT:              type-id: type-id
// CHECK-NEXT:                type-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: int
// CHECK-NEXT:                declarator: declarator
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            consteval-if-statement
// CHECK-NEXT:              statement: compound-statement
// CHECK-NEXT:            consteval-if-statement
// CHECK-NEXT:              is-not: true
// CHECK-NEXT:              statement: compound-statement
// CHECK-NEXT:            consteval-if-statement
// CHECK-NEXT:              statement: compound-statement
// CHECK-NEXT:              else-statement: compound-statement
// CHECK-NEXT:            consteval-if-statement
// CHECK-NEXT:              is-not: true
// CHECK-NEXT:              statement: compound-statement
// CHECK-NEXT:              else-statement: compound-statement
