// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

int main() {
  for (int i : {10, 20, 30}) {
  }
}
// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        integral-type-specifier
// CHECK-NEXT:          specifier: int
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          declarator-id: id-expression
// CHECK-NEXT:            unqualified-id: name-id
// CHECK-NEXT:              identifier: main
// CHECK-NEXT:        modifiers
// CHECK-NEXT:          function-declarator
// CHECK-NEXT:            parameters-and-qualifiers: parameters-and-qualifiers
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            for-range-statement
// CHECK-NEXT:              range-declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: int
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        declarator-id: id-expression
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: i
// CHECK-NEXT:              range-initializer: braced-init-list
// CHECK-NEXT:                expression-list
// CHECK-NEXT:                  int-literal-expression
// CHECK-NEXT:                    literal: 10
// CHECK-NEXT:                  int-literal-expression
// CHECK-NEXT:                    literal: 20
// CHECK-NEXT:                  int-literal-expression
// CHECK-NEXT:                    literal: 30
// CHECK-NEXT:              statement: compound-statement
