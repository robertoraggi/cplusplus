// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

int main() {
  int map[][2] = {{1, 2}};
  for (const auto& [key, value] : map) {
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
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: int
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        declarator-id: id-expression
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: map
// CHECK-NEXT:                      declarator-chunk-list
// CHECK-NEXT:                        array-declarator-chunk
// CHECK-NEXT:                        array-declarator-chunk
// CHECK-NEXT:                          expression: int-literal-expression
// CHECK-NEXT:                            literal: 2
// CHECK-NEXT:                    initializer: equal-initializer
// CHECK-NEXT:                      expression: braced-init-list
// CHECK-NEXT:                        expression-list
// CHECK-NEXT:                          braced-init-list
// CHECK-NEXT:                            expression-list
// CHECK-NEXT:                              int-literal-expression
// CHECK-NEXT:                                literal: 1
// CHECK-NEXT:                              int-literal-expression
// CHECK-NEXT:                                literal: 2
// CHECK-NEXT:            for-range-statement
// CHECK-NEXT:              range-declaration: structured-binding-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  const-qualifier
// CHECK-NEXT:                  auto-type-specifier
// CHECK-NEXT:                binding-list
// CHECK-NEXT:                  name-id
// CHECK-NEXT:                    identifier: key
// CHECK-NEXT:                  name-id
// CHECK-NEXT:                    identifier: value
// CHECK-NEXT:              range-initializer: id-expression
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: map
// CHECK-NEXT:              statement: compound-statement
