// RUN: %cxx -ast-dump -verify -fcheck %s | %filecheck %s

void f(int a, int b);

auto main() -> int {
  void (*p)(int, int);
  p = f;
}

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: f
// CHECK-NEXT:            declarator-chunk-list
// CHECK-NEXT:              function-declarator-chunk
// CHECK-NEXT:                parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                  parameter-declaration-list
// CHECK-NEXT:                    parameter-declaration
// CHECK-NEXT:                      identifier: a
// CHECK-NEXT:                      type-specifier-list
// CHECK-NEXT:                        integral-type-specifier
// CHECK-NEXT:                          specifier: int
// CHECK-NEXT:                      declarator: declarator
// CHECK-NEXT:                        core-declarator: id-declarator
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: a
// CHECK-NEXT:                    parameter-declaration
// CHECK-NEXT:                      identifier: b
// CHECK-NEXT:                      type-specifier-list
// CHECK-NEXT:                        integral-type-specifier
// CHECK-NEXT:                          specifier: int
// CHECK-NEXT:                      declarator: declarator
// CHECK-NEXT:                        core-declarator: id-declarator
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: b
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        auto-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: main
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:            trailing-return-type: trailing-return-type
// CHECK-NEXT:              type-id: type-id
// CHECK-NEXT:                type-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: int
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  void-type-specifier
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: nested-declarator
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:                          ptr-op-list
// CHECK-NEXT:                            pointer-operator
// CHECK-NEXT:                          core-declarator: id-declarator
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: p
// CHECK-NEXT:                      declarator-chunk-list
// CHECK-NEXT:                        function-declarator-chunk
// CHECK-NEXT:                          parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                            parameter-declaration-list
// CHECK-NEXT:                              parameter-declaration
// CHECK-NEXT:                                type-specifier-list
// CHECK-NEXT:                                  integral-type-specifier
// CHECK-NEXT:                                    specifier: int
// CHECK-NEXT:                              parameter-declaration
// CHECK-NEXT:                                type-specifier-list
// CHECK-NEXT:                                  integral-type-specifier
// CHECK-NEXT:                                    specifier: int
// CHECK-NEXT:            expression-statement
// CHECK-NEXT:              expression: assignment-expression
// CHECK-NEXT:                op: =
// CHECK-NEXT:                left-expression: id-expression
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: p
// CHECK-NEXT:                right-expression: implicit-cast-expression
// CHECK-NEXT:                  cast-kind: function-pointer-conversion
// CHECK-NEXT:                  expression: implicit-cast-expression
// CHECK-NEXT:                    cast-kind: function-to-pointer-conversion
// CHECK-NEXT:                    expression: id-expression
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: f
