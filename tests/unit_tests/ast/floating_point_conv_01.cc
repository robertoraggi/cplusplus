// RUN: %cxx -ast-dump -verify -fcheck %s | %filecheck %s

void f(int a, int b);

auto main() -> int {
  float f;
  double d;
  f = d;
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
// CHECK-NEXT:                  floating-point-type-specifier
// CHECK-NEXT:                    specifier: float
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: f
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  floating-point-type-specifier
// CHECK-NEXT:                    specifier: double
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: d
// CHECK-NEXT:            expression-statement
// CHECK-NEXT:              expression: assignment-expression [lvalue float]
// CHECK-NEXT:                op: =
// CHECK-NEXT:                left-expression: id-expression [lvalue float]
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: f
// CHECK-NEXT:                right-expression: implicit-cast-expression [prvalue float]
// CHECK-NEXT:                  cast-kind: floating-point-conversion
// CHECK-NEXT:                  expression: implicit-cast-expression [prvalue double]
// CHECK-NEXT:                    cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                    expression: id-expression [lvalue double]
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: d
