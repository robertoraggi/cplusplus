// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

void foo() {}

void foo(int x, auto... xs) { foo(xs...); }

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: foo
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: foo
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:            parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:              parameter-declaration-list
// CHECK-NEXT:                parameter-declaration
// CHECK-NEXT:                  identifier: x
// CHECK-NEXT:                  type-specifier-list
// CHECK-NEXT:                    integral-type-specifier
// CHECK-NEXT:                      specifier: int
// CHECK-NEXT:                  declarator: declarator
// CHECK-NEXT:                    core-declarator: id-declarator
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: x
// CHECK-NEXT:                parameter-declaration
// CHECK-NEXT:                  identifier: xs
// CHECK-NEXT:                  is-pack: true
// CHECK-NEXT:                  type-specifier-list
// CHECK-NEXT:                    named-type-specifier
// CHECK-NEXT:                  declarator: declarator
// CHECK-NEXT:                    core-declarator: parameter-pack
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: xs
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            expression-statement
// CHECK-NEXT:              expression: call-expression
// CHECK-NEXT:                base-expression: id-expression
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: foo
// CHECK-NEXT:                expression-list
// CHECK-NEXT:                  pack-expansion-expression
// CHECK-NEXT:                    expression: id-expression
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: xs
