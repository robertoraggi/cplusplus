// RUN: %cxx -fsyntax-only -verify -ast-dump %s | %filecheck %s

void f(int& __restrict__ a, int* __restrict__ b);
void g(int& __restrict a);

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
// CHECK-NEXT:                        ptr-op-list
// CHECK-NEXT:                          reference-operator
// CHECK-NEXT:                            ref-op: &
// CHECK-NEXT:                            is-restrict: true
// CHECK-NEXT:                        core-declarator: id-declarator
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: a
// CHECK-NEXT:                    parameter-declaration
// CHECK-NEXT:                      identifier: b
// CHECK-NEXT:                      type-specifier-list
// CHECK-NEXT:                        integral-type-specifier
// CHECK-NEXT:                          specifier: int
// CHECK-NEXT:                      declarator: declarator
// CHECK-NEXT:                        ptr-op-list
// CHECK-NEXT:                          pointer-operator
// CHECK-NEXT:                            cv-qualifier-list
// CHECK-NEXT:                              restrict-qualifier
// CHECK-NEXT:                        core-declarator: id-declarator
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: b
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: g
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
// CHECK-NEXT:                        ptr-op-list
// CHECK-NEXT:                          reference-operator
// CHECK-NEXT:                            ref-op: &
// CHECK-NEXT:                            is-restrict: true
// CHECK-NEXT:                        core-declarator: id-declarator
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: a
