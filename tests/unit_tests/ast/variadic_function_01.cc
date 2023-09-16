// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

void print_like(const char* fmt, ...);
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
// CHECK-NEXT:              declarator-id: id-expression
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: print_like
// CHECK-NEXT:            modifiers
// CHECK-NEXT:              function-declarator
// CHECK-NEXT:                parameters-and-qualifiers: parameters-and-qualifiers
// CHECK-NEXT:                  parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                    is-variadic: true
// CHECK-NEXT:                    parameter-declaration-list
// CHECK-NEXT:                      parameter-declaration
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          const-qualifier
// CHECK-NEXT:                          integral-type-specifier
// CHECK-NEXT:                            specifier: char
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:                          ptr-op-list
// CHECK-NEXT:                            pointer-operator
// CHECK-NEXT:                          core-declarator: id-declarator
// CHECK-NEXT:                            declarator-id: id-expression
// CHECK-NEXT:                              unqualified-id: name-id
// CHECK-NEXT:                                identifier: fmt
