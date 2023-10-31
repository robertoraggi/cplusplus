// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

template <typename T>
struct S {
  void f() {
    auto add = []<typename T1, typename T2>(T1 a, T2 b) { return a + b; };
  }
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    template-declaration
// CHECK-NEXT:      template-parameter-list
// CHECK-NEXT:        typename-type-parameter
// CHECK-NEXT:          depth: 0
// CHECK-NEXT:          index: 0
// CHECK-NEXT:          identifier: T
// CHECK-NEXT:      declaration: simple-declaration
// CHECK-NEXT:        decl-specifier-list
// CHECK-NEXT:          class-specifier
// CHECK-NEXT:            class-key: struct
// CHECK-NEXT:            unqualified-id: name-id
// CHECK-NEXT:              identifier: S
// CHECK-NEXT:            declaration-list
// CHECK-NEXT:              function-definition
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  void-type-specifier
// CHECK-NEXT:                declarator: declarator
// CHECK-NEXT:                  core-declarator: id-declarator
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: f
// CHECK-NEXT:                  declarator-chunk-list
// CHECK-NEXT:                    function-declarator-chunk
// CHECK-NEXT:                function-body: compound-statement-function-body
// CHECK-NEXT:                  statement: compound-statement
// CHECK-NEXT:                    statement-list
// CHECK-NEXT:                      declaration-statement
// CHECK-NEXT:                        declaration: simple-declaration
// CHECK-NEXT:                          decl-specifier-list
// CHECK-NEXT:                            auto-type-specifier
// CHECK-NEXT:                          init-declarator-list
// CHECK-NEXT:                            init-declarator
// CHECK-NEXT:                              declarator: declarator
// CHECK-NEXT:                                core-declarator: id-declarator
// CHECK-NEXT:                                  unqualified-id: name-id
// CHECK-NEXT:                                    identifier: add
// CHECK-NEXT:                              initializer: equal-initializer
// CHECK-NEXT:                                expression: lambda-expression
// CHECK-NEXT:                                  template-parameter-list
// CHECK-NEXT:                                    typename-type-parameter
// CHECK-NEXT:                                      depth: 1
// CHECK-NEXT:                                      index: 0
// CHECK-NEXT:                                      identifier: T1
// CHECK-NEXT:                                    typename-type-parameter
// CHECK-NEXT:                                      depth: 1
// CHECK-NEXT:                                      index: 1
// CHECK-NEXT:                                      identifier: T2
// CHECK-NEXT:                                  parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                                    parameter-declaration-list
// CHECK-NEXT:                                      parameter-declaration
// CHECK-NEXT:                                        identifier: a
// CHECK-NEXT:                                        type-specifier-list
// CHECK-NEXT:                                          named-type-specifier
// CHECK-NEXT:                                            unqualified-id: name-id
// CHECK-NEXT:                                              identifier: T1
// CHECK-NEXT:                                        declarator: declarator
// CHECK-NEXT:                                          core-declarator: id-declarator
// CHECK-NEXT:                                            unqualified-id: name-id
// CHECK-NEXT:                                              identifier: a
// CHECK-NEXT:                                      parameter-declaration
// CHECK-NEXT:                                        identifier: b
// CHECK-NEXT:                                        type-specifier-list
// CHECK-NEXT:                                          named-type-specifier
// CHECK-NEXT:                                            unqualified-id: name-id
// CHECK-NEXT:                                              identifier: T2
// CHECK-NEXT:                                        declarator: declarator
// CHECK-NEXT:                                          core-declarator: id-declarator
// CHECK-NEXT:                                            unqualified-id: name-id
// CHECK-NEXT:                                              identifier: b
// CHECK-NEXT:                                  statement: compound-statement
// CHECK-NEXT:                                    statement-list
// CHECK-NEXT:                                      return-statement
// CHECK-NEXT:                                        expression: binary-expression
// CHECK-NEXT:                                          op: +
// CHECK-NEXT:                                          left-expression: id-expression
// CHECK-NEXT:                                            unqualified-id: name-id
// CHECK-NEXT:                                              identifier: a
// CHECK-NEXT:                                          right-expression: id-expression
// CHECK-NEXT:                                            unqualified-id: name-id
// CHECK-NEXT:                                              identifier: b
