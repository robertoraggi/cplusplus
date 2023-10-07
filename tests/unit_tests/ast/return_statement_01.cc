// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

struct Pair {
  int first;
  int second;

  static auto zero() -> Pair { return {0, 0}; }
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: struct
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: Pair
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            simple-declaration
// CHECK-NEXT:              decl-specifier-list
// CHECK-NEXT:                integral-type-specifier
// CHECK-NEXT:                  specifier: int
// CHECK-NEXT:              init-declarator-list
// CHECK-NEXT:                init-declarator
// CHECK-NEXT:                  declarator: declarator
// CHECK-NEXT:                    core-declarator: id-declarator
// CHECK-NEXT:                      declarator-id: id-expression
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: first
// CHECK-NEXT:            simple-declaration
// CHECK-NEXT:              decl-specifier-list
// CHECK-NEXT:                integral-type-specifier
// CHECK-NEXT:                  specifier: int
// CHECK-NEXT:              init-declarator-list
// CHECK-NEXT:                init-declarator
// CHECK-NEXT:                  declarator: declarator
// CHECK-NEXT:                    core-declarator: id-declarator
// CHECK-NEXT:                      declarator-id: id-expression
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: second
// CHECK-NEXT:            function-definition
// CHECK-NEXT:              decl-specifier-list
// CHECK-NEXT:                static-specifier
// CHECK-NEXT:                auto-type-specifier
// CHECK-NEXT:              declarator: declarator
// CHECK-NEXT:                core-declarator: id-declarator
// CHECK-NEXT:                  declarator-id: id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: zero
// CHECK-NEXT:                declarator-chunk-list
// CHECK-NEXT:                  function-declarator-chunk
// CHECK-NEXT:                    trailing-return-type: trailing-return-type
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          named-type-specifier
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: Pair
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:              function-body: compound-statement-function-body
// CHECK-NEXT:                statement: compound-statement
// CHECK-NEXT:                  statement-list
// CHECK-NEXT:                    return-statement
// CHECK-NEXT:                      expression: braced-init-list
// CHECK-NEXT:                        expression-list
// CHECK-NEXT:                          int-literal-expression
// CHECK-NEXT:                            literal: 0
// CHECK-NEXT:                          int-literal-expression
// CHECK-NEXT:                            literal: 0
