// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

auto main() -> int {
  auto consteval_lambda = []() consteval { return 0; };
  auto constexpr_lambda = []() constexpr { return 0; };
  auto mutable_lambda = []() mutable { return 0; };
  auto static_lambda = []() static { return 0; };
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
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
// CHECK-NEXT:                  auto-type-specifier
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: consteval_lambda
// CHECK-NEXT:                    initializer: equal-initializer
// CHECK-NEXT:                      expression: lambda-expression
// CHECK-NEXT:                        lambda-specifier-list
// CHECK-NEXT:                          lambda-specifier
// CHECK-NEXT:                            specifier: consteval
// CHECK-NEXT:                        statement: compound-statement
// CHECK-NEXT:                          statement-list
// CHECK-NEXT:                            return-statement
// CHECK-NEXT:                              expression: int-literal-expression
// CHECK-NEXT:                                literal: 0
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  auto-type-specifier
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: constexpr_lambda
// CHECK-NEXT:                    initializer: equal-initializer
// CHECK-NEXT:                      expression: lambda-expression
// CHECK-NEXT:                        lambda-specifier-list
// CHECK-NEXT:                          lambda-specifier
// CHECK-NEXT:                            specifier: constexpr
// CHECK-NEXT:                        statement: compound-statement
// CHECK-NEXT:                          statement-list
// CHECK-NEXT:                            return-statement
// CHECK-NEXT:                              expression: int-literal-expression
// CHECK-NEXT:                                literal: 0
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  auto-type-specifier
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: mutable_lambda
// CHECK-NEXT:                    initializer: equal-initializer
// CHECK-NEXT:                      expression: lambda-expression
// CHECK-NEXT:                        lambda-specifier-list
// CHECK-NEXT:                          lambda-specifier
// CHECK-NEXT:                            specifier: mutable
// CHECK-NEXT:                        statement: compound-statement
// CHECK-NEXT:                          statement-list
// CHECK-NEXT:                            return-statement
// CHECK-NEXT:                              expression: int-literal-expression
// CHECK-NEXT:                                literal: 0
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  auto-type-specifier
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: static_lambda
// CHECK-NEXT:                    initializer: equal-initializer
// CHECK-NEXT:                      expression: lambda-expression
// CHECK-NEXT:                        lambda-specifier-list
// CHECK-NEXT:                          lambda-specifier
// CHECK-NEXT:                            specifier: static
// CHECK-NEXT:                        statement: compound-statement
// CHECK-NEXT:                          statement-list
// CHECK-NEXT:                            return-statement
// CHECK-NEXT:                              expression: int-literal-expression
// CHECK-NEXT:                                literal: 0
// CHECK-NEXT:    empty-declaration
