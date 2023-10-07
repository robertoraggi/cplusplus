// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

struct Pair {
  int first;
  int second;

  auto sum() -> int {
    auto [a, b] = *this;
    return a + b;
  }
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
// CHECK-NEXT:                auto-type-specifier
// CHECK-NEXT:              declarator: declarator
// CHECK-NEXT:                core-declarator: id-declarator
// CHECK-NEXT:                  declarator-id: id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: sum
// CHECK-NEXT:                declarator-chunk-list
// CHECK-NEXT:                  function-declarator-chunk
// CHECK-NEXT:                    trailing-return-type: trailing-return-type
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          integral-type-specifier
// CHECK-NEXT:                            specifier: int
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:              function-body: compound-statement-function-body
// CHECK-NEXT:                statement: compound-statement
// CHECK-NEXT:                  statement-list
// CHECK-NEXT:                    declaration-statement
// CHECK-NEXT:                      declaration: structured-binding-declaration
// CHECK-NEXT:                        decl-specifier-list
// CHECK-NEXT:                          auto-type-specifier
// CHECK-NEXT:                        binding-list
// CHECK-NEXT:                          name-id
// CHECK-NEXT:                            identifier: a
// CHECK-NEXT:                          name-id
// CHECK-NEXT:                            identifier: b
// CHECK-NEXT:                        initializer: equal-initializer
// CHECK-NEXT:                          expression: unary-expression
// CHECK-NEXT:                            op: *
// CHECK-NEXT:                            expression: this-expression
// CHECK-NEXT:                    return-statement
// CHECK-NEXT:                      expression: binary-expression
// CHECK-NEXT:                        op: +
// CHECK-NEXT:                        left-expression: id-expression
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: a
// CHECK-NEXT:                        right-expression: id-expression
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: b
