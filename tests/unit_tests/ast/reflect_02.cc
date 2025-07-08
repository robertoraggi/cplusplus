// clang-format off
// RUN: %cxx -verify -fcheck -ast-dump %s | %filecheck %s --match-full-lines
// clang-format on

struct S {
  int x;
};

auto main() -> int { return S{.x = 10}.[:^S::x:]; }

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: struct
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: S
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            simple-declaration
// CHECK-NEXT:              decl-specifier-list
// CHECK-NEXT:                integral-type-specifier
// CHECK-NEXT:                  specifier: int
// CHECK-NEXT:              init-declarator-list
// CHECK-NEXT:                init-declarator
// CHECK-NEXT:                  declarator: declarator
// CHECK-NEXT:                    core-declarator: id-declarator
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: x
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
// CHECK-NEXT:            return-statement
// CHECK-NEXT:              expression: splice-member-expression
// CHECK-NEXT:                access-op: .
// CHECK-NEXT:                base-expression: braced-type-construction
// CHECK-NEXT:                  type-specifier: named-type-specifier
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: S
// CHECK-NEXT:                  braced-init-list: braced-init-list
// CHECK-NEXT:                    expression-list
// CHECK-NEXT:                      designated-initializer-clause
// CHECK-NEXT:                        designator-list
// CHECK-NEXT:                          dot-designator
// CHECK-NEXT:                            identifier: x
// CHECK-NEXT:                        initializer: equal-initializer [prvalue int]
// CHECK-NEXT:                          expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                            literal: 10
// CHECK-NEXT:                splicer: splicer
// CHECK-NEXT:                  expression: reflect-expression
// CHECK-NEXT:                    expression: id-expression [lvalue int]
// CHECK-NEXT:                      nested-name-specifier: simple-nested-name-specifier
// CHECK-NEXT:                        identifier: S
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: x
