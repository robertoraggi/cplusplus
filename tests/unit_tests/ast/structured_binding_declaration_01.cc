// RUN: %cxx -fsyntax-only -verify -ast-dump %s | %filecheck %s --match-full-lines

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
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: first
// CHECK-NEXT:            simple-declaration
// CHECK-NEXT:              decl-specifier-list
// CHECK-NEXT:                integral-type-specifier
// CHECK-NEXT:                  specifier: int
// CHECK-NEXT:              init-declarator-list
// CHECK-NEXT:                init-declarator
// CHECK-NEXT:                  declarator: declarator
// CHECK-NEXT:                    core-declarator: id-declarator
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: second
// CHECK-NEXT:            function-definition
// CHECK-NEXT:              decl-specifier-list
// CHECK-NEXT:                auto-type-specifier
// CHECK-NEXT:              declarator: declarator
// CHECK-NEXT:                core-declarator: id-declarator
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: sum
// CHECK-NEXT:                declarator-chunk-list
// CHECK-NEXT:                  function-declarator-chunk
// CHECK-NEXT:                    trailing-return-type: trailing-return-type
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          integral-type-specifier
// CHECK-NEXT:                            specifier: int
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
// CHECK-NEXT:                        initializer: equal-initializer [lvalue ::Pair]
// CHECK-NEXT:                          expression: implicit-cast-expression [lvalue const ::Pair]
// CHECK-NEXT:                            cast-kind: qualification-conversion
// CHECK-NEXT:                            expression: unary-expression [lvalue ::Pair]
// CHECK-NEXT:                              op: *
// CHECK-NEXT:                              expression: this-expression [prvalue ::Pair*]
// CHECK-NEXT:                        hidden-variable: init-declarator
// CHECK-NEXT:                          declarator: declarator
// CHECK-NEXT:                            core-declarator: id-declarator
// CHECK-NEXT:                              unqualified-id: name-id
// CHECK-NEXT:                                identifier: $e{{[0-9]+}}
// CHECK-NEXT:                          initializer: equal-initializer [lvalue ::Pair]
// CHECK-NEXT:                            expression: implicit-cast-expression [lvalue const ::Pair]
// CHECK-NEXT:                              cast-kind: qualification-conversion
// CHECK-NEXT:                              expression: unary-expression [lvalue ::Pair]
// CHECK-NEXT:                                op: *
// CHECK-NEXT:                                expression: this-expression [prvalue ::Pair*]
// CHECK-NEXT:                        binding-declarator-list
// CHECK-NEXT:                          init-declarator
// CHECK-NEXT:                            declarator: declarator
// CHECK-NEXT:                              ptr-op-list
// CHECK-NEXT:                                reference-operator
// CHECK-NEXT:                                  ref-op: &
// CHECK-NEXT:                              core-declarator: id-declarator
// CHECK-NEXT:                                unqualified-id: name-id
// CHECK-NEXT:                                  identifier: a
// CHECK-NEXT:                            initializer: equal-initializer [lvalue int]
// CHECK-NEXT:                              expression: member-expression [lvalue int]
// CHECK-NEXT:                                access-op: .
// CHECK-NEXT:                                base-expression: id-expression [lvalue ::Pair]
// CHECK-NEXT:                                  unqualified-id: name-id
// CHECK-NEXT:                                    identifier: $e{{[0-9]+}}
// CHECK-NEXT:                                unqualified-id: name-id
// CHECK-NEXT:                                  identifier: first
// CHECK-NEXT:                          init-declarator
// CHECK-NEXT:                            declarator: declarator
// CHECK-NEXT:                              ptr-op-list
// CHECK-NEXT:                                reference-operator
// CHECK-NEXT:                                  ref-op: &
// CHECK-NEXT:                              core-declarator: id-declarator
// CHECK-NEXT:                                unqualified-id: name-id
// CHECK-NEXT:                                  identifier: b
// CHECK-NEXT:                            initializer: equal-initializer [lvalue int]
// CHECK-NEXT:                              expression: member-expression [lvalue int]
// CHECK-NEXT:                                access-op: .
// CHECK-NEXT:                                base-expression: id-expression [lvalue ::Pair]
// CHECK-NEXT:                                  unqualified-id: name-id
// CHECK-NEXT:                                    identifier: $e{{[0-9]+}}
// CHECK-NEXT:                                unqualified-id: name-id
// CHECK-NEXT:                                  identifier: second
// CHECK-NEXT:                    return-statement
// CHECK-NEXT:                      expression: binary-expression [prvalue int]
// CHECK-NEXT:                        op: +
// CHECK-NEXT:                        left-expression: implicit-cast-expression [prvalue int]
// CHECK-NEXT:                          cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                          expression: id-expression [lvalue int]
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: a
// CHECK-NEXT:                        right-expression: implicit-cast-expression [prvalue int]
// CHECK-NEXT:                          cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                          expression: id-expression [lvalue int]
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: b
