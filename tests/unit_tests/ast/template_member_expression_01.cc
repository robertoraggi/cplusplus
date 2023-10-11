// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

struct Allocator {
  template <typename T>
  auto Allocate(int) -> T * {
    return nullptr;
  }
};

auto copy(Allocator &A) -> void * { return A.template Allocate<char>(128); }

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: struct
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: Allocator
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            template-declaration
// CHECK-NEXT:              template-parameter-list
// CHECK-NEXT:                typename-type-parameter
// CHECK-NEXT:                  depth: 0
// CHECK-NEXT:                  index: 0
// CHECK-NEXT:                  identifier: T
// CHECK-NEXT:              declaration: function-definition
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  auto-type-specifier
// CHECK-NEXT:                declarator: declarator
// CHECK-NEXT:                  core-declarator: id-declarator
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: Allocate
// CHECK-NEXT:                  declarator-chunk-list
// CHECK-NEXT:                    function-declarator-chunk
// CHECK-NEXT:                      parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                        parameter-declaration-list
// CHECK-NEXT:                          parameter-declaration
// CHECK-NEXT:                            type-specifier-list
// CHECK-NEXT:                              integral-type-specifier
// CHECK-NEXT:                                specifier: int
// CHECK-NEXT:                      trailing-return-type: trailing-return-type
// CHECK-NEXT:                        type-id: type-id
// CHECK-NEXT:                          type-specifier-list
// CHECK-NEXT:                            named-type-specifier
// CHECK-NEXT:                              unqualified-id: name-id
// CHECK-NEXT:                                identifier: T
// CHECK-NEXT:                          declarator: declarator
// CHECK-NEXT:                            ptr-op-list
// CHECK-NEXT:                              pointer-operator
// CHECK-NEXT:                function-body: compound-statement-function-body
// CHECK-NEXT:                  statement: compound-statement
// CHECK-NEXT:                    statement-list
// CHECK-NEXT:                      return-statement
// CHECK-NEXT:                        expression: nullptr-literal-expression
// CHECK-NEXT:                          literal: nullptr
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        auto-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: copy
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:            parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:              parameter-declaration-list
// CHECK-NEXT:                parameter-declaration
// CHECK-NEXT:                  type-specifier-list
// CHECK-NEXT:                    named-type-specifier
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: Allocator
// CHECK-NEXT:                  declarator: declarator
// CHECK-NEXT:                    ptr-op-list
// CHECK-NEXT:                      reference-operator
// CHECK-NEXT:                        ref-op: &
// CHECK-NEXT:                    core-declarator: id-declarator
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: A
// CHECK-NEXT:            trailing-return-type: trailing-return-type
// CHECK-NEXT:              type-id: type-id
// CHECK-NEXT:                type-specifier-list
// CHECK-NEXT:                  void-type-specifier
// CHECK-NEXT:                declarator: declarator
// CHECK-NEXT:                  ptr-op-list
// CHECK-NEXT:                    pointer-operator
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            return-statement
// CHECK-NEXT:              expression: call-expression
// CHECK-NEXT:                base-expression: member-expression
// CHECK-NEXT:                  access-op: .
// CHECK-NEXT:                  base-expression: id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: A
// CHECK-NEXT:                  member-id: id-expression
// CHECK-NEXT:                    is-template-introduced: true
// CHECK-NEXT:                    unqualified-id: simple-template-id
// CHECK-NEXT:                      identifier: Allocate
// CHECK-NEXT:                      template-argument-list
// CHECK-NEXT:                        type-template-argument
// CHECK-NEXT:                          type-id: type-id
// CHECK-NEXT:                            type-specifier-list
// CHECK-NEXT:                              integral-type-specifier
// CHECK-NEXT:                                specifier: char
// CHECK-NEXT:                expression-list
// CHECK-NEXT:                  int-literal-expression
// CHECK-NEXT:                    literal: 128
