// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

template <typename T>
concept ok = true;

struct object {
  auto clone(this auto self) { return self; }
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    template-declaration
// CHECK-NEXT:      template-parameter-list
// CHECK-NEXT:        typename-type-parameter
// CHECK-NEXT:          identifier: T
// CHECK-NEXT:      declaration: concept-definition
// CHECK-NEXT:        identifier: ok
// CHECK-NEXT:        expression: bool-literal-expression
// CHECK-NEXT:          is-true: true
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: struct
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: object
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            function-definition
// CHECK-NEXT:              decl-specifier-list
// CHECK-NEXT:                auto-type-specifier
// CHECK-NEXT:              declarator: declarator
// CHECK-NEXT:                core-declarator: id-declarator
// CHECK-NEXT:                  declarator-id: id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: clone
// CHECK-NEXT:                declarator-chunk-list
// CHECK-NEXT:                  function-declarator-chunk
// CHECK-NEXT:                    parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                      parameter-declaration-list
// CHECK-NEXT:                        parameter-declaration
// CHECK-NEXT:                          is-this-introduced: true
// CHECK-NEXT:                          type-specifier-list
// CHECK-NEXT:                            auto-type-specifier
// CHECK-NEXT:                          declarator: declarator
// CHECK-NEXT:                            core-declarator: id-declarator
// CHECK-NEXT:                              declarator-id: id-expression
// CHECK-NEXT:                                unqualified-id: name-id
// CHECK-NEXT:                                  identifier: self
// CHECK-NEXT:              function-body: compound-statement-function-body
// CHECK-NEXT:                statement: compound-statement
// CHECK-NEXT:                  statement-list
// CHECK-NEXT:                    return-statement
// CHECK-NEXT:                      expression: id-expression
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: self
