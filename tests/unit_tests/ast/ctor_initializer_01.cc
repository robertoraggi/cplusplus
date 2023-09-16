// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

struct Handle {
  int zero_;
  int value_;
  int otherValue_;

  Handle(int value, int otherValue)
      : zero_(), value_(value), otherValue_{otherValue} {}
};
// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: struct
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: Handle
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
// CHECK-NEXT:                          identifier: zero_
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
// CHECK-NEXT:                          identifier: value_
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
// CHECK-NEXT:                          identifier: otherValue_
// CHECK-NEXT:            function-definition
// CHECK-NEXT:              declarator: declarator
// CHECK-NEXT:                core-declarator: id-declarator
// CHECK-NEXT:                  declarator-id: id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: Handle
// CHECK-NEXT:                modifiers
// CHECK-NEXT:                  function-declarator
// CHECK-NEXT:                    parameters-and-qualifiers: parameters-and-qualifiers
// CHECK-NEXT:                      parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                        parameter-declaration-list
// CHECK-NEXT:                          parameter-declaration
// CHECK-NEXT:                            type-specifier-list
// CHECK-NEXT:                              integral-type-specifier
// CHECK-NEXT:                                specifier: int
// CHECK-NEXT:                            declarator: declarator
// CHECK-NEXT:                              core-declarator: id-declarator
// CHECK-NEXT:                                declarator-id: id-expression
// CHECK-NEXT:                                  unqualified-id: name-id
// CHECK-NEXT:                                    identifier: value
// CHECK-NEXT:                          parameter-declaration
// CHECK-NEXT:                            type-specifier-list
// CHECK-NEXT:                              integral-type-specifier
// CHECK-NEXT:                                specifier: int
// CHECK-NEXT:                            declarator: declarator
// CHECK-NEXT:                              core-declarator: id-declarator
// CHECK-NEXT:                                declarator-id: id-expression
// CHECK-NEXT:                                  unqualified-id: name-id
// CHECK-NEXT:                                    identifier: otherValue
// CHECK-NEXT:              function-body: compound-statement-function-body
// CHECK-NEXT:                ctor-initializer: ctor-initializer
// CHECK-NEXT:                  mem-initializer-list
// CHECK-NEXT:                    paren-mem-initializer
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: zero_
// CHECK-NEXT:                    paren-mem-initializer
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: value_
// CHECK-NEXT:                      expression-list
// CHECK-NEXT:                        id-expression
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: value
// CHECK-NEXT:                    braced-mem-initializer
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: otherValue_
// CHECK-NEXT:                      braced-init-list: braced-init-list
// CHECK-NEXT:                        expression-list
// CHECK-NEXT:                          id-expression
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: otherValue
// CHECK-NEXT:                statement: compound-statement
