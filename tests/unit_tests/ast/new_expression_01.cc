// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

template <typename T>
auto make(void* where, T init) {
  auto a = new (where) T{init};
  auto b = new (where) T(init);
}

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    template-declaration
// CHECK-NEXT:      template-parameter-list
// CHECK-NEXT:        typename-type-parameter
// CHECK-NEXT:          depth: 0
// CHECK-NEXT:          index: 0
// CHECK-NEXT:          identifier: T
// CHECK-NEXT:      declaration: function-definition
// CHECK-NEXT:        decl-specifier-list
// CHECK-NEXT:          auto-type-specifier
// CHECK-NEXT:        declarator: declarator
// CHECK-NEXT:          core-declarator: id-declarator
// CHECK-NEXT:            unqualified-id: name-id
// CHECK-NEXT:              identifier: make
// CHECK-NEXT:          declarator-chunk-list
// CHECK-NEXT:            function-declarator-chunk
// CHECK-NEXT:              parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                parameter-declaration-list
// CHECK-NEXT:                  parameter-declaration
// CHECK-NEXT:                    type-specifier-list
// CHECK-NEXT:                      void-type-specifier
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      ptr-op-list
// CHECK-NEXT:                        pointer-operator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: where
// CHECK-NEXT:                  parameter-declaration
// CHECK-NEXT:                    type-specifier-list
// CHECK-NEXT:                      named-type-specifier
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: T
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: init
// CHECK-NEXT:        function-body: compound-statement-function-body
// CHECK-NEXT:          statement: compound-statement
// CHECK-NEXT:            statement-list
// CHECK-NEXT:              declaration-statement
// CHECK-NEXT:                declaration: simple-declaration
// CHECK-NEXT:                  decl-specifier-list
// CHECK-NEXT:                    auto-type-specifier
// CHECK-NEXT:                  init-declarator-list
// CHECK-NEXT:                    init-declarator
// CHECK-NEXT:                      declarator: declarator
// CHECK-NEXT:                        core-declarator: id-declarator
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: a
// CHECK-NEXT:                      initializer: equal-initializer
// CHECK-NEXT:                        expression: new-expression
// CHECK-NEXT:                          new-placement: new-placement
// CHECK-NEXT:                            expression-list
// CHECK-NEXT:                              id-expression
// CHECK-NEXT:                                unqualified-id: name-id
// CHECK-NEXT:                                  identifier: where
// CHECK-NEXT:                          type-specifier-list
// CHECK-NEXT:                            named-type-specifier
// CHECK-NEXT:                              unqualified-id: name-id
// CHECK-NEXT:                                identifier: T
// CHECK-NEXT:                          new-initalizer: new-braced-initializer
// CHECK-NEXT:                            braced-init-list: braced-init-list
// CHECK-NEXT:                              expression-list
// CHECK-NEXT:                                id-expression
// CHECK-NEXT:                                  unqualified-id: name-id
// CHECK-NEXT:                                    identifier: init
// CHECK-NEXT:              declaration-statement
// CHECK-NEXT:                declaration: simple-declaration
// CHECK-NEXT:                  decl-specifier-list
// CHECK-NEXT:                    auto-type-specifier
// CHECK-NEXT:                  init-declarator-list
// CHECK-NEXT:                    init-declarator
// CHECK-NEXT:                      declarator: declarator
// CHECK-NEXT:                        core-declarator: id-declarator
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: b
// CHECK-NEXT:                      initializer: equal-initializer
// CHECK-NEXT:                        expression: new-expression
// CHECK-NEXT:                          new-placement: new-placement
// CHECK-NEXT:                            expression-list
// CHECK-NEXT:                              id-expression
// CHECK-NEXT:                                unqualified-id: name-id
// CHECK-NEXT:                                  identifier: where
// CHECK-NEXT:                          type-specifier-list
// CHECK-NEXT:                            named-type-specifier
// CHECK-NEXT:                              unqualified-id: name-id
// CHECK-NEXT:                                identifier: T
// CHECK-NEXT:                          new-initalizer: new-paren-initializer
// CHECK-NEXT:                            expression-list
// CHECK-NEXT:                              id-expression
// CHECK-NEXT:                                unqualified-id: name-id
// CHECK-NEXT:                                  identifier: init
