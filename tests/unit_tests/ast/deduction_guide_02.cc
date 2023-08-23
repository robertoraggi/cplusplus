// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

struct List {
  template <typename T>
  struct Node {
    Node(T) {}
  };
  Node(int) -> Node<int>;
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: struct
// CHECK-NEXT:          is-final: false
// CHECK-NEXT:          name: simple-name
// CHECK-NEXT:            identifier: List
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            template-declaration
// CHECK-NEXT:              template-parameter-list
// CHECK-NEXT:                typename-type-parameter
// CHECK-NEXT:                  identifier: T
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  class-specifier
// CHECK-NEXT:                    class-key: struct
// CHECK-NEXT:                    is-final: false
// CHECK-NEXT:                    name: simple-name
// CHECK-NEXT:                      identifier: Node
// CHECK-NEXT:                    declaration-list
// CHECK-NEXT:                      function-definition
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:                          core-declarator: id-declarator
// CHECK-NEXT:                            name: simple-name
// CHECK-NEXT:                              identifier: Node
// CHECK-NEXT:                          modifiers
// CHECK-NEXT:                            function-declarator
// CHECK-NEXT:                              parameters-and-qualifiers: parameters-and-qualifiers
// CHECK-NEXT:                                parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                                  parameter-declaration-list
// CHECK-NEXT:                                    parameter-declaration
// CHECK-NEXT:                                      type-specifier-list
// CHECK-NEXT:                                        named-type-specifier
// CHECK-NEXT:                                          name: simple-name
// CHECK-NEXT:                                            identifier: T
// CHECK-NEXT:                                      declarator: declarator
// CHECK-NEXT:                        function-body: compound-statement-function-body
// CHECK-NEXT:                          statement: compound-statement
// CHECK-NEXT:            deduction-guide
// CHECK-NEXT:              identifier: Node
// CHECK-NEXT:              parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                parameter-declaration-list
// CHECK-NEXT:                  parameter-declaration
// CHECK-NEXT:                    type-specifier-list
// CHECK-NEXT:                      integral-type-specifier
// CHECK-NEXT:                        specifier: int
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:              template-id: template-name
// CHECK-NEXT:                id: simple-name
// CHECK-NEXT:                  identifier: Node
// CHECK-NEXT:                template-argument-list
// CHECK-NEXT:                  type-template-argument
// CHECK-NEXT:                    type-id: type-id
// CHECK-NEXT:                      type-specifier-list
// CHECK-NEXT:                        integral-type-specifier
// CHECK-NEXT:                          specifier: int
// CHECK-NEXT:                      declarator: declarator
