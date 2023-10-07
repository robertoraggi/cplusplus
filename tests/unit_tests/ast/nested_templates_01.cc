// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

template <typename T, typename A>
class List {
  template <typename U>
  struct Rebind {
    using type = List<U, A>;
  };
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    template-declaration
// CHECK-NEXT:      template-parameter-list
// CHECK-NEXT:        typename-type-parameter
// CHECK-NEXT:          depth: 0
// CHECK-NEXT:          index: 0
// CHECK-NEXT:          identifier: T
// CHECK-NEXT:        typename-type-parameter
// CHECK-NEXT:          depth: 0
// CHECK-NEXT:          index: 1
// CHECK-NEXT:          identifier: A
// CHECK-NEXT:      declaration: simple-declaration
// CHECK-NEXT:        decl-specifier-list
// CHECK-NEXT:          class-specifier
// CHECK-NEXT:            class-key: class
// CHECK-NEXT:            unqualified-id: name-id
// CHECK-NEXT:              identifier: List
// CHECK-NEXT:            declaration-list
// CHECK-NEXT:              template-declaration
// CHECK-NEXT:                template-parameter-list
// CHECK-NEXT:                  typename-type-parameter
// CHECK-NEXT:                    depth: 1
// CHECK-NEXT:                    index: 0
// CHECK-NEXT:                    identifier: U
// CHECK-NEXT:                declaration: simple-declaration
// CHECK-NEXT:                  decl-specifier-list
// CHECK-NEXT:                    class-specifier
// CHECK-NEXT:                      class-key: struct
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: Rebind
// CHECK-NEXT:                      declaration-list
// CHECK-NEXT:                        alias-declaration
// CHECK-NEXT:                          identifier: type
// CHECK-NEXT:                          type-id: type-id
// CHECK-NEXT:                            type-specifier-list
// CHECK-NEXT:                              named-type-specifier
// CHECK-NEXT:                                unqualified-id: simple-template-id
// CHECK-NEXT:                                  identifier: List
// CHECK-NEXT:                                  template-argument-list
// CHECK-NEXT:                                    type-template-argument
// CHECK-NEXT:                                      type-id: type-id
// CHECK-NEXT:                                        type-specifier-list
// CHECK-NEXT:                                          named-type-specifier
// CHECK-NEXT:                                            unqualified-id: name-id
// CHECK-NEXT:                                              identifier: U
// CHECK-NEXT:                                        declarator: declarator
// CHECK-NEXT:                                    type-template-argument
// CHECK-NEXT:                                      type-id: type-id
// CHECK-NEXT:                                        type-specifier-list
// CHECK-NEXT:                                          named-type-specifier
// CHECK-NEXT:                                            unqualified-id: name-id
// CHECK-NEXT:                                              identifier: A
// CHECK-NEXT:                                        declarator: declarator
// CHECK-NEXT:                            declarator: declarator
