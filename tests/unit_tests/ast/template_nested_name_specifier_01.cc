// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

struct HelperT {
  template <typename T>
  struct Lookup {
    using PointerT = T*;
  };
};

template <int N>
void set(typename HelperT::template Lookup<int>::PointerT p);

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: struct
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: HelperT
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            template-declaration
// CHECK-NEXT:              depth: 0
// CHECK-NEXT:              template-parameter-list
// CHECK-NEXT:                typename-type-parameter
// CHECK-NEXT:                  depth: 0
// CHECK-NEXT:                  index: 0
// CHECK-NEXT:                  identifier: T
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  class-specifier
// CHECK-NEXT:                    class-key: struct
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: Lookup
// CHECK-NEXT:                    declaration-list
// CHECK-NEXT:                      alias-declaration
// CHECK-NEXT:                        identifier: PointerT
// CHECK-NEXT:                        type-id: type-id
// CHECK-NEXT:                          type-specifier-list
// CHECK-NEXT:                            named-type-specifier
// CHECK-NEXT:                              unqualified-id: name-id
// CHECK-NEXT:                                identifier: T
// CHECK-NEXT:                          declarator: declarator
// CHECK-NEXT:                            ptr-op-list
// CHECK-NEXT:                              pointer-operator
// CHECK-NEXT:    template-declaration
// CHECK-NEXT:      depth: 0
// CHECK-NEXT:      template-parameter-list
// CHECK-NEXT:        non-type-template-parameter
// CHECK-NEXT:          depth: 0
// CHECK-NEXT:          index: 0
// CHECK-NEXT:          declaration: parameter-declaration
// CHECK-NEXT:            identifier: N
// CHECK-NEXT:            type-specifier-list
// CHECK-NEXT:              integral-type-specifier
// CHECK-NEXT:                specifier: int
// CHECK-NEXT:            declarator: declarator
// CHECK-NEXT:              core-declarator: id-declarator
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: N
// CHECK-NEXT:      declaration: simple-declaration
// CHECK-NEXT:        decl-specifier-list
// CHECK-NEXT:          void-type-specifier
// CHECK-NEXT:        init-declarator-list
// CHECK-NEXT:          init-declarator
// CHECK-NEXT:            declarator: declarator
// CHECK-NEXT:              core-declarator: id-declarator
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: set
// CHECK-NEXT:              declarator-chunk-list
// CHECK-NEXT:                function-declarator-chunk
// CHECK-NEXT:                  parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                    parameter-declaration-list
// CHECK-NEXT:                      parameter-declaration
// CHECK-NEXT:                        identifier: p
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          typename-specifier
// CHECK-NEXT:                            nested-name-specifier: template-nested-name-specifier
// CHECK-NEXT:                              is-template-introduced: true
// CHECK-NEXT:                              nested-name-specifier: simple-nested-name-specifier
// CHECK-NEXT:                                identifier: HelperT
// CHECK-NEXT:                              template-id: simple-template-id
// CHECK-NEXT:                                identifier: Lookup
// CHECK-NEXT:                                template-argument-list
// CHECK-NEXT:                                  type-template-argument
// CHECK-NEXT:                                    type-id: type-id
// CHECK-NEXT:                                      type-specifier-list
// CHECK-NEXT:                                        integral-type-specifier
// CHECK-NEXT:                                          specifier: int
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: PointerT
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:                          core-declarator: id-declarator
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: p
