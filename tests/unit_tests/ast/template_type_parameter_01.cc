// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

template <typename T, template <typename U> class C>
struct S {
  template <typename K>
  struct S2 {};
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    template-declaration
// CHECK-NEXT:      depth: 0
// CHECK-NEXT:      template-parameter-list
// CHECK-NEXT:        typename-type-parameter
// CHECK-NEXT:          depth: 0
// CHECK-NEXT:          index: 0
// CHECK-NEXT:          identifier: T
// CHECK-NEXT:        template-type-parameter
// CHECK-NEXT:          depth: 0
// CHECK-NEXT:          index: 1
// CHECK-NEXT:          identifier: C
// CHECK-NEXT:          template-parameter-list
// CHECK-NEXT:            typename-type-parameter
// CHECK-NEXT:              depth: 1
// CHECK-NEXT:              index: 0
// CHECK-NEXT:              identifier: U
// CHECK-NEXT:      declaration: simple-declaration
// CHECK-NEXT:        decl-specifier-list
// CHECK-NEXT:          class-specifier
// CHECK-NEXT:            class-key: struct
// CHECK-NEXT:            unqualified-id: name-id
// CHECK-NEXT:              identifier: S
// CHECK-NEXT:            declaration-list
// CHECK-NEXT:              template-declaration
// CHECK-NEXT:                depth: 1
// CHECK-NEXT:                template-parameter-list
// CHECK-NEXT:                  typename-type-parameter
// CHECK-NEXT:                    depth: 1
// CHECK-NEXT:                    index: 0
// CHECK-NEXT:                    identifier: K
// CHECK-NEXT:                declaration: simple-declaration
// CHECK-NEXT:                  decl-specifier-list
// CHECK-NEXT:                    class-specifier
// CHECK-NEXT:                      class-key: struct
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: S2
