// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

struct Pair {
  int first;
  int second;
};

auto pair = Pair{
    .first = 1,
    .second = 2,
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
// CHECK-NEXT:                      declarator-id: id-expression
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: first
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
// CHECK-NEXT:                          identifier: second
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        auto-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              declarator-id: id-expression
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: pair
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: braced-type-construction
// CHECK-NEXT:              type-specifier: named-type-specifier
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: Pair
// CHECK-NEXT:              braced-init-list: braced-init-list
// CHECK-NEXT:                expression-list
// CHECK-NEXT:                  designated-initializer-clause
// CHECK-NEXT:                    designator: designator
// CHECK-NEXT:                      identifier: first
// CHECK-NEXT:                    initializer: equal-initializer
// CHECK-NEXT:                      expression: int-literal-expression
// CHECK-NEXT:                        literal: 1
// CHECK-NEXT:                  designated-initializer-clause
// CHECK-NEXT:                    designator: designator
// CHECK-NEXT:                      identifier: second
// CHECK-NEXT:                    initializer: equal-initializer
// CHECK-NEXT:                      expression: int-literal-expression
// CHECK-NEXT:                        literal: 2
