// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

const int values[] = {
    1,
    2,
    3,
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        const-qualifier
// CHECK-NEXT:        integral-type-specifier
// CHECK-NEXT:          specifier: int
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: values
// CHECK-NEXT:            declarator-chunk-list
// CHECK-NEXT:              array-declarator-chunk
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: braced-init-list
// CHECK-NEXT:              expression-list
// CHECK-NEXT:                int-literal-expression
// CHECK-NEXT:                  literal: 1
// CHECK-NEXT:                int-literal-expression
// CHECK-NEXT:                  literal: 2
// CHECK-NEXT:                int-literal-expression
// CHECK-NEXT:                  literal: 3
