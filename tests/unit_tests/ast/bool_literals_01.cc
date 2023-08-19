// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

const bool ok = true;
const bool ko = false;

//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        const-qualifier
// CHECK-NEXT:        integral-type-specifier
// CHECK-NEXT:          specifier: bool
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              name: simple-name
// CHECK-NEXT:                identifier: ok
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: bool-literal-expression
// CHECK-NEXT:              value: true
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        const-qualifier
// CHECK-NEXT:        integral-type-specifier
// CHECK-NEXT:          specifier: bool
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              name: simple-name
// CHECK-NEXT:                identifier: ko
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: bool-literal-expression
// CHECK-NEXT:              value: false