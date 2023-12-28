// clang-format off
// RUN: %cxx -verify -freflect -fcheck -ast-dump %s | %filecheck %s --match-full-lines
// clang-format on

constexpr auto int_ty = ^int;
constexpr auto ptr_ty = ^const void*;

constexpr[:int_ty:] i = 0;
constexpr[:ptr_ty:] ptr = nullptr;

constexpr auto z = ^123;
constexpr auto x = [:z:];

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        constexpr-specifier
// CHECK-NEXT:        auto-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: int_ty
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: type-id-reflect-expression
// CHECK-NEXT:              type-id: type-id
// CHECK-NEXT:                type-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: int
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        constexpr-specifier
// CHECK-NEXT:        auto-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: ptr_ty
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: type-id-reflect-expression
// CHECK-NEXT:              type-id: type-id
// CHECK-NEXT:                type-specifier-list
// CHECK-NEXT:                  const-qualifier
// CHECK-NEXT:                  void-type-specifier
// CHECK-NEXT:                declarator: declarator
// CHECK-NEXT:                  ptr-op-list
// CHECK-NEXT:                    pointer-operator
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        constexpr-specifier
// CHECK-NEXT:        splicer-type-specifier
// CHECK-NEXT:          splicer: splicer
// CHECK-NEXT:            expression: id-expression
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: int_ty
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: i
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: int-literal-expression
// CHECK-NEXT:              literal: 0
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        constexpr-specifier
// CHECK-NEXT:        splicer-type-specifier
// CHECK-NEXT:          splicer: splicer
// CHECK-NEXT:            expression: id-expression
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: ptr_ty
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: ptr
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: nullptr-literal-expression
// CHECK-NEXT:              literal: nullptr
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        constexpr-specifier
// CHECK-NEXT:        auto-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: z
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: reflect-expression
// CHECK-NEXT:              expression: int-literal-expression
// CHECK-NEXT:                literal: 123
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        constexpr-specifier
// CHECK-NEXT:        auto-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: x
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: splice-expression
// CHECK-NEXT:              splicer: splicer
// CHECK-NEXT:                expression: id-expression
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: z
