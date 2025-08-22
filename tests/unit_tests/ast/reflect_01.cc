// clang-format off
// RUN: %cxx -verify -fcheck -ast-dump %s | %filecheck %s --match-full-lines
// clang-format on

constexpr auto int_ty = ^^int;
constexpr auto ptr_ty = ^^const void*;

constexpr[:int_ty:] i = 0;
constexpr[:ptr_ty:] ptr = nullptr;

constexpr auto z = ^^123;
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
// CHECK-NEXT:          initializer: equal-initializer [prvalue __builtin_meta_info]
// CHECK-NEXT:            expression: type-id-reflect-expression [prvalue __builtin_meta_info]
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
// CHECK-NEXT:          initializer: equal-initializer [prvalue __builtin_meta_info]
// CHECK-NEXT:            expression: type-id-reflect-expression [prvalue __builtin_meta_info]
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
// CHECK-NEXT:            expression: id-expression [lvalue const __builtin_meta_info]
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: int_ty
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: i
// CHECK-NEXT:          initializer: equal-initializer [prvalue int]
// CHECK-NEXT:            expression: int-literal-expression [prvalue int]
// CHECK-NEXT:              literal: 0
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        constexpr-specifier
// CHECK-NEXT:        splicer-type-specifier
// CHECK-NEXT:          splicer: splicer
// CHECK-NEXT:            expression: id-expression [lvalue const __builtin_meta_info]
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: ptr_ty
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: ptr
// CHECK-NEXT:          initializer: implicit-cast-expression [prvalue const void*]
// CHECK-NEXT:            cast-kind: pointer-conversion
// CHECK-NEXT:            expression: equal-initializer [prvalue decltype(nullptr)]
// CHECK-NEXT:              expression: nullptr-literal-expression [prvalue decltype(nullptr)]
// CHECK-NEXT:                literal: nullptr
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
// CHECK-NEXT:          initializer: equal-initializer [prvalue __builtin_meta_info]
// CHECK-NEXT:            expression: reflect-expression [prvalue __builtin_meta_info]
// CHECK-NEXT:              expression: int-literal-expression [prvalue int]
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
// CHECK-NEXT:          initializer: equal-initializer [prvalue __builtin_meta_info]
// CHECK-NEXT:            expression: splice-expression [prvalue const __builtin_meta_info]
// CHECK-NEXT:              splicer: splicer
// CHECK-NEXT:                expression: id-expression [lvalue const __builtin_meta_info]
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: z
