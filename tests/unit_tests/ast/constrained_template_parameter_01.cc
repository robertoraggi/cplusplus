// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

template <typename T>
concept Any = true;

template <Any auto x>
struct ident {
  static constexpr auto value = x;
};

template <Any auto x>
constexpr auto ident_v = ident<x>::value;
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
// CHECK-NEXT:      declaration: concept-definition
// CHECK-NEXT:        identifier: Any
// CHECK-NEXT:        expression: bool-literal-expression [prvalue bool]
// CHECK-NEXT:          is-true: true
// CHECK-NEXT:    template-declaration
// CHECK-NEXT:      depth: 0
// CHECK-NEXT:      template-parameter-list
// CHECK-NEXT:        non-type-template-parameter
// CHECK-NEXT:          depth: 0
// CHECK-NEXT:          index: 0
// CHECK-NEXT:          declaration: parameter-declaration
// CHECK-NEXT:            identifier: x
// CHECK-NEXT:            type-specifier-list
// CHECK-NEXT:              placeholder-type-specifier
// CHECK-NEXT:                type-constraint: type-constraint
// CHECK-NEXT:                  identifier: Any
// CHECK-NEXT:                specifier: auto-type-specifier
// CHECK-NEXT:            declarator: declarator
// CHECK-NEXT:              core-declarator: id-declarator
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: x
// CHECK-NEXT:      declaration: simple-declaration
// CHECK-NEXT:        decl-specifier-list
// CHECK-NEXT:          class-specifier
// CHECK-NEXT:            class-key: struct
// CHECK-NEXT:            unqualified-id: name-id
// CHECK-NEXT:              identifier: ident
// CHECK-NEXT:            declaration-list
// CHECK-NEXT:              simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  static-specifier
// CHECK-NEXT:                  constexpr-specifier
// CHECK-NEXT:                  auto-type-specifier
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: value
// CHECK-NEXT:                    initializer: equal-initializer
// CHECK-NEXT:                      expression: id-expression
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: x
// CHECK-NEXT:    template-declaration
// CHECK-NEXT:      depth: 0
// CHECK-NEXT:      template-parameter-list
// CHECK-NEXT:        non-type-template-parameter
// CHECK-NEXT:          depth: 0
// CHECK-NEXT:          index: 0
// CHECK-NEXT:          declaration: parameter-declaration
// CHECK-NEXT:            identifier: x
// CHECK-NEXT:            type-specifier-list
// CHECK-NEXT:              placeholder-type-specifier
// CHECK-NEXT:                type-constraint: type-constraint
// CHECK-NEXT:                  identifier: Any
// CHECK-NEXT:                specifier: auto-type-specifier
// CHECK-NEXT:            declarator: declarator
// CHECK-NEXT:              core-declarator: id-declarator
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: x
// CHECK-NEXT:      declaration: simple-declaration
// CHECK-NEXT:        decl-specifier-list
// CHECK-NEXT:          constexpr-specifier
// CHECK-NEXT:          auto-type-specifier
// CHECK-NEXT:        init-declarator-list
// CHECK-NEXT:          init-declarator
// CHECK-NEXT:            declarator: declarator
// CHECK-NEXT:              core-declarator: id-declarator
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: ident_v
// CHECK-NEXT:            initializer: equal-initializer
// CHECK-NEXT:              expression: id-expression
// CHECK-NEXT:                nested-name-specifier: template-nested-name-specifier
// CHECK-NEXT:                  template-id: simple-template-id
// CHECK-NEXT:                    identifier: ident
// CHECK-NEXT:                    template-argument-list
// CHECK-NEXT:                      type-template-argument
// CHECK-NEXT:                        type-id: type-id
// CHECK-NEXT:                          type-specifier-list
// CHECK-NEXT:                            named-type-specifier
// CHECK-NEXT:                              unqualified-id: name-id
// CHECK-NEXT:                                identifier: x
// CHECK-NEXT:                unqualified-id: name-id
// CHECK-NEXT:                  identifier: value
