// RUN: %cxx -ast-dump -verify -fcheck %s | %filecheck %s

auto main() -> int {
  void* ptr;
  bool b;
  b = ptr;
}

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        auto-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: main
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:            trailing-return-type: trailing-return-type
// CHECK-NEXT:              type-id: type-id
// CHECK-NEXT:                type-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: int
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  void-type-specifier
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      ptr-op-list
// CHECK-NEXT:                        pointer-operator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: ptr
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: bool
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: b
// CHECK-NEXT:            expression-statement
// CHECK-NEXT:              expression: assignment-expression [prvalue bool]
// CHECK-NEXT:                op: =
// CHECK-NEXT:                left-expression: id-expression [lvalue bool]
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: b
// CHECK-NEXT:                right-expression: implicit-cast-expression [prvalue bool]
// CHECK-NEXT:                  cast-kind: boolean-conversion
// CHECK-NEXT:                  expression: implicit-cast-expression [prvalue void*]
// CHECK-NEXT:                    cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                    expression: id-expression [lvalue void*]
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: ptr
