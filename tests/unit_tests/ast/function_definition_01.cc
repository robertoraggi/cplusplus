// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

void (*ptr)(){};

void (*f(int))() {}

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: nested-declarator
// CHECK-NEXT:              declarator: declarator
// CHECK-NEXT:                ptr-op-list
// CHECK-NEXT:                  pointer-operator
// CHECK-NEXT:                core-declarator: id-declarator
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: ptr
// CHECK-NEXT:            declarator-chunk-list
// CHECK-NEXT:              function-declarator-chunk
// CHECK-NEXT:          initializer: braced-init-list [prvalue void (*)()]
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: nested-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            ptr-op-list
// CHECK-NEXT:              pointer-operator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: f
// CHECK-NEXT:            declarator-chunk-list
// CHECK-NEXT:              function-declarator-chunk
// CHECK-NEXT:                parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                  parameter-declaration-list
// CHECK-NEXT:                    parameter-declaration
// CHECK-NEXT:                      type-specifier-list
// CHECK-NEXT:                        integral-type-specifier
// CHECK-NEXT:                          specifier: int
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
