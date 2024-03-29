// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

void func(auto... args);

void other_func(auto&&...);

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: func
// CHECK-NEXT:            declarator-chunk-list
// CHECK-NEXT:              function-declarator-chunk
// CHECK-NEXT:                parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                  parameter-declaration-list
// CHECK-NEXT:                    parameter-declaration
// CHECK-NEXT:                      identifier: args
// CHECK-NEXT:                      is-pack: true
// CHECK-NEXT:                      type-specifier-list
// CHECK-NEXT:                        auto-type-specifier
// CHECK-NEXT:                      declarator: declarator
// CHECK-NEXT:                        core-declarator: parameter-pack
// CHECK-NEXT:                          core-declarator: id-declarator
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: args
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: other_func
// CHECK-NEXT:            declarator-chunk-list
// CHECK-NEXT:              function-declarator-chunk
// CHECK-NEXT:                parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:                  parameter-declaration-list
// CHECK-NEXT:                    parameter-declaration
// CHECK-NEXT:                      is-pack: true
// CHECK-NEXT:                      type-specifier-list
// CHECK-NEXT:                        auto-type-specifier
// CHECK-NEXT:                      declarator: declarator
// CHECK-NEXT:                        ptr-op-list
// CHECK-NEXT:                          reference-operator
// CHECK-NEXT:                            ref-op: &&
// CHECK-NEXT:                        core-declarator: parameter-pack
