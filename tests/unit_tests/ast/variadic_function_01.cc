// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

void print_like(const char* fmt, ...);

// clang-format off
//      CHECK:translation-unit
// CHEC-NEXT:  declaration-list
// CHEC-NEXT:    simple-declaration
// CHEC-NEXT:      decl-specifier-list
// CHEC-NEXT:        void-type-specifier
// CHEC-NEXT:      init-declarator-list
// CHEC-NEXT:        init-declarator
// CHEC-NEXT:          declarator: declarator
// CHEC-NEXT:            core-declarator: id-declarator
// CHEC-NEXT:              name: simple-name
// CHEC-NEXT:                identifier: print_like
// CHEC-NEXT:            modifiers
// CHEC-NEXT:              function-declarator
// CHEC-NEXT:                parameters-and-qualifiers: parameters-and-qualifiers
// CHEC-NEXT:                  parameter-declaration-clause: parameter-declaration-clause
// CHEC-NEXT:                    is-variadic: true
// CHEC-NEXT:                    parameter-declaration-list
// CHEC-NEXT:                      parameter-declaration
// CHEC-NEXT:                        type-specifier-list
// CHEC-NEXT:                          const-qualifier
// CHEC-NEXT:                          integral-type-specifier
// CHEC-NEXT:                            specifier: char
// CHEC-NEXT:                        declarator: declarator
// CHEC-NEXT:                          ptr-op-list
// CHEC-NEXT:                            pointer-operator
// CHEC-NEXT:                          core-declarator: id-declarator
// CHEC-NEXT:                            name: simple-name
// CHEC-NEXT:                              identifier: fmt
// CHEC-NEXT: