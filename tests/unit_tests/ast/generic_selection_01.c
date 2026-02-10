// clang-format off
// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines
// clang-format on

int main() {
  (void)_Generic(1, default: "default");
  (void)_Generic(1, default: "default", int: "int", const char*: "const char*");
  (void)_Generic(1, int: "int", const char*: "const char*", default: "default");
  (void)_Generic(1, int: "int", const char*: "const char*");
}

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        integral-type-specifier
// CHECK-NEXT:          specifier: int
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: main
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            expression-statement
// CHECK-NEXT:              expression: cast-expression [prvalue void]
// CHECK-NEXT:                type-id: type-id
// CHECK-NEXT:                  type-specifier-list
// CHECK-NEXT:                    void-type-specifier
// CHECK-NEXT:                expression: generic-selection-expression [lvalue char [8]]
// CHECK-NEXT:                  matched-assoc-index: 0
// CHECK-NEXT:                  expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                    literal: 1
// CHECK-NEXT:                  generic-association-list
// CHECK-NEXT:                    default-generic-association
// CHECK-NEXT:                      expression: string-literal-expression [lvalue char [8]]
// CHECK-NEXT:                        literal: "default"
// CHECK-NEXT:                        encoding: <string_literal>
// CHECK-NEXT:            expression-statement
// CHECK-NEXT:              expression: cast-expression [prvalue void]
// CHECK-NEXT:                type-id: type-id
// CHECK-NEXT:                  type-specifier-list
// CHECK-NEXT:                    void-type-specifier
// CHECK-NEXT:                expression: generic-selection-expression [lvalue char [4]]
// CHECK-NEXT:                  matched-assoc-index: 1
// CHECK-NEXT:                  expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                    literal: 1
// CHECK-NEXT:                  generic-association-list
// CHECK-NEXT:                    default-generic-association
// CHECK-NEXT:                      expression: string-literal-expression [lvalue char [8]]
// CHECK-NEXT:                        literal: "default"
// CHECK-NEXT:                        encoding: <string_literal>
// CHECK-NEXT:                    type-generic-association
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          integral-type-specifier
// CHECK-NEXT:                            specifier: int
// CHECK-NEXT:                      expression: string-literal-expression [lvalue char [4]]
// CHECK-NEXT:                        literal: "int"
// CHECK-NEXT:                        encoding: <string_literal>
// CHECK-NEXT:                    type-generic-association
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          const-qualifier
// CHECK-NEXT:                          integral-type-specifier
// CHECK-NEXT:                            specifier: char
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:                          ptr-op-list
// CHECK-NEXT:                            pointer-operator
// CHECK-NEXT:                      expression: string-literal-expression [lvalue char [12]]
// CHECK-NEXT:                        literal: "const char*"
// CHECK-NEXT:                        encoding: <string_literal>
// CHECK-NEXT:            expression-statement
// CHECK-NEXT:              expression: cast-expression [prvalue void]
// CHECK-NEXT:                type-id: type-id
// CHECK-NEXT:                  type-specifier-list
// CHECK-NEXT:                    void-type-specifier
// CHECK-NEXT:                expression: generic-selection-expression [lvalue char [4]]
// CHECK-NEXT:                  matched-assoc-index: 0
// CHECK-NEXT:                  expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                    literal: 1
// CHECK-NEXT:                  generic-association-list
// CHECK-NEXT:                    type-generic-association
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          integral-type-specifier
// CHECK-NEXT:                            specifier: int
// CHECK-NEXT:                      expression: string-literal-expression [lvalue char [4]]
// CHECK-NEXT:                        literal: "int"
// CHECK-NEXT:                        encoding: <string_literal>
// CHECK-NEXT:                    type-generic-association
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          const-qualifier
// CHECK-NEXT:                          integral-type-specifier
// CHECK-NEXT:                            specifier: char
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:                          ptr-op-list
// CHECK-NEXT:                            pointer-operator
// CHECK-NEXT:                      expression: string-literal-expression [lvalue char [12]]
// CHECK-NEXT:                        literal: "const char*"
// CHECK-NEXT:                        encoding: <string_literal>
// CHECK-NEXT:                    default-generic-association
// CHECK-NEXT:                      expression: string-literal-expression [lvalue char [8]]
// CHECK-NEXT:                        literal: "default"
// CHECK-NEXT:                        encoding: <string_literal>
// CHECK-NEXT:            expression-statement
// CHECK-NEXT:              expression: cast-expression [prvalue void]
// CHECK-NEXT:                type-id: type-id
// CHECK-NEXT:                  type-specifier-list
// CHECK-NEXT:                    void-type-specifier
// CHECK-NEXT:                expression: generic-selection-expression [lvalue char [4]]
// CHECK-NEXT:                  matched-assoc-index: 0
// CHECK-NEXT:                  expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                    literal: 1
// CHECK-NEXT:                  generic-association-list
// CHECK-NEXT:                    type-generic-association
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          integral-type-specifier
// CHECK-NEXT:                            specifier: int
// CHECK-NEXT:                      expression: string-literal-expression [lvalue char [4]]
// CHECK-NEXT:                        literal: "int"
// CHECK-NEXT:                        encoding: <string_literal>
// CHECK-NEXT:                    type-generic-association
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          const-qualifier
// CHECK-NEXT:                          integral-type-specifier
// CHECK-NEXT:                            specifier: char
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:                          ptr-op-list
// CHECK-NEXT:                            pointer-operator
// CHECK-NEXT:                      expression: string-literal-expression [lvalue char [12]]
// CHECK-NEXT:                        literal: "const char*"
// CHECK-NEXT:                        encoding: <string_literal>
