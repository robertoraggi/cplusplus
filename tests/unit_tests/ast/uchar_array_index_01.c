// RUN: %cxx -verify -fcheck -ast-dump %s | %filecheck %s --match-full-lines

int main() {
  unsigned char c = 200;
  int arr[256];
  int i;
  for (i = 0; i < 256; i++) arr[i] = i;
  return arr[c] == 200 ? 0 : 1;
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
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  sign-type-specifier
// CHECK-NEXT:                    specifier: unsigned
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: char
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: c
// CHECK-NEXT:                    initializer: implicit-cast-expression [prvalue unsigned char]
// CHECK-NEXT:                      cast-kind: integral-conversion
// CHECK-NEXT:                      expression: equal-initializer [prvalue int]
// CHECK-NEXT:                        expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                          literal: 200
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: int
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: arr
// CHECK-NEXT:                      declarator-chunk-list
// CHECK-NEXT:                        array-declarator-chunk
// CHECK-NEXT:                          expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                            literal: 256
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: int
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: i
// CHECK-NEXT:            for-statement
// CHECK-NEXT:              initializer: expression-statement
// CHECK-NEXT:                expression: assignment-expression [prvalue int]
// CHECK-NEXT:                  op: =
// CHECK-NEXT:                  left-expression: id-expression [lvalue int]
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: i
// CHECK-NEXT:                  right-expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                    literal: 0
// CHECK-NEXT:              condition: binary-expression [prvalue bool]
// CHECK-NEXT:                op: <
// CHECK-NEXT:                left-expression: implicit-cast-expression [prvalue int]
// CHECK-NEXT:                  cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                  expression: id-expression [lvalue int]
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: i
// CHECK-NEXT:                right-expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                  literal: 256
// CHECK-NEXT:              expression: post-incr-expression [prvalue int]
// CHECK-NEXT:                op: ++
// CHECK-NEXT:                base-expression: id-expression [lvalue int]
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: i
// CHECK-NEXT:              statement: expression-statement
// CHECK-NEXT:                expression: assignment-expression [prvalue int]
// CHECK-NEXT:                  op: =
// CHECK-NEXT:                  left-expression: subscript-expression [lvalue int]
// CHECK-NEXT:                    base-expression: id-expression [lvalue int [256]]
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: arr
// CHECK-NEXT:                    index-expression: implicit-cast-expression [prvalue int]
// CHECK-NEXT:                      cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                      expression: id-expression [lvalue int]
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: i
// CHECK-NEXT:                  right-expression: implicit-cast-expression [prvalue int]
// CHECK-NEXT:                    cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                    expression: id-expression [lvalue int]
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: i
// CHECK-NEXT:            return-statement
// CHECK-NEXT:              expression: conditional-expression [prvalue int]
// CHECK-NEXT:                condition: binary-expression [prvalue bool]
// CHECK-NEXT:                  op: ==
// CHECK-NEXT:                  left-expression: implicit-cast-expression [prvalue int]
// CHECK-NEXT:                    cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                    expression: subscript-expression [lvalue int]
// CHECK-NEXT:                      base-expression: id-expression [lvalue int [256]]
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: arr
// CHECK-NEXT:                      index-expression: implicit-cast-expression [prvalue int]
// CHECK-NEXT:                        cast-kind: integral-promotion
// CHECK-NEXT:                        expression: implicit-cast-expression [prvalue unsigned char]
// CHECK-NEXT:                          cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                          expression: id-expression [lvalue unsigned char]
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: c
// CHECK-NEXT:                  right-expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                    literal: 200
// CHECK-NEXT:                iftrue-expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                  literal: 0
// CHECK-NEXT:                iffalse-expression: int-literal-expression [prvalue int]
// CHECK-NEXT:                  literal: 1
