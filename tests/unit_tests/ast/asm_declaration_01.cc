// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

void asm_qualifiers() {
  asm("nop");
  asm inline("nop");
  asm goto("nop");
  asm volatile("nop");
}

void asm_output() {
  int a;
  char* p = nullptr;
  asm("nop" : "=r"(a), "+rm"(*p));
}

void asm_clobbers() {
  asm("nop" : /* output */ : /* input */ : "r1", "r2", "r3");
}
void asm_goto() {
  asm goto("nop" : : : : end);
end:;
}

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: asm_qualifiers
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: asm-declaration
// CHECK-NEXT:                literal: "nop"
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: asm-declaration
// CHECK-NEXT:                literal: "nop"
// CHECK-NEXT:                asm-qualifier-list
// CHECK-NEXT:                  asm-qualifier
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: asm-declaration
// CHECK-NEXT:                literal: "nop"
// CHECK-NEXT:                asm-qualifier-list
// CHECK-NEXT:                  asm-qualifier
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: asm-declaration
// CHECK-NEXT:                literal: "nop"
// CHECK-NEXT:                asm-qualifier-list
// CHECK-NEXT:                  asm-qualifier
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: asm_output
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
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
// CHECK-NEXT:                          identifier: a
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: char
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      ptr-op-list
// CHECK-NEXT:                        pointer-operator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: p
// CHECK-NEXT:                    initializer: implicit-cast-expression [prvalue char*]
// CHECK-NEXT:                      cast-kind: pointer-conversion
// CHECK-NEXT:                      expression: equal-initializer [prvalue decltype(nullptr)]
// CHECK-NEXT:                        expression: nullptr-literal-expression [prvalue decltype(nullptr)]
// CHECK-NEXT:                          literal: nullptr
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: asm-declaration
// CHECK-NEXT:                literal: "nop"
// CHECK-NEXT:                output-operand-list
// CHECK-NEXT:                  asm-operand
// CHECK-NEXT:                    constraint-literal: "=r"
// CHECK-NEXT:                    expression: id-expression [lvalue int]
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: a
// CHECK-NEXT:                  asm-operand
// CHECK-NEXT:                    constraint-literal: "+rm"
// CHECK-NEXT:                    expression: unary-expression [lvalue char]
// CHECK-NEXT:                      op: *
// CHECK-NEXT:                      expression: implicit-cast-expression [prvalue char*]
// CHECK-NEXT:                        cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                        expression: id-expression [lvalue char*]
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: p
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: asm_clobbers
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: asm-declaration
// CHECK-NEXT:                literal: "nop"
// CHECK-NEXT:                clobber-list
// CHECK-NEXT:                  asm-clobber
// CHECK-NEXT:                    literal: "r1"
// CHECK-NEXT:                  asm-clobber
// CHECK-NEXT:                    literal: "r2"
// CHECK-NEXT:                  asm-clobber
// CHECK-NEXT:                    literal: "r3"
// CHECK-NEXT:    function-definition
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        void-type-specifier
// CHECK-NEXT:      declarator: declarator
// CHECK-NEXT:        core-declarator: id-declarator
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: asm_goto
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: asm-declaration
// CHECK-NEXT:                literal: "nop"
// CHECK-NEXT:                asm-qualifier-list
// CHECK-NEXT:                  asm-qualifier
// CHECK-NEXT:                goto-label-list
// CHECK-NEXT:                  asm-goto-label
// CHECK-NEXT:                    identifier: end
// CHECK-NEXT:            labeled-statement
// CHECK-NEXT:              identifier: end
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: empty-declaration
