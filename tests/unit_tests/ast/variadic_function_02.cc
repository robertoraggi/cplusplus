// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

void ff(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  for (int i = 0; i < count; ++i) {
    (void)__builtin_va_arg(args, int);
  }
  __builtin_va_end(args);
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
// CHECK-NEXT:            identifier: ff
// CHECK-NEXT:        declarator-chunk-list
// CHECK-NEXT:          function-declarator-chunk
// CHECK-NEXT:            parameter-declaration-clause: parameter-declaration-clause
// CHECK-NEXT:              is-variadic: true
// CHECK-NEXT:              parameter-declaration-list
// CHECK-NEXT:                parameter-declaration
// CHECK-NEXT:                  identifier: count
// CHECK-NEXT:                  type-specifier-list
// CHECK-NEXT:                    integral-type-specifier
// CHECK-NEXT:                      specifier: int
// CHECK-NEXT:                  declarator: declarator
// CHECK-NEXT:                    core-declarator: id-declarator
// CHECK-NEXT:                      unqualified-id: name-id
// CHECK-NEXT:                        identifier: count
// CHECK-NEXT:      function-body: compound-statement-function-body
// CHECK-NEXT:        statement: compound-statement
// CHECK-NEXT:          statement-list
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  va-list-type-specifier
// CHECK-NEXT:                    specifier: __builtin_va_list
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        unqualified-id: name-id
// CHECK-NEXT:                          identifier: args
// CHECK-NEXT:            expression-statement
// CHECK-NEXT:              expression: call-expression
// CHECK-NEXT:                base-expression: id-expression
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: __builtin_va_start
// CHECK-NEXT:                expression-list
// CHECK-NEXT:                  id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: args
// CHECK-NEXT:                  id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: count
// CHECK-NEXT:            for-statement
// CHECK-NEXT:              initializer: declaration-statement
// CHECK-NEXT:                declaration: simple-declaration
// CHECK-NEXT:                  decl-specifier-list
// CHECK-NEXT:                    integral-type-specifier
// CHECK-NEXT:                      specifier: int
// CHECK-NEXT:                  init-declarator-list
// CHECK-NEXT:                    init-declarator
// CHECK-NEXT:                      declarator: declarator
// CHECK-NEXT:                        core-declarator: id-declarator
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: i
// CHECK-NEXT:                      initializer: equal-initializer
// CHECK-NEXT:                        expression: int-literal-expression
// CHECK-NEXT:                          literal: 0
// CHECK-NEXT:              condition: binary-expression
// CHECK-NEXT:                op: <
// CHECK-NEXT:                left-expression: implicit-cast-expression
// CHECK-NEXT:                  cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                  expression: id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: i
// CHECK-NEXT:                right-expression: implicit-cast-expression
// CHECK-NEXT:                  cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                  expression: id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: count
// CHECK-NEXT:              expression: unary-expression
// CHECK-NEXT:                op: ++
// CHECK-NEXT:                expression: id-expression
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: i
// CHECK-NEXT:              statement: compound-statement
// CHECK-NEXT:                statement-list
// CHECK-NEXT:                  expression-statement
// CHECK-NEXT:                    expression: cast-expression
// CHECK-NEXT:                      type-id: type-id
// CHECK-NEXT:                        type-specifier-list
// CHECK-NEXT:                          void-type-specifier
// CHECK-NEXT:                      expression: va-arg-expression
// CHECK-NEXT:                        expression: id-expression
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: args
// CHECK-NEXT:                        type-id: type-id
// CHECK-NEXT:                          type-specifier-list
// CHECK-NEXT:                            integral-type-specifier
// CHECK-NEXT:                              specifier: int
// CHECK-NEXT:            declaration-statement
// CHECK-NEXT:              declaration: simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  named-type-specifier
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: __builtin_va_end
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: nested-declarator
// CHECK-NEXT:                        declarator: declarator
// CHECK-NEXT:                          core-declarator: id-declarator
// CHECK-NEXT:                            unqualified-id: name-id
// CHECK-NEXT:                              identifier: args
