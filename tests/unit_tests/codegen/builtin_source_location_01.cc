// RUN: %cxx -emit-ir %s -o - | %filecheck %s

// __builtin_LINE() folds to an integer constant.
int get_line() { return __builtin_LINE(); }

// __builtin_FILE() folds to a string constant.
const char* get_file() { return __builtin_FILE(); }

// __builtin_FUNCTION() folds to a string constant.
const char* get_function() { return __builtin_FUNCTION(); }

// Functions are emitted in reverse order in the IR, so check them accordingly.
// CHECK: cxx.func @_Z12get_functionv
// CHECK: cxx.address_of

// CHECK: cxx.func @_Z8get_filev
// CHECK: cxx.address_of

// CHECK: cxx.func @_Z8get_linev
// CHECK: arith.constant {{[0-9]+}} : i32

// CHECK-NOT: cxx.builtin_call @"__builtin_LINE"
// CHECK-NOT: cxx.builtin_call @"__builtin_FILE"
// CHECK-NOT: cxx.builtin_call @"__builtin_FUNCTION"
