// RUN: %cxx -emit-llvm %s -o - | %filecheck %s

constexpr int NPtr = 10;
static void* ptr[NPtr];
static void** pool = ptr;

// CHECK: @pool = internal global ptr @ptr
// CHECK: @ptr = internal global
