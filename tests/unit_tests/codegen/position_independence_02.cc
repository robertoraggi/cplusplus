// RUN: %cxx -target x86_64-linux -S %s -o - | %filecheck %s

const char* text() { return "hello"; }

// CHECK: leaq .str{{[0-9]*}}(%rip)
