// RUN: %cxx -target x86_64-linux -emit-llvm %s -o - | %filecheck %s --check-prefix=DEFAULT
// RUN: %cxx -target x86_64-linux -fPIC -emit-llvm %s -o - | %filecheck %s --check-prefix=PIC
// RUN: %cxx -target x86_64-linux -fpie -emit-llvm %s -o - | %filecheck %s --check-prefix=PIE
// RUN: %cxx -target x86_64-linux -fno-pic -fPIE -emit-llvm %s -o - | %filecheck %s --check-prefix=DEFAULT
// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s --check-prefix=MACOS

int value;

// DEFAULT: !{i32 8, !"PIC Level", i32 2}
// DEFAULT: !{i32 7, !"PIE Level", i32 2}
// PIC: !{i32 8, !"PIC Level", i32 2}
// PIE: !{i32 8, !"PIC Level", i32 1}
// PIE: !{i32 7, !"PIE Level", i32 1}
// MACOS: !{i32 8, !"PIC Level", i32 2}
