// RUN: %cxx -toolchain macos -emit-ir %s -o - | %filecheck %s

int vla_sum(int n) {
  int arr[n];
  for (int i = 0; i < n; i++) arr[i] = i;
  int s = 0;
  for (int i = 0; i < n; i++) s += arr[i];
  return s;
}

// CHECK: cxx.dyn_alloca
// CHECK: cxx.ptradd
// CHECK-NOT: cxx.subscript
