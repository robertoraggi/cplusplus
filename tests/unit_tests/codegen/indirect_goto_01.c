// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s

// CHECK: @dispatch_test
// CHECK: blockaddress(@dispatch_test
// CHECK: blockaddress(@dispatch_test
// CHECK: blockaddress(@dispatch_test
// CHECK: indirectbr ptr{{.*}}[label %{{.*}}, label %{{.*}}, label %
// CHECK: @conditional_test
// CHECK: blockaddress(@conditional_test
// CHECK: blockaddress(@conditional_test
// CHECK: indirectbr ptr{{.*}}[label %{{.*}}, label %

int conditional_test(int flag) {
  void *ptr = flag ? &&yes : &&no;
  goto *ptr;
yes:
  return 1;
no:
  return 0;
}

void dispatch_test(void) {
  void *table[3];
  table[0] = &&step1;
  table[1] = &&step2;
  table[2] = &&done;
  int i = 0;
  goto *table[i];
step1:
  i = 1;
  goto *table[i];
step2:
  i = 2;
  goto *table[i];
done:;
}
