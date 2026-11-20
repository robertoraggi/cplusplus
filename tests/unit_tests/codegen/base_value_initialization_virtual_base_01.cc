// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s

struct Value {
  int value;
  Value();
};

struct Holder : virtual Value {
  int held;
};

struct Derived : Holder {
  Derived();
};

Derived::Derived() : Holder() {}

// CHECK-LABEL: define {{.*}}void @_ZN7DerivedC2Ev(
// CHECK: call void @llvm.memset.p0.i64(ptr %{{[0-9]+}}, i8 0, i64 12, i1 false)
// CHECK: ret void
