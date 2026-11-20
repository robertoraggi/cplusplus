// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s

struct Pool {
  Pool(int capacity, int alignment);
  Pool() : Pool(64, 8) {}
  virtual ~Pool();
};

struct Base {
  virtual ~Base();
};

struct Derived : virtual Base {
  Derived(int value);
  Derived() : Derived(0) {}
  ~Derived() override;
};

void use(Pool&, Derived&);

void run() {
  Pool pool;
  Derived derived;
  use(pool, derived);
}

// CHECK-LABEL: define linkonce_odr {{.*}}void @_ZN7DerivedC1Ev(
// CHECK-NOT: @_ZTV7Derived
// CHECK-NOT: @_ZTT7Derived
// CHECK: call void @_ZN7DerivedC1Ei(
// CHECK-NOT: @_ZTV7Derived
// CHECK-NOT: @_ZTT7Derived
// CHECK: ret void

// CHECK-LABEL: define linkonce_odr {{.*}}void @_ZN4PoolC2Ev(
// CHECK-NOT: @_ZTV4Pool
// CHECK: call void @_ZN4PoolC2Eii(
// CHECK-NOT: @_ZTV4Pool
// CHECK: ret void
