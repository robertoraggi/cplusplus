// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s

struct Value {
  int value;
};

struct Shared : virtual Value {
  int shared;
};

struct Left : Shared {};

struct Right : virtual Shared {};

struct Bottom : Left, Right {
  Bottom();
};

Bottom::Bottom() {}

// CHECK-LABEL: define {{.*}}void @_ZN6BottomC2Ev(
// CHECK: [[ENTRY:%[0-9]+]] = getelementptr inbounds ptr, ptr %1, i32 6
// CHECK: [[TABLE:%[0-9]+]] = load ptr, ptr [[ENTRY]]
// CHECK: [[OFFSET_SLOT:%[0-9]+]] = getelementptr inbounds i8, ptr %{{[0-9]+}}, i64 -32
// CHECK: [[OFFSET:%[0-9]+]] = load i64, ptr [[OFFSET_SLOT]]
// CHECK: [[SHARED:%[0-9]+]] = getelementptr inbounds i8, ptr %{{[0-9]+}}, i64 [[OFFSET]]
// CHECK: store ptr [[TABLE]], ptr [[SHARED]]
// CHECK: ret void
