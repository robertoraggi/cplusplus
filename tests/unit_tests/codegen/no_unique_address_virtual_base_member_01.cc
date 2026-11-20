// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s

struct Value {
  char value;
};

struct Holder : virtual Value {
  int held;
};

struct Outer {
  int first;
  [[no_unique_address]] Holder holder;
  char last;
};

Outer outer{};

// CHECK-DAG: %Outer = type { i32, [4 x i8], %Holder.member, i8, [2 x i8] }
// CHECK-DAG: %Holder.member = type <{ ptr, i32, %Value }>
// CHECK-DAG: @outer = {{.*}}global %Outer { i32 0, [4 x i8] zeroinitializer, %Holder.member <{ ptr getelementptr inbounds nuw (i8, ptr @_ZTV6Holder, i64 24), i32 0, %Value zeroinitializer }>, i8 0, [2 x i8] zeroinitializer }
