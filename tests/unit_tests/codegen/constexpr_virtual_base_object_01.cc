// RUN: %cxx --target=arm64-apple-macosx -emit-llvm %s -o - | %filecheck %s
// RUN: %cxx --target=arm64-apple-macosx -emit-llvm %s -o - | %filecheck %s --check-prefix=VALUE

struct Value {
  char value = 7;
  virtual int get() const { return value; }
};

struct Holder : virtual Value {
  int held = 5;
  int get() const override { return held; }
};

struct Derived : Holder {
  int extra = 9;
};

struct Outer {
  int first;
  Holder holder;
};

struct Wrapper {
  int first;
  [[no_unique_address]] Holder holder;
  char last;
};

Holder holder{};
Derived derived{};
Outer outer{};
Wrapper wrapper{};
Holder plain;

auto valueOf(const Value& object) -> int { return object.get(); }

int main() {
  if (holder.value != 7 || holder.held != 5) return 1;
  if (derived.value != 7 || derived.held != 5 || derived.extra != 9) return 2;
  if (outer.holder.value != 7 || outer.holder.held != 5) return 3;
  if (wrapper.holder.value != 7 || wrapper.last != 0) return 4;
  if (valueOf(holder) != 5 || valueOf(derived) != 5) return 5;
  if (valueOf(outer.holder) != 5 || valueOf(wrapper.holder) != 5) return 6;
  if (plain.value != 7 || valueOf(plain) != 5) return 7;
  return 0;
}

// CHECK-NOT: @llvm.global_ctors

// VALUE-DAG: @holder = global %Holder { ptr getelementptr inbounds nuw (i8, ptr @_ZTV6Holder, i64 24), i32 5, [4 x i8] zeroinitializer, %Value.base <{ ptr getelementptr inbounds nuw (i8, ptr @_ZTV6Holder, i64 56), i8 7 }>, [7 x i8] zeroinitializer }
// VALUE-DAG: @derived = global %Derived { %Holder.base <{ ptr getelementptr inbounds nuw (i8, ptr @_ZTV7Derived, i64 24), i32 5 }>, i32 9, %Value.base <{ ptr getelementptr inbounds nuw (i8, ptr @_ZTV7Derived, i64 56), i8 7 }>, [7 x i8] zeroinitializer }
// VALUE-DAG: @plain = global %Holder { ptr getelementptr inbounds nuw (i8, ptr @_ZTV6Holder, i64 24), i32 5, [4 x i8] zeroinitializer, %Value.base <{ ptr getelementptr inbounds nuw (i8, ptr @_ZTV6Holder, i64 56), i8 7 }>, [7 x i8] zeroinitializer }
