// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s

struct Value {
  int value;
};

struct Empty {};

struct Dynamic : virtual Value {};

struct PrimaryBesideEmptyBase : Dynamic, Empty, virtual Value {
  int data[2];
};

struct EmptyDerived : Empty {};

struct EmptyMemberPastData : Empty, virtual EmptyDerived {
  [[no_unique_address]] EmptyDerived member;
};

struct DerivesEmptyMemberPastData : EmptyMemberPastData {};

struct Pair {
  int first, second;
};

struct EmptyMemberExtendsNonVirtualSize : Empty, virtual Pair, Value {
  [[no_unique_address]] Empty member;
};

struct AfterNonVirtualSize : EmptyMemberExtendsNonVirtualSize, Empty {
  char after;
};

struct NearlyEmpty : virtual Value {};

struct IndirectPrimary : virtual NearlyEmpty, Empty {};

struct SharesIndirectPrimary : IndirectPrimary, virtual NearlyEmpty {
  int data;
};

PrimaryBesideEmptyBase primaryBesideEmptyBase;
DerivesEmptyMemberPastData derivesEmptyMemberPastData;
AfterNonVirtualSize afterNonVirtualSize;
SharesIndirectPrimary sharesIndirectPrimary;

// CHECK-DAG: %PrimaryBesideEmptyBase = type { %Dynamic.base, [0 x i8], [2 x i32], %Value, [4 x i8] }
// CHECK-DAG: %Dynamic.base = type { ptr }
// CHECK-DAG: %DerivesEmptyMemberPastData = type { %EmptyMemberPastData.base, [0 x i8], [7 x i8] }
// CHECK-DAG: %EmptyMemberPastData.base = type <{ ptr, [0 x i8], [0 x i8], [0 x i8], i8 }>
// CHECK-DAG: %AfterNonVirtualSize = type { %EmptyMemberExtendsNonVirtualSize.base, [0 x i8], i8, [2 x i8], %Pair }
// CHECK-DAG: %EmptyMemberExtendsNonVirtualSize.base = type <{ ptr, [0 x i8], %Value, [0 x i8], i8 }>
// CHECK-DAG: %SharesIndirectPrimary = type { %IndirectPrimary.base, i32, %Value }
