// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s

struct Base {
  virtual ~Base();
};

struct Node : virtual Base {
  Node(int depth);
  Node(int depth, int width) : Node(depth + width) {}
  int depth;
};

Node::Node(int value) : depth(value) {}

struct Value {
  int value;
};

struct Holder : virtual Value {
  Holder();
  ~Holder();
};

Holder::Holder() {}
Holder::~Holder() {}

template <typename T>
struct Box : virtual Base {
  Box(T value);
  T value;
};

template <typename T>
Box<T>::Box(T v) : value(v) {}

template struct Box<int>;

// CHECK-DAG: define {{.*}}void @_ZN4NodeC1Ei(
// CHECK-DAG: define {{.*}}void @_ZN4NodeC2Ei(
// CHECK-DAG: define {{.*}}void @_ZN6HolderC1Ev(
// CHECK-DAG: define {{.*}}void @_ZN6HolderC2Ev(
// CHECK-DAG: define {{.*}}void @_ZN6HolderD1Ev(
// CHECK-DAG: define {{.*}}void @_ZN6HolderD2Ev(
// CHECK-DAG: define weak_odr {{.*}}void @_ZN3BoxIiEC1Ei(
// CHECK-DAG: define weak_odr {{.*}}void @_ZN3BoxIiEC2Ei(
// CHECK-NOT: define {{.*}}@_ZN4NodeC1Eii(
// CHECK-NOT: define {{.*}}@_ZN4NodeC2Eii(
