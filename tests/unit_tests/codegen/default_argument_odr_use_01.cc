// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s
// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s --check-prefix=ABSENT

namespace std {
class type_info;
}

struct Polymorphic {
  virtual ~Polymorphic() {}
};

template <class T>
int evaluated() {
  return 1;
}

template <class T>
int unevaluated() {
  return 2;
}

template <class T>
Polymorphic& object() {
  static Polymorphic instance;
  return instance;
}

template <class T>
int call(int a = evaluated<T>(), int b = sizeof(unevaluated<T>()),
         const std::type_info& info = typeid(object<T>())) {
  return a + b;
}

template <class T>
struct Aggregate {
  int a = evaluated<T>();
  int b = sizeof(unevaluated<T>());
};

int use() {
  Aggregate<char> aggregate{};
  return call<int>() + aggregate.a + aggregate.b;
}

// CHECK-DAG: define {{.*}}@_Z9evaluatedIiEiv(
// CHECK-DAG: define {{.*}}@_Z9evaluatedIcEiv(
// CHECK-DAG: define {{.*}}@_Z6objectIiER11Polymorphicv(

// ABSENT: define {{.*}}@_Z3usev(
// ABSENT-NOT: unevaluated
