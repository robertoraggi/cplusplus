// RUN: %cxx -fcheck -c %s 2>&1 | %filecheck %s --allow-empty
// CHECK-NOT: error

struct ConstValue {
  void zero();
};

class Outer {
 public:
  struct Inner {
    ConstValue result;
    bool done;
    Inner* parent;

    Inner(Inner* p) {
      parent = p;
      done = false;
      result.zero();
    }
  };
};
