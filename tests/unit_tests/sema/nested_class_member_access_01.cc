// RUN: %cxx -fsyntax-only -verify %s

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
