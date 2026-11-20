// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s --check-prefix=C1
// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s --check-prefix=C2
// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s --check-prefix=D1

class Shared {
  Shared();
  ~Shared();
  friend struct Concrete;

 public:
  Shared(int);
};

struct Abstract : virtual Shared {
  Abstract();
  ~Abstract();
  virtual void run() = 0;
};

Abstract::Abstract() {}
Abstract::~Abstract() {}

// C1-LABEL: define {{.*}}void @_ZN8AbstractC1Ev(
// C1-NOT: @_ZN6Shared{{C|D}}
// C1: ret void

// C2-LABEL: define {{.*}}void @_ZN8AbstractC2Ev(
// C2-NOT: @_ZN6Shared{{C|D}}
// C2: ret void

// D1-LABEL: define {{.*}}void @_ZN8AbstractD1Ev(
// D1-NOT: @_ZN6Shared{{C|D}}
// D1: ret void
