// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

// Brace form: extern "C" { ... }
extern "C" {
char* strcpy(char* dst, const char* src);
int strcmp(const char* a, const char* b);
}

// Single-declaration form: extern "C" void f();
extern "C" void abort();

// extern "C" with namespace (the minicc/libc++ pattern)
extern "C" {
namespace std {
void* malloc(unsigned long size);
void free(void* ptr);
int printf(const char* fmt, ...);
}  // namespace std
}  // extern "C"

// C++ function (no extern "C")
int cpp_func(int x);

// Member functions inside extern "C" block should still be C++ linkage
extern "C" {
struct Widget {
  void draw();
};
}

// Redeclare a C-linkage function outside extern "C" —
// the redeclaration should inherit C linkage from the canonical.
void abort();

void test() {
  char buf[32];
  strcpy(buf, "hello");
  int r = strcmp(buf, "hello");
  void* p = std::malloc(100);
  std::free(p);
  abort();
}

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  function extern "C" char* strcpy(char*, const char*)
// CHECK-NEXT:  function extern "C" int strcmp(const char*, const char*)
// CHECK-NEXT:  function extern "C" void abort()
// CHECK-NEXT:    [redeclarations]
// CHECK-NEXT:      function extern "C" void abort()
// CHECK-NEXT:  namespace std
// CHECK-NEXT:    function extern "C" void* malloc(unsigned long)
// CHECK-NEXT:    function extern "C" void free(void*)
// CHECK-NEXT:    function extern "C" int printf(const char*...)
// CHECK-NEXT:  function int cpp_func(int)
// CHECK-NEXT:  class Widget
// CHECK-NEXT:    constructor defaulted void Widget()
// CHECK-NEXT:    constructor defaulted void Widget(const ::Widget&)
// CHECK-NEXT:    constructor defaulted void Widget(::Widget&&)
// CHECK-NEXT:    function void draw()
// CHECK-NEXT:    function defaulted void ~Widget()
// CHECK-NEXT:  function void test()
