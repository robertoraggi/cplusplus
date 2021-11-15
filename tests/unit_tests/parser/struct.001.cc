// RUN: %cxx -verify -dump-symbols %s -o - | %filecheck %s

struct vec2i {
  // CHECK: struct: vec2i{{$}}

  int x;
  // CHECK-NEXT: field: int x{{$}}

  int y;
  // CHECK-NEXT: field: int y{{$}}

  vec2i& set(int xx, int yy);
  // CHECK-NEXT: function: struct vec2i &set(int, int){{$}}

  vec2i& copy(vec2i& other) {
    x = other.x;
    y = other.y;
    return *this;
  }
  // CHECK-NEXT: function: struct vec2i &copy(struct vec2i &){{$}}
};

struct vec3i;
// CHECK-NEXT: struct: vec3i{{$}}

class Renderer {};
// CHECK-NEXT: class: Renderer{{$}}

union Value {
  char c;
  int i;
  long l;
  void* p;

  char get_c() { return c; }
  int get_i() { return i; }
  long get_l() { return l; }
  void* get_p() { return p; }
};

// CHECK-NEXT: union: Value{{$}}
// CHECK-NEXT: field: char c{{$}}
// CHECK-NEXT: field: int i{{$}}
// CHECK-NEXT: field: long l{{$}}
// CHECK-NEXT: field: void *p{{$}}

// CHECK-NEXT: function: char get_c(){{$}}
// CHECK-NEXT: function: int get_i(){{$}}
// CHECK-NEXT: function: long get_l(){{$}}
// CHECK-NEXT: function: void *get_p(){{$}}
