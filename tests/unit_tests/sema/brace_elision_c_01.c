// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct point {
  int x;
  int y;
};

struct nested {
  struct point p;
  int z;
};

struct deep {
  struct nested n;
  int w;
};

struct with_array {
  int a[3];
  int b;
};
struct arr_of_struct {
  struct point pts[2];
};

union u {
  struct point p;
  int x;
};

struct has_union {
  int tag;
  union u val;
};

// Basic aggregate init
struct point p1 = {1, 2};

// Partial init
struct point p2 = {1};

// Empty braces
struct point p3 = {};

// Nested struct with braces
struct nested n1 = {{1, 2}, 3};

// Brace elision: nested struct without inner braces
struct nested n2 = {1, 2, 3};

// Deep brace elision
struct deep d1 = {1, 2, 3, 4};

// Partial init with brace elision (fewer elements than slots)
struct nested n3 = {1};

// Struct with array member, fully braced
struct with_array wa1 = {{1, 2, 3}, 4};

// Struct with array member, brace elision
struct with_array wa2 = {1, 2, 3, 4};

// Array of structs with braces
struct point arr1[2] = {{1, 2}, {3, 4}};

// Array of structs with brace elision
struct point arr2[2] = {1, 2, 3, 4};

// Array of structs with 3 elements
struct point arr3[3] = {1, 2, 3, 4, 5, 6};

// Struct containing array of structs
struct arr_of_struct as1 = {1, 2, 3, 4};

// Designated initializers
struct point p4 = {.x = 1, .y = 2};
struct point p5 = {.y = 5};
struct point p6 = {0};

// Union init
union u u1 = {42};

// Union brace elision for struct first member
union u u2 = {1, 2};

// Struct containing a union
struct has_union hu1 = {1, {42}};

// String literal for char array member (not brace elision)
struct Token {
  int id;
  char zName[7];
  double start;
  double span;
};

struct Token t1 = {3, "hi", 123.0, 321.0};

int main() {
  struct point lp1 = {1, 2};
  struct nested ln1 = {1, 2, 3};
  struct deep ld1 = {1, 2, 3, 4};
  struct point larr[2] = {1, 2, 3, 4};
  struct nested ln2 = {1};
  return 0;
}
