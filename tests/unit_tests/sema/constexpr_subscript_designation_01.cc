// RUN: %cxx -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

constexpr int values[] = {2, 4, 6};
static_assert(1[values] == 4);
constexpr bool sequence() {
  int data[] = {2, 4, 6};
  int index = 0;
  if ((index = 1, data)[index] != 4) return false;
  int* address = &(index = 2, data)[index];
  if (address != data + 2) return false;
  (index = 0, data)[index] = 9;
  if (data[0] != 9) return false;
  1[data] = 7;
  return data[1] == 7;
}
static_assert(sequence());
struct Data {
  inline static constexpr int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};
  struct Value { int values[3]; };
  inline static constexpr Value value{{2, 4, 6}};
};
constexpr bool copies() {
  auto first = Data::value;
  auto second = Data::value;
  first.values[1] = 30;
  return second.values[1] == 4 && Data::value.values[1] == 4 && Data::matrix[1][2] == 6;
}
static_assert(copies());

struct S { int a[3]; };
constexpr int prvalueMember = S{{1, 2, 3}}.a[1];
static_assert(prvalueMember == 2);
static_assert("abc"[1] == 'b');
static_assert(2["abc"] == 'c');
constexpr const char* text = "xyz";
static_assert(text[2] == 'z');
constexpr int table[2][2] = {{1, 2}, {3, 4}};
static_assert(table[1][0] == 3);
static_assert(1[table][1] == 4);
constexpr const int* row = table[1];
static_assert(row[1] == 4);
constexpr int f() { int a[2] = {5, 6}; int* p = a; return p[1] + 1[p]; }
static_assert(f() == 12);
constexpr int g() { S s{{7, 8, 9}}; return s.a[2]; }
static_assert(g() == 9);
constexpr int h(const int (&r)[3]) { return r[1]; }
static_assert(h({4, 5, 6}) == 5);
