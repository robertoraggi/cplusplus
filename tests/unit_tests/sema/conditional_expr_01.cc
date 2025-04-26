// RUN: %cxx -verify -fcheck %s

struct Base {};
struct Derived : Base {};

void returns_void();
int returns_int();
int& returns_int_ref();

void test_conditional(bool x) {
  // clang-format off

  // expected-error@1 {{left operand to ? is 'void', but right operand is of type 'int'}}
  x ? returns_void() : returns_int();

  // expected-error@1 {{left operand to ? is 'int', but right operand is of type 'void'}}
  x ? returns_int() : returns_void();

  // clang-format on

  static_assert(__is_same(void, decltype(x ? throw 0 : throw 0)));

  static_assert(__is_same(int, decltype(x ? throw 0 : returns_int())));
  static_assert(__is_same(int, decltype(x ? returns_int() : throw 0)));

  static_assert(__is_same(int&, decltype(x ? returns_int_ref() : throw 0)));
  static_assert(__is_same(int&, decltype(x ? throw 0 : returns_int_ref())));

  static_assert(__is_same(int, decltype(false ? 1 : 0)));

  void* void_ptr;
  int* int_ptr;
  const int* const_int_ptr;

  static_assert(__is_same(decltype(nullptr), decltype(x ? nullptr : nullptr)));
  static_assert(__is_same(void*&, decltype(x ? void_ptr : throw 0)));
  static_assert(__is_same(int*, decltype(x ? int_ptr : nullptr)));
  static_assert(__is_same(int*, decltype(x ? nullptr : int_ptr)));

  static_assert(__is_same(void*, decltype(x ? void_ptr : int_ptr)));
  static_assert(__is_same(void*, decltype(x ? int_ptr : void_ptr)));

  static_assert(__is_same(const void*, decltype(x ? void_ptr : const_int_ptr)));
  static_assert(__is_same(const void*, decltype(x ? const_int_ptr : void_ptr)));

  static_assert(__is_same(int, decltype(x ? 0 : '0')));
  static_assert(__is_same(unsigned, decltype(x ? 0 : 1u)));
  static_assert(__is_same(long, decltype(x ? 0 : 1l)));

  const char cs[] = "";
  char s[10];
  char* p;
  const char* cp = p;
  const char* const cpc = p;

  static_assert(__is_same(char*, decltype(x ? s : p)));

  static_assert(__is_same(const char*, decltype(true ? p : cp)));

  static_assert(__is_same(const char*, decltype(true ? p : "")));
  static_assert(__is_same(const char*, decltype(true ? "" : cp)));

  static_assert(__is_same(const char*, decltype(x ? cs : cp)));
  static_assert(__is_same(const char*, decltype(x ? cs : p)));
  static_assert(__is_same(const char*, decltype(x ? cs : cpc)));

  Base* b;
  Derived* d;

  static_assert(__is_same(Base*, decltype(true ? b : d)));

  char (*a10)[10];
  char (*a100)[100];
  char (*a)[];
  const char (*ca)[];

  // clang-format off
  // expected-error@1 {{left operand to ? is 'char (*)[10]', but right operand is of type 'char (*)[100]'}}
  x ? a10 : a100;
  // clang-format on

  static_assert(__is_same(char (*&)[10], decltype(x ? a10 : a10)));
  static_assert(__is_same(const char (*)[], decltype(x ? a10 : a)));
  static_assert(__is_same(char (*&)[], decltype(x ? a : a)));
  static_assert(__is_same(const char (*)[], decltype(x ? a : ca)));
}
