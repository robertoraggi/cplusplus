// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

using repr_type = unsigned char[16];

struct Data {
  operator repr_type&() noexcept { return bytes; }
  operator repr_type const&() const noexcept { return bytes; }
  unsigned char bytes[16];
};

using ints_type = int[4];

struct Single {
  operator const ints_type&() const;
};

unsigned char* mutable_object(Data& d) { return d + 4; }
const unsigned char* const_object(const Data& d) { return d + 4; }
unsigned char* integer_first(Data& d) { return 4 + d; }
const int* single_conversion(const Single& s) { return s + 1; }
