// RUN: %cxx -fcheck %s
// expected-no-diagnostics

template <int i>
void bit_signed() {
  _BitInt(i) x;
}

template <unsigned i>
void bit_unsigned() {
  unsigned _BitInt(i) x;
}

void use() {
  bit_signed<4>();
  bit_signed<128>();
  bit_unsigned<1>();
  bit_unsigned<64>();
}
