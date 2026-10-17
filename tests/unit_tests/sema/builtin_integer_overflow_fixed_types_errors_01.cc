// RUN: %cxx -verify -fsyntax-only %s

void signedness_mismatch(unsigned* u, int* i) {
  // expected-error@+1 {{invalid argument of type 'unsigned int*' for parameter of type 'int*'}}
  (void)__builtin_sadd_overflow(1, 2, u);
  // expected-error@+1 {{invalid argument of type 'int*' for parameter of type 'unsigned long long*'}}
  (void)__builtin_uaddll_overflow(1ull, 2ull, i);
}

void arity(int* i) {
  // expected-error@+1 {{too few arguments for function of type 'bool (int, int, int*) noexcept'}}
  (void)__builtin_sadd_overflow(1, 2);
  // expected-error@+1 {{too few arguments for function of type 'unsigned int (unsigned int, unsigned int, unsigned int, unsigned int*) noexcept'}}
  (void)__builtin_addc(1u, 2u, 3u);
}

void constness(const long* c) {
  // expected-error@+1 {{invalid argument of type 'const long*' for parameter of type 'long*'}}
  (void)__builtin_ssubl_overflow(1L, 2L, c);
}

void carry_out_type(unsigned char* byte) {
  // expected-error@+1 {{invalid argument of type 'unsigned char*' for parameter of type 'unsigned int*'}}
  (void)__builtin_addc(1u, 2u, 3u, byte);
}
