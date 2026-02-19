// RUN: %cxx -verify -fcheck %s

template <long N, long D>
struct ratio {
  static constexpr long num = N;
  static constexpr long den = D;
};

template <long A, long B>
struct gcd {
  static constexpr long value = 1;
};

template <class R1, class R2>
struct ratio_divide {
  static const long gcd_n1_n2 = gcd<R1::num, R2::num>::value;
  static const long gcd_d1_d2 = gcd<R1::den, R2::den>::value;
};

using out = ratio_divide<ratio<1, 2>, ratio<3, 4>>;
out* ptr = nullptr;
