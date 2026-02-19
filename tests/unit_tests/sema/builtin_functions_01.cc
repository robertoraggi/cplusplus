// clang-format off
// RUN: %cxx -verify -fcheck %s

// expected-no-diagnostics

double d = __builtin_inf();
float f = __builtin_inff();
long double ld = __builtin_infl();

double fabs_result = __builtin_fabs(-1.0);
float fabsf_result = __builtin_fabsf(-1.0f);

bool fin = __builtin_isfinite(1.0);
bool inf = __builtin_isinf(1.0);
bool nan_check = __builtin_isnan(1.0);
bool norm = __builtin_isnormal(1.0);

int ctz_val = __builtin_ctz(8u);
int ctzl_val = __builtin_ctzl(8ul);
int ctzll_val = __builtin_ctzll(8ull);

[[noreturn]] void foo() {
  __builtin_unreachable();
}
