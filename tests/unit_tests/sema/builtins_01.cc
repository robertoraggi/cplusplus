// RUN: %cxx -verify -fcheck %s

// Test builtin math functions
void test_math_builtins() {
  float f = __builtin_sinf(1.0f);
  double d = __builtin_cos(2.0);
  float s = __builtin_sqrtf(4.0f);
  double p = __builtin_pow(2.0, 3.0);
  float a = __builtin_fabsf(-1.0f);
  double c = __builtin_ceil(1.5);
  float fl = __builtin_floorf(1.9f);
  double r = __builtin_round(2.5);
  double l = __builtin_log(1.0);
  double e = __builtin_exp(1.0);
}

// Test builtin string/memory functions
void test_string_builtins() {
  const char* s1 = "hello";
  const char* s2 = "world";
  char buf[64];

  int cmp = __builtin_strcmp(s1, s2);
  __SIZE_TYPE__ len = __builtin_strlen(s1);
  void* p = __builtin_memcpy(buf, s1, 5);
  void* q = __builtin_memset(buf, 0, 64);
  int mcmp = __builtin_memcmp(s1, s2, 5);
  void* m = __builtin_memmove(buf, s1, 5);
  char* chr = __builtin_strchr(s1, 'l');
  char* cat = __builtin_strcat(buf, s2);
  char* cpy = __builtin_strcpy(buf, s1);
  char* ncpy = __builtin_strncpy(buf, s1, 3);
  int ncmp = __builtin_strncmp(s1, s2, 3);
  char* ncat = __builtin_strncat(buf, s2, 3);
  char* str = __builtin_strstr(s1, "ll");
  char* pbr = __builtin_strpbrk(s1, "aeiou");
  char* rch = __builtin_strrchr(s1, 'l');
}

// Test __has_builtin
#if !__has_builtin(__builtin_strcmp)
#error "__builtin_strcmp should be available"
#endif

#if !__has_builtin(__builtin_strlen)
#error "__builtin_strlen should be available"
#endif

#if !__has_builtin(__builtin_memcpy)
#error "__builtin_memcpy should be available"
#endif

#if !__has_builtin(__builtin_sin)
#error "__builtin_sin should be available"
#endif

#if !__has_builtin(__builtin_sqrt)
#error "__builtin_sqrt should be available"
#endif
