#if !defined(__need_ptrdiff_t) && !defined(__need_size_t) && \
    !defined(__need_wchar_t) && !defined(__need_NULL) &&     \
    !defined(__need_STDDEF_H_misc)
#define __need_ptrdiff_t
#define __need_size_t
#define __need_wchar_t
#define __need_NULL
#define __need_STDDEF_H_misc
#endif

#ifdef __need_ptrdiff_t
#undef __need_ptrdiff_t
typedef long int ptrdiff_t;
#endif /* __need_ptrdiff_t */

#ifdef __need_size_t
#undef __need_size_t
typedef long unsigned int size_t;
#endif /* __need_size_t */

#ifdef __need_wchar_t
#undef __need_wchar_t
#endif /* __need_wchar_t */

#ifdef __need_NULL
#undef __need_NULL
#undef NULL
#define NULL 0
#endif /* __need_NULL */

#ifdef __need_STDDEF_H_misc
#undef __need_STDDEF_H_misc
typedef long unsigned int rsize_t;

typedef struct {
  long long __clang_max_align_nonce1
      __attribute__((__aligned__(__alignof__(long long))));

  long double __clang_max_align_nonce2
      __attribute__((__aligned__(__alignof__(long double))));
} max_align_t;

#endif /* __need_STDDEF_H_misc */

#ifdef __need_wint_t
#undef __need_wint_t

#ifdef __WINT_TYPE__
typedef __WINT_TYPE__ wint_t;
#else
typedef int wint_t;
#endif

#endif /* __need_wint_t */

#if !defined(offsetof)
#define offsetof(t, d) __builtin_offsetof(t, d)
#endif
