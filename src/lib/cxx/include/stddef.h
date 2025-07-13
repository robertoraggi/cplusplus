#if !defined(__need_ptrdiff_t) && !defined(__need_size_t) &&     \
    !defined(__need_wchar_t) && !defined(__need_NULL) &&         \
    !defined(__need_rsize_t) && !defined(__need_nullptr_t) &&    \
    !defined(__need_max_align_t) && !defined(__need_offsetof) && \
    !defined(__need_wint_t)
#define __need_ptrdiff_t
#define __need_size_t
#define __need_wchar_t
#define __need_offsetof
#define __need_max_align_t

#if defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1
#define __need_rsize_t
#endif

#if !defined(__STDDEF_H)
#define __need_NULL
#endif

#endif /* no needs */

#ifdef __need_ptrdiff_t
#undef __need_ptrdiff_t
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#endif /* __need_ptrdiff_t */

#ifdef __need_size_t
#undef __need_size_t
typedef __SIZE_TYPE__ size_t;
#endif /* __need_size_t */

#ifdef __need_rsize_t
#undef __need_rsize_t
typedef __SIZE_TYPE__ rsize_t;
#endif /* __need_size_t */

#ifdef __need_ptrdiff_t
#undef __need_ptrdiff_t
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#endif /* __need_ptrdiff_t */

#ifdef __need_wchar_t
#undef __need_wchar_t
#ifndef __cplusplus
#if !defined(_WCHAR_T)
#define _WCHAR_T
typedef __WCHAR_TYPE__ wchar_t;
#endif /* _WCHAR_T */
#endif /* __cplusplus */
#endif /* __need_wchar_t */

#ifdef __need_NULL
#undef __need_NULL
#if !defined(NULL)
#define NULL 0
#endif /* NULL */
#endif /* __need_NULL */

#ifdef __need_wint_t
#undef __need_wint_t
typedef __WINT_TYPE__ wint_t;
#endif /* __need_wint_t */

#ifdef __need_offsetof
#undef __need_offsetof
#if !defined(offsetof)
#define offsetof(t, d) __builtin_offsetof(t, d)
#endif

#endif /* __need_offsetof */

#ifdef __need_max_align_t
#undef __need_max_align_t

typedef struct {
  long long __clang_max_align_nonce1
      __attribute__((__aligned__(__alignof__(long long))));

  long double __clang_max_align_nonce2
      __attribute__((__aligned__(__alignof__(long double))));
} max_align_t;

#endif /* __need_max_align_t */
