// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

#include "stddef_max_align_t.h"

#endif /* __need_max_align_t */
