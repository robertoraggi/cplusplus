// RUN: %cxx -toolchain wasm32 -verify -xc -U__cplusplus -D__STDC_VERSION__=202311L %s

#include <assert.h>
#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <iso646.h>
#include <limits.h>
#include <locale.h>
#include <math.h>

#ifndef __wasi__
#include <setjmp.h>
#include <signal.h>
#endif

#include <stdalign.h>
#include <stdarg.h>
#include <stdatomic.h>

#if __has_include(<stdbit.h>)
#include <stdbit.h>
#endif
#include <stdbool.h>

#if __has_include(<stdckdint.h>)
#include <stdckdint.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <string.h>
#include <tgmath.h>

#if __has_include(<threads.h>)
#include <threads.h>
#endif

#include <time.h>

#if __has_include(<uchar.h>)
#include <uchar.h>
#endif

#include <wchar.h>
#include <wctype.h>