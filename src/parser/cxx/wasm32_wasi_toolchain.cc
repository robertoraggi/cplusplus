// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/wasm32_wasi_toolchain.h>

// cxx
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>

#include <format>

namespace cxx {

Wasm32WasiToolchain::Wasm32WasiToolchain(Preprocessor* preprocessor)
    : Toolchain(preprocessor) {
  setMemoryLayout(std::make_unique<MemoryLayout>(32));
  memoryLayout()->setSizeOfLongDouble(16);
  memoryLayout()->setSizeOfLongLong(8);
}

auto Wasm32WasiToolchain::appdir() const -> const std::string& {
  return appdir_;
}

void Wasm32WasiToolchain::setAppdir(std::string appdir) {
  appdir_ = std::move(appdir);

  if (!appdir_.empty() && appdir_.back() == '/') {
    appdir_.pop_back();
  }
}

auto Wasm32WasiToolchain::sysroot() const -> const std::string& {
  return sysroot_;
}

void Wasm32WasiToolchain::setSysroot(std::string sysroot) {
  sysroot_ = std::move(sysroot);

  if (!sysroot_.empty() && sysroot_.back() == '/') {
    sysroot_.pop_back();
  }
}

void Wasm32WasiToolchain::addSystemIncludePaths() {
  addSystemIncludePath(std::format("{}/include", sysroot_));
  addSystemIncludePath(std::format("{}/include/wasm32-wasi", sysroot_));
  addSystemIncludePath(std::format("{}/../lib/cxx/include", appdir_));
}

void Wasm32WasiToolchain::addSystemCppIncludePaths() {
  addSystemIncludePath(std::format("{}/include/c++/v1", sysroot_));
  addSystemIncludePath(std::format("{}/include/wasm32-wasi/c++/v1", sysroot_));
}

void Wasm32WasiToolchain::addPredefinedMacros() {
  // clang-format off
  defineMacro("__autoreleasing", "");
  defineMacro("__strong", "");
  defineMacro("__unsafe_unretained", "");
  defineMacro("__weak", "");
  defineMacro("_Nonnull", "");
  defineMacro("_Nullable", "");
  defineMacro("_Pragma(x)", "");
  defineMacro("_Thread_local", "thread_local");

  defineMacro("_GNU_SOURCE", "1");
  defineMacro("_ILP32", "1");
  defineMacro("__ATOMIC_ACQUIRE", "2");
  defineMacro("__ATOMIC_ACQ_REL", "4");
  defineMacro("__ATOMIC_CONSUME", "1");
  defineMacro("__ATOMIC_RELAXED", "0");
  defineMacro("__ATOMIC_RELEASE", "3");
  defineMacro("__ATOMIC_SEQ_CST", "5");
  defineMacro("__BIGGEST_ALIGNMENT__", "16");
  defineMacro("__BITINT_MAXWIDTH__", "128");
  defineMacro("__BOOL_WIDTH__", "1");
  defineMacro("__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__");
  defineMacro("__CHAR16_TYPE__", "unsigned short");
  defineMacro("__CHAR32_TYPE__", "unsigned int");
  defineMacro("__CHAR_BIT__", "8");
  defineMacro("__CLANG_ATOMIC_BOOL_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_CHAR16_T_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_CHAR32_T_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_CHAR8_T_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_CHAR_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_INT_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_LLONG_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_LONG_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_POINTER_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_SHORT_LOCK_FREE", "2");
  defineMacro("__CLANG_ATOMIC_WCHAR_T_LOCK_FREE", "2");
  defineMacro("__CONSTANT_CFSTRINGS__", "1");
  defineMacro("__DBL_DECIMAL_DIG__", "17");
  defineMacro("__DBL_DENORM_MIN__", "4.9406564584124654e-324");
  defineMacro("__DBL_DIG__", "15");
  defineMacro("__DBL_EPSILON__", "2.2204460492503131e-16");
  defineMacro("__DBL_HAS_DENORM__", "1");
  defineMacro("__DBL_HAS_INFINITY__", "1");
  defineMacro("__DBL_HAS_QUIET_NAN__", "1");
  defineMacro("__DBL_MANT_DIG__", "53");
  defineMacro("__DBL_MAX_10_EXP__", "308");
  defineMacro("__DBL_MAX_EXP__", "1024");
  defineMacro("__DBL_MAX__", "1.7976931348623157e+308");
  defineMacro("__DBL_MIN_10_EXP__", "(-307)");
  defineMacro("__DBL_MIN_EXP__", "(-1021)");
  defineMacro("__DBL_MIN__", "2.2250738585072014e-308");
  defineMacro("__DBL_NORM_MAX__", "1.7976931348623157e+308");
  defineMacro("__DECIMAL_DIG__", "__LDBL_DECIMAL_DIG__");
  defineMacro("__DEPRECATED", "1");
  defineMacro("__EXCEPTIONS", "1");
  defineMacro("__FINITE_MATH_ONLY__", "0");
  defineMacro("__FLOAT128__", "1");
  defineMacro("__FLT_DECIMAL_DIG__", "9");
  defineMacro("__FLT_DENORM_MIN__", "1.40129846e-45F");
  defineMacro("__FLT_DIG__", "6");
  defineMacro("__FLT_EPSILON__", "1.19209290e-7F");
  defineMacro("__FLT_HAS_DENORM__", "1");
  defineMacro("__FLT_HAS_INFINITY__", "1");
  defineMacro("__FLT_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT_MANT_DIG__", "24");
  defineMacro("__FLT_MAX_10_EXP__", "38");
  defineMacro("__FLT_MAX_EXP__", "128");
  defineMacro("__FLT_MAX__", "3.40282347e+38F");
  defineMacro("__FLT_MIN_10_EXP__", "(-37)");
  defineMacro("__FLT_MIN_EXP__", "(-125)");
  defineMacro("__FLT_MIN__", "1.17549435e-38F");
  defineMacro("__FLT_NORM_MAX__", "3.40282347e+38F");
  defineMacro("__FLT_RADIX__", "2");
  defineMacro("__FPCLASS_NEGINF", "0x0004");
  defineMacro("__FPCLASS_NEGNORMAL", "0x0008");
  defineMacro("__FPCLASS_NEGSUBNORMAL", "0x0010");
  defineMacro("__FPCLASS_NEGZERO", "0x0020");
  defineMacro("__FPCLASS_POSINF", "0x0200");
  defineMacro("__FPCLASS_POSNORMAL", "0x0100");
  defineMacro("__FPCLASS_POSSUBNORMAL", "0x0080");
  defineMacro("__FPCLASS_POSZERO", "0x0040");
  defineMacro("__FPCLASS_QNAN", "0x0002");
  defineMacro("__FPCLASS_SNAN", "0x0001");
  defineMacro("__GCC_ATOMIC_BOOL_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR16_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR32_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR8_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_INT_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_LLONG_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_LONG_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_POINTER_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_SHORT_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_TEST_AND_SET_TRUEVAL", "1");
  defineMacro("__GCC_ATOMIC_WCHAR_T_LOCK_FREE", "2");
  defineMacro("__GCC_CONSTRUCTIVE_SIZE", "64");
  defineMacro("__GCC_DESTRUCTIVE_SIZE", "64");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8", "1");
  defineMacro("__GNUC_GNU_INLINE__", "1");
  defineMacro("__GNUC_MINOR__", "2");
  defineMacro("__GNUC_PATCHLEVEL__", "1");
  defineMacro("__GNUC__", "4");
  defineMacro("__GNUG__", "4");
  defineMacro("__GXX_ABI_VERSION", "1002");
  defineMacro("__GXX_EXPERIMENTAL_CXX0X__", "1");
  defineMacro("__GXX_RTTI", "1");
  defineMacro("__GXX_WEAK__", "1");
  defineMacro("__ILP32__", "1");
  defineMacro("__INT16_C_SUFFIX__", "");
  defineMacro("__INT16_FMTd__", "\"hd\"");
  defineMacro("__INT16_FMTi__", "\"hi\"");
  defineMacro("__INT16_MAX__", "32767");
  defineMacro("__INT16_TYPE__", "short");
  defineMacro("__INT32_C_SUFFIX__", "");
  defineMacro("__INT32_FMTd__", "\"d\"");
  defineMacro("__INT32_FMTi__", "\"i\"");
  defineMacro("__INT32_MAX__", "2147483647");
  defineMacro("__INT32_TYPE__", "int");
  defineMacro("__INT64_C_SUFFIX__", "LL");
  defineMacro("__INT64_FMTd__", "\"lld\"");
  defineMacro("__INT64_FMTi__", "\"lli\"");
  defineMacro("__INT64_MAX__", "9223372036854775807LL");
  defineMacro("__INT64_TYPE__", "long long int");
  defineMacro("__INT8_C_SUFFIX__", "");
  defineMacro("__INT8_FMTd__", "\"hhd\"");
  defineMacro("__INT8_FMTi__", "\"hhi\"");
  defineMacro("__INT8_MAX__", "127");
  defineMacro("__INT8_TYPE__", "signed char");
  defineMacro("__INTMAX_C_SUFFIX__", "LL");
  defineMacro("__INTMAX_FMTd__", "\"lld\"");
  defineMacro("__INTMAX_FMTi__", "\"lli\"");
  defineMacro("__INTMAX_MAX__", "9223372036854775807LL");
  defineMacro("__INTMAX_TYPE__", "long long int");
  defineMacro("__INTMAX_WIDTH__", "64");
  defineMacro("__INTPTR_FMTd__", "\"ld\"");
  defineMacro("__INTPTR_FMTi__", "\"li\"");
  defineMacro("__INTPTR_MAX__", "2147483647L");
  defineMacro("__INTPTR_TYPE__", "long int");
  defineMacro("__INTPTR_WIDTH__", "32");
  defineMacro("__INT_FAST16_FMTd__", "\"hd\"");
  defineMacro("__INT_FAST16_FMTi__", "\"hi\"");
  defineMacro("__INT_FAST16_MAX__", "32767");
  defineMacro("__INT_FAST16_TYPE__", "short");
  defineMacro("__INT_FAST16_WIDTH__", "16");
  defineMacro("__INT_FAST32_FMTd__", "\"d\"");
  defineMacro("__INT_FAST32_FMTi__", "\"i\"");
  defineMacro("__INT_FAST32_MAX__", "2147483647");
  defineMacro("__INT_FAST32_TYPE__", "int");
  defineMacro("__INT_FAST32_WIDTH__", "32");
  defineMacro("__INT_FAST64_FMTd__", "\"lld\"");
  defineMacro("__INT_FAST64_FMTi__", "\"lli\"");
  defineMacro("__INT_FAST64_MAX__", "9223372036854775807LL");
  defineMacro("__INT_FAST64_TYPE__", "long long int");
  defineMacro("__INT_FAST64_WIDTH__", "64");
  defineMacro("__INT_FAST8_FMTd__", "\"hhd\"");
  defineMacro("__INT_FAST8_FMTi__", "\"hhi\"");
  defineMacro("__INT_FAST8_MAX__", "127");
  defineMacro("__INT_FAST8_TYPE__", "signed char");
  defineMacro("__INT_FAST8_WIDTH__", "8");
  defineMacro("__INT_LEAST16_FMTd__", "\"hd\"");
  defineMacro("__INT_LEAST16_FMTi__", "\"hi\"");
  defineMacro("__INT_LEAST16_MAX__", "32767");
  defineMacro("__INT_LEAST16_TYPE__", "short");
  defineMacro("__INT_LEAST16_WIDTH__", "16");
  defineMacro("__INT_LEAST32_FMTd__", "\"d\"");
  defineMacro("__INT_LEAST32_FMTi__", "\"i\"");
  defineMacro("__INT_LEAST32_MAX__", "2147483647");
  defineMacro("__INT_LEAST32_TYPE__", "int");
  defineMacro("__INT_LEAST32_WIDTH__", "32");
  defineMacro("__INT_LEAST64_FMTd__", "\"lld\"");
  defineMacro("__INT_LEAST64_FMTi__", "\"lli\"");
  defineMacro("__INT_LEAST64_MAX__", "9223372036854775807LL");
  defineMacro("__INT_LEAST64_TYPE__", "long long int");
  defineMacro("__INT_LEAST64_WIDTH__", "64");
  defineMacro("__INT_LEAST8_FMTd__", "\"hhd\"");
  defineMacro("__INT_LEAST8_FMTi__", "\"hhi\"");
  defineMacro("__INT_LEAST8_MAX__", "127");
  defineMacro("__INT_LEAST8_TYPE__", "signed char");
  defineMacro("__INT_LEAST8_WIDTH__", "8");
  defineMacro("__INT_MAX__", "2147483647");
  defineMacro("__INT_WIDTH__", "32");
  defineMacro("__LDBL_DECIMAL_DIG__", "36");
  defineMacro("__LDBL_DENORM_MIN__", "6.47517511943802511092443895822764655e-4966L");
  defineMacro("__LDBL_DIG__", "33");
  defineMacro("__LDBL_EPSILON__", "1.92592994438723585305597794258492732e-34L");
  defineMacro("__LDBL_HAS_DENORM__", "1");
  defineMacro("__LDBL_HAS_INFINITY__", "1");
  defineMacro("__LDBL_HAS_QUIET_NAN__", "1");
  defineMacro("__LDBL_MANT_DIG__", "113");
  defineMacro("__LDBL_MAX_10_EXP__", "4932");
  defineMacro("__LDBL_MAX_EXP__", "16384");
  defineMacro("__LDBL_MAX__", "1.18973149535723176508575932662800702e+4932L");
  defineMacro("__LDBL_MIN_10_EXP__", "(-4931)");
  defineMacro("__LDBL_MIN_EXP__", "(-16381)");
  defineMacro("__LDBL_MIN__", "3.36210314311209350626267781732175260e-4932L");
  defineMacro("__LDBL_NORM_MAX__", "1.18973149535723176508575932662800702e+4932L");
  defineMacro("__LITTLE_ENDIAN__", "1");
  defineMacro("__LLONG_WIDTH__", "64");
  defineMacro("__LONG_LONG_MAX__", "9223372036854775807LL");
  defineMacro("__LONG_MAX__", "2147483647L");
  defineMacro("__LONG_WIDTH__", "32");
  defineMacro("__MEMORY_SCOPE_DEVICE", "1");
  defineMacro("__MEMORY_SCOPE_SINGLE", "4");
  defineMacro("__MEMORY_SCOPE_SYSTEM", "0");
  defineMacro("__MEMORY_SCOPE_WRKGRP", "2");
  defineMacro("__MEMORY_SCOPE_WVFRNT", "3");
  defineMacro("__NO_INLINE__", "1");
  defineMacro("__NO_MATH_ERRNO__", "1");
  defineMacro("__OBJC_BOOL_IS_BOOL", "0");
  defineMacro("__OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES", "3");
  defineMacro("__OPENCL_MEMORY_SCOPE_DEVICE", "2");
  defineMacro("__OPENCL_MEMORY_SCOPE_SUB_GROUP", "4");
  defineMacro("__OPENCL_MEMORY_SCOPE_WORK_GROUP", "1");
  defineMacro("__OPENCL_MEMORY_SCOPE_WORK_ITEM", "0");
  defineMacro("__ORDER_BIG_ENDIAN__", "4321");
  defineMacro("__ORDER_LITTLE_ENDIAN__", "1234");
  defineMacro("__ORDER_PDP_ENDIAN__", "3412");
  defineMacro("__POINTER_WIDTH__", "32");
  defineMacro("__PRAGMA_REDEFINE_EXTNAME", "1");
  defineMacro("__PTRDIFF_FMTd__", "\"ld\"");
  defineMacro("__PTRDIFF_FMTi__", "\"li\"");
  defineMacro("__PTRDIFF_MAX__", "2147483647L");
  defineMacro("__PTRDIFF_TYPE__", "long int");
  defineMacro("__PTRDIFF_WIDTH__", "32");
  defineMacro("__SCHAR_MAX__", "127");
  defineMacro("__SHRT_MAX__", "32767");
  defineMacro("__SHRT_WIDTH__", "16");
  defineMacro("__SIG_ATOMIC_MAX__", "2147483647L");
  defineMacro("__SIG_ATOMIC_WIDTH__", "32");
  defineMacro("__SIZEOF_DOUBLE__", "8");
  defineMacro("__SIZEOF_FLOAT__", "4");
  defineMacro("__SIZEOF_INT128__", "16");
  defineMacro("__SIZEOF_INT__", "4");
  defineMacro("__SIZEOF_LONG_DOUBLE__", "16");
  defineMacro("__SIZEOF_LONG_LONG__", "8");
  defineMacro("__SIZEOF_LONG__", "4");
  defineMacro("__SIZEOF_POINTER__", "4");
  defineMacro("__SIZEOF_PTRDIFF_T__", "4");
  defineMacro("__SIZEOF_SHORT__", "2");
  defineMacro("__SIZEOF_SIZE_T__", "4");
  defineMacro("__SIZEOF_WCHAR_T__", "4");
  defineMacro("__SIZEOF_WINT_T__", "4");
  defineMacro("__SIZE_FMTX__", "\"lX\"");
  defineMacro("__SIZE_FMTo__", "\"lo\"");
  defineMacro("__SIZE_FMTu__", "\"lu\"");
  defineMacro("__SIZE_FMTx__", "\"lx\"");
  defineMacro("__SIZE_MAX__", "4294967295UL");
  defineMacro("__SIZE_TYPE__", "long unsigned int");
  defineMacro("__SIZE_WIDTH__", "32");
  defineMacro("__STDCPP_DEFAULT_NEW_ALIGNMENT__", "16UL");
  defineMacro("__STDC_EMBED_EMPTY__", "2");
  defineMacro("__STDC_EMBED_FOUND__", "1");
  defineMacro("__STDC_EMBED_NOT_FOUND__", "0");
  defineMacro("__STDC_HOSTED__", "1");
  defineMacro("__STDC_UTF_16__", "1");
  defineMacro("__STDC_UTF_32__", "1");
  defineMacro("__STDC__", "1");
  defineMacro("__STRICT_ANSI__", "1");
  defineMacro("__UINT16_C_SUFFIX__", "");
  defineMacro("__UINT16_FMTX__", "\"hX\"");
  defineMacro("__UINT16_FMTo__", "\"ho\"");
  defineMacro("__UINT16_FMTu__", "\"hu\"");
  defineMacro("__UINT16_FMTx__", "\"hx\"");
  defineMacro("__UINT16_MAX__", "65535");
  defineMacro("__UINT16_TYPE__", "unsigned short");
  defineMacro("__UINT32_C_SUFFIX__", "U");
  defineMacro("__UINT32_FMTX__", "\"X\"");
  defineMacro("__UINT32_FMTo__", "\"o\"");
  defineMacro("__UINT32_FMTu__", "\"u\"");
  defineMacro("__UINT32_FMTx__", "\"x\"");
  defineMacro("__UINT32_MAX__", "4294967295U");
  defineMacro("__UINT32_TYPE__", "unsigned int");
  defineMacro("__UINT64_C_SUFFIX__", "ULL");
  defineMacro("__UINT64_FMTX__", "\"llX\"");
  defineMacro("__UINT64_FMTo__", "\"llo\"");
  defineMacro("__UINT64_FMTu__", "\"llu\"");
  defineMacro("__UINT64_FMTx__", "\"llx\"");
  defineMacro("__UINT64_MAX__", "18446744073709551615ULL");
  defineMacro("__UINT64_TYPE__", "long long unsigned int");
  defineMacro("__UINT8_C_SUFFIX__", "");
  defineMacro("__UINT8_FMTX__", "\"hhX\"");
  defineMacro("__UINT8_FMTo__", "\"hho\"");
  defineMacro("__UINT8_FMTu__", "\"hhu\"");
  defineMacro("__UINT8_FMTx__", "\"hhx\"");
  defineMacro("__UINT8_MAX__", "255");
  defineMacro("__UINT8_TYPE__", "unsigned char");
  defineMacro("__UINTMAX_C_SUFFIX__", "ULL");
  defineMacro("__UINTMAX_FMTX__", "\"llX\"");
  defineMacro("__UINTMAX_FMTo__", "\"llo\"");
  defineMacro("__UINTMAX_FMTu__", "\"llu\"");
  defineMacro("__UINTMAX_FMTx__", "\"llx\"");
  defineMacro("__UINTMAX_MAX__", "18446744073709551615ULL");
  defineMacro("__UINTMAX_TYPE__", "long long unsigned int");
  defineMacro("__UINTMAX_WIDTH__", "64");
  defineMacro("__UINTPTR_FMTX__", "\"lX\"");
  defineMacro("__UINTPTR_FMTo__", "\"lo\"");
  defineMacro("__UINTPTR_FMTu__", "\"lu\"");
  defineMacro("__UINTPTR_FMTx__", "\"lx\"");
  defineMacro("__UINTPTR_MAX__", "4294967295UL");
  defineMacro("__UINTPTR_TYPE__", "long unsigned int");
  defineMacro("__UINTPTR_WIDTH__", "32");
  defineMacro("__UINT_FAST16_FMTX__", "\"hX\"");
  defineMacro("__UINT_FAST16_FMTo__", "\"ho\"");
  defineMacro("__UINT_FAST16_FMTu__", "\"hu\"");
  defineMacro("__UINT_FAST16_FMTx__", "\"hx\"");
  defineMacro("__UINT_FAST16_MAX__", "65535");
  defineMacro("__UINT_FAST16_TYPE__", "unsigned short");
  defineMacro("__UINT_FAST32_FMTX__", "\"X\"");
  defineMacro("__UINT_FAST32_FMTo__", "\"o\"");
  defineMacro("__UINT_FAST32_FMTu__", "\"u\"");
  defineMacro("__UINT_FAST32_FMTx__", "\"x\"");
  defineMacro("__UINT_FAST32_MAX__", "4294967295U");
  defineMacro("__UINT_FAST32_TYPE__", "unsigned int");
  defineMacro("__UINT_FAST64_FMTX__", "\"llX\"");
  defineMacro("__UINT_FAST64_FMTo__", "\"llo\"");
  defineMacro("__UINT_FAST64_FMTu__", "\"llu\"");
  defineMacro("__UINT_FAST64_FMTx__", "\"llx\"");
  defineMacro("__UINT_FAST64_MAX__", "18446744073709551615ULL");
  defineMacro("__UINT_FAST64_TYPE__", "long long unsigned int");
  defineMacro("__UINT_FAST8_FMTX__", "\"hhX\"");
  defineMacro("__UINT_FAST8_FMTo__", "\"hho\"");
  defineMacro("__UINT_FAST8_FMTu__", "\"hhu\"");
  defineMacro("__UINT_FAST8_FMTx__", "\"hhx\"");
  defineMacro("__UINT_FAST8_MAX__", "255");
  defineMacro("__UINT_FAST8_TYPE__", "unsigned char");
  defineMacro("__UINT_LEAST16_FMTX__", "\"hX\"");
  defineMacro("__UINT_LEAST16_FMTo__", "\"ho\"");
  defineMacro("__UINT_LEAST16_FMTu__", "\"hu\"");
  defineMacro("__UINT_LEAST16_FMTx__", "\"hx\"");
  defineMacro("__UINT_LEAST16_MAX__", "65535");
  defineMacro("__UINT_LEAST16_TYPE__", "unsigned short");
  defineMacro("__UINT_LEAST32_FMTX__", "\"X\"");
  defineMacro("__UINT_LEAST32_FMTo__", "\"o\"");
  defineMacro("__UINT_LEAST32_FMTu__", "\"u\"");
  defineMacro("__UINT_LEAST32_FMTx__", "\"x\"");
  defineMacro("__UINT_LEAST32_MAX__", "4294967295U");
  defineMacro("__UINT_LEAST32_TYPE__", "unsigned int");
  defineMacro("__UINT_LEAST64_FMTX__", "\"llX\"");
  defineMacro("__UINT_LEAST64_FMTo__", "\"llo\"");
  defineMacro("__UINT_LEAST64_FMTu__", "\"llu\"");
  defineMacro("__UINT_LEAST64_FMTx__", "\"llx\"");
  defineMacro("__UINT_LEAST64_MAX__", "18446744073709551615ULL");
  defineMacro("__UINT_LEAST64_TYPE__", "long long unsigned int");
  defineMacro("__UINT_LEAST8_FMTX__", "\"hhX\"");
  defineMacro("__UINT_LEAST8_FMTo__", "\"hho\"");
  defineMacro("__UINT_LEAST8_FMTu__", "\"hhu\"");
  defineMacro("__UINT_LEAST8_FMTx__", "\"hhx\"");
  defineMacro("__UINT_LEAST8_MAX__", "255");
  defineMacro("__UINT_LEAST8_TYPE__", "unsigned char");
  defineMacro("__USER_LABEL_PREFIX__", "");
  defineMacro("__VERSION__", "\"Clang 20.0.0git\"");
  defineMacro("__WCHAR_MAX__", "2147483647");
  defineMacro("__WCHAR_TYPE__", "int");
  defineMacro("__WCHAR_WIDTH__", "32");
  defineMacro("__WINT_MAX__", "2147483647");
  defineMacro("__WINT_TYPE__", "int");
  defineMacro("__WINT_WIDTH__", "32");
  defineMacro("__clang__", "1");
  defineMacro("__clang_literal_encoding__", "\"UTF-8\"");
  defineMacro("__clang_major__", "20");
  defineMacro("__clang_minor__", "0");
  defineMacro("__clang_patchlevel__", "0");
  defineMacro("__clang_version__", "\"20.0.0git \"");
  defineMacro("__clang_wide_literal_encoding__", "\"UTF-32\"");
  defineMacro("__cplusplus", "202400L");
  defineMacro("__cpp_aggregate_bases", "201603L");
  defineMacro("__cpp_aggregate_nsdmi", "201304L");
  defineMacro("__cpp_aggregate_paren_init", "201902L");
  defineMacro("__cpp_alias_templates", "200704L");
  defineMacro("__cpp_aligned_new", "201606L");
  defineMacro("__cpp_attributes", "200809L");
  defineMacro("__cpp_auto_cast", "202110L");
  defineMacro("__cpp_binary_literals", "201304L");
  defineMacro("__cpp_capture_star_this", "201603L");
  defineMacro("__cpp_char8_t", "202207L");
  defineMacro("__cpp_concepts", "202002");
  defineMacro("__cpp_conditional_explicit", "201806L");
  defineMacro("__cpp_consteval", "202211L");
  defineMacro("__cpp_constexpr", "202406L");
  defineMacro("__cpp_constexpr_dynamic_alloc", "201907L");
  defineMacro("__cpp_constexpr_in_decltype", "201711L");
  defineMacro("__cpp_constinit", "201907L");
  defineMacro("__cpp_decltype", "200707L");
  defineMacro("__cpp_decltype_auto", "201304L");
  defineMacro("__cpp_deduction_guides", "201703L");
  defineMacro("__cpp_delegating_constructors", "200604L");
  defineMacro("__cpp_deleted_function", "202403L");
  defineMacro("__cpp_designated_initializers", "201707L");
  defineMacro("__cpp_digit_separators", "201309L");
  defineMacro("__cpp_enumerator_attributes", "201411L");
  defineMacro("__cpp_exceptions", "199711L");
  defineMacro("__cpp_fold_expressions", "201603L");
  defineMacro("__cpp_generic_lambdas", "201707L");
  defineMacro("__cpp_guaranteed_copy_elision", "201606L");
  defineMacro("__cpp_hex_float", "201603L");
  defineMacro("__cpp_if_consteval", "202106L");
  defineMacro("__cpp_if_constexpr", "201606L");
  defineMacro("__cpp_impl_coroutine", "201902L");
  defineMacro("__cpp_impl_destroying_delete", "201806L");
  defineMacro("__cpp_impl_three_way_comparison", "201907L");
  defineMacro("__cpp_implicit_move", "202207L");
  defineMacro("__cpp_inheriting_constructors", "201511L");
  defineMacro("__cpp_init_captures", "201803L");
  defineMacro("__cpp_initializer_lists", "200806L");
  defineMacro("__cpp_inline_variables", "201606L");
  defineMacro("__cpp_lambdas", "200907L");
  defineMacro("__cpp_multidimensional_subscript", "202211L");
  defineMacro("__cpp_named_character_escapes", "202207L");
  defineMacro("__cpp_namespace_attributes", "201411L");
  defineMacro("__cpp_nested_namespace_definitions", "201411L");
  defineMacro("__cpp_noexcept_function_type", "201510L");
  defineMacro("__cpp_nontype_template_args", "201411L");
  defineMacro("__cpp_nontype_template_parameter_auto", "201606L");
  defineMacro("__cpp_nsdmi", "200809L");
  defineMacro("__cpp_pack_indexing", "202311L");
  defineMacro("__cpp_placeholder_variables", "202306L");
  defineMacro("__cpp_range_based_for", "202211L");
  defineMacro("__cpp_raw_strings", "200710L");
  defineMacro("__cpp_ref_qualifiers", "200710L");
  defineMacro("__cpp_return_type_deduction", "201304L");
  defineMacro("__cpp_rtti", "199711L");
  defineMacro("__cpp_rvalue_references", "200610L");
  defineMacro("__cpp_size_t_suffix", "202011L");
  defineMacro("__cpp_sized_deallocation", "201309L");
  defineMacro("__cpp_static_assert", "202306L");
  defineMacro("__cpp_static_call_operator", "202207L");
  defineMacro("__cpp_structured_bindings", "202403L");
  defineMacro("__cpp_template_auto", "201606L");
  defineMacro("__cpp_template_template_args", "201611L");
  defineMacro("__cpp_unicode_characters", "200704L");
  defineMacro("__cpp_unicode_literals", "200710L");
  defineMacro("__cpp_user_defined_literals", "200809L");
  defineMacro("__cpp_using_enum", "201907L");
  defineMacro("__cpp_variable_templates", "201304L");
  defineMacro("__cpp_variadic_friend", "202403L");
  defineMacro("__cpp_variadic_templates", "200704L");
  defineMacro("__cpp_variadic_using", "201611L");
  defineMacro("__llvm__", "1");
  defineMacro("__private_extern__", "extern");
  defineMacro("__wasi__", "1");
  defineMacro("__wasm", "1");
  defineMacro("__wasm32", "1");
  defineMacro("__wasm32__", "1");
  defineMacro("__wasm__", "1");
  defineMacro("__wasm_bulk_memory__", "1");
  defineMacro("__wasm_bulk_memory_opt__", "1");
  defineMacro("__wasm_multivalue__", "1");
  defineMacro("__wasm_mutable_globals__", "1");
  defineMacro("__wasm_nontrapping_fptoint__", "1");
  defineMacro("__wasm_reference_types__", "1");
  defineMacro("__wasm_sign_ext__", "1");
  // clang-format off
}

}  // namespace cxx
