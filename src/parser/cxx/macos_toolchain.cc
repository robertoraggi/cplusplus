// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/macos_toolchain.h>
#include <cxx/preprocessor.h>

namespace cxx {

void MacOSToolchain::addSystemIncludePaths() {
  addSystemIncludePath(
      "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/"
      "Frameworks");

  addSystemIncludePath(
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "XcodeDefault.xctoolchain/usr/include");

  addSystemIncludePath(
      "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include");

  addSystemIncludePath(
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "XcodeDefault.xctoolchain/usr/lib/clang/14.0.3/include");
}

void MacOSToolchain::addSystemCppIncludePaths() {
  addSystemIncludePath(
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "XcodeDefault.xctoolchain/usr/bin/../include/c++/v1");
}

void MacOSToolchain::addPredefinedMacros() {
  // clang-format off
  defineMacro("__extension__", "");
  defineMacro("__null", "nullptr");
  defineMacro("__signed__", "signed");
  defineMacro("_Pragma(x)", "");
  defineMacro("_Nullable", "");
  defineMacro("_Nonnull", "");
  defineMacro("__autoreleasing", "");

  defineMacro("_LP64", "1");
  defineMacro("__APPLE_CC__", "6000");
  defineMacro("__APPLE__", "1");
  defineMacro("__ATOMIC_ACQUIRE", "2");
  defineMacro("__ATOMIC_ACQ_REL", "4");
  defineMacro("__ATOMIC_CONSUME", "1");
  defineMacro("__ATOMIC_RELAXED", "0");
  defineMacro("__ATOMIC_RELEASE", "3");
  defineMacro("__ATOMIC_SEQ_CST", "5");
  defineMacro("__BIGGEST_ALIGNMENT__", "16");
  defineMacro("__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__");
  defineMacro("__CHAR16_TYPE__ unsigned", "short");
  defineMacro("__CHAR32_TYPE__ unsigned", "int");
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
  defineMacro("__DECIMAL_DIG__", "__LDBL_DECIMAL_DIG__");
  defineMacro("__DEPRECATED", "1");
  defineMacro("__DYNAMIC__", "1");
  defineMacro("__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__", "101500");
  defineMacro("__EXCEPTIONS", "1");
  defineMacro("__FINITE_MATH_ONLY__", "0");
  defineMacro("__FLT16_DECIMAL_DIG__", "5");
  defineMacro("__FLT16_DENORM_MIN__", "5.9604644775390625e-8F16");
  defineMacro("__FLT16_DIG__", "3");
  defineMacro("__FLT16_EPSILON__", "9.765625e-4F16");
  defineMacro("__FLT16_HAS_DENORM__", "1");
  defineMacro("__FLT16_HAS_INFINITY__", "1");
  defineMacro("__FLT16_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT16_MANT_DIG__", "11");
  defineMacro("__FLT16_MAX_10_EXP__", "4");
  defineMacro("__FLT16_MAX_EXP__", "16");
  defineMacro("__FLT16_MAX__", "6.5504e+4F16");
  defineMacro("__FLT16_MIN_10_EXP__", "(-4)");
  defineMacro("__FLT16_MIN_EXP__", "(-13)");
  defineMacro("__FLT16_MIN__", "6.103515625e-5F16");
  defineMacro("__FLT_DECIMAL_DIG__", "9");
  defineMacro("__FLT_DENORM_MIN__", "1.40129846e-45F");
  defineMacro("__FLT_DIG__", "6");
  defineMacro("__FLT_EPSILON__", "1.19209290e-7F");
  defineMacro("__FLT_EVAL_METHOD__", "0");
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
  defineMacro("__FLT_RADIX__", "2");
  defineMacro("__FXSR__", "1");
  defineMacro("__GCC_ASM_FLAG_OUTPUTS__", "1");
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
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16", "1");
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
  defineMacro("__INT64_TYPE__ long long", "int");
  defineMacro("__INT8_C_SUFFIX__", "");
  defineMacro("__INT8_FMTd__", "\"hhd\"");
  defineMacro("__INT8_FMTi__", "\"hhi\"");
  defineMacro("__INT8_MAX__", "127");
  defineMacro("__INT8_TYPE__ signed", "char");
  defineMacro("__INTMAX_C_SUFFIX__", "L");
  defineMacro("__INTMAX_FMTd__", "\"ld\"");
  defineMacro("__INTMAX_FMTi__", "\"li\"");
  defineMacro("__INTMAX_MAX__", "9223372036854775807L");
  defineMacro("__INTMAX_TYPE__ long", "int");
  defineMacro("__INTMAX_WIDTH__", "64");
  defineMacro("__INTPTR_FMTd__", "\"ld\"");
  defineMacro("__INTPTR_FMTi__", "\"li\"");
  defineMacro("__INTPTR_MAX__", "9223372036854775807L");
  defineMacro("__INTPTR_TYPE__ long", "int");
  defineMacro("__INTPTR_WIDTH__", "64");
  defineMacro("__INT_FAST16_FMTd__", "\"hd\"");
  defineMacro("__INT_FAST16_FMTi__", "\"hi\"");
  defineMacro("__INT_FAST16_MAX__", "32767");
  defineMacro("__INT_FAST16_TYPE__", "short");
  defineMacro("__INT_FAST32_FMTd__", "\"d\"");
  defineMacro("__INT_FAST32_FMTi__", "\"i\"");
  defineMacro("__INT_FAST32_MAX__", "2147483647");
  defineMacro("__INT_FAST32_TYPE__", "int");
  defineMacro("__INT_FAST64_FMTd__", "\"lld\"");
  defineMacro("__INT_FAST64_FMTi__", "\"lli\"");
  defineMacro("__INT_FAST64_MAX__", "9223372036854775807LL");
  defineMacro("__INT_FAST64_TYPE__ long long", "int");
  defineMacro("__INT_FAST8_FMTd__", "\"hhd\"");
  defineMacro("__INT_FAST8_FMTi__", "\"hhi\"");
  defineMacro("__INT_FAST8_MAX__", "127");
  defineMacro("__INT_FAST8_TYPE__ signed", "char");
  defineMacro("__INT_LEAST16_FMTd__", "\"hd\"");
  defineMacro("__INT_LEAST16_FMTi__", "\"hi\"");
  defineMacro("__INT_LEAST16_MAX__", "32767");
  defineMacro("__INT_LEAST16_TYPE__", "short");
  defineMacro("__INT_LEAST32_FMTd__", "\"d\"");
  defineMacro("__INT_LEAST32_FMTi__", "\"i\"");
  defineMacro("__INT_LEAST32_MAX__", "2147483647");
  defineMacro("__INT_LEAST32_TYPE__", "int");
  defineMacro("__INT_LEAST64_FMTd__", "\"lld\"");
  defineMacro("__INT_LEAST64_FMTi__", "\"lli\"");
  defineMacro("__INT_LEAST64_MAX__", "9223372036854775807LL");
  defineMacro("__INT_LEAST64_TYPE__ long long", "int");
  defineMacro("__INT_LEAST8_FMTd__", "\"hhd\"");
  defineMacro("__INT_LEAST8_FMTi__", "\"hhi\"");
  defineMacro("__INT_LEAST8_MAX__", "127");
  defineMacro("__INT_LEAST8_TYPE__ signed", "char");
  defineMacro("__INT_MAX__", "2147483647");
  defineMacro("__LDBL_DECIMAL_DIG__", "21");
  defineMacro("__LDBL_DENORM_MIN__", "3.64519953188247460253e-4951L");
  defineMacro("__LDBL_DIG__", "18");
  defineMacro("__LDBL_EPSILON__", "1.08420217248550443401e-19L");
  defineMacro("__LDBL_HAS_DENORM__", "1");
  defineMacro("__LDBL_HAS_INFINITY__", "1");
  defineMacro("__LDBL_HAS_QUIET_NAN__", "1");
  defineMacro("__LDBL_MANT_DIG__", "64");
  defineMacro("__LDBL_MAX_10_EXP__", "4932");
  defineMacro("__LDBL_MAX_EXP__", "16384");
  defineMacro("__LDBL_MAX__", "1.18973149535723176502e+4932L");
  defineMacro("__LDBL_MIN_10_EXP__", "(-4931)");
  defineMacro("__LDBL_MIN_EXP__", "(-16381)");
  defineMacro("__LDBL_MIN__", "3.36210314311209350626e-4932L");
  defineMacro("__LITTLE_ENDIAN__", "1");
  defineMacro("__LONG_LONG_MAX__", "9223372036854775807LL");
  defineMacro("__LONG_MAX__", "9223372036854775807L");
  defineMacro("__LP64__", "1");
  defineMacro("__MACH__", "1");
  defineMacro("__MMX__", "1");
  defineMacro("__NO_INLINE__", "1");
  defineMacro("__NO_MATH_INLINES", "1");
  defineMacro("__OBJC_BOOL_IS_BOOL", "0");
  defineMacro("__OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES", "3");
  defineMacro("__OPENCL_MEMORY_SCOPE_DEVICE", "2");
  defineMacro("__OPENCL_MEMORY_SCOPE_SUB_GROUP", "4");
  defineMacro("__OPENCL_MEMORY_SCOPE_WORK_GROUP", "1");
  defineMacro("__OPENCL_MEMORY_SCOPE_WORK_ITEM", "0");
  defineMacro("__ORDER_BIG_ENDIAN__", "4321");
  defineMacro("__ORDER_LITTLE_ENDIAN__", "1234");
  defineMacro("__ORDER_PDP_ENDIAN__", "3412");
  defineMacro("__PIC__", "2");
  defineMacro("__POINTER_WIDTH__", "64");
  defineMacro("__PRAGMA_REDEFINE_EXTNAME", "1");
  defineMacro("__PTRDIFF_FMTd__", "\"ld\"");
  defineMacro("__PTRDIFF_FMTi__", "\"li\"");
  defineMacro("__PTRDIFF_MAX__", "9223372036854775807L");
  defineMacro("__PTRDIFF_TYPE__ long", "int");
  defineMacro("__PTRDIFF_WIDTH__", "64");
  defineMacro("__REGISTER_PREFIX__", "");
  defineMacro("__SCHAR_MAX__", "127");
  defineMacro("__SEG_FS", "1");
  defineMacro("__SEG_GS", "1");
  defineMacro("__SHRT_MAX__", "32767");
  defineMacro("__SIG_ATOMIC_MAX__", "2147483647");
  defineMacro("__SIG_ATOMIC_WIDTH__", "32");
  defineMacro("__SIZEOF_DOUBLE__", "8");
  defineMacro("__SIZEOF_FLOAT__", "4");
  defineMacro("__SIZEOF_INT128__", "16");
  defineMacro("__SIZEOF_INT__", "4");
  defineMacro("__SIZEOF_LONG_DOUBLE__", "16");
  defineMacro("__SIZEOF_LONG_LONG__", "8");
  defineMacro("__SIZEOF_LONG__", "8");
  defineMacro("__SIZEOF_POINTER__", "8");
  defineMacro("__SIZEOF_PTRDIFF_T__", "8");
  defineMacro("__SIZEOF_SHORT__", "2");
  defineMacro("__SIZEOF_SIZE_T__", "8");
  defineMacro("__SIZEOF_WCHAR_T__", "4");
  defineMacro("__SIZEOF_WINT_T__", "4");
  defineMacro("__SIZE_FMTX__", "\"lX\"");
  defineMacro("__SIZE_FMTo__", "\"lo\"");
  defineMacro("__SIZE_FMTu__", "\"lu\"");
  defineMacro("__SIZE_FMTx__", "\"lx\"");
  defineMacro("__SIZE_MAX__", "18446744073709551615UL");
  defineMacro("__SIZE_TYPE__ long unsigned", "int");
  defineMacro("__SIZE_WIDTH__", "64");
  defineMacro("__SSE2_MATH__", "1");
  defineMacro("__SSE2__", "1");
  defineMacro("__SSE3__", "1");
  defineMacro("__SSE4_1__", "1");
  defineMacro("__SSE_MATH__", "1");
  defineMacro("__SSE__", "1");
  defineMacro("__SSP__", "1");
  defineMacro("__SSSE3__", "1");
  defineMacro("__STDCPP_DEFAULT_NEW_ALIGNMENT__", "16UL");
  defineMacro("__STDC_HOSTED__", "1");
  defineMacro("__STDC_NO_THREADS__", "1");
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
  defineMacro("__UINT16_TYPE__ unsigned", "short");
  defineMacro("__UINT32_C_SUFFIX__", "U");
  defineMacro("__UINT32_FMTX__", "\"X\"");
  defineMacro("__UINT32_FMTo__", "\"o\"");
  defineMacro("__UINT32_FMTu__", "\"u\"");
  defineMacro("__UINT32_FMTx__", "\"x\"");
  defineMacro("__UINT32_MAX__", "4294967295U");
  defineMacro("__UINT32_TYPE__ unsigned", "int");
  defineMacro("__UINT64_C_SUFFIX__", "ULL");
  defineMacro("__UINT64_FMTX__", "\"llX\"");
  defineMacro("__UINT64_FMTo__", "\"llo\"");
  defineMacro("__UINT64_FMTu__", "\"llu\"");
  defineMacro("__UINT64_FMTx__", "\"llx\"");
  defineMacro("__UINT64_MAX__", "18446744073709551615ULL");
  defineMacro("__UINT64_TYPE__ long long unsigned", "int");
  defineMacro("__UINT8_C_SUFFIX__", "");
  defineMacro("__UINT8_FMTX__", "\"hhX\"");
  defineMacro("__UINT8_FMTo__", "\"hho\"");
  defineMacro("__UINT8_FMTu__", "\"hhu\"");
  defineMacro("__UINT8_FMTx__", "\"hhx\"");
  defineMacro("__UINT8_MAX__", "255");
  defineMacro("__UINT8_TYPE__ unsigned", "char");
  defineMacro("__UINTMAX_C_SUFFIX__", "UL");
  defineMacro("__UINTMAX_FMTX__", "\"lX\"");
  defineMacro("__UINTMAX_FMTo__", "\"lo\"");
  defineMacro("__UINTMAX_FMTu__", "\"lu\"");
  defineMacro("__UINTMAX_FMTx__", "\"lx\"");
  defineMacro("__UINTMAX_MAX__", "18446744073709551615UL");
  defineMacro("__UINTMAX_TYPE__ long unsigned", "int");
  defineMacro("__UINTMAX_WIDTH__", "64");
  defineMacro("__UINTPTR_FMTX__", "\"lX\"");
  defineMacro("__UINTPTR_FMTo__", "\"lo\"");
  defineMacro("__UINTPTR_FMTu__", "\"lu\"");
  defineMacro("__UINTPTR_FMTx__", "\"lx\"");
  defineMacro("__UINTPTR_MAX__", "18446744073709551615UL");
  defineMacro("__UINTPTR_TYPE__ long unsigned", "int");
  defineMacro("__UINTPTR_WIDTH__", "64");
  defineMacro("__UINT_FAST16_FMTX__", "\"hX\"");
  defineMacro("__UINT_FAST16_FMTo__", "\"ho\"");
  defineMacro("__UINT_FAST16_FMTu__", "\"hu\"");
  defineMacro("__UINT_FAST16_FMTx__", "\"hx\"");
  defineMacro("__UINT_FAST16_MAX__", "65535");
  defineMacro("__UINT_FAST16_TYPE__ unsigned", "short");
  defineMacro("__UINT_FAST32_FMTX__", "\"X\"");
  defineMacro("__UINT_FAST32_FMTo__", "\"o\"");
  defineMacro("__UINT_FAST32_FMTu__", "\"u\"");
  defineMacro("__UINT_FAST32_FMTx__", "\"x\"");
  defineMacro("__UINT_FAST32_MAX__", "4294967295U");
  defineMacro("__UINT_FAST32_TYPE__ unsigned", "int");
  defineMacro("__UINT_FAST64_FMTX__", "\"llX\"");
  defineMacro("__UINT_FAST64_FMTo__", "\"llo\"");
  defineMacro("__UINT_FAST64_FMTu__", "\"llu\"");
  defineMacro("__UINT_FAST64_FMTx__", "\"llx\"");
  defineMacro("__UINT_FAST64_MAX__", "18446744073709551615ULL");
  defineMacro("__UINT_FAST64_TYPE__ long long unsigned", "int");
  defineMacro("__UINT_FAST8_FMTX__", "\"hhX\"");
  defineMacro("__UINT_FAST8_FMTo__", "\"hho\"");
  defineMacro("__UINT_FAST8_FMTu__", "\"hhu\"");
  defineMacro("__UINT_FAST8_FMTx__", "\"hhx\"");
  defineMacro("__UINT_FAST8_MAX__", "255");
  defineMacro("__UINT_FAST8_TYPE__ unsigned", "char");
  defineMacro("__UINT_LEAST16_FMTX__", "\"hX\"");
  defineMacro("__UINT_LEAST16_FMTo__", "\"ho\"");
  defineMacro("__UINT_LEAST16_FMTu__", "\"hu\"");
  defineMacro("__UINT_LEAST16_FMTx__", "\"hx\"");
  defineMacro("__UINT_LEAST16_MAX__", "65535");
  defineMacro("__UINT_LEAST16_TYPE__ unsigned", "short");
  defineMacro("__UINT_LEAST32_FMTX__", "\"X\"");
  defineMacro("__UINT_LEAST32_FMTo__", "\"o\"");
  defineMacro("__UINT_LEAST32_FMTu__", "\"u\"");
  defineMacro("__UINT_LEAST32_FMTx__", "\"x\"");
  defineMacro("__UINT_LEAST32_MAX__", "4294967295U");
  defineMacro("__UINT_LEAST32_TYPE__ unsigned", "int");
  defineMacro("__UINT_LEAST64_FMTX__", "\"llX\"");
  defineMacro("__UINT_LEAST64_FMTo__", "\"llo\"");
  defineMacro("__UINT_LEAST64_FMTu__", "\"llu\"");
  defineMacro("__UINT_LEAST64_FMTx__", "\"llx\"");
  defineMacro("__UINT_LEAST64_MAX__", "18446744073709551615ULL");
  defineMacro("__UINT_LEAST64_TYPE__ long long unsigned", "int");
  defineMacro("__UINT_LEAST8_FMTX__", "\"hhX\"");
  defineMacro("__UINT_LEAST8_FMTo__", "\"hho\"");
  defineMacro("__UINT_LEAST8_FMTu__", "\"hhu\"");
  defineMacro("__UINT_LEAST8_FMTx__", "\"hhx\"");
  defineMacro("__UINT_LEAST8_MAX__", "255");
  defineMacro("__UINT_LEAST8_TYPE__ unsigned", "char");
  defineMacro("__USER_LABEL_PREFIX__", "_");
  defineMacro("__WCHAR_MAX__", "2147483647");
  defineMacro("__WCHAR_TYPE__", "int");
  defineMacro("__WCHAR_WIDTH__", "32");
  defineMacro("__WINT_MAX__", "2147483647");
  defineMacro("__WINT_TYPE__", "int");
  defineMacro("__WINT_WIDTH__", "32");
  defineMacro("__amd64", "1");
  defineMacro("__amd64__", "1");
  defineMacro("__apple_build_version__", "12000032");
  defineMacro("__clang__", "1");
  defineMacro("__clang_major__", "12");
  defineMacro("__clang_minor__", "0");
  defineMacro("__clang_patchlevel__", "0");
  defineMacro("__code_model_small_", "1");
  defineMacro("__core2", "1");
  defineMacro("__core2__", "1");
  defineMacro("__cplusplus", "201707L");
  defineMacro("__cpp_aggregate_bases", "201603L");
  defineMacro("__cpp_aggregate_nsdmi", "201304L");
  defineMacro("__cpp_alias_templates", "200704L");
  defineMacro("__cpp_aligned_new", "201606L");
  defineMacro("__cpp_attributes", "200809L");
  defineMacro("__cpp_binary_literals", "201304L");
  defineMacro("__cpp_capture_star_this", "201603L");
  defineMacro("__cpp_char8_t", "201811L");
  defineMacro("__cpp_conditional_explicit", "201806L");
  defineMacro("__cpp_constexpr", "201907L");
  defineMacro("__cpp_constexpr_dynamic_alloc", "201907L");
  defineMacro("__cpp_constexpr_in_decltype", "201711L");
  defineMacro("__cpp_constinit", "201907L");
  defineMacro("__cpp_coroutines", "201703L");
  defineMacro("__cpp_decltype", "200707L");
  defineMacro("__cpp_decltype_auto", "201304L");
  defineMacro("__cpp_deduction_guides", "201703L");
  defineMacro("__cpp_delegating_constructors", "200604L");
  defineMacro("__cpp_designated_initializers", "201707L");
  defineMacro("__cpp_digit_separators", "201309L");
  defineMacro("__cpp_enumerator_attributes", "201411L");
  defineMacro("__cpp_exceptions", "199711L");
  defineMacro("__cpp_fold_expressions", "201603L");
  defineMacro("__cpp_generic_lambdas", "201707L");
  defineMacro("__cpp_guaranteed_copy_elision", "201606L");
  defineMacro("__cpp_hex_float", "201603L");
  defineMacro("__cpp_if_constexpr", "201606L");
  defineMacro("__cpp_impl_destroying_delete", "201806L");
  defineMacro("__cpp_impl_three_way_comparison", "201907L");
  defineMacro("__cpp_inheriting_constructors", "201511L");
  defineMacro("__cpp_init_captures", "201803L");
  defineMacro("__cpp_initializer_lists", "200806L");
  defineMacro("__cpp_inline_variables", "201606L");
  defineMacro("__cpp_lambdas", "200907L");
  defineMacro("__cpp_namespace_attributes", "201411L");
  defineMacro("__cpp_nested_namespace_definitions", "201411L");
  defineMacro("__cpp_noexcept_function_type", "201510L");
  defineMacro("__cpp_nontype_template_args", "201411L");
  defineMacro("__cpp_nontype_template_parameter_auto", "201606L");
  defineMacro("__cpp_nsdmi", "200809L");
  defineMacro("__cpp_range_based_for", "201603L");
  defineMacro("__cpp_raw_strings", "200710L");
  defineMacro("__cpp_ref_qualifiers", "200710L");
  defineMacro("__cpp_return_type_deduction", "201304L");
  defineMacro("__cpp_rtti", "199711L");
  defineMacro("__cpp_rvalue_references", "200610L");
  defineMacro("__cpp_static_assert", "201411L");
  defineMacro("__cpp_structured_bindings", "201606L");
  defineMacro("__cpp_template_auto", "201606L");
  defineMacro("__cpp_threadsafe_static_init", "200806L");
  defineMacro("__cpp_unicode_characters", "200704L");
  defineMacro("__cpp_unicode_literals", "200710L");
  defineMacro("__cpp_user_defined_literals", "200809L");
  defineMacro("__cpp_variable_templates", "201304L");
  defineMacro("__cpp_variadic_templates", "200704L");
  defineMacro("__cpp_variadic_using", "201611L");
  defineMacro("__llvm__", "1");
  defineMacro("__nonnull", "_Nonnull");
  defineMacro("__null_unspecified", "_Null_unspecified");
  defineMacro("__nullable", "_Nullable");
  defineMacro("__pic__", "2");
  defineMacro("__private_extern__", "extern");
  defineMacro("__seg_fs", "__attribute__((address_space(257)))");
  defineMacro("__seg_gs", "__attribute__((address_space(256)))");
  defineMacro("__strong", "");
  defineMacro("__tune_core2__", "1");
  defineMacro("__unsafe_unretained", "");
  defineMacro("__weak", "__attribute__((objc_gc(weak)))");
  defineMacro("__x86_64", "1");
  defineMacro("__x86_64__", "1");
  // clang-format on
}

}  // namespace cxx
