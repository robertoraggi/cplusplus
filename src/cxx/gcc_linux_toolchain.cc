// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/gcc_linux_toolchain.h>
#include <cxx/preprocessor.h>
#include <fmt/format.h>

#include <filesystem>

namespace cxx {

GCCLinuxToolchain::GCCLinuxToolchain(Preprocessor* preprocessor)
    : Toolchain(preprocessor) {
  for (int version : {12, 11, 10, 9}) {
    const auto path = std::filesystem::path(
        fmt::format("/usr/lib/gcc/x86_64-linux-gnu/{}/include", version));

    if (exists(path)) {
      version_ = version;
      break;
    }
  }
}

void GCCLinuxToolchain::addSystemIncludePaths() {
  auto addSystemIncludePathForGCCVersion = [this](int version) {
    addSystemIncludePath(
        fmt::format("/usr/lib/gcc/x86_64-linux-gnu/{}/include", version));
  };

  addSystemIncludePath("/usr/include");
  addSystemIncludePath("/usr/include/x86_64-linux-gnu");
  addSystemIncludePath("/usr/local/include");

  if (version_) addSystemIncludePathForGCCVersion(*version_);
}

void GCCLinuxToolchain::addSystemCppIncludePaths() {
  auto addSystemIncludePathForGCCVersion = [this](int version) {
    addSystemIncludePath(fmt::format("/usr/include/c++/{}/backward", version));

    addSystemIncludePath(
        fmt::format("/usr/include/x86_64-linux-gnu/c++/{}", version));

    addSystemIncludePath(fmt::format("/usr/include/c++/{}", version));
  };

  if (version_) addSystemIncludePathForGCCVersion(*version_);
}

void GCCLinuxToolchain::addPredefinedMacros() {
  // clang-format off
  defineMacro("__extension__", "");
  defineMacro("__null", "nullptr");
  defineMacro("__signed__", "signed");
  defineMacro("_Pragma(x)", "");
  defineMacro("__amd64", "1");
  defineMacro("__amd64__", "1");
  defineMacro("__ATOMIC_ACQ_REL", "4");
  defineMacro("__ATOMIC_ACQUIRE", "2");
  defineMacro("__ATOMIC_CONSUME", "1");
  defineMacro("__ATOMIC_HLE_ACQUIRE", "65536");
  defineMacro("__ATOMIC_HLE_RELEASE", "131072");
  defineMacro("__ATOMIC_RELAXED", "0");
  defineMacro("__ATOMIC_RELEASE", "3");
  defineMacro("__ATOMIC_SEQ_CST", "5");
  defineMacro("__BIGGEST_ALIGNMENT__", "16");
  defineMacro("__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__");
  defineMacro("__CET__", "3");
  defineMacro("__CHAR_BIT__", "8");
  defineMacro("__CHAR16_TYPE__", "short unsigned int");
  defineMacro("__CHAR32_TYPE__", "unsigned int");
  defineMacro("__CHAR8_TYPE__", "unsigned char");
  defineMacro("__code_model_small__", "1");
  defineMacro("__cplusplus", "201709L");
  defineMacro("__cpp_aggregate_bases", "201603");
  defineMacro("__cpp_aggregate_nsdmi", "201304");
  defineMacro("__cpp_alias_templates", "200704");
  defineMacro("__cpp_aligned_new", "201606");
  defineMacro("__cpp_attributes", "200809");
  defineMacro("__cpp_binary_literals", "201304");
  defineMacro("__cpp_capture_star_this", "201603");
  defineMacro("__cpp_char8_t", "201811");
  defineMacro("__cpp_conditional_explicit", "201806");
  defineMacro("__cpp_constexpr", "201603");
  defineMacro("__cpp_decltype", "200707");
  defineMacro("__cpp_decltype_auto", "201304");
  defineMacro("__cpp_deduction_guides", "201703");
  defineMacro("__cpp_delegating_constructors", "200604");
  defineMacro("__cpp_digit_separators", "201309");
  defineMacro("__cpp_enumerator_attributes", "201411");
  defineMacro("__cpp_exceptions", "199711");
  defineMacro("__cpp_fold_expressions", "201603");
  defineMacro("__cpp_generic_lambdas", "201304");
  defineMacro("__cpp_guaranteed_copy_elision", "201606");
  defineMacro("__cpp_hex_float", "201603");
  defineMacro("__cpp_if_constexpr", "201606");
  defineMacro("__cpp_impl_destroying_delete", "201806");
  defineMacro("__cpp_inheriting_constructors", "201511");
  defineMacro("__cpp_init_captures", "201304");
  defineMacro("__cpp_initializer_lists", "200806");
  defineMacro("__cpp_inline_variables", "201606");
  defineMacro("__cpp_lambdas", "200907");
  defineMacro("__cpp_namespace_attributes", "201411");
  defineMacro("__cpp_nested_namespace_definitions", "201411");
  defineMacro("__cpp_noexcept_function_type", "201510");
  defineMacro("__cpp_nontype_template_args", "201411");
  defineMacro("__cpp_nontype_template_parameter_auto", "201606");
  defineMacro("__cpp_nontype_template_parameter_class", "201806");
  defineMacro("__cpp_nsdmi", "200809");
  defineMacro("__cpp_range_based_for", "201603");
  defineMacro("__cpp_raw_strings", "200710");
  defineMacro("__cpp_ref_qualifiers", "200710");
  defineMacro("__cpp_return_type_deduction", "201304");
  defineMacro("__cpp_rtti", "199711");
  defineMacro("__cpp_runtime_arrays", "198712");
  defineMacro("__cpp_rvalue_reference", "200610");
  defineMacro("__cpp_rvalue_references", "200610");
  defineMacro("__cpp_sized_deallocation", "201309");
  defineMacro("__cpp_static_assert", "201411");
  defineMacro("__cpp_structured_bindings", "201606");
  defineMacro("__cpp_template_auto", "201606");
  defineMacro("__cpp_template_template_args", "201611");
  defineMacro("__cpp_threadsafe_static_init", "200806");
  defineMacro("__cpp_unicode_characters", "201411");
  defineMacro("__cpp_unicode_literals", "200710");
  defineMacro("__cpp_user_defined_literals", "200809");
  defineMacro("__cpp_variable_templates", "201304");
  defineMacro("__cpp_variadic_templates", "200704");
  defineMacro("__cpp_variadic_using", "201611");
  defineMacro("__DBL_DECIMAL_DIG__", "17");
  defineMacro("__DBL_DENORM_MIN__", "double(4.94065645841246544176568792868221372e-324L)");
  defineMacro("__DBL_DIG__", "15");
  defineMacro("__DBL_EPSILON__", "double(2.22044604925031308084726333618164062e-16L)");
  defineMacro("__DBL_HAS_DENORM__", "1");
  defineMacro("__DBL_HAS_INFINITY__", "1");
  defineMacro("__DBL_HAS_QUIET_NAN__", "1");
  defineMacro("__DBL_MANT_DIG__", "53");
  defineMacro("__DBL_MAX__", "double(1.79769313486231570814527423731704357e+308L)");
  defineMacro("__DBL_MAX_10_EXP__", "308");
  defineMacro("__DBL_MAX_EXP__", "1024");
  defineMacro("__DBL_MIN__", "double(2.22507385850720138309023271733240406e-308L)");
  defineMacro("__DBL_MIN_10_EXP__", "(-307)");
  defineMacro("__DBL_MIN_EXP__", "(-1021)");
  defineMacro("__DEC_EVAL_METHOD__", "2");
  defineMacro("__DEC128_EPSILON__", "1E-33DL");
  defineMacro("__DEC128_MANT_DIG__", "34");
  defineMacro("__DEC128_MAX__", "9.999999999999999999999999999999999E6144DL");
  defineMacro("__DEC128_MAX_EXP__", "6145");
  defineMacro("__DEC128_MIN__", "1E-6143DL");
  defineMacro("__DEC128_MIN_EXP__", "(-6142)");
  defineMacro("__DEC128_SUBNORMAL_MIN__", "0.000000000000000000000000000000001E-6143DL");
  defineMacro("__DEC32_EPSILON__", "1E-6DF");
  defineMacro("__DEC32_MANT_DIG__", "7");
  defineMacro("__DEC32_MAX__", "9.999999E96DF");
  defineMacro("__DEC32_MAX_EXP__", "97");
  defineMacro("__DEC32_MIN__", "1E-95DF");
  defineMacro("__DEC32_MIN_EXP__", "(-94)");
  defineMacro("__DEC32_SUBNORMAL_MIN__", "0.000001E-95DF");
  defineMacro("__DEC64_EPSILON__", "1E-15DD");
  defineMacro("__DEC64_MANT_DIG__", "16");
  defineMacro("__DEC64_MAX__", "9.999999999999999E384DD");
  defineMacro("__DEC64_MAX_EXP__", "385");
  defineMacro("__DEC64_MIN__", "1E-383DD");
  defineMacro("__DEC64_MIN_EXP__", "(-382)");
  defineMacro("__DEC64_SUBNORMAL_MIN__", "0.000000000000001E-383DD");
  defineMacro("__DECIMAL_BID_FORMAT__", "1");
  defineMacro("__DECIMAL_DIG__", "21");
  defineMacro("__DEPRECATED", "1");
  defineMacro("__ELF__", "1");
  defineMacro("__EXCEPTIONS", "1");
  defineMacro("__FINITE_MATH_ONLY__", "0");
  defineMacro("__FLOAT_WORD_ORDER__", "__ORDER_LITTLE_ENDIAN__");
  defineMacro("__FLT_DECIMAL_DIG__", "9");
  defineMacro("__FLT_DENORM_MIN__", "1.40129846432481707092372958328991613e-45F");
  defineMacro("__FLT_DIG__", "6");
  defineMacro("__FLT_EPSILON__", "1.19209289550781250000000000000000000e-7F");
  defineMacro("__FLT_EVAL_METHOD__", "0");
  defineMacro("__FLT_EVAL_METHOD_TS_18661_3__", "0");
  defineMacro("__FLT_HAS_DENORM__", "1");
  defineMacro("__FLT_HAS_INFINITY__", "1");
  defineMacro("__FLT_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT_MANT_DIG__", "24");
  defineMacro("__FLT_MAX__", "3.40282346638528859811704183484516925e+38F");
  defineMacro("__FLT_MAX_10_EXP__", "38");
  defineMacro("__FLT_MAX_EXP__", "128");
  defineMacro("__FLT_MIN__", "1.17549435082228750796873653722224568e-38F");
  defineMacro("__FLT_MIN_10_EXP__", "(-37)");
  defineMacro("__FLT_MIN_EXP__", "(-125)");
  defineMacro("__FLT_RADIX__", "2");
  defineMacro("__FLT128_DECIMAL_DIG__", "36");
  defineMacro("__FLT128_DENORM_MIN__", "6.47517511943802511092443895822764655e-4966F128");
  defineMacro("__FLT128_DIG__", "33");
  defineMacro("__FLT128_EPSILON__", "1.92592994438723585305597794258492732e-34F128");
  defineMacro("__FLT128_HAS_DENORM__", "1");
  defineMacro("__FLT128_HAS_INFINITY__", "1");
  defineMacro("__FLT128_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT128_MANT_DIG__", "113");
  defineMacro("__FLT128_MAX__", "1.18973149535723176508575932662800702e+4932F128");
  defineMacro("__FLT128_MAX_10_EXP__", "4932");
  defineMacro("__FLT128_MAX_EXP__", "16384");
  defineMacro("__FLT128_MIN__", "3.36210314311209350626267781732175260e-4932F128");
  defineMacro("__FLT128_MIN_10_EXP__", "(-4931)");
  defineMacro("__FLT128_MIN_EXP__", "(-16381)");
  defineMacro("__FLT32_DECIMAL_DIG__", "9");
  defineMacro("__FLT32_DENORM_MIN__", "1.40129846432481707092372958328991613e-45F32");
  defineMacro("__FLT32_DIG__", "6");
  defineMacro("__FLT32_EPSILON__", "1.19209289550781250000000000000000000e-7F32");
  defineMacro("__FLT32_HAS_DENORM__", "1");
  defineMacro("__FLT32_HAS_INFINITY__", "1");
  defineMacro("__FLT32_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT32_MANT_DIG__", "24");
  defineMacro("__FLT32_MAX__", "3.40282346638528859811704183484516925e+38F32");
  defineMacro("__FLT32_MAX_10_EXP__", "38");
  defineMacro("__FLT32_MAX_EXP__", "128");
  defineMacro("__FLT32_MIN__", "1.17549435082228750796873653722224568e-38F32");
  defineMacro("__FLT32_MIN_10_EXP__", "(-37)");
  defineMacro("__FLT32_MIN_EXP__", "(-125)");
  defineMacro("__FLT32X_DECIMAL_DIG__", "17");
  defineMacro("__FLT32X_DENORM_MIN__", "4.94065645841246544176568792868221372e-324F32x");
  defineMacro("__FLT32X_DIG__", "15");
  defineMacro("__FLT32X_EPSILON__", "2.22044604925031308084726333618164062e-16F32x");
  defineMacro("__FLT32X_HAS_DENORM__", "1");
  defineMacro("__FLT32X_HAS_INFINITY__", "1");
  defineMacro("__FLT32X_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT32X_MANT_DIG__", "53");
  defineMacro("__FLT32X_MAX__", "1.79769313486231570814527423731704357e+308F32x");
  defineMacro("__FLT32X_MAX_10_EXP__", "308");
  defineMacro("__FLT32X_MAX_EXP__", "1024");
  defineMacro("__FLT32X_MIN__", "2.22507385850720138309023271733240406e-308F32x");
  defineMacro("__FLT32X_MIN_10_EXP__", "(-307)");
  defineMacro("__FLT32X_MIN_EXP__", "(-1021)");
  defineMacro("__FLT64_DECIMAL_DIG__", "17");
  defineMacro("__FLT64_DENORM_MIN__", "4.94065645841246544176568792868221372e-324F64");
  defineMacro("__FLT64_DIG__", "15");
  defineMacro("__FLT64_EPSILON__", "2.22044604925031308084726333618164062e-16F64");
  defineMacro("__FLT64_HAS_DENORM__", "1");
  defineMacro("__FLT64_HAS_INFINITY__", "1");
  defineMacro("__FLT64_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT64_MANT_DIG__", "53");
  defineMacro("__FLT64_MAX__", "1.79769313486231570814527423731704357e+308F64");
  defineMacro("__FLT64_MAX_10_EXP__", "308");
  defineMacro("__FLT64_MAX_EXP__", "1024");
  defineMacro("__FLT64_MIN__", "2.22507385850720138309023271733240406e-308F64");
  defineMacro("__FLT64_MIN_10_EXP__", "(-307)");
  defineMacro("__FLT64_MIN_EXP__", "(-1021)");
  defineMacro("__FLT64X_DECIMAL_DIG__", "21");
  defineMacro("__FLT64X_DENORM_MIN__", "3.64519953188247460252840593361941982e-4951F64x");
  defineMacro("__FLT64X_DIG__", "18");
  defineMacro("__FLT64X_EPSILON__", "1.08420217248550443400745280086994171e-19F64x");
  defineMacro("__FLT64X_HAS_DENORM__", "1");
  defineMacro("__FLT64X_HAS_INFINITY__", "1");
  defineMacro("__FLT64X_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT64X_MANT_DIG__", "64");
  defineMacro("__FLT64X_MAX__", "1.18973149535723176502126385303097021e+4932F64x");
  defineMacro("__FLT64X_MAX_10_EXP__", "4932");
  defineMacro("__FLT64X_MAX_EXP__", "16384");
  defineMacro("__FLT64X_MIN__", "3.36210314311209350626267781732175260e-4932F64x");
  defineMacro("__FLT64X_MIN_10_EXP__", "(-4931)");
  defineMacro("__FLT64X_MIN_EXP__", "(-16381)");
  defineMacro("__FXSR__", "1");
  defineMacro("__GCC_ASM_FLAG_OUTPUTS__", "1");
  defineMacro("__GCC_ATOMIC_BOOL_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR16_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR32_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR8_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_INT_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_LLONG_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_LONG_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_POINTER_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_SHORT_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_TEST_AND_SET_TRUEVAL", "1");
  defineMacro("__GCC_ATOMIC_WCHAR_T_LOCK_FREE", "2");
  defineMacro("__GCC_HAVE_DWARF2_CFI_ASM", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8", "1");
  defineMacro("__GCC_IEC_559", "2");
  defineMacro("__GCC_IEC_559_COMPLEX", "2");
  defineMacro("__gnu_linux__", "1");
  defineMacro("__GNUC__", "9");
  defineMacro("__GNUC_MINOR__", "3");
  defineMacro("__GNUC_PATCHLEVEL__", "0");
  defineMacro("__GNUC_STDC_INLINE__", "1");
  defineMacro("__GNUG__", "9");
  defineMacro("__GXX_ABI_VERSION", "1013");
  defineMacro("__GXX_EXPERIMENTAL_CXX0X__", "1");
  defineMacro("__GXX_RTTI", "1");
  defineMacro("__GXX_WEAK__", "1");
  defineMacro("__has_include_next(STR)", "__has_include_next__(STR)");
  defineMacro("__has_include(STR)", "__has_include__(STR)");
  defineMacro("__HAVE_SPECULATION_SAFE_VALUE", "1");
  defineMacro("__INT_FAST16_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INT_FAST16_TYPE__", "long int");
  defineMacro("__INT_FAST16_WIDTH__", "64");
  defineMacro("__INT_FAST32_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INT_FAST32_TYPE__", "long int");
  defineMacro("__INT_FAST32_WIDTH__", "64");
  defineMacro("__INT_FAST64_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INT_FAST64_TYPE__", "long int");
  defineMacro("__INT_FAST64_WIDTH__", "64");
  defineMacro("__INT_FAST8_MAX__", "0x7f");
  defineMacro("__INT_FAST8_TYPE__", "signed char");
  defineMacro("__INT_FAST8_WIDTH__", "8");
  defineMacro("__INT_LEAST16_MAX__", "0x7fff");
  defineMacro("__INT_LEAST16_TYPE__", "short int");
  defineMacro("__INT_LEAST16_WIDTH__", "16");
  defineMacro("__INT_LEAST32_MAX__", "0x7fffffff");
  defineMacro("__INT_LEAST32_TYPE__", "int");
  defineMacro("__INT_LEAST32_WIDTH__", "32");
  defineMacro("__INT_LEAST64_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INT_LEAST64_TYPE__", "long int");
  defineMacro("__INT_LEAST64_WIDTH__", "64");
  defineMacro("__INT_LEAST8_MAX__", "0x7f");
  defineMacro("__INT_LEAST8_TYPE__", "signed char");
  defineMacro("__INT_LEAST8_WIDTH__", "8");
  defineMacro("__INT_MAX__", "0x7fffffff");
  defineMacro("__INT_WIDTH__", "32");
  defineMacro("__INT16_C(c)", "c");
  defineMacro("__INT16_MAX__", "0x7fff");
  defineMacro("__INT16_TYPE__", "short int");
  defineMacro("__INT32_C(c)", "c");
  defineMacro("__INT32_MAX__", "0x7fffffff");
  defineMacro("__INT32_TYPE__", "int");
  defineMacro("__INT64_C(c)", "c ## L");
  defineMacro("__INT64_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INT64_TYPE__", "long int");
  defineMacro("__INT8_C(c)", "c");
  defineMacro("__INT8_MAX__", "0x7f");
  defineMacro("__INT8_TYPE__", "signed char");
  defineMacro("__INTMAX_C(c)", "c ## L");
  defineMacro("__INTMAX_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INTMAX_TYPE__", "long int");
  defineMacro("__INTMAX_WIDTH__", "64");
  defineMacro("__INTPTR_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INTPTR_TYPE__", "long int");
  defineMacro("__INTPTR_WIDTH__", "64");
  defineMacro("__k8", "1");
  defineMacro("__k8__", "1");
  defineMacro("__LDBL_DECIMAL_DIG__", "21");
  defineMacro("__LDBL_DENORM_MIN__", "3.64519953188247460252840593361941982e-4951L");
  defineMacro("__LDBL_DIG__", "18");
  defineMacro("__LDBL_EPSILON__", "1.08420217248550443400745280086994171e-19L");
  defineMacro("__LDBL_HAS_DENORM__", "1");
  defineMacro("__LDBL_HAS_INFINITY__", "1");
  defineMacro("__LDBL_HAS_QUIET_NAN__", "1");
  defineMacro("__LDBL_MANT_DIG__", "64");
  defineMacro("__LDBL_MAX__", "1.18973149535723176502126385303097021e+4932L");
  defineMacro("__LDBL_MAX_10_EXP__", "4932");
  defineMacro("__LDBL_MAX_EXP__", "16384");
  defineMacro("__LDBL_MIN__", "3.36210314311209350626267781732175260e-4932L");
  defineMacro("__LDBL_MIN_10_EXP__", "(-4931)");
  defineMacro("__LDBL_MIN_EXP__", "(-16381)");
  defineMacro("__linux", "1");
  defineMacro("__linux__", "1");
  defineMacro("__LONG_LONG_MAX__", "0x7fffffffffffffffLL");
  defineMacro("__LONG_LONG_WIDTH__", "64");
  defineMacro("__LONG_MAX__", "0x7fffffffffffffffL");
  defineMacro("__LONG_WIDTH__", "64");
  defineMacro("__LP64__", "1");
  defineMacro("__MMX__", "1");
  defineMacro("__NO_INLINE__", "1");
  defineMacro("__ORDER_BIG_ENDIAN__", "4321");
  defineMacro("__ORDER_LITTLE_ENDIAN__", "1234");
  defineMacro("__ORDER_PDP_ENDIAN__", "3412");
  defineMacro("__pic__", "2");
  defineMacro("__PIC__", "2");
  defineMacro("__pie__", "2");
  defineMacro("__PIE__", "2");
  defineMacro("__PRAGMA_REDEFINE_EXTNAME", "1");
  defineMacro("__PTRDIFF_MAX__", "0x7fffffffffffffffL");
  defineMacro("__PTRDIFF_TYPE__", "long int");
  defineMacro("__PTRDIFF_WIDTH__", "64");
  defineMacro("__REGISTER_PREFIX__", "");
  defineMacro("__SCHAR_MAX__", "0x7f");
  defineMacro("__SCHAR_WIDTH__", "8");
  defineMacro("__SEG_FS", "1");
  defineMacro("__SEG_GS", "1");
  defineMacro("__SHRT_MAX__", "0x7fff");
  defineMacro("__SHRT_WIDTH__", "16");
  defineMacro("__SIG_ATOMIC_MAX__", "0x7fffffff");
  defineMacro("__SIG_ATOMIC_MIN__", "(-__SIG_ATOMIC_MAX__ - 1)");
  defineMacro("__SIG_ATOMIC_TYPE__", "int");
  defineMacro("__SIG_ATOMIC_WIDTH__", "32");
  defineMacro("__SIZE_MAX__", "0xffffffffffffffffUL");
  defineMacro("__SIZE_TYPE__", "long unsigned int");
  defineMacro("__SIZE_WIDTH__", "64");
  defineMacro("__SIZEOF_DOUBLE__", "8");
  defineMacro("__SIZEOF_FLOAT__", "4");
  defineMacro("__SIZEOF_FLOAT128__", "16");
  defineMacro("__SIZEOF_FLOAT80__", "16");
  defineMacro("__SIZEOF_INT__", "4");
  defineMacro("__SIZEOF_INT128__", "16");
  defineMacro("__SIZEOF_LONG__", "8");
  defineMacro("__SIZEOF_LONG_DOUBLE__", "16");
  defineMacro("__SIZEOF_LONG_LONG__", "8");
  defineMacro("__SIZEOF_POINTER__", "8");
  defineMacro("__SIZEOF_PTRDIFF_T__", "8");
  defineMacro("__SIZEOF_SHORT__", "2");
  defineMacro("__SIZEOF_SIZE_T__", "8");
  defineMacro("__SIZEOF_WCHAR_T__", "4");
  defineMacro("__SIZEOF_WINT_T__", "4");
  defineMacro("__SSE__", "1");
  defineMacro("__SSE_MATH__", "1");
  defineMacro("__SSE2__", "1");
  defineMacro("__SSE2_MATH__", "1");
  defineMacro("__SSP_STRONG__", "3");
  defineMacro("__STDC__", "1");
  defineMacro("__STDC_HOSTED__", "1");
  defineMacro("__STDC_IEC_559__", "1");
  defineMacro("__STDC_IEC_559_COMPLEX__", "1");
  defineMacro("__STDC_ISO_10646__", "201706L");
  defineMacro("__STDC_UTF_16__", "1");
  defineMacro("__STDC_UTF_32__", "1");
  defineMacro("__STDCPP_DEFAULT_NEW_ALIGNMENT__", "16");
  defineMacro("__STRICT_ANSI__", "1");
  defineMacro("__UINT_FAST16_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINT_FAST16_TYPE__", "long unsigned int");
  defineMacro("__UINT_FAST32_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINT_FAST32_TYPE__", "long unsigned int");
  defineMacro("__UINT_FAST64_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINT_FAST64_TYPE__", "long unsigned int");
  defineMacro("__UINT_FAST8_MAX__", "0xff");
  defineMacro("__UINT_FAST8_TYPE__", "unsigned char");
  defineMacro("__UINT_LEAST16_MAX__", "0xffff");
  defineMacro("__UINT_LEAST16_TYPE__", "short unsigned int");
  defineMacro("__UINT_LEAST32_MAX__", "0xffffffffU");
  defineMacro("__UINT_LEAST32_TYPE__", "unsigned int");
  defineMacro("__UINT_LEAST64_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINT_LEAST64_TYPE__", "long unsigned int");
  defineMacro("__UINT_LEAST8_MAX__", "0xff");
  defineMacro("__UINT_LEAST8_TYPE__", "unsigned char");
  defineMacro("__UINT16_C(c)", "c");
  defineMacro("__UINT16_MAX__", "0xffff");
  defineMacro("__UINT16_TYPE__", "short unsigned int");
  defineMacro("__UINT32_C(c)", "c ## U");
  defineMacro("__UINT32_MAX__", "0xffffffffU");
  defineMacro("__UINT32_TYPE__", "unsigned int");
  defineMacro("__UINT64_C(c)", "c ## UL");
  defineMacro("__UINT64_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINT64_TYPE__", "long unsigned int");
  defineMacro("__UINT8_C(c)", "c");
  defineMacro("__UINT8_MAX__", "0xff");
  defineMacro("__UINT8_TYPE__", "unsigned char");
  defineMacro("__UINTMAX_C(c)", "c ## UL");
  defineMacro("__UINTMAX_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINTMAX_TYPE__", "long unsigned int");
  defineMacro("__UINTPTR_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINTPTR_TYPE__", "long unsigned int");
  defineMacro("__unix", "1");
  defineMacro("__unix__", "1");
  defineMacro("__USER_LABEL_PREFIX__", "");
  defineMacro("__VERSION__", "\"9.3.0\"");
  defineMacro("__WCHAR_MAX__", "0x7fffffff");
  defineMacro("__WCHAR_MIN__", "(-__WCHAR_MAX__ - 1)");
  defineMacro("__WCHAR_TYPE__", "int");
  defineMacro("__WCHAR_WIDTH__", "32");
  defineMacro("__WINT_MAX__", "0xffffffffU");
  defineMacro("__WINT_MIN__", "0U");
  defineMacro("__WINT_TYPE__", "unsigned int");
  defineMacro("__WINT_WIDTH__", "32");
  defineMacro("__x86_64", "1");
  defineMacro("__x86_64__", "1");
  defineMacro("_GNU_SOURCE", "1");
  defineMacro("_LP64", "1");
  defineMacro("_STDC_PREDEF_H", "1");
  // clang-format on
}

}  // namespace cxx
