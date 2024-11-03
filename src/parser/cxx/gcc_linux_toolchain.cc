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

#include <cxx/gcc_linux_toolchain.h>
#include <cxx/preprocessor.h>
#include <cxx/private/path.h>

#include <format>

namespace cxx {

GCCLinuxToolchain::GCCLinuxToolchain(Preprocessor* preprocessor,
                                     std::string arch)
    : Toolchain(preprocessor), arch_(std::move(arch)) {
  for (int version : {14, 13, 12, 11, 10, 9}) {
    const auto path = fs::path(
        std::format("/usr/lib/gcc/{}-linux-gnu/{}/include", arch_, version));

    if (exists(path)) {
      version_ = version;
      break;
    }
  }
}

void GCCLinuxToolchain::addSystemIncludePaths() {
  auto addSystemIncludePathForGCCVersion = [this](int version) {
    addSystemIncludePath(
        std::format("/usr/lib/gcc/{}-linux-gnu/{}/include", arch_, version));
  };

  addSystemIncludePath("/usr/include");
  addSystemIncludePath(std::format("/usr/include/{}-linux-gnu", arch_));
  addSystemIncludePath("/usr/local/include");

  if (version_) addSystemIncludePathForGCCVersion(*version_);
}

void GCCLinuxToolchain::addSystemCppIncludePaths() {
  auto addSystemIncludePathForGCCVersion = [this](int version) {
    addSystemIncludePath(std::format("/usr/include/c++/{}/backward", version));

    addSystemIncludePath(
        std::format("/usr/include/{}-linux-gnu/c++/{}", arch_, version));

    addSystemIncludePath(std::format("/usr/include/c++/{}", version));
  };

  if (version_) addSystemIncludePathForGCCVersion(*version_);
}

void GCCLinuxToolchain::addPredefinedMacros() {
  defineMacro("__extension__", "");
  defineMacro("__null", "nullptr");
  defineMacro("__restrict__", "");
  defineMacro("__restrict", "");
  defineMacro("__signed__", "signed");
  defineMacro("_Nonnull", "");
  defineMacro("_Nullable", "");
  defineMacro("_Pragma(x)", "");

  // std=c++26
  defineMacro("__cplusplus", "202400L");

  addCommonMacros();
  addCxx26Macros();

  if (arch_ == "aarch64") {
    addArm64Macros();
  } else if (arch_ == "x86_64") {
    addAmd64Macros();
  }
}

// clang-format off
void GCCLinuxToolchain::addCommonMacros() {
  defineMacro("__unix__", "1");
  defineMacro("__unix", "1");
  defineMacro("__linux__", "1");
  defineMacro("__linux", "1");
  defineMacro("__gnu_linux__", "1");
  defineMacro("__cpp_variadic_using", "201611L");
  defineMacro("__cpp_variadic_templates", "200704L");
  defineMacro("__cpp_variable_templates", "201304L");
  defineMacro("__cpp_using_enum", "201907L");
  defineMacro("__cpp_user_defined_literals", "200809L");
  defineMacro("__cpp_unicode_literals", "200710L");
  defineMacro("__cpp_unicode_characters", "201411L");
  defineMacro("__cpp_threadsafe_static_init", "200806L");
  defineMacro("__cpp_template_template_args", "201611L");
  defineMacro("__cpp_template_auto", "201606L");
  defineMacro("__cpp_structured_bindings", "201606L");
  defineMacro("__cpp_sized_deallocation", "201309L");
  defineMacro("__cpp_rvalue_references", "200610L");
  defineMacro("__cpp_rvalue_reference", "200610L");
  defineMacro("__cpp_runtime_arrays", "198712L");
  defineMacro("__cpp_rtti", "199711L");
  defineMacro("__cpp_return_type_deduction", "201304L");
  defineMacro("__cpp_ref_qualifiers", "200710L");
  defineMacro("__cpp_raw_strings", "200710L");
  defineMacro("__cpp_range_based_for", "201603L");
  defineMacro("__cpp_nsdmi", "200809L");
  defineMacro("__cpp_nontype_template_parameter_class", "201806L");
  defineMacro("__cpp_nontype_template_parameter_auto", "201606L");
  defineMacro("__cpp_nontype_template_args", "201911L");
  defineMacro("__cpp_noexcept_function_type", "201510L");
  defineMacro("__cpp_nested_namespace_definitions", "201411L");
  defineMacro("__cpp_namespace_attributes", "201411L");
  defineMacro("__cpp_lambdas", "200907L");
  defineMacro("__cpp_inline_variables", "201606L");
  defineMacro("__cpp_initializer_lists", "200806L");
  defineMacro("__cpp_init_captures", "201803L");
  defineMacro("__cpp_inheriting_constructors", "201511L");
  defineMacro("__cpp_impl_three_way_comparison", "201907L");
  defineMacro("__cpp_impl_destroying_delete", "201806L");
  defineMacro("__cpp_impl_coroutine", "201902L");
  defineMacro("__cpp_if_constexpr", "201606L");
  defineMacro("__cpp_hex_float", "201603L");
  defineMacro("__cpp_guaranteed_copy_elision", "201606L");
  defineMacro("__cpp_generic_lambdas", "201707L");
  defineMacro("__cpp_fold_expressions", "201603L");
  defineMacro("__cpp_exceptions", "199711L");
  defineMacro("__cpp_enumerator_attributes", "201411L");
  defineMacro("__cpp_digit_separators", "201309L");
  defineMacro("__cpp_designated_initializers", "201707L");
  defineMacro("__cpp_delegating_constructors", "200604L");
  defineMacro("__cpp_deduction_guides", "201907L");
  defineMacro("__cpp_decltype_auto", "201304L");
  defineMacro("__cpp_decltype", "200707L");
  defineMacro("__cpp_constinit", "201907L");
  defineMacro("__cpp_constexpr_in_decltype", "201711L");
  defineMacro("__cpp_constexpr_dynamic_alloc", "201907L");
  defineMacro("__cpp_consteval", "202211L");
  defineMacro("__cpp_conditional_explicit", "201806L");
  defineMacro("__cpp_concepts", "202002L");
  defineMacro("__cpp_char8_t", "202207L");
  defineMacro("__cpp_capture_star_this", "201603L");
  defineMacro("__cpp_binary_literals", "201304L");
  defineMacro("__cpp_attributes", "200809L");
  defineMacro("__cpp_aligned_new", "201606L");
  defineMacro("__cpp_alias_templates", "200704L");
  defineMacro("__cpp_aggregate_paren_init", "201902L");
  defineMacro("__cpp_aggregate_nsdmi", "201304L");
  defineMacro("__cpp_aggregate_bases", "201603L");
  defineMacro("__WINT_WIDTH__", "32");
  defineMacro("__WINT_TYPE__", "unsigned");
  defineMacro("__WINT_MIN__", "0U");
  defineMacro("__WINT_MAX__", "0xffffffffU");
  defineMacro("__WCHAR_WIDTH__", "32");
  defineMacro("__VERSION__", "\"14.2.0\"");
  defineMacro("__USER_LABEL_PREFIX__", "");
  defineMacro("__UINT_LEAST8_TYPE__", "unsigned");
  defineMacro("__UINT_LEAST8_MAX__", "0xff");
  defineMacro("__UINT_LEAST64_TYPE__", "long");
  defineMacro("__UINT_LEAST64_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINT_LEAST32_TYPE__", "unsigned");
  defineMacro("__UINT_LEAST32_MAX__", "0xffffffffU");
  defineMacro("__UINT_LEAST16_TYPE__", "short");
  defineMacro("__UINT_LEAST16_MAX__", "0xffff");
  defineMacro("__UINT_FAST8_TYPE__", "unsigned");
  defineMacro("__UINT_FAST8_MAX__", "0xff");
  defineMacro("__UINT_FAST64_TYPE__", "long");
  defineMacro("__UINT_FAST64_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINT_FAST32_TYPE__", "long");
  defineMacro("__UINT_FAST32_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINT_FAST16_TYPE__", "long");
  defineMacro("__UINT_FAST16_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINTPTR_TYPE__", "long");
  defineMacro("__UINTPTR_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINTMAX_TYPE__", "long");
  defineMacro("__UINTMAX_MAX__", "0xffffffffffffffffUL");
  defineMacro("__UINTMAX_C(c)", "c");
  defineMacro("__UINT8_TYPE__", "unsigned");
  defineMacro("__UINT8_MAX__", "0xff");
  defineMacro("__UINT8_C(c)", "c");
  defineMacro("__UINT64_TYPE__", "long");
  defineMacro("__UINT64_MAX__", "0xffffffffffffffffUL");
  // defineMacro("__UINT64_C(c)", "c");
  defineMacro("__UINT32_TYPE__", "unsigned");
  defineMacro("__UINT32_MAX__", "0xffffffffU");
  defineMacro("__UINT32_C(c)", "c");
  defineMacro("__UINT16_TYPE__", "short");
  defineMacro("__UINT16_MAX__", "0xffff");
  defineMacro("__UINT16_C(c)", "c");
  defineMacro("__STRICT_ANSI__", "1");
  defineMacro("__STDC__", "1");
  defineMacro("__STDC_UTF_32__", "1");
  defineMacro("__STDC_UTF_16__", "1");
  defineMacro("__STDC_ISO_10646__", "201706L");
  defineMacro("__STDC_IEC_60559_COMPLEX__", "201404L");
  defineMacro("__STDC_IEC_60559_BFP__", "201404L");
  defineMacro("__STDC_IEC_559__", "1");
  defineMacro("__STDC_IEC_559_COMPLEX__", "1");
  defineMacro("__STDC_HOSTED__", "1");
  defineMacro("__STDCPP_THREADS__", "1");
  defineMacro("__STDCPP_DEFAULT_NEW_ALIGNMENT__", "16");
  defineMacro("__SIZE_WIDTH__", "64");
  defineMacro("__SIZE_TYPE__", "long");
  defineMacro("__SIZE_MAX__", "0xffffffffffffffffUL");
  defineMacro("__SIZEOF_WINT_T__", "4");
  defineMacro("__SIZEOF_WCHAR_T__", "4");
  defineMacro("__SIZEOF_SIZE_T__", "8");
  defineMacro("__SIZEOF_SHORT__", "2");
  defineMacro("__SIZEOF_PTRDIFF_T__", "8");
  defineMacro("__SIZEOF_POINTER__", "8");
  defineMacro("__SIZEOF_LONG__", "8");
  defineMacro("__SIZEOF_LONG_LONG__", "8");
  defineMacro("__SIZEOF_LONG_DOUBLE__", "16");
  defineMacro("__SIZEOF_INT__", "4");
  defineMacro("__SIZEOF_INT128__", "16");
  defineMacro("__SIZEOF_FLOAT__", "4");
  defineMacro("__SIZEOF_DOUBLE__", "8");
  defineMacro("__SIG_ATOMIC_WIDTH__", "32");
  defineMacro("__SIG_ATOMIC_TYPE__", "int");
  defineMacro("__SIG_ATOMIC_MIN__", "(-__SIG_ATOMIC_MAX__");
  defineMacro("__SIG_ATOMIC_MAX__", "0x7fffffff");
  defineMacro("__SHRT_WIDTH__", "16");
  defineMacro("__SHRT_MAX__", "0x7fff");
  defineMacro("__SCHAR_WIDTH__", "8");
  defineMacro("__SCHAR_MAX__", "0x7f");
  defineMacro("__REGISTER_PREFIX__", "");
  defineMacro("__PTRDIFF_WIDTH__", "64");
  defineMacro("__PTRDIFF_TYPE__", "long");
  defineMacro("__PTRDIFF_MAX__", "0x7fffffffffffffffL");
  defineMacro("__PRAGMA_REDEFINE_EXTNAME", "1");
  defineMacro("__ORDER_PDP_ENDIAN__", "3412");
  defineMacro("__ORDER_LITTLE_ENDIAN__", "1234");
  defineMacro("__ORDER_BIG_ENDIAN__", "4321");
  defineMacro("__NO_INLINE__", "1");
  defineMacro("__LP64__", "1");
  defineMacro("__LONG_WIDTH__", "64");
  defineMacro("__LONG_MAX__", "0x7fffffffffffffffL");
  defineMacro("__LONG_LONG_WIDTH__", "64");
  defineMacro("__LONG_LONG_MAX__", "0x7fffffffffffffffLL");
  defineMacro("__LDBL_MIN__", "3.36210314311209350626267781732175260e-4932L");
  defineMacro("__LDBL_MIN_EXP__", "(-16381)");
  defineMacro("__LDBL_MIN_10_EXP__", "(-4931)");
  defineMacro("__LDBL_MAX_EXP__", "16384");
  defineMacro("__LDBL_MAX_10_EXP__", "4932");
  defineMacro("__LDBL_IS_IEC_60559__", "1");
  defineMacro("__LDBL_HAS_QUIET_NAN__", "1");
  defineMacro("__LDBL_HAS_INFINITY__", "1");
  defineMacro("__LDBL_HAS_DENORM__", "1");
  defineMacro("__INT_WIDTH__", "32");
  defineMacro("__INT_MAX__", "0x7fffffff");
  defineMacro("__INT_LEAST8_WIDTH__", "8");
  defineMacro("__INT_LEAST8_TYPE__", "signed");
  defineMacro("__INT_LEAST8_MAX__", "0x7f");
  defineMacro("__INT_LEAST64_WIDTH__", "64");
  defineMacro("__INT_LEAST64_TYPE__", "long");
  defineMacro("__INT_LEAST64_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INT_LEAST32_WIDTH__", "32");
  defineMacro("__INT_LEAST32_TYPE__", "int");
  defineMacro("__INT_LEAST32_MAX__", "0x7fffffff");
  defineMacro("__INT_LEAST16_WIDTH__", "16");
  defineMacro("__INT_LEAST16_TYPE__", "short");
  defineMacro("__INT_LEAST16_MAX__", "0x7fff");
  defineMacro("__INT_FAST8_WIDTH__", "8");
  defineMacro("__INT_FAST8_TYPE__", "signed");
  defineMacro("__INT_FAST8_MAX__", "0x7f");
  defineMacro("__INT_FAST64_WIDTH__", "64");
  defineMacro("__INT_FAST64_TYPE__", "long");
  defineMacro("__INT_FAST64_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INT_FAST32_WIDTH__", "64");
  defineMacro("__INT_FAST32_TYPE__", "long");
  defineMacro("__INT_FAST32_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INT_FAST16_WIDTH__", "64");
  defineMacro("__INT_FAST16_TYPE__", "long");
  defineMacro("__INT_FAST16_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INTPTR_WIDTH__", "64");
  defineMacro("__INTPTR_TYPE__", "long");
  defineMacro("__INTPTR_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INTMAX_WIDTH__", "64");
  defineMacro("__INTMAX_TYPE__", "long");
  defineMacro("__INTMAX_MAX__", "0x7fffffffffffffffL");
  defineMacro("__INTMAX_C(c)", "c");
  defineMacro("__INT8_TYPE__", "signed");
  defineMacro("__INT8_MAX__", "0x7f");
  defineMacro("__INT8_C(c)", "c");
  defineMacro("__INT64_TYPE__", "long");
  defineMacro("__INT64_MAX__", "0x7fffffffffffffffL");
  // defineMacro("__INT64_C(c)", "c");
  defineMacro("__INT32_TYPE__", "int");
  defineMacro("__INT32_MAX__", "0x7fffffff");
  defineMacro("__INT32_C(c)", "c");
  defineMacro("__INT16_TYPE__", "short");
  defineMacro("__INT16_MAX__", "0x7fff");
  defineMacro("__INT16_C(c)", "c");
  defineMacro("__HAVE_SPECULATION_SAFE_VALUE", "1");
  defineMacro("__GXX_WEAK__", "1");
  defineMacro("__GXX_RTTI", "1");
  defineMacro("__GXX_EXPERIMENTAL_CXX0X__", "1");
  defineMacro("__GXX_ABI_VERSION", "1019");
  defineMacro("__GNUG__", "14");
  defineMacro("__GNUC__", "14");
  defineMacro("__GNUC_WIDE_EXECUTION_CHARSET_NAME", "\"UTF-32LE\"");
  defineMacro("__GNUC_STDC_INLINE__", "1");
  defineMacro("__GNUC_PATCHLEVEL__", "0");
  defineMacro("__GNUC_MINOR__", "2");
  defineMacro("__GNUC_EXECUTION_CHARSET_NAME", "\"UTF-8\"");
  defineMacro("__GCC_IEC_559_COMPLEX", "2");
  defineMacro("__GCC_IEC_559", "2");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2", "1");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1", "1");
  defineMacro("__GCC_HAVE_DWARF2_CFI_ASM", "1");
  defineMacro("__GCC_CONSTRUCTIVE_SIZE", "64");
  defineMacro("__GCC_ATOMIC_WCHAR_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_TEST_AND_SET_TRUEVAL", "1");
  defineMacro("__GCC_ATOMIC_SHORT_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_POINTER_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_LONG_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_LLONG_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_INT_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR8_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR32_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_CHAR16_T_LOCK_FREE", "2");
  defineMacro("__GCC_ATOMIC_BOOL_LOCK_FREE", "2");
  defineMacro("__GCC_ASM_FLAG_OUTPUTS__", "1");
  defineMacro("__FLT_RADIX__", "2");
  defineMacro("__FLT_NORM_MAX__", "3.40282346638528859811704183484516925e+38F");
  defineMacro("__FLT_MIN__", "1.17549435082228750796873653722224568e-38F");
  defineMacro("__FLT_MIN_EXP__", "(-125)");
  defineMacro("__FLT_MIN_10_EXP__", "(-37)");
  defineMacro("__FLT_MAX__", "3.40282346638528859811704183484516925e+38F");
  defineMacro("__FLT_MAX_EXP__", "128");
  defineMacro("__FLT_MAX_10_EXP__", "38");
  defineMacro("__FLT_MANT_DIG__", "24");
  defineMacro("__FLT_IS_IEC_60559__", "1");
  defineMacro("__FLT_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT_HAS_INFINITY__", "1");
  defineMacro("__FLT_HAS_DENORM__", "1");
  defineMacro("__FLT_EVAL_METHOD__", "0");
  defineMacro("__FLT_EVAL_METHOD_TS_18661_3__", "0");
  defineMacro("__FLT_EPSILON__", "1.19209289550781250000000000000000000e-7F");
  defineMacro("__FLT_DIG__", "6");
  defineMacro("__FLT_DENORM_MIN__", "1.40129846432481707092372958328991613e-45F");
  defineMacro("__FLT_DECIMAL_DIG__", "9");
  defineMacro("__FLT64_NORM_MAX__", "1.79769313486231570814527423731704357e+308F64");
  defineMacro("__FLT64_MIN__", "2.22507385850720138309023271733240406e-308F64");
  defineMacro("__FLT64_MIN_EXP__", "(-1021)");
  defineMacro("__FLT64_MIN_10_EXP__", "(-307)");
  defineMacro("__FLT64_MAX__", "1.79769313486231570814527423731704357e+308F64");
  defineMacro("__FLT64_MAX_EXP__", "1024");
  defineMacro("__FLT64_MAX_10_EXP__", "308");
  defineMacro("__FLT64_MANT_DIG__", "53");
  defineMacro("__FLT64_IS_IEC_60559__", "1");
  defineMacro("__FLT64_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT64_HAS_INFINITY__", "1");
  defineMacro("__FLT64_HAS_DENORM__", "1");
  defineMacro("__FLT64_EPSILON__", "2.22044604925031308084726333618164062e-16F64");
  defineMacro("__FLT64_DIG__", "15");
  defineMacro("__FLT64_DENORM_MIN__", "4.94065645841246544176568792868221372e-324F64");
  defineMacro("__FLT64_DECIMAL_DIG__", "17");
  defineMacro("__FLT64X_MIN__", "3.36210314311209350626267781732175260e-4932F64x");
  defineMacro("__FLT64X_MIN_EXP__", "(-16381)");
  defineMacro("__FLT64X_MIN_10_EXP__", "(-4931)");
  defineMacro("__FLT64X_MAX_EXP__", "16384");
  defineMacro("__FLT64X_MAX_10_EXP__", "4932");
  defineMacro("__FLT64X_IS_IEC_60559__", "1");
  defineMacro("__FLT64X_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT64X_HAS_INFINITY__", "1");
  defineMacro("__FLT64X_HAS_DENORM__", "1");
  defineMacro("__FLT32_NORM_MAX__", "3.40282346638528859811704183484516925e+38F32");
  defineMacro("__FLT32_MIN__", "1.17549435082228750796873653722224568e-38F32");
  defineMacro("__FLT32_MIN_EXP__", "(-125)");
  defineMacro("__FLT32_MIN_10_EXP__", "(-37)");
  defineMacro("__FLT32_MAX__", "3.40282346638528859811704183484516925e+38F32");
  defineMacro("__FLT32_MAX_EXP__", "128");
  defineMacro("__FLT32_MAX_10_EXP__", "38");
  defineMacro("__FLT32_MANT_DIG__", "24");
  defineMacro("__FLT32_IS_IEC_60559__", "1");
  defineMacro("__FLT32_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT32_HAS_INFINITY__", "1");
  defineMacro("__FLT32_HAS_DENORM__", "1");
  defineMacro("__FLT32_EPSILON__", "1.19209289550781250000000000000000000e-7F32");
  defineMacro("__FLT32_DIG__", "6");
  defineMacro("__FLT32_DENORM_MIN__", "1.40129846432481707092372958328991613e-45F32");
  defineMacro("__FLT32_DECIMAL_DIG__", "9");
  defineMacro("__FLT32X_NORM_MAX__", "1.79769313486231570814527423731704357e+308F32x");
  defineMacro("__FLT32X_MIN__", "2.22507385850720138309023271733240406e-308F32x");
  defineMacro("__FLT32X_MIN_EXP__", "(-1021)");
  defineMacro("__FLT32X_MIN_10_EXP__", "(-307)");
  defineMacro("__FLT32X_MAX__", "1.79769313486231570814527423731704357e+308F32x");
  defineMacro("__FLT32X_MAX_EXP__", "1024");
  defineMacro("__FLT32X_MAX_10_EXP__", "308");
  defineMacro("__FLT32X_MANT_DIG__", "53");
  defineMacro("__FLT32X_IS_IEC_60559__", "1");
  defineMacro("__FLT32X_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT32X_HAS_INFINITY__", "1");
  defineMacro("__FLT32X_HAS_DENORM__", "1");
  defineMacro("__FLT32X_EPSILON__", "2.22044604925031308084726333618164062e-16F32x");
  defineMacro("__FLT32X_DIG__", "15");
  defineMacro("__FLT32X_DENORM_MIN__", "4.94065645841246544176568792868221372e-324F32x");
  defineMacro("__FLT32X_DECIMAL_DIG__", "17");
  defineMacro("__FLT16_NORM_MAX__", "6.55040000000000000000000000000000000e+4F16");
  defineMacro("__FLT16_MIN__", "6.10351562500000000000000000000000000e-5F16");
  defineMacro("__FLT16_MIN_EXP__", "(-13)");
  defineMacro("__FLT16_MIN_10_EXP__", "(-4)");
  defineMacro("__FLT16_MAX__", "6.55040000000000000000000000000000000e+4F16");
  defineMacro("__FLT16_MAX_EXP__", "16");
  defineMacro("__FLT16_MAX_10_EXP__", "4");
  defineMacro("__FLT16_MANT_DIG__", "11");
  defineMacro("__FLT16_IS_IEC_60559__", "1");
  defineMacro("__FLT16_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT16_HAS_INFINITY__", "1");
  defineMacro("__FLT16_HAS_DENORM__", "1");
  defineMacro("__FLT16_EPSILON__", "9.76562500000000000000000000000000000e-4F16");
  defineMacro("__FLT16_DIG__", "3");
  defineMacro("__FLT16_DENORM_MIN__", "5.96046447753906250000000000000000000e-8F16");
  defineMacro("__FLT16_DECIMAL_DIG__", "5");
  defineMacro("__FLT128_NORM_MAX__", "1.18973149535723176508575932662800702e+4932F128");
  defineMacro("__FLT128_MIN__", "3.36210314311209350626267781732175260e-4932F128");
  defineMacro("__FLT128_MIN_EXP__", "(-16381)");
  defineMacro("__FLT128_MIN_10_EXP__", "(-4931)");
  defineMacro("__FLT128_MAX__", "1.18973149535723176508575932662800702e+4932F128");
  defineMacro("__FLT128_MAX_EXP__", "16384");
  defineMacro("__FLT128_MAX_10_EXP__", "4932");
  defineMacro("__FLT128_MANT_DIG__", "113");
  defineMacro("__FLT128_IS_IEC_60559__", "1");
  defineMacro("__FLT128_HAS_QUIET_NAN__", "1");
  defineMacro("__FLT128_HAS_INFINITY__", "1");
  defineMacro("__FLT128_HAS_DENORM__", "1");
  defineMacro("__FLT128_EPSILON__", "1.92592994438723585305597794258492732e-34F128");
  defineMacro("__FLT128_DIG__", "33");
  defineMacro("__FLT128_DENORM_MIN__", "6.47517511943802511092443895822764655e-4966F128");
  defineMacro("__FLT128_DECIMAL_DIG__", "36");
  defineMacro("__FLOAT_WORD_ORDER__", "__ORDER_LITTLE_ENDIAN__");
  defineMacro("__FINITE_MATH_ONLY__", "0");
  defineMacro("__EXCEPTIONS", "1");
  defineMacro("__ELF__", "1");
  defineMacro("__DEPRECATED", "1");
  defineMacro("__DEC_EVAL_METHOD__", "2");
  defineMacro("__DECIMAL_BID_FORMAT__", "1");
  defineMacro("__DEC64_SUBNORMAL_MIN__", "0.000000000000001E-383DD");
  defineMacro("__DEC64_MIN__", "1E-383DD");
  defineMacro("__DEC64_MIN_EXP__", "(-382)");
  defineMacro("__DEC64_MAX__", "9.999999999999999E384DD");
  defineMacro("__DEC64_MAX_EXP__", "385");
  defineMacro("__DEC64_MANT_DIG__", "16");
  defineMacro("__DEC64_EPSILON__", "1E-15DD");
  defineMacro("__DEC32_SUBNORMAL_MIN__", "0.000001E-95DF");
  defineMacro("__DEC32_MIN__", "1E-95DF");
  defineMacro("__DEC32_MIN_EXP__", "(-94)");
  defineMacro("__DEC32_MAX__", "9.999999E96DF");
  defineMacro("__DEC32_MAX_EXP__", "97");
  defineMacro("__DEC32_MANT_DIG__", "7");
  defineMacro("__DEC32_EPSILON__", "1E-6DF");
  defineMacro("__DEC128_SUBNORMAL_MIN__", "0.000000000000000000000000000000001E-6143DL");
  defineMacro("__DEC128_MIN__", "1E-6143DL");
  defineMacro("__DEC128_MIN_EXP__", "(-6142)");
  defineMacro("__DEC128_MAX__", "9.999999999999999999999999999999999E6144DL");
  defineMacro("__DEC128_MAX_EXP__", "6145");
  defineMacro("__DEC128_MANT_DIG__", "34");
  defineMacro("__DEC128_EPSILON__", "1E-33DL");
  defineMacro("__DBL_NORM_MAX__", "double(1.79769313486231570814527423731704357e+308L)");
  defineMacro("__DBL_MIN__", "double(2.22507385850720138309023271733240406e-308L)");
  defineMacro("__DBL_MIN_EXP__", "(-1021)");
  defineMacro("__DBL_MIN_10_EXP__", "(-307)");
  defineMacro("__DBL_MAX__", "double(1.79769313486231570814527423731704357e+308L)");
  defineMacro("__DBL_MAX_EXP__", "1024");
  defineMacro("__DBL_MAX_10_EXP__", "308");
  defineMacro("__DBL_MANT_DIG__", "53");
  defineMacro("__DBL_IS_IEC_60559__", "1");
  defineMacro("__DBL_HAS_QUIET_NAN__", "1");
  defineMacro("__DBL_HAS_INFINITY__", "1");
  defineMacro("__DBL_HAS_DENORM__", "1");
  defineMacro("__DBL_EPSILON__", "double(2.22044604925031308084726333618164062e-16L)");
  defineMacro("__DBL_DIG__", "15");
  defineMacro("__DBL_DENORM_MIN__", "double(4.94065645841246544176568792868221372e-324L)");
  defineMacro("__DBL_DECIMAL_DIG__", "17");
  defineMacro("__CHAR_BIT__", "8");
  defineMacro("__CHAR8_TYPE__", "unsigned");
  defineMacro("__CHAR32_TYPE__", "unsigned");
  defineMacro("__CHAR16_TYPE__", "short");
  defineMacro("__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__");
  defineMacro("__BIGGEST_ALIGNMENT__", "16");
  defineMacro("__BFLT16_NORM_MAX__", "3.38953138925153547590470800371487867e+38BF16");
  defineMacro("__BFLT16_MIN__", "1.17549435082228750796873653722224568e-38BF16");
  defineMacro("__BFLT16_MIN_EXP__", "(-125)");
  defineMacro("__BFLT16_MIN_10_EXP__", "(-37)");
  defineMacro("__BFLT16_MAX__", "3.38953138925153547590470800371487867e+38BF16");
  defineMacro("__BFLT16_MAX_EXP__", "128");
  defineMacro("__BFLT16_MAX_10_EXP__", "38");
  defineMacro("__BFLT16_MANT_DIG__", "8");
  defineMacro("__BFLT16_IS_IEC_60559__", "0");
  defineMacro("__BFLT16_HAS_QUIET_NAN__", "1");
  defineMacro("__BFLT16_HAS_INFINITY__", "1");
  defineMacro("__BFLT16_HAS_DENORM__", "1");
  defineMacro("__BFLT16_EPSILON__", "7.81250000000000000000000000000000000e-3BF16");
  defineMacro("__BFLT16_DIG__", "2");
  defineMacro("__BFLT16_DENORM_MIN__", "9.18354961579912115600575419704879436e-41BF16");
  defineMacro("__BFLT16_DECIMAL_DIG__", "4");
  defineMacro("__ATOMIC_SEQ_CST", "5");
  defineMacro("__ATOMIC_RELEASE", "3");
  defineMacro("__ATOMIC_RELAXED", "0");
  defineMacro("__ATOMIC_CONSUME", "1");
  defineMacro("__ATOMIC_ACQ_REL", "4");
  defineMacro("__ATOMIC_ACQUIRE", "2");
  defineMacro("_STDC_PREDEF_H", "1");
  defineMacro("_LP64", "1");
  defineMacro("_GNU_SOURCE", "1");
}

void GCCLinuxToolchain::addCxx26Macros() {
  defineMacro("__cpp_auto_cast", "202110L");
  defineMacro("__cpp_constexpr", "202306L");
  defineMacro("__cpp_explicit_this_parameter", "202110L");
  defineMacro("__cpp_if_consteval", "202106L");
  defineMacro("__cpp_implicit_move", "202207L");
  defineMacro("__cpp_multidimensional_subscript", "202211L");
  defineMacro("__cpp_named_character_escapes", "202207L");
  defineMacro("__cpp_placeholder_variables", "202306L");
  defineMacro("__cpp_size_t_suffix", "202011L");
  defineMacro("__cpp_static_assert", "202306L");
  defineMacro("__cpp_static_call_operator", "202207L");
  defineMacro("__STDCPP_BFLOAT16_T__", "1");
  defineMacro("__STDCPP_FLOAT128_T__", "1");
  defineMacro("__STDCPP_FLOAT16_T__", "1");
  defineMacro("__STDCPP_FLOAT32_T__", "1");
  defineMacro("__STDCPP_FLOAT64_T__", "1");
}

void GCCLinuxToolchain::addAmd64Macros() {
  defineMacro("__ATOMIC_HLE_ACQUIRE", "65536");
  defineMacro("__ATOMIC_HLE_RELEASE", "131072");
  defineMacro("__DECIMAL_DIG__", "21");
  defineMacro("__FLT64X_DECIMAL_DIG__", "21");
  defineMacro("__FLT64X_DENORM_MIN__", "3.64519953188247460252840593361941982e-4951F64x");
  defineMacro("__FLT64X_DIG__", "18");
  defineMacro("__FLT64X_EPSILON__", "1.08420217248550443400745280086994171e-19F64x");
  defineMacro("__FLT64X_MANT_DIG__", "64");
  defineMacro("__FLT64X_MAX__", "1.18973149535723176502126385303097021e+4932F64x");
  defineMacro("__FLT64X_NORM_MAX__", "1.18973149535723176502126385303097021e+4932F64x");
  defineMacro("__FXSR__", "1");
  defineMacro("__GCC_DESTRUCTIVE_SIZE", "64");
  defineMacro("__LDBL_DECIMAL_DIG__", "21");
  defineMacro("__LDBL_DENORM_MIN__", "3.64519953188247460252840593361941982e-4951L");
  defineMacro("__LDBL_DIG__", "18");
  defineMacro("__LDBL_EPSILON__", "1.08420217248550443400745280086994171e-19L");
  defineMacro("__LDBL_MANT_DIG__", "64");
  defineMacro("__LDBL_MAX__", "1.18973149535723176502126385303097021e+4932L");
  defineMacro("__LDBL_NORM_MAX__", "1.18973149535723176502126385303097021e+4932L");
  defineMacro("__MMX_WITH_SSE__", "1");
  defineMacro("__MMX__", "1");
  defineMacro("__SEG_FS", "1");
  defineMacro("__SEG_GS", "1");
  defineMacro("__SIZEOF_FLOAT128__", "16");
  defineMacro("__SIZEOF_FLOAT80__", "16");
  defineMacro("__SSE2_MATH__", "1");
  defineMacro("__SSE2__", "1");
  defineMacro("__SSE_MATH__", "1");
  defineMacro("__SSE__", "1");
  defineMacro("__WCHAR_MAX__", "0x7fffffff");
  defineMacro("__WCHAR_MIN__", "(-__WCHAR_MAX__");
  defineMacro("__WCHAR_TYPE__", "int");
  defineMacro("__amd64", "1");
  defineMacro("__amd64__", "1");
  defineMacro("__code_model_small__", "1");
  defineMacro("__k8", "1");
  defineMacro("__k8__", "1");
  defineMacro("__x86_64", "1");
  defineMacro("__x86_64__", "1");
}

void GCCLinuxToolchain::addArm64Macros() {
  defineMacro("__AARCH64EL__", "1");
  defineMacro("__AARCH64_CMODEL_SMALL__", "1");
  defineMacro("__ARM_64BIT_STATE", "1");
  defineMacro("__ARM_ALIGN_MAX_PWR", "28");
  defineMacro("__ARM_ALIGN_MAX_STACK_PWR", "16");
  defineMacro("__ARM_ARCH", "8");
  defineMacro("__ARM_ARCH_8A", "1");
  defineMacro("__ARM_ARCH_ISA_A64", "1");
  defineMacro("__ARM_ARCH_PROFILE", "65");
  defineMacro("__ARM_FEATURE_CLZ", "1");
  defineMacro("__ARM_FEATURE_FMA", "1");
  defineMacro("__ARM_FEATURE_IDIV", "1");
  defineMacro("__ARM_FEATURE_NUMERIC_MAXMIN", "1");
  defineMacro("__ARM_FEATURE_UNALIGNED", "1");
  defineMacro("__ARM_FP", "14");
  defineMacro("__ARM_FP16_ARGS", "1");
  defineMacro("__ARM_FP16_FORMAT_IEEE", "1");
  defineMacro("__ARM_NEON", "1");
  defineMacro("__ARM_NEON_SVE_BRIDGE", "1");
  defineMacro("__ARM_PCS_AAPCS64", "1");
  defineMacro("__ARM_SIZEOF_MINIMAL_ENUM", "4");
  defineMacro("__ARM_SIZEOF_WCHAR_T", "4");
  defineMacro("__ARM_STATE_ZA", "1");
  defineMacro("__ARM_STATE_ZT0", "1");
  defineMacro("__CHAR_UNSIGNED__", "1");
  defineMacro("__DECIMAL_DIG__", "36");
  defineMacro("__FLT64X_DECIMAL_DIG__", "36");
  defineMacro("__FLT64X_DENORM_MIN__", "6.47517511943802511092443895822764655e-4966F64x");
  defineMacro("__FLT64X_DIG__", "33");
  defineMacro("__FLT64X_EPSILON__", "1.92592994438723585305597794258492732e-34F64x");
  defineMacro("__FLT64X_MANT_DIG__", "113");
  defineMacro("__FLT64X_MAX__", "1.18973149535723176508575932662800702e+4932F64x");
  defineMacro("__FLT64X_NORM_MAX__", "1.18973149535723176508575932662800702e+4932F64x");
  defineMacro("__FLT_EVAL_METHOD_C99__", "0");
  defineMacro("__FP_FAST_FMA", "1");
  defineMacro("__FP_FAST_FMAF", "1");
  defineMacro("__FP_FAST_FMAF32", "1");
  defineMacro("__FP_FAST_FMAF32x", "1");
  defineMacro("__FP_FAST_FMAF64", "1");
  defineMacro("__GCC_DESTRUCTIVE_SIZE", "256");
  defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16", "1");
  defineMacro("__LDBL_DECIMAL_DIG__", "36");
  defineMacro("__LDBL_DENORM_MIN__", "6.47517511943802511092443895822764655e-4966L");
  defineMacro("__LDBL_DIG__", "33");
  defineMacro("__LDBL_EPSILON__", "1.92592994438723585305597794258492732e-34L");
  defineMacro("__LDBL_MANT_DIG__", "113");
  defineMacro("__LDBL_MAX__", "1.18973149535723176508575932662800702e+4932L");
  defineMacro("__LDBL_NORM_MAX__", "1.18973149535723176508575932662800702e+4932L");
  defineMacro("__WCHAR_MAX__", "0xffffffffU");
  defineMacro("__WCHAR_MIN__", "0U");
  defineMacro("__WCHAR_TYPE__", "unsigned");
  defineMacro("__WCHAR_UNSIGNED__", "1");
  defineMacro("__aarch64__", "1");
  defineMacro("__arm_in(...)", "[[arm::in(__VA_ARGS__)]]");
  defineMacro("__arm_inout(...)", "[[arm::inout(__VA_ARGS__)]]");
  defineMacro("__arm_locally_streaming", "[[arm::locally_streaming]]");
  defineMacro("__arm_new(...)", "[[arm::new(__VA_ARGS__)]]");
  defineMacro("__arm_out(...)", "[[arm::out(__VA_ARGS__)]]");
  defineMacro("__arm_preserves(...)", "[[arm::preserves(__VA_ARGS__)]]");
  defineMacro("__arm_streaming", "[[arm::streaming]]");
  defineMacro("__arm_streaming_compatible", "[[arm::streaming_compatible]]");
}
// clang-format on

}  // namespace cxx
