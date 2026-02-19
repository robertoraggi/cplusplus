// Generated file by: gen_token_fwd_h.ts
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

#pragma once

#include <cxx/cxx_fwd.h>

#include <cstdint>
#include <optional>

namespace cxx {

union TokenValue;
class Token;

#define FOR_EACH_BASE_TOKEN(V)                                    \
  V(EOF_SYMBOL, "<eof_symbol>")                                   \
  V(ERROR, "<error>")                                             \
  V(COMMENT, "<comment>")                                         \
  V(IDENTIFIER, "<identifier>")                                   \
  V(CHARACTER_LITERAL, "<character_literal>")                     \
  V(FLOATING_POINT_LITERAL, "<floating_point_literal>")           \
  V(INTEGER_LITERAL, "<integer_literal>")                         \
  V(STRING_LITERAL, "<string_literal>")                           \
  V(USER_DEFINED_STRING_LITERAL, "<user_defined_string_literal>") \
  V(UTF16_STRING_LITERAL, "<utf16_string_literal>")               \
  V(UTF32_STRING_LITERAL, "<utf32_string_literal>")               \
  V(UTF8_STRING_LITERAL, "<utf8_string_literal>")                 \
  V(WIDE_STRING_LITERAL, "<wide_string_literal>")                 \
  V(PP_INTERNAL_VARIABLE, "<pp_internal_variable>")               \
  V(CODE_COMPLETION, "<code_completion>")

#define FOR_EACH_OPERATOR(V)      \
  V(AMP_AMP, "&&")                \
  V(AMP_EQUAL, "&=")              \
  V(AMP, "&")                     \
  V(BAR_BAR, "||")                \
  V(BAR_EQUAL, "|=")              \
  V(BAR, "|")                     \
  V(CARET_CARET, "^^")            \
  V(CARET_EQUAL, "^=")            \
  V(CARET, "^")                   \
  V(COLON_COLON, "::")            \
  V(COLON, ":")                   \
  V(COMMA, ",")                   \
  V(DELETE_ARRAY, "delete[]")     \
  V(DOT_DOT_DOT, "...")           \
  V(DOT_STAR, ".*")               \
  V(DOT, ".")                     \
  V(EQUAL_EQUAL, "==")            \
  V(EQUAL, "=")                   \
  V(EXCLAIM_EQUAL, "!=")          \
  V(EXCLAIM, "!")                 \
  V(GREATER_EQUAL, ">=")          \
  V(GREATER_GREATER_EQUAL, ">>=") \
  V(GREATER_GREATER, ">>")        \
  V(GREATER, ">")                 \
  V(HASH_HASH, "##")              \
  V(HASH, "#")                    \
  V(LBRACE, "{")                  \
  V(LBRACKET, "[")                \
  V(LESS_EQUAL_GREATER, "<=>")    \
  V(LESS_EQUAL, "<=")             \
  V(LESS_LESS_EQUAL, "<<=")       \
  V(LESS_LESS, "<<")              \
  V(LESS, "<")                    \
  V(LPAREN, "(")                  \
  V(MINUS_EQUAL, "-=")            \
  V(MINUS_GREATER_STAR, "->*")    \
  V(MINUS_GREATER, "->")          \
  V(MINUS_MINUS, "--")            \
  V(MINUS, "-")                   \
  V(NEW_ARRAY, "new[]")           \
  V(PERCENT_EQUAL, "%=")          \
  V(PERCENT, "%")                 \
  V(PLUS_EQUAL, "+=")             \
  V(PLUS_PLUS, "++")              \
  V(PLUS, "+")                    \
  V(QUESTION, "?")                \
  V(RBRACE, "}")                  \
  V(RBRACKET, "]")                \
  V(RPAREN, ")")                  \
  V(SEMICOLON, ";")               \
  V(SLASH_EQUAL, "/=")            \
  V(SLASH, "/")                   \
  V(STAR_EQUAL, "*=")             \
  V(STAR, "*")                    \
  V(TILDE, "~")

#define FOR_EACH_KEYWORD(V)                     \
  V(_ATOMIC, "_Atomic")                         \
  V(_BITINT, "_BitInt")                         \
  V(_COMPLEX, "_Complex")                       \
  V(_DECIMAL128, "_Decimal128")                 \
  V(_DECIMAL32, "_Decimal32")                   \
  V(_DECIMAL64, "_Decimal64")                   \
  V(_FLOAT16, "_Float16")                       \
  V(_GENERIC, "_Generic")                       \
  V(_IMAGINARY, "_Imaginary")                   \
  V(_NORETURN, "_Noreturn")                     \
  V(__ATTRIBUTE__, "__attribute__")             \
  V(__BUILTIN_BIT_CAST, "__builtin_bit_cast")   \
  V(__BUILTIN_META_INFO, "__builtin_meta_info") \
  V(__BUILTIN_OFFSETOF, "__builtin_offsetof")   \
  V(__BUILTIN_VA_ARG, "__builtin_va_arg")       \
  V(__BUILTIN_VA_LIST, "__builtin_va_list")     \
  V(__COMPLEX__, "__complex__")                 \
  V(__EXTENSION__, "__extension__")             \
  V(__FLOAT128, "__float128")                   \
  V(__FLOAT80, "__float80")                     \
  V(__IMAG__, "__imag__")                       \
  V(__INT128, "__int128")                       \
  V(__INT128_T, "__int128_t")                   \
  V(__INT64, "__int64")                         \
  V(__REAL__, "__real__")                       \
  V(__RESTRICT__, "__restrict__")               \
  V(__THREAD, "__thread")                       \
  V(__UINT128_T, "__uint128_t")                 \
  V(__UNDERLYING_TYPE, "__underlying_type")     \
  V(ALIGNAS, "alignas")                         \
  V(ALIGNOF, "alignof")                         \
  V(ASM, "asm")                                 \
  V(AUTO, "auto")                               \
  V(BOOL, "bool")                               \
  V(BREAK, "break")                             \
  V(CASE, "case")                               \
  V(CATCH, "catch")                             \
  V(CHAR, "char")                               \
  V(CHAR16_T, "char16_t")                       \
  V(CHAR32_T, "char32_t")                       \
  V(CHAR8_T, "char8_t")                         \
  V(CLASS, "class")                             \
  V(CO_AWAIT, "co_await")                       \
  V(CO_RETURN, "co_return")                     \
  V(CO_YIELD, "co_yield")                       \
  V(CONCEPT, "concept")                         \
  V(CONST, "const")                             \
  V(CONST_CAST, "const_cast")                   \
  V(CONSTEVAL, "consteval")                     \
  V(CONSTEXPR, "constexpr")                     \
  V(CONSTINIT, "constinit")                     \
  V(CONTINUE, "continue")                       \
  V(DECLTYPE, "decltype")                       \
  V(DEFAULT, "default")                         \
  V(DELETE, "delete")                           \
  V(DO, "do")                                   \
  V(DOUBLE, "double")                           \
  V(DYNAMIC_CAST, "dynamic_cast")               \
  V(ELSE, "else")                               \
  V(ENUM, "enum")                               \
  V(EXPLICIT, "explicit")                       \
  V(EXPORT, "export")                           \
  V(EXTERN, "extern")                           \
  V(FALSE, "false")                             \
  V(FLOAT, "float")                             \
  V(FOR, "for")                                 \
  V(FRIEND, "friend")                           \
  V(GOTO, "goto")                               \
  V(IF, "if")                                   \
  V(IMPORT, "import")                           \
  V(INLINE, "inline")                           \
  V(INT, "int")                                 \
  V(LONG, "long")                               \
  V(MODULE, "module")                           \
  V(MUTABLE, "mutable")                         \
  V(NAMESPACE, "namespace")                     \
  V(NEW, "new")                                 \
  V(NOEXCEPT, "noexcept")                       \
  V(NULLPTR, "nullptr")                         \
  V(OPERATOR, "operator")                       \
  V(PRIVATE, "private")                         \
  V(PROTECTED, "protected")                     \
  V(PUBLIC, "public")                           \
  V(REGISTER, "register")                       \
  V(REINTERPRET_CAST, "reinterpret_cast")       \
  V(REQUIRES, "requires")                       \
  V(RETURN, "return")                           \
  V(SHORT, "short")                             \
  V(SIGNED, "signed")                           \
  V(SIZEOF, "sizeof")                           \
  V(STATIC, "static")                           \
  V(STATIC_ASSERT, "static_assert")             \
  V(STATIC_CAST, "static_cast")                 \
  V(STRUCT, "struct")                           \
  V(SWITCH, "switch")                           \
  V(TEMPLATE, "template")                       \
  V(THIS, "this")                               \
  V(THREAD_LOCAL, "thread_local")               \
  V(THROW, "throw")                             \
  V(TRUE, "true")                               \
  V(TRY, "try")                                 \
  V(TYPEDEF, "typedef")                         \
  V(TYPEID, "typeid")                           \
  V(TYPENAME, "typename")                       \
  V(TYPEOF, "typeof")                           \
  V(TYPEOF_UNQUAL, "typeof_unqual")             \
  V(UNION, "union")                             \
  V(UNSIGNED, "unsigned")                       \
  V(USING, "using")                             \
  V(VIRTUAL, "virtual")                         \
  V(VOID, "void")                               \
  V(VOLATILE, "volatile")                       \
  V(WCHAR_T, "wchar_t")                         \
  V(WHILE, "while")

#define FOR_EACH_BUILTIN_TYPE_TRAIT(V)                            \
  V(__HAS_UNIQUE_OBJECT_REPRESENTATIONS,                          \
    "__has_unique_object_representations")                        \
  V(__HAS_VIRTUAL_DESTRUCTOR, "__has_virtual_destructor")         \
  V(__IS_ABSTRACT, "__is_abstract")                               \
  V(__IS_AGGREGATE, "__is_aggregate")                             \
  V(__IS_ARITHMETIC, "__is_arithmetic")                           \
  V(__IS_ARRAY, "__is_array")                                     \
  V(__IS_ASSIGNABLE, "__is_assignable")                           \
  V(__IS_BASE_OF, "__is_base_of")                                 \
  V(__IS_BOUNDED_ARRAY, "__is_bounded_array")                     \
  V(__IS_CLASS, "__is_class")                                     \
  V(__IS_COMPOUND, "__is_compound")                               \
  V(__IS_CONST, "__is_const")                                     \
  V(__IS_CONSTRUCTIBLE, "__is_constructible")                     \
  V(__IS_CONVERTIBLE, "__is_convertible")                         \
  V(__IS_CONVERTIBLE_TO, "__is_convertible_to")                   \
  V(__IS_DESTRUCTIBLE, "__is_destructible")                       \
  V(__IS_EMPTY, "__is_empty")                                     \
  V(__IS_ENUM, "__is_enum")                                       \
  V(__IS_FINAL, "__is_final")                                     \
  V(__IS_FLOATING_POINT, "__is_floating_point")                   \
  V(__IS_FUNCTION, "__is_function")                               \
  V(__IS_FUNDAMENTAL, "__is_fundamental")                         \
  V(__IS_INTEGRAL, "__is_integral")                               \
  V(__IS_LAYOUT_COMPATIBLE, "__is_layout_compatible")             \
  V(__IS_LITERAL_TYPE, "__is_literal_type")                       \
  V(__IS_LVALUE_REFERENCE, "__is_lvalue_reference")               \
  V(__IS_MEMBER_FUNCTION_POINTER, "__is_member_function_pointer") \
  V(__IS_MEMBER_OBJECT_POINTER, "__is_member_object_pointer")     \
  V(__IS_MEMBER_POINTER, "__is_member_pointer")                   \
  V(__IS_NOTHROW_CONSTRUCTIBLE, "__is_nothrow_constructible")     \
  V(__IS_NULL_POINTER, "__is_null_pointer")                       \
  V(__IS_OBJECT, "__is_object")                                   \
  V(__IS_POD, "__is_pod")                                         \
  V(__IS_POINTER, "__is_pointer")                                 \
  V(__IS_POLYMORPHIC, "__is_polymorphic")                         \
  V(__IS_REFERENCE, "__is_reference")                             \
  V(__IS_RVALUE_REFERENCE, "__is_rvalue_reference")               \
  V(__IS_SAME_AS, "__is_same_as")                                 \
  V(__IS_SAME, "__is_same")                                       \
  V(__IS_SCALAR, "__is_scalar")                                   \
  V(__IS_SCOPED_ENUM, "__is_scoped_enum")                         \
  V(__IS_SIGNED, "__is_signed")                                   \
  V(__IS_STANDARD_LAYOUT, "__is_standard_layout")                 \
  V(__IS_SWAPPABLE_WITH, "__is_swappable_with")                   \
  V(__IS_TRIVIAL, "__is_trivial")                                 \
  V(__IS_TRIVIALLY_ASSIGNABLE, "__is_trivially_assignable")       \
  V(__IS_TRIVIALLY_CONSTRUCTIBLE, "__is_trivially_constructible") \
  V(__IS_TRIVIALLY_COPYABLE, "__is_trivially_copyable")           \
  V(__IS_UNBOUNDED_ARRAY, "__is_unbounded_array")                 \
  V(__IS_UNION, "__is_union")                                     \
  V(__IS_UNSIGNED, "__is_unsigned")                               \
  V(__IS_VOID, "__is_void")                                       \
  V(__IS_VOLATILE, "__is_volatile")

#define FOR_EACH_UNARY_BUILTIN_TYPE_TRAIT(V)          \
  V(__ADD_LVALUE_REFERENCE, "__add_lvalue_reference") \
  V(__ADD_POINTER, "__add_pointer")                   \
  V(__ADD_RVALUE_REFERENCE, "__add_rvalue_reference") \
  V(__DECAY, "__decay")                               \
  V(__MAKE_SIGNED, "__make_signed")                   \
  V(__MAKE_UNSIGNED, "__make_unsigned")               \
  V(__REMOVE_ALL_EXTENTS, "__remove_all_extents")     \
  V(__REMOVE_CONST, "__remove_const")                 \
  V(__REMOVE_CV, "__remove_cv")                       \
  V(__REMOVE_CVREF, "__remove_cvref")                 \
  V(__REMOVE_EXTENT, "__remove_extent")               \
  V(__REMOVE_POINTER, "__remove_pointer")             \
  V(__REMOVE_REFERENCE_T, "__remove_reference_t")     \
  V(__REMOVE_RESTRICT, "__remove_restrict")           \
  V(__REMOVE_VOLATILE, "__remove_volatile")

#define FOR_EACH_BINARY_BUILTIN_TYPE_TRAIT(V)

#define FOR_EACH_BUILTIN_FUNCTION(V)                                    \
  V(__BUILTIN___COSPI, "__builtin___cospi")                             \
  V(__BUILTIN___COSPIF, "__builtin___cospif")                           \
  V(__BUILTIN___EXP10, "__builtin___exp10")                             \
  V(__BUILTIN___EXP10F, "__builtin___exp10f")                           \
  V(__BUILTIN___FINITE, "__builtin___finite")                           \
  V(__BUILTIN___FINITEF, "__builtin___finitef")                         \
  V(__BUILTIN___FINITEL, "__builtin___finitel")                         \
  V(__BUILTIN___SINPI, "__builtin___sinpi")                             \
  V(__BUILTIN___SINPIF, "__builtin___sinpif")                           \
  V(__BUILTIN___TANPI, "__builtin___tanpi")                             \
  V(__BUILTIN___TANPIF, "__builtin___tanpif")                           \
  V(__BUILTIN_ABS, "__builtin_abs")                                     \
  V(__BUILTIN_ACOS, "__builtin_acos")                                   \
  V(__BUILTIN_ACOSF, "__builtin_acosf")                                 \
  V(__BUILTIN_ACOSH, "__builtin_acosh")                                 \
  V(__BUILTIN_ACOSHF, "__builtin_acoshf")                               \
  V(__BUILTIN_ACOSHL, "__builtin_acoshl")                               \
  V(__BUILTIN_ACOSL, "__builtin_acosl")                                 \
  V(__BUILTIN_ASIN, "__builtin_asin")                                   \
  V(__BUILTIN_ASINF, "__builtin_asinf")                                 \
  V(__BUILTIN_ASINH, "__builtin_asinh")                                 \
  V(__BUILTIN_ASINHF, "__builtin_asinhf")                               \
  V(__BUILTIN_ASINHL, "__builtin_asinhl")                               \
  V(__BUILTIN_ASINL, "__builtin_asinl")                                 \
  V(__BUILTIN_ATAN, "__builtin_atan")                                   \
  V(__BUILTIN_ATAN2, "__builtin_atan2")                                 \
  V(__BUILTIN_ATAN2F, "__builtin_atan2f")                               \
  V(__BUILTIN_ATAN2L, "__builtin_atan2l")                               \
  V(__BUILTIN_ATANF, "__builtin_atanf")                                 \
  V(__BUILTIN_ATANH, "__builtin_atanh")                                 \
  V(__BUILTIN_ATANHF, "__builtin_atanhf")                               \
  V(__BUILTIN_ATANHL, "__builtin_atanhl")                               \
  V(__BUILTIN_ATANL, "__builtin_atanl")                                 \
  V(__BUILTIN_BCMP, "__builtin_bcmp")                                   \
  V(__BUILTIN_BCOPY, "__builtin_bcopy")                                 \
  V(__BUILTIN_BSWAP32, "__builtin_bswap32")                             \
  V(__BUILTIN_BSWAP64, "__builtin_bswap64")                             \
  V(__BUILTIN_BZERO, "__builtin_bzero")                                 \
  V(__BUILTIN_CBRT, "__builtin_cbrt")                                   \
  V(__BUILTIN_CBRTF, "__builtin_cbrtf")                                 \
  V(__BUILTIN_CBRTL, "__builtin_cbrtl")                                 \
  V(__BUILTIN_CEIL, "__builtin_ceil")                                   \
  V(__BUILTIN_CEILF, "__builtin_ceilf")                                 \
  V(__BUILTIN_CEILL, "__builtin_ceill")                                 \
  V(__BUILTIN_COPYSIGN, "__builtin_copysign")                           \
  V(__BUILTIN_COPYSIGNF, "__builtin_copysignf")                         \
  V(__BUILTIN_COPYSIGNL, "__builtin_copysignl")                         \
  V(__BUILTIN_COS, "__builtin_cos")                                     \
  V(__BUILTIN_COSF, "__builtin_cosf")                                   \
  V(__BUILTIN_COSH, "__builtin_cosh")                                   \
  V(__BUILTIN_COSHF, "__builtin_coshf")                                 \
  V(__BUILTIN_COSHL, "__builtin_coshl")                                 \
  V(__BUILTIN_COSL, "__builtin_cosl")                                   \
  V(__BUILTIN_CTZ, "__builtin_ctz")                                     \
  V(__BUILTIN_CTZL, "__builtin_ctzl")                                   \
  V(__BUILTIN_CTZLL, "__builtin_ctzll")                                 \
  V(__BUILTIN_ERF, "__builtin_erf")                                     \
  V(__BUILTIN_ERFC, "__builtin_erfc")                                   \
  V(__BUILTIN_ERFCF, "__builtin_erfcf")                                 \
  V(__BUILTIN_ERFCL, "__builtin_erfcl")                                 \
  V(__BUILTIN_ERFF, "__builtin_erff")                                   \
  V(__BUILTIN_ERFL, "__builtin_erfl")                                   \
  V(__BUILTIN_EXP, "__builtin_exp")                                     \
  V(__BUILTIN_EXP2, "__builtin_exp2")                                   \
  V(__BUILTIN_EXP2F, "__builtin_exp2f")                                 \
  V(__BUILTIN_EXP2L, "__builtin_exp2l")                                 \
  V(__BUILTIN_EXPECT, "__builtin_expect")                               \
  V(__BUILTIN_EXPF, "__builtin_expf")                                   \
  V(__BUILTIN_EXPL, "__builtin_expl")                                   \
  V(__BUILTIN_EXPM1, "__builtin_expm1")                                 \
  V(__BUILTIN_EXPM1F, "__builtin_expm1f")                               \
  V(__BUILTIN_EXPM1L, "__builtin_expm1l")                               \
  V(__BUILTIN_FABS, "__builtin_fabs")                                   \
  V(__BUILTIN_FABSF, "__builtin_fabsf")                                 \
  V(__BUILTIN_FABSL, "__builtin_fabsl")                                 \
  V(__BUILTIN_FDIM, "__builtin_fdim")                                   \
  V(__BUILTIN_FDIMF, "__builtin_fdimf")                                 \
  V(__BUILTIN_FDIML, "__builtin_fdiml")                                 \
  V(__BUILTIN_FINITE, "__builtin_finite")                               \
  V(__BUILTIN_FINITEF, "__builtin_finitef")                             \
  V(__BUILTIN_FINITEL, "__builtin_finitel")                             \
  V(__BUILTIN_FLOOR, "__builtin_floor")                                 \
  V(__BUILTIN_FLOORF, "__builtin_floorf")                               \
  V(__BUILTIN_FLOORL, "__builtin_floorl")                               \
  V(__BUILTIN_FMA, "__builtin_fma")                                     \
  V(__BUILTIN_FMAF, "__builtin_fmaf")                                   \
  V(__BUILTIN_FMAL, "__builtin_fmal")                                   \
  V(__BUILTIN_FMAX, "__builtin_fmax")                                   \
  V(__BUILTIN_FMAXF, "__builtin_fmaxf")                                 \
  V(__BUILTIN_FMAXIMUM_NUM, "__builtin_fmaximum_num")                   \
  V(__BUILTIN_FMAXIMUM_NUMF, "__builtin_fmaximum_numf")                 \
  V(__BUILTIN_FMAXIMUM_NUML, "__builtin_fmaximum_numl")                 \
  V(__BUILTIN_FMAXL, "__builtin_fmaxl")                                 \
  V(__BUILTIN_FMIN, "__builtin_fmin")                                   \
  V(__BUILTIN_FMINF, "__builtin_fminf")                                 \
  V(__BUILTIN_FMINIMUM_NUM, "__builtin_fminimum_num")                   \
  V(__BUILTIN_FMINIMUM_NUMF, "__builtin_fminimum_numf")                 \
  V(__BUILTIN_FMINIMUM_NUML, "__builtin_fminimum_numl")                 \
  V(__BUILTIN_FMINL, "__builtin_fminl")                                 \
  V(__BUILTIN_FMOD, "__builtin_fmod")                                   \
  V(__BUILTIN_FMODF, "__builtin_fmodf")                                 \
  V(__BUILTIN_FMODL, "__builtin_fmodl")                                 \
  V(__BUILTIN_FREXP, "__builtin_frexp")                                 \
  V(__BUILTIN_FREXPF, "__builtin_frexpf")                               \
  V(__BUILTIN_FREXPL, "__builtin_frexpl")                               \
  V(__BUILTIN_HYPOT, "__builtin_hypot")                                 \
  V(__BUILTIN_HYPOTF, "__builtin_hypotf")                               \
  V(__BUILTIN_HYPOTL, "__builtin_hypotl")                               \
  V(__BUILTIN_ILOGB, "__builtin_ilogb")                                 \
  V(__BUILTIN_ILOGBF, "__builtin_ilogbf")                               \
  V(__BUILTIN_ILOGBL, "__builtin_ilogbl")                               \
  V(__BUILTIN_INDEX, "__builtin_index")                                 \
  V(__BUILTIN_INF, "__builtin_inf")                                     \
  V(__BUILTIN_INFF, "__builtin_inff")                                   \
  V(__BUILTIN_INFL, "__builtin_infl")                                   \
  V(__BUILTIN_IS_CONSTANT_EVALUATED, "__builtin_is_constant_evaluated") \
  V(__BUILTIN_ISFINITE, "__builtin_isfinite")                           \
  V(__BUILTIN_ISINF, "__builtin_isinf")                                 \
  V(__BUILTIN_ISNAN, "__builtin_isnan")                                 \
  V(__BUILTIN_ISNORMAL, "__builtin_isnormal")                           \
  V(__BUILTIN_LABS, "__builtin_labs")                                   \
  V(__BUILTIN_LDEXP, "__builtin_ldexp")                                 \
  V(__BUILTIN_LDEXPF, "__builtin_ldexpf")                               \
  V(__BUILTIN_LDEXPL, "__builtin_ldexpl")                               \
  V(__BUILTIN_LGAMMA, "__builtin_lgamma")                               \
  V(__BUILTIN_LGAMMAF, "__builtin_lgammaf")                             \
  V(__BUILTIN_LGAMMAL, "__builtin_lgammal")                             \
  V(__BUILTIN_LLABS, "__builtin_llabs")                                 \
  V(__BUILTIN_LLRINT, "__builtin_llrint")                               \
  V(__BUILTIN_LLRINTF, "__builtin_llrintf")                             \
  V(__BUILTIN_LLRINTL, "__builtin_llrintl")                             \
  V(__BUILTIN_LLROUND, "__builtin_llround")                             \
  V(__BUILTIN_LLROUNDF, "__builtin_llroundf")                           \
  V(__BUILTIN_LLROUNDL, "__builtin_llroundl")                           \
  V(__BUILTIN_LOG, "__builtin_log")                                     \
  V(__BUILTIN_LOG10, "__builtin_log10")                                 \
  V(__BUILTIN_LOG10F, "__builtin_log10f")                               \
  V(__BUILTIN_LOG10L, "__builtin_log10l")                               \
  V(__BUILTIN_LOG1P, "__builtin_log1p")                                 \
  V(__BUILTIN_LOG1PF, "__builtin_log1pf")                               \
  V(__BUILTIN_LOG1PL, "__builtin_log1pl")                               \
  V(__BUILTIN_LOG2, "__builtin_log2")                                   \
  V(__BUILTIN_LOG2F, "__builtin_log2f")                                 \
  V(__BUILTIN_LOG2L, "__builtin_log2l")                                 \
  V(__BUILTIN_LOGB, "__builtin_logb")                                   \
  V(__BUILTIN_LOGBF, "__builtin_logbf")                                 \
  V(__BUILTIN_LOGBL, "__builtin_logbl")                                 \
  V(__BUILTIN_LOGF, "__builtin_logf")                                   \
  V(__BUILTIN_LOGL, "__builtin_logl")                                   \
  V(__BUILTIN_LRINT, "__builtin_lrint")                                 \
  V(__BUILTIN_LRINTF, "__builtin_lrintf")                               \
  V(__BUILTIN_LRINTL, "__builtin_lrintl")                               \
  V(__BUILTIN_LROUND, "__builtin_lround")                               \
  V(__BUILTIN_LROUNDF, "__builtin_lroundf")                             \
  V(__BUILTIN_LROUNDL, "__builtin_lroundl")                             \
  V(__BUILTIN_MEMCCPY, "__builtin_memccpy")                             \
  V(__BUILTIN_MEMCHR, "__builtin_memchr")                               \
  V(__BUILTIN_MEMCMP, "__builtin_memcmp")                               \
  V(__BUILTIN_MEMCPY, "__builtin_memcpy")                               \
  V(__BUILTIN_MEMMOVE, "__builtin_memmove")                             \
  V(__BUILTIN_MEMPCPY, "__builtin_mempcpy")                             \
  V(__BUILTIN_MEMSET, "__builtin_memset")                               \
  V(__BUILTIN_MODF, "__builtin_modf")                                   \
  V(__BUILTIN_MODFF, "__builtin_modff")                                 \
  V(__BUILTIN_MODFL, "__builtin_modfl")                                 \
  V(__BUILTIN_NAN, "__builtin_nan")                                     \
  V(__BUILTIN_NANF, "__builtin_nanf")                                   \
  V(__BUILTIN_NANL, "__builtin_nanl")                                   \
  V(__BUILTIN_NEARBYINT, "__builtin_nearbyint")                         \
  V(__BUILTIN_NEARBYINTF, "__builtin_nearbyintf")                       \
  V(__BUILTIN_NEARBYINTL, "__builtin_nearbyintl")                       \
  V(__BUILTIN_NEXTAFTER, "__builtin_nextafter")                         \
  V(__BUILTIN_NEXTAFTERF, "__builtin_nextafterf")                       \
  V(__BUILTIN_NEXTAFTERL, "__builtin_nextafterl")                       \
  V(__BUILTIN_NEXTTOWARD, "__builtin_nexttoward")                       \
  V(__BUILTIN_NEXTTOWARDF, "__builtin_nexttowardf")                     \
  V(__BUILTIN_NEXTTOWARDL, "__builtin_nexttowardl")                     \
  V(__BUILTIN_POW, "__builtin_pow")                                     \
  V(__BUILTIN_POWF, "__builtin_powf")                                   \
  V(__BUILTIN_POWL, "__builtin_powl")                                   \
  V(__BUILTIN_REMAINDER, "__builtin_remainder")                         \
  V(__BUILTIN_REMAINDERF, "__builtin_remainderf")                       \
  V(__BUILTIN_REMAINDERL, "__builtin_remainderl")                       \
  V(__BUILTIN_REMQUO, "__builtin_remquo")                               \
  V(__BUILTIN_REMQUOF, "__builtin_remquof")                             \
  V(__BUILTIN_REMQUOL, "__builtin_remquol")                             \
  V(__BUILTIN_RINDEX, "__builtin_rindex")                               \
  V(__BUILTIN_RINT, "__builtin_rint")                                   \
  V(__BUILTIN_RINTF, "__builtin_rintf")                                 \
  V(__BUILTIN_RINTL, "__builtin_rintl")                                 \
  V(__BUILTIN_ROUND, "__builtin_round")                                 \
  V(__BUILTIN_ROUNDEVEN, "__builtin_roundeven")                         \
  V(__BUILTIN_ROUNDEVENF, "__builtin_roundevenf")                       \
  V(__BUILTIN_ROUNDEVENL, "__builtin_roundevenl")                       \
  V(__BUILTIN_ROUNDF, "__builtin_roundf")                               \
  V(__BUILTIN_ROUNDL, "__builtin_roundl")                               \
  V(__BUILTIN_SCALBLN, "__builtin_scalbln")                             \
  V(__BUILTIN_SCALBLNF, "__builtin_scalblnf")                           \
  V(__BUILTIN_SCALBLNL, "__builtin_scalblnl")                           \
  V(__BUILTIN_SCALBN, "__builtin_scalbn")                               \
  V(__BUILTIN_SCALBNF, "__builtin_scalbnf")                             \
  V(__BUILTIN_SCALBNL, "__builtin_scalbnl")                             \
  V(__BUILTIN_SIN, "__builtin_sin")                                     \
  V(__BUILTIN_SINCOS, "__builtin_sincos")                               \
  V(__BUILTIN_SINCOSF, "__builtin_sincosf")                             \
  V(__BUILTIN_SINCOSL, "__builtin_sincosl")                             \
  V(__BUILTIN_SINF, "__builtin_sinf")                                   \
  V(__BUILTIN_SINH, "__builtin_sinh")                                   \
  V(__BUILTIN_SINHF, "__builtin_sinhf")                                 \
  V(__BUILTIN_SINHL, "__builtin_sinhl")                                 \
  V(__BUILTIN_SINL, "__builtin_sinl")                                   \
  V(__BUILTIN_SQRT, "__builtin_sqrt")                                   \
  V(__BUILTIN_SQRTF, "__builtin_sqrtf")                                 \
  V(__BUILTIN_SQRTL, "__builtin_sqrtl")                                 \
  V(__BUILTIN_STPCPY, "__builtin_stpcpy")                               \
  V(__BUILTIN_STPNCPY, "__builtin_stpncpy")                             \
  V(__BUILTIN_STRCASECMP, "__builtin_strcasecmp")                       \
  V(__BUILTIN_STRCAT, "__builtin_strcat")                               \
  V(__BUILTIN_STRCHR, "__builtin_strchr")                               \
  V(__BUILTIN_STRCMP, "__builtin_strcmp")                               \
  V(__BUILTIN_STRCPY, "__builtin_strcpy")                               \
  V(__BUILTIN_STRCSPN, "__builtin_strcspn")                             \
  V(__BUILTIN_STRDUP, "__builtin_strdup")                               \
  V(__BUILTIN_STRERROR, "__builtin_strerror")                           \
  V(__BUILTIN_STRLCAT, "__builtin_strlcat")                             \
  V(__BUILTIN_STRLCPY, "__builtin_strlcpy")                             \
  V(__BUILTIN_STRLEN, "__builtin_strlen")                               \
  V(__BUILTIN_STRNCASECMP, "__builtin_strncasecmp")                     \
  V(__BUILTIN_STRNCAT, "__builtin_strncat")                             \
  V(__BUILTIN_STRNCMP, "__builtin_strncmp")                             \
  V(__BUILTIN_STRNCPY, "__builtin_strncpy")                             \
  V(__BUILTIN_STRNDUP, "__builtin_strndup")                             \
  V(__BUILTIN_STRPBRK, "__builtin_strpbrk")                             \
  V(__BUILTIN_STRRCHR, "__builtin_strrchr")                             \
  V(__BUILTIN_STRSPN, "__builtin_strspn")                               \
  V(__BUILTIN_STRSTR, "__builtin_strstr")                               \
  V(__BUILTIN_STRTOK, "__builtin_strtok")                               \
  V(__BUILTIN_STRXFRM, "__builtin_strxfrm")                             \
  V(__BUILTIN_TAN, "__builtin_tan")                                     \
  V(__BUILTIN_TANF, "__builtin_tanf")                                   \
  V(__BUILTIN_TANH, "__builtin_tanh")                                   \
  V(__BUILTIN_TANHF, "__builtin_tanhf")                                 \
  V(__BUILTIN_TANHL, "__builtin_tanhl")                                 \
  V(__BUILTIN_TANL, "__builtin_tanl")                                   \
  V(__BUILTIN_TGAMMA, "__builtin_tgamma")                               \
  V(__BUILTIN_TGAMMAF, "__builtin_tgammaf")                             \
  V(__BUILTIN_TGAMMAL, "__builtin_tgammal")                             \
  V(__BUILTIN_TRUNC, "__builtin_trunc")                                 \
  V(__BUILTIN_TRUNCF, "__builtin_truncf")                               \
  V(__BUILTIN_TRUNCL, "__builtin_truncl")                               \
  V(__BUILTIN_UNREACHABLE, "__builtin_unreachable")                     \
  V(__BUILTIN_VA_COPY, "__builtin_va_copy")                             \
  V(__BUILTIN_VA_END, "__builtin_va_end")                               \
  V(__BUILTIN_VA_START, "__builtin_va_start")

#define FOR_EACH_BUILTIN_TEMPLATE(V)          \
  V(__MAKE_INTEGER_SEQ, "__make_integer_seq") \
  V(__TYPE_PACK_ELEMENT, "__type_pack_element")

#define FOR_EACH_TOKEN_ALIAS(V)    \
  V(RESTRICT, __RESTRICT__)        \
  V(__ALIGNOF__, ALIGNOF)          \
  V(__ALIGNOF, ALIGNOF)            \
  V(__ASM__, ASM)                  \
  V(__ASM, ASM)                    \
  V(__ATTRIBUTE, __ATTRIBUTE__)    \
  V(__DECLTYPE__, DECLTYPE)        \
  V(__DECLTYPE, DECLTYPE)          \
  V(__INLINE__, INLINE)            \
  V(__INLINE, INLINE)              \
  V(__RESTRICT, __RESTRICT__)      \
  V(__TYPEOF__, TYPEOF)            \
  V(__TYPEOF, TYPEOF)              \
  V(__VOLATILE__, VOLATILE)        \
  V(__VOLATILE, VOLATILE)          \
  V(_ALIGNAS, ALIGNAS)             \
  V(_ALIGNOF, ALIGNOF)             \
  V(_ASM, ASM)                     \
  V(_BOOL, BOOL)                   \
  V(_STATIC_ASSERT, STATIC_ASSERT) \
  V(_THREAD_LOCAL, THREAD_LOCAL)   \
  V(AND_EQ, AMP_EQUAL)             \
  V(AND, AMP_AMP)                  \
  V(BITAND, AMP)                   \
  V(BITOR, BAR)                    \
  V(COMPL, TILDE)                  \
  V(NOT_EQ, EXCLAIM_EQUAL)         \
  V(NOT, EXCLAIM)                  \
  V(OR_EQ, BAR_EQUAL)              \
  V(OR, BAR_BAR)                   \
  V(XOR_EQ, CARET_EQUAL)           \
  V(XOR, CARET)

#define FOR_EACH_TOKEN(V) \
  FOR_EACH_BASE_TOKEN(V)  \
  FOR_EACH_OPERATOR(V)    \
  FOR_EACH_KEYWORD(V)

// clang-format off
#define TOKEN_ENUM(tk, _) T_##tk,
#define TOKEN_ALIAS_ENUM(tk, other) T_##tk = T_##other,
enum class TokenKind : std::uint8_t {
  FOR_EACH_TOKEN(TOKEN_ENUM)
  FOR_EACH_TOKEN_ALIAS(TOKEN_ALIAS_ENUM)
};

enum class BuiltinTypeTraitKind {
  T_NONE,
  FOR_EACH_BUILTIN_TYPE_TRAIT(TOKEN_ENUM)
};

enum class UnaryBuiltinTypeKind {
  T_NONE,
  FOR_EACH_UNARY_BUILTIN_TYPE_TRAIT(TOKEN_ENUM)
};

enum class BinaryBuiltinTypeKind {
  T_NONE,
  FOR_EACH_BINARY_BUILTIN_TYPE_TRAIT(TOKEN_ENUM)
};

enum class BuiltinFunctionKind {
  T_NONE,
  FOR_EACH_BUILTIN_FUNCTION(TOKEN_ENUM)
};

enum class BuiltinTemplateKind {
  T_NONE,
  FOR_EACH_BUILTIN_TEMPLATE(TOKEN_ENUM)
};

#undef TOKEN_ENUM
#undef TOKEN_ALIAS_ENUM
// clang-format on

[[nodiscard]] inline auto is_keyword(TokenKind tk) -> bool {
  return tk >= TokenKind::T__ATOMIC && tk <= TokenKind::T_WHILE;
}

[[nodiscard]] inline auto get_underlying_binary_op(TokenKind op) -> TokenKind {
  switch (op) {
    case TokenKind::T_STAR_EQUAL:
      return TokenKind::T_STAR;
    case TokenKind::T_SLASH_EQUAL:
      return TokenKind::T_SLASH;
    case TokenKind::T_PERCENT_EQUAL:
      return TokenKind::T_PERCENT;
    case TokenKind::T_PLUS_EQUAL:
      return TokenKind::T_PLUS;
    case TokenKind::T_MINUS_EQUAL:
      return TokenKind::T_MINUS;
    case TokenKind::T_LESS_LESS_EQUAL:
      return TokenKind::T_LESS_LESS;
    case TokenKind::T_AMP_EQUAL:
      return TokenKind::T_AMP;
    case TokenKind::T_CARET_EQUAL:
      return TokenKind::T_CARET;
    case TokenKind::T_BAR_EQUAL:
      return TokenKind::T_BAR;
    case TokenKind::T_GREATER_GREATER_EQUAL:
      return TokenKind::T_GREATER_GREATER;
    default:
      return TokenKind::T_EOF_SYMBOL;
  }  // switch
}

}  // namespace cxx
