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

#pragma once

#include <cxx/cxx_fwd.h>

#include <cstdint>

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
  V(WIDE_STRING_LITERAL, "<wide_string_literal>")

#define FOR_EACH_OPERATOR(V)      \
  V(AMP_AMP, "&&")                \
  V(AMP_EQUAL, "&=")              \
  V(AMP, "&")                     \
  V(BAR_BAR, "||")                \
  V(BAR_EQUAL, "|=")              \
  V(BAR, "|")                     \
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

#define FOR_EACH_KEYWORD(V)                   \
  V(ALIGNAS, "alignas")                       \
  V(ALIGNOF, "alignof")                       \
  V(ASM, "asm")                               \
  V(AUTO, "auto")                             \
  V(BOOL, "bool")                             \
  V(BREAK, "break")                           \
  V(CASE, "case")                             \
  V(CATCH, "catch")                           \
  V(CHAR, "char")                             \
  V(CHAR16_T, "char16_t")                     \
  V(CHAR32_T, "char32_t")                     \
  V(CHAR8_T, "char8_t")                       \
  V(CLASS, "class")                           \
  V(CO_AWAIT, "co_await")                     \
  V(CO_RETURN, "co_return")                   \
  V(CO_YIELD, "co_yield")                     \
  V(CONCEPT, "concept")                       \
  V(CONST_CAST, "const_cast")                 \
  V(CONST, "const")                           \
  V(CONSTEVAL, "consteval")                   \
  V(CONSTEXPR, "constexpr")                   \
  V(CONSTINIT, "constinit")                   \
  V(CONTINUE, "continue")                     \
  V(DECLTYPE, "decltype")                     \
  V(DEFAULT, "default")                       \
  V(DELETE, "delete")                         \
  V(DO, "do")                                 \
  V(DOUBLE, "double")                         \
  V(DYNAMIC_CAST, "dynamic_cast")             \
  V(ELSE, "else")                             \
  V(ENUM, "enum")                             \
  V(EXPLICIT, "explicit")                     \
  V(EXPORT, "export")                         \
  V(EXTERN, "extern")                         \
  V(FALSE, "false")                           \
  V(FLOAT, "float")                           \
  V(FOR, "for")                               \
  V(FRIEND, "friend")                         \
  V(GOTO, "goto")                             \
  V(IF, "if")                                 \
  V(IMPORT, "import")                         \
  V(INLINE, "inline")                         \
  V(INT, "int")                               \
  V(LONG, "long")                             \
  V(MODULE, "module")                         \
  V(MUTABLE, "mutable")                       \
  V(NAMESPACE, "namespace")                   \
  V(NEW, "new")                               \
  V(NOEXCEPT, "noexcept")                     \
  V(NULLPTR, "nullptr")                       \
  V(OPERATOR, "operator")                     \
  V(PRIVATE, "private")                       \
  V(PROTECTED, "protected")                   \
  V(PUBLIC, "public")                         \
  V(REINTERPRET_CAST, "reinterpret_cast")     \
  V(REQUIRES, "requires")                     \
  V(RETURN, "return")                         \
  V(SHORT, "short")                           \
  V(SIGNED, "signed")                         \
  V(SIZEOF, "sizeof")                         \
  V(STATIC_ASSERT, "static_assert")           \
  V(STATIC_CAST, "static_cast")               \
  V(STATIC, "static")                         \
  V(STRUCT, "struct")                         \
  V(SWITCH, "switch")                         \
  V(TEMPLATE, "template")                     \
  V(THIS, "this")                             \
  V(THREAD_LOCAL, "thread_local")             \
  V(THROW, "throw")                           \
  V(TRUE, "true")                             \
  V(TRY, "try")                               \
  V(TYPEDEF, "typedef")                       \
  V(TYPEID, "typeid")                         \
  V(TYPENAME, "typename")                     \
  V(UNION, "union")                           \
  V(UNSIGNED, "unsigned")                     \
  V(USING, "using")                           \
  V(VIRTUAL, "virtual")                       \
  V(VOID, "void")                             \
  V(VOLATILE, "volatile")                     \
  V(WCHAR_T, "wchar_t")                       \
  V(WHILE, "while")                           \
  V(__ATTRIBUTE__, "__attribute__")           \
  V(__BUILTIN_BIT_CAST, "__builtin_bit_cast") \
  V(__BUILTIN_VA_ARG, "__builtin_va_arg")     \
  V(__BUILTIN_VA_LIST, "__builtin_va_list")   \
  V(__COMPLEX__, "__complex__")               \
  V(__EXTENSION__, "__extension__")           \
  V(__FLOAT128, "__float128")                 \
  V(__FLOAT80, "__float80")                   \
  V(__IMAG__, "__imag__")                     \
  V(__INT128, "__int128")                     \
  V(__INT64, "__int64")                       \
  V(__REAL__, "__real__")                     \
  V(__RESTRICT__, "__restrict__")             \
  V(__THREAD, "__thread")                     \
  V(__UNDERLYING_TYPE, "__underlying_type")   \
  V(_ATOMIC, "_Atomic")                       \
  V(_COMPLEX, "_Complex")

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
  V(__IS_UNBOUNDED_ARRAY, "__is_unbounded_array")                 \
  V(__IS_UNION, "__is_union")                                     \
  V(__IS_UNSIGNED, "__is_unsigned")                               \
  V(__IS_VOID, "__is_void")                                       \
  V(__IS_VOLATILE, "__is_volatile")

#define FOR_EACH_TOKEN_ALIAS(V) \
  V(AND_EQ, AMP_EQUAL)          \
  V(AND, AMP_AMP)               \
  V(BITAND, AMP)                \
  V(BITOR, BAR)                 \
  V(COMPL, TILDE)               \
  V(NOT_EQ, EXCLAIM_EQUAL)      \
  V(NOT, EXCLAIM)               \
  V(OR_EQ, BAR_EQUAL)           \
  V(OR, BAR_BAR)                \
  V(XOR_EQ, CARET_EQUAL)        \
  V(XOR, CARET)                 \
  V(__ALIGNOF__, ALIGNOF)       \
  V(__ALIGNOF, ALIGNOF)         \
  V(__ASM__, ASM)               \
  V(__ASM, ASM)                 \
  V(__ATTRIBUTE, __ATTRIBUTE__) \
  V(__DECLTYPE__, DECLTYPE)     \
  V(__DECLTYPE, DECLTYPE)       \
  V(__INLINE__, INLINE)         \
  V(__INLINE, INLINE)           \
  V(__RESTRICT, __RESTRICT__)   \
  V(__TYPEOF__, DECLTYPE)       \
  V(__TYPEOF, DECLTYPE)         \
  V(__VOLATILE__, VOLATILE)     \
  V(__VOLATILE, VOLATILE)       \
  V(_ALIGNOF, ALIGNOF)          \
  V(_STATIC_ASSERT, STATIC_ASSERT)

#define FOR_EACH_TOKEN(V) \
  FOR_EACH_BASE_TOKEN(V)  \
  FOR_EACH_OPERATOR(V)    \
  FOR_EACH_KEYWORD(V)

// clang-format off
#define TOKEN_ENUM(tk, _) T_##tk,
#define TOKEN_ALIAS_ENUM(tk, other) T_##tk = T_##other,
enum struct TokenKind : std::uint8_t {
  FOR_EACH_TOKEN(TOKEN_ENUM)
  FOR_EACH_TOKEN_ALIAS(TOKEN_ALIAS_ENUM)
};

enum struct BuiltinTypeTraitKind {
  T_NONE,
  FOR_EACH_BUILTIN_TYPE_TRAIT(TOKEN_ENUM)
};

#undef TOKEN_ENUM
#undef TOKEN_ALIAS_ENUM
// clang-format on

}  // namespace cxx
