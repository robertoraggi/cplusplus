// Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cstdint>
#include <string_view>

#include "cxx-fwd.h"

#define FOR_EACH_TOKEN(V)                                             \
  V(EOF_SYMBOL, "eof")                                                \
  V(ERROR, "error")                                                   \
  V(IDENTIFIER, "identifier")                                         \
  V(CHARACTER_LITERAL, "character_literal")                           \
  V(FLOATING_POINT_LITERAL, "floating_point_literal")                 \
  V(INTEGER_LITERAL, "integer_literal")                               \
  V(STRING_LITERAL, "string_literal")                                 \
  V(USER_DEFINED_LITERAL, "user_defined_literal")                     \
  V(USER_DEFINED_STRING_LITERAL, "user_defined_string_literal")       \
  V(EXCLAIM, "!")                                                     \
  V(EXCLAIM_EQUAL, "!=")                                              \
  V(PERCENT, "%")                                                     \
  V(PERCENT_EQUAL, "%=")                                              \
  V(AMP, "&")                                                         \
  V(AMP_AMP, "&&")                                                    \
  V(AMP_EQUAL, "&=")                                                  \
  V(LPAREN, "(")                                                      \
  V(RPAREN, ")")                                                      \
  V(STAR, "*")                                                        \
  V(STAR_EQUAL, "*=")                                                 \
  V(PLUS, "+")                                                        \
  V(PLUS_PLUS, "++")                                                  \
  V(PLUS_EQUAL, "+=")                                                 \
  V(COMMA, ",")                                                       \
  V(MINUS, "-")                                                       \
  V(MINUS_MINUS, "--")                                                \
  V(MINUS_EQUAL, "-=")                                                \
  V(MINUS_GREATER, "->")                                              \
  V(MINUS_GREATER_STAR, "->*")                                        \
  V(DOT, ".")                                                         \
  V(DOT_STAR, ".*")                                                   \
  V(DOT_DOT_DOT, "...")                                               \
  V(SLASH, "/")                                                       \
  V(SLASH_EQUAL, "/=")                                                \
  V(COLON, ":")                                                       \
  V(COLON_COLON, "::")                                                \
  V(SEMICOLON, ";")                                                   \
  V(LESS, "<")                                                        \
  V(LESS_LESS, "<<")                                                  \
  V(LESS_LESS_EQUAL, "<<=")                                           \
  V(LESS_EQUAL, "<=")                                                 \
  V(LESS_EQUAL_GREATER, "<=>")                                        \
  V(EQUAL, "=")                                                       \
  V(EQUAL_EQUAL, "==")                                                \
  V(GREATER, ">")                                                     \
  V(GREATER_EQUAL, ">=")                                              \
  V(GREATER_GREATER, ">>")                                            \
  V(GREATER_GREATER_EQUAL, ">>=")                                     \
  V(QUESTION, "?")                                                    \
  V(LBRACKET, "[")                                                    \
  V(RBRACKET, "]")                                                    \
  V(CARET, "^")                                                       \
  V(CARET_EQUAL, "^=")                                                \
  V(LBRACE, "{")                                                      \
  V(BAR, "|")                                                         \
  V(BAR_EQUAL, "|=")                                                  \
  V(BAR_BAR, "||")                                                    \
  V(RBRACE, "}")                                                      \
  V(TILDE, "~")                                                       \
  V(NEW_ARRAY, "new[]")                                               \
  V(DELETE_ARRAY, "delete[]")                                         \
  V(__INT64, "__int64")                                               \
  V(__INT128, "__int128")                                             \
  V(__FLOAT80, "__float80")                                           \
  V(__FLOAT128, "__float128")                                         \
  V(__ALIGNOF, "__alignof")                                           \
  V(__ASM__, "__asm__")                                               \
  V(__ASM, "__asm")                                                   \
  V(__ATTRIBUTE__, "__attribute__")                                   \
  V(__ATTRIBUTE, "__attribute")                                       \
  V(__COMPLEX__, "__complex__")                                       \
  V(__DECLTYPE, "__decltype")                                         \
  V(__DECLTYPE__, "__decltype__")                                     \
  V(__EXTENSION__, "__extension__")                                   \
  V(__HAS_UNIQUE_OBJECT_REPRESENTATIONS,                              \
    "__has_unique_object_representations")                            \
  V(__HAS_VIRTUAL_DESTRUCTOR, "__has_virtual_destructor")             \
  V(__IMAG__, "__imag__")                                             \
  V(__INLINE, "__inline")                                             \
  V(__INLINE__, "__inline__")                                         \
  V(__IS_ABSTRACT, "__is_abstract")                                   \
  V(__IS_AGGREGATE, "__is_aggregate")                                 \
  V(__IS_BASE_OF, "__is_base_of")                                     \
  V(__IS_CLASS, "__is_class")                                         \
  V(__IS_CONSTRUCTIBLE, "__is_constructible")                         \
  V(__IS_CONVERTIBLE_TO, "__is_convertible_to")                       \
  V(__IS_EMPTY, "__is_empty")                                         \
  V(__IS_ENUM, "__is_enum")                                           \
  V(__IS_FINAL, "__is_final")                                         \
  V(__IS_FUNCTION, "__is_function")                                   \
  V(__IS_LITERAL, "__is_literal")                                     \
  V(__IS_NOTHROW_ASSIGNABLE, "__is_nothrow_assignable")               \
  V(__IS_NOTHROW_CONSTRUCTIBLE, "__is_nothrow_constructible")         \
  V(__IS_POD, "__is_pod")                                             \
  V(__IS_POLYMORPHIC, "__is_polymorphic")                             \
  V(__IS_SAME, "__is_same")                                           \
  V(__IS_STANDARD_LAYOUT, "__is_standard_layout")                     \
  V(__IS_TRIVIAL, "__is_trivial")                                     \
  V(__IS_TRIVIALLY_ASSIGNABLE, "__is_trivially_assignable")           \
  V(__IS_TRIVIALLY_CONSTRUCTIBLE, "__is_trivially_constructible")     \
  V(__IS_TRIVIALLY_COPYABLE, "__is_trivially_copyable")               \
  V(__IS_TRIVIALLY_DESTRUCTIBLE, "__is_trivially_destructible")       \
  V(__IS_UNION, "__is_union")                                         \
  V(__REAL__, "__real__")                                             \
  V(__REFERENCE_BINDS_TO_TEMPORARY, "__reference_binds_to_temporary") \
  V(__RESTRICT__, "__restrict__")                                     \
  V(__RESTRICT, "__restrict")                                         \
  V(__THREAD, "__thread")                                             \
  V(__TYPEOF__, "__typeof__")                                         \
  V(__TYPEOF, "__typeof")                                             \
  V(__UNDERLYING_TYPE, "__underlying_type")                           \
  V(_ATOMIC, "_Atomic")                                               \
  V(ALIGNAS, "alignas")                                               \
  V(ALIGNOF, "alignof")                                               \
  V(ASM, "asm")                                                       \
  V(AUTO, "auto")                                                     \
  V(BOOL, "bool")                                                     \
  V(BREAK, "break")                                                   \
  V(CASE, "case")                                                     \
  V(CATCH, "catch")                                                   \
  V(CHAR, "char")                                                     \
  V(CHAR16_T, "char16_t")                                             \
  V(CHAR32_T, "char32_t")                                             \
  V(CHAR8_T, "char8_t")                                               \
  V(CLASS, "class")                                                   \
  V(CO_AWAIT, "co_await")                                             \
  V(CO_RETURN, "co_return")                                           \
  V(CO_YIELD, "co_yield")                                             \
  V(CONCEPT, "concept")                                               \
  V(CONST, "const")                                                   \
  V(CONST_CAST, "const_cast")                                         \
  V(CONSTEVAL, "consteval")                                           \
  V(CONSTEXPR, "constexpr")                                           \
  V(CONSTINIT, "constinit")                                           \
  V(CONTINUE, "continue")                                             \
  V(DECLTYPE, "decltype")                                             \
  V(DEFAULT, "default")                                               \
  V(DELETE, "delete")                                                 \
  V(DO, "do")                                                         \
  V(DOUBLE, "double")                                                 \
  V(DYNAMIC_CAST, "dynamic_cast")                                     \
  V(ELSE, "else")                                                     \
  V(ENUM, "enum")                                                     \
  V(EXPLICIT, "explicit")                                             \
  V(EXPORT, "export")                                                 \
  V(EXTERN, "extern")                                                 \
  V(FALSE, "false")                                                   \
  V(FLOAT, "float")                                                   \
  V(FOR, "for")                                                       \
  V(FRIEND, "friend")                                                 \
  V(GOTO, "goto")                                                     \
  V(IF, "if")                                                         \
  V(INLINE, "inline")                                                 \
  V(INT, "int")                                                       \
  V(LONG, "long")                                                     \
  V(MUTABLE, "mutable")                                               \
  V(NAMESPACE, "namespace")                                           \
  V(NEW, "new")                                                       \
  V(NOEXCEPT, "noexcept")                                             \
  V(NULLPTR, "nullptr")                                               \
  V(OPERATOR, "operator")                                             \
  V(PRIVATE, "private")                                               \
  V(PROTECTED, "protected")                                           \
  V(PUBLIC, "public")                                                 \
  V(REINTERPRET_CAST, "reinterpret_cast")                             \
  V(REQUIRES, "requires")                                             \
  V(RETURN, "return")                                                 \
  V(SHORT, "short")                                                   \
  V(SIGNED, "signed")                                                 \
  V(SIZEOF, "sizeof")                                                 \
  V(STATIC, "static")                                                 \
  V(STATIC_ASSERT, "static_assert")                                   \
  V(STATIC_CAST, "static_cast")                                       \
  V(STRUCT, "struct")                                                 \
  V(SWITCH, "switch")                                                 \
  V(TEMPLATE, "template")                                             \
  V(THIS, "this")                                                     \
  V(THREAD_LOCAL, "thread_local")                                     \
  V(THROW, "throw")                                                   \
  V(TRUE, "true")                                                     \
  V(TRY, "try")                                                       \
  V(TYPEDEF, "typedef")                                               \
  V(TYPEID, "typeid")                                                 \
  V(TYPENAME, "typename")                                             \
  V(UNION, "union")                                                   \
  V(UNSIGNED, "unsigned")                                             \
  V(USING, "using")                                                   \
  V(VIRTUAL, "virtual")                                               \
  V(VOID, "void")                                                     \
  V(VOLATILE, "volatile")                                             \
  V(WCHAR_T, "wchar_t")                                               \
  V(WHILE, "while")

namespace cxx {

#define TOKEN_ENUM(tk, _) T_##tk,
enum struct TokenKind : uint16_t { FOR_EACH_TOKEN(TOKEN_ENUM) };
#undef TOKEN_ENUM

extern const char* token_spell[];
extern const char* token_name[];

class Token {
  friend class TranslationUnit;
  TokenKind kind_ : 16;
  uint16_t startOfLine_ : 1;
  uint16_t leadingSpace_ : 1;
  unsigned offset_;
  const void* priv_;

 public:
  Token(TokenKind kind = TokenKind::T_ERROR, unsigned offset = 0,
        const void* priv = nullptr)
      : kind_(kind), offset_(offset), priv_(priv) {}
  inline TokenKind kind() const { return kind_; }
  inline unsigned offset() const { return offset_; }
  inline bool startOfLine() const { return startOfLine_; }
  inline bool leadingSpace() const { return leadingSpace_; }

  static std::string_view spell(TokenKind kind);
  std::string_view spell() const;

  static std::string_view name(TokenKind kind);
  std::string_view name() const;
};

}  // namespace cxx
