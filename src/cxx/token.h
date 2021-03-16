// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>

#include <cstdint>
#include <string>

#define FOR_EACH_TOKEN(V)                                             \
  V(EOF_SYMBOL, "eof")                                                \
  V(ERROR, "error")                                                   \
  V(IDENTIFIER, "identifier")                                         \
  V(CHARACTER_LITERAL, "character_literal")                           \
  V(FLOATING_POINT_LITERAL, "floating_point_literal")                 \
  V(INTEGER_LITERAL, "integer_literal")                               \
  V(STRING_LITERAL, "string_literal")                                 \
  V(USER_DEFINED_STRING_LITERAL, "user_defined_string_literal")       \
  V(HASH, "#")                                                        \
  V(HASH_HASH, "##")                                                  \
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
  V(MODULE, "module")                                                 \
  V(IMPORT, "import")                                                 \
  V(__INT64, "__int64")                                               \
  V(__INT128, "__int128")                                             \
  V(__FLOAT80, "__float80")                                           \
  V(__FLOAT128, "__float128")                                         \
  V(__ALIGNOF, "__alignof")                                           \
  V(__ALIGNOF__, "__alignof__")                                       \
  V(__ASM__, "__asm__")                                               \
  V(__ASM, "__asm")                                                   \
  V(__ATTRIBUTE__, "__attribute__")                                   \
  V(__ATTRIBUTE, "__attribute")                                       \
  V(__BUILTIN_VA_LIST, "__builtin_va_list")                           \
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
  V(_COMPLEX, "_Complex")                                             \
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
enum struct TokenKind : uint8_t { FOR_EACH_TOKEN(TOKEN_ENUM) };
#undef TOKEN_ENUM

union TokenValue {
  const void* ptrValue;
  std::string* stringValue;
  const Identifier* idValue;
  const Literal* literalValue;
  TokenKind tokenKindValue;
  int intValue;
};

class Token {
 public:
  Token(const Token&) = default;
  Token& operator=(const Token&) = default;

  Token(Token&&) = default;
  Token& operator=(Token&&) = default;

  Token() = default;

  explicit Token(TokenKind kind, unsigned offset = 0, unsigned length = 0,
                 TokenValue value = {});

  inline TokenKind kind() const;
  inline void setKind(TokenKind kind);

  inline TokenValue value() const;
  inline void setValue(TokenValue value);

  inline unsigned offset() const;
  inline unsigned length() const;

  inline unsigned fileId() const;
  inline void setFileId(unsigned fileId);

  inline bool startOfLine() const;
  inline void setStartOfLine(bool startOfLine);

  inline bool leadingSpace() const;
  inline void setLeadingSpace(bool leadingSpace);

  explicit operator bool() const;
  explicit operator TokenKind() const;

  bool is(TokenKind k) const;
  bool isNot(TokenKind k) const;

  const std::string& spell() const;
  const std::string& name() const;

  static const std::string& spell(TokenKind kind);
  static const std::string& name(TokenKind kind);

 private:
  TokenKind kind_ : 8;
  uint32_t startOfLine_ : 1;
  uint32_t leadingSpace_ : 1;
  uint32_t fileId_ : 10;
  uint32_t length_ : 16;
  uint32_t offset_ : 28;
  TokenValue value_;
};

inline Token::Token(TokenKind kind, unsigned offset, unsigned length,
                    TokenValue value)
    : kind_(kind),
      startOfLine_(0),
      leadingSpace_(0),
      fileId_(0),
      length_(length),
      offset_(offset),
      value_(value) {}

inline TokenKind Token::kind() const { return kind_; }

inline void Token::setKind(TokenKind kind) { kind_ = kind; }

inline TokenValue Token::value() const { return value_; }

inline void Token::setValue(TokenValue value) { value_ = value; }

inline unsigned Token::offset() const { return offset_; }

inline unsigned Token::length() const { return length_; }

inline unsigned Token::fileId() const { return fileId_; }

inline void Token::setFileId(unsigned fileId) { fileId_ = fileId; }

inline bool Token::startOfLine() const { return startOfLine_; }

inline void Token::setStartOfLine(bool startOfLine) {
  startOfLine_ = startOfLine;
}

inline bool Token::leadingSpace() const { return leadingSpace_; }

inline void Token::setLeadingSpace(bool leadingSpace) {
  leadingSpace_ = leadingSpace;
}

inline Token::operator bool() const {
  return kind() != TokenKind::T_EOF_SYMBOL;
}

inline Token::operator TokenKind() const { return kind(); }

inline bool Token::is(TokenKind k) const { return kind() == k; }

inline bool Token::isNot(TokenKind k) const { return kind() != k; }

}  // namespace cxx
