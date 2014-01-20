// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TOKEN_H
#define TOKEN_H

#define FOR_EACH_TOKEN(V) \
  V(EOF_SYMBOL, "<eof symbol>") \
  V(ERROR, "<error symbol>") \
  V(INT_LITERAL, "<int literal>") \
  V(CHAR_LITERAL, "<char literal>") \
  V(STRING_LITERAL, "<string literal>") \
  V(IDENTIFIER, "<identifier>") \
  V(AMP, "&") \
  V(AMP_AMP, "&&") \
  V(AMP_EQUAL, "&=") \
  V(BAR, "|") \
  V(BAR_BAR, "||") \
  V(BAR_EQUAL, "|=") \
  V(CARET, "^") \
  V(CARET_EQUAL, "^=") \
  V(COLON, ":") \
  V(COLON_COLON, "::") \
  V(COMMA, ",") \
  V(DOT, ".") \
  V(DOT_DOT_DOT, "...") \
  V(DOT_STAR, ".*") \
  V(EQUAL, "=") \
  V(EQUAL_EQUAL, "==") \
  V(EXCLAIM, "!") \
  V(EXCLAIM_EQUAL, "!=") \
  V(GREATER, ">") \
  V(GREATER_EQUAL, ">=") \
  V(GREATER_GREATER, ">>") \
  V(GREATER_GREATER_EQUAL, ">>=") \
  V(LBRACE, "{") \
  V(LBRACKET, "[") \
  V(LESS, "<") \
  V(LESS_EQUAL, "<=") \
  V(LESS_LESS, "<<") \
  V(LESS_LESS_EQUAL, "<<=") \
  V(LPAREN, "(") \
  V(MINUS, "-") \
  V(MINUS_EQUAL, "-=") \
  V(MINUS_GREATER, "->") \
  V(MINUS_GREATER_STAR, "->*") \
  V(MINUS_MINUS, "--") \
  V(PERCENT, "%") \
  V(PERCENT_EQUAL, "%=") \
  V(PLUS, "+") \
  V(PLUS_EQUAL, "+=") \
  V(PLUS_PLUS, "++") \
  V(POUND, "#") \
  V(POUND_POUND, "##") \
  V(QUESTION, "?") \
  V(RBRACE, "}") \
  V(RBRACKET, "]") \
  V(RPAREN, ")") \
  V(SEMICOLON, ";") \
  V(SLASH, "/") \
  V(SLASH_EQUAL, "/=") \
  V(STAR, "*") \
  V(STAR_EQUAL, "*=") \
  V(TILDE, "~") \
  V(TILDE_EQUAL, "~=") \
  V(ALIGNAS, "alignas") \
  V(ALIGNOF, "alignof") \
  V(ASM, "asm") \
  V(AUTO, "auto") \
  V(BOOL, "bool") \
  V(BREAK, "break") \
  V(CASE, "case") \
  V(CATCH, "catch") \
  V(CHAR, "char") \
  V(CHAR16_T, "char16_t") \
  V(CHAR32_T, "char32_t") \
  V(CLASS, "class") \
  V(CONST, "const") \
  V(CONST_CAST, "const_cast") \
  V(CONSTEXPR, "constexpr") \
  V(CONTINUE, "continue") \
  V(DECLTYPE, "decltype") \
  V(DEFAULT, "default") \
  V(DELETE, "delete") \
  V(DO, "do") \
  V(DOUBLE, "double") \
  V(DYNAMIC_CAST, "dynamic_cast") \
  V(ELSE, "else") \
  V(ENUM, "enum") \
  V(EXPLICIT, "explicit") \
  V(EXPORT, "export") \
  V(EXTERN, "extern") \
  V(FALSE, "false") \
  V(FLOAT, "float") \
  V(FOR, "for") \
  V(FRIEND, "friend") \
  V(GOTO, "goto") \
  V(IF, "if") \
  V(INLINE, "inline") \
  V(INT, "int") \
  V(LONG, "long") \
  V(MUTABLE, "mutable") \
  V(NAMESPACE, "namespace") \
  V(NEW, "new") \
  V(NOEXCEPT, "noexcept") \
  V(NULLPTR, "nullptr") \
  V(OPERATOR, "operator") \
  V(PRIVATE, "private") \
  V(PROTECTED, "protected") \
  V(PUBLIC, "public") \
  V(REGISTER, "register") \
  V(REINTERPRET_CAST, "reintepret_cast") \
  V(RETURN, "return") \
  V(SHORT, "short") \
  V(SIGNED, "signed") \
  V(SIZEOF, "sizeof") \
  V(STATIC, "static") \
  V(STATIC_ASSERT, "static_assert") \
  V(STATIC_CAST, "static_cast") \
  V(STRUCT, "struct") \
  V(SWITCH, "switch") \
  V(TEMPLATE, "template") \
  V(THIS, "this") \
  V(THREAD_LOCAL, "thread_local") \
  V(THROW, "throw") \
  V(TRUE, "true") \
  V(TRY, "try") \
  V(TYPEDEF, "typedef") \
  V(TYPEID, "typeid") \
  V(TYPENAME, "typename") \
  V(UNION, "union") \
  V(UNSIGNED, "unsigned") \
  V(USING, "using") \
  V(VIRTUAL, "virtual") \
  V(VOID, "void") \
  V(VOLATILE, "volatile") \
  V(WCHAR_T, "wchar_t") \
  V(WHILE, "while")

enum TokenKind {
#define TOKEN_ENUM(tk, _) T_##tk,
  FOR_EACH_TOKEN(TOKEN_ENUM)
};

extern const char* token_spell[];

class Token {
  friend class TranslationUnit;
  TokenKind kind_;
  unsigned offset_;
  const void* priv_;
public:
  Token(TokenKind kind = T_ERROR, unsigned offset = 0, const void* priv = 0)
    : kind_(kind), offset_(offset), priv_(priv) {}
  inline TokenKind kind() const { return kind_; }
  inline unsigned offset() const { return offset_; }
};

#endif // TOKEN_H
