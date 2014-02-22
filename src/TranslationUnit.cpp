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

#include "TranslationUnit.h"
#include "Control.h"
#include "Names.h"
#include "KeywordsP.h"
#include "AST.h"
#include "Symbols.h"
#include "Names.h"
#include "Types.h"
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <cassert>

bool yyparse(TranslationUnit* unit, const std::function<void(TranslationUnitAST*)>& consume);

inline void TranslationUnit::yyinp() {
  if (yychar == '\n')
    lines_.push_back(yypos);
  yychar = *yyptr++;
  ++yypos;
}

TokenKind TranslationUnit::yylex(unsigned* offset, const void** priv) {
again:
  while (isspace(yychar))
    yyinp();
  *offset = yypos;
  *priv = nullptr;
  if (yychar == 0)
    return T_EOF_SYMBOL;
  auto ch = yychar;
  yyinp();

  // ### TODO: wide and unicode literals
  if (ch == 'L' && (yychar == '\'' || yychar == '"')) {
    ch = yychar;
    yyinp();
  } else if (toupper(ch) == 'U' && (yychar == '\'' || yychar == '"')) {
    ch = yychar;
    yyinp();
  }


  switch (ch) {
  case '\'':
  case '"': {
    const auto quote = ch;
    yytext = quote;
    while (yychar) {
      if (yychar == quote)
        break;
      else if (yychar == '\\') {
        yytext += yychar;
        yyinp();
        if (yychar) {
          yytext += yychar;
          yyinp();
        }
      } else {
        yytext += yychar;
        yyinp();
      }
    }
    assert(yychar == quote);
    yytext += yychar;
    yyinp();
    *priv = control_->getIdentifier(yytext);
    return quote == '"' ? T_STRING_LITERAL : T_CHAR_LITERAL;
  }

  case '=':
    if (yychar == '=') {
      yyinp();
      return T_EQUAL_EQUAL;
    }
    return T_EQUAL;

  case ',':
    return T_COMMA;

  case '~':
    if (yychar == '=') {
      yyinp();
      return T_TILDE_EQUAL;
    }
    return T_TILDE;

  case '{':
    return T_LBRACE;

  case '}':
    return T_RBRACE;

  case '[':
    return T_LBRACKET;

  case ']':
    return T_RBRACKET;

  case '#':
    while (auto ch = yychar) {
      yyinp();
      if (ch == '\n')
        break;
      if (ch == '\\' && yychar)
        yyinp();
    }
    goto again;
#if 0
    if (yychar == '#') {
      yyinp();
      return T_POUND_POUND;
    }
    return T_POUND;
#endif

  case '(':
    return T_LPAREN;

  case ')':
    return T_RPAREN;

  case ';':
    return T_SEMICOLON;

  case ':':
    if (yychar == ':') {
      yyinp();
      return T_COLON_COLON;
    }
    return T_COLON;

  case '.':
    if (yychar == '.') {
      yyinp();
      if (yychar == '.') {
        yyinp();
        return T_DOT_DOT_DOT;
      }
      return T_ERROR;
    } else if (yychar == '*') {
      yyinp();
      return T_DOT_STAR;
    }
    return T_DOT;

  case '?':
    return T_QUESTION;

  case '*':
    if (yychar == '=') {
      yyinp();
      return T_STAR_EQUAL;
    }
    return T_STAR;

  case '%':
    if (yychar == '=') {
      yyinp();
      return T_PERCENT_EQUAL;
    }
    return T_PERCENT;

  case '^':
    if (yychar == '=') {
      yyinp();
      return T_CARET_EQUAL;
    }
    return T_CARET;

  case '&':
    if (yychar == '=') {
      yyinp();
      return T_AMP_EQUAL;
    } else if (yychar == '&') {
      yyinp();
      return T_AMP_AMP;
    }
    return T_AMP;

  case '|':
    if (yychar == '=') {
      yyinp();
      return T_BAR_EQUAL;
    } else if (yychar == '|') {
      yyinp();
      return T_BAR_BAR;
    }
    return T_BAR;

  case '!':
    if (yychar == '=') {
      yyinp();
      return T_EXCLAIM_EQUAL;
    }
    return T_EXCLAIM;

  case '+':
    if (yychar == '+') {
      yyinp();
      return T_PLUS_PLUS;
    } else if (yychar == '=') {
      yyinp();
      return T_PLUS_EQUAL;
    }
    return T_PLUS;

  case '-':
    if (yychar == '-') {
      yyinp();
      return T_MINUS_MINUS;
    } else if (yychar == '=') {
      yyinp();
      return T_MINUS_EQUAL;
    } else if (yychar == '>') {
      yyinp();
      if (yychar == '*') {
        yyinp();
        return T_MINUS_GREATER_STAR;
      } else {
        return T_MINUS_GREATER;
      }
    }
    return T_MINUS;

  case '<':
    if (yychar == '=') {
      yyinp();
      return T_LESS_EQUAL;
    } else if (yychar == '<') {
      yyinp();
      if (yychar == '=') {
        yyinp();
        return T_LESS_LESS_EQUAL;
      }
      return T_LESS_LESS;
    }
    return T_LESS;

  case '>':
#if 0
    if (yychar == '=') {
      yyinp();
      return T_GREATER_EQUAL;
    } else if (yychar == '>') {
      yyinp();
      if (yychar == '=') {
        yyinp();
        return T_GREATER_GREATER_EQUAL;
      }
      return T_GREATER_GREATER;
    }
#endif
    return T_GREATER;

  case '/':
    if (yychar == '/') {
      yyinp();
      while (yychar && yychar != '\n')
        yyinp();
      goto again;
    } else if (yychar == '*') {
      yyinp();
      while (yychar) {
        if (yychar == '*') {
          yyinp();
          if (yychar == '/') {
            yyinp();
            goto again;
          }
        } else {
          yyinp();
        }
      }
      assert(!"unexpected end of file");
    } else if (yychar == '=') {
      yyinp();
      return T_SLASH_EQUAL;
    }
    return T_SLASH;

  default:
    if (std::isalpha(ch) || ch == '_' || ch == '$') {
      yytext = ch;
      while (std::isalnum(yychar) || yychar == '_' || yychar == '$') {
        yytext += yychar;
        yyinp();
      }

      auto k = classify(yytext.c_str(), yytext.size());
      if (k != T_IDENTIFIER)
        return (TokenKind) k;

      *priv = control_->getIdentifier(yytext);
      return T_IDENTIFIER;
    } else if (std::isdigit(ch)) {
      yytext = ch;
      while (std::isalnum(yychar) || yychar == '.') {
        yytext += yychar;
        yyinp();
      }

      *priv = control_->getIdentifier(yytext);
      return T_INT_LITERAL;
    }
  } // switch
  return T_ERROR;
}

void TranslationUnit::warning(unsigned index, const char* format...) {
  unsigned line, column;
  getTokenStartPosition(index, &line, &column);
  fprintf(stderr, "%s:%d:%d: warning: ", yyfilename.c_str(), line, column);
  va_list args, ap;
  va_start(args, format);
  va_copy(ap, args);
  vfprintf(stderr, format, args);
  va_end(ap);
  va_end(args);
  fprintf(stderr, "\n");
}

void TranslationUnit::error(unsigned index, const char* format...) {
  unsigned line, column;
  getTokenStartPosition(index, &line, &column);
  fprintf(stderr, "%s:%d:%d: error: ", yyfilename.c_str(), line, column);
  va_list args, ap;
  va_start(args, format);
  va_copy(ap, args);
  vfprintf(stderr, format, args);
  va_end(ap);
  va_end(args);
  fprintf(stderr, "\n");
  if (fatalErrors_)
    exit(EXIT_FAILURE);
}

void TranslationUnit::fatal(unsigned index, const char* format...) {
  unsigned line, column;
  getTokenStartPosition(index, &line, &column);
  fprintf(stderr, "%s:%d:%d: fatal: ", yyfilename.c_str(), line, column);
  va_list args, ap;
  va_start(args, format);
  va_copy(ap, args);
  vfprintf(stderr, format, args);
  va_end(ap);
  va_end(args);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

int TranslationUnit::tokenLength(unsigned index) const
{
  auto&& tk = tokens_[index];
  if (tk.kind() == T_IDENTIFIER) {
    const std::string* id = reinterpret_cast<const std::string*>(tk.priv_);
    return id->size();
  }
  return ::strlen(token_spell[tk.kind()]);
}

const char* TranslationUnit::tokenText(unsigned index) const {
  auto&& tk = tokens_[index];
  switch (tk.kind()) {
  case T_IDENTIFIER:
  case T_STRING_LITERAL:
  case T_CHAR_LITERAL:
  case T_INT_LITERAL: {
    const Identifier* id = reinterpret_cast<const Identifier*>(tk.priv_);
    return id->c_str();
  }
  default:
    return token_spell[tk.kind()];
  }
}

const Identifier* TranslationUnit::identifier(unsigned index) const {
  if (! index)
    return 0;
  auto&& tk = tokens_[index];
  return reinterpret_cast<const Identifier*>(tk.priv_);
}

void TranslationUnit::getTokenStartPosition(unsigned index, unsigned* line, unsigned* column) const {
  auto offset = tokens_[index].offset();
  auto it = std::lower_bound(lines_.cbegin(), lines_.cend(), offset);
  if (it != lines_.cbegin()) {
    --it;
    assert(*it <= offset);
    *line = std::distance(lines_.cbegin(), it) + 1;
    *column = offset - *it;
  } else {
    *line = 1;
    *column = offset + 1;
  }
}

void TranslationUnit::tokenize() {
  TokenKind kind;
  tokens_.emplace_back(T_ERROR, 0, nullptr);
  do {
    unsigned offset{0};
    const void* value{nullptr};
    kind = yylex(&offset, &value);
    tokens_.emplace_back(kind, offset, value);
  } while (kind != T_EOF_SYMBOL);
}

bool TranslationUnit::parse(const std::function<void(TranslationUnitAST*)>& consume) {
  if (tokens_.empty())
    tokenize();
  return yyparse(this, consume);
}
