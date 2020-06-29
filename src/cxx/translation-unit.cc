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

#include "translation-unit.h"

#include <cassert>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <functional>

#include "ast.h"
#include "control.h"
#include "lexer.h"
#include "names.h"
#include "symbols.h"
#include "types.h"

// generated keyword classifier
#include "keywords-priv.h"

namespace cxx {

bool yyparse(TranslationUnit* unit,
             const std::function<void(TranslationUnitAST*)>& consume);

void TranslationUnit::initializeLineMap() {
  // ### remove
  lines_.clear();
  lines_.push_back(-1);
  const auto start = yycode.c_str();
  for (auto ptr = start; *ptr; ++ptr) {
    if (*ptr == '\n') {
      lines_.push_back(ptr - start);
    }
  }
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

  const auto start = lines_.at(line - 1) + 1;
  const auto end = line < lines_.size() ? lines_.at(line) : yycode.size();
  std::string textLine = yycode.substr(start, end - start);
  std::string cursor(column - 1, ' ');
  cursor += "^";
  fprintf(stderr, "%s\n", textLine.c_str());
  fprintf(stderr, "%s\n", cursor.c_str());
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

  const auto start = lines_.at(line - 1) + 1;
  const auto end = line < lines_.size() ? lines_.at(line) : yycode.size();
  std::string textLine = yycode.substr(start, end - start);
  std::string cursor(column - 1, ' ');
  cursor += "^";
  fprintf(stderr, "%s\n", textLine.c_str());
  fprintf(stderr, "%s\n", cursor.c_str());
  if (fatalErrors_) exit(EXIT_FAILURE);
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

int TranslationUnit::tokenLength(unsigned index) const {
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
    case T_CHARACTER_LITERAL:
    case T_INTEGER_LITERAL: {
      const Identifier* id = reinterpret_cast<const Identifier*>(tk.priv_);
      return id->c_str();
    }
    default:
      return token_spell[tk.kind()];
  }
}

const Identifier* TranslationUnit::identifier(unsigned index) const {
  if (!index) return 0;
  auto&& tk = tokens_[index];
  return reinterpret_cast<const Identifier*>(tk.priv_);
}

void TranslationUnit::getTokenStartPosition(unsigned index, unsigned* line,
                                            unsigned* column) const {
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
  Lexer lexer(yycode);
  TokenKind kind;
  tokens_.emplace_back(TokenKind::T_ERROR, 0, nullptr);
  do {
    kind = lexer.next();
    const void* value = nullptr;
    switch (kind) {
      case TokenKind::T_IDENTIFIER:
      case TokenKind::T_STRING_LITERAL:
      case TokenKind::T_CHARACTER_LITERAL:
      case TokenKind::T_INTEGER_LITERAL:
      case TokenKind::T_FLOATING_POINT_LITERAL:
        value = control_->getIdentifier(lexer.tokenText());
        break;
      default:
        break;
    }
    tokens_.emplace_back(kind, lexer.tokenPos(), value);
    auto& tok = tokens_.back();
    tok.leadingSpace_ = lexer.tokenLeadingSpace();
    tok.startOfLine_ = lexer.tokenStartOfLine();
  } while (kind != TokenKind::T_EOF_SYMBOL);
}

bool TranslationUnit::parse(
    const std::function<void(TranslationUnitAST*)>& consume) {
  if (tokens_.empty()) tokenize();
  return yyparse(this, consume);
}

}  // namespace cxx
