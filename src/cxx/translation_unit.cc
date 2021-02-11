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

#include <cxx/arena.h>
#include <cxx/control.h>
#include <cxx/lexer.h>
#include <cxx/parser.h>
#include <cxx/translation_unit.h>
#include <utf8.h>

#include <cassert>

namespace cxx {

TranslationUnit::TranslationUnit(Control* control) : control_(control) {
  arena_ = new Arena();
}

TranslationUnit::~TranslationUnit() { delete arena_; }

void TranslationUnit::initializeLineMap() {
  // ### remove
  lines_.clear();
  lines_.push_back(-1);
  const auto start = code_.c_str();
  for (auto ptr = start; *ptr; ++ptr) {
    if (*ptr == '\n') {
      lines_.push_back(int(ptr - start));
    }
  }
}

int TranslationUnit::tokenLength(SourceLocation loc) const {
  const auto& tk = tokenAt(loc);
  if (tk.kind() == TokenKind::T_IDENTIFIER) {
    const std::string* id = tk.value().stringValue;
    return int(id->size());
  }
  return int(Token::spell(tk.kind()).size());
}

const Identifier* TranslationUnit::identifier(SourceLocation loc) const {
  const auto& tk = tokenAt(loc);
  return tk.value().idValue;
}

std::string_view TranslationUnit::tokenText(SourceLocation loc) const {
  const auto& tk = tokenAt(loc);
  switch (tk.kind()) {
    case TokenKind::T_IDENTIFIER:
    case TokenKind::T_STRING_LITERAL:
    case TokenKind::T_CHARACTER_LITERAL:
    case TokenKind::T_INTEGER_LITERAL: {
      const Identifier* id = tk.value().idValue;
      return id->toString();
    }
    default:
      return Token::spell(tk.kind());
  }
}

void TranslationUnit::getTokenPosition(unsigned offset, unsigned* line,
                                       unsigned* column) const {
  auto it = std::lower_bound(lines_.cbegin(), lines_.cend(), int(offset));
  assert(it != cbegin(lines_));
  --it;
  assert(*it <= int(offset));
  *line = int(std::distance(cbegin(lines_), it) + 1);
  auto start = cbegin(source()) + *it;
  auto end = cbegin(source()) + offset;
  *column = utf8::distance(start, end);
}

void TranslationUnit::getTokenStartPosition(SourceLocation loc, unsigned* line,
                                            unsigned* column) const {
  auto offset = tokenAt(loc).offset();
  getTokenPosition(offset, line, column);
}

void TranslationUnit::getTokenEndPosition(SourceLocation loc, unsigned* line,
                                          unsigned* column) const {
  const auto& tk = tokenAt(loc);
  auto offset = tk.offset() + tk.length();
  getTokenPosition(offset, line, column);
}

void TranslationUnit::tokenize() {
  Lexer lexer(code_);
  TokenKind kind;
  tokens_.emplace_back(TokenKind::T_ERROR);
  do {
    kind = lexer.next();
    TokenValue value;
    switch (kind) {
      case TokenKind::T_IDENTIFIER:
      case TokenKind::T_STRING_LITERAL:
      case TokenKind::T_CHARACTER_LITERAL:
      case TokenKind::T_INTEGER_LITERAL:
      case TokenKind::T_FLOATING_POINT_LITERAL:
        value.idValue = control_->getIdentifier(lexer.tokenText());
        break;
      case TokenKind::T_GREATER:
        value = lexer.tokenValue();
        break;
      default:
        break;
    }
    tokens_.emplace_back(kind, lexer.tokenPos(), lexer.tokenLength(), value);
    auto& tok = tokens_.back();
    tok.leadingSpace_ = lexer.tokenLeadingSpace();
    tok.startOfLine_ = lexer.tokenStartOfLine();
  } while (kind != TokenKind::T_EOF_SYMBOL);
}

bool TranslationUnit::parse() {
  if (tokens_.empty()) tokenize();
  Parser parse;
  return parse(this, ast_);
}

}  // namespace cxx
