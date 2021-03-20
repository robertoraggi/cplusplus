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

#include <cxx/translation_unit.h>

// cxx
#include <cxx/arena.h>
#include <cxx/control.h>
#include <cxx/lexer.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/parser.h>
#include <cxx/preprocessor.h>
#include <utf8.h>

#include <cassert>

namespace cxx {

TranslationUnit::TranslationUnit(Control* control) : control_(control) {
  arena_ = new Arena();
  preprocessor_ = std::make_unique<Preprocessor>(control_);
}

TranslationUnit::~TranslationUnit() { delete arena_; }

void TranslationUnit::setSource(std::string source, std::string fileName) {
  fileName_ = std::move(fileName);
  preprocessor_->preprocess(std::move(source), fileName_, tokens_);
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

const std::string& TranslationUnit::tokenText(SourceLocation loc) const {
  const auto& tk = tokenAt(loc);
  switch (tk.kind()) {
    case TokenKind::T_IDENTIFIER:
      return tk.value().idValue->name();

    case TokenKind::T_STRING_LITERAL:
    case TokenKind::T_CHARACTER_LITERAL:
    case TokenKind::T_INTEGER_LITERAL:
      return tk.value().literalValue->value();

    default:
      return Token::spell(tk.kind());
  }  // switch
}

void TranslationUnit::getTokenStartPosition(SourceLocation loc, unsigned* line,
                                            unsigned* column,
                                            std::string_view* fileName) const {
  preprocessor_->getTokenStartPosition(tokenAt(loc), line, column, fileName);
}

void TranslationUnit::getTokenEndPosition(SourceLocation loc, unsigned* line,
                                          unsigned* column,
                                          std::string_view* fileName) const {
  preprocessor_->getTokenEndPosition(tokenAt(loc), line, column, fileName);
}

bool TranslationUnit::parse() {
  Parser parse(this);
  return parse(ast_);
}

}  // namespace cxx
