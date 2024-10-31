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

#include <cxx/translation_unit.h>

// cxx
#include <cxx/arena.h>
#include <cxx/control.h>
#include <cxx/lexer.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/parser.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>

#ifndef CXX_NO_FLATBUFFERS
#include <cxx/private/ast_decoder.h>
#include <cxx/private/ast_encoder.h>
#endif

#include <utf8/unchecked.h>

#include <ostream>

namespace cxx {

TranslationUnit::TranslationUnit(Control* control,
                                 DiagnosticsClient* diagnosticsClient)
    : control_(control), diagnosticsClient_(diagnosticsClient) {
  arena_ = std::make_unique<Arena>();
  globalNamespace_ = control_->newNamespaceSymbol(nullptr, {});

  preprocessor_ = std::make_unique<Preprocessor>(control_, diagnosticsClient_);

  if (diagnosticsClient_) {
    diagnosticsClient_->setPreprocessor(preprocessor_.get());
  }
}

TranslationUnit::~TranslationUnit() = default;

auto TranslationUnit::diagnosticsClient() const -> DiagnosticsClient* {
  return diagnosticsClient_;
}

auto TranslationUnit::changeDiagnosticsClient(
    DiagnosticsClient* diagnosticsClient) -> DiagnosticsClient* {
  std::swap(diagnosticsClient_, diagnosticsClient);
  return diagnosticsClient;
}

void TranslationUnit::setSource(std::string source, std::string fileName) {
  fileName_ = std::move(fileName);
  preprocessor_->preprocess(std::move(source), fileName_, tokens_);
}

auto TranslationUnit::tokenLength(SourceLocation loc) const -> int {
  const auto& tk = tokenAt(loc);
  if (tk.kind() == TokenKind::T_IDENTIFIER) {
    const std::string* id = tk.value().stringValue;
    return static_cast<int>(id->size());
  }
  return static_cast<int>(Token::spell(tk.kind()).size());
}

auto TranslationUnit::identifier(SourceLocation loc) const
    -> const Identifier* {
  const auto& tk = tokenAt(loc);
  return tk.value().idValue;
}

auto TranslationUnit::literal(SourceLocation loc) const -> const Literal* {
  const auto& tk = tokenAt(loc);
  return tk.value().literalValue;
}

auto TranslationUnit::tokenText(SourceLocation loc) const
    -> const std::string& {
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

void TranslationUnit::parse(const ParserConfiguration& config) {
  Parser parse(this);
  parse.setConfig(config);
  parse(ast_);
}

auto TranslationUnit::globalScope() const -> Scope* {
  if (!globalNamespace_) return nullptr;
  return globalNamespace_->scope();
}

auto TranslationUnit::load(std::span<const std::uint8_t> data) -> bool {
#ifndef CXX_NO_FLATBUFFERS
  ASTDecoder decode{this};
  return decode(data);
#else
  return false;
#endif
}

auto TranslationUnit::serialize(std::ostream& out) -> bool {
  return serialize([&out](auto data) {
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
  });
}

auto TranslationUnit::serialize(
    const std::function<void(std::span<const std::uint8_t>)>& block) -> bool {
#ifndef CXX_NO_FLATBUFFERS
  ASTEncoder encode;
  auto data = encode(this);
  block(data);
  return true;
#else
  return false;
#endif
}

}  // namespace cxx
