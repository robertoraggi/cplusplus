// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

TranslationUnit::TranslationUnit(DiagnosticsClient* diagnosticsClient)
    : diagnosticsClient_(diagnosticsClient) {
  control_ = std::make_unique<Control>();
  arena_ = std::make_unique<Arena>();
  globalNamespace_ = control_->newNamespaceSymbol(nullptr, {});

  preprocessor_ =
      std::make_unique<Preprocessor>(control_.get(), diagnosticsClient_);

  if (diagnosticsClient_) {
    diagnosticsClient_->setPreprocessor(preprocessor_.get());
  }
}

TranslationUnit::~TranslationUnit() {}

auto TranslationUnit::diagnosticsClient() const -> DiagnosticsClient* {
  return diagnosticsClient_;
}

auto TranslationUnit::changeDiagnosticsClient(
    DiagnosticsClient* diagnosticsClient) -> DiagnosticsClient* {
  std::swap(diagnosticsClient_, diagnosticsClient);

  if (diagnosticsClient_) {
    diagnosticsClient_->setPreprocessor(preprocessor_.get());
  }

  return diagnosticsClient;
}

void TranslationUnit::setSource(std::string source, std::string fileName) {
  beginPreprocessing(std::move(source), std::move(fileName));
  DefaultPreprocessorState state{*preprocessor_};
  while (state) {
    std::visit(state, continuePreprocessing());
  }
  endPreprocessing();
}

void TranslationUnit::beginPreprocessing(std::string source,
                                         std::string fileName) {
  fileName_ = std::move(fileName);
  preprocessor_->beginPreprocessing(std::move(source), fileName_, tokens_);
}

auto TranslationUnit::continuePreprocessing() -> PreprocessingState {
  return preprocessor_->continuePreprocessing(tokens_);
}

void TranslationUnit::endPreprocessing() {
  preprocessor_->endPreprocessing(tokens_);
}

auto TranslationUnit::fatalErrors() const -> bool {
  return diagnosticsClient_->fatalErrors();
}

void TranslationUnit::setFatalErrors(bool fatalErrors) {
  diagnosticsClient_->setFatalErrors(fatalErrors);
}

auto TranslationUnit::blockErrors(bool blockErrors) -> bool {
  return diagnosticsClient_->blockErrors(blockErrors);
}

void TranslationUnit::error(SourceLocation loc, std::string message) const {
  diagnosticsClient_->report(tokenAt(loc), Severity::Error, std::move(message));
}

void TranslationUnit::warning(SourceLocation loc, std::string message) const {
  TranslationUnit::diagnosticsClient_->report(tokenAt(loc), Severity::Warning,
                                              std::move(message));
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

auto TranslationUnit::tokenStartPosition(SourceLocation loc) const
    -> SourcePosition {
  return preprocessor_->tokenStartPosition(tokenAt(loc));
}

auto TranslationUnit::tokenEndPosition(SourceLocation loc) const
    -> SourcePosition {
  return preprocessor_->tokenEndPosition(tokenAt(loc));
}

void TranslationUnit::parse(ParserConfiguration config) {
  if (ast_) {
    cxx_runtime_error("translation unit already parsed");
  }

  config_ = std::move(config);

  preprocessor_->squeeze();
  Parser parse(this);
  parse(ast_);
}

auto TranslationUnit::language() const -> LanguageKind {
  return preprocessor_->language();
}

auto TranslationUnit::config() const -> const ParserConfiguration& {
  return config_;
}

auto TranslationUnit::globalScope() const -> Scope* {
  if (!globalNamespace_) return nullptr;
  return globalNamespace_->scope();
}

auto TranslationUnit::fileName() const -> const std::string& {
  return fileName_;
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
