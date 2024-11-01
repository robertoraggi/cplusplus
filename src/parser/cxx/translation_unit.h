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

#include <cxx/ast_fwd.h>
#include <cxx/diagnostic.h>
#include <cxx/diagnostics_client.h>
#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/parser_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token.h>

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cxx {

class TranslationUnit {
 public:
  explicit TranslationUnit(Control* control,
                           DiagnosticsClient* diagosticsClient);

  ~TranslationUnit();

  [[nodiscard]] auto control() const -> Control* { return control_; }

  [[nodiscard]] auto arena() const -> Arena* { return arena_.get(); }

  [[nodiscard]] auto diagnosticsClient() const -> DiagnosticsClient*;

  auto changeDiagnosticsClient(DiagnosticsClient* diagnosticsClient)
      -> DiagnosticsClient*;

  [[nodiscard]] auto ast() const -> UnitAST* { return ast_; }

  void setAST(UnitAST* ast) { ast_ = ast; }

  [[nodiscard]] auto globalScope() const -> Scope*;

  [[nodiscard]] auto fileName() const -> const std::string& {
    return fileName_;
  }
  [[nodiscard]] auto preprocessor() const -> Preprocessor* {
    return preprocessor_.get();
  }

  void setSource(std::string source, std::string fileName);

  [[nodiscard]] auto fatalErrors() const -> bool {
    return diagnosticsClient_->fatalErrors();
  }

  void setFatalErrors(bool fatalErrors) {
    diagnosticsClient_->setFatalErrors(fatalErrors);
  }

  auto blockErrors(bool blockErrors = true) -> bool {
    return diagnosticsClient_->blockErrors(blockErrors);
  }

  void error(SourceLocation loc, std::string message) const {
    diagnosticsClient_->report(tokenAt(loc), Severity::Error,
                               std::move(message));
  }

  void warning(SourceLocation loc, std::string message) const {
    diagnosticsClient_->report(tokenAt(loc), Severity::Warning,
                               std::move(message));
  }

  // tokens
  [[nodiscard]] inline auto tokenCount() const -> unsigned {
    return static_cast<unsigned>(tokens_.size());
  }

  [[nodiscard]] inline auto tokenAt(SourceLocation loc) const -> const Token& {
    return tokens_[loc.index()];
  }

  void setTokenKind(SourceLocation loc, TokenKind kind) {
    tokens_[loc.index()].setKind(kind);
  }

  [[nodiscard]] inline auto tokenKind(SourceLocation loc) const -> TokenKind {
    return tokenAt(loc).kind();
  }

  void setTokenValue(SourceLocation loc, TokenValue value) {
    tokens_[loc.index()].setValue(value);
  }

  [[nodiscard]] auto tokenLength(SourceLocation loc) const -> int;

  [[nodiscard]] auto tokenText(SourceLocation loc) const -> const std::string&;

  [[nodiscard]] auto tokenStartPosition(SourceLocation loc) const
      -> SourcePosition;

  [[nodiscard]] auto tokenEndPosition(SourceLocation loc) const
      -> SourcePosition;

  [[nodiscard]] auto identifier(SourceLocation loc) const -> const Identifier*;

  [[nodiscard]] auto literal(SourceLocation loc) const -> const Literal*;

  void parse(const ParserConfiguration& config = {});

  [[nodiscard]] auto load(std::span<const std::uint8_t> data) -> bool;

  [[nodiscard]] auto serialize(std::ostream& out) -> bool;

  [[nodiscard]] auto serialize(
      const std::function<void(std::span<const std::uint8_t>)>& onData) -> bool;

 private:
  Control* control_;
  std::unique_ptr<Arena> arena_;
  std::vector<Token> tokens_;
  std::string fileName_;
  UnitAST* ast_ = nullptr;
  const char* yyptr = nullptr;
  DiagnosticsClient* diagnosticsClient_ = nullptr;
  NamespaceSymbol* globalNamespace_ = nullptr;
  std::unique_ptr<Preprocessor> preprocessor_;
};

}  // namespace cxx
