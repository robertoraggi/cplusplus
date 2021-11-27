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

#include <cxx/ast_fwd.h>
#include <cxx/diagnostic.h>
#include <cxx/diagnostics_client.h>
#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/token.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cxx {

class TranslationUnit {
 public:
  explicit TranslationUnit(Control* control,
                           DiagnosticsClient* diagosticsClient);

  ~TranslationUnit();

  Control* control() const { return control_; }

  Arena* arena() const { return arena_.get(); }

  DiagnosticsClient* diagnosticsClient() const { return diagnosticsClient_; }

  UnitAST* ast() const { return ast_; }

  const std::string& fileName() const { return fileName_; }
  Preprocessor* preprocessor() const { return preprocessor_.get(); }

  void setSource(std::string source, std::string fileName);

  bool fatalErrors() const { return diagnosticsClient_->fatalErrors(); }

  void setFatalErrors(bool fatalErrors) {
    diagnosticsClient_->setFatalErrors(fatalErrors);
  }

  bool blockErrors(bool blockErrors = true) {
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
  inline unsigned tokenCount() const { return unsigned(tokens_.size()); }

  inline const Token& tokenAt(SourceLocation loc) const {
    return tokens_[loc.index()];
  }

  void setTokenKind(SourceLocation loc, TokenKind kind) {
    tokens_[loc.index()].setKind(kind);
  }

  inline TokenKind tokenKind(SourceLocation loc) const {
    return tokenAt(loc).kind();
  }

  void setTokenValue(SourceLocation loc, TokenValue value) {
    tokens_[loc.index()].setValue(value);
  }

  int tokenLength(SourceLocation loc) const;

  const std::string& tokenText(SourceLocation loc) const;

  void getTokenStartPosition(SourceLocation loc, unsigned* line,
                             unsigned* column = nullptr,
                             std::string_view* fileName = nullptr) const;

  void getTokenEndPosition(SourceLocation loc, unsigned* line,
                           unsigned* column = nullptr,
                           std::string_view* fileName = nullptr) const;

  const Identifier* identifier(SourceLocation loc) const;

  const Literal* literal(SourceLocation loc) const;

  bool parse(bool checkTypes = false);

  void replaceWithIdentifier(SourceLocation loc);

 private:
  Control* control_;
  std::unique_ptr<Arena> arena_;
  std::vector<Token> tokens_;
  std::string fileName_;
  UnitAST* ast_ = nullptr;
  const char* yyptr = nullptr;
  DiagnosticsClient* diagnosticsClient_;
  std::unique_ptr<Preprocessor> preprocessor_;
};

}  // namespace cxx
