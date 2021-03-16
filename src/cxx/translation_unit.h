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
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/token.h>

// fmt
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cxx {

class Preprocessor;

class DiagnosticClient {
 public:
  virtual ~DiagnosticClient() = default;

  virtual void report(const Diagnostic& diagtic) = 0;
};

class TranslationUnit {
 public:
  TranslationUnit(const TranslationUnit&) = delete;
  TranslationUnit& operator=(const TranslationUnit&) = delete;

  TranslationUnit(Control* control);
  ~TranslationUnit();

  Control* control() const { return control_; }

  Arena* arena() const { return arena_; }

  DiagnosticClient* diagnosticClient() const { return diagnosticClient_; }

  void setDiagnosticClient(DiagnosticClient* diagnosticClient) {
    diagnosticClient_ = diagnosticClient;
  }

  UnitAST* ast() const { return ast_; }

  const std::string& fileName() const { return fileName_; }
  void setFileName(std::string fileName) { fileName_ = std::move(fileName); }

  bool preprocessed() const { return preprocessed_; }
  void setPreprocessed(bool preprocessed) { preprocessed_ = preprocessed; }

  void setPreprocessor(std::unique_ptr<Preprocessor> preprocessor);

  const std::string& source() const { return code_; }

  void setSource(std::string source) {
    code_ = std::move(source);
    yyptr = code_.c_str();
  }

  bool fatalErrors() const { return fatalErrors_; }
  void setFatalErrors(bool fatalErrors) { fatalErrors_ = fatalErrors; }

  bool blockErrors(bool blockErrors = true) {
    std::swap(blockErrors_, blockErrors);
    return blockErrors;
  }

  template <typename... Args>
  void report(SourceLocation loc, Severity kind, const std::string_view& format,
              const Args&... args) {
    if (blockErrors_) return;

    unsigned line = 0, column = 0;
    std::string_view fileName;
    getTokenStartPosition(loc, &line, &column, &fileName);

    if (diagnosticClient_) {
      diagnosticClient_->report(
          Diagnostic(kind, std::string(fileName), line, column,
                     fmt::vformat(format, fmt::make_format_args(args...))));
      return;
    }

    std::string_view Severity;

    switch (kind) {
      case Severity::Message:
        Severity = "message";
        break;
      case Severity::Warning:
        Severity = "warning";
        break;
      case Severity::Error:
        Severity = "error";
        break;
      case Severity::Fatal:
        Severity = "fatal";
        break;
    }  // switch

    fmt::print(stderr, "{}:{}:{}: {}: ", fileName, line, column, Severity);
    fmt::vprint(stderr, format, fmt::make_format_args(args...));
    fmt::print(stderr, "\n");
#if 0
    std::string cursor(column - 1, ' ');
    cursor += "^";
    fmt::print(stderr, "{}\n{}\n", preprocessor_->getTextLine(tokenAt(loc)),
               cursor);
#endif
    if (kind == Severity::Fatal || (kind == Severity::Error && fatalErrors_))
      exit(EXIT_FAILURE);
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

  int tokenLength(SourceLocation loc) const;

  const std::string& tokenText(SourceLocation loc) const;

  void getTokenStartPosition(SourceLocation loc, unsigned* line,
                             unsigned* column = nullptr,
                             std::string_view* fileName = nullptr) const;

  void getTokenEndPosition(SourceLocation loc, unsigned* line,
                           unsigned* column = nullptr,
                           std::string_view* fileName = nullptr) const;

  const Identifier* identifier(SourceLocation loc) const;

  void tokenize(bool preprocessing = false);

  bool parse();

 private:
  Control* control_;
  Arena* arena_;
  std::vector<Token> tokens_;
  std::string fileName_;
  std::string text_;
  std::string code_;
  UnitAST* ast_ = nullptr;
  const char* yyptr = nullptr;
  bool fatalErrors_ = false;
  bool blockErrors_ = false;
  bool preprocessed_ = true;
  DiagnosticClient* diagnosticClient_ = nullptr;
  std::unique_ptr<Preprocessor> preprocessor_;
};

}  // namespace cxx
