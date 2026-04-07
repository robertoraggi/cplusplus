// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/diagnostic.h>

#include <vector>

namespace cxx {

class Preprocessor;

class DiagnosticsClient {
 public:
  DiagnosticsClient(const DiagnosticsClient&) = delete;
  auto operator=(const DiagnosticsClient&) -> DiagnosticsClient& = delete;

  DiagnosticsClient() = default;

  virtual ~DiagnosticsClient();

  virtual void report(const Diagnostic& diagnostic);

  [[nodiscard]] auto preprocessor() const -> Preprocessor* {
    return preprocessor_;
  }

  void setPreprocessor(Preprocessor* preprocessor) {
    preprocessor_ = preprocessor;
  }

  [[nodiscard]] auto fatalErrors() const -> bool { return fatalErrors_; }
  void setFatalErrors(bool fatalErrors) { fatalErrors_ = fatalErrors; }

  [[nodiscard]] auto errorCount() const -> int { return errorCount_; }

  [[nodiscard]] auto errorLimit() const -> int { return errorLimit_; }
  void setErrorLimit(int errorLimit) { errorLimit_ = errorLimit; }

  [[nodiscard]] auto errorLimitReached() const -> bool {
    return errorLimit_ > 0 && errorCount_ >= errorLimit_;
  }

  [[nodiscard]] auto shouldReportErrors() const -> bool {
    return !blockErrors_;
  }

  auto blockErrors(bool blockErrors = true) -> bool {
    std::swap(blockErrors_, blockErrors);
    return blockErrors;
  }

  void report(const Token& token, Severity severity, std::string message) {
    if (blockErrors_) return;

    if (severity == Severity::Error || severity == Severity::Fatal) {
      ++errorCount_;
      if (errorLimit_ > 0 && errorCount_ > errorLimit_) return;
    }

    Diagnostic diag{severity, token, std::move(message)};

    report(diag);
  }

 private:
  Preprocessor* preprocessor_ = nullptr;
  int errorCount_ = 0;
  int errorLimit_ = 0;
  bool blockErrors_ = false;
  bool fatalErrors_ = false;
};

struct CapturingDiagnosticsClient final : DiagnosticsClient {
  DiagnosticsClient* parent = nullptr;
  std::vector<Diagnostic> diagnostics;

  explicit CapturingDiagnosticsClient(DiagnosticsClient* parent = nullptr)
      : parent(parent) {}

  void report(const Diagnostic& diagnostic) override {
    diagnostics.push_back(diagnostic);
    if (parent) parent->report(diagnostic);
  }
};

}  // namespace cxx
