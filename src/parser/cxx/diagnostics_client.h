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
class TranslationUnit;

class DiagnosticsClient {
 public:
  DiagnosticsClient(const DiagnosticsClient&) = delete;
  auto operator=(const DiagnosticsClient&) -> DiagnosticsClient& = delete;

  DiagnosticsClient() = default;

  virtual ~DiagnosticsClient();

  virtual void report(const Diagnostic& diagnostic);

  [[nodiscard]] virtual auto isSfinae() const -> bool { return false; }

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

class SilentDiagnosticsClient final : public DiagnosticsClient {
 public:
  explicit SilentDiagnosticsClient(bool sfinae = true) : sfinae_(sfinae) {}

  void report(const Diagnostic& diagnostic) override {
    if (diagnostic.severity() != Severity::Error) return;
    hadError_ = true;
    diagnostics_.push_back(diagnostic);
  }

  [[nodiscard]] auto isSfinae() const -> bool override { return sfinae_; }

  [[nodiscard]] auto hadError() const -> bool { return hadError_; }

  [[nodiscard]] auto diagnostics() const -> const std::vector<Diagnostic>& {
    return diagnostics_;
  }

 private:
  bool sfinae_ = true;
  bool hadError_ = false;
  std::vector<Diagnostic> diagnostics_;
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

auto reportOutsideImmediateContext(TranslationUnit* unit,
                                   const std::vector<Diagnostic>& diagnostics)
    -> bool;

class DiagnosticsClientScope {
 public:
  DiagnosticsClientScope(TranslationUnit* unit, DiagnosticsClient* client);
  ~DiagnosticsClientScope();

  DiagnosticsClientScope(const DiagnosticsClientScope&) = delete;
  auto operator=(const DiagnosticsClientScope&)
      -> DiagnosticsClientScope& = delete;

  void restore();

  [[nodiscard]] auto previousClient() const -> DiagnosticsClient* {
    return previousClient_;
  }

 private:
  TranslationUnit* unit_;
  DiagnosticsClient* previousClient_;
  DiagnosticsClient* previousReportingClient_;
  bool restored_ = false;
};

class SilentDiagnosticsScope {
 public:
  explicit SilentDiagnosticsScope(TranslationUnit* unit, bool sfinae = true)
      : client_(sfinae), scope_(unit, &client_) {}

  SilentDiagnosticsScope(const SilentDiagnosticsScope&) = delete;
  auto operator=(const SilentDiagnosticsScope&)
      -> SilentDiagnosticsScope& = delete;

  void finish() { scope_.restore(); }

  [[nodiscard]] auto hadError() const -> bool { return client_.hadError(); }

  [[nodiscard]] auto diagnostics() const -> const std::vector<Diagnostic>& {
    return client_.diagnostics();
  }

  [[nodiscard]] auto previousClient() const -> DiagnosticsClient* {
    return scope_.previousClient();
  }

 private:
  SilentDiagnosticsClient client_;
  DiagnosticsClientScope scope_;
};

class CapturingDiagnosticsScope {
 public:
  explicit CapturingDiagnosticsScope(TranslationUnit* unit)
      : scope_(unit, &client_) {}

  CapturingDiagnosticsScope(const CapturingDiagnosticsScope&) = delete;
  auto operator=(const CapturingDiagnosticsScope&)
      -> CapturingDiagnosticsScope& = delete;

  void forwardToPreviousClient() { client_.parent = scope_.previousClient(); }

  void finish() { scope_.restore(); }

  [[nodiscard]] auto diagnostics() const -> const std::vector<Diagnostic>& {
    return client_.diagnostics;
  }

  auto takeDiagnostics() -> std::vector<Diagnostic> {
    return std::move(client_.diagnostics);
  }

  [[nodiscard]] auto previousClient() const -> DiagnosticsClient* {
    return scope_.previousClient();
  }

 private:
  CapturingDiagnosticsClient client_;
  DiagnosticsClientScope scope_;
};
}  // namespace cxx
