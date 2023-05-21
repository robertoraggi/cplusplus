// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

  auto blockErrors(bool blockErrors = true) -> bool {
    std::swap(blockErrors_, blockErrors);
    return blockErrors;
  }

  void report(const Token& token, Severity severity, std::string message) {
    if (blockErrors_) return;

    Diagnostic diag{severity, token, std::move(message)};

    report(diag);
  }

 private:
  Preprocessor* preprocessor_ = nullptr;
  bool blockErrors_ = false;
  bool fatalErrors_ = false;
};

}  // namespace cxx
