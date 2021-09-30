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

#include <cxx/diagnostic.h>

// fmt
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cstdlib>

namespace cxx {

class Preprocessor;

class DiagnosticsClient {
 public:
  DiagnosticsClient(const DiagnosticsClient&) = delete;
  DiagnosticsClient& operator=(const DiagnosticsClient&) = delete;

  DiagnosticsClient() = default;

  virtual ~DiagnosticsClient();

  virtual void report(const Diagnostic& diagnostic);

  Preprocessor* preprocessor() const { return preprocessor_; }

  void setPreprocessor(Preprocessor* preprocessor) {
    preprocessor_ = preprocessor;
  }

  bool fatalErrors() const { return fatalErrors_; }
  void setFatalErrors(bool fatalErrors) { fatalErrors_ = fatalErrors; }

  bool blockErrors(bool blockErrors = true) {
    std::swap(blockErrors_, blockErrors);
    return blockErrors;
  }

  template <typename... Args>
  void report(const Token& token, Severity kind, const std::string_view& format,
              const Args&... args) {
    if (blockErrors_) return;

    Diagnostic diag(kind, token,
                    fmt::vformat(format, fmt::make_format_args(args...)));

    report(diag);

    if (diag.severity() == Severity::Fatal ||
        (diag.severity() == Severity::Error && fatalErrors_))
      exit(EXIT_FAILURE);
  }

 private:
  Preprocessor* preprocessor_ = nullptr;
  bool blockErrors_ = false;
  bool fatalErrors_ = false;
};

}  // namespace cxx