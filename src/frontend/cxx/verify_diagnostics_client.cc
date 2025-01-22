#// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "verify_diagnostics_client.h"

#include <cxx/source_location.h>

namespace cxx {

auto VerifyDiagnosticsClient::verify() const -> bool { return verify_; }

void VerifyDiagnosticsClient::setVerify(bool verify) { verify_ = verify; }

auto VerifyDiagnosticsClient::reportedDiagnostics() const
    -> const std::list<Diagnostic>& {
  return reportedDiagnostics_;
}

auto VerifyDiagnosticsClient::expectedDiagnostics() const
    -> const std::list<ExpectedDiagnostic>& {
  return expectedDiagnostics_;
}

void VerifyDiagnosticsClient::handleComment(Preprocessor* preprocessor,
                                            const Token& token) {
  const std::string text{preprocessor->getTokenText(token)};

  std::smatch match;

  if (!std::regex_match(text, match, rx)) {
    return;
  }

  const auto pos = preprocessor->tokenStartPosition(token);
  auto line = pos.line;

  Severity severity = Severity::Error;

  if (match[1] == "warning") {
    severity = Severity::Warning;
  }

  std::string offset = match[2];

  if (!offset.empty()) {
    line += std::stoi(offset);
  }

  const auto& message = match[3];

  expectedDiagnostics_.push_back(
      {token, severity, std::string(pos.fileName), message, line});
}

auto VerifyDiagnosticsClient::hasErrors() const -> bool {
  if (!reportedDiagnostics_.empty()) return true;
  if (!expectedDiagnostics_.empty()) return true;
  return hasErrors_;
}

void VerifyDiagnosticsClient::report(const Diagnostic& diagnostic) {
  if (!shouldReportErrors()) return;

  if (verify_) {
    reportedDiagnostics_.push_back(diagnostic);
    return;
  }

  if (diagnostic.severity() == Severity::Error ||
      diagnostic.severity() == Severity::Fatal) {
    hasErrors_ = true;
  }

  DiagnosticsClient::report(diagnostic);
}

void VerifyDiagnosticsClient::verifyExpectedDiagnostics() {
  if (!verify_) return;

  while (!reportedDiagnostics_.empty()) {
    if (expectedDiagnostics_.empty()) break;

    auto expected = expectedDiagnostics_.front();
    expectedDiagnostics_.pop_front();

    if (auto it = findDiagnostic(expected); it != cend(reportedDiagnostics_)) {
      reportedDiagnostics_.erase(it);
    } else {
      Diagnostic diag(Severity::Error, expected.token, "expected diagnostic");
      DiagnosticsClient::report(diag);

      hasErrors_ = true;
    }
  }

  for (const auto& diag : reportedDiagnostics_) {
    DiagnosticsClient::report(diag);
  }
}

auto VerifyDiagnosticsClient::findDiagnostic(const ExpectedDiagnostic& expected)
    const -> std::list<Diagnostic>::const_iterator {
  auto isExpectedDiagnostic = [&](const Diagnostic& d) {
    if (d.severity() != expected.severity) {
      return false;
    }

    const auto pos = preprocessor()->tokenStartPosition(d.token());

    if (pos.line != expected.line) return false;

    if (pos.fileName != expected.fileName) return false;

    if (d.message() != expected.message) return false;

    return true;
  };

  return std::find_if(begin(reportedDiagnostics_), end(reportedDiagnostics_),
                      isExpectedDiagnostic);
}

}  // namespace cxx