#// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

namespace cxx {

void VerifyCommentHandler::handleComment(Preprocessor* preprocessor,
                                         const Token& token) {
  const std::string text{preprocessor->getTokenText(token)};

  std::smatch match;

  if (!std::regex_match(text, match, rx)) {
    return;
  }

  std::string_view fileName;
  unsigned line = 0;
  unsigned column = 0;

  preprocessor->getTokenStartPosition(token, &line, &column, &fileName);

  Severity severity = Severity::Error;

  if (match[1] == "warning") {
    severity = Severity::Warning;
  }

  std::string offset = match[2];

  if (!offset.empty()) {
    line += std::stoi(offset);
  }

  const auto& message = match[3];

  expectedDiagnostics_.push_back({severity, fileName, message, line});
}

auto VerifyDiagnosticsClient::hasErrors() const -> bool {
  if (verify_) {
    return !reportedDiagnostics_.empty();
  }

  for (const auto& d : reportedDiagnostics_) {
    if (d.severity() == Severity::Error || d.severity() == Severity::Fatal) {
      return true;
    }
  }

  return false;
}

void VerifyDiagnosticsClient::report(const Diagnostic& diagnostic) {
  if (!verify_) {
    DiagnosticsClient::report(diagnostic);
    return;
  }

  reportedDiagnostics_.push_back(diagnostic);
}

void VerifyDiagnosticsClient::verifyExpectedDiagnostics(
    const std::list<ExpectedDiagnostic>& expectedDiagnostics) {
  if (!verify_) return;

  for (const auto& expected : expectedDiagnostics) {
    if (auto it = findDiagnostic(expected); it != cend(reportedDiagnostics_)) {
      reportedDiagnostics_.erase(it);
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

    unsigned line = 0;
    unsigned column = 0;
    std::string_view fileName;

    preprocessor()->getTokenStartPosition(d.token(), &line, &column, &fileName);

    if (line != expected.line) return false;

    if (fileName != expected.fileName) return false;

    if (d.message() != expected.message) return false;

    return true;
  };

  return std::find_if(begin(reportedDiagnostics_), end(reportedDiagnostics_),
                      isExpectedDiagnostic);
}

}  // namespace cxx