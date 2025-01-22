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

#include <cxx/diagnostics_client.h>

// cxx
#include <cxx/preprocessor.h>
#include <cxx/source_location.h>

#include <cctype>
#include <cstdlib>
#include <format>
#include <iostream>

namespace cxx {

DiagnosticsClient::~DiagnosticsClient() = default;

void DiagnosticsClient::report(const Diagnostic& diag) {
  std::string_view severity;

  switch (diag.severity()) {
    case Severity::Message:
      severity = "message";
      break;
    case Severity::Warning:
      severity = "warning";
      break;
    case Severity::Error:
      severity = "error";
      break;
    case Severity::Fatal:
      severity = "fatal";
      break;
  }  // switch

  const auto pos = preprocessor_->tokenStartPosition(diag.token());

  if (!pos.fileName.empty()) {
    std::cerr << std::format("{}:{}:{}: {}: {}\n", pos.fileName, pos.line,
                             pos.column, severity, diag.message());

    const auto textLine = preprocessor_->getTextLine(diag.token());

    const auto end = std::max(0, static_cast<int>(pos.column) - 1);

    std::string indent{textLine.substr(0, end)};

    for (auto& ch : indent) {
      if (!std::isspace(ch)) ch = ' ';
    }

    std::cerr << std::format("{0}\n{1}^\n", textLine, indent);
  } else {
    std::cerr << std::format("{}\n", diag.message());
  }

  if (diag.severity() == Severity::Fatal ||
      (diag.severity() == Severity::Error && fatalErrors_)) {
    exit(EXIT_FAILURE);
  }
}

}  // namespace cxx
