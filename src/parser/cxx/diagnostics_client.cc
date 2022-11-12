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

#include <cxx/diagnostics_client.h>

// cxx
#include <cxx/preprocessor.h>
#include <cxx/private/format.h>

#include <cctype>
#include <cstdlib>
#include <iostream>

namespace cxx {

DiagnosticsClient::~DiagnosticsClient() {}

void DiagnosticsClient::report(const Diagnostic& diag) {
  std::string_view Severity;

  switch (diag.severity()) {
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

  std::string_view fileName;
  uint32_t line = 0;
  uint32_t column = 0;

  preprocessor_->getTokenStartPosition(diag.token(), &line, &column, &fileName);

  if (!fileName.empty()) {
    fmt::print(std::cerr, "{}:{}:{}: {}\n", fileName, line, column,
               diag.message());

    const auto textLine = preprocessor_->getTextLine(diag.token());

    const auto end = std::max(0, int(column) - 1);

    std::string indent{textLine.substr(0, end)};

    for (auto& ch : indent) {
      if (!std::isspace(ch)) ch = ' ';
    }

    fmt::print(std::cerr, "{0}\n{1}^\n", textLine, indent);
  } else {
    fmt::print(std::cerr, "{}\n", diag.message());
  }

  if (diag.severity() == Severity::Fatal ||
      (diag.severity() == Severity::Error && fatalErrors_))
    exit(EXIT_FAILURE);
}

}  // namespace cxx
