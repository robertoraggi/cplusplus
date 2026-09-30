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

#include <cxx/diagnostics_client.h>

// cxx
#include <cxx/preprocessor.h>
#include <cxx/source_location.h>
#include <cxx/translation_unit.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <iostream>

namespace cxx {

DiagnosticsClient::~DiagnosticsClient() = default;

auto reportOutsideImmediateContext(TranslationUnit* unit,
                                   const std::vector<Diagnostic>& diagnostics)
    -> bool {
  if (diagnostics.empty()) return false;
  auto client = unit->reportingDiagnosticsClient();
  if (!client) return false;
  for (const auto& diagnostic : diagnostics) client->report(diagnostic);
  return true;
}

OutsideImmediateContextScope::OutsideImmediateContextScope(
    TranslationUnit* unit)
    : unit_(unit) {
  if (unit->diagnosticsClient()->isSfinae()) capture_.emplace(unit);
}

OutsideImmediateContextScope::~OutsideImmediateContextScope() {
  if (!capture_) return;
  capture_->finish();
  (void)reportOutsideImmediateContext(unit_, capture_->diagnostics());
}

DiagnosticsClientScope::DiagnosticsClientScope(TranslationUnit* unit,
                                               DiagnosticsClient* client)
    : unit_(unit),
      previousReportingClient_(unit->reportingDiagnosticsClient()) {
  previousClient_ = unit->changeDiagnosticsClient(client);
}

DiagnosticsClientScope::~DiagnosticsClientScope() { restore(); }

void DiagnosticsClientScope::restore() {
  if (restored_) return;
  restored_ = true;
  (void)unit_->changeDiagnosticsClient(previousClient_);
  unit_->setReportingDiagnosticsClient(previousReportingClient_);
}

void DiagnosticsClient::report(const Diagnostic& diag) {
  std::string_view severity;

  switch (diag.severity()) {
    case Severity::Message:
      severity = "message";
      break;
    case Severity::Note:
      severity = "note";
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

  SourcePosition pos;
  if (sourceResolver_) {
    pos = sourceResolver_->tokenStartPosition(diag.token());
  }

  if (sourceResolver_ && diag.token().fileId() != lastIncludeFileId_ &&
      (diag.severity() == Severity::Warning ||
       diag.severity() == Severity::Error ||
       diag.severity() == Severity::Fatal)) {
    lastIncludeFileId_ = diag.token().fileId();
    auto stack = sourceResolver_->includeStack(diag.token());
    std::string_view prefix = "In file included from ";
    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
      std::cerr << std::format("{}{}:{}:\n", prefix, it->fileName, it->line);
      prefix = "                 from ";
    }
  }

  if (pos.fileName.empty()) {
    std::cerr << std::format("{}\n", diag.message());
  } else {
    std::cerr << std::format("{}:{}:{}: {}: {}\n", pos.fileName, pos.line,
                             pos.column, severity, diag.message());

    const auto textLine = sourceResolver_->getTextLine(diag.token());

    if (!textLine.empty()) {
      const auto end = std::max(0, static_cast<int>(pos.column) - 1);

      std::string indent{textLine.substr(0, end)};

      for (auto& ch : indent) {
        if (!std::isspace(ch)) ch = ' ';
      }

      std::cerr << std::format("{0}\n{1}^\n", textLine, indent);
    }
  }

  if (diag.severity() == Severity::Fatal ||
      (diag.severity() == Severity::Error && fatalErrors_)) {
    exit(EXIT_FAILURE);
  }
  // Note diagnostics never abort compilation.
}

}  // namespace cxx
