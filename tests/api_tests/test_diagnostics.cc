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
#include <cxx/source_resolver.h>
#include <cxx/token.h>
#include <cxx/types.h>
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

#include "test_utils.h"

using namespace cxx;

namespace {

class ArchivedSourceResolver final : public SourceResolver {
 public:
  explicit ArchivedSourceResolver(std::string_view textLine)
      : textLine_(textLine) {}

  [[nodiscard]] auto tokenStartPosition(const Token& token) const
      -> SourcePosition override {
    return SourcePosition{"archived.cc", 12, 3};
  }

  [[nodiscard]] auto tokenEndPosition(const Token& token) const
      -> SourcePosition override {
    return SourcePosition{"archived.cc", 12, 4};
  }

  [[nodiscard]] auto getTextLine(const Token& token) const
      -> std::string_view override {
    return textLine_;
  }

  [[nodiscard]] auto getTokenText(const Token& token) const
      -> std::string_view override {
    return {};
  }

 private:
  std::string_view textLine_;
};

auto reportedText(SourceResolver& resolver) -> std::string {
  DiagnosticsClient client;
  client.setSourceResolver(&resolver);

  std::ostringstream captured;
  auto* saved = std::cerr.rdbuf(captured.rdbuf());
  client.report(Token{TokenKind::T_IDENTIFIER}, Severity::Error, "message");
  std::cerr.rdbuf(saved);

  return captured.str();
}

}  // namespace

TEST(Diagnostics, ReportsPositionWithoutSourceText) {
  ArchivedSourceResolver resolver{""};

  ASSERT_EQ(reportedText(resolver), "archived.cc:12:3: error: message\n");
}

TEST(Diagnostics, ReportsCaretWhenSourceTextIsAvailable) {
  ArchivedSourceResolver resolver{"  int x;"};

  ASSERT_EQ(reportedText(resolver),
            "archived.cc:12:3: error: message\n  int x;\n  ^\n");
}

TEST(Snippets, UnresolvedTypesPrintFromCapturedText) {
  auto source = R"(
template <int N>
struct Buffer {
  char data[N + 1];
};
)"_cxx;

  auto buffer =
      symbol_cast<ClassSymbol>(LookupMember{source}(source.scope(), "Buffer"));

  ASSERT_TRUE(buffer);

  auto data = symbol_cast<FieldSymbol>(LookupMember{source}(buffer, "data"));

  ASSERT_TRUE(data);
  ASSERT_EQ(to_string(data->type(), data->name()), "char data[N + 1]");
}

TEST(Snippets, UncapturedRangeHasNoText) {
  auto source = "int x;"_cxx;

  ASSERT_TRUE(
      source.unit.snippetText({SourceLocation(1), SourceLocation(2)}).empty());
}
