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

#include <cxx/control.h>
#include <cxx/diagnostics_client.h>
#include <cxx/preprocessor.h>
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

auto reportedText(SourceResolver& resolver,
                  Token token = Token{TokenKind::T_IDENTIFIER},
                  int repetitions = 1) -> std::string {
  DiagnosticsClient client;
  client.setSourceResolver(&resolver);

  std::ostringstream captured;
  auto* saved = std::cerr.rdbuf(captured.rdbuf());
  for (int i = 0; i < repetitions; ++i)
    client.report(token, Severity::Error, "message");
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

TEST(Diagnostics, RetainsIncludeChainForEachHeaderOccurrence) {
  Control control;
  DiagnosticsClient diagnostics;
  Preprocessor pp{&control, &diagnostics};
  std::vector<Token> tokens;
  pp.beginPreprocessing("#include \"first.h\"\n#include \"second.h\"\n",
                        "main.cc", tokens);
  while (true) {
    auto state = pp.continuePreprocessing(tokens);
    if (std::holds_alternative<ProcessingComplete>(state)) break;
    if (auto include = std::get_if<PendingInclude>(&state)) {
      include->resolveWith(std::get<QuoteInclude>(include->include).fileName);
    } else if (auto content = std::get_if<PendingFileContent>(&state)) {
      content->setContent(content->fileName == "shared.h"
                              ? "int broken;\n"
                              : "#include \"shared.h\"\n");
    }
  }
  pp.endPreprocessing(tokens);

  std::vector<Token> occurrences;
  for (const auto& token : tokens) {
    if (pp.getTokenText(token) == "broken") occurrences.push_back(token);
  }
  ASSERT_EQ(occurrences.size(), 2);
  EXPECT_NE(occurrences[0].fileId(), occurrences[1].fileId());

  auto first = pp.includeStack(occurrences[0]);
  auto second = pp.includeStack(occurrences[1]);
  ASSERT_EQ(first.size(), 2);
  ASSERT_EQ(second.size(), 2);
  EXPECT_EQ(first[0].fileName, "first.h");
  EXPECT_EQ(first[1].fileName, "main.cc");
  EXPECT_EQ(first[1].line, 1);
  EXPECT_EQ(second[0].fileName, "second.h");
  EXPECT_EQ(second[1].fileName, "main.cc");
  EXPECT_EQ(second[1].line, 2);

  const auto text = reportedText(pp, occurrences[1]);
  EXPECT_NE(text.find("shared.h:1:5: error: message"), std::string::npos);
  EXPECT_NE(text.find("In file included from main.cc:2:"), std::string::npos);
  EXPECT_NE(text.find("from second.h:1:"), std::string::npos);
  EXPECT_LT(text.find("In file included from main.cc:2:"),
            text.find("shared.h:1:5: error: message"));
  EXPECT_EQ(text.find("included from here"), std::string::npos);
  EXPECT_EQ(text.find("first.h:"), std::string::npos);

  const auto repeated = reportedText(pp, occurrences[1], 2);
  const auto firstTrace = repeated.find("In file included from");
  ASSERT_NE(firstTrace, std::string::npos);
  EXPECT_EQ(repeated.find("In file included from", firstTrace + 1),
            std::string::npos);
  const auto firstError = repeated.find("error: message");
  ASSERT_NE(firstError, std::string::npos);
  EXPECT_NE(repeated.find("error: message", firstError + 1), std::string::npos);
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
