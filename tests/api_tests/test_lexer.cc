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

#include <cxx/lexer.h>
#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

using namespace cxx;

namespace {

struct Scanned {
  TokenKind kind = TokenKind::T_EOF_SYMBOL;
  std::string text;
};

auto scan(std::string_view source, LanguageKind lang = LanguageKind::kCXX)
    -> std::vector<Scanned> {
  Lexer lexer{source, lang};
  std::vector<Scanned> tokens;

  for (;;) {
    const auto kind = lexer.next();
    if (kind == TokenKind::T_EOF_SYMBOL) break;
    tokens.push_back({kind, std::string{lexer.tokenText()}});
  }

  return tokens;
}

auto scanOne(std::string_view source) -> Scanned {
  const auto tokens = scan(source);
  if (tokens.size() != 1) return {};
  return tokens.front();
}

}  // namespace

TEST(Lexer, LooksAheadOverAsciiCodePoints) {
  Lexer lexer{std::string_view{"abc"}};

  EXPECT_EQ(lexer.LA(), 'a');
  EXPECT_EQ(lexer.LA(0), 'a');
  EXPECT_EQ(lexer.LA(1), 'b');
  EXPECT_EQ(lexer.LA(2), 'c');
  EXPECT_EQ(lexer.LA(3), 0u);
  EXPECT_EQ(lexer.LA(4), 0u);
  EXPECT_EQ(lexer.LA(100), 0u);
}

TEST(Lexer, LooksAheadOverMultiByteCodePoints) {
  Lexer lexer{std::string_view{"a\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80z"}};

  EXPECT_EQ(lexer.LA(0), 'a');
  EXPECT_EQ(lexer.LA(1), 0xe9u);
  EXPECT_EQ(lexer.LA(2), 0x20acu);
  EXPECT_EQ(lexer.LA(3), 0x1f600u);
  EXPECT_EQ(lexer.LA(4), 'z');
  EXPECT_EQ(lexer.LA(5), 0u);
}

TEST(Lexer, LooksBehindStopsAtTheStartOfTheSource) {
  Lexer lexer{std::string_view{"abc"}};

  EXPECT_EQ(lexer.LA(-1), 0u);
  EXPECT_EQ(lexer.LA(-2), 0u);
  EXPECT_EQ(lexer.LA(-100), 0u);
}

TEST(Lexer, LooksBehindOverAsciiCodePoints) {
  Lexer lexer{std::string_view{"ab cd"}};

  ASSERT_EQ(lexer.next(), TokenKind::T_IDENTIFIER);
  ASSERT_EQ(lexer.tokenText(), "ab");

  EXPECT_EQ(lexer.LA(), ' ');
  EXPECT_EQ(lexer.LA(-1), 'b');
  EXPECT_EQ(lexer.LA(-2), 'a');
  EXPECT_EQ(lexer.LA(-3), 0u);
  EXPECT_EQ(lexer.LA(1), 'c');
}

TEST(Lexer, LooksBehindOverMultiByteCodePoints) {
  Lexer lexer{std::string_view{"\"\xc3\xa9\xe2\x82\xac\""}};

  ASSERT_EQ(lexer.next(), TokenKind::T_STRING_LITERAL);

  EXPECT_EQ(lexer.LA(), 0u);
  EXPECT_EQ(lexer.LA(-1), '"');
  EXPECT_EQ(lexer.LA(-2), 0x20acu);
  EXPECT_EQ(lexer.LA(-3), 0xe9u);
  EXPECT_EQ(lexer.LA(-4), '"');
  EXPECT_EQ(lexer.LA(-5), 0u);
  EXPECT_EQ(lexer.LA(-6), 0u);
}

TEST(Lexer, LookaheadSkipsLineSplices) {
  Lexer lexer{std::string_view{"a\\\nbc"}};

  EXPECT_EQ(lexer.LA(0), 'a');
  EXPECT_EQ(lexer.LA(1), 'b');
  EXPECT_EQ(lexer.LA(2), 'c');

  Lexer windows{std::string_view{"a\\\r\nbc"}};

  EXPECT_EQ(windows.LA(0), 'a');
  EXPECT_EQ(windows.LA(1), 'b');
  EXPECT_EQ(windows.LA(2), 'c');
}

TEST(Lexer, ScansRawStringLiteralsWithoutADelimiter) {
  const auto token = scanOne(R"src(R"(hello)")src");
  EXPECT_EQ(token.kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(token.text, R"src(R"(hello)")src");
}

TEST(Lexer, ScansRawStringLiteralsWithADelimiter) {
  const auto token = scanOne(R"src(R"xy(hello)xy")src");
  EXPECT_EQ(token.kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(token.text, R"src(R"xy(hello)xy")src");
}

TEST(Lexer, ScansRawStringLiteralsWithTheLongestDelimiter) {
  const auto token = scanOne(R"src(R"0123456789abcdef(x)0123456789abcdef")src");
  EXPECT_EQ(token.kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(token.text, R"src(R"0123456789abcdef(x)0123456789abcdef")src");
}

TEST(Lexer, StopsARawStringLiteralAtTheFirstMatchingDelimiter) {
  const auto tokens = scan(R"src(R"(a)" + R"(b)")src");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0].kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(tokens[0].text, R"src(R"(a)")src");
  EXPECT_EQ(tokens[1].kind, TokenKind::T_PLUS);
  EXPECT_EQ(tokens[2].kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(tokens[2].text, R"src(R"(b)")src");
}

TEST(Lexer, IgnoresDelimiterPrefixesInsideARawStringLiteral) {
  const auto token = scanOne(R"src(R"xy(a)"b)x)xyz)xy")src");
  EXPECT_EQ(token.kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(token.text, R"src(R"xy(a)"b)x)xyz)xy")src");
}

TEST(Lexer, KeepsQuotesAndBackslashesInsideARawStringLiteral) {
  const auto token = scanOne(R"src(R"(a"b\c\)")src");
  EXPECT_EQ(token.kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(token.text, R"src(R"(a"b\c\)")src");
}

TEST(Lexer, KeepsNewlinesInsideARawStringLiteral) {
  const auto token = scanOne("R\"(one\ntwo)\"");
  EXPECT_EQ(token.kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(token.text, "R\"(one\ntwo)\"");
}

TEST(Lexer, KeepsMultiByteCodePointsInsideARawStringLiteral) {
  const auto token = scanOne("R\"(\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80)\"");
  EXPECT_EQ(token.kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(token.text, "R\"(\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80)\"");
}

TEST(Lexer, ScansRawStringLiteralsWithAnEncodingPrefix) {
  EXPECT_EQ(scanOne(R"src(LR"(x)")src").kind, TokenKind::T_WIDE_STRING_LITERAL);
  EXPECT_EQ(scanOne(R"src(u8R"(x)")src").kind,
            TokenKind::T_UTF8_STRING_LITERAL);
  EXPECT_EQ(scanOne(R"src(uR"(x)")src").kind,
            TokenKind::T_UTF16_STRING_LITERAL);
  EXPECT_EQ(scanOne(R"src(UR"(x)")src").kind,
            TokenKind::T_UTF32_STRING_LITERAL);
}

TEST(Lexer, ScansRawStringLiteralsWithAUserDefinedSuffix) {
  const auto token = scanOne(R"src(R"(x)"_s)src");
  EXPECT_EQ(token.kind, TokenKind::T_USER_DEFINED_STRING_LITERAL);
  EXPECT_EQ(token.text, R"src(R"(x)"_s)src");
}

TEST(Lexer, ScansUnterminatedRawStringLiteralsUpToTheEndOfTheSource) {
  const auto token = scanOne(R"src(R"delim(abc)src");
  EXPECT_EQ(token.kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(token.text, R"src(R"delim(abc)src");

  const auto truncated = scanOne(R"src(R"delim(abc)deli)src");
  EXPECT_EQ(truncated.kind, TokenKind::T_STRING_LITERAL);
  EXPECT_EQ(truncated.text, R"src(R"delim(abc)deli)src");
}

TEST(Lexer, ScansEmptyRawStringLiterals) {
  EXPECT_EQ(scanOne(R"src(R"()")src").text, R"src(R"()")src");
  EXPECT_EQ(scanOne(R"src(R"x()x")src").text, R"src(R"x()x")src");
}
