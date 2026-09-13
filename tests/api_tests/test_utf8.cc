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

#include <cxx/private/utf8.h>
#include <gtest/gtest.h>

#include <string_view>

using namespace cxx;

namespace {

struct Decoded {
  std::uint32_t codePoint = 0;
  std::ptrdiff_t consumed = 0;
};

auto decodeFirst(std::string_view text) -> Decoded {
  auto it = text.begin();
  const auto codePoint = utf8::next(it, text.end());
  return {codePoint, it - text.begin()};
}

auto decodeLast(std::string_view text) -> Decoded {
  auto it = text.end();
  const auto codePoint = utf8::prior(it, text.begin());
  return {codePoint, text.end() - it};
}

}  // namespace

TEST(Utf8, DecodesTheFourSequenceLengths) {
  EXPECT_EQ(decodeFirst("a").codePoint, 0x61u);
  EXPECT_EQ(decodeFirst("a").consumed, 1);

  EXPECT_EQ(decodeFirst("\xc3\xa9").codePoint, 0xe9u);
  EXPECT_EQ(decodeFirst("\xc3\xa9").consumed, 2);

  EXPECT_EQ(decodeFirst("\xe2\x82\xac").codePoint, 0x20acu);
  EXPECT_EQ(decodeFirst("\xe2\x82\xac").consumed, 3);

  EXPECT_EQ(decodeFirst("\xf0\x9f\x98\x80").codePoint, 0x1f600u);
  EXPECT_EQ(decodeFirst("\xf0\x9f\x98\x80").consumed, 4);
}

TEST(Utf8, DecodesTheSequenceBoundaries) {
  EXPECT_EQ(decodeFirst(std::string_view{"\0", 1}).codePoint, 0u);
  EXPECT_EQ(decodeFirst(std::string_view{"\0", 1}).consumed, 1);

  EXPECT_EQ(decodeFirst("\x7f").codePoint, 0x7fu);
  EXPECT_EQ(decodeFirst("\xc2\x80").codePoint, 0x80u);
  EXPECT_EQ(decodeFirst("\xdf\xbf").codePoint, 0x7ffu);
  EXPECT_EQ(decodeFirst("\xe0\xa0\x80").codePoint, 0x800u);
  EXPECT_EQ(decodeFirst("\xef\xbf\xbf").codePoint, 0xffffu);
  EXPECT_EQ(decodeFirst("\xf0\x90\x80\x80").codePoint, 0x10000u);
  EXPECT_EQ(decodeFirst("\xf4\x8f\xbf\xbf").codePoint, 0x10ffffu);
}

TEST(Utf8, NextStopsAtTheEndOfTheRange) {
  std::string_view empty;
  auto it = empty.begin();
  EXPECT_EQ(utf8::next(it, empty.end()), 0u);
  EXPECT_EQ(it, empty.end());
}

TEST(Utf8, NextConsumesOneOctetWhenTheSequenceIsTruncated) {
  EXPECT_EQ(decodeFirst("\xe2\x82").codePoint, 0xe2u);
  EXPECT_EQ(decodeFirst("\xe2\x82").consumed, 1);

  EXPECT_EQ(decodeFirst("\xf0\x9f").codePoint, 0xf0u);
  EXPECT_EQ(decodeFirst("\xf0\x9f").consumed, 1);

  EXPECT_EQ(decodeFirst("\xc3").codePoint, 0xc3u);
  EXPECT_EQ(decodeFirst("\xc3").consumed, 1);
}

TEST(Utf8, NextConsumesOneOctetWhenTheSequenceIsIllFormed) {
  EXPECT_EQ(decodeFirst("\x80").codePoint, 0x80u);
  EXPECT_EQ(decodeFirst("\x80").consumed, 1);

  EXPECT_EQ(decodeFirst("\xff").codePoint, 0xffu);
  EXPECT_EQ(decodeFirst("\xff").consumed, 1);

  EXPECT_EQ(decodeFirst("\xe2\x41\x42").codePoint, 0xe2u);
  EXPECT_EQ(decodeFirst("\xe2\x41\x42").consumed, 1);

  EXPECT_EQ(decodeFirst("\xf0\x9f\x98\x41").codePoint, 0xf0u);
  EXPECT_EQ(decodeFirst("\xf0\x9f\x98\x41").consumed, 1);
}

TEST(Utf8, PeekNextLeavesTheIteratorAlone) {
  const std::string_view text =
      "\xe2\x82\xac"
      "z";
  auto it = text.begin();
  EXPECT_EQ(utf8::peekNext(it, text.end()), 0x20acu);
  EXPECT_EQ(it, text.begin());
}

TEST(Utf8, PriorWalksBackOverCompleteSequences) {
  EXPECT_EQ(decodeLast("a").codePoint, 0x61u);
  EXPECT_EQ(decodeLast("a").consumed, 1);

  EXPECT_EQ(decodeLast("\xc3\xa9").codePoint, 0xe9u);
  EXPECT_EQ(decodeLast("\xc3\xa9").consumed, 2);

  EXPECT_EQ(decodeLast("\xe2\x82\xac").codePoint, 0x20acu);
  EXPECT_EQ(decodeLast("\xe2\x82\xac").consumed, 3);

  EXPECT_EQ(decodeLast("\xf0\x9f\x98\x80").codePoint, 0x1f600u);
  EXPECT_EQ(decodeLast("\xf0\x9f\x98\x80").consumed, 4);
}

TEST(Utf8, PriorStopsAtTheStartOfTheRange) {
  std::string_view empty;
  auto it = empty.begin();
  EXPECT_EQ(utf8::prior(it, empty.begin()), 0u);
  EXPECT_EQ(it, empty.begin());
}

TEST(Utf8, PriorConsumesOneOctetWhenTheSequenceIsIllFormed) {
  EXPECT_EQ(decodeLast("a\x80").codePoint, 0x80u);
  EXPECT_EQ(decodeLast("a\x80").consumed, 1);

  EXPECT_EQ(decodeLast("\xe2\x82\xac\x80").codePoint, 0x80u);
  EXPECT_EQ(decodeLast("\xe2\x82\xac\x80").consumed, 1);

  EXPECT_EQ(decodeLast("\x80\x80\x80\x80\x80").codePoint, 0x80u);
  EXPECT_EQ(decodeLast("\x80\x80\x80\x80\x80").consumed, 1);
}

TEST(Utf8, PriorAndNextRoundTrip) {
  const std::string_view text = "a\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80";

  auto forward = text.begin();
  while (forward != text.end()) {
    const auto start = forward;
    const auto codePoint = utf8::next(forward, text.end());

    auto backward = forward;
    EXPECT_EQ(utf8::prior(backward, text.begin()), codePoint);
    EXPECT_EQ(backward, start);
  }
}

TEST(Utf8, DistanceCountsCodePoints) {
  const std::string_view text = "a\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80";
  EXPECT_EQ(text.size(), 10u);
  EXPECT_EQ(utf8::distance(text.begin(), text.end()), 4);

  std::string_view empty;
  EXPECT_EQ(utf8::distance(empty.begin(), empty.end()), 0);
}

TEST(Utf8, DistanceCountsIllFormedOctetsOneByOne) {
  const std::string_view text = "\xff\xe2\x41";
  EXPECT_EQ(utf8::distance(text.begin(), text.end()), 3);
}
