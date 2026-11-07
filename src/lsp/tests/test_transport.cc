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

#include <cxx/lsp/transport.h>
#include <cxx/private/file.h>
#include <cxx/private/read_all.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <format>
#include <string>

using namespace cxx::lsp;

namespace {
using cxx::detail::File;

[[nodiscard]] auto temporaryFile() -> File { return File(std::tmpfile()); }
}  // namespace

TEST(Transport, StdioReadsConsecutiveByteCountedFrames) {
  auto input = temporaryFile();
  auto output = temporaryFile();
  ASSERT_TRUE(input);
  ASSERT_TRUE(output);
  std::string body = R"({"id":7,"text":"aé😀"})";
  auto frame = std::format(
      "Content-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
      body.size(), body);
  ASSERT_EQ(std::fwrite(frame.data(), 1, frame.size(), input.get()),
            frame.size());
  ASSERT_EQ(std::fwrite(frame.data(), 1, frame.size(), input.get()),
            frame.size());
  std::rewind(input.get());
  StdioTransport transport(input.get(), output.get());
  for (int i = 0; i < 2; ++i) {
    auto message = transport.nextMessage();
    ASSERT_TRUE(message);
    EXPECT_EQ(message->at("id").get<int>(), 7);
    EXPECT_EQ(message->at("text").get<std::string>(), "aé😀");
  }
  EXPECT_FALSE(transport.nextMessage());
  EXPECT_FALSE(transport.isOpen());
}

TEST(Transport, StdioFlushesCompleteLargeFrames) {
  auto input = temporaryFile();
  auto output = temporaryFile();
  ASSERT_TRUE(input);
  ASSERT_TRUE(output);
  auto message = json::parse(
      std::format("{{\"text\":\"{}\"}}", std::string(192 * 1024, 'x')));
  StdioTransport transport(input.get(), output.get());
  transport.sendMessage(message);
  std::rewind(output.get());
  auto frame = cxx::detail::readAll(output.get());
  ASSERT_TRUE(frame);
  auto body = message.dump();
  EXPECT_EQ(*frame,
            std::format("Content-Length: {}\r\n\r\n{}", body.size(), body));
}

TEST(Transport, JsonLinesHandlesCommentsAndUnterminatedFinalLines) {
  auto input = temporaryFile();
  auto output = temporaryFile();
  ASSERT_TRUE(input);
  ASSERT_TRUE(output);
  std::string text = "\n# comment\n{\"id\":1}\n{\"id\":2}";
  ASSERT_EQ(std::fwrite(text.data(), 1, text.size(), input.get()), text.size());
  std::rewind(input.get());
  JsonLinesTransport transport(input.get(), output.get());
  auto first = transport.nextMessage();
  auto last = transport.nextMessage();
  ASSERT_TRUE(first);
  ASSERT_TRUE(last);
  EXPECT_EQ(first->at("id").get<int>(), 1);
  EXPECT_EQ(last->at("id").get<int>(), 2);
  EXPECT_FALSE(transport.nextMessage());
  EXPECT_FALSE(transport.isOpen());
}

TEST(Transport, OutputFailuresCloseTheTransport) {
  auto input = temporaryFile();
  File output(std::fopen(__FILE__, "r"));
  ASSERT_TRUE(input);
  ASSERT_TRUE(output);
  JsonLinesTransport transport(input.get(), output.get());
  transport.sendMessage(json::parse("{\"id\":1}"));
  EXPECT_FALSE(transport.isOpen());
  EXPECT_NE(std::ferror(output.get()), 0);
}
