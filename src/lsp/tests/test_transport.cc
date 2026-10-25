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
