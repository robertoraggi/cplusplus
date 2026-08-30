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

#include "transport.h"

#include <cxx/cli.h>

#include <format>
#include <iostream>

namespace cxx::lsp {

Transport::~Transport() = default;

StreamTransport::StreamTransport(std::istream& input, std::ostream& output)
    : input_(input), output_(output) {}

auto StreamTransport::isOpen() const -> bool { return input_.good(); }

auto StreamTransport::readLine() -> std::optional<std::string> {
  std::string line;
  if (!std::getline(input_, line)) return std::nullopt;
  return line;
}

auto StreamTransport::readBytes(std::size_t count) -> std::string {
  std::string bytes(count, '\0');
  input_.read(bytes.data(), std::streamsize(count));
  bytes.resize(std::size_t(input_.gcount()));
  return bytes;
}

void StreamTransport::write(std::string_view text) {
#ifndef CXX_NO_THREADS
  auto locker = std::unique_lock(outputMutex_);
#endif

  output_.write(text.data(), std::streamsize(text.size()));
  output_.flush();
}

StdioTransport::StdioTransport() : StdioTransport(std::cin, std::cout) {}

StdioTransport::StdioTransport(std::istream& input, std::ostream& output)
    : StreamTransport(input, output) {}

auto StdioTransport::readHeaders()
    -> std::unordered_map<std::string, std::string> {
  std::unordered_map<std::string, std::string> headers;

  while (auto line = readLine()) {
    if (line->empty() || *line == "\r") break;

    const auto pos = line->find_first_of(':');

    if (pos == std::string::npos) continue;

    auto name = line->substr(0, pos);
    auto value = line->substr(pos + 1);

    name.erase(name.find_last_not_of(" \t\r\n") + 1);
    value.erase(0, value.find_first_not_of(" \t\r\n"));
    value.erase(value.find_last_not_of(" \t\r\n") + 1);

    headers.insert_or_assign(std::move(name), std::move(value));
  }

  return headers;
}

auto StdioTransport::nextMessage() -> std::optional<json> {
  const auto headers = readHeaders();

  const auto it = headers.find("Content-Length");

  if (it == headers.end()) return std::nullopt;

  const auto contentLength = std::stoul(it->second);

  return json::parse(readBytes(contentLength));
}

void StdioTransport::sendMessage(const json& message) {
  const auto text = message.dump();
  write(std::format("Content-Length: {}\r\n\r\n{}", text.size(), text));
}

JsonLinesTransport::JsonLinesTransport()
    : JsonLinesTransport(std::cin, std::cout) {}

JsonLinesTransport::JsonLinesTransport(std::istream& input,
                                       std::ostream& output)
    : StreamTransport(input, output) {}

auto JsonLinesTransport::nextMessage() -> std::optional<json> {
  while (auto line = readLine()) {
    if (line->empty()) continue;
    if (line->starts_with("#")) continue;
    return json::parse(*line);
  }

  return std::nullopt;
}

void JsonLinesTransport::sendMessage(const json& message) {
  write(std::format("{}\n", message.dump(2)));
}

auto createTransport(const CLI& cli) -> std::unique_ptr<Transport> {
  if (cli.opt_lsp_test) return std::make_unique<JsonLinesTransport>();
  return std::make_unique<StdioTransport>();
}

}  // namespace cxx::lsp
