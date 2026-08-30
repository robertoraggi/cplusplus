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

#pragma once

#include <cxx/lsp/json.h>

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#ifndef CXX_NO_THREADS
#include <mutex>
#endif

namespace cxx {
class CLI;
}

namespace cxx::lsp {

class Transport {
 public:
  Transport() = default;
  virtual ~Transport();

  Transport(const Transport&) = delete;
  auto operator=(const Transport&) -> Transport& = delete;

  [[nodiscard]] virtual auto isOpen() const -> bool = 0;

  [[nodiscard]] virtual auto nextMessage() -> std::optional<json> = 0;

  virtual void sendMessage(const json& message) = 0;
};

class StreamTransport : public Transport {
 public:
  StreamTransport(std::istream& input, std::ostream& output);

  [[nodiscard]] auto isOpen() const -> bool override;

 protected:
  [[nodiscard]] auto readLine() -> std::optional<std::string>;

  [[nodiscard]] auto readBytes(std::size_t count) -> std::string;

  void write(std::string_view text);

 private:
  std::istream& input_;
  std::ostream& output_;
#ifndef CXX_NO_THREADS
  std::mutex outputMutex_;
#endif
};

class StdioTransport final : public StreamTransport {
 public:
  StdioTransport();
  StdioTransport(std::istream& input, std::ostream& output);

  [[nodiscard]] auto nextMessage() -> std::optional<json> override;

  void sendMessage(const json& message) override;

 private:
  [[nodiscard]] auto readHeaders()
      -> std::unordered_map<std::string, std::string>;
};

class JsonLinesTransport final : public StreamTransport {
 public:
  JsonLinesTransport();
  JsonLinesTransport(std::istream& input, std::ostream& output);

  [[nodiscard]] auto nextMessage() -> std::optional<json> override;

  void sendMessage(const json& message) override;
};

[[nodiscard]] auto createTransport(const CLI& cli)
    -> std::unique_ptr<Transport>;

}  // namespace cxx::lsp
