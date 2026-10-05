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

#include <cxx/linker/bytes.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cxx::linker {

struct ArchiveMember {
  std::string name;
  ByteView content;
};

class Archive {
 public:
  [[nodiscard]] static auto isArchive(ByteView content) -> bool;

  [[nodiscard]] static auto create(std::string name,
                                   std::shared_ptr<const Bytes> content,
                                   std::string& error)
      -> std::shared_ptr<const Archive>;

  [[nodiscard]] auto find(std::string_view symbol) const
      -> std::optional<std::uint32_t>;

  [[nodiscard]] auto member(std::uint32_t offset) const
      -> std::optional<ArchiveMember>;

 private:
  Archive(std::string name, std::shared_ptr<const Bytes> content)
      : name_(std::move(name)), content_(std::move(content)) {}

  std::string name_;
  std::shared_ptr<const Bytes> content_;
  std::unordered_map<std::string_view, std::uint32_t> symbols_;

  [[nodiscard]] auto parse(std::string& error) -> bool;

  [[nodiscard]] auto header(std::size_t offset) const
      -> std::optional<ArchiveMember>;
};

}  // namespace cxx::linker
