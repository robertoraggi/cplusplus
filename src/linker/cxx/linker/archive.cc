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

#include <cxx/linker/archive.h>

#include <charconv>
#include <cstring>

namespace cxx::linker {

namespace {

constexpr std::string_view kMagic = "!<arch>\n";
constexpr std::size_t kHeaderSize = 60;
constexpr std::size_t kNameSize = 16;
constexpr std::size_t kSizeOffset = 48;
constexpr std::size_t kSizeLength = 10;

[[nodiscard]] auto trimmed(std::string_view text) -> std::string_view {
  while (!text.empty() && (text.back() == ' ' || text.back() == '/')) {
    text.remove_suffix(1);
  }
  return text;
}

[[nodiscard]] auto bigEndian(const std::uint8_t* at) -> std::uint32_t {
  return std::uint32_t{at[0]} << 24 | std::uint32_t{at[1]} << 16 |
         std::uint32_t{at[2]} << 8 | std::uint32_t{at[3]};
}

}  // namespace

auto Archive::isArchive(ByteView content) -> bool {
  const std::string_view head{reinterpret_cast<const char*>(content.data()),
                              std::min(content.size(), kMagic.size())};
  return head == kMagic;
}

auto Archive::create(std::string name, std::shared_ptr<const Bytes> content,
                     std::string& error) -> std::shared_ptr<const Archive> {
  std::shared_ptr<Archive> archive{
      new Archive(std::move(name), std::move(content))};
  if (!archive->parse(error)) return nullptr;
  return archive;
}

auto Archive::header(std::size_t offset) const -> std::optional<ArchiveMember> {
  const ByteView bytes = *content_;
  if (offset + kHeaderSize > bytes.size()) return std::nullopt;
  const std::string_view header{
      reinterpret_cast<const char*>(bytes.data()) + offset, kHeaderSize};
  const auto digits = header.substr(kSizeOffset, kSizeLength);
  std::size_t size = 0;
  const auto end = digits.data() + trimmed(digits).size();
  if (std::from_chars(digits.data(), end, size).ec != std::errc{}) {
    return std::nullopt;
  }
  if (offset + kHeaderSize + size > bytes.size()) return std::nullopt;
  auto name = std::string{trimmed(header.substr(0, kNameSize))};
  return ArchiveMember{std::move(name),
                       bytes.subspan(offset + kHeaderSize, size)};
}

auto Archive::parse(std::string& error) -> bool {
  if (!isArchive(*content_)) {
    error = concat(name_, ": not an archive");
    return false;
  }
  const auto index = header(kMagic.size());
  if (!index || !index->name.empty() || index->content.size() < 4) {
    error = concat(name_, ": missing symbol index");
    return false;
  }
  const auto* data = index->content.data();
  const auto count = bigEndian(data);
  if (index->content.size() < 4 + std::size_t{count} * 4) {
    error = concat(name_, ": malformed symbol index");
    return false;
  }
  const auto* names = reinterpret_cast<const char*>(data + 4 + count * 4);
  const auto* limit =
      reinterpret_cast<const char*>(data + index->content.size());
  for (std::uint32_t i = 0; i < count && names < limit; ++i) {
    const std::string_view name{names, strnlen(names, limit - names)};
    symbols_.try_emplace(name, bigEndian(data + 4 + i * 4));
    names += name.size() + 1;
  }
  return true;
}

auto Archive::find(std::string_view symbol) const
    -> std::optional<std::uint32_t> {
  const auto it = symbols_.find(symbol);
  if (it == symbols_.end()) return std::nullopt;
  return it->second;
}

auto Archive::member(std::uint32_t offset) const
    -> std::optional<ArchiveMember> {
  auto result = header(offset);
  if (result) result->name = concat(name_, "(", result->name, ")");
  return result;
}

}  // namespace cxx::linker
