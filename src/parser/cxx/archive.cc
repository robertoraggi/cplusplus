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

#include <cxx/archive.h>

#include <array>
#include <bit>
#include <cstring>

namespace cxx {

namespace {

auto readWord(const std::uint8_t* bytes) -> std::uint64_t {
  std::uint64_t word = 0;
  std::memcpy(&word, bytes, sizeof(word));
  if constexpr (std::endian::native == std::endian::big)
    word = std::byteswap(word);
  return word;
}

}  // namespace

auto archiveChecksum(std::span<const std::uint8_t> data) -> std::uint32_t {
  constexpr std::uint64_t kPrime = 1099511628211ull;
  constexpr std::uint64_t kBasis = 14695981039346656037ull;

  std::array<std::uint64_t, 4> lanes{kBasis, kBasis ^ 1, kBasis ^ 2,
                                     kBasis ^ 3};

  const auto* first = data.data();
  auto size = data.size();

  while (size >= 4 * sizeof(std::uint64_t)) {
    for (std::size_t lane = 0; lane < lanes.size(); ++lane)
      lanes[lane] = (lanes[lane] ^ readWord(first + lane * 8)) * kPrime;
    first += 4 * sizeof(std::uint64_t);
    size -= 4 * sizeof(std::uint64_t);
  }

  auto hash =
      ((lanes[0] ^ lanes[1]) * kPrime) ^ ((lanes[2] ^ lanes[3]) * kPrime);

  while (size >= sizeof(std::uint64_t)) {
    hash = (hash ^ readWord(first)) * kPrime;
    first += sizeof(std::uint64_t);
    size -= sizeof(std::uint64_t);
  }

  if (size) {
    std::uint64_t tail = 0;
    std::memcpy(&tail, first, size);
    if constexpr (std::endian::native == std::endian::big)
      tail = std::byteswap(tail);
    hash = (hash ^ tail) * kPrime;
  }

  hash ^= data.size();
  hash ^= hash >> 32;
  hash *= kPrime;
  hash ^= hash >> 29;

  return static_cast<std::uint32_t>(hash);
}

auto ArchiveWriter::operator()() -> std::vector<std::uint8_t> {
  ByteWriter out;

  out.str(kMagic);
  out.u8(static_cast<std::uint8_t>(envelope_.kind));
  out.u32(kSchemaMajor);
  out.u32(kSchemaMinor);
  out.str(envelope_.serializationAbi);
  out.str(envelope_.targetKey);
  out.str(envelope_.languageKey);
  out.str(envelope_.optionDigest);
  out.str(envelope_.artifactIdentity);

  out.u32(static_cast<std::uint32_t>(envelope_.dependencies.size()));
  for (const auto& dependency : envelope_.dependencies) {
    out.str(dependency.fileName);
    out.str(dependency.contentDigest);
    out.boolean(dependency.isSystemHeader);
  }

  out.u32(static_cast<std::uint32_t>(sections_.size()));

  std::vector<std::size_t> offsetSlots;
  offsetSlots.reserve(sections_.size());

  for (const auto& [section, data] : sections_) {
    out.u8(static_cast<std::uint8_t>(section));
    out.u32(static_cast<std::uint32_t>(data.size()));
    out.u32(archiveChecksum(data));
    offsetSlots.push_back(out.reserveU32());
  }

  auto slot = offsetSlots.begin();
  for (const auto& [section, data] : sections_) {
    out.patchU32(*slot++, static_cast<std::uint32_t>(out.size()));
    out.append(data);
  }

  return out.take();
}

auto ArchiveReader::operator()(std::span<const std::uint8_t> data) -> bool {
  ByteReader in{data};

  if (in.str() != ArchiveWriter::kMagic) {
    error_ = "not a cxx archive";
    return false;
  }

  const auto kind = in.u8();
  if (kind < static_cast<std::uint8_t>(ArchiveKind::kPrecompiledHeader) ||
      kind > static_cast<std::uint8_t>(ArchiveKind::kHeaderUnit)) {
    error_ = "unknown archive kind";
    return false;
  }

  envelope_.kind = static_cast<ArchiveKind>(kind);
  envelope_.schemaMajor = in.u32();
  envelope_.schemaMinor = in.u32();

  if (!in.ok()) {
    error_ = "archive envelope is truncated";
    return false;
  }

  if (envelope_.schemaMajor != ArchiveWriter::kSchemaMajor) {
    error_ = "archive was built by a different compiler version";
    return false;
  }

  envelope_.serializationAbi = in.str();
  envelope_.targetKey = in.str();
  envelope_.languageKey = in.str();
  envelope_.optionDigest = in.str();
  envelope_.artifactIdentity = in.str();

  const auto dependencyCount = in.count(3 * 4 + 1);
  for (std::uint32_t i = 0; in.ok() && i < dependencyCount; ++i) {
    ArchiveDependency dependency;
    dependency.fileName = in.str();
    dependency.contentDigest = in.str();
    dependency.isSystemHeader = in.boolean();
    envelope_.dependencies.push_back(std::move(dependency));
  }

  const auto sectionCount = in.count(1 + 4 + 4 + 4);

  struct Entry {
    ArchiveSection section;
    std::uint32_t size;
    std::uint32_t checksum;
    std::uint32_t offset;
  };

  std::vector<Entry> entries;

  for (std::uint32_t i = 0; in.ok() && i < sectionCount; ++i) {
    Entry entry{};
    entry.section = static_cast<ArchiveSection>(in.u8());
    entry.size = in.u32();
    entry.checksum = in.u32();
    entry.offset = in.u32();
    entries.push_back(entry);
  }

  if (!in.ok()) {
    error_ = "archive section directory is truncated";
    return false;
  }

  for (const auto& entry : entries) {
    if (entry.offset > data.size() || entry.size > data.size() - entry.offset) {
      error_ = "archive section is out of bounds";
      return false;
    }

    auto bytes = data.subspan(entry.offset, entry.size);

    if (verifyChecksums_ && archiveChecksum(bytes) != entry.checksum) {
      error_ = "archive section is corrupt";
      return false;
    }

    if (!sections_.emplace(entry.section, bytes).second) {
      error_ = "archive contains a duplicate section";
      return false;
    }
  }

  return true;
}

auto ArchiveReader::section(ArchiveSection section) const -> ByteReader {
  auto it = sections_.find(section);
  if (it == sections_.end()) return ByteReader{};
  return ByteReader{it->second};
}

}  // namespace cxx
