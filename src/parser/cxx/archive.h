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

#include <cxx/private/archive_io.h>

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace cxx {

enum class ArchiveKind : std::uint8_t {
  kPrecompiledHeader = 1,
  kModuleInterface = 2,
  kHeaderUnit = 3,
};

enum class ArchiveSection : std::uint8_t {
  kStrings = 1,
  kSourceMap = 2,
  kNames = 3,
  kTypes = 4,
  kSymbols = 5,
  kConstants = 6,
  kAst = 7,
  kPreprocessor = 8,
  kSession = 9,
  kDependencies = 10,
};

struct ArchiveDependency {
  std::string fileName;
  std::string contentDigest;
  bool isSystemHeader = false;
};

struct ArchiveEnvelope {
  ArchiveKind kind = ArchiveKind::kPrecompiledHeader;
  std::uint32_t schemaMajor = 0;
  std::uint32_t schemaMinor = 0;
  std::string serializationAbi;
  std::string targetKey;
  std::string languageKey;
  std::string optionDigest;
  std::string artifactIdentity;
  std::vector<ArchiveDependency> dependencies;
};

class ArchiveWriter {
 public:
  static constexpr std::string_view kMagic = "CXXARCH";
  static constexpr std::uint32_t kSchemaMajor = 4;
  static constexpr std::uint32_t kSchemaMinor = 0;

  void setEnvelope(ArchiveEnvelope envelope) {
    envelope_ = std::move(envelope);
  }

  [[nodiscard]] auto envelope() -> ArchiveEnvelope& { return envelope_; }

  void addSection(ArchiveSection section, std::vector<std::uint8_t> data) {
    sections_[section] = std::move(data);
  }

  [[nodiscard]] auto operator()() -> std::vector<std::uint8_t>;

 private:
  ArchiveEnvelope envelope_;
  std::map<ArchiveSection, std::vector<std::uint8_t>> sections_;
};

class ArchiveReader {
 public:
  /**
   * Whether each section is checked against the checksum the writer recorded.
   * Off by default; a consumer decides that an artifact is current from the
   * inputs that produced it. The section directory and every record are
   * range-checked either way, so a truncated archive is still rejected.
   */
  void setVerifyChecksums(bool verifyChecksums) {
    verifyChecksums_ = verifyChecksums;
  }

  [[nodiscard]] auto operator()(std::span<const std::uint8_t> data) -> bool;

  [[nodiscard]] auto envelope() const -> const ArchiveEnvelope& {
    return envelope_;
  }

  [[nodiscard]] auto hasSection(ArchiveSection section) const -> bool {
    return sections_.contains(section);
  }

  [[nodiscard]] auto section(ArchiveSection section) const -> ByteReader;

  [[nodiscard]] auto error() const -> const std::string& { return error_; }

 private:
  ArchiveEnvelope envelope_;
  std::map<ArchiveSection, std::span<const std::uint8_t>> sections_;
  std::string error_;
  bool verifyChecksums_ = false;
};

[[nodiscard]] auto archiveChecksum(std::span<const std::uint8_t> data)
    -> std::uint32_t;

}  // namespace cxx
