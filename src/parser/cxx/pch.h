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

#include <cxx/archive.h>
#include <cxx/preprocessor_snapshot.h>
#include <cxx/token_fwd.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace cxx {

class TranslationUnit;

struct PrecompiledHeaderKeys {
  std::string serializationAbi;
  std::string targetKey;
  std::string languageKey;
  std::string optionDigest;
};

/**
 * Writes a precompiled header: the preprocessor continuation state of 9.4 and
 * the semantic graph of 9.1, reachable from the committed prefix (7.4).
 */
class PrecompiledHeaderWriter {
 public:
  PrecompiledHeaderWriter(TranslationUnit* unit, PrecompiledHeaderKeys keys)
      : unit_(unit), keys_(std::move(keys)) {}

  void setPreprocessorState(PreprocessorSnapshot snapshot) {
    preprocessorState_ = std::move(snapshot);
  }

  void addDependency(ArchiveDependency dependency) {
    dependencies_.push_back(std::move(dependency));
  }

  [[nodiscard]] auto operator()() -> std::vector<std::uint8_t>;

  [[nodiscard]] auto errors() const -> const std::vector<std::string>& {
    return errors_;
  }

 private:
  TranslationUnit* unit_ = nullptr;
  PrecompiledHeaderKeys keys_;
  PreprocessorSnapshot preprocessorState_;
  std::vector<ArchiveDependency> dependencies_;
  std::vector<std::string> errors_;
};

/**
 * Validates a precompiled header, restores the preprocessor and adopts the
 * decoded graph as the committed prefix of `unit` (8.1, 9.5). On failure
 * nothing is adopted and `unit` is left untouched.
 */
class PrecompiledHeaderReader {
 public:
  PrecompiledHeaderReader(TranslationUnit* unit, PrecompiledHeaderKeys keys)
      : unit_(unit), keys_(std::move(keys)) {}

  /** See `ArchiveReader::setVerifyChecksums`. */
  void setVerifyChecksums(bool verifyChecksums) {
    verifyChecksums_ = verifyChecksums;
  }

  [[nodiscard]] auto operator()(std::span<const std::uint8_t> data) -> bool;

  [[nodiscard]] auto error() const -> const std::string& { return error_; }

  [[nodiscard]] auto dependencies() const
      -> const std::vector<ArchiveDependency>& {
    return dependencies_;
  }

 private:
  TranslationUnit* unit_ = nullptr;
  PrecompiledHeaderKeys keys_;
  std::vector<ArchiveDependency> dependencies_;
  std::string error_;
  bool verifyChecksums_ = false;
};

[[nodiscard]] auto precompiledHeaderSerializationAbi() -> std::string;

}  // namespace cxx
