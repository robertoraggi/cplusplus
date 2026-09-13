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

#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/semantic_archive.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>

#include <algorithm>
#include <limits>
#include <numeric>

namespace cxx {

auto SemanticEncoderBase::stringRef(std::string_view text) -> StringRef {
  if (text.empty()) return StringRef{0};

  if (auto it = stringIndexOf_.find(text); it != stringIndexOf_.end())
    return StringRef{it->second};

  const auto index = static_cast<std::uint32_t>(strings_.size() + 1);
  strings_.emplace_back(text);
  stringIndexOf_.emplace(strings_.back(), index);

  return StringRef{index};
}

auto SemanticEncoderBase::locationRef(SourceLocation loc) -> LocationRef {
  if (!loc) return LocationRef{0};

  if (referencedLocations_.insert(loc.index()).second) {
    LocationRecord record;
    record.ordinal = loc.index();
    locations_.push_back(record);
  }

  return LocationRef{loc.index()};
}

void SemanticEncoderBase::resolveLocations() {
  std::ranges::sort(locations_, {}, &LocationRecord::ordinal);

  int lastFileId = -1;
  std::uint32_t lastFile = 0;

  auto resolved = locations_.begin();

  for (auto& record : locations_) {
    const SourceLocation loc{record.ordinal};
    if (!unit_->ownsLocation(loc)) continue;

    auto start = unit_->tokenStartPosition(loc);
    auto end = unit_->tokenEndPosition(loc);

    const auto fileId = unit_->tokenAt(loc).fileId();
    if (fileId != lastFileId) {
      lastFileId = fileId;
      auto [it, inserted] = locationFileIndexOf_.try_emplace(
          std::string{start.fileName},
          static_cast<std::uint32_t>(locationFiles_.size()));
      if (inserted) locationFiles_.emplace_back(start.fileName);
      lastFile = it->second;
    }

    auto presumed = unit_->presumedTokenStartPosition(loc);
    auto [presumedFile, inserted] = locationFileIndexOf_.try_emplace(
        std::string{presumed.fileName},
        static_cast<std::uint32_t>(locationFiles_.size()));
    if (inserted) locationFiles_.emplace_back(presumed.fileName);
    record.presumedFileName = presumedFile->second;
    record.presumedLine = presumed.line;
    record.fileName = lastFile;
    record.startLine = start.line;
    record.startColumn = start.column;
    record.endLine = end.line;
    record.endColumn = end.column;

    *resolved++ = record;
  }

  locations_.erase(resolved, locations_.end());
}

auto SemanticEncoderBase::stringAt(StringRef ref) const -> std::string_view {
  const auto index = static_cast<std::uint32_t>(ref);
  if (index == 0 || index > strings_.size()) return {};
  return strings_[index - 1];
}

auto SemanticEncoderBase::constNodeRef(std::shared_ptr<void> owner, int kind)
    -> std::uint32_t {
  if (!owner) return 0;

  auto [it, inserted] = constIndexOf_.emplace(
      owner.get(), static_cast<std::uint32_t>(constIndexOf_.size() + 1));

  if (inserted) {
    constPending_.push_back({std::move(owner), kind});
    constRecords_.emplace_back();
  }

  return it->second;
}

auto SemanticEncoderBase::entrySortKey(const Symbol* symbol) const
    -> std::pair<unsigned, std::string_view> {
  if (!symbol) return {0, {}};

  std::string_view name;
  if (auto id = name_cast<Identifier>(symbol->name())) name = id->name();

  return {symbol->location().index(), name};
}

void SemanticEncoderBase::flushConstants(ByteWriter& out) const {
  out.varU32(static_cast<std::uint32_t>(constRecords_.size()));
  for (const auto& record : constRecords_) out.bytes(record);
}

void SemanticEncoderBase::flushStrings(ByteWriter& out) const {
  out.varU32(static_cast<std::uint32_t>(strings_.size()));
  for (const auto& text : strings_) out.str(text);
}

void SemanticEncoderBase::flushSourceMap(ByteWriter& out) const {
  out.varU32(static_cast<std::uint32_t>(locationFiles_.size()));
  for (const auto& fileName : locationFiles_) out.str(fileName);

  out.varU32(static_cast<std::uint32_t>(locations_.size()));

  unsigned previousOrdinal = 0;

  for (const auto& record : locations_) {
    out.varU32(record.ordinal - previousOrdinal);
    previousOrdinal = record.ordinal;
    out.varU32(record.fileName);
    out.varU32(record.startLine);
    out.varU32(record.startColumn);
    out.varI32(static_cast<std::int32_t>(record.endLine) -
               static_cast<std::int32_t>(record.startLine));
    out.varU32(record.endColumn);
    out.varU32(record.presumedFileName);
    out.varU32(record.presumedLine);
  }
}

auto SemanticDecoderBase::readStrings(ByteReader& in) -> bool {
  const auto count = in.varCount(1);
  strings_.reserve(count);
  for (std::uint32_t i = 0; in.ok() && i < count; ++i)
    strings_.push_back(in.str());
  if (!in.ok()) fail("string section is truncated");
  return in.ok();
}

auto SemanticDecoderBase::readSourceMap(ByteReader& in) -> bool {
  const auto fileCount = in.varCount(1);

  std::vector<std::string> files;
  files.reserve(fileCount);
  for (std::uint32_t i = 0; in.ok() && i < fileCount; ++i)
    files.push_back(in.str());

  const auto count = in.varCount(8);
  auto entries = in.rest();

  if (!in.ok()) {
    fail("source map section is truncated");
    return false;
  }

  sourceMap_ = std::make_unique<PrefixSourceMap>(
      std::move(files),
      std::vector<std::uint8_t>(entries.begin(), entries.end()), count);

  return true;
}

auto PrefixSourceMap::decodedEntries() const -> const std::vector<Entry>& {
  if (decoded_) return entries_;
  decoded_ = true;

  ByteReader in{encodedEntries_};
  entries_.reserve(count_);

  std::uint64_t ordinal = 0;

  for (std::size_t i = 0; in.ok() && i < count_; ++i) {
    ordinal += in.varU32();

    const auto fileName = in.varU32();

    PrefixSourceLocationInfo info;
    if (fileName < files_.size()) info.fileName = files_[fileName];
    info.startLine = in.varU32();
    info.startColumn = in.varU32();
    info.endLine = static_cast<std::uint32_t>(
        static_cast<std::int64_t>(info.startLine) + in.varI32());
    info.endColumn = in.varU32();
    auto presumedFile = in.varU32();
    if (presumedFile < files_.size())
      info.presumedFileName = files_[presumedFile];
    info.presumedLine = in.varU32();

    if (ordinal > std::numeric_limits<unsigned>::max()) break;

    entries_.push_back({static_cast<unsigned>(ordinal), info});
  }

  if (!in.ok()) entries_.clear();

  return entries_;
}

auto SemanticDecoderBase::stringAt(StringRef ref) const -> std::string_view {
  const auto index = static_cast<std::uint32_t>(ref);
  if (index == 0 || index > strings_.size()) return {};
  return strings_[index - 1];
}

auto SemanticDecoderBase::identifierAt(StringRef ref) -> const Identifier* {
  return internedAt(
      identifiers_, ref, [](Control* control, std::string_view text) {
        return text.empty() ? nullptr : control->getIdentifier(text);
      });
}

auto SemanticDecoderBase::readStringLiteral(ByteReader& in)
    -> const StringLiteral* {
  if (!in.boolean()) return nullptr;
  return internedAt(stringLiterals_, StringRef{in.varU32()},
                    [](Control* control, std::string_view text) {
                      return control->stringLiteral(text);
                    });
}

auto SemanticDecoderBase::readCharLiteral(ByteReader& in)
    -> const CharLiteral* {
  if (!in.boolean()) return nullptr;
  return internedAt(charLiterals_, StringRef{in.varU32()},
                    [](Control* control, std::string_view text) {
                      return control->charLiteral(text);
                    });
}

auto SemanticDecoderBase::readIntegerLiteral(ByteReader& in)
    -> const IntegerLiteral* {
  if (!in.boolean()) return nullptr;
  return internedAt(integerLiterals_, StringRef{in.varU32()},
                    [](Control* control, std::string_view text) {
                      return control->integerLiteral(text);
                    });
}

auto SemanticDecoderBase::readFloatLiteral(ByteReader& in)
    -> const FloatLiteral* {
  if (!in.boolean()) return nullptr;
  return internedAt(floatLiterals_, StringRef{in.varU32()},
                    [](Control* control, std::string_view text) {
                      return control->floatLiteral(text);
                    });
}

auto SemanticDecoderBase::control() const -> Control* {
  return unit_->control();
}

auto SemanticDecoderBase::arena() const -> Arena* { return unit_->arena(); }

auto SemanticDecoderBase::readRecords(ByteReader& in,
                                      std::vector<Record>& records) -> bool {
  const auto count = in.varCount(1);
  records.reserve(count);
  for (std::uint32_t i = 0; in.ok() && i < count; ++i)
    records.push_back(Record{in.byteSpan()});
  return in.ok();
}

}  // namespace cxx
