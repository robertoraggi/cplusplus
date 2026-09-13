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
#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>
#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cxx {

class Control;
class Arena;
class TranslationUnit;

enum class StringRef : std::uint32_t {};
enum class NameRef : std::uint32_t {};
enum class TypeRef : std::uint32_t {};
enum class SymbolRef : std::uint32_t {};
enum class AstRef : std::uint32_t {};
enum class ConstRef : std::uint32_t {};
enum class LocationRef : std::uint32_t {};

struct PrefixSourceLocationInfo {
  std::string_view fileName;
  std::string_view presumedFileName;
  std::uint32_t presumedLine = 0;
  std::uint32_t startLine = 0;
  std::uint32_t startColumn = 0;
  std::uint32_t endLine = 0;
  std::uint32_t endColumn = 0;
};

/**
 * The token-position table of an adopted prefix, expanded on the first query.
 * The encoder writes the block in ascending ordinal order, and the block
 * carries its own file names, so the map outlives both the archive and the
 * decoder's string table.
 */
class PrefixSourceMap {
 public:
  PrefixSourceMap(std::vector<std::string> files,
                  std::vector<std::uint8_t> entries, std::size_t count)
      : files_(std::move(files)),
        encodedEntries_(std::move(entries)),
        count_(count) {}

  [[nodiscard]] auto positionOf(unsigned ordinal) const
      -> std::optional<PrefixSourceLocationInfo> {
    const auto& entries = decodedEntries();
    auto it = std::ranges::upper_bound(entries, ordinal, {}, &Entry::ordinal);
    if (it == entries.begin()) return std::nullopt;
    --it;
    if (it->ordinal == ordinal) return it->info;
    return afterLastTokenOf(it->info);
  }

  [[nodiscard]] auto empty() const -> bool { return count_ == 0; }

 private:
  struct Entry {
    unsigned ordinal = 0;
    PrefixSourceLocationInfo info;
  };

  [[nodiscard]] auto decodedEntries() const -> const std::vector<Entry>&;

  [[nodiscard]] static auto afterLastTokenOf(
      const PrefixSourceLocationInfo& recorded) -> PrefixSourceLocationInfo {
    return {.fileName = recorded.fileName,
            .presumedFileName = recorded.presumedFileName,
            .presumedLine =
                recorded.presumedLine + recorded.endLine - recorded.startLine,
            .startLine = recorded.endLine,
            .startColumn = recorded.endColumn,
            .endLine = recorded.endLine,
            .endColumn = recorded.endColumn};
  }

  std::vector<std::string> files_;
  std::vector<std::uint8_t> encodedEntries_;
  std::size_t count_ = 0;
  mutable std::vector<Entry> entries_;
  mutable bool decoded_ = false;
};

struct SemanticArchiveRoots {
  ScopeSymbol* globalScope = nullptr;
  UnitAST* ast = nullptr;
  int anonymousIdCount = 0;
  int closureNameCount = 0;
  unsigned prefixTokenCount = 0;
  std::vector<FunctionSymbol*> pendingBodyCompletions;
  std::vector<ClassSymbol*> pendingMemberInstantiations;
  std::vector<ClassSymbol*> instantiatedMemberClasses;
  std::vector<std::pair<std::uint64_t, std::string>> snippets;
};

/**
 * A pointer-keyed index table. Encoding a translation unit does millions of
 * these lookups — one per reference in every record — and an open-addressing
 * table with a power-of-two mask beats a node-based map by avoiding both the
 * pointer chase and the modulo.
 */
class PointerIndexTable {
 public:
  PointerIndexTable() : slots_(1u << 12) {}

  [[nodiscard]] auto size() const -> std::size_t { return size_; }

  /** The stored index, or 0 when the key is absent. */
  [[nodiscard]] auto find(const void* key) const -> std::uint32_t {
    auto mask = slots_.size() - 1;
    auto i = hash(key) & mask;
    for (;;) {
      const auto& slot = slots_[i];
      if (!slot.key) return 0;
      if (slot.key == key) return slot.index;
      i = (i + 1) & mask;
    }
  }

  void insert(const void* key, std::uint32_t index) {
    if ((size_ + 1) * 4 >= slots_.size() * 3) grow();
    place(slots_, key, index);
    ++size_;
  }

 private:
  struct Slot {
    const void* key = nullptr;
    std::uint32_t index = 0;
  };

  [[nodiscard]] static auto hash(const void* key) -> std::size_t {
    return std::hash<const void*>{}(key);
  }

  static void place(std::vector<Slot>& slots, const void* key,
                    std::uint32_t index) {
    auto mask = slots.size() - 1;
    auto i = hash(key) & mask;
    while (slots[i].key) i = (i + 1) & mask;
    slots[i] = {key, index};
  }

  void grow() {
    std::vector<Slot> slots(slots_.size() * 2);
    for (const auto& slot : slots_) {
      if (slot.key) place(slots, slot.key, slot.index);
    }
    slots_.swap(slots);
  }

  std::vector<Slot> slots_;
  std::size_t size_ = 0;
};

class SemanticEncoderBase {
 public:
  explicit SemanticEncoderBase(TranslationUnit* unit) : unit_(unit) {}

  [[nodiscard]] auto unit() const -> TranslationUnit* { return unit_; }

  [[nodiscard]] auto errors() const -> const std::vector<std::string>& {
    return errors_;
  }

 protected:
  /**
   * One interned reference domain. Records live end to end in a single buffer
   * addressed by extent, rather than one heap allocation each: a translation
   * unit has hundreds of thousands of them.
   */
  template <typename T>
  struct Domain {
    struct Extent {
      std::uint32_t offset = 0;
      std::uint32_t size = 0;
    };

    /**
     * An entity that carries scratch space numbers itself: the reference is a
     * field read rather than a hash probe, which is what most of encoding does.
     */
    static constexpr bool kNumbersItself =
        requires(T entity) { entity->setInternalId(0u); };

    ~Domain() {
      if constexpr (kNumbersItself) {
        for (auto entity : pending) entity->setInternalId(0);
      }
    }

    PointerIndexTable indexOf;
    std::vector<T> pending;
    std::vector<Extent> extents;
    std::vector<std::uint8_t> blob;
    std::size_t cursor = 0;

    [[nodiscard]] auto reference(T entity) -> std::uint32_t {
      if (!entity) return 0;

      if constexpr (kNumbersItself) {
        if (auto index = entity->internalId()) return index;

        const auto index = static_cast<std::uint32_t>(pending.size() + 1);
        entity->setInternalId(index);
        pending.push_back(entity);
        extents.emplace_back();

        return index;
      } else {
        const auto key = static_cast<const void*>(entity);

        if (auto index = indexOf.find(key)) return index;

        const auto index = static_cast<std::uint32_t>(indexOf.size() + 1);
        indexOf.insert(key, index);
        pending.push_back(entity);
        extents.emplace_back();

        return index;
      }
    }

    [[nodiscard]] auto hasPending() const -> bool {
      return cursor < pending.size();
    }

    [[nodiscard]] auto takePending() -> T { return pending[cursor++]; }

    void store(T entity, const ByteWriter& record) {
      const auto index = indexOfEntity(entity) - 1;
      const auto& bytes = record.view();
      extents[index] = {static_cast<std::uint32_t>(blob.size()),
                        static_cast<std::uint32_t>(bytes.size())};
      blob.insert(blob.end(), bytes.begin(), bytes.end());
    }

    [[nodiscard]] auto indexOfEntity(T entity) const -> std::uint32_t {
      if constexpr (kNumbersItself) return entity->internalId();
      return indexOf.find(static_cast<const void*>(entity));
    }

    void flush(ByteWriter& out) const {
      out.varU32(static_cast<std::uint32_t>(extents.size()));
      for (const auto& extent : extents) {
        out.varU32(extent.size);
        out.append(blob, extent.offset, extent.size);
      }
    }
  };

  struct ConstNode {
    std::shared_ptr<void> owner;
    int kind = 0;
  };

  [[nodiscard]] auto stringRef(std::string_view text) -> StringRef;
  [[nodiscard]] auto locationRef(SourceLocation loc) -> LocationRef;

  [[nodiscard]] auto constNodeRef(std::shared_ptr<void> owner, int kind)
      -> std::uint32_t;

  /**
   * A deterministic ordering key for an unordered container's symbol key, so
   * the archive bytes never depend on hash iteration order (7.5).
   */
  [[nodiscard]] auto entrySortKey(const Symbol* symbol) const
      -> std::pair<unsigned, std::string_view>;

  /**
   * Resolves every referenced location to a file, line and column, leaving the
   * records in ascending ordinal order so that the decoder can binary-search
   * the block as written. Positions are resolved in token order rather than in
   * the order the closure happened to reach them, because computing a column
   * means walking the line: in ordinal order each walk continues where the
   * previous one stopped.
   */
  void resolveLocations();

  void flushStrings(ByteWriter& out) const;
  void flushSourceMap(ByteWriter& out) const;
  [[nodiscard]] auto stringAt(StringRef ref) const -> std::string_view;
  void flushConstants(ByteWriter& out) const;

  void reportError(std::string message) {
    errors_.push_back(std::move(message));
  }

  Domain<const Name*> names_;
  Domain<const Type*> types_;
  Domain<Symbol*> symbols_;
  Domain<AST*> nodes_;

  std::unordered_map<const void*, std::uint32_t> constIndexOf_;
  std::vector<ConstNode> constPending_;
  std::vector<std::vector<std::uint8_t>> constRecords_;
  std::size_t constCursor_ = 0;

  /** Reused for every record, so the buffer is grown once. */
  ByteWriter scratch_;

 private:
  TranslationUnit* unit_ = nullptr;
  struct TransparentStringHash {
    using is_transparent = void;

    auto operator()(std::string_view text) const -> std::size_t {
      return std::hash<std::string_view>{}(text);
    }
  };

  // Heterogeneous lookup: interning a spelling that is already in the table
  // must not construct a `std::string` to throw away.
  std::unordered_map<std::string, std::uint32_t, TransparentStringHash,
                     std::equal_to<>>
      stringIndexOf_;
  std::vector<std::string> strings_;
  struct LocationRecord {
    unsigned ordinal = 0;
    std::uint32_t fileName = 0;
    std::uint32_t presumedFileName = 0;
    std::uint32_t presumedLine = 0;
    std::uint32_t startLine = 0;
    std::uint32_t startColumn = 0;
    std::uint32_t endLine = 0;
    std::uint32_t endColumn = 0;
  };

  std::unordered_set<unsigned> referencedLocations_;
  std::vector<LocationRecord> locations_;
  std::vector<std::string> locationFiles_;
  std::unordered_map<std::string, std::uint32_t> locationFileIndexOf_;
  std::vector<std::string> errors_;
};

class SemanticDecoderBase {
 public:
  explicit SemanticDecoderBase(TranslationUnit* unit) : unit_(unit) {}

  [[nodiscard]] auto unit() const -> TranslationUnit* { return unit_; }

  [[nodiscard]] auto ok() const -> bool { return ok_; }

  [[nodiscard]] auto error() const -> const std::string& { return error_; }

  [[nodiscard]] auto takeSourceMap() -> std::unique_ptr<PrefixSourceMap> {
    return std::move(sourceMap_);
  }

 protected:
  void fail(std::string message) {
    if (ok_) error_ = std::move(message);
    ok_ = false;
  }

  [[nodiscard]] auto readStrings(ByteReader& in) -> bool;
  [[nodiscard]] auto readSourceMap(ByteReader& in) -> bool;

  [[nodiscard]] auto stringAt(StringRef ref) const -> std::string_view;

  /** Each string reference is interned once and remembered. */
  [[nodiscard]] auto identifierAt(StringRef ref) -> const Identifier*;

  [[nodiscard]] auto readStringLiteral(ByteReader& in) -> const StringLiteral*;
  [[nodiscard]] auto readCharLiteral(ByteReader& in) -> const CharLiteral*;
  [[nodiscard]] auto readIntegerLiteral(ByteReader& in)
      -> const IntegerLiteral*;
  [[nodiscard]] auto readFloatLiteral(ByteReader& in) -> const FloatLiteral*;

  /**
   * A location reference is the token ordinal itself: an adopted prefix keeps
   * its tokens at the same indices.
   */
  [[nodiscard]] static auto locationAt(LocationRef ref) -> SourceLocation {
    return SourceLocation{static_cast<unsigned>(ref)};
  }

  [[nodiscard]] auto control() const -> Control*;
  [[nodiscard]] auto arena() const -> Arena*;

  /** A view into the archive the decoder was handed, never a copy. */
  struct Record {
    std::span<const std::uint8_t> bytes;
  };

  [[nodiscard]] static auto readRecords(ByteReader& in,
                                        std::vector<Record>& records) -> bool;

  std::vector<Record> nameRecords_;
  std::vector<Record> typeRecords_;
  std::vector<Record> symbolRecords_;
  std::vector<Record> nodeRecords_;
  std::vector<Record> constRecords_;

  std::vector<const Name*> names_;
  std::vector<const Type*> types_;
  std::vector<Symbol*> symbols_;
  std::vector<AST*> nodes_;
  std::vector<std::shared_ptr<void>> constants_;

  std::vector<bool> nameDecoded_;
  std::vector<bool> typeDecoded_;

 private:
  template <typename T, typename Intern>
  [[nodiscard]] auto internedAt(std::vector<const T*>& cache, StringRef ref,
                                Intern intern) -> const T* {
    const auto index = static_cast<std::uint32_t>(ref);
    if (index == 0 || index > strings_.size()) return nullptr;
    if (cache.size() != strings_.size() + 1)
      cache.assign(strings_.size() + 1, nullptr);
    auto& slot = cache[index];
    if (!slot) slot = intern(control(), strings_[index - 1]);
    return slot;
  }

  TranslationUnit* unit_ = nullptr;
  std::vector<std::string> strings_;
  std::vector<const Identifier*> identifiers_;
  std::vector<const StringLiteral*> stringLiterals_;
  std::vector<const CharLiteral*> charLiterals_;
  std::vector<const IntegerLiteral*> integerLiterals_;
  std::vector<const FloatLiteral*> floatLiterals_;
  std::unique_ptr<PrefixSourceMap> sourceMap_;
  std::string error_;
  bool ok_ = true;
};

[[nodiscard]] auto encodeSemanticGraph(TranslationUnit* unit,
                                       const SemanticArchiveRoots& roots,
                                       ArchiveWriter& archive,
                                       std::vector<std::string>& errors)
    -> bool;

[[nodiscard]] auto decodeSemanticGraph(TranslationUnit* unit,
                                       const ArchiveReader& archive,
                                       SemanticArchiveRoots& roots,
                                       std::string& error) -> bool;

}  // namespace cxx
