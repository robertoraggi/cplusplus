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

#include <cxx/linker/object.h>
#include <cxx/linker/symbols.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace cxx::linker {

namespace {

constexpr std::string_view kMagic{"\0asm", 4};

struct Section {
  std::uint8_t id = 0;
  std::string name;
  ByteView payload;
};

[[nodiscard]] auto hasAddend(std::uint8_t type) -> bool {
  switch (type) {
    case kMemoryAddrLeb:
    case kMemoryAddrSleb:
    case kMemoryAddrI32:
    case kFunctionOffsetI32:
    case kSectionOffsetI32:
    case kMemoryAddrRelSleb:
    case kMemoryAddrLocrelI32:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] auto isLinkerMetadata(const std::string& name) -> bool {
  return name == "linking" || name.starts_with("reloc.") || name == "name" ||
         name == "producers" || name == "target_features";
}

struct SegmentInfo {
  std::string name;
  std::uint32_t alignment = 0;
  std::uint32_t flags = 0;
};

class Parser {
 public:
  Parser(Object& object, std::string& error) : object_(object), error_(error) {}

  [[nodiscard]] auto run() -> bool {
    if (!splitSections()) return false;
    for (const auto& section : sections_) {
      if (!parseSection(section)) return fail("malformed section");
    }
    for (auto& info : object_.symbolInfos) nameImportedSymbol(info);
    return buildChunks();
  }

 private:
  Object& object_;
  std::string& error_;
  std::vector<Section> sections_;
  std::vector<std::vector<Relocation>> relocations_;
  std::vector<SegmentInfo> segmentInfos_;
  std::vector<std::size_t> customSections_;
  bool hasLinking_ = false;

  [[nodiscard]] auto fail(std::string_view message) -> bool {
    error_ = concat(object_.name, ": ", message);
    return false;
  }

  [[nodiscard]] auto splitSections() -> bool {
    Reader reader(object_.content);
    const auto magic = reader.bytes(4);
    const auto version = reader.u32();
    if (!reader.ok() || std::memcmp(magic.data(), kMagic.data(), 4) != 0) {
      return fail("not a wasm object");
    }
    if (version != 1) return fail("unsupported wasm version");
    while (reader.remaining()) {
      Section section;
      section.id = reader.u8();
      section.payload = reader.bytes(reader.uleb());
      if (!reader.ok()) return fail("truncated section");
      if (section.id == kCustomSection) {
        Reader payload(section.payload);
        section.name = payload.string();
        if (!payload.ok()) return fail("truncated section name");
        section.payload = section.payload.subspan(payload.position());
      }
      sections_.push_back(std::move(section));
    }
    relocations_.resize(sections_.size());
    return true;
  }

  [[nodiscard]] auto parseSection(const Section& section) -> bool {
    Reader reader(section.payload);
    switch (section.id) {
      case kTypeSection:
        return parseTypes(reader);
      case kImportSection:
        parseImports(reader);
        break;
      case kFunctionSection:
        parseFunctions(reader);
        break;
      case kGlobalSection:
        return parseGlobals(reader);
      case kTagSection:
        parseTags(reader);
        break;
      case kCustomSection:
        return parseCustom(section, reader);
      default:
        break;
    }
    return reader.ok();
  }

  [[nodiscard]] auto parseTypes(Reader& reader) -> bool {
    for (auto count = reader.uleb(); count && reader.ok(); --count) {
      if (reader.u8() != kFunctionTypeForm)
        return fail("unsupported type form");
      FuncType type;
      for (auto n = reader.uleb(); n && reader.ok(); --n) {
        type.params.push_back(reader.u8());
      }
      for (auto n = reader.uleb(); n && reader.ok(); --n) {
        type.results.push_back(reader.u8());
      }
      object_.types.push_back(std::move(type));
    }
    return reader.ok();
  }

  void skipLimits(Reader& reader) {
    const auto flags = reader.u8();
    static_cast<void>(reader.uleb());
    if (flags & kLimitsHasMax) static_cast<void>(reader.uleb());
  }

  void parseImports(Reader& reader) {
    for (auto count = reader.uleb(); count && reader.ok(); --count) {
      Import import;
      import.module = reader.string();
      import.field = reader.string();
      const auto kind = reader.u8();
      if (kind > kExternalTag) {
        reader.skip(reader.remaining() + 1);
        return;
      }
      if (kind == kExternalFunction) {
        import.typeIndex = reader.uleb();
      } else if (kind == kExternalTable) {
        reader.skip(1);
        skipLimits(reader);
      } else if (kind == kExternalMemory) {
        skipLimits(reader);
      } else if (kind == kExternalGlobal) {
        import.valueType = reader.u8();
        import.isMutable = reader.u8();
      } else {
        reader.skip(1);
        import.typeIndex = reader.uleb();
      }
      object_.imports[kind].push_back(std::move(import));
    }
  }

  void parseFunctions(Reader& reader) {
    for (auto count = reader.uleb(); count && reader.ok(); --count) {
      object_.functionTypes.push_back(reader.uleb());
    }
  }

  [[nodiscard]] auto parseGlobals(Reader& reader) -> bool {
    for (auto count = reader.uleb(); count && reader.ok(); --count) {
      GlobalDefinition global;
      global.valueType = reader.u8();
      global.isMutable = reader.u8();
      global.initOpcode = reader.u8();
      switch (global.initOpcode) {
        case kOpI32Const:
        case kOpI64Const:
          global.init = static_cast<std::uint64_t>(reader.sleb());
          break;
        case kOpF32Const:
          global.init = reader.u32();
          break;
        case kOpF64Const:
          global.init = reader.u32();
          global.init |= std::uint64_t{reader.u32()} << 32;
          break;
        default:
          return fail("unsupported global initializer");
      }
      reader.skip(1);
      object_.globals.push_back(global);
    }
    return reader.ok();
  }

  void parseTags(Reader& reader) {
    for (auto count = reader.uleb(); count && reader.ok(); --count) {
      reader.skip(1);
      object_.tags.push_back(reader.uleb());
    }
  }

  [[nodiscard]] auto parseCustom(const Section& section, Reader& reader)
      -> bool {
    if (section.name == "linking") return parseLinking(reader);
    if (section.name.starts_with("reloc.")) return parseRelocations(reader);
    if (isLinkerMetadata(section.name)) return true;
    customSections_.push_back(
        static_cast<std::size_t>(&section - sections_.data()));
    return true;
  }

  [[nodiscard]] auto parseLinking(Reader& reader) -> bool {
    hasLinking_ = true;
    if (reader.uleb() != 2) return fail("unsupported linking version");
    while (reader.remaining() && reader.ok()) {
      const auto id = reader.u8();
      Reader body(reader.bytes(reader.uleb()));
      if (id == kSymbolTableSubsection) parseSymbolTable(body);
      if (id == kSegmentInfo) parseSegmentInfo(body);
      if (id == kInitFunctions) parseInitFunctions(body);
    }
    return reader.ok();
  }

  void parseSymbolTable(Reader& reader) {
    for (auto count = reader.uleb(); count && reader.ok(); --count) {
      SymbolInfo info;
      info.kind = reader.u8();
      info.flags = reader.uleb();
      const bool undefined = info.flags & kUndefined;
      const bool named = !undefined || (info.flags & kExplicitName);
      if (info.kind == kSymbolData) {
        info.name = reader.string();
        if (!undefined) {
          info.index = reader.uleb();
          info.offset = reader.uleb();
          static_cast<void>(reader.uleb());
        }
      } else if (info.kind == kSymbolSection) {
        info.index = reader.uleb();
      } else {
        info.index = reader.uleb();
        if (named) info.name = reader.string();
      }
      object_.symbolInfos.push_back(std::move(info));
    }
  }

  void nameImportedSymbol(SymbolInfo& info) {
    if (!info.name.empty() || !(info.flags & kUndefined)) return;
    if (info.kind == kSymbolData || info.kind == kSymbolSection) return;
    const auto& imports = object_.imports[kImportKindOfSymbol[info.kind]];
    if (info.index < imports.size()) info.name = imports[info.index].field;
  }

  void parseSegmentInfo(Reader& reader) {
    for (auto count = reader.uleb(); count && reader.ok(); --count) {
      SegmentInfo info;
      info.name = reader.string();
      info.alignment = reader.uleb();
      info.flags = reader.uleb();
      segmentInfos_.push_back(std::move(info));
    }
  }

  void parseInitFunctions(Reader& reader) {
    for (auto count = reader.uleb(); count && reader.ok(); --count) {
      InitFunction init;
      init.priority = reader.uleb();
      init.symbol = reader.uleb();
      object_.initFunctions.push_back(init);
    }
  }

  [[nodiscard]] auto parseRelocations(Reader& reader) -> bool {
    const auto target = reader.uleb();
    if (target >= relocations_.size()) return fail("bad relocation target");
    auto& relocations = relocations_[target];
    for (auto count = reader.uleb(); count && reader.ok(); --count) {
      Relocation relocation;
      relocation.type = reader.u8();
      relocation.offset = reader.uleb();
      relocation.index = reader.uleb();
      if (hasAddend(relocation.type)) {
        relocation.addend = static_cast<std::int32_t>(reader.sleb());
      }
      relocations.push_back(relocation);
    }
    return reader.ok();
  }

  [[nodiscard]] auto buildChunks() -> bool {
    if (!hasLinking_) return fail("missing linking section");
    for (std::size_t i = 0; i < sections_.size(); ++i) {
      std::ranges::stable_sort(relocations_[i], {}, &Relocation::offset);
      if (sections_[i].id == kCodeSection && !buildFunctions(i)) return false;
      if (sections_[i].id == kDataSection && !buildSegments(i)) return false;
    }
    for (const auto index : customSections_) buildSection(index);
    return validate();
  }

  [[nodiscard]] auto validate() -> bool {
    const auto symbolCount = object_.symbolInfos.size();
    for (const auto type : object_.functionTypes) {
      if (type >= object_.types.size()) return fail("bad function type");
    }
    for (const auto type : object_.tags) {
      if (type >= object_.types.size()) return fail("bad tag type");
    }
    for (const auto kind : {kExternalFunction, kExternalTag}) {
      for (const auto& import : object_.imports[kind]) {
        if (import.typeIndex >= object_.types.size()) {
          return fail("bad import type");
        }
      }
    }
    for (const auto& info : object_.symbolInfos) {
      if (!isValid(info)) return fail("bad symbol");
    }
    for (const auto& init : object_.initFunctions) {
      if (init.symbol >= symbolCount) return fail("bad init function");
    }
    return validateRelocations(object_.functions) &&
           validateRelocations(object_.segments) &&
           validateRelocations(object_.sections);
  }

  [[nodiscard]] auto isValid(const SymbolInfo& info) const -> bool {
    if (info.kind > kSymbolTable) return false;
    if (info.kind == kSymbolSection) return true;
    const bool undefined = info.flags & kUndefined;
    if (info.kind == kSymbolData) {
      return undefined || info.index < object_.segments.size();
    }
    const auto& imports = object_.imports[kImportKindOfSymbol[info.kind]];
    if (undefined) return info.index < imports.size();
    if (info.index < imports.size()) return false;
    return info.index - imports.size() < definitionCount(info.kind);
  }

  [[nodiscard]] auto definitionCount(std::uint8_t kind) const -> std::size_t {
    switch (kind) {
      case kSymbolFunction:
        return object_.functions.size();
      case kSymbolGlobal:
        return object_.globals.size();
      case kSymbolTag:
        return object_.tags.size();
      default:
        return std::numeric_limits<std::size_t>::max();
    }
  }

  [[nodiscard]] auto validateRelocations(const std::vector<Chunk>& chunks)
      -> bool {
    for (const auto& chunk : chunks) {
      for (const auto& relocation : chunk.relocations) {
        const auto limit = relocation.type == kTypeIndexLeb
                               ? object_.types.size()
                               : object_.symbolInfos.size();
        if (relocation.index >= limit) return fail("bad relocation");
      }
    }
    return true;
  }

  [[nodiscard]] auto buildFunctions(std::size_t index) -> bool {
    Reader reader(sections_[index].payload);
    const auto count = reader.uleb();
    if (count != object_.functionTypes.size()) return fail("bad code section");
    object_.functions.resize(count);
    RelocationCursor cursor{relocations_[index]};
    for (std::size_t i = 0; i < count; ++i) {
      auto& function = object_.functions[i];
      const auto size = reader.uleb();
      const auto start = reader.position();
      function.content = reader.bytes(size);
      if (!reader.ok()) return fail("truncated code section");
      function.object = &object_;
      function.kind = ChunkKind::Function;
      function.typeIndex = object_.functionTypes[i];
      function.relocations = cursor.take(start, size);
    }
    return true;
  }

  [[nodiscard]] auto buildSegments(std::size_t index) -> bool {
    Reader reader(sections_[index].payload);
    const auto count = reader.uleb();
    object_.segments.resize(count);
    RelocationCursor cursor{relocations_[index]};
    for (std::size_t i = 0; i < count; ++i) {
      auto& segment = object_.segments[i];
      const auto flags = reader.uleb();
      if (flags != 0) return fail("unsupported data segment");
      reader.skip(1);
      static_cast<void>(reader.sleb());
      reader.skip(1);
      const auto size = reader.uleb();
      const auto start = reader.position();
      segment.content = reader.bytes(size);
      if (!reader.ok()) return fail("truncated data section");
      segment.object = &object_;
      segment.kind = ChunkKind::Segment;
      segment.relocations = cursor.take(start, size);
      if (i >= segmentInfos_.size()) continue;
      segment.name = segmentInfos_[i].name;
      segment.alignment = segmentInfos_[i].alignment;
      segment.flags = segmentInfos_[i].flags;
      if (segment.flags & kSegmentTls) return fail("unsupported TLS segment");
    }
    return true;
  }

  void buildSection(std::size_t index) {
    auto& section = object_.sections.emplace_back();
    section.object = &object_;
    section.kind = ChunkKind::Section;
    section.name = sections_[index].name;
    section.content = sections_[index].payload;
    section.relocations = std::move(relocations_[index]);
    section.index = static_cast<std::uint32_t>(index);
  }

  class RelocationCursor {
   public:
    explicit RelocationCursor(const std::vector<Relocation>& relocations)
        : relocations_(relocations) {}

    [[nodiscard]] auto take(std::size_t start, std::size_t size)
        -> std::vector<Relocation> {
      std::vector<Relocation> result;
      for (; next_ < relocations_.size(); ++next_) {
        auto relocation = relocations_[next_];
        if (relocation.offset >= start + size) break;
        relocation.offset -= static_cast<std::uint32_t>(start);
        result.push_back(relocation);
      }
      return result;
    }

   private:
    const std::vector<Relocation>& relocations_;
    std::size_t next_ = 0;
  };
};

}  // namespace

Object::Object(std::string name, ByteView content)
    : name(std::move(name)), content(content) {}

Object::~Object() = default;

auto Object::parse(std::string& error) -> bool {
  return Parser{*this, error}.run();
}

}  // namespace cxx::linker
