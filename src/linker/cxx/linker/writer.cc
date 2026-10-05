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

#include <cxx/linker/relocate.h>
#include <cxx/linker/writer.h>

#include <algorithm>
#include <cstring>

namespace cxx::linker {

namespace {

constexpr std::uint8_t kModuleHeader[] = {0x00, 0x61, 0x73, 0x6d,
                                          0x01, 0x00, 0x00, 0x00};
constexpr std::uint8_t kFunctionNames = 1;
constexpr std::size_t kF32Size = 4;
constexpr std::size_t kF64Size = 8;

void emitInit(Buffer& out, const GlobalDefinition& global) {
  out.u8(global.initOpcode);
  if (global.initOpcode == kOpI32Const) {
    out.sleb(static_cast<std::int32_t>(global.init));
  } else if (global.initOpcode == kOpI64Const) {
    out.sleb(static_cast<std::int64_t>(global.init));
  } else {
    out.fixed(global.init,
              global.initOpcode == kOpF32Const ? kF32Size : kF64Size);
  }
  out.u8(kOpEnd);
}

void emitConstant(Buffer& out, std::uint32_t value) {
  GlobalDefinition constant;
  constant.init = value;
  emitInit(out, constant);
}

class Writer {
 public:
  explicit Writer(Layout& layout) : layout_(layout) {}

  [[nodiscard]] auto run(Bytes& output, std::string& error) -> bool {
    Buffer code;
    Buffer data;
    std::vector<Buffer> customs;
    if (!buildCode(code, error) || !buildData(data, error)) return false;
    if (!buildCustoms(customs, error)) return false;
    const bool hasImports = !layout_.functionImports.empty() ||
                            !layout_.globalImports.empty() ||
                            !layout_.tagImports.empty();
    Buffer imports = importSection();
    Buffer functions = functionSection();
    Buffer module;
    module.bytes(kModuleHeader);
    module.section(kTypeSection, typeSection());
    if (hasImports) module.section(kImportSection, imports);
    if (!layout_.functions.empty()) module.section(kFunctionSection, functions);
    if (layout_.hasTable) module.section(kTableSection, tableSection());
    module.section(kMemorySection, memorySection());
    if (!layout_.tags.empty()) module.section(kTagSection, tagSection());
    if (!layout_.globals.empty() || !layout_.gotEntries.empty()) {
      module.section(kGlobalSection, globalSection());
    }
    module.section(kExportSection, exportSection());
    if (!layout_.tableEntries.empty()) {
      module.section(kElementSection, elementSection());
    }
    if (!layout_.functions.empty()) module.section(kCodeSection, code);
    if (data.size()) module.section(kDataSection, data);
    for (std::size_t i = 0; i < customs.size(); ++i) {
      module.section(kCustomSection, customs[i]);
    }
    if (layout_.hasNames) module.section(kCustomSection, nameSection());
    output.assign(module.view().begin(), module.view().end());
    return true;
  }

 private:
  Layout& layout_;

  [[nodiscard]] auto buildCode(Buffer& code, std::string& error) -> bool {
    code.uleb(static_cast<std::uint32_t>(layout_.functions.size()));
    for (const Chunk* function : layout_.functions) {
      code.uleb(static_cast<std::uint32_t>(function->content.size()));
      const auto start = code.size();
      code.bytes(function->content);
      if (!relocate(layout_, *function, code.data() + start, error)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] auto buildData(Buffer& data, std::string& error) -> bool {
    std::uint32_t end = layout_.dataStart;
    for (const Chunk* chunk : layout_.dataChunks) {
      end = std::max(end, chunk->offset + static_cast<std::uint32_t>(
                                              chunk->content.size()));
    }
    if (layout_.dataChunks.empty()) return true;
    Buffer image;
    image.zeros(end - layout_.dataStart);
    for (const Chunk* chunk : layout_.dataChunks) {
      auto* at = image.data() + (chunk->offset - layout_.dataStart);
      std::memcpy(at, chunk->content.data(), chunk->content.size());
      if (!relocate(layout_, *chunk, at, error)) return false;
    }
    data.uleb(1);
    data.uleb(0);
    data.u8(kOpI32Const);
    data.sleb(static_cast<std::int32_t>(layout_.dataStart));
    data.u8(kOpEnd);
    data.uleb(static_cast<std::uint32_t>(image.size()));
    data.bytes(image.view());
    return true;
  }

  [[nodiscard]] auto buildCustoms(std::vector<Buffer>& customs,
                                  std::string& error) -> bool {
    for (const auto& section : layout_.customSections) {
      Buffer payload;
      payload.string(section.name);
      for (const Chunk* chunk : section.chunks) {
        const auto start = payload.size();
        payload.bytes(chunk->content);
        if (!relocate(layout_, *chunk, payload.data() + start, error)) {
          return false;
        }
      }
      customs.push_back(std::move(payload));
    }
    return true;
  }

  [[nodiscard]] auto typeSection() -> Buffer {
    Buffer out;
    out.uleb(static_cast<std::uint32_t>(layout_.types.size()));
    for (const auto& type : layout_.types) {
      out.u8(kFunctionTypeForm);
      out.uleb(static_cast<std::uint32_t>(type.params.size()));
      for (const auto param : type.params) out.u8(param);
      out.uleb(static_cast<std::uint32_t>(type.results.size()));
      for (const auto result : type.results) out.u8(result);
    }
    return out;
  }

  [[nodiscard]] auto importSection() -> Buffer {
    Buffer out;
    out.uleb(static_cast<std::uint32_t>(layout_.functionImports.size() +
                                        layout_.globalImports.size() +
                                        layout_.tagImports.size()));
    for (const Symbol* symbol : layout_.functionImports) {
      importHeader(out, *symbol, kExternalFunction);
      out.uleb(layout_.typeIndex(symbol->type));
    }
    for (const Symbol* symbol : layout_.globalImports) {
      importHeader(out, *symbol, kExternalGlobal);
      out.u8(symbol->global.valueType);
      out.u8(symbol->global.isMutable);
    }
    for (const Symbol* symbol : layout_.tagImports) {
      importHeader(out, *symbol, kExternalTag);
      out.u8(0);
      out.uleb(layout_.typeIndex(symbol->type));
    }
    return out;
  }

  static void importHeader(Buffer& out, const Symbol& symbol,
                           std::uint8_t kind) {
    out.string(symbol.importModule);
    out.string(symbol.importField);
    out.u8(kind);
  }

  [[nodiscard]] auto functionSection() -> Buffer {
    Buffer out;
    out.uleb(static_cast<std::uint32_t>(layout_.functions.size()));
    for (const Chunk* function : layout_.functions) {
      out.uleb(layout_.typeIndex(*function->object, function->typeIndex));
    }
    return out;
  }

  [[nodiscard]] auto tableSection() const -> Buffer {
    Buffer out;
    const auto size = static_cast<std::uint32_t>(layout_.tableEntries.size()) +
                      layout_.tableBase;
    out.uleb(1);
    out.u8(kFuncRef);
    out.u8(kLimitsHasMax);
    out.uleb(size);
    out.uleb(size);
    return out;
  }

  [[nodiscard]] auto memorySection() const -> Buffer {
    Buffer out;
    out.uleb(1);
    out.u8(layout_.maxMemoryPages ? kLimitsHasMax : 0);
    out.uleb(layout_.memoryPages);
    if (layout_.maxMemoryPages) out.uleb(layout_.maxMemoryPages);
    return out;
  }

  [[nodiscard]] auto tagSection() -> Buffer {
    Buffer out;
    out.uleb(static_cast<std::uint32_t>(layout_.tags.size()));
    for (const Symbol* symbol : layout_.tags) {
      out.u8(0);
      out.uleb(layout_.typeIndex(symbol->type));
    }
    return out;
  }

  [[nodiscard]] auto globalSection() const -> Buffer {
    Buffer out;
    out.uleb(static_cast<std::uint32_t>(layout_.globals.size() +
                                        layout_.gotEntries.size()));
    for (const Symbol* symbol : layout_.globals) {
      out.u8(symbol->global.valueType);
      out.u8(symbol->global.isMutable);
      emitInit(out, symbol->global);
    }
    for (const Symbol* symbol : layout_.gotEntries) {
      out.u8(kI32);
      out.u8(0);
      emitConstant(out, gotValue(*symbol));
    }
    return out;
  }

  [[nodiscard]] static auto gotValue(const Symbol& symbol) -> std::uint32_t {
    if (symbol.kind == SymbolKind::Function)
      return symbol.tableIndex.value_or(0);
    return symbol.defined ? symbol.address() : 0;
  }

  [[nodiscard]] auto exportSection() const -> Buffer {
    Buffer out;
    out.uleb(static_cast<std::uint32_t>(layout_.exports.size()) + 1);
    out.string("memory");
    out.u8(kExternalMemory);
    out.uleb(0);
    for (const Symbol* symbol : layout_.exports) {
      const bool isFunction = symbol->kind == SymbolKind::Function;
      out.string(symbol->name);
      out.u8(isFunction ? kExternalFunction : kExternalGlobal);
      out.uleb(isFunction ? symbol->outputIndex() : symbol->index);
    }
    return out;
  }

  [[nodiscard]] auto elementSection() const -> Buffer {
    Buffer out;
    out.uleb(1);
    out.uleb(0);
    out.u8(kOpI32Const);
    out.sleb(layout_.tableBase);
    out.u8(kOpEnd);
    out.uleb(static_cast<std::uint32_t>(layout_.tableEntries.size()));
    for (const Symbol* symbol : layout_.tableEntries) {
      out.uleb(symbol->outputIndex());
    }
    return out;
  }

  [[nodiscard]] auto nameSection() const -> Buffer {
    Buffer names;
    names.uleb(static_cast<std::uint32_t>(layout_.functionImports.size() +
                                          layout_.functions.size()));
    std::uint32_t index = 0;
    for (const Symbol* symbol : layout_.functionImports) {
      names.uleb(index++);
      names.string(symbol->name);
    }
    for (const Chunk* function : layout_.functions) {
      names.uleb(index++);
      names.string(function->name);
    }
    Buffer out;
    out.string("name");
    out.u8(kFunctionNames);
    out.uleb(static_cast<std::uint32_t>(names.size()));
    out.bytes(names.view());
    return out;
  }
};

}  // namespace

auto writeModule(Layout& layout, Bytes& output, std::string& error) -> bool {
  return Writer{layout}.run(output, error);
}

}  // namespace cxx::linker
