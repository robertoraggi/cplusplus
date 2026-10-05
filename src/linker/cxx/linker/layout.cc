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

#include <cxx/linker/layout.h>

#include <algorithm>
#include <limits>

namespace cxx::linker {

namespace {

enum class Synthetic {
  StackPointer,
  MemoryBase,
  TableBase,
  TlsBase,
  DsoHandle,
  DataEnd,
  GlobalBase,
  HeapBase,
  HeapEnd,
  FirstPageEnd,
};

struct SyntheticDescription {
  std::string_view name;
  SymbolKind kind;
  Synthetic which;
};

constexpr SyntheticDescription kSynthetics[] = {
    {"__stack_pointer", SymbolKind::Global, Synthetic::StackPointer},
    {"__memory_base", SymbolKind::Global, Synthetic::MemoryBase},
    {"__table_base", SymbolKind::Global, Synthetic::TableBase},
    {"__tls_base", SymbolKind::Global, Synthetic::TlsBase},
    {"__dso_handle", SymbolKind::Data, Synthetic::DsoHandle},
    {"__data_end", SymbolKind::Data, Synthetic::DataEnd},
    {"__global_base", SymbolKind::Data, Synthetic::GlobalBase},
    {"__heap_base", SymbolKind::Data, Synthetic::HeapBase},
    {"__heap_end", SymbolKind::Data, Synthetic::HeapEnd},
    {"__wasm_first_page_end", SymbolKind::Data, Synthetic::FirstPageEnd},
};

struct SyntheticSymbol {
  Symbol* symbol;
  Synthetic which;
};

struct MemoryMap {
  std::uint32_t stackTop = 0;
  std::uint32_t dataStart = 0;
  std::uint32_t dataEnd = 0;
  std::uint32_t heapBase = 0;
  std::uint32_t memoryEnd = 0;
};

constexpr std::uint32_t kStackAlignment = 16;
constexpr std::uint32_t kDefaultGlobalBase = 1024;

[[nodiscard]] constexpr auto alignTo(std::uint32_t value,
                                     std::uint32_t alignment) -> std::uint32_t {
  return (value + alignment - 1) & ~(alignment - 1);
}

[[nodiscard]] auto stackTopFrom(std::uint32_t address, std::uint32_t size)
    -> std::uint32_t {
  return alignTo(address, kStackAlignment) + size;
}

[[nodiscard]] auto valueOf(Synthetic which, const MemoryMap& memory)
    -> std::uint32_t {
  switch (which) {
    case Synthetic::StackPointer:
      return memory.stackTop;
    case Synthetic::MemoryBase:
      return 0;
    case Synthetic::TableBase:
      return 1;
    case Synthetic::TlsBase:
    case Synthetic::DsoHandle:
    case Synthetic::GlobalBase:
      return memory.dataStart;
    case Synthetic::DataEnd:
      return memory.dataEnd;
    case Synthetic::HeapBase:
      return memory.heapBase;
    case Synthetic::HeapEnd:
      return memory.memoryEnd;
    case Synthetic::FirstPageEnd:
      return kPageSize;
  }
  return 0;
}

class Layouter {
 public:
  Layouter(SymbolTable& symbols, std::vector<std::unique_ptr<Object>>& objects,
           const LayoutOptions& options, Layout& layout)
      : symbols_(symbols), options_(options), layout_(layout) {
    for (auto& object : objects) objects_.push_back(object.get());
    objects_.push_back(&layout.synthetic);
  }

  [[nodiscard]] auto run(std::string& error) -> bool {
    defineSynthetics();
    if (!markRoots(error)) return false;
    drain();
    if (!resolveUndefined(error)) return false;
    assignFunctions();
    assignGlobals();
    assignTags();
    if (!assignMemory(error)) return false;
    assignCustomSections();
    return true;
  }

 private:
  SymbolTable& symbols_;
  const LayoutOptions& options_;
  Layout& layout_;
  std::vector<Object*> objects_;
  std::vector<SyntheticSymbol> synthetics_;
  std::vector<Symbol*> undefined_;
  std::vector<Chunk*> work_;
  Chunk* stub_ = nullptr;

  void defineSynthetics() {
    for (const auto& description : kSynthetics) {
      auto* symbol =
          symbols_.defineSynthetic(description.name, description.kind);
      if (!symbol) continue;
      symbol->global.isMutable = description.which == Synthetic::StackPointer;
      synthetics_.push_back({symbol, description.which});
    }
    if (auto* ctors = symbols_.defineSynthetic("__wasm_call_ctors",
                                               SymbolKind::Function)) {
      defineCallCtors(*ctors);
    }
  }

  [[nodiscard]] auto addSyntheticFunction(Bytes body) -> Chunk* {
    auto& stored = layout_.syntheticBodies.emplace_back(std::move(body));
    auto& chunk = layout_.synthetic.functions.emplace_back();
    chunk.object = &layout_.synthetic;
    chunk.content = stored;
    return &chunk;
  }

  void defineCallCtors(Symbol& symbol) {
    std::vector<std::pair<std::uint32_t, Symbol*>> ctors;
    for (Object* object : objects_) {
      for (const auto& init : object->initFunctions) {
        ctors.emplace_back(init.priority, object->symbols[init.symbol]);
      }
    }
    std::ranges::stable_sort(ctors, {}, &decltype(ctors)::value_type::first);
    Bytes body{0};
    std::vector<Relocation> relocations;
    for (const auto& [priority, target] : ctors) {
      Relocation call;
      call.type = kFunctionIndexLeb;
      call.offset = static_cast<std::uint32_t>(body.size() + 1);
      call.index = static_cast<std::uint32_t>(layout_.synthetic.symbols.size());
      relocations.push_back(call);
      layout_.synthetic.symbols.push_back(target);
      body.push_back(kOpCall);
      body.resize(body.size() + kPaddedLebSize);
    }
    body.push_back(kOpEnd);
    auto* chunk = addSyntheticFunction(std::move(body));
    chunk->relocations = std::move(relocations);
    chunk->name = symbol.name;
    symbol.chunk = chunk;
  }

  [[nodiscard]] auto stub() -> Chunk* {
    if (!stub_) {
      stub_ = addSyntheticFunction({0, kOpUnreachable, kOpEnd});
      stub_->name = "__wasm_undefined_function";
    }
    return stub_;
  }

  void addExport(Symbol* symbol) {
    if (std::ranges::find(layout_.exports, symbol) == layout_.exports.end()) {
      layout_.exports.push_back(symbol);
    }
  }

  [[nodiscard]] auto exportable(const Symbol& symbol) const -> bool {
    return symbol.kind == SymbolKind::Function ||
           symbol.kind == SymbolKind::Global;
  }

  [[nodiscard]] auto rootNamed(const std::string& name, std::string_view what,
                               std::string& error) -> bool {
    auto* symbol = symbols_.find(name);
    if (symbol && symbol->defined) {
      reach(symbol);
      addExport(symbol);
      return true;
    }
    error = concat(what, " symbol not defined: ", name);
    return false;
  }

  [[nodiscard]] auto markRoots(std::string& error) -> bool {
    if (!options_.noEntry && !rootNamed(options_.entry, "entry", error)) {
      return false;
    }
    for (const auto& name : options_.exports) {
      if (!rootNamed(name, "exported", error)) return false;
    }
    for (auto* symbol : symbols_.symbols()) {
      if (!symbol->defined) continue;
      if (symbol->exported && exportable(*symbol)) addExport(symbol);
      if (symbol->exported || symbol->noStrip) reach(symbol);
    }
    if (options_.gcSections) return true;
    for (Object* object : objects_) {
      for (auto* symbol : object->symbols) {
        if (symbol->defined) reach(symbol);
      }
    }
    return true;
  }

  void reach(Chunk* chunk) {
    if (chunk->live) return;
    chunk->live = true;
    work_.push_back(chunk);
  }

  void reach(Symbol* symbol) {
    if (symbol->live) return;
    symbol->live = true;
    if (symbol->kind == SymbolKind::Section) return;
    if (symbol->chunk) {
      reach(symbol->chunk);
    } else if (!symbol->defined) {
      undefined_.push_back(symbol);
    } else if (symbol->kind == SymbolKind::Global) {
      layout_.globals.push_back(symbol);
    } else if (symbol->kind == SymbolKind::Tag) {
      layout_.tags.push_back(symbol);
    }
  }

  void drain() {
    while (!work_.empty()) {
      const Chunk* chunk = work_.back();
      work_.pop_back();
      for (const auto& relocation : chunk->relocations) {
        scan(*chunk, relocation);
      }
    }
  }

  void scan(const Chunk& chunk, const Relocation& relocation) {
    if (relocation.type == kTypeIndexLeb) return;
    Symbol* symbol = chunk.object->symbols[relocation.index];
    switch (relocation.type) {
      case kTableIndexSleb:
      case kTableIndexI32:
      case kTableIndexRelSleb:
        addTableEntry(symbol);
        break;
      case kGlobalIndexLeb:
      case kGlobalIndexI32:
        addGotEntry(symbol);
        break;
      case kTableNumberLeb:
        layout_.hasTable = true;
        break;
      default:
        break;
    }
    reach(symbol);
  }

  [[nodiscard]] static auto isWeakUndefined(const Symbol& symbol) -> bool {
    return !symbol.defined && symbol.weak && !symbol.imported;
  }

  void addTableEntry(Symbol* symbol) {
    layout_.hasTable = true;
    if (symbol->kind != SymbolKind::Function || symbol->tableIndex) return;
    if (isWeakUndefined(*symbol)) return;
    symbol->tableIndex =
        static_cast<std::uint32_t>(layout_.tableEntries.size()) +
        layout_.tableBase;
    layout_.tableEntries.push_back(symbol);
  }

  void addGotEntry(Symbol* symbol) {
    const bool isAddress = symbol->kind == SymbolKind::Data ||
                           symbol->kind == SymbolKind::Function;
    if (!isAddress || symbol->gotIndex) return;
    symbol->gotIndex = 0;
    layout_.gotEntries.push_back(symbol);
    addTableEntry(symbol);
  }

  [[nodiscard]] auto resolveUndefined(std::string& error) -> bool {
    std::string message;
    for (Symbol* symbol : undefined_) {
      if (!importUndefined(*symbol)) {
        message += concat(message.empty() ? "" : "\n",
                          "undefined symbol: ", symbol->name);
      }
    }
    error = std::move(message);
    return error.empty();
  }

  [[nodiscard]] auto importUndefined(Symbol& symbol) -> bool {
    if (symbol.kind == SymbolKind::Table ||
        symbol.kind == SymbolKind::Section) {
      return true;
    }
    if (!symbol.imported && options_.allowUndefined) {
      symbol.imported = true;
      symbol.importModule = "env";
      symbol.importField = symbol.name;
    }
    if (!symbol.imported) return resolveWeak(symbol);
    switch (symbol.kind) {
      case SymbolKind::Function:
        layout_.functionImports.push_back(&symbol);
        break;
      case SymbolKind::Global:
        layout_.globalImports.push_back(&symbol);
        break;
      case SymbolKind::Tag:
        layout_.tagImports.push_back(&symbol);
        break;
      default:
        break;
    }
    return true;
  }

  [[nodiscard]] auto resolveWeak(Symbol& symbol) -> bool {
    if (!symbol.weak) return false;
    if (symbol.kind != SymbolKind::Function) return true;
    symbol.chunk = stub();
    reach(symbol.chunk);
    drain();
    return true;
  }

  void assignFunctions() {
    std::uint32_t index = 0;
    for (auto* symbol : layout_.functionImports) symbol->index = index++;
    for (Object* object : objects_) {
      for (auto& function : object->functions) {
        if (!function.live) continue;
        function.index = index++;
        layout_.functions.push_back(&function);
      }
    }
    auto offset = static_cast<std::uint32_t>(
        ulebSize(static_cast<std::uint32_t>(layout_.functions.size())));
    for (auto* function : layout_.functions) {
      const auto size = static_cast<std::uint32_t>(function->content.size());
      offset += static_cast<std::uint32_t>(ulebSize(size));
      function->offset = offset;
      offset += size;
    }
  }

  void assignGlobals() {
    std::uint32_t index = 0;
    for (auto* symbol : layout_.globalImports) symbol->index = index++;
    for (auto* symbol : layout_.globals) symbol->index = index++;
    for (auto* symbol : layout_.gotEntries) symbol->gotIndex = index++;
  }

  void assignTags() {
    std::uint32_t index = 0;
    for (auto* symbol : layout_.tagImports) symbol->index = index++;
    for (auto* symbol : layout_.tags) symbol->index = index++;
  }

  static void place(const std::vector<Chunk*>& chunks, std::uint32_t& address) {
    for (auto* chunk : chunks) {
      address = alignTo(address, 1u << chunk->alignment);
      chunk->offset = address;
      address += static_cast<std::uint32_t>(chunk->content.size());
    }
  }

  [[nodiscard]] auto assignMemory(std::string& error) -> bool {
    std::vector<Chunk*> readOnly;
    std::vector<Chunk*> data;
    std::vector<Chunk*> zero;
    for (Object* object : objects_) {
      for (auto& segment : object->segments) {
        if (!segment.live) continue;
        if (segment.name.starts_with(".bss")) {
          zero.push_back(&segment);
        } else if (segment.name.starts_with(".rodata")) {
          readOnly.push_back(&segment);
        } else {
          data.push_back(&segment);
        }
      }
    }
    MemoryMap memory;
    std::uint32_t address = 0;
    if (options_.stackFirst) {
      address = memory.stackTop = stackTopFrom(address, options_.stackSize);
    }
    if (options_.globalBase && options_.globalBase < address) {
      error =
          "--global-base cannot be less than stack size when --stack-first "
          "is used";
      return false;
    }
    if (options_.globalBase) {
      address = options_.globalBase;
    } else if (!options_.stackFirst) {
      address = kDefaultGlobalBase;
    }
    memory.dataStart = address;
    place(readOnly, address);
    place(data, address);
    layout_.dataChunks = std::move(readOnly);
    layout_.dataChunks.insert(layout_.dataChunks.end(), data.begin(),
                              data.end());
    place(zero, address);
    memory.dataEnd = address;
    if (!options_.stackFirst) {
      address = memory.stackTop = stackTopFrom(address, options_.stackSize);
    }
    memory.heapBase = address = alignTo(address, kStackAlignment);
    memory.memoryEnd =
        alignTo(std::max(address, options_.initialMemory), kPageSize);
    layout_.dataStart = memory.dataStart;
    layout_.memoryPages = memory.memoryEnd / kPageSize;
    layout_.maxMemoryPages = alignTo(options_.maxMemory, kPageSize) / kPageSize;
    for (const auto& [symbol, which] : synthetics_) {
      const auto value = valueOf(which, memory);
      if (symbol->kind == SymbolKind::Global) {
        symbol->global.init = value;
      } else {
        symbol->offset = value;
      }
    }
    return true;
  }

  [[nodiscard]] auto isStripped(const Chunk& section) const -> bool {
    if (options_.stripAll) return true;
    return options_.stripDebug && section.name.starts_with(".debug_");
  }

  void assignCustomSections() {
    layout_.hasNames = !options_.stripAll;
    for (Object* object : objects_) {
      for (auto& section : object->sections) {
        if (isStripped(section)) continue;
        section.live = true;
        auto it = std::ranges::find(layout_.customSections, section.name,
                                    &OutputSection::name);
        if (it == layout_.customSections.end()) {
          it = layout_.customSections.insert(it, {section.name, {}, 0});
        }
        section.offset = it->size;
        it->size += static_cast<std::uint32_t>(section.content.size());
        it->chunks.push_back(&section);
      }
    }
  }
};

}  // namespace

Layout::Layout() : synthetic("<linker>", {}) {
  synthetic.types.emplace_back();
  synthetic.functions.reserve(2);
}

auto Layout::typeIndex(const FuncType& type) -> std::uint32_t {
  const auto it = std::ranges::find(types, type);
  if (it != types.end()) return static_cast<std::uint32_t>(it - types.begin());
  types.push_back(type);
  return static_cast<std::uint32_t>(types.size() - 1);
}

auto Layout::typeIndex(Object& object, std::uint32_t index) -> std::uint32_t {
  constexpr auto kUnassigned = std::numeric_limits<std::uint32_t>::max();
  if (object.typeMap.size() != object.types.size()) {
    object.typeMap.assign(object.types.size(), kUnassigned);
  }
  auto& mapped = object.typeMap[index];
  if (mapped == kUnassigned) mapped = typeIndex(object.types[index]);
  return mapped;
}

auto buildLayout(SymbolTable& symbols,
                 std::vector<std::unique_ptr<Object>>& objects,
                 const LayoutOptions& options, Layout& layout,
                 std::string& error) -> bool {
  return Layouter{symbols, objects, options, layout}.run(error);
}

}  // namespace cxx::linker
