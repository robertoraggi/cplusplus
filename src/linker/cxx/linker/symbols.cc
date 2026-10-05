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

#include <cxx/linker/symbols.h>

#include <utility>

namespace cxx::linker {

namespace {

constexpr SymbolKind kKinds[] = {
    SymbolKind::Function, SymbolKind::Data, SymbolKind::Global,
    SymbolKind::Section,  SymbolKind::Tag,  SymbolKind::Table,
};

[[nodiscard]] auto importOf(Object& object, const SymbolInfo& info)
    -> const Import* {
  const auto& imports = object.imports[kImportKindOfSymbol[info.kind]];
  return info.index < imports.size() ? &imports[info.index] : nullptr;
}

void describeUndefined(Symbol& symbol, Object& object, const SymbolInfo& info) {
  if (info.kind == kSymbolData || info.kind == kSymbolSection) return;
  const auto* import = importOf(object, info);
  if (!import) return;
  symbol.importModule = import->module;
  symbol.importField = import->field;
  if (symbol.kind == SymbolKind::Function || symbol.kind == SymbolKind::Tag) {
    symbol.type = object.types[import->typeIndex];
  }
  if (symbol.kind == SymbolKind::Global) {
    symbol.global.valueType = import->valueType;
    symbol.global.isMutable = import->isMutable;
  }
}

void describeFunction(Symbol& symbol, Object& object, const SymbolInfo& info) {
  symbol.chunk = &object.functions[info.index - object.functionImportCount()];
  symbol.type = object.types[symbol.chunk->typeIndex];
  if (symbol.chunk->name.empty()) symbol.chunk->name = info.name;
}

void describeDefined(Symbol& symbol, Object& object, const SymbolInfo& info) {
  const auto locals = [&](ExternalKind kind) {
    return info.index - object.imports[kind].size();
  };
  switch (symbol.kind) {
    case SymbolKind::Function:
      describeFunction(symbol, object, info);
      break;
    case SymbolKind::Data:
      symbol.chunk = &object.segments[info.index];
      symbol.offset = info.offset;
      break;
    case SymbolKind::Global:
      symbol.global = object.globals[locals(kExternalGlobal)];
      break;
    case SymbolKind::Tag:
      symbol.type = object.types[object.tags[locals(kExternalTag)]];
      break;
    case SymbolKind::Section:
      for (auto& section : object.sections) {
        if (section.index == info.index) symbol.chunk = &section;
      }
      break;
    case SymbolKind::Table:
      break;
  }
}

[[nodiscard]] auto describe(Object& object, const SymbolInfo& info) -> Symbol {
  Symbol symbol;
  symbol.name = info.name;
  symbol.kind = kKinds[info.kind];
  symbol.defined = !(info.flags & kUndefined);
  symbol.weak = info.flags & kBindingWeak;
  symbol.imported = !symbol.defined && (info.flags & kExplicitName);
  symbol.exported = info.flags & kExported;
  symbol.noStrip = info.flags & kNoStrip;
  symbol.object = &object;
  if (symbol.defined) {
    describeDefined(symbol, object, info);
  } else {
    describeUndefined(symbol, object, info);
  }
  return symbol;
}

}  // namespace

auto SymbolTable::find(std::string_view name) const -> Symbol* {
  const auto it = table_.find(std::string{name});
  return it == table_.end() ? nullptr : it->second.get();
}

auto SymbolTable::addGlobal(Object& object, const SymbolInfo& info,
                            std::string& error) -> Symbol* {
  auto candidate = describe(object, info);
  auto& slot = table_[info.name];
  if (!slot) {
    slot = std::make_unique<Symbol>(std::move(candidate));
    order_.push_back(slot.get());
    if (!slot->defined && !slot->weak) pending_.push_back(slot.get());
    return slot.get();
  }
  Symbol& existing = *slot;
  existing.exported |= candidate.exported;
  existing.noStrip |= candidate.noStrip;
  if (!candidate.defined) {
    if (!existing.defined && existing.weak && !candidate.weak) {
      existing.weak = false;
      pending_.push_back(&existing);
    }
    return &existing;
  }
  if (existing.defined && !existing.weak && !candidate.weak) {
    error = concat("duplicate symbol: ", info.name, " (defined in ",
                   existing.object->name, " and ", object.name, ")");
    return nullptr;
  }
  if (existing.defined && !(existing.weak && !candidate.weak)) return &existing;
  candidate.exported = existing.exported;
  candidate.noStrip = existing.noStrip;
  existing = std::move(candidate);
  return &existing;
}

auto SymbolTable::add(Object& object, std::string& error) -> bool {
  object.symbols.reserve(object.symbolInfos.size());
  for (const auto& info : object.symbolInfos) {
    if (!(info.flags & kBindingLocal)) {
      auto* symbol = addGlobal(object, info, error);
      if (!symbol) return false;
      object.symbols.push_back(symbol);
      continue;
    }
    auto local = std::make_unique<Symbol>(describe(object, info));
    object.symbols.push_back(local.get());
    object.localSymbols.push_back(std::move(local));
  }
  return true;
}

auto SymbolTable::defineSynthetic(std::string_view name, SymbolKind kind)
    -> Symbol* {
  auto* symbol = find(name);
  if (!symbol || symbol->defined) return nullptr;
  symbol->kind = kind;
  symbol->defined = true;
  symbol->weak = false;
  symbol->imported = false;
  return symbol;
}

auto SymbolTable::takePending() -> std::vector<Symbol*> {
  return std::exchange(pending_, {});
}

}  // namespace cxx::linker
