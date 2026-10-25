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

#include <cxx/cxxir/cxxir_emitter.h>
#include <cxx/translation_unit.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <format>
#include <iterator>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cxx::ir {
namespace {

[[nodiscard]] auto quote(std::string_view text) -> std::string {
  std::string result = "\"";
  for (unsigned char ch : text) {
    if (ch >= 32 && ch < 127 && ch != '"' && ch != '\\')
      result += static_cast<char>(ch);
    else
      result += std::format("\\{:02X}", ch);
  }
  return result + '"';
}

template <typename Range, typename Format>
[[nodiscard]] auto join(const Range& range, Format format) -> std::string {
  std::string text;
  for (const auto& element : range) {
    if (!text.empty()) text += ", ";
    text += format(element);
  }
  return text;
}

[[nodiscard]] auto linkageAttr(Linkage linkage) -> std::string {
  static constexpr std::string_view names[] = {"external",
                                               "internal",
                                               "linkonce_odr",
                                               "weak_odr",
                                               "available_externally",
                                               "appending",
                                               "weak",
                                               "extern_weak"};
  return std::format("#cxx.linkage_kind<{}>", names[static_cast<int>(linkage)]);
}

[[nodiscard]] auto orderAttr(MemoryOrder order) -> std::string {
  static constexpr std::string_view names[] = {"relaxed", "acquire", "release",
                                               "acq_rel", "seq_cst"};
  return std::format("#cxx.memory_order<{}>", names[static_cast<int>(order)]);
}

[[nodiscard]] auto scopeAttr(MemoryScope scope) -> std::string {
  static constexpr std::string_view names[] = {"system", "device", "workgroup",
                                               "subgroup", "invocation"};
  return std::format("#cxx.memory_scope<{}>", names[static_cast<int>(scope)]);
}

[[nodiscard]] auto regionsAttr(MemoryRegions regions) -> std::string {
  if (regions.storage && regions.workgroup)
    return "#cxx.memory_regions<storage, workgroup>";
  if (regions.storage) return "#cxx.memory_regions<storage>";
  auto name = regions.workgroup ? "workgroup" : "none";
  return std::format("#cxx.memory_regions<{}>", name);
}

template <typename Range>
[[nodiscard]] auto denseArray(std::string_view type, const Range& values)
    -> std::string {
  auto text = std::format("array<{}", type);
  if (!values.empty())
    text +=
        ": " + join(values, [](auto value) { return std::to_string(value); });
  return text + '>';
}

void addAttr(std::string& attrs, std::string_view name, std::string value) {
  if (!attrs.empty()) attrs += ", ";
  attrs += std::format("{} = {}", quote(name), value);
}

void addStringAttr(std::string& attrs, std::string_view name,
                   std::string_view value) {
  if (!value.empty()) addAttr(attrs, name, quote(value));
}

[[nodiscard]] auto trim(std::string_view text) -> std::string_view {
  auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return {};
  return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

[[nodiscard]] auto splitList(std::string_view text)
    -> std::vector<std::string_view> {
  std::vector<std::string_view> values;
  int depth = 0;
  bool quoted = false;
  std::size_t start = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    auto ch = text[i];
    if (quoted && ch == '\\') {
      ++i;
      continue;
    }
    if (ch == '"') quoted = !quoted;
    if (quoted) continue;
    if (ch == '<' || ch == '(' || ch == '[' || ch == '{') ++depth;
    if (ch == ')' || ch == ']' || ch == '}' ||
        (ch == '>' && (i == 0 || text[i - 1] != '-')))
      --depth;
    if (ch == ',' && depth == 0) {
      values.push_back(trim(text.substr(start, i - start)));
      start = i + 1;
    }
  }
  if (auto last = trim(text.substr(start)); !last.empty())
    values.push_back(last);
  return values;
}

[[nodiscard]] auto arrayElements(std::string_view text)
    -> std::vector<std::string_view> {
  text = trim(text);
  if (text.starts_with("array<")) {
    auto colon = text.find(':');
    if (colon == std::string_view::npos) return {};
    text = text.substr(colon + 1, text.size() - colon - 2);
  } else if (text.starts_with('[')) {
    text = text.substr(1, text.size() - 2);
  }
  return splitList(text);
}

[[nodiscard]] auto scalarText(std::string_view text) -> std::string_view {
  return trim(text.substr(0, text.find(" : ")));
}

[[nodiscard]] auto arrayText(std::string_view text) -> std::string {
  return "[" +
         join(arrayElements(text),
              [](auto value) { return std::string(scalarText(value)); }) +
         "]";
}

[[nodiscard]] auto integerValue(std::string_view text) -> std::size_t {
  text = scalarText(text);
  assert(!text.empty());
  std::size_t value = 0;
  for (auto ch : text) {
    assert(ch >= '0' && ch <= '9');
    value = value * 10 + (ch - '0');
  }
  return value;
}

[[nodiscard]] auto enumBody(std::string_view text) -> std::string_view {
  auto first = text.find('<');
  assert(first != std::string_view::npos && text.ends_with('>'));
  return text.substr(first + 1, text.size() - first - 2);
}

[[nodiscard]] auto enumText(std::string_view text) -> std::string_view {
  return text.substr(text.find('<'));
}

[[nodiscard]] auto isBareIdentifier(std::string_view text) -> bool {
  if (text.empty()) return false;
  auto initial = [](unsigned char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_' ||
           ch == '$';
  };
  if (!initial(text.front())) return false;
  for (unsigned char ch : text)
    if (!initial(ch) && !(ch >= '0' && ch <= '9') && ch != '.') return false;
  return true;
}

[[nodiscard]] auto symbolText(std::string_view text) -> std::string {
  if (text.starts_with('@')) text.remove_prefix(1);
  if (text.starts_with('"') && text.ends_with('"')) {
    auto name = text.substr(1, text.size() - 2);
    if (isBareIdentifier(name)) text = name;
  }
  return "@" + std::string(text);
}

class Attributes {
 public:
  explicit Attributes(std::string_view attrs,
                      std::string_view properties = {}) {
    append(attrs);
    append(properties);
  }

  [[nodiscard]] auto get(std::string_view name) const -> std::string_view {
    for (const auto& entry : entries_)
      if (entry.name == name) return entry.value;
    return {};
  }

  auto take(std::string_view name) -> std::string_view {
    for (auto& entry : entries_) {
      if (entry.name != name) continue;
      entry.name = {};
      return entry.value;
    }
    return {};
  }

  [[nodiscard]] auto dictionary(bool keyword = false) const -> std::string {
    std::string text;
    for (const auto& entry : entries_) {
      if (entry.name.empty()) continue;
      if (!text.empty()) text += ", ";
      text += isBareIdentifier(entry.name) ? std::string(entry.name)
                                           : quote(entry.name);
      if (entry.value != "unit") text += " = " + std::string(entry.value);
    }
    if (text.empty()) return {};
    return (keyword ? " attributes {" : " {") + text + "}";
  }

 private:
  struct Entry {
    std::string_view name;
    std::string_view value;
  };
  std::vector<Entry> entries_;

  void append(std::string_view text) {
    for (auto entry : splitList(text)) {
      auto equal = entry.find('=');
      assert(equal != std::string_view::npos);
      auto name = trim(entry.substr(0, equal));
      if (name.starts_with('"')) name = name.substr(1, name.size() - 2);
      entries_.push_back({name, trim(entry.substr(equal + 1))});
    }
  }
};

[[nodiscard]] auto dictionaryText(std::string_view text) -> std::string {
  text = trim(text);
  if (text.empty()) return {};
  assert(text.starts_with('{') && text.ends_with('}'));
  return Attributes(text.substr(1, text.size() - 2)).dictionary();
}

[[nodiscard]] auto isOneOf(std::string_view name,
                           std::initializer_list<std::string_view> names)
    -> bool {
  return std::ranges::find(names, name) != names.end();
}

[[nodiscard]] auto floatBits(const ConstInt& bits, unsigned width)
    -> std::string {
  std::string text = "0x";
  for (unsigned i = width / 4; i > 0; --i) {
    auto word = i > 16 ? bits.highBits() : bits.lowBits();
    auto shift = ((i - 1) % 16) * 4;
    text += "0123456789ABCDEF"[(word >> shift) & 15];
  }
  return text;
}

[[nodiscard]] auto specialFloat(unsigned width, bool negative, bool nan,
                                bool signaling) -> std::string {
  std::string hex(width / 4, '0');
  auto setBit = [&](unsigned bit) {
    auto& digit = hex.at(hex.size() - 1 - bit / 4);
    auto value = digit <= '9' ? digit - '0' : digit - 'A' + 10;
    digit = "0123456789ABCDEF"[value | (1u << (bit % 4))];
  };
  unsigned exponent = 15;
  switch (width) {
    case 16:
      exponent = 5;
      break;
    case 32:
      exponent = 8;
      break;
    case 64:
      exponent = 11;
      break;
  }
  auto fraction = width - exponent - 1;
  for (unsigned i = 0; i < exponent; ++i) setBit(fraction + i);
  if (width == 80) setBit(63);
  if (nan) {
    auto quietBit = fraction - 1;
    if (width == 80) --quietBit;
    setBit(signaling ? 0 : quietBit);
  }
  if (negative) setBit(width - 1);
  return "0x" + hex;
}

}  // namespace

struct CxxIREmitter::Private {
  Private(CxxIREmitter& emitter, TranslationUnit* unit)
      : emitter(emitter), unit(unit) {}

  struct Type {
    TypeKind kind = TypeKind::Other;
    std::string text;
    std::string name;
    unsigned width = 0;
    TypeRef element;
    std::optional<std::uint32_t> addressSpace;
    std::uint64_t count = 0;
    std::vector<TypeRef> parameters;
    std::vector<TypeRef> results;
    std::vector<TypeRef> members;
    std::vector<std::uint32_t> padding;
    bool variadic = false;
    bool defined = false;
    bool packed = false;
    std::uint64_t alignment = 0;
  };

  struct Value {
    TypeRef type;
    bool isZero = false;
    SourceLocation location;
    std::string nameHint;
  };

  struct Operation {
    std::string name;
    std::vector<ValueRef> results;
    std::vector<ValueRef> operands;
    std::string attrs;
    std::string properties;
    SourceLocation location;
    FunctionRef function;
    GlobalRef global;
    std::vector<BlockRef> successors;
    bool terminator = false;
    bool allocation = false;
    std::string gotoLabel;
    std::vector<CleanupAction> cleanups;
    ValueRef indirectTarget;
  };

  using Operations = std::list<Operation>;

  struct Point {
    Operations* operations = nullptr;
    Operations::iterator next;
    BlockRef block;
  };

  struct Block {
    FunctionRef function;
    GlobalRef global;
    std::vector<ValueRef> parameters;
    Operations operations;
    bool erased = false;
  };

  struct Label {
    BlockRef block;
    std::int64_t depth = 0;
    SourceLocation location;
  };

  struct Function {
    std::string name;
    TypeRef type;
    SourceLocation location;
    std::string attrs;
    std::string aliasee;
    std::string personality;
    std::vector<BlockRef> blocks;
    std::unordered_map<std::string, Label> labels;
    std::vector<std::string> addressLabels;
  };

  struct Global {
    std::string attrs;
    SourceLocation location;
    Linkage linkage;
    std::vector<BlockRef> blocks;
  };

  struct Module {
    std::string name;
    std::string sourceFile;
    std::string attrs;
    Operations operations;
    std::unordered_map<std::string, FunctionRef> functions;
    std::unordered_map<std::string, GlobalRef> globals;
    std::unordered_set<std::string> symbols;
  };

  struct ModuleScope {
    ModuleRef module;
    Point point;
  };

  struct CleanupRegion {
    BlockRef block;
    Operation* anchor = nullptr;
  };

  std::vector<Type> types{1};
  CxxIREmitter& emitter;
  TranslationUnit* unit;
  std::unordered_map<std::string, TypeRef> internedTypes;
  std::vector<Value> values{1};
  std::vector<std::unique_ptr<Block>> blocks{1};
  std::vector<std::unique_ptr<Function>> functions{1};
  std::vector<std::unique_ptr<Global>> globals{1};
  std::vector<std::unique_ptr<Module>> modules{1};
  std::vector<Point> points{1};
  std::vector<std::uint32_t> freePoints;
  std::vector<ModuleScope> moduleScopes;
  std::vector<FunctionRef> functionBodies;
  std::unordered_map<CleanupRegionRef, CleanupRegion> cleanupRegions;
  std::uint32_t nextCleanupRegion = 1;
  ModuleRef currentModule;
  Point point;

  [[nodiscard]] auto sourceLocation(SourceLocation loc) const
      -> SourceLocation {
    return loc ? loc : emitter.enclosingLocation();
  }

  struct LocationPrinter {
    TranslationUnit* unit;
    bool enabled;
    std::unordered_map<SourceLocation, std::size_t> sources;
    std::unordered_map<std::string, std::size_t> interned;
    std::vector<std::string> definitions;

    [[nodiscard]] auto reference(std::size_t index) const -> std::string {
      return std::format(" loc(#loc{})", index);
    }

    auto intern(std::string text) -> std::size_t {
      auto [it, inserted] = interned.try_emplace(text, definitions.size());
      if (inserted) definitions.push_back(std::move(text));
      return it->second;
    }

    auto suffix(SourceLocation loc) -> std::string {
      if (!enabled) return {};
      if (!unit || !loc) return reference(intern("unknown"));
      if (auto it = sources.find(loc); it != sources.end())
        return reference(it->second);
      auto [file, line, column] = unit->tokenStartPosition(loc);
      auto index = intern(std::format("{}:{}:{}", quote(file), line, column));
      sources.emplace(loc, index);
      return reference(index);
    }

    auto suffix(std::string_view file) -> std::string {
      if (!enabled) return {};
      return reference(intern(file.empty() ? "unknown" : quote(file) + ":0:0"));
    }

    void print(std::format_context::iterator& out) const {
      for (std::size_t i = 0; i < definitions.size(); ++i)
        out = std::format_to(out, "#loc{} = loc({})\n", i, definitions[i]);
    }
  };

  struct Names {
    struct Counter {
      std::size_t nextValue = 0;
      std::size_t nextArgument = 0;
      std::size_t nextConflict = 0;
      std::unordered_set<std::string> used;
    };

    const Private& ir;
    std::vector<std::string> values;
    std::vector<std::string> blocks;
    std::vector<std::size_t> groupSizes;

    Names(const Private& ir, const Module& module)
        : ir(ir),
          values(ir.values.size()),
          blocks(ir.blocks.size()),
          groupSizes(ir.values.size()) {
      Counter counter;
      numberOperations(module.operations, counter);
      numberRegions(module.operations, counter);
    }

    [[nodiscard]] auto value(ValueRef ref) const -> const std::string& {
      const auto& name = values.at(HandleAccess::id(ref));
      assert(!name.empty() && "value is outside the printed region");
      return name;
    }

    [[nodiscard]] auto block(BlockRef ref) const -> const std::string& {
      const auto& name = blocks.at(HandleAccess::id(ref));
      assert(!name.empty() && "block is outside the printed region");
      return name;
    }

    [[nodiscard]] auto definition(std::span<const ValueRef> results) const
        -> std::string {
      if (results.empty()) return {};
      std::string text;
      for (std::size_t i = 0; i < results.size();) {
        if (i) text += ", ";
        auto count = groupSizes[HandleAccess::id(results[i])];
        assert(count && "missing result group");
        const auto& name = value(results[i]);
        text += name.substr(0, name.find('#'));
        if (count != 1) text += ":" + std::to_string(count);
        i += count;
      }
      return text + " = ";
    }

    auto unique(std::string name, Counter& counter) -> std::string {
      auto candidate = name;
      while (!counter.used.insert(candidate).second)
        candidate = name + "_" + std::to_string(counter.nextConflict++);
      return "%" + candidate;
    }

    void numberOperations(const Operations& operations, Counter& counter) {
      for (const auto& op : operations) {
        if (op.results.empty()) continue;
        for (std::size_t first = 0; first < op.results.size();) {
          auto last = first + 1;
          while (last < op.results.size() &&
                 ir.values[HandleAccess::id(op.results[last])].nameHint.empty())
            ++last;
          auto hint = ir.values[HandleAccess::id(op.results[first])].nameHint;
          auto name = hint.empty() ? "%" + std::to_string(counter.nextValue++)
                                   : unique(std::move(hint), counter);
          groupSizes[HandleAccess::id(op.results[first])] = last - first;
          for (auto i = first; i < last; ++i)
            values[HandleAccess::id(op.results[i])] =
                last - first == 1 ? name
                                  : name + "#" + std::to_string(i - first);
          first = last;
        }
      }
    }

    void numberRegions(const Operations& operations, const Counter& parent) {
      for (const auto& op : operations) {
        if (op.function)
          numberBlocks(ir.function(op.function).blocks, parent);
        else if (op.global)
          numberBlocks(ir.global(op.global).blocks, parent);
      }
    }

    void numberBlocks(std::span<const BlockRef> refs, Counter counter) {
      std::size_t nextBlock = 0;
      for (auto ref : refs) {
        const auto& b = ir.block(ref);
        if (b.erased) continue;
        bool entry = nextBlock == 0;
        blocks[HandleAccess::id(ref)] = "^bb" + std::to_string(nextBlock++);
        for (auto parameter : b.parameters)
          values[HandleAccess::id(parameter)] =
              entry ? unique("arg" + std::to_string(counter.nextArgument++),
                             counter)
                    : "%" + std::to_string(counter.nextValue++);
        numberOperations(b.operations, counter);
      }
      for (auto ref : refs)
        if (!ir.block(ref).erased)
          numberRegions(ir.block(ref).operations, counter);
    }
  };

  [[nodiscard]] auto type(TypeRef ref) const -> const Type& {
    return types.at(HandleAccess::id(ref));
  }
  [[nodiscard]] auto typeText(TypeRef ref) const -> const std::string& {
    assert(ref);
    return type(ref).text;
  }
  [[nodiscard]] auto valueType(ValueRef value) const -> TypeRef {
    return values.at(HandleAccess::id(value)).type;
  }
  [[nodiscard]] auto block(BlockRef ref) const -> Block& {
    return *blocks.at(HandleAccess::id(ref));
  }
  [[nodiscard]] auto function(FunctionRef ref) const -> Function& {
    return *functions.at(HandleAccess::id(ref));
  }
  [[nodiscard]] auto global(GlobalRef ref) const -> Global& {
    return *globals.at(HandleAccess::id(ref));
  }
  [[nodiscard]] auto module() const -> Module& {
    return *modules.at(HandleAccess::id(currentModule));
  }

  auto intern(std::string key, Type info) -> TypeRef {
    if (auto it = internedTypes.find(key); it != internedTypes.end())
      return it->second;
    auto ref = HandleAccess::make<TypeTag>(types.size());
    if (info.text.empty()) info.text = std::format("!cxxir{}", types.size());
    types.push_back(std::move(info));
    internedTypes.emplace(std::move(key), ref);
    return ref;
  }

  enum class TypePrintState { Unseen, Visiting, Printed };

  auto printTypeAlias(TypeRef ref, std::format_context::iterator& out,
                      std::span<TypePrintState> states) const -> std::string {
    const auto& t = type(ref);
    if (t.text != std::format("!cxxir{}", HandleAccess::id(ref))) return t.text;
    auto& state = states[HandleAccess::id(ref)];
    if (state == TypePrintState::Printed) return t.text;
    if (state == TypePrintState::Visiting) {
      if (t.kind == TypeKind::Class) return "!cxx.class<" + quote(t.name) + ">";
      return renderType(ref, out, states);
    }
    state = TypePrintState::Visiting;
    auto definition = renderType(ref, out, states);
    out = std::format_to(out, "{} = {}\n", t.text, definition);
    state = TypePrintState::Printed;
    return t.text;
  }

  auto renderTypes(std::span<const TypeRef> refs,
                   std::format_context::iterator& out,
                   std::span<TypePrintState> states) const -> std::string {
    std::string text;
    for (auto ref : refs) {
      if (!text.empty()) text += ", ";
      text += printTypeAlias(ref, out, states);
    }
    return text;
  }

  auto renderType(TypeRef ref, std::format_context::iterator& out,
                  std::span<TypePrintState> states) const -> std::string {
    const auto& t = type(ref);
    switch (t.kind) {
      case TypeKind::Pointer:
        return "!cxx.ptr<" + printTypeAlias(t.element, out, states) +
               (t.addressSpace ? std::format(", {}", *t.addressSpace) : "") +
               ">";
      case TypeKind::Array:
        return std::format("!cxx.array<{}, {}>",
                           printTypeAlias(t.element, out, states), t.count);
      case TypeKind::Vector:
        return std::format("vector<{}x{}>", t.count,
                           printTypeAlias(t.element, out, states));
      case TypeKind::Function: {
        auto parameters = renderTypes(t.parameters, out, states);
        auto results = renderTypes(t.results, out, states);
        return std::format("!cxx.function<({}), ({}), {}>", parameters, results,
                           t.variadic ? "true" : "false");
      }
      case TypeKind::Class: {
        auto result = "!cxx.class<" + quote(t.name);
        if (!t.defined) return result + ">";
        result += ", ";
        if (t.packed) result += "packed ";
        if (t.alignment) result += std::format("align {} ", t.alignment);
        result += '(';
        for (std::size_t i = 0; i < t.members.size(); ++i) {
          if (i) result += ", ";
          if (std::ranges::find(t.padding, i) != t.padding.end())
            result += "padding ";
          result += printTypeAlias(t.members[i], out, states);
        }
        return result + ")>";
      }
      case TypeKind::Other:
        if (!t.members.empty())
          return "tuple<" + renderTypes(t.members, out, states) + ">";
        [[fallthrough]];
      default:
        return t.text;
    }
  }

  auto newValue(TypeRef type) -> ValueRef {
    auto ref = HandleAccess::make<ValueTag>(values.size());
    values.push_back({type});
    return ref;
  }

  [[nodiscard]] auto operandTypes(std::span<const ValueRef> operands) const
      -> std::string {
    return join(operands,
                [&](auto value) { return typeText(valueType(value)); });
  }

  struct OperationFormatter {
    const Private& ir;
    const Operation& op;
    const Names& names;
    Attributes attributes;
    std::string prefix;

    OperationFormatter(const Private& ir, const Operation& op,
                       const Names& names)
        : ir(ir),
          op(op),
          names(names),
          attributes(op.attrs, op.properties),
          prefix(names.definition(op.results)) {}

    [[nodiscard]] auto value(std::size_t index) const -> const std::string& {
      return names.value(op.operands[index]);
    }

    [[nodiscard]] auto operandType(std::size_t index) const -> std::string {
      return ir.typeText(ir.valueType(op.operands[index]));
    }

    [[nodiscard]] auto resultType() const -> std::string {
      return ir.typeText(ir.valueType(op.results.front()));
    }

    [[nodiscard]] auto resultTypes() const -> std::string {
      if (op.results.size() == 1) return resultType();
      return std::format("({})", ir.operandTypes(op.results));
    }

    [[nodiscard]] auto signature(std::span<const ValueRef> inputs) const
        -> std::string {
      return std::format(" : ({}) -> {}", ir.operandTypes(inputs),
                         resultTypes());
    }

    [[nodiscard]] auto finish(std::string_view arguments = {},
                              std::string_view types = {}) const
        -> std::string {
      auto text = std::format("{}{}", prefix, op.name);
      if (!arguments.empty())
        std::format_to(std::back_inserter(text), " {}", arguments);
      std::format_to(std::back_inserter(text), "{}{}", attributes.dictionary(),
                     types);
      return text;
    }

    [[nodiscard]] auto valueNames(std::span<const ValueRef> values) const
        -> std::string {
      return join(values, [&](auto ref) { return names.value(ref); });
    }

    [[nodiscard]] auto blockNames(std::span<const BlockRef> blocks) const
        -> std::string {
      return join(blocks, [&](auto ref) { return names.block(ref); });
    }

    [[nodiscard]] auto group(std::span<const ValueRef> values) const
        -> std::string {
      return std::format("({} : {})", valueNames(values),
                         ir.operandTypes(values));
    }

    [[nodiscard]] auto format() -> std::string;
  };

  auto emit(SourceLocation loc, std::string_view name,
            std::span<const TypeRef> resultTypes,
            std::span<const ValueRef> operands, std::string attrs = {},
            std::span<const BlockRef> successors = {}, bool terminator = false,
            std::string properties = {}) -> std::vector<ValueRef> {
    assert(point.operations && "emission requires an insertion point");
    std::vector<ValueRef> results;
    for (auto type : resultTypes) results.push_back(newValue(type));
    if (name == "cxx.atomic.cmpxchg" || name == "cxx.landing_pad") {
      values[HandleAccess::id(results[0])].nameHint =
          name == "cxx.atomic.cmpxchg" ? "old" : "exception";
      values[HandleAccess::id(results[1])].nameHint =
          name == "cxx.atomic.cmpxchg" ? "exchanged" : "selector";
    }
    Operation op;
    op.name = name;
    op.results = results;
    op.operands.assign(operands.begin(), operands.end());
    op.attrs = std::move(attrs);
    op.properties = std::move(properties);
    op.location = sourceLocation(loc);
    op.successors.assign(successors.begin(), successors.end());
    op.terminator = terminator;
    point.operations->insert(point.next, std::move(op));
    return results;
  }

  auto one(SourceLocation loc, std::string_view name, TypeRef result,
           std::initializer_list<ValueRef> operands = {},
           std::string attrs = {}, std::string properties = {}) -> ValueRef {
    return emit(loc, name, std::array{result},
                {operands.begin(), operands.size()}, std::move(attrs), {},
                false, std::move(properties))
        .front();
  }

  void op(SourceLocation loc, std::string_view name,
          std::initializer_list<ValueRef> operands = {}, std::string attrs = {},
          std::initializer_list<BlockRef> successors = {},
          bool terminator = false, std::string properties = {}) {
    (void)emit(loc, name, {}, {operands.begin(), operands.size()},
               std::move(attrs), {successors.begin(), successors.size()},
               terminator, std::move(properties));
  }

  void appendModule(Operation op) {
    auto& operations = module().operations;
    auto position =
        point.operations == &operations ? point.next : operations.end();
    operations.insert(position, std::move(op));
  }

  [[nodiscard]] auto abiAttrs(std::span<const ParameterAbi> parameters,
                              std::size_t count) const -> std::string {
    std::vector<std::string> entries;
    bool any = false;
    for (std::size_t i = 0; i < count; ++i) {
      auto abi = i < parameters.size() ? parameters[i] : ParameterAbi{};
      std::string attrs;
      if (abi.kind == ParameterAbiKind::ZeroExtend)
        attrs = "llvm.zeroext = unit";
      if (abi.kind == ParameterAbiKind::StructReturn ||
          abi.kind == ParameterAbiKind::ByValue) {
        addAttr(attrs,
                abi.kind == ParameterAbiKind::StructReturn ? "llvm.sret"
                                                           : "llvm.byval",
                typeText(abi.indirectType));
        if (abi.alignment)
          addAttr(attrs, "llvm.align", std::format("{} : i64", abi.alignment));
      }
      any |= !attrs.empty();
      entries.push_back("{" + attrs + "}");
    }
    if (!any) return {};
    return "[" + join(entries, [](const auto& s) { return s; }) + "]";
  }

  void signatureAttrs(std::string& attrs,
                      std::span<const ParameterAbi> parameters,
                      std::size_t parameterCount, ParameterAbi result,
                      std::size_t resultCount,
                      CallingConvention convention) const {
    if (auto args = abiAttrs(parameters, parameterCount); !args.empty())
      addAttr(attrs, "arg_attrs", args);
    if (auto results = abiAttrs(std::array{result}, resultCount);
        !results.empty())
      addAttr(attrs, "res_attrs", results);
    if (convention != CallingConvention::C)
      addAttr(attrs, "calling_convention",
              "#cxx.calling_convention<cxx_fast_tls>");
  }

  [[nodiscard]] auto initializer(const Initializer& init,
                                 TypeRef fallback = {}) const -> std::string;
  void printBlocks(std::span<const BlockRef> blocks,
                   std::format_context::iterator& out,
                   LocationPrinter& locations, const Names& names,
                   bool skipEntryLabel = false) const;
  void printOperations(const Operations& operations,
                       std::format_context::iterator& out,
                       LocationPrinter& locations, const Names& names,
                       std::string_view indent) const;
};

auto CxxIREmitter::Private::OperationFormatter::format() -> std::string {
  std::string_view name = op.name;
  std::span<const ValueRef> results = op.results;
  std::span<const ValueRef> operands = op.operands;
  std::span<const BlockRef> successors = op.successors;
  const auto& attrs = op.attrs;
  const auto& properties = op.properties;

  if (name == "arith.constant") {
    auto literal = attributes.take("value");
    return prefix + std::string(name) + attributes.dictionary() + " " +
           std::string(literal);
  }
  if (name == "arith.cmpi" || name == "arith.cmpf") {
    static constexpr std::string_view integers[] = {
        "eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"};
    static constexpr std::string_view floats[] = {
        "false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord",
        "ueq",   "ugt", "uge", "ult", "ule", "une", "uno", "true"};
    auto predicate = integerValue(attributes.take("predicate"));
    auto spelling =
        name == "arith.cmpi" ? integers[predicate] : floats[predicate];
    return finish(std::string(spelling) + ", " + valueNames(operands),
                  " : " + operandType(0));
  }
  if (name == "arith.select") {
    std::string types = " : ";
    if (operandType(0) != "i1") types += operandType(0) + ", ";
    return finish(valueNames(operands), types + resultType());
  }
  if (name == "arith.negf") return finish(value(0), " : " + resultType());
  if (name.starts_with("arith.") && operands.size() == 2 &&
      results.size() == 1) {
    auto arguments = valueNames(operands);
    if (auto flags = attributes.take("overflowFlags"); !flags.empty())
      arguments += " overflow" + std::string(enumText(flags));
    return finish(arguments, " : " + resultType());
  }
  if (name.starts_with("arith.") && operands.size() == 1 && results.size() == 1)
    return finish(value(0), " : " + operandType(0) + " to " + resultType());

  if (name == "cf.br") {
    auto target = names.block(successors.front());
    if (!operands.empty())
      target +=
          "(" + valueNames(operands) + " : " + ir.operandTypes(operands) + ")";
    return finish(target);
  }
  if (name == "cf.cond_br") {
    (void)attributes.take("operandSegmentSizes");
    return finish(value(0) + ", " + blockNames(successors));
  }

  if (name == "vector.broadcast")
    return finish(value(0), " : " + operandType(0) + " to " + resultType());
  if (name == "vector.from_elements")
    return finish(valueNames(operands), " : " + resultType());
  if (name == "vector.shuffle") {
    auto mask = arrayElements(attributes.take("mask"));
    auto indices =
        join(mask, [](auto index) { return std::string(scalarText(index)); });
    return finish(valueNames(operands) + " [" + indices + "]",
                  " : " + ir.operandTypes(operands));
  }
  if (name == "vector.extract" || name == "vector.insert") {
    bool insert = name == "vector.insert";
    auto positions = arrayElements(attributes.take("static_position"));
    std::string indices;
    std::size_t dynamic = insert ? 2 : 1;
    for (auto position : positions) {
      if (!indices.empty()) indices += ", ";
      indices += scalarText(position) == "-9223372036854775808"
                     ? value(dynamic++)
                     : std::string(scalarText(position));
    }
    auto arguments = insert ? value(0) + ", " + value(1) : value(0);
    arguments += "[" + indices + "]";
    return finish(arguments,
                  insert ? " : " + operandType(0) + " into " + operandType(1)
                         : " : " + resultType() + " from " + operandType(0));
  }
  if (name == "llvm.blocktag")
    return finish(std::string(enumText(attributes.take("tag"))));

  if (name == "cxx.return")
    return finish(valueNames(operands),
                  operands.empty() ? "" : " : " + ir.operandTypes(operands));
  if (name == "cxx.unreachable") return finish();
  if (isOneOf(name, {"cxx.alloca", "cxx.workgroup_alloc", "cxx.dyn_alloca"})) {
    auto alignment = scalarText(attributes.take("alignment"));
    auto types = name == "cxx.dyn_alloca" ? operandType(0) + " -> " : "";
    return finish(valueNames(operands), " : " + types + resultType() +
                                            " align " + std::string(alignment));
  }
  if (isOneOf(name, {"cxx.load", "cxx.store", "cxx.bitfield.load",
                     "cxx.bitfield.store"})) {
    auto qualifier = attributes.take("volatile_").empty() ? "" : "volatile ";
    bool bitfield = name.starts_with("cxx.bitfield.");
    auto alignment =
        bitfield ? std::string_view{} : attributes.take("alignment");
    bool load = name == "cxx.load" || name == "cxx.bitfield.load";
    return finish(
        std::string(qualifier) + valueNames(operands),
        " : " +
            (load ? operandType(0) + " -> " + resultType()
                  : ir.operandTypes(operands)) +
            (bitfield ? "" : " align " + std::string(scalarText(alignment))));
  }
  if (name == "cxx.atomic.fence" || name == "cxx.control_barrier") {
    std::string arguments;
    if (name == "cxx.control_barrier")
      arguments = std::string(enumText(attributes.take("execution_scope"))) +
                  " " + std::string(enumText(attributes.take("memory_scope"))) +
                  " ";
    arguments += std::string(enumText(attributes.take("ordering"))) + " ";
    if (name == "cxx.atomic.fence")
      arguments += std::string(enumText(attributes.take("scope"))) + " ";
    arguments += enumText(attributes.take("memory_regions"));
    return finish(arguments);
  }
  if (name.starts_with("cxx.atomic.")) {
    std::string arguments =
        attributes.take("volatile_").empty() ? "" : "volatile ";
    if (name == "cxx.atomic.rmw")
      arguments += std::string(enumText(attributes.take("kind"))) + " ";
    arguments += valueNames(operands) + " ";
    if (name == "cxx.atomic.cmpxchg") {
      arguments +=
          std::string(enumText(attributes.take("success_order"))) + " " +
          std::string(enumText(attributes.take("failure_order"))) + " ";
    } else {
      arguments += std::string(enumText(attributes.take("ordering"))) + " ";
    }
    arguments += enumText(attributes.take("scope"));
    if (!attributes.take("weak").empty()) arguments += " weak";
    if (name == "cxx.atomic.load")
      return finish(arguments, " : " + operandType(0) + " -> " + resultType());
    if (name == "cxx.atomic.cmpxchg")
      return finish(arguments, " : " + operandType(0) + ", " + operandType(1));
    return finish(arguments, " : " + ir.operandTypes(operands));
  }
  if (name == "cxx.memzero" || name == "cxx.memcpy") {
    auto size = scalarText(attributes.take("size"));
    return finish(valueNames(operands), " : " + ir.operandTypes(operands) +
                                            " size " + std::string(size));
  }
  if (name == "cxx.zero_init" || name == "cxx.copy")
    return finish(valueNames(operands), " : " + ir.operandTypes(operands));
  if (name == "cxx.member") {
    auto index = scalarText(attributes.take("member_index"));
    return finish(value(0) + "[" + std::string(index) + "]",
                  " : " + operandType(0) + " -> " + resultType());
  }
  if (name == "cxx.subscript")
    return finish(value(0) + "[" + value(1) + "]", signature(operands));
  if (name == "cxx.ptradd" || name == "cxx.ptrdiff")
    return finish(valueNames(operands), signature(operands));
  if (isOneOf(name, {"cxx.bitcast", "cxx.reshape", "cxx.array_to_ptr",
                     "cxx.ptrtoint", "cxx.inttoptr", "cxx.addrspace_cast"}))
    return finish(value(0), " : " + operandType(0) + " to " + resultType());
  if (name == "cxx.extractvalue" || name == "cxx.insertvalue") {
    auto position = scalarText(attributes.take("position"));
    auto arguments = value(0) + "[" + std::string(position) + "]";
    if (name == "cxx.insertvalue") arguments = value(1) + ", " + arguments;
    return finish(arguments,
                  " : " + operandType(0) +
                      (name == "cxx.extractvalue" ? " to " + resultType()
                                                  : ", " + operandType(1)));
  }
  if (name == "cxx.address_of")
    return finish(symbolText(attributes.take("sym_name")),
                  " : " + resultType());
  if (name == "cxx.constant.aggregate")
    return finish(std::string(attributes.take("value")), " : " + resultType());
  if (isOneOf(name, {"cxx.constant.null", "cxx.zero", "cxx.undef"}))
    return finish({}, " : " + resultType());
  if (name == "cxx.todo.expr" || name == "cxx.todo.stmt")
    return finish(std::string(attributes.take("message")),
                  results.empty() ? "" : " : " + resultType());
  if (name == "cxx.label_address")
    return finish(std::string(attributes.take("label_name")),
                  " : " + resultType());
  if (name == "cxx.indirect_goto")
    return finish(value(0) + " [" + blockNames(successors) + "]",
                  " : " + operandType(0));
  if (name == "cxx.global_ctor")
    return finish(symbolText(attributes.take("ctor")));
  if (name == "cxx.module_asm")
    return finish(std::string(attributes.take("assembly")));
  if (name == "cxx.exception_typeid")
    return finish(value(0), " : " + operandType(0));
  if (name == "cxx.resume")
    return finish(valueNames(operands), " : " + operandType(0));
  if (name == "cxx.landing_pad") {
    auto cleanup = attributes.take("cleanup").empty() ? "" : "cleanup ";
    return finish(std::string(cleanup) + "(" + valueNames(operands) + ")",
                  signature(operands));
  }
  if (name == "cxx.builtin.call")
    return finish(std::string(attributes.take("builtin_name")) + "(" +
                      valueNames(operands) + ")",
                  signature(operands));
  if (name == "cxx.call" || name == "cxx.invoke") {
    (void)attributes.take("operandSegmentSizes");
    auto callee = attributes.take("callee");
    auto arguments = callee.empty()
                         ? "indirect " + value(0) + " : " + operandType(0)
                         : symbolText(callee);
    auto inputs = operands.subspan(callee.empty() ? 1 : 0);
    arguments += "(" + valueNames(inputs) + ")";
    if (name == "cxx.invoke")
      arguments += " to " + names.block(successors[0]) + " unwind " +
                   names.block(successors[1]);
    return finish(arguments, signature(inputs));
  }
  if (name == "cxx.inline_asm" || name == "cxx.asm_goto") {
    auto sideEffects =
        attributes.take("has_side_effects").empty() ? "" : "has_side_effects ";
    auto arguments = std::string(sideEffects) +
                     std::string(attributes.take("asm_string")) + ", " +
                     std::string(attributes.take("constraints")) + "(" +
                     valueNames(operands) + ")";
    if (name == "cxx.asm_goto")
      arguments += " to " + names.block(successors.front()) + " [" +
                   blockNames(successors.subspan(1)) + "]";
    return finish(arguments, signature(operands));
  }
  if (name == "cxx.cleanup_branch") {
    auto segments = arrayElements(attributes.take("operandSegmentSizes"));
    auto addressCount = integerValue(segments[0]);
    auto flagCount = integerValue(segments[1]);
    auto addresses = operands.first(addressCount);
    auto flags = operands.subspan(addressCount, flagCount);
    auto initialized = operands.subspan(addressCount + flagCount);
    auto arguments = group(addresses) + " destructors " +
                     std::string(attributes.take("destructors")) + " flags " +
                     group(flags) + " active " +
                     arrayText(attributes.take("activeFlagIndices")) +
                     " counts " + arrayText(attributes.take("elementCounts"));
    if (!initialized.empty()) arguments += " initialized " + group(initialized);
    return finish(arguments + " " + names.block(successors.front()));
  }
  if (name == "cxx.vtable") {
    auto arguments = symbolText(attributes.take("sym_name")) + " ";
    if (auto linkage = attributes.take("linkage_kind"); !linkage.empty())
      arguments += std::string(enumBody(linkage)) + " ";
    if (auto info = attributes.take("type_info"); !info.empty())
      arguments += "type_info " + symbolText(info) + " ";
    auto offsets = arrayElements(attributes.take("offsets"));
    auto tops = arrayElements(attributes.take("offsets_to_top"));
    auto slots = arrayElements(attributes.take("slots"));
    arguments += "{";
    for (std::size_t i = 0; i < slots.size(); ++i) {
      arguments += i ? ", " : " ";
      auto tableOffsets = arrayElements(offsets[i]);
      if (!tableOffsets.empty())
        arguments += "offsets [" +
                     join(tableOffsets,
                          [](auto n) { return std::string(scalarText(n)); }) +
                     "] ";
      arguments += "offset_to_top " + std::string(scalarText(tops[i])) + " [";
      arguments +=
          join(arrayElements(slots[i]),
               [](auto slot) {
                 return slot == "unit" ? std::string("null") : symbolText(slot);
               }) +
          "]";
    }
    return finish(arguments + " }");
  }

  auto text = prefix + quote(name) + "(" + valueNames(operands) + ")";
  if (!successors.empty()) text += "[" + blockNames(successors) + "]";
  if (!properties.empty()) text += " <{" + properties + "}>";
  if (!attrs.empty()) text += " {" + attrs + "}";
  return text + " : (" + ir.operandTypes(operands) + ") -> (" +
         ir.operandTypes(results) + ")";
}

CxxIREmitter::CxxIREmitter(TranslationUnit* unit)
    : priv_(std::make_unique<Private>(*this, unit)) {}
CxxIREmitter::~CxxIREmitter() = default;

auto CxxIREmitter::debug() -> DebugEmitter* { return nullptr; }

auto CxxIREmitter::beginModule(const ModuleInfo& info) -> ModuleRef {
  auto& p = *priv_;
  auto module = std::make_unique<Private::Module>();
  module->name = info.name;
  module->sourceFile = info.sourceFile;
  auto& attrs = module->attrs;
  addStringAttr(attrs, "cxx.triple", info.targetTriple);
  addStringAttr(attrs, "cxx.data-layout", info.dataLayout);
  addStringAttr(attrs, "cxx.frame-pointer", info.framePointer);
  addStringAttr(attrs, "cxx.device_profile", info.deviceProfile);
  addStringAttr(attrs, "cxx.debug-compilation-dir",
                info.debugCompilationDirectory);
  if (info.picLevel)
    addAttr(attrs, "cxx.pic-level", std::format("{} : i32", info.picLevel));
  if (info.pieLevel)
    addAttr(attrs, "cxx.pie-level", std::format("{} : i32", info.pieLevel));
  auto ref = HandleAccess::make<ModuleTag>(p.modules.size());
  p.modules.push_back(std::move(module));
  p.moduleScopes.push_back({p.currentModule, p.point});
  p.currentModule = ref;
  setModuleInsertionPoint(false);
  return ref;
}

void CxxIREmitter::endModule() {
  auto& p = *priv_;
  auto scope = p.moduleScopes.back();
  p.moduleScopes.pop_back();
  p.currentModule = scope.module;
  p.point = scope.point;
}

auto CxxIREmitter::module(ModuleRef ref, bool printLocations) const
    -> ModuleView {
  return {*this, ref, printLocations};
}

auto CxxIREmitter::formatTo(ModuleRef ref, std::format_context::iterator out,
                            bool printLocations) const
    -> std::format_context::iterator {
  const auto& p = *priv_;
  Private::LocationPrinter locations{.unit = p.unit, .enabled = printLocations};
  std::vector<Private::TypePrintState> states(p.types.size());
  for (std::size_t i = 1; i < p.types.size(); ++i)
    (void)p.printTypeAlias(HandleAccess::make<TypeTag>(i), out, states);
  const auto& module = *p.modules.at(HandleAccess::id(ref));
  Private::Names names(p, module);
  out = std::format_to(out, "module");
  if (!module.name.empty())
    out = std::format_to(out, " @{}", quote(module.name));
  if (!module.attrs.empty())
    out = std::format_to(out, " attributes {{{}}}", module.attrs);
  out = std::format_to(out, " {{\n");
  p.printOperations(module.operations, out, locations, names, "  ");
  out = std::format_to(out, "}}{}\n", locations.suffix(module.sourceFile));
  locations.print(out);
  return out;
}

void CxxIREmitter::Private::printBlocks(std::span<const BlockRef> refs,
                                        std::format_context::iterator& out,
                                        LocationPrinter& locations,
                                        const Names& names,
                                        bool skipEntryLabel) const {
  bool entry = true;
  for (auto ref : refs) {
    const auto& b = block(ref);
    if (b.erased) continue;
    if (entry && (skipEntryLabel || b.parameters.empty())) {
      entry = false;
      printOperations(b.operations, out, locations, names, "    ");
      continue;
    }
    entry = false;
    out = std::format_to(out, "  {}", names.block(ref));
    if (!b.parameters.empty()) {
      auto parameters = join(b.parameters, [&](ValueRef v) {
        return names.value(v) + ": " + typeText(valueType(v)) +
               locations.suffix(values[HandleAccess::id(v)].location);
      });
      out = std::format_to(out, "({})", parameters);
    }
    out = std::format_to(out, ":\n");
    printOperations(b.operations, out, locations, names, "    ");
  }
}

void CxxIREmitter::Private::printOperations(const Operations& operations,
                                            std::format_context::iterator& out,
                                            LocationPrinter& locations,
                                            const Names& names,
                                            std::string_view indent) const {
  for (const auto& op : operations) {
    if (op.function) {
      const auto& fn = function(op.function);
      auto attrs = fn.attrs;
      addStringAttr(attrs, "aliasee", fn.aliasee);
      if (!fn.personality.empty())
        addAttr(attrs, "personality", "@" + quote(fn.personality));
      Attributes attributes(attrs);
      auto& signature = type(fn.type);
      (void)attributes.take("function_type");
      auto parameterAttrs = arrayElements(attributes.take("arg_attrs"));
      auto resultAttrs = arrayElements(attributes.take("res_attrs"));
      out = std::format_to(out, "{}cxx.func ", indent);
      if (auto linkage = attributes.get("linkage_kind");
          !linkage.empty() && enumBody(linkage) != "external")
        out = std::format_to(out, "{} ",
                             enumBody(attributes.take("linkage_kind")));
      if (auto kind = attributes.get("inline_kind");
          !kind.empty() && enumBody(kind) != "no_inline")
        out = std::format_to(out, "{} ",
                             enumBody(attributes.take("inline_kind")));
      out = std::format_to(out, "{}(", symbolText(attributes.take("sym_name")));
      bool hasBody = !fn.blocks.empty();
      for (std::size_t i = 0; i < signature.parameters.size(); ++i) {
        if (i) out = std::format_to(out, ", ");
        if (hasBody)
          out = std::format_to(
              out, "{}: ", names.value(block(fn.blocks.front()).parameters[i]));
        out = std::format_to(out, "{}", typeText(signature.parameters[i]));
        if (i < parameterAttrs.size() && parameterAttrs[i] != "{}")
          out = std::format_to(out, "{}", dictionaryText(parameterAttrs[i]));
        if (hasBody) {
          auto parameter = block(fn.blocks.front()).parameters[i];
          out = std::format_to(
              out, "{}",
              locations.suffix(values[HandleAccess::id(parameter)].location));
        }
      }
      if (signature.variadic)
        out = std::format_to(out, "{}",
                             signature.parameters.empty() ? "..." : ", ...");
      out = std::format_to(out, ")");
      if (!signature.results.empty()) {
        out = std::format_to(out, " -> ");
        bool parenthesized =
            signature.results.size() != 1 ||
            (!resultAttrs.empty() && resultAttrs.front() != "{}");
        if (parenthesized) out = std::format_to(out, "(");
        for (std::size_t i = 0; i < signature.results.size(); ++i) {
          if (i) out = std::format_to(out, ", ");
          out = std::format_to(out, "{}", typeText(signature.results[i]));
          if (i < resultAttrs.size() && resultAttrs[i] != "{}")
            out = std::format_to(out, "{}", dictionaryText(resultAttrs[i]));
        }
        if (parenthesized) out = std::format_to(out, ")");
      }
      out = std::format_to(out, "{}", attributes.dictionary(true));
      if (hasBody) {
        out = std::format_to(out, " {{\n");
        printBlocks(fn.blocks, out, locations, names, true);
        out = std::format_to(out, "{}}}", indent);
      }
      out = std::format_to(out, "{}\n", locations.suffix(fn.location));
    } else if (op.global) {
      const auto& g = global(op.global);
      Attributes attributes(g.attrs);
      out = std::format_to(out, "{}cxx.global {}", indent,
                           symbolText(attributes.take("sym_name")));
      if (auto linkage = attributes.get("linkage_kind");
          !linkage.empty() && enumBody(linkage) != "external")
        out = std::format_to(out, " {}",
                             enumBody(attributes.take("linkage_kind")));
      if (!attributes.take("constant").empty())
        out = std::format_to(out, " constant");
      out = std::format_to(out, " : {}", attributes.take("global_type"));
      if (auto value = attributes.take("value"); !value.empty())
        out = std::format_to(out, " = {}", value);
      if (!g.blocks.empty()) {
        out = std::format_to(out, " {{\n");
        printBlocks(g.blocks, out, locations, names);
        out = std::format_to(out, "{}}}", indent);
      }
      if (auto alignment = attributes.take("alignment"); !alignment.empty())
        out = std::format_to(out, " alignment({})", scalarText(alignment));
      out = std::format_to(out, "{}{}\n", attributes.dictionary(true),
                           locations.suffix(g.location));
    } else {
      out = std::format_to(out, "{}{}{}\n", indent,
                           OperationFormatter{*this, op, names}.format(),
                           locations.suffix(op.location));
    }
  }
}

auto CxxIREmitter::voidType() -> TypeRef {
  return priv_->intern("void", {.kind = TypeKind::Void, .text = "!cxx.void"});
}

auto CxxIREmitter::unresolvedType() -> TypeRef {
  return priv_->intern("expr",
                       {.kind = TypeKind::Unresolved, .text = "!cxx.expr"});
}

auto CxxIREmitter::integerType(unsigned bits) -> TypeRef {
  auto text = std::format("i{}", bits);
  return priv_->intern(
      text, {.kind = TypeKind::Integer, .text = text, .width = bits});
}

auto CxxIREmitter::floatingType(FloatKind kind) -> TypeRef {
  static constexpr unsigned widths[] = {16, 32, 64, 80, 128};
  auto bits = widths[static_cast<int>(kind)];
  auto text = std::format("f{}", bits);
  return priv_->intern(
      text, {.kind = TypeKind::Floating, .text = text, .width = bits});
}

auto CxxIREmitter::pointerType(TypeRef element,
                               std::optional<std::uint32_t> space) -> TypeRef {
  if (space == 0) space.reset();
  auto key =
      std::format("ptr:{}:{}", HandleAccess::id(element), space.value_or(0));
  return priv_->intern(
      key,
      {.kind = TypeKind::Pointer, .element = element, .addressSpace = space});
}

auto CxxIREmitter::rebindPointerType(TypeRef pointer, TypeRef element)
    -> TypeRef {
  return pointerType(element, priv_->type(pointer).addressSpace);
}

auto CxxIREmitter::arrayType(TypeRef element, std::uint64_t size) -> TypeRef {
  auto key = std::format("array:{}:{}", HandleAccess::id(element), size);
  return priv_->intern(
      key, {.kind = TypeKind::Array, .element = element, .count = size});
}

auto CxxIREmitter::vectorType(TypeRef element, std::uint64_t size) -> TypeRef {
  auto key = std::format("vector:{}:{}", HandleAccess::id(element), size);
  return priv_->intern(
      key, {.kind = TypeKind::Vector, .element = element, .count = size});
}

auto CxxIREmitter::functionType(std::span<const TypeRef> parameters,
                                std::span<const TypeRef> results, bool variadic)
    -> TypeRef {
  auto key = std::format(
      "function:({})->({}):{}",
      join(parameters,
           [](auto t) { return std::to_string(HandleAccess::id(t)); }),
      join(results, [](auto t) { return std::to_string(HandleAccess::id(t)); }),
      variadic);
  Private::Type type;
  type.kind = TypeKind::Function;
  type.parameters.assign(parameters.begin(), parameters.end());
  type.results.assign(results.begin(), results.end());
  type.variadic = variadic;
  return priv_->intern(key, std::move(type));
}

auto CxxIREmitter::declareClassType(std::string_view name) -> TypeRef {
  return priv_->intern("class:" + std::string(name),
                       {.kind = TypeKind::Class, .name = std::string(name)});
}

void CxxIREmitter::defineClassType(TypeRef ref,
                                   std::span<const TypeRef> members,
                                   bool packed, std::uint64_t alignment,
                                   std::span<const std::uint32_t> padding) {
  auto& t = priv_->types.at(HandleAccess::id(ref));
  t.members.assign(members.begin(), members.end());
  t.padding.assign(padding.begin(), padding.end());
  t.packed = packed;
  t.alignment = alignment;
  t.defined = true;
}

auto CxxIREmitter::typeKind(TypeRef type) -> TypeKind {
  return priv_->type(type).kind;
}
auto CxxIREmitter::scalarWidth(TypeRef type) -> unsigned {
  return priv_->type(type).width;
}
auto CxxIREmitter::elementType(TypeRef type) -> TypeRef {
  return priv_->type(type).element;
}
auto CxxIREmitter::typeOf(ValueRef value) -> TypeRef {
  return priv_->valueType(value);
}

auto CxxIREmitter::Private::initializer(const Initializer& init,
                                        TypeRef fallback) const -> std::string {
  auto ref = init.type ? init.type : fallback;
  auto typed = [&](std::string value) { return value + " : " + typeText(ref); };
  switch (init.kind) {
    case Initializer::Kind::None:
    case Initializer::Kind::Undef:
      return {};
    case Initializer::Kind::Integer:
      return typed(init.integer.toString());
    case Initializer::Kind::Floating:
      if (std::isfinite(init.floating))
        return typed(std::format("{:.17e}", init.floating));
      [[fallthrough]];
    case Initializer::Kind::SignalingNaN: {
      bool signaling = init.kind == Initializer::Kind::SignalingNaN;
      return typed(specialFloat(
          type(ref).width, !signaling && std::signbit(init.floating),
          signaling || std::isnan(init.floating), signaling));
    }
    case Initializer::Kind::FloatingBits: {
      return typed(floatBits(init.integer, type(ref).width));
    }
    case Initializer::Kind::Bytes:
      return quote(init.bytes);
    case Initializer::Kind::Aggregate:
      return "[" +
             join(init.elements,
                  [&](const auto& element) { return initializer(element); }) +
             "]";
    case Initializer::Kind::Vector:
      return typed("dense<[" +
                   join(init.elements,
                        [&](const auto& element) {
                          auto value = initializer(element);
                          return value.substr(0, value.rfind(" : "));
                        }) +
                   "]>");
    case Initializer::Kind::Null:
      return "unit";
    case Initializer::Kind::Zero:
      return "#llvm.zero";
    case Initializer::Kind::ScalarZero:
      if (type(ref).kind == TypeKind::Vector)
        return typed(type(type(ref).element).kind == TypeKind::Floating
                         ? "dense<0.000000e+00>"
                         : "dense<0>");
      return typed(type(ref).kind == TypeKind::Floating ? "0.000000e+00" : "0");
  }
  return {};
}

auto CxxIREmitter::constant(SourceLocation loc, TypeRef type,
                            const Initializer& init) -> ValueRef {
  auto& p = *priv_;
  std::string_view name = "arith.constant";
  std::string attrs;
  switch (init.kind) {
    case Initializer::Kind::None:
      return {};
    case Initializer::Kind::Null:
      name = "cxx.constant.null";
      break;
    case Initializer::Kind::Zero:
      name = "cxx.zero";
      break;
    case Initializer::Kind::Undef:
      name = "cxx.undef";
      break;
    case Initializer::Kind::Aggregate:
      name = "cxx.constant.aggregate";
      [[fallthrough]];
    default:
      addAttr(attrs, "value", p.initializer(init, type));
      break;
  }
  auto value = p.one(loc, name, type, {}, attrs);
  auto& result = p.values[HandleAccess::id(value)];
  if (name == "arith.constant") {
    result.nameHint = "cst";
    const auto& t = p.type(type);
    if (t.kind == TypeKind::Integer &&
        (init.kind == Initializer::Kind::Integer ||
         init.kind == Initializer::Kind::ScalarZero)) {
      auto integer = init.kind == Initializer::Kind::ScalarZero ? ConstInt{0}
                                                                : init.integer;
      if (auto signedValue = ConstInt::make(integer.toWide(), t.width, true))
        integer = *signedValue;
      if (t.width == 1)
        result.nameHint = integer.lowBits() & 1 ? "true" : "false";
      else
        result.nameHint = std::format("c{}_{}", integer.toString(), t.text);
    }
  }
  p.values[HandleAccess::id(value)].isZero =
      init.kind == Initializer::Kind::Zero ||
      init.kind == Initializer::Kind::ScalarZero ||
      init.kind == Initializer::Kind::Null ||
      (init.kind == Initializer::Kind::Integer && init.integer.isZero());
  return value;
}

auto CxxIREmitter::isZeroConstant(ValueRef value) -> bool {
  return priv_->values.at(HandleAccess::id(value)).isZero;
}

auto CxxIREmitter::todo(SourceLocation loc, TodoKind kind,
                        std::string_view message) -> ValueRef {
  auto attrs = "message = " + quote(message);
  if (kind == TodoKind::Expression)
    return priv_->one(loc, "cxx.todo.expr", unresolvedType(), {}, {}, attrs);
  priv_->op(loc, "cxx.todo.stmt", {}, {}, {}, false, attrs);
  return {};
}

auto CxxIREmitter::unaryOp(SourceLocation loc, UnaryOp, TypeRef type,
                           ValueRef value) -> ValueRef {
  return priv_->one(loc, "arith.negf", type, {value});
}

auto CxxIREmitter::binaryOp(SourceLocation loc, BinaryOp op, ValueRef lhs,
                            ValueRef rhs) -> ValueRef {
  static constexpr std::string_view names[] = {
      "arith.addi",  "arith.subi",  "arith.muli",  "arith.addi",  "arith.subi",
      "arith.muli",  "arith.divsi", "arith.divui", "arith.remsi", "arith.remui",
      "arith.andi",  "arith.ori",   "arith.xori",  "arith.shli",  "arith.shrsi",
      "arith.shrui", "arith.addf",  "arith.subf",  "arith.mulf",  "arith.divf"};
  std::string attrs;
  if (op == BinaryOp::AddSignedInt || op == BinaryOp::SubSignedInt ||
      op == BinaryOp::MulSignedInt)
    attrs = "overflowFlags = #arith.overflow<nsw>";
  return priv_->one(loc, names[static_cast<int>(op)], typeOf(lhs), {lhs, rhs},
                    attrs);
}

auto CxxIREmitter::compareInt(SourceLocation loc, IntPredicate predicate,
                              ValueRef lhs, ValueRef rhs) -> ValueRef {
  auto result = integerType(1);
  const auto& operand = priv_->type(typeOf(lhs));
  if (operand.kind == TypeKind::Vector)
    result = vectorType(result, operand.count);
  return priv_->one(
      loc, "arith.cmpi", result, {lhs, rhs},
      std::format("predicate = {} : i64", static_cast<int>(predicate)));
}

auto CxxIREmitter::compareFloat(SourceLocation loc, FloatPredicate predicate,
                                ValueRef lhs, ValueRef rhs) -> ValueRef {
  static constexpr int predicates[] = {1, 6, 4, 5, 2, 3, 13, 14};
  auto result = integerType(1);
  const auto& operand = priv_->type(typeOf(lhs));
  if (operand.kind == TypeKind::Vector)
    result = vectorType(result, operand.count);
  return priv_->one(loc, "arith.cmpf", result, {lhs, rhs},
                    std::format("predicate = {} : i64",
                                predicates[static_cast<int>(predicate)]));
}

auto CxxIREmitter::select(SourceLocation loc, ValueRef condition,
                          ValueRef ifTrue, ValueRef ifFalse) -> ValueRef {
  return priv_->one(loc, "arith.select", typeOf(ifTrue),
                    {condition, ifTrue, ifFalse});
}

auto CxxIREmitter::convert(SourceLocation loc, CastKind kind, ValueRef value,
                           TypeRef type) -> ValueRef {
  static constexpr std::string_view names[] = {
      "arith.trunci",      "arith.extsi",  "arith.extui",
      "arith.extf",        "arith.truncf", "arith.sitofp",
      "arith.uitofp",      "arith.fptosi", "arith.fptoui",
      "arith.bitcast",     "cxx.bitcast",  "cxx.reshape",
      "cxx.array_to_ptr",  "cxx.ptrtoint", "cxx.inttoptr",
      "cxx.addrspace_cast"};
  return priv_->one(loc, names[static_cast<int>(kind)], type, {value});
}

auto CxxIREmitter::vectorSplat(SourceLocation loc, TypeRef type,
                               ValueRef scalar) -> ValueRef {
  return priv_->one(loc, "vector.broadcast", type, {scalar});
}

auto CxxIREmitter::vectorFromElements(SourceLocation loc, TypeRef type,
                                      std::span<const ValueRef> elements)
    -> ValueRef {
  return priv_->emit(loc, "vector.from_elements", std::array{type}, elements)
      .front();
}

auto CxxIREmitter::vectorExtract(SourceLocation loc, TypeRef type,
                                 ValueRef vector, ValueRef index,
                                 bool indexIsSigned) -> ValueRef {
  auto indexType = priv_->intern("index", {.text = "index"});
  auto position =
      priv_->one(loc, indexIsSigned ? "arith.index_cast" : "arith.index_castui",
                 indexType, {index});
  return priv_->one(loc, "vector.extract", type, {vector, position},
                    "static_position = array<i64: -9223372036854775808>");
}

auto CxxIREmitter::vectorInsert(SourceLocation loc, ValueRef vector,
                                ValueRef element, ValueRef index,
                                bool indexIsSigned) -> ValueRef {
  auto indexType = priv_->intern("index", {.text = "index"});
  auto position =
      priv_->one(loc, indexIsSigned ? "arith.index_cast" : "arith.index_castui",
                 indexType, {index});
  return priv_->one(loc, "vector.insert", typeOf(vector),
                    {element, vector, position},
                    "static_position = array<i64: -9223372036854775808>");
}

auto CxxIREmitter::vectorShuffle(SourceLocation loc, ValueRef first,
                                 ValueRef second,
                                 std::span<const std::int64_t> mask)
    -> ValueRef {
  auto type = vectorType(elementType(typeOf(first)), mask.size());
  return priv_->one(loc, "vector.shuffle", type, {first, second},
                    "mask = " + denseArray("i64", mask));
}

auto CxxIREmitter::allocate(SourceLocation loc, TypeRef type, ValueRef size,
                            std::uint64_t alignment) -> ValueRef {
  auto& p = *priv_;
  auto attrs = std::format("alignment = {} : i64", alignment);
  if (size) return p.one(loc, "cxx.dyn_alloca", type, {size}, {}, attrs);
  auto saved = p.point;
  if (auto block = insertionBlock(); block && p.block(block).function) {
    auto entry = p.function(p.block(block).function).blocks.front();
    setInsertionBlockStart(entry);
    while (p.point.next != p.point.operations->end() &&
           p.point.next->allocation)
      ++p.point.next;
  }
  auto value = p.one(loc, "cxx.alloca", type, {}, {}, attrs);
  std::prev(p.point.next)->allocation = true;
  p.point = saved;
  return value;
}

auto CxxIREmitter::allocateWorkgroup(SourceLocation loc, TypeRef type,
                                     std::uint64_t alignment) -> ValueRef {
  auto saved = priv_->point;
  if (auto block = insertionBlock(); block && priv_->block(block).function)
    setInsertionBlockStart(
        priv_->function(priv_->block(block).function).blocks.front());
  auto value = priv_->one(loc, "cxx.workgroup_alloc", type, {}, {},
                          std::format("alignment = {} : i64", alignment));
  priv_->point = saved;
  return value;
}

namespace {
[[nodiscard]] auto accessAttrs(const Access& access, bool load) -> std::string {
  std::string attrs;
  if (access.isVolatile) addAttr(attrs, "volatile_", "unit");
  if (access.bitfield) {
    addAttr(attrs, "alignment", std::format("{} : i64", access.alignment));
    const auto& field = *access.bitfield;
    addAttr(attrs, "bit_offset", std::format("{} : i32", field.bitOffset));
    addAttr(attrs, "bit_width", std::format("{} : i32", field.bitWidth));
    addAttr(attrs, "storage_size", std::format("{} : i64", field.storageSize));
    if (load) addAttr(attrs, "is_signed", access.isSigned ? "true" : "false");
  }
  return attrs;
}

[[nodiscard]] auto atomicAttrs(const AtomicAccess& access) -> std::string {
  auto attrs = "ordering = " + orderAttr(access.order) +
               ", scope = " + scopeAttr(access.scope);
  if (access.isVolatile) addAttr(attrs, "volatile_", "unit");
  return attrs;
}
}  // namespace

auto CxxIREmitter::load(SourceLocation loc, TypeRef type, ValueRef address,
                        const Access& access) -> ValueRef {
  return priv_->one(
      loc, access.bitfield ? "cxx.bitfield.load" : "cxx.load", type, {address},
      accessAttrs(access, true),
      access.bitfield ? ""
                      : std::format("alignment = {} : i64", access.alignment));
}

void CxxIREmitter::store(SourceLocation loc, ValueRef value, ValueRef address,
                         const Access& access) {
  priv_->op(loc, access.bitfield ? "cxx.bitfield.store" : "cxx.store",
            {value, address}, accessAttrs(access, false), {}, false,
            access.bitfield
                ? ""
                : std::format("alignment = {} : i64", access.alignment));
}

auto CxxIREmitter::atomicLoad(SourceLocation loc, TypeRef type,
                              ValueRef address, const AtomicAccess& access)
    -> ValueRef {
  return priv_->one(loc, "cxx.atomic.load", type, {address},
                    atomicAttrs(access));
}

void CxxIREmitter::atomicStore(SourceLocation loc, ValueRef value,
                               ValueRef address, const AtomicAccess& access) {
  priv_->op(loc, "cxx.atomic.store", {value, address}, atomicAttrs(access));
}

auto CxxIREmitter::atomicReadModifyWrite(SourceLocation loc,
                                         AtomicOperation operation,
                                         ValueRef value, ValueRef address,
                                         const AtomicAccess& access)
    -> ValueRef {
  static constexpr std::string_view names[] = {"exchange", "add", "sub", "and",
                                               "or",       "xor", "nand"};
  auto attrs = atomicAttrs(access);
  addAttr(attrs, "kind",
          std::format("#cxx.atomic_rmw_kind<{}>",
                      names[static_cast<int>(operation)]));
  return priv_->one(loc, "cxx.atomic.rmw", typeOf(value), {value, address},
                    attrs);
}

auto CxxIREmitter::atomicCompareExchange(SourceLocation loc, ValueRef address,
                                         ValueRef expected, ValueRef desired,
                                         const CompareExchangeAccess& access)
    -> std::vector<ValueRef> {
  auto attrs = "success_order = " + orderAttr(access.success) +
               ", failure_order = " + orderAttr(access.failure) +
               ", scope = " + scopeAttr(access.scope);
  if (access.isWeak) addAttr(attrs, "weak", "unit");
  if (access.isVolatile) addAttr(attrs, "volatile_", "unit");
  return priv_->emit(loc, "cxx.atomic.cmpxchg",
                     std::array{typeOf(expected), integerType(1)},
                     std::array{address, expected, desired}, attrs);
}

void CxxIREmitter::memoryBarrier(SourceLocation loc,
                                 const MemoryBarrierInfo& barrier) {
  priv_->op(loc, "cxx.atomic.fence", {},
            "ordering = " + orderAttr(barrier.order) +
                ", scope = " + scopeAttr(barrier.scope) +
                ", memory_regions = " + regionsAttr(barrier.regions));
}

void CxxIREmitter::controlBarrier(SourceLocation loc, MemoryScope scope,
                                  const MemoryBarrierInfo& barrier) {
  priv_->op(loc, "cxx.control_barrier", {},
            "execution_scope = " + scopeAttr(scope) +
                ", memory_scope = " + scopeAttr(barrier.scope) +
                ", ordering = " + orderAttr(barrier.order) +
                ", memory_regions = " + regionsAttr(barrier.regions));
}

auto CxxIREmitter::pointerAdd(SourceLocation loc, TypeRef type, ValueRef base,
                              ValueRef offset) -> ValueRef {
  return priv_->one(loc, "cxx.ptradd", type, {base, offset});
}

auto CxxIREmitter::pointerDiff(SourceLocation loc, TypeRef type, ValueRef lhs,
                               ValueRef rhs) -> ValueRef {
  return priv_->one(loc, "cxx.ptrdiff", type, {lhs, rhs});
}

auto CxxIREmitter::subscript(SourceLocation loc, TypeRef type, ValueRef base,
                             ValueRef index) -> ValueRef {
  return priv_->one(loc, "cxx.subscript", type, {base, index});
}

auto CxxIREmitter::memberAddress(SourceLocation loc, TypeRef type,
                                 ValueRef base, std::uint32_t index)
    -> ValueRef {
  return priv_->one(loc, "cxx.member", type, {base}, {},
                    std::format("member_index = {} : i32", index));
}

auto CxxIREmitter::extractValue(SourceLocation loc, TypeRef type,
                                ValueRef container, std::int64_t position)
    -> ValueRef {
  return priv_->one(loc, "cxx.extractvalue", type, {container},
                    std::format("position = {} : i64", position));
}

auto CxxIREmitter::insertValue(SourceLocation loc, TypeRef type,
                               ValueRef container, ValueRef value,
                               std::int64_t position) -> ValueRef {
  return priv_->one(loc, "cxx.insertvalue", type, {container, value},
                    std::format("position = {} : i64", position));
}

auto CxxIREmitter::addressOfSymbol(SourceLocation loc, TypeRef type,
                                   std::string_view symbol) -> ValueRef {
  return priv_->one(loc, "cxx.address_of", type, {},
                    "sym_name = @" + quote(symbol));
}

void CxxIREmitter::memsetZero(SourceLocation loc, ValueRef address,
                              std::uint64_t size) {
  priv_->op(loc, "cxx.memzero", {address}, {}, {}, false,
            std::format("size = {} : i64", size));
}

void CxxIREmitter::memcpy(SourceLocation loc, ValueRef destination,
                          ValueRef source, std::uint64_t size) {
  priv_->op(loc, "cxx.memcpy", {destination, source}, {}, {}, false,
            std::format("size = {} : i64", size));
}

void CxxIREmitter::zeroInitialize(SourceLocation loc, ValueRef address,
                                  TypeRef type) {
  auto pointer = rebindPointerType(typeOf(address), type);
  if (pointer != typeOf(address)) address = bitcast(loc, pointer, address);
  priv_->op(loc, "cxx.zero_init", {address});
}

void CxxIREmitter::copyObject(SourceLocation loc, ValueRef destination,
                              ValueRef source, TypeRef type) {
  auto destinationType = rebindPointerType(typeOf(destination), type);
  auto sourceType = rebindPointerType(typeOf(source), type);
  if (destinationType != typeOf(destination))
    destination = bitcast(loc, destinationType, destination);
  if (sourceType != typeOf(source)) source = bitcast(loc, sourceType, source);
  priv_->op(loc, "cxx.copy", {destination, source});
}

auto CxxIREmitter::saveInsertionPoint() -> InsertionPointRef {
  auto& p = *priv_;
  std::uint32_t id;
  if (p.freePoints.empty()) {
    id = p.points.size();
    p.points.push_back(p.point);
  } else {
    id = p.freePoints.back();
    p.freePoints.pop_back();
    p.points[id] = p.point;
  }
  return HandleAccess::make<InsertionPointTag>(id);
}

void CxxIREmitter::restoreInsertionPoint(InsertionPointRef ref) {
  auto& p = *priv_;
  auto id = HandleAccess::id(ref);
  p.point = p.points.at(id);
  p.points[id] = {};
  p.freePoints.push_back(id);
}

void CxxIREmitter::setInsertionPoint(InsertionPoint point) {
  auto& p = *priv_;
  bool module = point.kind == InsertionPoint::Kind::ModuleStart ||
                point.kind == InsertionPoint::Kind::ModuleEnd;
  bool start = point.kind == InsertionPoint::Kind::ModuleStart ||
               point.kind == InsertionPoint::Kind::BlockStart;
  auto& operations =
      module ? p.module().operations : p.block(point.block).operations;
  p.point = {&operations, start ? operations.begin() : operations.end(),
             module ? BlockRef{} : point.block};
}

auto CxxIREmitter::insertionBlock() -> BlockRef { return priv_->point.block; }

void CxxIREmitter::beginFunctionBody(FunctionRef function) {
  priv_->functionBodies.push_back(function);
}

void CxxIREmitter::endFunctionBody(FunctionRef function) {
  assert(!priv_->functionBodies.empty() &&
         priv_->functionBodies.back() == function);
  priv_->functionBodies.pop_back();
}

auto CxxIREmitter::createBlock(FunctionRef function) -> BlockRef {
  auto& p = *priv_;
  GlobalRef global;
  if (!function && insertionBlock()) {
    auto& owner = p.block(insertionBlock());
    function = owner.function;
    global = owner.global;
  }
  if (!function && !global && !p.functionBodies.empty())
    function = p.functionBodies.back();
  assert((function || global) && "a block needs an owning region");
  auto ref = HandleAccess::make<BlockTag>(p.blocks.size());
  auto block = std::make_unique<Private::Block>();
  block->function = function;
  block->global = global;
  p.blocks.push_back(std::move(block));
  if (function)
    p.function(function).blocks.push_back(ref);
  else
    p.global(global).blocks.push_back(ref);
  return ref;
}

void CxxIREmitter::eraseBlock(BlockRef block) {
  priv_->block(block).erased = true;
}

auto CxxIREmitter::hasTerminator(BlockRef block) -> bool {
  auto& operations = priv_->block(block).operations;
  return !operations.empty() && operations.back().terminator;
}

auto CxxIREmitter::addBlockParameter(BlockRef block, TypeRef type,
                                     SourceLocation loc) -> ValueRef {
  auto value = priv_->newValue(type);
  priv_->values[HandleAccess::id(value)].location = priv_->sourceLocation(loc);
  priv_->block(block).parameters.push_back(value);
  return value;
}

auto CxxIREmitter::blockParameter(BlockRef block, unsigned index) -> ValueRef {
  return priv_->block(block).parameters.at(index);
}

auto CxxIREmitter::blockParameterCount(BlockRef block) -> unsigned {
  return priv_->block(block).parameters.size();
}

void CxxIREmitter::branch(SourceLocation loc, BlockRef target,
                          std::span<const ValueRef> operands) {
  (void)priv_->emit(loc, "cf.br", {}, operands, {}, std::array{target}, true);
}

void CxxIREmitter::condBranch(SourceLocation loc, ValueRef condition,
                              BlockRef trueDest, BlockRef falseDest) {
  priv_->op(loc, "cf.cond_br", {condition},
            "operandSegmentSizes = array<i32: 1, 0, 0>", {trueDest, falseDest},
            true);
}

void CxxIREmitter::switchBranch(SourceLocation loc, ValueRef flag,
                                BlockRef defaultDest,
                                std::span<const ConstInt> caseValues,
                                std::span<const BlockRef> caseDestinations) {
  assert(caseValues.size() == caseDestinations.size());
  for (std::size_t i = 0; i < caseValues.size(); ++i) {
    auto next = i + 1 == caseValues.size() ? defaultDest : createBlock({});
    auto value = constantInt(loc, typeOf(flag), caseValues[i]);
    auto matches = compareInt(loc, IntPredicate::Equal, flag, value);
    condBranch(loc, matches, caseDestinations[i], next);
    if (next != defaultDest) setInsertionBlock(next);
  }
  if (caseValues.empty()) branch(loc, defaultDest);
}

void CxxIREmitter::ret(SourceLocation loc, std::span<const ValueRef> values) {
  (void)priv_->emit(loc, "cxx.return", {}, values, {}, {}, true);
}

void CxxIREmitter::unreachable(SourceLocation loc) {
  priv_->op(loc, "cxx.unreachable", {}, {}, {}, true);
}

auto CxxIREmitter::findFunction(std::string_view name) -> FunctionRef {
  auto& p = *priv_;
  if (!p.currentModule) return {};
  auto it = p.module().functions.find(std::string(name));
  return it == p.module().functions.end() ? FunctionRef{} : it->second;
}

auto CxxIREmitter::declareFunction(SourceLocation loc, const FunctionInfo& info)
    -> FunctionRef {
  if (auto ref = findFunction(info.name)) return ref;
  auto& p = *priv_;
  auto function = std::make_unique<Private::Function>();
  function->name = info.name;
  function->type = info.type;
  function->location = p.sourceLocation(loc);
  function->aliasee = info.aliasee;
  auto& attrs = function->attrs;
  addAttr(attrs, "sym_name", quote(info.name));
  addAttr(attrs, "function_type", p.typeText(info.type));
  addAttr(attrs, "linkage_kind", linkageAttr(info.linkage));
  if (info.inlineKind != InlineKind::Default) {
    static constexpr std::string_view kinds[] = {"", "no_inline", "inline",
                                                 "always_inline"};
    addAttr(attrs, "inline_kind",
            std::format("#cxx.inline_kind<{}>",
                        kinds[static_cast<int>(info.inlineKind)]));
  }
  if (info.visibility != Visibility::Default)
    addAttr(attrs, "visibility_",
            info.visibility == Visibility::Hidden
                ? "#cxx.visibility<hidden>"
                : "#cxx.visibility<protected>");
  addStringAttr(attrs, "wasm_import_module", info.importModule);
  addStringAttr(attrs, "wasm_import_name", info.importName);
  addStringAttr(attrs, "wasm_export_name", info.exportName);
  addStringAttr(attrs, "section", info.section);
  if (info.isUsed) addAttr(attrs, "used", "unit");
  const auto& type = p.type(info.type);
  p.signatureAttrs(attrs, info.parameters, type.parameters.size(),
                   info.resultAbi, type.results.size(), info.callingConvention);
  if (info.kernel) {
    const auto& k = *info.kernel;
    addAttr(attrs, "cxx.kernel_workgroup_size",
            denseArray("i32", std::array{k.x, k.y, k.z}));
    auto buffers = join(k.buffers, [&](const auto& buffer) {
      return std::format(
          "{{parameter = {} : i32, binding = {} : i32, kind = {}, read_only = "
          "{}, name = {}, unsigned = {}}}",
          buffer.parameter, buffer.binding,
          quote(buffer.kind == KernelBufferKind::Storage ? "storage"
                                                         : "uniform"),
          buffer.readOnly, quote(buffer.name), buffer.unsignedElements);
    });
    auto ids = join(k.invocationIds, [&](const auto& id) {
      static constexpr std::string_view kinds[] = {"global", "local",
                                                   "workgroup"};
      return std::format("{{parameter = {} : i32, kind = {}}}", id.parameter,
                         quote(kinds[static_cast<int>(id.kind)]));
    });
    addAttr(attrs, "cxx.kernel_buffers", "[" + buffers + "]");
    addAttr(attrs, "cxx.kernel_invocation_ids", "[" + ids + "]");
  }
  auto ref = HandleAccess::make<FunctionTag>(p.functions.size());
  p.functions.push_back(std::move(function));
  p.module().functions.emplace(info.name, ref);
  p.module().symbols.emplace(info.name);
  p.appendModule({.function = ref});
  return ref;
}

auto CxxIREmitter::functionParameterTypes(FunctionRef ref)
    -> std::vector<TypeRef> {
  return priv_->type(priv_->function(ref).type).parameters;
}

auto CxxIREmitter::functionResultTypes(FunctionRef ref)
    -> std::vector<TypeRef> {
  return priv_->type(priv_->function(ref).type).results;
}

auto CxxIREmitter::functionHasBody(FunctionRef ref) -> bool {
  return !priv_->function(ref).blocks.empty();
}

void CxxIREmitter::setFunctionAliasee(FunctionRef ref,
                                      std::string_view aliasee) {
  priv_->function(ref).aliasee = aliasee;
}

auto CxxIREmitter::findGlobal(std::string_view name) -> GlobalRef {
  auto& p = *priv_;
  if (!p.currentModule) return {};
  auto it = p.module().globals.find(std::string(name));
  return it == p.module().globals.end() ? GlobalRef{} : it->second;
}

auto CxxIREmitter::declareGlobal(SourceLocation loc, const GlobalInfo& info)
    -> GlobalRef {
  if (auto ref = findGlobal(info.name)) return ref;
  auto& p = *priv_;
  auto global = std::make_unique<Private::Global>();
  global->linkage = info.linkage;
  global->location =
      info.unknownLocation ? SourceLocation{} : p.sourceLocation(loc);
  auto& attrs = global->attrs;
  addAttr(attrs, "sym_name", quote(info.name));
  addAttr(attrs, "global_type", p.typeText(info.type));
  addAttr(attrs, "linkage_kind", linkageAttr(info.linkage));
  if (info.isConstant) addAttr(attrs, "constant", "unit");
  if (info.isUsed) addAttr(attrs, "used", "unit");
  if (info.isThreadLocal) addAttr(attrs, "thread_local_", "unit");
  if (info.alignment)
    addAttr(attrs, "alignment", std::format("{} : i64", info.alignment));
  if (info.addressSpace)
    addAttr(attrs, "address_space",
            std::format("{} : i32", *info.addressSpace));
  addStringAttr(attrs, "section", info.section);
  if (auto value = p.initializer(info.initializer, info.type); !value.empty())
    addAttr(attrs, "value", value);
  auto ref = HandleAccess::make<GlobalTag>(p.globals.size());
  p.globals.push_back(std::move(global));
  p.module().globals.emplace(info.name, ref);
  p.module().symbols.emplace(info.name);
  p.appendModule({.global = ref});
  return ref;
}

auto CxxIREmitter::globalLinkage(GlobalRef ref) -> Linkage {
  return priv_->global(ref).linkage;
}

auto CxxIREmitter::symbolExists(std::string_view name) -> bool {
  return priv_->currentModule &&
         priv_->module().symbols.contains(std::string(name));
}

void CxxIREmitter::beginGlobalInitializer(GlobalRef ref) {
  auto& p = *priv_;
  auto block = std::make_unique<Private::Block>();
  block->global = ref;
  auto b = HandleAccess::make<BlockTag>(p.blocks.size());
  p.blocks.push_back(std::move(block));
  p.global(ref).blocks.push_back(b);
  setInsertionBlock(b);
}

void CxxIREmitter::globalConstructor(SourceLocation loc, FunctionRef ref,
                                     std::uint32_t priority) {
  priv_->appendModule({.name = "cxx.global_ctor",
                       .attrs = "ctor = @" + quote(priv_->function(ref).name) +
                                std::format(", priority = {} : i32", priority),
                       .location = priv_->sourceLocation(loc)});
}

void CxxIREmitter::defineVTable(SourceLocation loc, const VTableInfo& info) {
  auto& p = *priv_;
  std::vector<std::string> offsets, tops, slots;
  for (const auto& table : info.tables) {
    offsets.push_back(
        "[" +
        join(table.offsets,
             [](auto offset) { return std::format("{} : i64", offset); }) +
        "]");
    tops.push_back(std::format("{} : i64", table.offsetToTop));
    slots.push_back("[" +
                    join(table.slots,
                         [&](auto f) {
                           return f ? "@" + quote(p.function(f).name) : "unit";
                         }) +
                    "]");
  }
  auto list = [](const auto& entries) {
    return "[" + join(entries, [](const auto& text) { return text; }) + "]";
  };
  auto attrs = "sym_name = " + quote(info.name) +
               ", offsets = " + list(offsets) +
               ", offsets_to_top = " + list(tops) + ", slots = " + list(slots) +
               ", linkage_kind = " + linkageAttr(info.linkage);
  if (!info.typeInfo.empty())
    addAttr(attrs, "type_info", "@" + quote(info.typeInfo));
  p.module().symbols.emplace(info.name);
  p.appendModule({.name = "cxx.vtable",
                  .attrs = std::move(attrs),
                  .location = p.sourceLocation(loc)});
}

auto CxxIREmitter::call(SourceLocation loc, const CallInfo& info)
    -> std::vector<ValueRef> {
  auto& p = *priv_;
  if (info.kind == CallKind::Builtin)
    return p.emit(loc, "cxx.builtin.call", info.results, info.arguments, {}, {},
                  false, "builtin_name = " + quote(info.callee));
  std::vector<ValueRef> operands;
  if (info.indirectCallee) operands.push_back(info.indirectCallee);
  operands.insert(operands.end(), info.arguments.begin(), info.arguments.end());
  auto attrs = std::format("operandSegmentSizes = array<i32: {}, {}>",
                           info.indirectCallee ? 1 : 0, info.arguments.size());
  if (!info.indirectCallee) addAttr(attrs, "callee", "@" + quote(info.callee));
  if (info.variadicCalleeType)
    addAttr(attrs, "var_callee_type", p.typeText(info.variadicCalleeType));
  p.signatureAttrs(attrs, info.parameters, info.arguments.size(),
                   info.resultAbi, info.results.size(), info.callingConvention);
  if (info.unwindDestination)
    return p.emit(loc, "cxx.invoke", info.results, operands, attrs,
                  std::array{info.normalDestination, info.unwindDestination},
                  true);
  return p.emit(loc, "cxx.call", info.results, operands, attrs);
}

void CxxIREmitter::functionPersonality(FunctionRef ref,
                                       std::string_view personality) {
  priv_->function(ref).personality = personality;
}

auto CxxIREmitter::landingPad(SourceLocation loc,
                              std::span<const ValueRef> clauses, bool cleanup)
    -> std::vector<ValueRef> {
  return priv_->emit(loc, "cxx.landing_pad",
                     std::array{pointerType(integerType(8)), integerType(32)},
                     clauses, cleanup ? "cleanup = unit" : "");
}

auto CxxIREmitter::exceptionTypeId(SourceLocation loc, ValueRef typeInfo)
    -> ValueRef {
  return priv_->one(loc, "cxx.exception_typeid", integerType(32), {typeInfo});
}

void CxxIREmitter::resumeException(SourceLocation loc, ValueRef exception,
                                   ValueRef selector) {
  priv_->op(loc, "cxx.resume", {exception, selector}, {}, {}, true);
}

void CxxIREmitter::moduleAsm(SourceLocation loc, std::string_view assembly) {
  priv_->appendModule({.name = "cxx.module_asm",
                       .attrs = "assembly = " + quote(assembly),
                       .location = priv_->sourceLocation(loc)});
}

auto CxxIREmitter::inlineAsm(SourceLocation loc, const InlineAsmInfo& info)
    -> std::vector<ValueRef> {
  auto& p = *priv_;
  auto attrs = "asm_string = " + quote(info.asmString) +
               ", constraints = " + quote(info.constraints);
  if (info.hasSideEffects) addAttr(attrs, "has_side_effects", "unit");
  addAttr(attrs, "argument_element_types",
          "[" + join(info.argumentElementTypes, [&](auto type) {
            return type ? p.typeText(type) : "unit";
          }) + "]");
  return p.emit(loc, "cxx.inline_asm", info.results, info.arguments, attrs);
}

auto CxxIREmitter::inlineAsmGoto(SourceLocation loc, const InlineAsmInfo& info,
                                 BlockRef fallthrough,
                                 std::span<const BlockRef> destinations)
    -> ValueRef {
  auto& p = *priv_;
  auto attrs = "asm_string = " + quote(info.asmString) +
               ", constraints = " + quote(info.constraints);
  addAttr(attrs, "argument_element_types",
          "[" + join(info.argumentElementTypes, [&](auto type) {
            return type ? p.typeText(type) : "unit";
          }) + "]");
  std::vector<BlockRef> successors{fallthrough};
  successors.insert(successors.end(), destinations.begin(), destinations.end());
  std::vector<TypeRef> results;
  if (info.results.size() == 1) results.push_back(info.results.front());
  if (info.results.size() > 1) {
    auto key =
        "tuple:" + join(info.results, [&](auto t) { return p.typeText(t); });
    Private::Type tuple;
    tuple.members.assign(info.results.begin(), info.results.end());
    results.push_back(p.intern(key, std::move(tuple)));
  }
  return singleValue(p.emit(loc, "cxx.asm_goto", results, info.arguments, attrs,
                            successors, true));
}

auto CxxIREmitter::beginCleanupRegion() -> CleanupRegionRef {
  auto& p = *priv_;
  auto ref = HandleAccess::make<CleanupRegionTag>(p.nextCleanupRegion++);
  Private::CleanupRegion region{.block = insertionBlock()};
  if (p.point.operations) {
    auto it = p.point.next;
    while (it != p.point.operations->begin()) {
      --it;
      if (it->allocation) continue;
      region.anchor = &*it;
      break;
    }
  }
  p.cleanupRegions.emplace(ref, region);
  return ref;
}

void CxxIREmitter::endCleanupRegion(CleanupRegionRef ref) {
  priv_->cleanupRegions.erase(ref);
}

auto CxxIREmitter::activateConditionalCleanup(ValueRef, BlockRef entry,
                                              CleanupRegionRef ref)
    -> ValueRef {
  auto& p = *priv_;
  auto saved = p.point;
  setInsertionBlockStart(entry);
  auto boolean = integerType(1);
  auto flag = allocate({}, pointerType(boolean), 1);
  auto region = p.cleanupRegions.at(ref);
  setInsertionBlockStart(region.block ? region.block : entry);
  if (region.anchor) {
    p.point.next = std::ranges::find_if(
        *p.point.operations, [&](auto& op) { return &op == region.anchor; });
    assert(p.point.next != p.point.operations->end());
    ++p.point.next;
  } else {
    while (p.point.next != p.point.operations->end() &&
           p.point.next->allocation)
      ++p.point.next;
  }
  auto inactive = constantInt({}, boolean, 0);
  store({}, inactive, flag, 1);
  p.point = saved;
  auto active = constantInt({}, boolean, 1);
  store({}, active, flag, 1);
  return flag;
}

void CxxIREmitter::defineLabel(SourceLocation loc, std::string_view name,
                               std::int64_t depth) {
  auto& p = *priv_;
  p.function(p.block(insertionBlock()).function).labels[std::string(name)] = {
      insertionBlock(), depth, p.sourceLocation(loc)};
}

void CxxIREmitter::branchWithCleanups(SourceLocation loc, CleanupTarget target,
                                      std::span<const CleanupAction> cleanups) {
  auto& p = *priv_;
  if (!target.block) {
    Private::Operation op;
    op.location = p.sourceLocation(loc);
    op.gotoLabel = target.label;
    op.cleanups.assign(cleanups.begin(), cleanups.end());
    p.point.operations->insert(p.point.next, std::move(op));
    return;
  }
  if (cleanups.empty()) {
    branch(loc, target.block);
    return;
  }
  std::vector<ValueRef> addresses, flags, initialized;
  std::vector<std::string> destructors;
  std::vector<std::int32_t> flagIndices, initializedIndices;
  std::vector<std::int64_t> counts;
  for (const auto& action : cleanups) {
    addresses.push_back(action.address);
    destructors.push_back("@" + quote(p.function(action.destructor).name));
    counts.push_back(action.elementCount);
    flagIndices.push_back(
        action.activeFlag ? static_cast<std::int32_t>(flags.size()) : -1);
    initializedIndices.push_back(
        action.initializedCount ? static_cast<std::int32_t>(initialized.size())
                                : -1);
    if (action.activeFlag) flags.push_back(action.activeFlag);
    if (action.initializedCount) initialized.push_back(action.initializedCount);
  }
  auto attrs = "destructors = [" +
               join(destructors, [](const auto& text) { return text; }) + "]";
  addAttr(attrs, "activeFlagIndices", denseArray("i32", flagIndices));
  addAttr(attrs, "initializedCountIndices",
          denseArray("i32", initializedIndices));
  addAttr(attrs, "elementCounts", denseArray("i64", counts));
  addAttr(attrs, "operandSegmentSizes",
          denseArray("i32", std::array{addresses.size(), flags.size(),
                                       initialized.size()}));
  addresses.insert(addresses.end(), flags.begin(), flags.end());
  addresses.insert(addresses.end(), initialized.begin(), initialized.end());
  (void)p.emit(loc, "cxx.cleanup_branch", {}, addresses, attrs,
               std::array{target.block}, true);
}

void CxxIREmitter::indirectGoto(SourceLocation loc, ValueRef target) {
  priv_->op(loc, "cxx.indirect_goto", {target}, {}, {}, true);
  std::prev(priv_->point.next)->indirectTarget = target;
}

auto CxxIREmitter::labelAddress(SourceLocation loc, TypeRef type,
                                std::string_view name, FunctionRef function)
    -> ValueRef {
  std::string attrs;
  if (function) {
    addAttr(attrs, "func_name", quote(priv_->function(function).name));
    auto& labels = priv_->function(function).addressLabels;
    auto it = std::ranges::find(labels, name);
    auto tag = std::distance(labels.begin(), it);
    if (it == labels.end()) labels.emplace_back(name);
    addAttr(attrs, "tag_id", std::format("{} : i32", tag));
  }
  return priv_->one(loc, "cxx.label_address", type, {}, attrs,
                    "label_name = " + quote(name));
}

void CxxIREmitter::resolveFunctionControlFlow(FunctionRef ref) {
  auto& p = *priv_;
  auto saved = p.point;
  auto& fn = p.function(ref);
  std::vector<BlockRef> labels;
  for (std::size_t tag = 0; tag < fn.addressLabels.size(); ++tag) {
    auto label = fn.labels.find(fn.addressLabels[tag]);
    assert(label != fn.labels.end() && "undefined address-taken label");
    auto block = label->second.block;
    labels.push_back(block);
    auto& operations = p.block(block).operations;
    operations.push_front(
        {.name = "llvm.blocktag",
         .attrs = std::format("tag = #llvm.blocktag<id = {}>", tag),
         .location = label->second.location});
  }
  for (auto block : fn.blocks) {
    auto& operations = p.block(block).operations;
    for (auto it = operations.begin(); it != operations.end(); ++it) {
      if (it->indirectTarget) {
        it->successors = labels;
      }
      if (it->gotoLabel.empty()) continue;
      auto label = fn.labels.find(it->gotoLabel);
      assert(label != fn.labels.end() && "undefined goto label");
      std::vector<CleanupAction> cleanups;
      for (const auto& action : it->cleanups)
        if (action.depth >= label->second.depth) cleanups.push_back(action);
      p.point = {&operations, it, block};
      branchWithCleanups(it->location,
                         CleanupTarget{.block = label->second.block}, cleanups);
      operations.erase(it, operations.end());
      break;
    }
  }
  std::unordered_set<BlockRef> reachable;
  std::vector<BlockRef> work;
  if (!fn.blocks.empty()) work.push_back(fn.blocks.front());
  while (!work.empty()) {
    auto block = work.back();
    work.pop_back();
    if (!reachable.insert(block).second || p.block(block).erased) continue;
    for (const auto& op : p.block(block).operations)
      work.insert(work.end(), op.successors.begin(), op.successors.end());
  }
  for (auto block : fn.blocks)
    if (!reachable.contains(block)) p.block(block).erased = true;
  p.point = saved;
}

}  // namespace cxx::ir
