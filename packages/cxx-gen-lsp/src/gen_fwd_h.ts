// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

import * as path from "node:path";
import { getEnumBaseType, MetaModel, toCppType, Type } from "./MetaModel.js";
import { writeFileSync } from "node:fs";
import { copyrightHeader } from "./copyrightHeader.js";

const fragment = `
[[noreturn]] void lsp_runtime_error(const std::string& msg);

using LSPAny = json;
using Pattern = std::string;

class LSPObject {
  public:
    LSPObject() = default;
    explicit LSPObject(json& repr): repr_(&repr) {}

    [[nodiscard]] explicit operator bool() const { return repr_ != nullptr; }
    [[nodiscard]] operator const json&() const { return *repr_; }
    [[nodiscard]] auto get() const -> json& { return *repr_; }

  protected:
    json* repr_{nullptr};
};

template <typename T>
class Vector final : public LSPObject {
public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const { return repr_ && repr_->is_array(); }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }
  [[nodiscard]] auto at(int index) const -> const T& { return repr_->at(index); }
};

namespace details {

template <typename T>
struct TryEmplace;

template <std::derived_from<LSPObject> T>
struct TryEmplace<T> {
  auto operator()(auto& result, json& value) const -> bool {
    auto obj = T{value};
    if (!obj) return false;
    result.template emplace<T>(std::move(obj));
    return true;
  }
};

template <typename... Ts>
auto try_emplace(std::variant<Ts...>& result, json& value) -> bool {
  return (details::TryEmplace<Ts>{}(result, value) || ...);
}

template <>
struct TryEmplace<std::nullptr_t> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_null()) return false;
    result.template emplace<std::nullptr_t>(nullptr);
    return true;
  }
};

template <>
struct TryEmplace<bool> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_boolean()) return false;
    result.template emplace<bool>(value);
    return true;
  }
};

template <>
struct TryEmplace<int> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_number_integer()) return false;
    result.template emplace<int>(value);
    return true;
  }
};

template <>
struct TryEmplace<long> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_number_integer()) return false;
    result.template emplace<long>(value);
    return true;
  }
};

template <>
struct TryEmplace<double> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_number_float()) return false;
    result.template emplace<double>(value);
    return true;
  }
};

template <>
struct TryEmplace<std::string> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_string()) return false;
    result.template emplace<std::string>(value);
    return true;
  }
};

template <typename... Ts>
struct TryEmplace<std::variant<Ts...>> {
  auto operator()(auto& result, json& value) const -> bool {
    return try_emplace(result, value);
  }
};

template <typename... Ts>
struct TryEmplace<std::tuple<Ts...>> {
  auto operator()(auto& result, json& value) const -> bool {
    lsp_runtime_error("todo: TryEmplace<std::tuple<Ts...>>");
    return false;
  }
};

template <>
struct TryEmplace<json> {
  auto operator()(auto& result, json& value) const -> bool {
    result = value;
    return true;
  }
};

template <>
struct TryEmplace<TextDocumentSyncKind> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_number_integer()) return false;
    result = TextDocumentSyncKind(value.get<int>());
    return true;
  }
};

}  // namespace details

template <typename... Ts>
class Vector<std::variant<Ts...>> final : public LSPObject {
  public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const { return repr_ && repr_->is_array(); }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }
  [[nodiscard]] auto at(int index) const -> std::variant<Ts...> {
    std::variant<Ts...> result;
    details::try_emplace(result, repr_->at(index));
    return result;
  }
};

template <typename Key, typename Value>
class Map final : public LSPObject {
public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const { return repr_ && repr_->is_object(); }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }
  [[nodiscard]] auto at(const Key& key) const -> const Value& { return repr_->at(key); }
};

template <typename Key, typename... Ts>
class Map<Key, std::variant<Ts...>> final : public LSPObject {
  public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const { return repr_ && repr_->is_object(); }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }

  [[nodiscard]] auto at(const Key& key) const -> std::variant<Ts...> {
    std::variant<Ts...> result;
    details::try_emplace(result, repr_->at(key));
    return result;
  }
};
`;

function dependenciesHelper(type: Type, deps: Set<string>) {
  switch (type.kind) {
    case "base":
      break;
    case "reference":
      deps.add(type.name);
      break;
    case "array":
      dependenciesHelper(type.element, deps);
      break;
    case "map":
      dependenciesHelper(type.key, deps);
      dependenciesHelper(type.value, deps);
      break;
    case "tuple":
      type.items.forEach((item) => dependenciesHelper(item, deps));
      break;
    case "or":
      type.items.forEach((item) => dependenciesHelper(item, deps));
      break;
    case "and":
      type.items.forEach((item) => dependenciesHelper(item, deps));
      break;
    default:
      throw new Error(`Unknown type kind: ${JSON.stringify(type)}`);
  } // switch
}

function dependencies(type: Type): Set<string> {
  const deps = new Set<string>();
  dependenciesHelper(type, deps);
  if (type.kind === "reference") {
    deps.delete(type.name);
  }
  return deps;
}

export function gen_fwd_h({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
  let out = "";

  const emit = (s: string = "") => {
    out += `${s}\n`;
  };

  emit(copyrightHeader);
  emit();
  emit(`#pragma once`);
  emit();
  emit(`#include <nlohmann/json.hpp>`);
  emit(`#include <variant>`);
  emit();
  emit(`namespace cxx::lsp {`);
  emit();
  emit(`using json = nlohmann::json;`);
  emit();
  model.enumerations.forEach((enumeration) => {
    const enumBaseType = getEnumBaseType(enumeration);
    emit(`enum class ${enumeration.name}${enumBaseType};`);
  });
  emit();
  model.structures.forEach((structure) => {
    emit(`class ${structure.name};`);
  });
  emit();

  const knownTypes = new Set<string>();
  model.enumerations.forEach((enumeration) => knownTypes.add(enumeration.name));
  model.structures.forEach((structure) => knownTypes.add(structure.name));
  // add builtins
  knownTypes.add("LSPAny");
  knownTypes.add("LSPObject");
  knownTypes.add("LSPArray");
  knownTypes.add("Pattern");
  knownTypes.add("DocumentFilter");
  knownTypes.add("TextDocumentFilter");

  const todo = model.typeAliases.map((typeAlias) => typeAlias);

  const builtinTypes = new Set(["LSPAny", "LSPObject", "LSPArray", "Pattern"]);

  emit(fragment);

  while (todo.length > 0) {
    const typeAlias = todo.pop()!;
    const deps = Array.from(dependencies(typeAlias.type));
    const hasUnknownDeps = deps.some((dep) => !knownTypes.has(dep));
    if (hasUnknownDeps) {
      const unknownDeps = deps.filter((dep) => !knownTypes.has(dep));
      console.log(`unknown ${unknownDeps} for type alias ${typeAlias.name}`);
      todo.unshift(typeAlias);
      continue;
    }

    if (builtinTypes.has(typeAlias.name)) {
      // skip builtins
      continue;
    }

    const cppType = toCppType(typeAlias.type);
    emit();
    emit(`using ${typeAlias.name} = ${cppType};`);
  }
  emit();
  emit(`}`);

  const outputFile = path.join(outputDirectory, "fwd.h");
  writeFileSync(outputFile, out);
}
