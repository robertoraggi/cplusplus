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

import * as fs from "node:fs";
import { spawnSync } from "node:child_process";
import {
  ModelIndex,
  type ModelClass,
  type ModelType,
  substituteType,
  unqualified,
  typeArguments,
} from "./parseModel.ts";
import { cpy_header } from "./cpy_header.ts";

const short = (s: string) => s.replace(/^::cxx::/, "").replaceAll("::", "_");

type Channel = "num" | "big" | "str" | "val";

type Decode = (value: string) => string;

type Wire = {
  channel: Channel;
  cpp: string;
  ts: string;
  decode?: Decode | undefined;
};

type Reader = { name: string; type: ModelType; expression: string };

const kindedRoots = ["AST", "Symbol", "Type", "Name"];
const modelRoots = [...kindedRoots, "Literal"];
const families = [...modelRoots, "Misc"];
const rootlessClasses = [
  "Meta",
  "InitializerList",
  "ConstComplex",
  "ConstObject",
  "ConstAddress",
  "ConstLabelAddress",
  "DefaultInitializerContext",
];

const wrapperOf: Record<string, string> = {
  AST: "astOf",
  Symbol: "symbolOf",
  Type: "typeOf",
  Name: "nameOf",
};

const enumeratorOf: Record<string, (name: string) => string> = {
  AST: (name) => name.replace(/AST$/, ""),
  Symbol: (name) => `k${name.replace(/Symbol$/, "")}`,
  Type: (name) => `k${name.replace(/Type$/, "")}`,
  Name: (name) => `k${name}`,
};

const containerClasses = ["::std::vector", "::std::span", "::std::deque"];

const channelSuffix: Record<Channel, string> = {
  num: "",
  big: "BigInt",
  str: "String",
  val: "Val",
};

const itemChannel: Record<Channel, string> = {
  num: "item",
  big: "itemBig",
  str: "itemStr",
  val: "itemVal",
};

function peel(type: ModelType): ModelType {
  switch (type.kind) {
    case "qual":
    case "lvalue-reference":
    case "rvalue-reference":
      return peel(type.element);
    default:
      return type;
  }
}

export function gen_reflection(index: ModelIndex, root: string) {
  const rootOf = (entry: ModelClass): string | undefined => {
    if (modelRoots.includes(entry.unqualifiedName))
      return entry.unqualifiedName;
    for (const base of entry.bases) {
      const parent = index.classOf(base.type);
      if (parent) {
        const found = rootOf(parent);
        if (found) return found;
      }
    }
  };

  const classes = index.model.classes.filter(
    (entry) =>
      !entry.isTemplate &&
      (rootOf(entry) || rootlessClasses.includes(entry.unqualifiedName)),
  );
  const classNames = new Set(classes.map((entry) => entry.name));
  const familyOf = (entry: ModelClass) => rootOf(entry) ?? "Misc";
  const isKinded = (entry: ModelClass) =>
    kindedRoots.includes(rootOf(entry) ?? "");
  const familyKey = (family: string) => family.toLowerCase();

  const enumNames = new Set<string>(
    kindedRoots.map((family) => `::cxx::${family}Kind`),
  );
  const report: string[] = [];
  const usedItemHelpers = new Map<
    string,
    { family: string; channel: Channel }
  >();

  const handleCpp = (target: ModelClass, expr: string) => {
    const family = rootOf(target);
    const pointer = family
      ? `static_cast<const ::cxx::${family}*>(${expr})`
      : expr;
    return `static_cast<double>(reinterpret_cast<std::intptr_t>(${pointer}))`;
  };

  const selfCpp = (entry: ModelClass) => {
    const family = rootOf(entry);
    const handle = `reinterpret_cast<const ::cxx::${family ?? entry.unqualifiedName}*>(handle)`;
    if (!family || entry.unqualifiedName === family)
      return `reinterpret_cast<const ${entry.name}*>(handle)`;
    return `static_cast<const ${entry.name}*>(${handle})`;
  };

  const decodeRef = (target: ModelClass): Decode => {
    const family = rootOf(target);
    const wrapper = family ? wrapperOf[family] : undefined;
    if (wrapper) return (value) => `${wrapper}(${value}, this.modelOwner)`;
    return (value) => `objOf(${value}, this.modelOwner, ${short(target.name)})`;
  };

  const decodeAt =
    (target: ModelClass): Decode =>
    (value) =>
      `objAt(${value}, this.modelOwner, ${short(target.name)})`;

  const asVal = (value: Wire) =>
    value.channel === "val" ? value.cpp : `val(${value.cpp})`;

  const optionalDecode = (value: Wire): Decode | undefined => {
    const { decode } = value;
    if (!decode) return;
    return (raw) =>
      `((item: any) => (item === undefined ? undefined : ${decode("item")}))(${raw})`;
  };

  const arrayDecode = (value: Wire): Decode | undefined => {
    const { decode } = value;
    if (!decode) return;
    return (raw) => `(${raw} as any[]).map((item: any) => ${decode("item")})`;
  };

  function wire(type: ModelType, expr: string): Wire | undefined {
    switch (type.kind) {
      case "qual":
      case "lvalue-reference":
      case "rvalue-reference":
        return wire(type.element, expr);

      case "builtin": {
        if (type.name === "void") return;
        if (type.name === "bool")
          return {
            channel: "num",
            cpp: `static_cast<double>(${expr})`,
            ts: "boolean",
            decode: (value) => `${value} !== 0`,
          };
        if (type.name.includes("long long")) {
          const native = type.name.startsWith("unsigned")
            ? "std::uint64_t"
            : "std::int64_t";
          return {
            channel: "big",
            cpp: `static_cast<${native}>(${expr})`,
            ts: "bigint",
          };
        }
        return {
          channel: "num",
          cpp: `static_cast<double>(${expr})`,
          ts: "number",
        };
      }

      case "enum":
        enumNames.add(type.name);
        return {
          channel: "num",
          cpp: `static_cast<double>(${expr})`,
          ts: short(type.name),
        };

      case "pointer": {
        const target = unqualified(type.element);
        if (target.kind !== "class") return;
        if (classNames.has(target.name)) {
          const record = index.classOf(target.name)!;
          return {
            channel: "num",
            cpp: handleCpp(record, expr),
            ts: `${short(target.name)} | undefined`,
            decode: decodeRef(record),
          };
        }
        const value = wire(target, "item");
        if (!value) return;
        return {
          channel: "val",
          cpp: `optionalValue(${expr}, [&](const auto& item) { return ${asVal(value)}; })`,
          ts: `${value.ts} | undefined`,
          decode: optionalDecode(value),
        };
      }

      case "class":
        return classWire(type, expr);
    }
  }

  function classWire(type: ModelType, expr: string): Wire | undefined {
    if (type.kind !== "class") return;
    const args = typeArguments(type);

    if (["::std::basic_string", "::std::basic_string_view"].includes(type.name))
      return { channel: "str", cpp: `std::string(${expr})`, ts: "string" };

    if (type.name === "::cxx::SourceLocation")
      return {
        channel: "num",
        cpp: `static_cast<double>(${expr}.index())`,
        ts: "number",
      };

    if (type.name === "::std::shared_ptr") {
      const target = unqualified(args[0]!);
      if (target.kind !== "class" || !classNames.has(target.name)) return;
      const record = index.classOf(target.name)!;
      return {
        channel: "num",
        cpp: handleCpp(record, `${expr}.get()`),
        ts: `${short(target.name)} | undefined`,
        decode: decodeRef(record),
      };
    }

    if (type.name === "::std::optional") {
      const value = wire(args[0]!, "item");
      if (!value) return;
      return {
        channel: "val",
        cpp: `optionalValue(${expr}, [&](const auto& item) { return ${asVal(value)}; })`,
        ts: `${value.ts} | undefined`,
        decode: optionalDecode(value),
      };
    }

    if (containerClasses.includes(type.name)) {
      const value = wire(args[0]!, "item");
      if (!value) return;
      return {
        channel: "val",
        cpp: `arrayValue(${expr}, [&](const auto& item) { return ${asVal(value)}; })`,
        ts: `ReadonlyArray<${value.ts}>`,
        decode: arrayDecode(value),
      };
    }

    if (type.name === "::std::ranges::ref_view") return wire(args[0]!, expr);

    if (type.name === "::std::tuple") {
      const values = args.map((argument, i) =>
        wire(argument, `std::get<${i}>(${expr})`),
      );
      if (values.some((value) => !value)) return;
      const decoded = values.map(
        (value, i) => value!.decode?.(`item[${i}]`) ?? `item[${i}]`,
      );
      return {
        channel: "val",
        cpp: `[&]() -> val { auto result = val::array(); ${values
          .map((value) => `result.call<void>("push", ${asVal(value!)});`)
          .join(" ")} return result; }()`,
        ts: `readonly [${values.map((value) => value!.ts).join(", ")}]`,
        decode: values.some((value) => value!.decode)
          ? (raw) => `((item: any) => [${decoded.join(", ")}] as const)(${raw})`
          : undefined,
      };
    }

    if (type.name === "::cxx::List") return;

    if (type.name === "::std::variant") {
      const values = args.map((argument, i) =>
        wire(argument, `std::get<${i}>(${expr})`),
      );
      if (values.some((value) => !value)) return;
      const alternatives = values
        .map((value, i) =>
          value!.decode
            ? `item.index === ${i} ? { index: ${i}, value: ${value!.decode("item.value")} } : `
            : "",
        )
        .join("");
      return {
        channel: "val",
        cpp: `[&]() -> val { auto result = val::object(); result.set("index", ${expr}.index()); switch (${expr}.index()) { ${values
          .map(
            (value, i) =>
              `case ${i}: result.set("value", ${asVal(value!)}); break;`,
          )
          .join("\n")} } return result; }()`,
        ts: values
          .map(
            (value, i) =>
              `{ readonly index: ${i}; readonly value: ${value!.ts} }`,
          )
          .join(" | "),
        decode: alternatives
          ? (raw) => `((item: any) => ${alternatives}item)(${raw})`
          : undefined,
      };
    }

    if (classNames.has(type.name)) {
      const record = index.classOf(type.name)!;
      return {
        channel: "num",
        cpp: handleCpp(record, `&(${expr})`),
        ts: short(type.name),
        decode: decodeAt(record),
      };
    }

    const record = index.classOf(type.name);
    if (!record || record.isTemplate) return;
    if (record.fields.some((field) => field.access !== "public")) return;
    const fields = record.fields.map((field) => ({
      name: field.name,
      value: wire(field.type, `(${expr}).${field.name}`),
    }));
    if (fields.some((field) => !field.value)) return;
    return {
      channel: "val",
      cpp: `[&]() -> val { auto result = val::object(); ${fields
        .map((field) => `result.set("${field.name}", ${asVal(field.value!)});`)
        .join(" ")} return result; }()`,
      ts: `{ ${fields
        .map((field) => `readonly ${field.name}: ${field.value!.ts}`)
        .join("; ")} }`,
      decode: fields.some((field) => field.value!.decode)
        ? (raw) =>
            `((item: any) => ({ ${fields
              .map(
                (field) =>
                  `${field.name}: ${field.value!.decode?.(`item.${field.name}`) ?? `item.${field.name}`}`,
              )
              .join(", ")} }))(${raw})`
        : undefined,
    };
  }

  function collectReaders(): Map<string, Reader[]> {
    const result = new Map<string, Reader[]>();
    const text = (): ModelType => ({
      kind: "class",
      name: "::std::basic_string",
      isPolymorphic: false,
      arguments: [],
    });

    for (const entry of classes) {
      const items = new Map<string, Reader>();
      const family = familyOf(entry);

      for (const { owner, substitution } of index.layoutOf(entry)) {
        if (family === "AST")
          for (const field of owner.fields) {
            if (field.access !== "public") continue;
            items.set(field.name, {
              name: field.name,
              type: substituteType(field.type, substitution),
              expression: `self->${field.name}`,
            });
          }
        for (const method of owner.methods) {
          if (!method.isConst || method.isStatic || method.parameters.length)
            continue;
          if (
            method.returnType.kind === "builtin" &&
            method.returnType.name === "void"
          )
            continue;
          items.set(method.name, {
            name: method.name,
            type: substituteType(method.returnType, substitution),
            expression: `self->${method.name}()`,
          });
        }
      }

      if (isKinded(entry)) items.delete("kind");

      if (["Symbol", "Type", "Name"].includes(family))
        items.set("text", {
          name: "text",
          type: text(),
          expression:
            family === "Symbol" ? "to_string(self->name())" : "to_string(self)",
        });

      if (family === "AST")
        for (const name of ["firstSourceLocation", "lastSourceLocation"])
          items.set(name, {
            name,
            type: {
              kind: "class",
              name: "::cxx::SourceLocation",
              isPolymorphic: false,
              arguments: [],
            },
            expression: `const_cast<${entry.name}*>(self)->${name}()`,
          });

      if (entry.unqualifiedName === "ClassSymbol") {
        items.set("templatePattern", {
          name: "templatePattern",
          type: {
            kind: "pointer",
            element: {
              kind: "class",
              name: "::cxx::ClassSymbol",
              isPolymorphic: true,
              arguments: [],
            },
          },
          expression: "class_template_of(const_cast<ClassSymbol*>(self))",
        });
        items.set("expandedTemplateArguments", {
          name: "expandedTemplateArguments",
          type: items.get("templateArguments")!.type,
          expression:
            "expand_template_arguments(class_template_arguments(const_cast<ClassSymbol*>(self)))",
        });
        items.set("expandedTemplateArgumentTexts", {
          name: "expandedTemplateArgumentTexts",
          type: {
            kind: "class",
            name: "::std::vector",
            isPolymorphic: false,
            arguments: [{ kind: "type", text: "std::string", type: text() }],
          },
          expression:
            "[&] { std::vector<std::string> result; for (const auto& argument : expand_template_arguments(class_template_arguments(const_cast<ClassSymbol*>(self)))) result.push_back(to_string(argument)); return result; }()",
        });
      }

      if (family === "Symbol")
        items.set("isType", {
          name: "isType",
          type: { kind: "builtin", name: "bool" },
          expression:
            "is_type(const_cast<Symbol*>(static_cast<const Symbol*>(self)))",
        });

      result.set(entry.name, [...items.values()]);
    }
    return result;
  }

  const readers = collectReaders();

  function isASTChild(reader: Reader): boolean {
    if (!reader.expression.startsWith("self->")) return false;
    if (reader.expression.endsWith("()")) return false;
    let type = peel(reader.type);
    if (type.kind !== "pointer") return false;
    type = unqualified(type.element);
    if (type.kind === "class" && type.name === "::cxx::List") {
      const element = typeArguments(type)[0];
      if (!element) return false;
      type = unqualified(element);
      if (type.kind !== "pointer") return false;
      type = unqualified(type.element);
    }
    if (type.kind !== "class") return false;
    const target = index.classOf(type.name);
    return !!target && rootOf(target) === "AST";
  }

  type FamilyCode = Record<string, string[]>;
  const channels = [
    "num",
    "big",
    "str",
    "val",
    "size",
    "item",
    "itemBig",
    "itemStr",
    "itemVal",
  ];
  const code = new Map<string, FamilyCode>(
    families.map((family) => [
      family,
      Object.fromEntries(channels.map((channel) => [channel, []])),
    ]),
  );
  const slots = new Map<string, number>(families.map((family) => [family, 0]));
  const slotBases: { name: string; expression: string }[] = [];
  const previousClass = new Map<string, { name: string; count: number }>();

  const ts: string[] = [];
  const childSlots: string[] = [];
  const visits: { node: string; visit: string }[] = [];
  const constructorTable = new Map<string, string[]>(
    kindedRoots.map((family) => [family, []]),
  );

  for (const entry of classes) {
    const family = familyOf(entry);
    const bodies = code.get(family)!;
    const name = short(entry.name);
    const slotBase = `${name}SlotBase`;
    const firstSlot = slots.get(family)!;
    const self = selfCpp(entry);
    const kindType = short(`::cxx::${family}Kind`);

    let base = "ModelObject";
    for (const item of entry.bases)
      if (classNames.has(item.type)) {
        base = short(item.type);
        break;
      }

    const abstract = kindedRoots.includes(family) && !entry.isFinal;
    ts.push(
      `export ${abstract ? "abstract " : ""}class ${name} extends ${base} {`,
    );

    if (entry.unqualifiedName === family && kindedRoots.includes(family))
      ts.push(
        `readonly kind: ${kindType};`,
        `constructor(handle: number, owner: ModelOwner, kind: ${kindType}) { super(handle, owner); this.kind = kind; }`,
      );

    if (family === "AST") {
      if (entry.unqualifiedName === "AST")
        ts.push(
          `abstract accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result;`,
          `get startLocation(): SourceLocation | undefined { return cxx.getStartLocation(this.handle, this.modelOwner.getUnitHandle()); }`,
          `get endLocation(): SourceLocation | undefined { return cxx.getEndLocation(this.handle, this.modelOwner.getUnitHandle()); }`,
        );
      else if (entry.isFinal) {
        const visit = `visit${enumeratorOf.AST!(entry.unqualifiedName)}`;
        visits.push({ node: name, visit });
        ts.push(
          `accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result { return visitor.${visit}(this, context); }`,
        );
      }
    }

    if (kindedRoots.includes(family) && entry.isFinal)
      constructorTable
        .get(family)!
        .push(
          `${familyKey(family)}Constructors[${short(`::cxx::${family}Kind`)}.${enumeratorOf[family]!(entry.unqualifiedName)}] = ${name};`,
        );

    const children: string[] = [];

    for (const reader of readers.get(entry.name)!) {
      if (["handle", "modelOwner", "kind", "accept"].includes(reader.name))
        throw new Error(`reserved reflection property ${reader.name}`);

      const slotIndex = slots.get(family)!;
      const slot = `${slotBase} + ${slotIndex - firstSlot}`;
      const declared = peel(reader.type);
      const property =
        reader.name === "constructor" ? "constructorSymbol" : reader.name;
      const element = unqualified(
        declared.kind === "pointer" ? declared.element : declared,
      );

      if (
        declared.kind === "pointer" &&
        element.kind === "class" &&
        element.name === "::cxx::List"
      ) {
        const item = wire(typeArguments(element)[0]!, "item");
        if (!item?.decode) {
          report.push(
            `${entry.name}.${reader.name}: ${JSON.stringify(reader.type)}`,
          );
          continue;
        }
        bodies.num!.push(
          `case ${slot}: { auto self = ${self}; return static_cast<double>(reinterpret_cast<std::intptr_t>(${reader.expression})); }`,
        );
        ts.push(
          `get ${property}(): Iterable<${item.ts}> { return listOf(this.modelOwner, cxx.read${family}(this.handle, ${slot}), (item: any) => ${item.decode("item")}); }`,
        );
        if (isASTChild(reader)) children.push(`[${slot}, true]`);
        slots.set(family, slotIndex + 1);
        continue;
      }

      if (
        declared.kind === "class" &&
        containerClasses.includes(declared.name)
      ) {
        const item = wire(typeArguments(declared)[0]!, "item");
        if (!item) {
          report.push(
            `${entry.name}.${reader.name}: ${JSON.stringify(reader.type)}`,
          );
          continue;
        }
        const container = reader.expression;
        bodies.size!.push(
          `case ${slot}: { auto self = ${self}; return static_cast<int>(std::size(${container})); }`,
        );
        bodies[itemChannel[item.channel]]!.push(
          `case ${slot}: { auto self = ${self}; const auto& container = ${container}; const auto& item = *std::next(std::begin(container), index); return ${item.cpp}; }`,
        );
        const helper = `${familyKey(family)}${channelSuffix[item.channel]}Items`;
        usedItemHelpers.set(helper, { family, channel: item.channel });
        ts.push(
          `get ${property}(): Iterable<${item.ts}> { return ${helper}(this.modelOwner, this.handle, ${slot}, (item: any) => ${item.decode?.("item") ?? "item"}); }`,
        );
        slots.set(family, slotIndex + 1);
        continue;
      }

      const value = wire(reader.type, reader.expression);
      if (!value) {
        report.push(
          `${entry.name}.${reader.name}: ${JSON.stringify(reader.type)}`,
        );
        continue;
      }

      bodies[value.channel]!.push(
        `case ${slot}: { auto self = ${self}; return ${value.cpp}; }`,
      );
      const raw = `cxx.read${family}${channelSuffix[value.channel]}(this.handle, ${slot})`;
      const decoded = value.decode
        ? value.decode(raw)
        : value.ts === "number"
          ? raw
          : `${raw} as ${value.ts}`;
      ts.push(`get ${property}(): ${value.ts} { return ${decoded}; }`);
      if (isASTChild(reader)) children.push(`[${slot}, false]`);
      slots.set(family, slotIndex + 1);
    }

    ts.push("}");

    const count = slots.get(family)! - firstSlot;
    if (count) {
      // Chain bases within each family so adding a reader changes only this
      // class's offsets and the next base, not every subsequent class.
      const previous = previousClass.get(family);
      slotBases.push({
        name: slotBase,
        expression: previous ? `${previous.name} + ${previous.count}` : "0",
      });
      previousClass.set(family, { name: slotBase, count });
    }

    if (family === "AST" && entry.isFinal)
      childSlots.push(
        `childSlots[ASTKind.${enumeratorOf.AST!(entry.unqualifiedName)}] = [${children.join(", ")}];`,
      );
  }

  writeCpp();
  writeTs();

  function writeCpp() {
    const cpp: string[] = [];
    cpp.push(
      `// Generated file by: gen_reflection.ts\n${cpy_header}\n#include <cxx/private/model_inputs.h>\n#include <cxx/translation_unit.h>\n#include <emscripten/bind.h>\n#include <emscripten/val.h>\n\n#include <cstdint>\n#include <iterator>\n#include <string>\n#include <type_traits>\n\nnamespace cxx::js {\nnamespace {\nusing emscripten::val;`,
    );
    cpp.push(
      ...slotBases.map(
        ({ name, expression }) => `constexpr int ${name} = ${expression};`,
      ),
    );
    cpp.push(`template <typename T, typename F>
auto optionalValue(const T& value, F convert) -> val {
  if (!value) return val::undefined();
  return convert(*value);
}

template <typename T, typename F>
auto arrayValue(const T& values, F convert) -> val {
  auto result = val::array();
  for (const auto& item : values) result.call<void>("push", convert(item));
  return result;
}
`);

    const signature: Record<string, (family: string) => string> = {
      num: (family) =>
        `auto read${family}(std::intptr_t handle, int slot) -> double`,
      big: (family) =>
        `auto read${family}BigInt(std::intptr_t handle, int slot) -> std::int64_t`,
      str: (family) =>
        `auto read${family}String(std::intptr_t handle, int slot) -> std::string`,
      val: (family) =>
        `auto read${family}Val(std::intptr_t handle, int slot) -> val`,
      size: (family) =>
        `auto read${family}Size(std::intptr_t handle, int slot) -> int`,
      item: (family) =>
        `auto read${family}Item(std::intptr_t handle, int slot, int index) -> double`,
      itemBig: (family) =>
        `auto read${family}ItemBigInt(std::intptr_t handle, int slot, int index) -> std::int64_t`,
      itemStr: (family) =>
        `auto read${family}ItemString(std::intptr_t handle, int slot, int index) -> std::string`,
      itemVal: (family) =>
        `auto read${family}ItemVal(std::intptr_t handle, int slot, int index) -> val`,
    };
    const exportName: Record<string, (family: string) => string> = {
      num: (family) => `read${family}`,
      big: (family) => `read${family}BigInt`,
      str: (family) => `read${family}String`,
      val: (family) => `read${family}Val`,
      size: (family) => `read${family}Size`,
      item: (family) => `read${family}Item`,
      itemBig: (family) => `read${family}ItemBigInt`,
      itemStr: (family) => `read${family}ItemString`,
      itemVal: (family) => `read${family}ItemVal`,
    };

    const bindings: string[] = [];
    for (const family of families) {
      const bodies = code.get(family)!;
      for (const channel of channels) {
        const cases = bodies[channel]!;
        if (!cases.length) continue;
        cpp.push(
          `${signature[channel]!(family)} {\n  switch (slot) {\n${cases.join("\n")}\n  }\n  cxx_runtime_error("unknown model slot");\n}`,
        );
        bindings.push(exportName[channel]!(family));
      }
    }

    for (const family of kindedRoots) {
      cpp.push(
        `auto get${family}Kind(std::intptr_t handle) -> int {\n  return static_cast<int>(reinterpret_cast<const ::cxx::${family}*>(handle)->kind());\n}`,
      );
      bindings.push(`get${family}Kind`);
    }

    cpp.push(`auto getListValue(std::intptr_t handle) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(
      reinterpret_cast<const List<AST*>*>(handle)->value);
}

auto getListNext(std::intptr_t handle) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(
      reinterpret_cast<const List<AST*>*>(handle)->next);
}

auto getUnitAST(std::intptr_t handle) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(static_cast<const ::cxx::AST*>(
      reinterpret_cast<TranslationUnit*>(handle)->ast()));
}

auto getGlobalScope(std::intptr_t handle) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(static_cast<const ::cxx::Symbol*>(
      reinterpret_cast<TranslationUnit*>(handle)->globalScope()));
}`);
    bindings.push(
      "getListValue",
      "getListNext",
      "getUnitAST",
      "getGlobalScope",
    );

    cpp.push(
      `}  // namespace\n}  // namespace cxx::js\n\nEMSCRIPTEN_BINDINGS(cxx_reflection) {\n${bindings
        .map((name) => `  emscripten::function("${name}", &cxx::js::${name});`)
        .join("\n")}\n}`,
    );

    fs.writeFileSync(`${root}/src/js/cxx/reflection.cc`, cpp.join("\n\n"));
  }

  function writeTs() {
    const head: string[] = [];
    head.push(`// Generated file by: gen_reflection.ts\n${cpy_header}
import { cxx } from "./cxx.js";
import { type SourceLocation } from "./SourceLocation.js";

export interface ModelOwner {
  getUnitHandle(): number;
  readonly disposed: boolean;
}

function disposedError(): Error {
  return new Error("Parser has been disposed");
}

export abstract class ModelObject {
  readonly #handle: number;

  constructor(
    handle: number,
    readonly modelOwner: ModelOwner,
  ) {
    this.#handle = handle;
  }

  get handle(): number {
    if (this.modelOwner.disposed) throw disposedError();
    return this.#handle;
  }
}

function objOf<T>(
  handle: number,
  owner: ModelOwner,
  ctor: new (handle: number, owner: ModelOwner) => T,
): T | undefined {
  if (!handle) return undefined;
  return new ctor(handle, owner);
}

function objAt<T>(
  handle: number,
  owner: ModelOwner,
  ctor: new (handle: number, owner: ModelOwner) => T,
): T {
  return new ctor(handle, owner);
}

function* listOf(
  owner: ModelOwner,
  head: number,
  of: (handle: number) => any,
): Iterable<any> {
  let it = head;
  while (it) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.getListValue(it));
    it = cxx.getListNext(it);
  }
}`);

    for (const family of kindedRoots) {
      const key = familyKey(family);
      head.push(`const ${key}Constructors: Array<
  new (handle: number, owner: ModelOwner, kind: ${short(`::cxx::${family}Kind`)}) => ${family}
> = [];

function ${wrapperOf[family]}(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = cxx.get${family}Kind(handle);
  return new ${key}Constructors[kind]!(handle, owner, kind);
}`);
    }

    for (const [helper, { family, channel }] of usedItemHelpers) {
      const size = `cxx.read${family}Size`;
      const item = `cxx.read${family}Item${channelSuffix[channel]}`;
      head.push(`function* ${helper}(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const size = ${size}(handle, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(${item}(handle, slot, i));
  }
}`);
    }

    const tail: string[] = [];

    for (const name of enumNames) {
      const entry = index.enumOf(name);
      if (!entry) {
        tail.push(`export type ${short(name)} = number;`);
        continue;
      }
      tail.push(
        `export enum ${short(name)} {\n${entry.enumerators
          .map((enumerator) => {
            if (enumerator.value === undefined)
              throw new Error(`missing enum value ${name}.${enumerator.name}`);
            return `  ${enumerator.name} = ${enumerator.value},`;
          })
          .join("\n")}\n}`,
      );
    }

    for (const family of kindedRoots)
      tail.push(constructorTable.get(family)!.join("\n"));

    tail.push(`const childSlots: Array<ReadonlyArray<readonly [number, boolean]>> = [];
${childSlots.join("\n")}

export function* children(node: AST): Iterable<AST> {
  for (const [slot, isList] of childSlots[node.kind] ?? []) {
    const value = cxx.readAST(node.handle, slot);
    if (!isList) {
      const child = astOf(value, node.modelOwner);
      if (child) yield child;
      continue;
    }
    for (const child of listOf(node.modelOwner, value, (item: any) =>
      astOf(item, node.modelOwner),
    ))
      if (child) yield child;
  }
}

export function modelOf(owner: ModelOwner): {
  ast: UnitAST;
  globalScope: ScopeSymbol;
} {
  const unit = owner.getUnitHandle();
  return {
    ast: astOf(cxx.getUnitAST(unit), owner),
    globalScope: symbolOf(cxx.getGlobalScope(unit), owner),
  };
}`);

    tail.push(`export abstract class ASTVisitor<Context, Result> {
${visits
  .map(
    ({ node, visit }) =>
      `  abstract ${visit}(node: ${node}, context: Context): Result;`,
  )
  .join("\n")}
}

export class RecursiveASTVisitor<Context> extends ASTVisitor<Context, void> {
  accept(node: AST | undefined, context: Context): void {
    node?.accept(this, context);
  }

  visitChildren(node: AST, context: Context): void {
    for (const child of children(node)) this.accept(child, context);
  }

${visits
  .map(
    ({ node, visit }) =>
      `  ${visit}(node: ${node}, context: Context): void { this.visitChildren(node, context); }`,
  )
  .join("\n")}
}`);

    fs.writeFileSync(
      `${root}/packages/cxx-frontend/src/Semantic.ts`,
      [
        ...head,
        ...slotBases.map(
          ({ name, expression }) => `const ${name} = ${expression};`,
        ),
        ...ts,
        ...tail,
      ].join("\n"),
    );
  }

  const gaps = new Set(report).size;
  if (gaps)
    console.error(
      `gen_reflection: ${gaps} accessors have no binding and were skipped`,
    );

  const commands = [
    ["clang-format", ["-i", `${root}/src/js/cxx/reflection.cc`]],
    [
      `${root}/node_modules/.bin/prettier`,
      ["--write", `${root}/packages/cxx-frontend/src/Semantic.ts`],
    ],
  ] as const;
  for (const [command, args] of commands) {
    const result = spawnSync(command, [...args], { encoding: "utf8" });
    if (result.error) throw result.error;
    if (result.status !== 0)
      throw new Error(`${command} failed: ${result.stderr}`);
  }
}
