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
  typeKey,
  unqualified,
  typeArguments,
} from "./parseModel.ts";
import { cpy_header } from "./cpy_header.ts";
import * as tokens from "./tokens.ts";

const short = (s: string) => s.replace(/^::cxx::/, "").replaceAll("::", "_");

type Channel = "num" | "big" | "str" | "val";

type Decode = (value: string, owner: string) => string;

type Wire = {
  channel: Channel;
  cpp: string;
  ts: string;
  decode?: Decode | undefined;
};

type Reader = {
  name: string;
  type: ModelType;
  expression: string;
  owner: string;
  isVirtual: boolean;
};

type Slot = { reader: Reader; child?: string | undefined };

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
  Symbol: (name) => name.replace(/Symbol$/, ""),
  Type: (name) => name.replace(/Type$/, ""),
  Name: (name) => name,
};

const spellingLists: Record<string, string[]> = {
  "::cxx::BuiltinTypeTraitKind": tokens.BUILTIN_TYPE_TRAITS,
  "::cxx::UnaryBuiltinTypeKind": tokens.UNARY_BUILTIN_TYPE_SPECIFIERS,
  "::cxx::BinaryBuiltinTypeKind": tokens.BINARY_BUILTIN_TYPE_SPECIFIERS,
  "::cxx::BuiltinFunctionKind": tokens.BUILTIN_FUNCTIONS,
  "::cxx::BuiltinTemplateKind": tokens.BUILTIN_TEMPLATES,
  "::cxx::WellKnownName": tokens.WELL_KNOWN_NAMES,
};

const memberName = (enumName: string, name: string) => {
  const spellings = spellingLists[enumName];
  if (!spellings) return name.replace(/^k(?=[A-Z])/, "");

  const base = name.replace(/^T_/, "");
  if (base === "NONE") return "none";

  const spelling = spellings.find((item) => item.toUpperCase() === base);
  if (!spelling) throw new Error(`no spelling for ${enumName}::${name}`);
  return spelling;
};

const tableOf = (name: string) =>
  `${short(name).replace(/^[A-Z]+(?![a-z])|^[A-Z]/, (prefix) => prefix.toLowerCase())}Names`;

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

  const selected = index.model.classes.filter(
    (entry) =>
      !entry.isTemplate &&
      (rootOf(entry) || rootlessClasses.includes(entry.unqualifiedName)),
  );
  const classNames = new Set(selected.map((entry) => entry.name));

  const classes: ModelClass[] = [];
  const ordered = new Set<string>();
  const order = (entry: ModelClass) => {
    if (ordered.has(entry.name)) return;
    ordered.add(entry.name);
    for (const item of entry.bases) {
      const owner = index.classOf(item.type);
      if (owner && classNames.has(owner.name)) order(owner);
    }
    classes.push(entry);
  };
  for (const entry of selected) order(entry);

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
    if (wrapper) return (value, owner) => `${wrapper}(${value}, ${owner})`;
    return (value, owner) => `objOf(${value}, ${owner}, ${short(target.name)})`;
  };

  const decodeAt =
    (target: ModelClass): Decode =>
    (value, owner) =>
      `objAt(${value}, ${owner}, ${short(target.name)})`;

  const asVal = (value: Wire) =>
    value.channel === "val" ? value.cpp : `val(${value.cpp})`;

  const optionalDecode = (value: Wire): Decode | undefined => {
    const { decode } = value;
    if (!decode) return;
    return (raw, owner) =>
      `optionalOf(${raw}, (item: any) => ${decode("item", owner)})`;
  };

  const arrayDecode = (value: Wire): Decode | undefined => {
    const { decode } = value;
    if (!decode) return;
    return (raw, owner) =>
      `(${raw} as any[]).map((element: any) => ${decode("element", owner)})`;
  };

  const declarations = new Map<string, string>();
  const decoders = new Map<string, { name: string; lines: string[] }>();
  const decoderNames = new Set<string>();

  function declareType(name: string, text: string) {
    if (!declarations.has(name)) declarations.set(name, text);
  }

  function declareDecoder(
    key: string,
    preferred: string,
    returnType: string,
    build: () => string[],
  ): string {
    const declared = decoders.get(key);
    if (declared) return declared.name;

    let name = preferred;
    for (let n = 2; decoderNames.has(name); ++n) name = `${preferred}${n}`;
    decoderNames.add(name);

    const decoder = { name, lines: [] as string[] };
    decoders.set(key, decoder);
    decoder.lines = [
      `function ${name}(value: any, owner: ModelOwner): ${returnType} {`,
      ...build(),
      `}`,
    ];
    return name;
  }

  function unionOf(parts: string[]): string {
    const items: string[] = [];
    let optional = false;
    for (const part of parts) {
      let text = part;
      if (text.endsWith(" | undefined")) {
        text = text.slice(0, -" | undefined".length);
        optional = true;
      }
      if (text === "undefined") {
        optional = true;
        continue;
      }
      if (!items.includes(text)) items.push(text);
    }
    if (optional) items.push("undefined");
    return items.join(" | ");
  }

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

      case "enum": {
        enumNames.add(type.name);
        const wire: Wire = {
          channel: "num",
          cpp: `static_cast<double>(${expr})`,
          ts: short(type.name),
        };
        if (index.enumOf(type.name))
          wire.decode = (value) => `${tableOf(type.name)}[${value}]!`;
        return wire;
      }

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

    if (type.name === "::cxx::ConstInt")
      return {
        channel: "str",
        cpp: `${expr}.toString()`,
        ts: "bigint",
        decode: (value) => `BigInt(${value})`,
      };

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
      const parts = args.map((argument, i) =>
        wire(argument, `std::get<${i}>(${expr})`),
      );
      if (parts.some((part) => !part)) return;
      const values = parts as Wire[];
      const cpp = `[&]() -> val { auto result = val::array(); ${values
        .map((value) => `result.call<void>("push", ${asVal(value)});`)
        .join(" ")} return result; }()`;
      const alias = index.aliasOf(type);
      const elements = `readonly [${values.map((value) => value.ts).join(", ")}]`;
      const ts = alias ? short(alias.name) : elements;
      if (alias) declareType(ts, `export type ${ts} = ${elements};`);
      if (!values.some((value) => value.decode))
        return { channel: "val", cpp, ts };
      const decode = declareDecoder(
        typeKey(type),
        `decode${alias ? ts : "Tuple"}`,
        ts,
        () => [
          `  return [${values
            .map(
              (value, i) =>
                value.decode?.(`value[${i}]`, "owner") ?? `value[${i}]`,
            )
            .join(", ")}] as const;`,
        ],
      );
      return {
        channel: "val",
        cpp,
        ts,
        decode: (raw, owner) => `${decode}(${raw}, ${owner})`,
      };
    }

    if (type.name === "::cxx::List") return;

    if (type.name === "::std::variant") {
      const alternatives = args.map((argument, i) =>
        wire(argument, `std::get<${i}>(${expr})`),
      );
      if (alternatives.some((alternative) => !alternative)) return;
      const values = alternatives as Wire[];
      const cpp = `[&]() -> val { auto result = val::object(); result.set("index", ${expr}.index()); switch (${expr}.index()) { ${values
        .map(
          (value, i) =>
            `case ${i}: result.set("value", ${asVal(value)}); break;`,
        )
        .join("\n")} } return result; }()`;

      const alias = index.aliasOf(type);
      const union = unionOf(values.map((value) => value.ts));
      const ts = alias ? short(alias.name) : union;
      if (alias) declareType(ts, `export type ${ts} = ${union};`);

      if (!values.some((value) => value.decode))
        return { channel: "val", cpp, ts, decode: (raw) => `${raw}.value` };

      const decode = declareDecoder(
        typeKey(type),
        `decode${alias ? ts : "Variant"}`,
        ts,
        () => {
          const lines = ["  switch (value.index) {"];
          values.forEach((value, i) => {
            if (!value.decode) return;
            lines.push(
              `    case ${i}:`,
              `      return ${value.decode("value.value", "owner")};`,
            );
          });
          lines.push("    default:", "      return value.value;", "  }");
          return lines;
        },
      );
      return {
        channel: "val",
        cpp,
        ts,
        decode: (raw, owner) => `${decode}(${raw}, ${owner})`,
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

    if (record.fields.length === 0)
      return {
        channel: "val",
        cpp: "val::undefined()",
        ts: "undefined",
        decode: () => "undefined",
      };

    const members = record.fields.map((field) => ({
      name: field.name,
      value: wire(field.type, `(${expr}).${field.name}`),
    }));
    if (members.some((member) => !member.value)) return;
    const fields = members as { name: string; value: Wire }[];

    const cpp = `[&]() -> val { auto result = val::object(); ${fields
      .map((field) => `result.set("${field.name}", ${asVal(field.value)});`)
      .join(" ")} return result; }()`;
    const ts = short(record.name);
    declareType(
      ts,
      `export interface ${ts} {\n${fields
        .map((field) => `  readonly ${field.name}: ${field.value.ts};`)
        .join("\n")}\n}`,
    );

    if (!fields.some((field) => field.value.decode))
      return { channel: "val", cpp, ts };

    const decode = declareDecoder(typeKey(type), `decode${ts}`, ts, () => [
      "  return {",
      ...fields.map(
        (field) =>
          `    ${field.name}: ${field.value.decode?.(`value.${field.name}`, "owner") ?? `value.${field.name}`},`,
      ),
      "  };",
    ]);
    return {
      channel: "val",
      cpp,
      ts,
      decode: (raw, owner) => `${decode}(${raw}, ${owner})`,
    };
  }

  function collectReaders(): Map<string, Map<string, Reader>> {
    const result = new Map<string, Map<string, Reader>>();
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
              owner: owner.name,
              isVirtual: false,
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
            owner: owner.name,
            isVirtual: method.isVirtual,
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
          owner: `::cxx::${family}`,
          isVirtual: false,
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
            owner: "::cxx::AST",
            isVirtual: true,
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
          owner: entry.name,
          isVirtual: false,
        });
        items.set("expandedTemplateArguments", {
          name: "expandedTemplateArguments",
          type: items.get("templateArguments")!.type,
          expression:
            "expand_template_arguments(class_template_arguments(const_cast<ClassSymbol*>(self)))",
          owner: entry.name,
          isVirtual: false,
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
          owner: entry.name,
          isVirtual: false,
        });
      }

      if (family === "Symbol")
        items.set("isType", {
          name: "isType",
          type: { kind: "builtin", name: "bool" },
          expression:
            "is_type(const_cast<Symbol*>(static_cast<const Symbol*>(self)))",
          owner: "::cxx::Symbol",
          isVirtual: false,
        });

      result.set(entry.name, items);
    }
    return result;
  }

  const readers = collectReaders();

  function sharesSlot(inherited: Reader, reader: Reader): boolean {
    if (JSON.stringify(inherited.type) !== JSON.stringify(reader.type))
      return false;
    return inherited.owner === reader.owner || inherited.isVirtual;
  }

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
  const classSlots = new Map<string, Map<string, Slot>>();

  const ts: string[] = [];
  const childSlots: string[] = [];
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

    const baseEntry = index.primaryBaseOf(entry, (candidate) =>
      classNames.has(candidate.name),
    );
    const base = baseEntry ? short(baseEntry.name) : "ModelObject";
    const inherited: Map<string, Slot> = baseEntry
      ? classSlots.get(baseEntry.name)!
      : new Map();
    const slotMap = new Map<string, Slot>(inherited);
    classSlots.set(entry.name, slotMap);

    const abstract = kindedRoots.includes(family) && !entry.isFinal;
    ts.push(
      `export ${abstract ? "abstract " : ""}class ${name} extends ${base} {`,
    );

    if (entry.unqualifiedName === family && kindedRoots.includes(family))
      ts.push(
        `readonly kind: ${kindType};`,
        `constructor(handle: number, owner: ModelOwner, kind: ${kindType}) { super(handle, owner); this.kind = kind; }`,
      );

    if (family === "AST" && entry.unqualifiedName === "AST")
      ts.push(
        `get startLocation(): SourceLocation | undefined { return cxx.getStartLocation(this.handle, this.modelOwner.getUnitHandle()); }`,
        `get endLocation(): SourceLocation | undefined { return cxx.getEndLocation(this.handle, this.modelOwner.getUnitHandle()); }`,
      );

    if (kindedRoots.includes(family) && entry.isFinal)
      constructorTable
        .get(family)!
        .push(`  ${enumeratorOf[family]!(entry.unqualifiedName)}: ${name},`);

    for (const reader of readers.get(entry.name)!.values()) {
      if (["handle", "modelOwner", "kind", "accept"].includes(reader.name))
        throw new Error(`reserved reflection property ${reader.name}`);

      const inheritedSlot = inherited.get(reader.name);
      if (inheritedSlot && sharesSlot(inheritedSlot.reader, reader)) continue;

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
          `get ${property}(): Iterable<${item.ts}> { return listOf(this.modelOwner, cxx.read${family}(this.handle, ${slot}), (item: any) => ${item.decode("item", "this.modelOwner")}); }`,
        );
        slotMap.set(reader.name, {
          reader,
          child: isASTChild(reader)
            ? `[${slot}, true, "${property}"]`
            : undefined,
        });
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
          `get ${property}(): Iterable<${item.ts}> { return ${helper}(this.modelOwner, this.handle, ${slot}, (item: any) => ${item.decode?.("item", "this.modelOwner") ?? "item"}); }`,
        );
        slotMap.set(reader.name, { reader });
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
        ? value.decode(raw, "this.modelOwner")
        : value.ts === "number"
          ? raw
          : `${raw} as ${value.ts}`;
      ts.push(`get ${property}(): ${value.ts} { return ${decoded}; }`);
      slotMap.set(reader.name, {
        reader,
        child: isASTChild(reader)
          ? `[${slot}, false, "${property}"]`
          : undefined,
      });
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

    if (family === "AST" && entry.isFinal) {
      const children = [...slotMap.values()]
        .map((item) => item.child)
        .filter((child) => child !== undefined);
      childSlots.push(
        `  ${enumeratorOf.AST!(entry.unqualifiedName)}: [${children.join(", ")}],`,
      );
    }
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
    const imports: string[] = [];
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
}

function optionalOf<T>(value: any, of: (item: any) => T): T | undefined {
  if (value === undefined) return undefined;
  return of(value);
}`);

    for (const family of kindedRoots) {
      const key = familyKey(family);
      const kindType = short(`::cxx::${family}Kind`);
      head.push(`function ${wrapperOf[family]}(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = ${tableOf(`::cxx::${family}Kind`)}[cxx.get${family}Kind(handle)]!;
  return new ${key}Constructors[kind](handle, owner, kind);
}`);
      void kindType;
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
      const members = new Map<number, string>();
      for (const enumerator of entry.enumerators) {
        if (enumerator.value === undefined)
          throw new Error(`missing enum value ${name}.${enumerator.name}`);
        const member = memberName(name, enumerator.name);
        if ([...members.values()].includes(member))
          throw new Error(`duplicate enumerator ${name}.${member}`);
        members.set(enumerator.value, member);
      }
      if (name === "::cxx::TokenKind") {
        imports.push(
          `import { type TokenKind, tokenKindNames } from "./TokenKind.js";`,
        );
        continue;
      }
      tail.push(
        `export type ${short(name)} =\n${[...members.values()]
          .map((member) => `  | "${member}"`)
          .join("\n")};`,
        `const ${tableOf(name)}: Record<number, ${short(name)}> = {\n${[
          ...members,
        ]
          .map(([value, member]) => `  ${value}: "${member}",`)
          .join("\n")}\n};`,
      );
    }

    for (const family of kindedRoots) {
      const kindType = short(`::cxx::${family}Kind`);
      tail.push(
        `const ${familyKey(family)}Constructors: Record<
  ${kindType},
  new (handle: number, owner: ModelOwner, kind: ${kindType}) => ${family}
> = {
${constructorTable.get(family)!.join("\n")}
};`,
      );
    }

    tail.push(`const childSlots: Partial<
  Record<ASTKind, ReadonlyArray<readonly [number, boolean, string]>>
> = {
${childSlots.join("\n")}
};

export interface ASTChild {
  readonly node: AST;
  readonly key: string | number;
  readonly listKey: string | undefined;
}

export function* children(node: AST): Generator<ASTChild> {
  for (const [slot, isList, key] of childSlots[node.kind] ?? []) {
    const value = cxx.readAST(node.handle, slot);
    if (!isList) {
      const child = astOf(value, node.modelOwner);
      if (child) yield { node: child, key, listKey: undefined };
      continue;
    }
    let index = 0;
    for (const child of listOf(node.modelOwner, value, (item: any) =>
      astOf(item, node.modelOwner),
    )) {
      if (child) yield { node: child, key: index, listKey: key };
      ++index;
    }
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

    fs.writeFileSync(
      `${root}/packages/cxx-frontend/src/Semantic.ts`,
      [
        ...head,
        ...imports,
        ...declarations.values(),
        ...[...decoders.values()].map((decoder) => decoder.lines.join("\n")),
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
