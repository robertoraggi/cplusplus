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

import type {
  ModelClass,
  ModelEnum,
  ModelIndex,
  ModelMethod,
  ModelType,
} from "./parseModel.ts";
import { unqualified } from "./parseModel.ts";

export const EMITTER = "::cxx::ir::Emitter";

const NAMESPACE = "::cxx::ir::";

export const UNDELEGATED: Record<string, { cppBody: string }> = {
  debug: { cppBody: "return nullptr;" },
};

export type Wire =
  | { kind: "void" }
  | { kind: "bool" }
  | { kind: "number"; cpp: string }
  | { kind: "constint" }
  | { kind: "location" }
  | { kind: "string" }
  | { kind: "bytes" }
  | { kind: "handle"; tag: string; ref: string }
  | { kind: "enum"; name: string; cpp: string }
  | { kind: "struct"; name: string; cpp: string }
  | { kind: "span"; element: Wire }
  | { kind: "vector"; element: Wire }
  | { kind: "optional"; element: Wire };

export interface ProtocolParameter {
  name: string;
  wire: Wire;
  byConstRef: boolean;
}

export interface ProtocolMethod {
  name: string;
  result: Wire;
  parameters: ProtocolParameter[];
}

export interface ProtocolField {
  name: string;
  wire: Wire;
}

export interface ProtocolStruct {
  name: string;
  cpp: string;
  fields: ProtocolField[];
}

export interface ProtocolEnum {
  name: string;
  cpp: string;
  enumerators: { name: string; value: number }[];
}

export interface Protocol {
  methods: ProtocolMethod[];
  undelegated: { name: string; cppResult: string; cppBody: string }[];
  structs: ProtocolStruct[];
  enums: ProtocolEnum[];
  handles: string[];
}

function shortName(qualified: string): string {
  if (!qualified.startsWith(NAMESPACE))
    throw new Error(`${qualified} is not an emitter protocol type`);
  return qualified.slice(NAMESPACE.length).split("::").join("");
}

function cppName(qualified: string): string {
  return qualified.slice(2);
}

class ProtocolBuilder {
  readonly #index: ModelIndex;
  readonly #structs = new Map<string, ProtocolStruct>();
  readonly #enums = new Map<string, ProtocolEnum>();
  readonly #handles = new Set<string>();
  readonly #shortNames = new Map<string, string>();

  constructor(index: ModelIndex) {
    this.#index = index;
  }

  build(): Protocol {
    const emitter = this.#index.classOf(EMITTER);
    if (!emitter)
      throw new Error(
        `${EMITTER} is not in the semantic model; is <cxx/codegen/emitter.h> ` +
          "included by src/parser/cxx/private/model_inputs.h?",
      );

    const methods: ProtocolMethod[] = [];
    const undelegated: Protocol["undelegated"] = [];

    for (const method of this.protocolMethodsOf(emitter)) {
      const exception = UNDELEGATED[method.name];
      if (exception) {
        undelegated.push({
          name: method.name,
          cppResult: this.cppTypeName(method.returnType),
          cppBody: exception.cppBody,
        });
        continue;
      }
      methods.push({
        name: method.name,
        result: this.wireOf(method.returnType, `${method.name}()`),
        parameters: method.parameters.map((parameter) => ({
          name: parameter.name,
          wire: this.wireOf(
            parameter.type,
            `${method.name}(${parameter.name})`,
          ),
          byConstRef: isConstReference(parameter.type),
        })),
      });
    }

    return {
      methods,
      undelegated,
      structs: [...this.#structs.values()],
      enums: [...this.#enums.values()],
      handles: [...this.#handles].sort(),
    };
  }

  protocolMethodsOf(entry: ModelClass): ModelMethod[] {
    const methods = entry.methods.filter(
      (method) =>
        method.isVirtual && method.isPure && method.access === "public",
    );
    const names = new Set<string>();
    for (const method of methods) {
      if (names.has(method.name))
        throw new Error(
          `${entry.name}::${method.name} is an overloaded protocol method; ` +
            "a JavaScript delegate cannot dispatch on the signature",
        );
      names.add(method.name);
    }
    return methods;
  }

  claim(qualified: string): string {
    const short = shortName(qualified);
    const owner = this.#shortNames.get(short);
    if (owner && owner !== qualified)
      throw new Error(
        `${qualified} and ${owner} both shorten to ${short}; ` +
          "the protocol needs distinct TypeScript names",
      );
    this.#shortNames.set(short, qualified);
    return short;
  }

  wireOf(type: ModelType, where: string): Wire {
    const wire = this.tryWire(type, where);
    if (!wire)
      throw new Error(
        `${where}: no JavaScript representation for ${describe(type)}. ` +
          "Extend the wire table in emitterProtocol.ts, or keep the method " +
          "off the delegated protocol by listing it in UNDELEGATED.",
      );
    return wire;
  }

  tryWire(type: ModelType, where: string): Wire | undefined {
    if (type.kind === "lvalue-reference")
      return this.tryWire(type.element, where);
    if (type.kind === "qual") return this.tryWire(type.element, where);

    if (type.kind === "builtin") {
      if (type.name === "void") return { kind: "void" };
      if (type.name === "bool") return { kind: "bool" };
      if (NUMERIC.has(type.name)) return { kind: "number", cpp: type.name };
      return undefined;
    }

    if (type.kind === "enum") return this.enumWire(type.name);

    if (type.kind !== "class") return undefined;

    switch (type.name) {
      case "::cxx::ConstInt":
        return { kind: "constint" };
      case "::cxx::SourceLocation":
        return { kind: "location" };
      case "::std::basic_string_view":
        return { kind: "string" };
      case "::std::basic_string":
        return { kind: "bytes" };
      case NAMESPACE + "Handle":
        return this.handleWire(type, where);
      case "::std::span":
        return { kind: "span", element: this.argumentWire(type, 0, where) };
      case "::std::vector":
        return { kind: "vector", element: this.argumentWire(type, 0, where) };
      case "::std::optional":
        return { kind: "optional", element: this.argumentWire(type, 0, where) };
      default:
        return this.structWire(type.name, where);
    }
  }

  argumentWire(type: ModelType, index: number, where: string): Wire {
    const argument = type.kind === "class" ? type.arguments[index] : undefined;
    if (!argument || argument.kind !== "type")
      throw new Error(
        `${where}: ${describe(type)} has no type argument ${index}`,
      );
    return this.wireOf(argument.type, where);
  }

  handleWire(type: ModelType, where: string): Wire {
    const tag = unqualified(
      type.kind === "class" && type.arguments[0]?.kind === "type"
        ? type.arguments[0].type
        : { kind: "none" },
    );
    if (tag.kind !== "class" || !tag.name.endsWith("Tag"))
      throw new Error(`${where}: ${describe(type)} has no handle tag`);
    const short = shortName(tag.name);
    this.#handles.add(short);
    return {
      kind: "handle",
      tag: cppName(tag.name),
      ref: short.slice(0, -"Tag".length) + "Ref",
    };
  }

  enumWire(qualified: string): Wire {
    const name = this.claim(qualified);
    if (!this.#enums.has(qualified)) {
      const entry = this.#index.enumOf(qualified);
      if (!entry) throw new Error(`${qualified} is not in the semantic model`);
      this.#enums.set(qualified, {
        name,
        cpp: cppName(qualified),
        enumerators: enumeratorsOf(entry),
      });
    }
    return { kind: "enum", name, cpp: cppName(qualified) };
  }

  structWire(qualified: string, where: string): Wire | undefined {
    if (!qualified.startsWith(NAMESPACE)) return undefined;
    const entry = this.#index.classOf(qualified);
    if (!entry) return undefined;
    const name = this.claim(qualified);
    if (!this.#structs.has(qualified)) {
      const struct: ProtocolStruct = {
        name,
        cpp: cppName(qualified),
        fields: [],
      };
      this.#structs.set(qualified, struct);
      for (const field of entry.fields) {
        if (field.access !== "public") continue;
        struct.fields.push({
          name: field.name,
          wire: this.wireOf(field.type, `${where} -> ${name}.${field.name}`),
        });
      }
    }
    return { kind: "struct", name, cpp: cppName(qualified) };
  }

  cppTypeName(type: ModelType): string {
    if (type.kind === "pointer") return `${this.cppTypeName(type.element)}*`;
    if (type.kind === "class") return cppName(type.name);
    if (type.kind === "builtin") return type.name;
    throw new Error(`cannot spell ${describe(type)} in C++`);
  }
}

const NUMERIC = new Set([
  "char",
  "signed char",
  "unsigned char",
  "short",
  "unsigned short",
  "int",
  "unsigned int",
  "long",
  "unsigned long",
  "long long",
  "unsigned long long",
  "float",
  "double",
]);

function isConstReference(type: ModelType): boolean {
  return type.kind === "lvalue-reference" && type.element.kind === "qual";
}

function enumeratorsOf(entry: ModelEnum): { name: string; value: number }[] {
  return entry.enumerators.map((enumerator) => {
    if (enumerator.value === undefined)
      throw new Error(
        `${entry.name}::${enumerator.name} has no constant value`,
      );
    return { name: enumerator.name, value: enumerator.value };
  });
}

function describe(type: ModelType): string {
  switch (type.kind) {
    case "builtin":
      return type.name;
    case "enum":
      return `enum ${type.name}`;
    case "class":
      return type.arguments.length
        ? `${type.name}<${type.arguments.map((a) => a.text).join(", ")}>`
        : type.name;
    case "pointer":
      return `${describe(type.element)}*`;
    case "lvalue-reference":
      return `${describe(type.element)}&`;
    case "rvalue-reference":
      return `${describe(type.element)}&&`;
    case "qual":
      return `${type.isConst ? "const " : ""}${describe(type.element)}`;
    case "array":
      return `${describe(type.element)}[]`;
    default:
      return type.kind;
  }
}

export function emitterProtocol(index: ModelIndex): Protocol {
  return new ProtocolBuilder(index).build();
}

export function cppType(wire: Wire): string {
  switch (wire.kind) {
    case "void":
      return "void";
    case "bool":
      return "bool";
    case "number":
      return wire.cpp;
    case "constint":
      return "cxx::ConstInt";
    case "location":
      return "cxx::SourceLocation";
    case "string":
      return "std::string_view";
    case "bytes":
      return "std::string";
    case "handle":
      return `cxx::ir::${wire.ref}`;
    case "enum":
    case "struct":
      return wire.cpp;
    case "span":
      return `std::span<const ${cppType(wire.element)}>`;
    case "vector":
      return `std::vector<${cppType(wire.element)}>`;
    case "optional":
      return `std::optional<${cppType(wire.element)}>`;
  }
}

const TS_RESERVED = new Set([
  "await",
  "break",
  "case",
  "catch",
  "class",
  "const",
  "continue",
  "debugger",
  "default",
  "delete",
  "do",
  "else",
  "enum",
  "export",
  "extends",
  "false",
  "finally",
  "for",
  "function",
  "if",
  "import",
  "in",
  "instanceof",
  "let",
  "new",
  "null",
  "return",
  "static",
  "super",
  "switch",
  "this",
  "throw",
  "true",
  "try",
  "typeof",
  "var",
  "void",
  "while",
  "with",
  "yield",
]);

export function tsIdentifier(name: string): string {
  return TS_RESERVED.has(name) ? `${name}_` : name;
}

export function tsType(wire: Wire, position: "in" | "out"): string {
  switch (wire.kind) {
    case "void":
      return "void";
    case "bool":
      return "boolean";
    case "number":
    case "location":
      return "number";
    case "constint":
      return "bigint";
    case "string":
      return "string";
    case "bytes":
      return "Uint8Array";
    case "handle":
      return wire.ref;
    case "enum":
    case "struct":
      return wire.name;
    case "span":
    case "vector": {
      const element = tsType(wire.element, position);
      return position === "in" ? `readonly ${element}[]` : `${element}[]`;
    }
    case "optional":
      return `${tsType(wire.element, position)} | undefined`;
  }
}
