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

export interface Location {
  file: string;
  line: number;
  column: number;
}

export interface Span {
  file: string;
  startLine: number;
  startColumn: number;
  endLine: number;
  endColumn: number;
}

export type ModelType =
  | { kind: "none" }
  | { kind: "builtin"; name: string }
  | { kind: "enum"; name: string; isScoped: boolean }
  | { kind: "class"; name: string; isPolymorphic: boolean; arguments: ModelTemplateArgument[] }
  | { kind: "pointer"; element: ModelType }
  | { kind: "lvalue-reference"; element: ModelType }
  | { kind: "rvalue-reference"; element: ModelType }
  | { kind: "qual"; isConst: boolean; isVolatile: boolean; element: ModelType }
  | { kind: "array"; element: ModelType; size?: number }
  | { kind: "type-param"; index: number; depth: number; isPack: boolean }
  | { kind: "function"; returnType: ModelType; parameterTypes: ModelType[] }
  | { kind: "member-object-pointer"; classType: ModelType; element: ModelType }
  | { kind: "member-function-pointer"; classType: ModelType; functionType: ModelType };

export type ModelTemplateArgument =
  | { text: string; kind: "type"; type: ModelType }
  | { text: string; kind: "symbol"; name: string }
  | { text: string; kind: "value"; value?: number }
  | { text: string; kind: "expression" };

export type ModelAccess = "public" | "protected" | "private";

export interface ModelField {
  name: string;
  access: ModelAccess;
  isBitField: boolean;
  isMutable: boolean;
  anonymousPath: string[];
  typeName: string;
  type: ModelType;
  location?: Location;
  declaration?: Span;
  initializer?: Span;
  initializerText?: string;
  declaredType?: string;
}

export interface ModelBase {
  type: string;
  isVirtual: boolean;
  typeNode?: ModelType;
}

export interface ModelParameter {
  name: string;
  typeName: string;
  type: ModelType;
}

export interface ModelConstructor {
  isExplicit: boolean;
  parameters: ModelParameter[];
}

export interface ModelMethod {
  name: string;
  access: ModelAccess;
  isStatic: boolean;
  isConst: boolean;
  isVirtual: boolean;
  isPure: boolean;
  returnTypeName: string;
  returnType: ModelType;
  parameters: ModelParameter[];
}

export interface ModelClass {
  name: string;
  unqualifiedName: string;
  isUnion: boolean;
  isFinal: boolean;
  isTemplate: boolean;
  location?: Location;
  body?: Span;
  bases: ModelBase[];
  fields: ModelField[];
  constructors: ModelConstructor[];
  methods: ModelMethod[];
}

export interface ModelEnumerator {
  name: string;
  value?: number;
}

export interface ModelEnum {
  name: string;
  isScoped: boolean;
  underlyingType: string;
  location?: Location;
  enumerators: ModelEnumerator[];
}

export interface ModelAlias {
  name: string;
  unqualifiedName: string;
  typeName: string;
  type: ModelType;
  location?: Location;
}

export interface Model {
  enums: ModelEnum[];
  classes: ModelClass[];
  aliases: ModelAlias[];
}

export class ModelIndex {
  readonly classes = new Map<string, ModelClass>();
  readonly enums = new Map<string, ModelEnum>();
  readonly aliases = new Map<string, ModelAlias>();
  readonly model: Model;

  constructor(model: Model) {
    this.model = model;
    for (const entry of model.classes) this.classes.set(entry.name, entry);
    for (const entry of model.enums) this.enums.set(entry.name, entry);
    for (const entry of model.aliases ?? []) {
      const key = typeKey(entry.type);
      if (!this.aliases.has(key)) this.aliases.set(key, entry);
    }
  }

  aliasOf(type: ModelType): ModelAlias | undefined {
    return this.aliases.get(typeKey(unqualified(type)));
  }

  classOf(name: string): ModelClass | undefined {
    return this.classes.get(name);
  }

  enumOf(name: string): ModelEnum | undefined {
    return this.enums.get(name);
  }

  fileOf(entry: ModelClass): string {
    const file = entry.location?.file ?? "";
    const slash = file.lastIndexOf("/");
    return slash < 0 ? file : file.slice(slash + 1);
  }

  /**
   * The class, followed by every base in declaration order, with template
   * arguments substituted for the type parameters of a template base.
   */
  layoutOf(entry: ModelClass): { owner: ModelClass; substitution: ModelType[] }[] {
    const result: { owner: ModelClass; substitution: ModelType[] }[] = [];

    const walk = (current: ModelClass, substitution: ModelType[]) => {
      for (const base of current.bases) {
        const owner = this.classOf(base.type);
        if (!owner) continue;
        const args: ModelType[] = [];
        const node = base.typeNode;
        if (node && node.kind === "class") {
          for (const argument of node.arguments) {
            if (argument.kind === "type")
              args.push(substituteType(argument.type, substitution));
          }
        }
        walk(owner, args);
      }
      result.push({ owner: current, substitution });
    };

    walk(entry, []);

    return result;
  }

  primaryBaseOf(
    entry: ModelClass,
    accept: (base: ModelClass) => boolean,
  ): ModelClass | undefined {
    const base = entry.bases[0];
    if (!base) return undefined;
    const owner = this.classOf(base.type);
    if (!owner || !accept(owner)) return undefined;
    return owner;
  }

  ownLayoutOf(
    entry: ModelClass,
    base: ModelClass | undefined,
  ): { owner: ModelClass; substitution: ModelType[] }[] {
    const layout = this.layoutOf(entry);
    if (!base) return layout;

    const shared = this.layoutOf(base);
    for (let i = 0; i < shared.length; ++i) {
      const expected = shared[i]!;
      const actual = layout[i];
      if (
        !actual ||
        actual.owner.name !== expected.owner.name ||
        JSON.stringify(actual.substitution) !==
          JSON.stringify(expected.substitution)
      )
        throw new Error(
          `the layout of ${base.name} is not a prefix of the layout of ${entry.name}`,
        );
    }

    return layout.slice(shared.length);
  }
}

export function substituteType(
  type: ModelType,
  substitution: ModelType[],
): ModelType {
  switch (type.kind) {
    case "type-param": {
      const replacement = substitution[type.index];
      if (!replacement)
        throw new Error(
          `no template argument for type-param<${type.index}, ${type.depth}>`,
        );
      return replacement;
    }
    case "pointer":
    case "lvalue-reference":
    case "rvalue-reference":
      return { ...type, element: substituteType(type.element, substitution) };
    case "qual":
      return { ...type, element: substituteType(type.element, substitution) };
    case "array":
      return { ...type, element: substituteType(type.element, substitution) };
    case "class":
      return {
        ...type,
        arguments: type.arguments.map((argument) =>
          argument.kind === "type"
            ? { ...argument, type: substituteType(argument.type, substitution) }
            : argument,
        ),
      };
    default:
      return type;
  }
}

export function typeKey(type: ModelType): string {
  switch (type.kind) {
    case "none":
      return "none";
    case "builtin":
      return type.name;
    case "enum":
      return `enum ${type.name}`;
    case "class":
      return `${type.name}<${type.arguments
        .map((argument) =>
          argument.kind === "type"
            ? typeKey(argument.type)
            : `${argument.kind}:${argument.text}`,
        )
        .join(",")}>`;
    case "pointer":
      return `${typeKey(type.element)}*`;
    case "lvalue-reference":
      return `${typeKey(type.element)}&`;
    case "rvalue-reference":
      return `${typeKey(type.element)}&&`;
    case "qual":
      return `${type.isConst ? "const " : ""}${type.isVolatile ? "volatile " : ""}${typeKey(type.element)}`;
    case "array":
      return `${typeKey(type.element)}[${type.size ?? ""}]`;
    case "type-param":
      return `type-param<${type.index},${type.depth}${type.isPack ? ",..." : ""}>`;
    case "function":
      return `${typeKey(type.returnType)}(${type.parameterTypes.map(typeKey).join(",")})`;
    case "member-object-pointer":
      return `${typeKey(type.classType)}::*${typeKey(type.element)}`;
    case "member-function-pointer":
      return `${typeKey(type.classType)}::*${typeKey(type.functionType)}`;
  }
}

/** Drops top-level cv-qualification. */
export function unqualified(type: ModelType): ModelType {
  return type.kind === "qual" ? unqualified(type.element) : type;
}

export function typeArguments(type: ModelType): ModelType[] {
  const target = unqualified(type);
  if (target.kind !== "class") return [];
  return target.arguments
    .filter((argument) => argument.kind === "type")
    .map((argument) => (argument as { type: ModelType }).type);
}

export function className(type: ModelType): string | undefined {
  const target = unqualified(type);
  return target.kind === "class" ? target.name : undefined;
}

export function loadModel(source: string): ModelIndex {
  return new ModelIndex(JSON.parse(source) as Model);
}
