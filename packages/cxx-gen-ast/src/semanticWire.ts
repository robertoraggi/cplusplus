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

import {
  type ModelIndex,
  type ModelType,
  className,
  substituteType,
  typeArguments,
  unqualified,
} from "./parseModel.ts";

export type Wire =
  | { k: "bool" }
  | { k: "u32" }
  | { k: "i32" }
  | { k: "u64" }
  | { k: "i64" }
  | { k: "f32" }
  | { k: "f64" }
  | { k: "f80" }
  | { k: "string" }
  | { k: "enum"; name: string; count: number; last?: string }
  | { k: "location" }
  | { k: "name" }
  | { k: "type"; cpp: string }
  | { k: "symbol"; cpp: string }
  | { k: "ast"; cpp: string }
  | { k: "ast-list"; element: string }
  | { k: "identifier" }
  | { k: "literal"; cpp: string }
  | { k: "abi-tags" }
  | { k: "attributes" }
  | { k: "const-value" }
  | { k: "const-value-ptr" }
  | { k: "template-argument" }
  | { k: "indeterminate" }
  | { k: "shared"; cpp: string }
  | { k: "unique"; cpp: string; element: Wire }
  | { k: "optional"; cpp: string; element: Wire }
  | { k: "vector"; cpp: string; view: boolean; element: Wire }
  | { k: "deque"; cpp: string; element: Wire }
  | { k: "pair"; cpp: string; first: Wire; second: Wire }
  | { k: "tuple"; cpp: string; elements: Wire[] }
  | { k: "map"; cpp: string; key: Wire; value: Wire }
  | { k: "struct"; name: string; cpp: string }
  | { k: "variant"; cpp: string; alternatives: Wire[] }
  | { k: "unit" }
  | { k: "identifier-info" };

const literalClasses = new Set([
  "::cxx::Literal",
  "::cxx::IntegerLiteral",
  "::cxx::FloatLiteral",
  "::cxx::StringLiteral",
  "::cxx::CharLiteral",
  "::cxx::CommentLiteral",
]);

const nameClasses = new Set([
  "::cxx::Name",
  "::cxx::Identifier",
  "::cxx::OperatorId",
  "::cxx::DestructorId",
  "::cxx::LiteralOperatorId",
  "::cxx::ConversionFunctionId",
  "::cxx::TemplateId",
]);

const integerWires: Record<string, Wire> = {
  bool: { k: "bool" },
  char: { k: "i32" },
  "signed char": { k: "i32" },
  "unsigned char": { k: "u32" },
  short: { k: "i32" },
  "short int": { k: "i32" },
  "unsigned short": { k: "u32" },
  "unsigned short int": { k: "u32" },
  int: { k: "i32" },
  "unsigned int": { k: "u32" },
  long: { k: "i64" },
  "long int": { k: "i64" },
  "unsigned long": { k: "u64" },
  "unsigned long int": { k: "u64" },
  "long long": { k: "i64" },
  "long long int": { k: "i64" },
  "unsigned long long": { k: "u64" },
  "unsigned long long int": { k: "u64" },
  float: { k: "f32" },
  double: { k: "f64" },
  "long double": { k: "f80" },
};

export class WireMapper {
  readonly structs = new Map<string, string>();
  readonly index: ModelIndex;

  constructor(index: ModelIndex) {
    this.index = index;
  }

  isSymbolClass(name: string): boolean {
    const entry = this.index.classOf(name);
    if (!entry) return false;
    if (name === "::cxx::Symbol") return true;
    return entry.bases.some((base) => this.isSymbolClass(base.type));
  }

  isAstClass(name: string): boolean {
    const entry = this.index.classOf(name);
    if (!entry) return false;
    if (name === "::cxx::AST") return true;
    return entry.bases.some((base) => this.isAstClass(base.type));
  }

  isTypeClass(name: string): boolean {
    const entry = this.index.classOf(name);
    if (!entry) return false;
    if (name === "::cxx::Type") return true;
    return entry.bases.some((base) => this.isTypeClass(base.type));
  }

  isNameClass(name: string): boolean {
    if (nameClasses.has(name)) return true;
    const entry = this.index.classOf(name);
    if (!entry) return false;
    return entry.bases.some((base) => this.isNameClass(base.type));
  }

  cppTypeOf(type: ModelType): string {
    switch (type.kind) {
      case "none":
        return "void";
      case "builtin":
        return type.name;
      case "enum":
        return type.name;
      case "pointer":
        return `${this.cppTypeOf(type.element)}*`;
      case "lvalue-reference":
        return `${this.cppTypeOf(type.element)}&`;
      case "rvalue-reference":
        return `${this.cppTypeOf(type.element)}&&`;
      case "qual": {
        const inner = this.cppTypeOf(type.element);
        return type.isConst ? `const ${inner}` : inner;
      }
      case "array":
        return `${this.cppTypeOf(type.element)}[${type.size ?? ""}]`;
      case "class": {
        if (type.name === "::std::variant") {
          const alias = this.variantAliasOf(type);
          if (alias) return alias;
        }
        const args = type.arguments
          .filter((argument) => argument.kind === "type")
          .map((argument) =>
            this.cppTypeOf((argument as { type: ModelType }).type),
          );
        const name = normalizeClassName(type.name);
        if (args.length === 0) return name;
        if (isStdContainer(type.name)) {
          const kept = keptArguments(type.name, args);
          return kept.length === 0 ? name : `${name}<${kept.join(", ")}>`;
        }
        return `${name}<${args.join(", ")}>`;
      }
      case "function": {
        const parameters = type.parameterTypes
          .map((parameter) => this.cppTypeOf(parameter))
          .join(", ");
        return `${this.cppTypeOf(type.returnType)}(${parameters})`;
      }
      case "member-object-pointer":
        return `${this.cppTypeOf(type.element)} ${this.cppTypeOf(type.classType)}::*`;
      case "member-function-pointer":
        return `${this.cppTypeOf(type.functionType)} ${this.cppTypeOf(type.classType)}::*`;
      case "type-param":
        throw new Error("unsubstituted template parameter");
    }
  }

  variantAliasOf(type: ModelType): string | undefined {
    const alternatives = typeArguments(type).map((argument) => {
      const pointee = className(unqualified(pointerElement(argument)));
      if (pointee) return pointee;
      const inner = unqualified(argument);
      if (inner.kind === "builtin") return inner.name;
      return className(argument) ?? inner.kind;
    });

    if (
      alternatives.length === 4 &&
      alternatives[0] === "::cxx::Type" &&
      alternatives[1] === "::cxx::Symbol"
    )
      return "cxx::TemplateArgument";

    if (
      alternatives[0] === "long long" &&
      alternatives[1] === "::cxx::StringLiteral" &&
      alternatives.at(-1) === "::cxx::IndeterminateValue"
    )
      return "cxx::ConstValue";

    return undefined;
  }

  wireOf(type: ModelType, context: string): Wire {
    const target = unqualified(type);

    switch (target.kind) {
      case "builtin": {
        const wire = integerWires[target.name];
        if (wire) return wire;
        throw new Error(`${context}: unmapped builtin type '${target.name}'`);
      }

      case "enum": {
        const entry = this.index.enumOf(target.name);
        if (!entry)
          throw new Error(
            `${context}: enum '${target.name}' is not in the model`,
          );
        const last = entry.enumerators.at(-1);
        const isContiguous =
          last !== undefined && last.value === entry.enumerators.length - 1;
        const wire: Wire = {
          k: "enum",
          name: target.name,
          count: entry.enumerators.length,
        };
        if (isContiguous) wire.last = last.name;
        return wire;
      }

      case "pointer":
        return this.pointerWireOf(target.element, context);

      case "class":
        return this.classWireOf(target, context);

      case "array":
      case "lvalue-reference":
      case "rvalue-reference":
      case "function":
      case "member-object-pointer":
      case "member-function-pointer":
      case "none":
      case "qual":
      case "type-param":
        break;
    }

    throw new Error(`${context}: unmapped type kind '${target.kind}'`);
  }

  private pointerWireOf(pointee: ModelType, context: string): Wire {
    const target = unqualified(pointee);

    if (target.kind !== "class")
      throw new Error(`${context}: unmapped pointer to '${target.kind}'`);

    const name = target.name;

    if (name === "::cxx::TranslationUnit") return { k: "unit" };
    if (name === "::cxx::IdentifierInfo") return { k: "identifier-info" };

    if (name === "::cxx::List") {
      const [element] = typeArguments(target);
      if (!element) throw new Error(`${context}: List without an element type`);
      const elementName = className(unqualified(pointerElement(element)));
      if (!elementName || !this.isAstClass(elementName))
        throw new Error(`${context}: List of non-AST '${elementName}'`);
      return { k: "ast-list", element: normalizeClassName(elementName) };
    }

    if (name === "::std::vector") {
      const element = requireArgument(typeArguments(target), 0, context);
      const elementName = className(unqualified(pointerElement(element)));
      if (elementName === "::cxx::Identifier") return { k: "abi-tags" };
      if (elementName === "::cxx::Attribute") return { k: "attributes" };
      throw new Error(
        `${context}: unmapped pointer to vector of '${elementName}'`,
      );
    }

    if (name === "::std::variant") {
      const variant = this.variantWireOf(target, context);
      if (variant.k !== "const-value")
        throw new Error(`${context}: unmapped pointer to variant`);
      return { k: "const-value-ptr" };
    }

    if (name === "::cxx::Identifier") return { k: "identifier" };

    if (this.isNameClass(name)) return { k: "name" };
    if (literalClasses.has(name))
      return { k: "literal", cpp: normalizeClassName(name) };
    if (this.isTypeClass(name))
      return { k: "type", cpp: normalizeClassName(name) };
    if (this.isSymbolClass(name))
      return { k: "symbol", cpp: normalizeClassName(name) };
    if (this.isAstClass(name))
      return { k: "ast", cpp: normalizeClassName(name) };

    throw new Error(`${context}: unmapped pointer to class '${name}'`);
  }

  private classWireOf(
    target: ModelType & { kind: "class" },
    context: string,
  ): Wire {
    const name = target.name;
    const args = typeArguments(target);
    const cpp = this.cppTypeOf(target);

    switch (name) {
      case "::cxx::SourceLocation":
        return { k: "location" };

      case "::cxx::IndeterminateValue":
        return { k: "indeterminate" };

      case "::std::basic_string":
      case "::std::basic_string_view":
        return { k: "string" };

      case "::std::variant":
        return this.variantWireOf(target, context);

      case "::std::optional":
        return {
          k: "optional",
          cpp,
          element: this.wireOf(requireArgument(args, 0, context), context),
        };

      case "::std::vector":
      case "::std::span":
        return {
          k: "vector",
          cpp:
            name === "::std::span"
              ? `std::vector<${this.cppTypeOf(stripConst(requireArgument(args, 0, context)))}>`
              : cpp,
          view: name === "::std::span",
          element: this.wireOf(requireArgument(args, 0, context), context),
        };

      case "::std::deque":
        return {
          k: "deque",
          cpp,
          element: this.wireOf(requireArgument(args, 0, context), context),
        };

      case "::std::unique_ptr":
        return {
          k: "unique",
          cpp,
          element: this.wireOf(requireArgument(args, 0, context), context),
        };

      case "::std::shared_ptr": {
        const element = className(requireArgument(args, 0, context));
        if (!element)
          throw new Error(`${context}: shared_ptr of a non-class type`);
        this.noteStruct(element, normalizeClassName(element));
        return { k: "shared", cpp: normalizeClassName(element) };
      }

      case "::std::pair":
        return {
          k: "pair",
          cpp,
          first: this.wireOf(requireArgument(args, 0, context), context),
          second: this.wireOf(requireArgument(args, 1, context), context),
        };

      case "::std::tuple":
        return {
          k: "tuple",
          cpp,
          elements: args.map((argument) => this.wireOf(argument, context)),
        };

      case "::std::unordered_map":
      case "::std::map":
        return {
          k: "map",
          cpp,
          key: this.wireOf(requireArgument(args, 0, context), context),
          value: this.wireOf(requireArgument(args, 1, context), context),
        };

      default:
        break;
    }

    if (this.index.classOf(name)) {
      this.noteStruct(name, cpp);
      return { k: "struct", name, cpp };
    }

    throw new Error(`${context}: unmapped class '${name}'`);
  }

  variantWireOf(target: ModelType & { kind: "class" }, context: string): Wire {
    const alias = this.variantAliasOf(target);

    if (alias === "cxx::TemplateArgument") return { k: "template-argument" };
    if (alias === "cxx::ConstValue") return { k: "const-value" };

    return {
      k: "variant",
      cpp: this.cppTypeOf(target),
      alternatives: typeArguments(target).map((argument) =>
        this.wireOf(argument, context),
      ),
    };
  }

  requireStruct(name: string) {
    const entry = this.index.classOf(name);
    if (!entry) throw new Error(`struct '${name}' is not in the model`);
    this.noteStruct(name, normalizeClassName(name));
  }

  private noteStruct(name: string, cpp: string) {
    if (this.structs.has(name)) return;
    this.structs.set(name, cpp);
    const entry = this.index.classOf(name);
    if (!entry) return;
    for (const { owner, substitution } of this.index.layoutOf(entry)) {
      for (const field of owner.fields) {
        if (field.isBitField) continue;
        try {
          this.wireOf(
            substituteTypeSafely(field.type, substitution),
            `${owner.name}::${field.name}`,
          );
        } catch {
          // Reported by the caller that classifies this field.
        }
      }
    }
  }
}

function substituteTypeSafely(
  type: ModelType,
  substitution: ModelType[],
): ModelType {
  try {
    return substituteType(type, substitution);
  } catch {
    return type;
  }
}

function pointerElement(type: ModelType): ModelType {
  const target = unqualified(type);
  return target.kind === "pointer" ? target.element : target;
}

function requireArgument(
  args: ModelType[],
  index: number,
  context: string,
): ModelType {
  const argument = args[index];
  if (!argument)
    throw new Error(`${context}: missing template argument ${index}`);
  return argument;
}

function stripConst(type: ModelType): ModelType {
  return unqualified(type);
}

const keptArgumentCounts: Record<string, number> = {
  "::std::vector": 1,
  "::std::span": 1,
  "::std::deque": 1,
  "::std::optional": 1,
  "::std::unique_ptr": 1,
  "::std::shared_ptr": 1,
  "::std::unordered_map": 2,
  "::std::map": 2,
  "::std::unordered_set": 1,
  "::std::basic_string": 0,
};

function isStdContainer(name: string): boolean {
  return name in keptArgumentCounts;
}

function keptArguments(name: string, args: string[]): string[] {
  const count = keptArgumentCounts[name];
  return count === undefined ? args : args.slice(0, count);
}

export function normalizeClassName(name: string): string {
  if (name === "::std::basic_string") return "std::string";
  return name.startsWith("::") ? name.slice(2) : name;
}
