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

import type { Wire } from "./semanticWire.ts";

export class Names {
  private next = 0;

  fresh(prefix: string): string {
    return `${prefix}${++this.next}`;
  }
}

export function codecName(name: string): string {
  return name
    .replace(/^::/, "")
    .replace(/::/g, "")
    .replace(/[^A-Za-z0-9_]/g, "_");
}

export function substitute(template: string, self: string, value: string) {
  return template
    .replaceAll("$element", value)
    .replaceAll("$value", value)
    .replaceAll("$->", `${self}->`)
    .replaceAll("$", self);
}

const scalarWriters: Record<string, string> = {
  bool: "boolean",
  u32: "u32",
  i32: "i32",
  u64: "u64",
  i64: "i64",
  f32: "f32",
  f64: "f64",
  f80: "f80",
  string: "str",
};

const scalarCasts: Record<string, string> = {
  u32: "std::uint32_t",
  i32: "std::int32_t",
  u64: "std::uint64_t",
  i64: "std::int64_t",
};

export function emitWrite(
  lines: string[],
  indent: string,
  wire: Wire,
  value: string,
  names: Names,
): void {
  const push = (text: string) => lines.push(`${indent}${text}`);

  switch (wire.k) {
    case "bool":
    case "f32":
    case "f64":
    case "f80":
      push(`out.${scalarWriters[wire.k]}(${value});`);
      return;

    case "string":
      push(`out.varU32(static_cast<std::uint32_t>(stringRef(${value})));`);
      return;

    case "u32":
      push(`out.varU32(static_cast<std::uint32_t>(${value}));`);
      return;

    case "u64":
      push(`out.varU64(static_cast<std::uint64_t>(${value}));`);
      return;

    case "i32":
      push(`out.varI32(static_cast<std::int32_t>(${value}));`);
      return;

    case "i64":
      push(`out.varI64(static_cast<std::int64_t>(${value}));`);
      return;

    case "enum":
      push(`out.varU32(static_cast<std::uint32_t>(${value}));`);
      return;

    case "location":
      push(`out.varU32(static_cast<std::uint32_t>(locationRef(${value})));`);
      return;

    case "name":
      push(`out.varU32(static_cast<std::uint32_t>(nameRef(${value})));`);
      return;

    case "type":
      push(`out.varU32(static_cast<std::uint32_t>(typeRef(${value})));`);
      return;

    case "symbol":
      push(`out.varU32(static_cast<std::uint32_t>(symbolRef(${value})));`);
      return;

    case "ast":
      push(`out.varU32(static_cast<std::uint32_t>(astRef(${value})));`);
      return;

    case "ast-list":
      push(`writeAstList(out, ${value});`);
      return;

    case "identifier":
      push(`out.varU32(static_cast<std::uint32_t>(identifierRef(${value})));`);
      return;

    case "literal":
      push(`writeLiteral(out, ${value});`);
      return;

    case "abi-tags":
      push(`writeAbiTags(out, ${value});`);
      return;

    case "attributes":
      push(`writeAttributes(out, ${value});`);
      return;

    case "const-value":
      push(`writeConstValue(out, ${value});`);
      return;

    case "const-value-ptr":
      push(`out.boolean(${value} != nullptr);`);
      push(`if (${value}) writeConstValue(out, *${value});`);
      return;

    case "template-argument":
      push(`writeTemplateArgument(out, ${value});`);
      return;

    case "shared":
      push(`out.varU32(static_cast<std::uint32_t>(constRef(${value})));`);
      return;

    case "indeterminate":
    case "unit":
    case "identifier-info":
      return;

    case "struct":
      push(`write${codecName(wire.name)}(out, &${value});`);
      return;

    case "unique": {
      push(`out.boolean(${value} != nullptr);`);
      push(`if (${value}) {`);
      emitWrite(lines, `${indent}  `, wire.element, `(*${value})`, names);
      push(`}`);
      return;
    }

    case "optional": {
      push(`out.boolean(${value}.has_value());`);
      push(`if (${value}.has_value()) {`);
      emitWrite(lines, `${indent}  `, wire.element, `(*${value})`, names);
      push(`}`);
      return;
    }

    case "vector":
    case "deque": {
      const element = names.fresh("element");
      push(
        `out.varU32(static_cast<std::uint32_t>(std::ranges::size(${value})));`,
      );
      push(`for (const auto& ${element} : ${value}) {`);
      emitWrite(lines, `${indent}  `, wire.element, element, names);
      push(`}`);
      return;
    }

    case "pair":
      emitWrite(lines, indent, wire.first, `${value}.first`, names);
      emitWrite(lines, indent, wire.second, `${value}.second`, names);
      return;

    case "tuple":
      wire.elements.forEach((element, position) => {
        emitWrite(
          lines,
          indent,
          element,
          `std::get<${position}>(${value})`,
          names,
        );
      });
      return;

    case "variant": {
      push(`out.u8(static_cast<std::uint8_t>(${value}.index()));`);
      push(`switch (${value}.index()) {`);
      wire.alternatives.forEach((alternative, position) => {
        push(`  case ${position}: {`);
        emitWrite(
          lines,
          `${indent}    `,
          alternative,
          `std::get<${position}>(${value})`,
          names,
        );
        push(`    break;`);
        push(`  }`);
      });
      push(`}`);
      return;
    }

    case "map": {
      const entries = names.fresh("entries");
      const entry = names.fresh("entry");
      push(`{`);
      push(`  using MapT${entries} = std::remove_cvref_t<decltype(${value})>;`);
      push(`  std::vector<std::pair<typename MapT${entries}::key_type,`);
      push(
        `                        typename MapT${entries}::mapped_type>> ${entries};`,
      );
      push(`  for (const auto& ${entry} : ${value})`);
      push(`    ${entries}.emplace_back(${entry}.first, ${entry}.second);`);
      push(
        `  std::ranges::sort(${entries}, [this](const auto& lhs, const auto& rhs) {`,
      );
      push(`    return entrySortKey(lhs.first) < entrySortKey(rhs.first);`);
      push(`  });`);
      push(`  out.varU32(static_cast<std::uint32_t>(${entries}.size()));`);
      push(`  for (const auto& ${entry} : ${entries}) {`);
      emitWrite(lines, `${indent}    `, wire.key, `${entry}.first`, names);
      emitWrite(lines, `${indent}    `, wire.value, `${entry}.second`, names);
      push(`  }`);
      push(`}`);
      return;
    }
  }
}

/**
 * Declares `target` of type `cppType` and fills it from the reader. Every
 * bounds and range check lives in the reader helpers, so a corrupt archive
 * leaves the decoder failed rather than the graph half-built.
 */
export function derivesLocalType(wire: Wire): boolean {
  switch (wire.k) {
    case "vector":
      return !wire.view;
    case "deque":
    case "unique":
    case "map":
    case "variant":
      return true;
    default:
      return false;
  }
}

/**
 * The type of the local a value is read into. A declared type that is a view or
 * a reference — `std::string_view` on an interning factory's parameter, say —
 * would dangle the moment the reader's temporary died, so the local always owns
 * its value.
 */
export function localTypeOf(wire: Wire, declared: string): string {
  if (wire.k === "string") return "std::string";
  if (wire.k === "vector" && wire.view) return wire.cpp;
  return declared.replace(/\s*&+$/, "");
}

export function emitRead(
  lines: string[],
  indent: string,
  wire: Wire,
  declaredType: string,
  target: string,
  names: Names,
): void {
  const push = (text: string) => lines.push(`${indent}${text}`);
  const cppType = localTypeOf(wire, declaredType);

  switch (wire.k) {
    case "bool":
      push(`${cppType} ${target} = in.boolean();`);
      return;

    case "string":
      push(`${cppType} ${target}{stringAt(StringRef{in.varU32()})};`);
      return;

    case "f32":
      push(`${cppType} ${target} = in.f32();`);
      return;

    case "f64":
      push(`${cppType} ${target} = in.f64();`);
      return;

    case "f80":
      push(`${cppType} ${target} = in.f80();`);
      return;

    case "u32":
      push(`${cppType} ${target} = static_cast<${cppType}>(in.varU32());`);
      return;

    case "u64":
      push(`${cppType} ${target} = static_cast<${cppType}>(in.varU64());`);
      return;

    case "i32":
      push(`${cppType} ${target} = static_cast<${cppType}>(in.varI32());`);
      return;

    case "i64":
      push(`${cppType} ${target} = static_cast<${cppType}>(in.varI64());`);
      return;

    case "enum":
      if (wire.last) {
        push(
          `static_assert(static_cast<std::uint32_t>(${wire.name}::${wire.last}) + 1 ==` +
            ` ${wire.count});`,
        );
      }
      push(
        `${cppType} ${target} = static_cast<${cppType}>(readEnum(in, ${wire.count}));`,
      );
      return;

    case "location":
      push(`${cppType} ${target} = locationAt(LocationRef{in.varU32()});`);
      return;

    case "name":
      push(
        `${cppType} ${target} = ${downcast("name", wire, "nameAt(NameRef{in.varU32()})")};`,
      );
      return;

    case "type":
      push(
        `${cppType} ${target} = ${downcast("type", wire, "typeAt(TypeRef{in.varU32()})")};`,
      );
      return;

    case "symbol":
      push(
        `${cppType} ${target} = ${downcast("symbol", wire, "symbolAt(SymbolRef{in.varU32()})")};`,
      );
      return;

    case "ast":
      push(
        `${cppType} ${target} = ${downcast("ast", wire, "astAt(AstRef{in.varU32()})")};`,
      );
      return;

    case "ast-list":
      push(`${cppType} ${target} = readAstList<${wire.element}>(in);`);
      return;

    case "identifier":
      push(`${cppType} ${target} = identifierAt(StringRef{in.varU32()});`);
      return;

    case "literal":
      push(`${cppType} ${target} = read${literalReader(wire.cpp)}(in);`);
      return;

    case "abi-tags":
      push(`${cppType} ${target} = readAbiTags(in);`);
      return;

    case "attributes":
      push(`${cppType} ${target} = readAttributes(in);`);
      return;

    case "const-value":
      push(`${cppType} ${target} = readConstValue(in);`);
      return;

    case "const-value-ptr":
      push(`${cppType} ${target} = nullptr;`);
      push(`if (in.boolean())`);
      push(`  ${target} = arena()->make<cxx::ConstValue>(readConstValue(in));`);
      return;

    case "template-argument":
      push(`${cppType} ${target} = readTemplateArgument(in);`);
      return;

    case "shared":
      push(
        `${cppType} ${target} = std::static_pointer_cast<${wire.cpp}>(constantAt(ConstRef{in.varU32()}));`,
      );
      return;

    case "indeterminate":
      push(`${cppType} ${target}{};`);
      return;

    case "unit":
      push(`${cppType} ${target} = unit();`);
      return;

    case "identifier-info":
      push(`${cppType} ${target} = nullptr;`);
      return;

    case "struct":
      push(`${cppType} ${target}{};`);
      push(`read${codecName(wire.name)}(in, &${target});`);
      return;

    case "unique": {
      const element = nestedTypeOf(wire, cppType, target, "element_type");
      push(`${cppType} ${target};`);
      push(`if (in.boolean()) {`);
      push(`  ${target} = std::make_unique<${element}>();`);
      if (wire.element.k === "struct") {
        push(`  read${codecName(wire.element.name)}(in, ${target}.get());`);
      } else {
        const temp = names.fresh("value");
        emitRead(lines, `${indent}  `, wire.element, element, temp, names);
        push(`  *${target} = std::move(${temp});`);
      }
      push(`}`);
      return;
    }

    case "optional": {
      const temp = names.fresh("value");
      push(`${cppType} ${target};`);
      push(`if (in.boolean()) {`);
      emitRead(
        lines,
        `${indent}  `,
        wire.element,
        nestedTypeOf(wire, cppType, target, "value_type"),
        temp,
        names,
      );
      push(`  ${target} = std::move(${temp});`);
      push(`}`);
      return;
    }

    case "vector":
    case "deque": {
      const count = names.fresh("count");
      const position = names.fresh("i");
      const temp = names.fresh("element");
      push(`${cppType} ${target};`);
      push(`{`);
      push(`  const auto ${count} = in.varCount(1);`);
      push(
        `  for (std::uint32_t ${position} = 0; ok() && ${position} < ${count}; ++${position}) {`,
      );
      emitRead(
        lines,
        `${indent}    `,
        wire.element,
        nestedTypeOf(wire, cppType, target, "value_type"),
        temp,
        names,
      );
      push(`    ${target}.push_back(std::move(${temp}));`);
      push(`  }`);
      push(`}`);
      return;
    }

    case "pair": {
      const first = names.fresh("first");
      const second = names.fresh("second");
      push(`${cppType} ${target};`);
      push(`{`);
      emitRead(
        lines,
        `${indent}  `,
        wire.first,
        `decltype(${target}.first)`,
        first,
        names,
      );
      emitRead(
        lines,
        `${indent}  `,
        wire.second,
        `decltype(${target}.second)`,
        second,
        names,
      );
      push(`  ${target} = {std::move(${first}), std::move(${second})};`);
      push(`}`);
      return;
    }

    case "tuple": {
      const parts: string[] = [];
      push(`${cppType} ${target};`);
      push(`{`);
      wire.elements.forEach((element, position) => {
        const temp = names.fresh("item");
        parts.push(temp);
        emitRead(
          lines,
          `${indent}  `,
          element,
          `std::tuple_element_t<${position}, decltype(${target})>`,
          temp,
          names,
        );
      });
      push(
        `  ${target} = decltype(${target}){${parts.map((part) => `std::move(${part})`).join(", ")}};`,
      );
      push(`}`);
      return;
    }

    case "variant": {
      const tag = names.fresh("tag");
      push(`${cppType} ${target};`);
      push(`{`);
      push(`  const auto ${tag} = in.u8();`);
      push(`  switch (${tag}) {`);
      wire.alternatives.forEach((alternative, position) => {
        const temp = names.fresh("alternative");
        push(`    case ${position}: {`);
        emitRead(
          lines,
          `${indent}      `,
          alternative,
          `std::variant_alternative_t<${position}, decltype(${target})>`,
          temp,
          names,
        );
        push(`      ${target} = std::move(${temp});`);
        push(`      break;`);
        push(`    }`);
      });
      push(`    default:`);
      push(`      fail("unknown variant alternative");`);
      push(`      break;`);
      push(`  }`);
      push(`}`);
      return;
    }

    case "map": {
      const count = names.fresh("count");
      const position = names.fresh("i");
      const key = names.fresh("key");
      const value = names.fresh("value");
      push(`${cppType} ${target};`);
      push(`{`);
      push(`  const auto ${count} = in.varCount(1);`);
      push(
        `  for (std::uint32_t ${position} = 0; ok() && ${position} < ${count}; ++${position}) {`,
      );
      emitRead(
        lines,
        `${indent}    `,
        wire.key,
        `decltype(${target})::key_type`,
        key,
        names,
      );
      emitRead(
        lines,
        `${indent}    `,
        wire.value,
        `decltype(${target})::mapped_type`,
        value,
        names,
      );
      push(`    ${target}.emplace(${key}, std::move(${value}));`);
      push(`  }`);
      push(`}`);
      return;
    }
  }
}

/**
 * `Literal` has no kind discriminator, and every field declared with the base
 * type holds a string literal (an asm string, a `static_assert` message, a
 * literal-operator suffix), so the base reader interns a string literal.
 */
export function literalReader(cpp: string): string {
  switch (cpp) {
    case "cxx::CharLiteral":
      return "CharLiteral";
    case "cxx::IntegerLiteral":
      return "IntegerLiteral";
    case "cxx::FloatLiteral":
      return "FloatLiteral";
    default:
      return "StringLiteral";
  }
}

function downcast(domain: string, wire: Wire, expression: string): string {
  const base: Record<string, string> = {
    name: "cxx::Name",
    type: "cxx::Type",
    symbol: "cxx::Symbol",
    ast: "cxx::AST",
  };

  const caster: Record<string, string> = {
    name: "name_cast",
    type: "type_cast",
    symbol: "symbol_cast",
    ast: "ast_cast",
  };

  const cpp = "cpp" in wire ? (wire as { cpp: string }).cpp : base[domain]!;

  if (cpp === base[domain]) return expression;

  return `${caster[domain]}<${cpp.replace(/^cxx::/, "")}>(${expression})`;
}

function elementTypeOf(cppType: string): string {
  const open = cppType.indexOf("<");
  const close = cppType.lastIndexOf(">");
  if (open < 0 || close < 0) return cppType;
  return cppType.slice(open + 1, close).trim();
}

function nestedTypeOf(
  wire: Wire,
  cppType: string,
  target: string,
  member: string,
): string {
  if ("cpp" in wire && cppType === wire.cpp) return elementTypeOf(cppType);
  return `decltype(${target})::${member}`;
}
