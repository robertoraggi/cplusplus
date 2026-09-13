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

import type { BuiltinDef } from "./builtins.ts";

export const BUILTIN_CONTROL_OPS = [
  "kEnd",
  "kVariadic",
  "kPointer",
  "kLvalueReference",
  "kRvalueReference",
  "kConst",
  "kVolatile",
  "kComplex",
] as const;

export interface BuiltinLeafType {
  op: string;
  spellings: string[];
  expr: string;
}

export const BUILTIN_LEAF_TYPES: BuiltinLeafType[] = [
  { op: "kVoid", spellings: ["void"], expr: "control->getVoidType()" },
  { op: "kBool", spellings: ["bool", "_Bool"], expr: "control->getBoolType()" },
  { op: "kChar", spellings: ["char"], expr: "control->getCharType()" },
  {
    op: "kSignedChar",
    spellings: ["signed char"],
    expr: "control->getSignedCharType()",
  },
  {
    op: "kUnsignedChar",
    spellings: ["unsigned char"],
    expr: "control->getUnsignedCharType()",
  },
  {
    op: "kShortInt",
    spellings: ["short", "short int", "signed short"],
    expr: "control->getShortIntType()",
  },
  {
    op: "kUnsignedShortInt",
    spellings: ["unsigned short", "unsigned short int"],
    expr: "control->getUnsignedShortIntType()",
  },
  {
    op: "kInt",
    spellings: ["int", "signed", "signed int"],
    expr: "control->getIntType()",
  },
  {
    op: "kUnsignedInt",
    spellings: ["unsigned", "unsigned int"],
    expr: "control->getUnsignedIntType()",
  },
  {
    op: "kLongInt",
    spellings: ["long", "long int", "signed long"],
    expr: "control->getLongIntType()",
  },
  {
    op: "kUnsignedLongInt",
    spellings: ["unsigned long", "unsigned long int"],
    expr: "control->getUnsignedLongIntType()",
  },
  {
    op: "kLongLongInt",
    spellings: ["long long", "long long int", "signed long long"],
    expr: "control->getLongLongIntType()",
  },
  {
    op: "kUnsignedLongLongInt",
    spellings: ["unsigned long long", "unsigned long long int"],
    expr: "control->getUnsignedLongLongIntType()",
  },
  { op: "kInt128", spellings: ["__int128"], expr: "control->getInt128Type()" },
  {
    op: "kUnsignedInt128",
    spellings: ["unsigned __int128"],
    expr: "control->getUnsignedInt128Type()",
  },
  { op: "kFloat", spellings: ["float"], expr: "control->getFloatType()" },
  { op: "kDouble", spellings: ["double"], expr: "control->getDoubleType()" },
  {
    op: "kLongDouble",
    spellings: ["long double"],
    expr: "control->getLongDoubleType()",
  },
  {
    op: "kFloat16",
    spellings: ["_Float16"],
    expr: "control->getFloat16Type()",
  },
  {
    op: "kWideChar",
    spellings: ["wchar_t"],
    expr: "control->getWideCharType()",
  },
  { op: "kChar8", spellings: ["char8_t"], expr: "control->getChar8Type()" },
  { op: "kChar16", spellings: ["char16_t"], expr: "control->getChar16Type()" },
  { op: "kChar32", spellings: ["char32_t"], expr: "control->getChar32Type()" },
  {
    op: "kNullptr",
    spellings: ["decltype(nullptr)"],
    expr: "control->getNullptrType()",
  },
  {
    op: "kSizeType",
    spellings: ["__SIZE_TYPE__", "size_t"],
    expr: "control->getSizeType()",
  },
  {
    op: "kBuiltinVaList",
    spellings: ["__builtin_va_list"],
    expr: "control->getBuiltinVaListType()",
  },
];

export const BUILTIN_OPS: string[] = [
  ...BUILTIN_CONTROL_OPS,
  ...BUILTIN_LEAF_TYPES.map((leaf) => leaf.op),
];

const OP_VALUE = new Map<string, number>(
  BUILTIN_OPS.map((op, index): [string, number] => [op, index]),
);

const LEAF_BY_SPELLING = new Map<string, string>(
  BUILTIN_LEAF_TYPES.flatMap(({ op, spellings }) =>
    spellings.map((spelling): [string, string] => [spelling, op]),
  ),
);

export function opValue(op: string): number {
  const value = OP_VALUE.get(op);
  if (value === undefined) throw new Error(`unknown builtin type op '${op}'`);
  return value;
}

export interface BuiltinPrototype {
  name: string;
  returnType: string;
  parameterTypes: string[];
  isVariadic: boolean;
}

export function parsePrototype(prototype: string): BuiltinPrototype {
  const match = prototype.match(
    /^(?:constexpr\s+)?(.*?)\s*\b([A-Za-z_]\w*)\s*\((.*)\)$/,
  );

  if (!match) throw new Error(`cannot parse builtin prototype '${prototype}'`);

  const returnType = match[1] ?? "";
  const name = match[2] ?? "";

  const parameterTypes = (match[3] ?? "")
    .split(",")
    .map((parameter) => parameter.trim())
    .filter((parameter) => parameter.length > 0);

  const isVariadic = parameterTypes.at(-1) === "...";

  if (isVariadic) parameterTypes.pop();

  if (parameterTypes.includes("...")) {
    throw new Error(`'...' must come last in '${prototype}'`);
  }

  return { name, returnType, parameterTypes, isVariadic };
}

export function encodeType(spelling: string): string[] {
  let text = spelling.trim();

  const declarators: string[] = [];

  while (true) {
    if (text.endsWith("&&")) {
      declarators.push("kRvalueReference");
      text = text.slice(0, -2).trim();
    } else if (text.endsWith("&")) {
      declarators.push("kLvalueReference");
      text = text.slice(0, -1).trim();
    } else if (text.endsWith("*")) {
      declarators.push("kPointer");
      text = text.slice(0, -1).trim();
    } else {
      break;
    }
  }

  const qualifiers: string[] = [];

  while (true) {
    if (text.startsWith("const ")) {
      qualifiers.push("kConst");
      text = text.slice(6).trim();
    } else if (text.startsWith("volatile ")) {
      qualifiers.push("kVolatile");
      text = text.slice(9).trim();
    } else {
      break;
    }
  }

  const complex: string[] = [];

  if (text.startsWith("_Complex ")) {
    complex.push("kComplex");
    text = text.slice(9).trim();
  }

  const leaf = LEAF_BY_SPELLING.get(text);

  if (!leaf) throw new Error(`no builtin type op for '${spelling}'`);

  return [...declarators, ...qualifiers.sort(), ...complex, leaf];
}

export function prototypesOf(builtin: BuiltinDef): string[] {
  const prototype = builtin.prototype;
  if (prototype === undefined) return [];
  return Array.isArray(prototype) ? prototype : [prototype];
}

export function encodeOne(prototype: string): string[] {
  const { returnType, parameterTypes, isVariadic } = parsePrototype(prototype);

  const ops = [returnType, ...parameterTypes].flatMap(encodeType);

  if (isVariadic) ops.push("kVariadic");

  ops.push("kEnd");

  return ops;
}

export function encodeSignature(builtin: BuiltinDef): string[] {
  return prototypesOf(builtin).flatMap(encodeOne);
}
