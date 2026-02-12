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

import { cpy_header } from "./cpy_header.ts";
import { BUILTINS } from "./builtins.ts";
import * as fs from "fs";

const converterFor: Record<string, string> = {
  float: "toFloat",
  double: "toDouble",
  "long double": "toLongDouble",
  int: "toInt",
  long: "toInt",
  "long long": "toInt",
};

const castArgFor: Record<string, string> = {
  int: "static_cast<int>",
  long: "static_cast<long>",
  "long long": "static_cast<long long>",
};

function isIntegralRet(ret: string): boolean {
  return ret === "int" || ret === "long" || ret === "long long";
}

function enumName(builtinName: string): string {
  return `BuiltinFunctionKind::T_${builtinName.toUpperCase()}`;
}

function emitMathCase(b: {
  name: string;
  eval: { fn: string; args: string[]; ret: string };
}): string[] {
  const lines: string[] = [];
  const { fn, args, ret } = b.eval;
  const caseName = enumName(b.name);
  const wrapRet = isIntegralRet(ret) ? "static_cast<std::intmax_t>" : "";

  if (args.length === 1) {
    const conv = converterFor[args[0]];
    const cast = castArgFor[args[0]] ?? "";
    const argExpr = cast ? `${cast}(*a)` : `*a`;
    const retExpr = wrapRet
      ? `${wrapRet}(${fn}(${argExpr}))`
      : `${fn}(${argExpr})`;
    lines.push(`    case ${caseName}: {`);
    lines.push(
      `      if (auto a = ${conv}(args[0])) return ConstValue{${retExpr}};`,
    );
    lines.push(`      return std::nullopt;`);
    lines.push(`    }`);
  } else {
    const varNames = ["a", "b", "c", "d"];
    const decls: string[] = [];
    const checks: string[] = [];
    const callArgs: string[] = [];

    for (let i = 0; i < args.length; i++) {
      const v = varNames[i];
      const conv = converterFor[args[i]];
      const cast = castArgFor[args[i]] ?? "";
      decls.push(`auto ${v} = ${conv}(args[${i}]);`);
      checks.push(v);
      callArgs.push(cast ? `${cast}(*${v})` : `*${v}`);
    }

    const retExpr = wrapRet
      ? `${wrapRet}(${fn}(${callArgs.join(", ")}))`
      : `${fn}(${callArgs.join(", ")})`;

    lines.push(`    case ${caseName}: {`);
    lines.push(`      ${decls.join(" ")}`);
    lines.push(
      `      if (${checks.join(" && ")}) return ConstValue{${retExpr}};`,
    );
    lines.push(`      return std::nullopt;`);
    lines.push(`    }`);
  }

  return lines;
}

function stringCmp(name: string, normalize: boolean): string[] {
  const caseName = enumName(name);
  const sign = normalize
    ? `static_cast<std::intmax_t>(r > 0 ? 1 : r < 0 ? -1 : 0)`
    : `static_cast<std::intmax_t>(r)`;
  return [
    `    case ${caseName}: {`,
    `      auto* a = std::get_if<const StringLiteral*>(&args[0]);`,
    `      auto* b = std::get_if<const StringLiteral*>(&args[1]);`,
    `      if (a && b && *a && *b) {`,
    `        auto sa = (*a)->stringValue();`,
    `        auto sb = (*b)->stringValue();`,
    `        int r = sa.compare(sb);`,
    `        return ConstValue{${sign}};`,
    `      }`,
    `      return std::nullopt;`,
    `    }`,
  ];
}

function stringNCmp(name: string): string[] {
  const caseName = enumName(name);
  return [
    `    case ${caseName}: {`,
    `      auto* a = std::get_if<const StringLiteral*>(&args[0]);`,
    `      auto* b = std::get_if<const StringLiteral*>(&args[1]);`,
    `      auto n = toInt(args[2]);`,
    `      if (a && b && *a && *b && n) {`,
    `        auto sa = (*a)->stringValue();`,
    `        auto sb = (*b)->stringValue();`,
    `        int r = sa.compare(0, static_cast<size_t>(*n), sb, 0, static_cast<size_t>(*n));`,
    `        return ConstValue{static_cast<std::intmax_t>(r > 0 ? 1 : r < 0 ? -1 : 0)};`,
    `      }`,
    `      return std::nullopt;`,
    `    }`,
  ];
}

function memCmp(name: string, boolResult: boolean): string[] {
  const caseName = enumName(name);
  const retExpr = boolResult
    ? `static_cast<std::intmax_t>(r != 0 ? 1 : 0)`
    : `static_cast<std::intmax_t>(r > 0 ? 1 : r < 0 ? -1 : 0)`;
  return [
    `    case ${caseName}: {`,
    `      auto* a = std::get_if<const StringLiteral*>(&args[0]);`,
    `      auto* b = std::get_if<const StringLiteral*>(&args[1]);`,
    `      auto n = toInt(args[2]);`,
    `      if (a && b && *a && *b && n) {`,
    `        auto sa = (*a)->stringValue();`,
    `        auto sb = (*b)->stringValue();`,
    `        auto len = static_cast<size_t>(*n);`,
    `        if (sa.size() >= len && sb.size() >= len) {`,
    `          int r = std::memcmp(sa.data(), sb.data(), len);`,
    `          return ConstValue{${retExpr}};`,
    `        }`,
    `      }`,
    `      return std::nullopt;`,
    `    }`,
  ];
}

function casecmp(name: string, withN: boolean): string[] {
  const excludeIfMsvc = [
    "__builtin_strcasecmp",
    "__builtin_strncasecmp",
  ].includes(name);

  const lines: string[] = [];

  if (excludeIfMsvc) {
    lines.push(`#ifndef _MSC_VER`);
  }

  const caseName = enumName(name);
  lines.push(
    `    case ${caseName}: {`,
    `      auto* a = std::get_if<const StringLiteral*>(&args[0]);`,
    `      auto* b = std::get_if<const StringLiteral*>(&args[1]);`,
  );
  if (withN) {
    lines.push(`      auto n = toInt(args[2]);`);
    lines.push(`      if (a && b && *a && *b && n) {`);
    lines.push(
      `        int r = strncasecmp(std::string((*a)->stringValue()).c_str(),`,
    );
    lines.push(
      `                            std::string((*b)->stringValue()).c_str(),`,
    );
    lines.push(`                            static_cast<size_t>(*n));`);
  } else {
    lines.push(`      if (a && b && *a && *b) {`);
    lines.push(
      `        int r = strcasecmp(std::string((*a)->stringValue()).c_str(),`,
    );
    lines.push(
      `                           std::string((*b)->stringValue()).c_str());`,
    );
  }
  lines.push(
    `        return ConstValue{static_cast<std::intmax_t>(r > 0 ? 1 : r < 0 ? -1 : 0)};`,
  );
  lines.push(`      }`);
  lines.push(`      return std::nullopt;`);
  lines.push(`    }`);

  if (excludeIfMsvc) {
    lines.push(`#endif`);
  }

  return lines;
}

function strspan(name: string, complement: boolean): string[] {
  const caseName = enumName(name);
  const func = complement ? "strcspn" : "strspn";
  return [
    `    case ${caseName}: {`,
    `      auto* a = std::get_if<const StringLiteral*>(&args[0]);`,
    `      auto* b = std::get_if<const StringLiteral*>(&args[1]);`,
    `      if (a && b && *a && *b) {`,
    `        auto sa = std::string((*a)->stringValue());`,
    `        auto sb = std::string((*b)->stringValue());`,
    `        return ConstValue{static_cast<std::intmax_t>(${func}(sa.c_str(), sb.c_str()))};`,
    `      }`,
    `      return std::nullopt;`,
    `    }`,
  ];
}

export function gen_builtins_interp_h({ output }: { output: string }) {
  const lines: string[] = [];

  lines.push(`// Generated by gen_builtins_interp_h.ts`);
  lines.push(cpy_header);
  lines.push(``);
  lines.push(`#include <cmath>`);
  lines.push(`#include <cstdlib>`);
  lines.push(`#include <cstring>`);
  lines.push(``);
  lines.push(
    `auto cxx::ASTInterpreter::evaluateBuiltinCall(cxx::BuiltinFunctionKind kind,`,
  );
  lines.push(
    `                                         std::vector<ConstValue> args)`,
  );
  lines.push(`    -> std::optional<ConstValue> {`);
  lines.push(`  switch (kind) {`);

  lines.push(`    case ${enumName("__builtin_is_constant_evaluated")}:`);
  lines.push(`      return ConstValue{std::intmax_t(1)};`);
  lines.push(``);
  lines.push(`    case ${enumName("__builtin_expect")}:`);
  lines.push(`      if (args.size() >= 1) return args[0];`);
  lines.push(`      return std::nullopt;`);
  lines.push(``);

  lines.push(`    case ${enumName("__builtin_bswap32")}: {`);
  lines.push(`      if (auto a = toInt(args[0])) {`);
  lines.push(`        auto v = static_cast<uint32_t>(*a);`);
  lines.push(
    `        v = ((v & 0xFF000000u) >> 24) | ((v & 0x00FF0000u) >> 8) |`,
  );
  lines.push(
    `            ((v & 0x0000FF00u) << 8) | ((v & 0x000000FFu) << 24);`,
  );
  lines.push(`        return ConstValue{static_cast<std::intmax_t>(v)};`);
  lines.push(`      }`);
  lines.push(`      return std::nullopt;`);
  lines.push(`    }`);
  lines.push(``);

  lines.push(`    case ${enumName("__builtin_bswap64")}: {`);
  lines.push(`      if (auto a = toInt(args[0])) {`);
  lines.push(`        auto v = static_cast<uint64_t>(*a);`);
  lines.push(`        v = ((v & 0xFF00000000000000ull) >> 56) |`);
  lines.push(`            ((v & 0x00FF000000000000ull) >> 40) |`);
  lines.push(`            ((v & 0x0000FF0000000000ull) >> 24) |`);
  lines.push(`            ((v & 0x000000FF00000000ull) >> 8) |`);
  lines.push(`            ((v & 0x00000000FF000000ull) << 8) |`);
  lines.push(`            ((v & 0x0000000000FF0000ull) << 24) |`);
  lines.push(`            ((v & 0x000000000000FF00ull) << 40) |`);
  lines.push(`            ((v & 0x00000000000000FFull) << 56);`);
  lines.push(`        return ConstValue{static_cast<std::intmax_t>(v)};`);
  lines.push(`      }`);
  lines.push(`      return std::nullopt;`);
  lines.push(`    }`);
  lines.push(``);

  lines.push(`    case ${enumName("__builtin_strlen")}: {`);
  lines.push(
    `      if (auto* lit = std::get_if<const StringLiteral*>(&args[0])) {`,
  );
  lines.push(`        if (*lit) {`);
  lines.push(
    `          return ConstValue{static_cast<std::intmax_t>((*lit)->stringValue().size())};`,
  );
  lines.push(`        }`);
  lines.push(`      }`);
  lines.push(`      return std::nullopt;`);
  lines.push(`    }`);
  lines.push(``);

  lines.push(...stringCmp("__builtin_strcmp", true));
  lines.push(``);
  lines.push(...stringNCmp("__builtin_strncmp"));
  lines.push(``);
  lines.push(...memCmp("__builtin_memcmp", false));
  lines.push(``);
  lines.push(...memCmp("__builtin_bcmp", true));
  lines.push(``);
  lines.push(...casecmp("__builtin_strcasecmp", false));
  lines.push(``);
  lines.push(...casecmp("__builtin_strncasecmp", true));
  lines.push(``);
  lines.push(...strspan("__builtin_strspn", false));
  lines.push(``);
  lines.push(...strspan("__builtin_strcspn", true));
  lines.push(``);

  let inCxx23Block = false;

  for (const b of BUILTINS) {
    if (!b.eval) continue;
    if (b.eval.cxx23 && !inCxx23Block) {
      lines.push(`#if __cplusplus >= 202302L`);
      inCxx23Block = true;
    } else if (!b.eval.cxx23 && inCxx23Block) {
      lines.push(`#endif`);
      inCxx23Block = false;
    }
    lines.push(...emitMathCase(b as any));
  }

  if (inCxx23Block) {
    lines.push(`#endif`);
  }

  lines.push(``);
  lines.push(`    default:`);
  lines.push(`      return std::nullopt;`);
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  fs.writeFileSync(output, lines.join("\n"));
}
