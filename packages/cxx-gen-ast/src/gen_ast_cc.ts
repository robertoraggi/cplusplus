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
import { groupNodesByBaseType } from "./groupNodesByBaseType.ts";
import type { AST } from "./parseAST.ts";
import * as fs from "fs";

export function gen_ast_cc({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_bases = groupNodesByBaseType(ast);

  const toKebapName = (name: string) =>
    name.replace(/([A-Z]+)/g, "-$1").toLocaleLowerCase();

  const astName = (name: string) => toKebapName(name.slice(0, -3)).slice(1);

  ast.nodes.forEach(({ name, members }) => {
    emit();
    emit(`auto ${name}::firstSourceLocation() -> SourceLocation {`);
    members.forEach((m) => {
      if (m.kind === "attribute") return;
      emit(`  if (auto loc = cxx::firstSourceLocation(${m.name})) return loc;`);
    });
    emit(`  return {};`);
    emit(`}`);

    emit();
    emit(`auto ${name}::lastSourceLocation() -> SourceLocation {`);
    [...members].reverse().forEach((m) => {
      if (m.kind === "attribute") return;
      emit(`  if (auto loc = cxx::lastSourceLocation(${m.name})) return loc;`);
    });
    emit(`  return {};`);
    emit(`}`);
  });

  emit();
  emit(`namespace {`);
  emit(`std::string_view kASTKindNames[] = {`);
  by_bases.forEach((nodes, base) => {
    emit();
    emit(`    // ${base}`);
    nodes.forEach(({ name }) => {
      emit(`    "${astName(name)}",`);
    });
  });
  emit(`};`);
  emit(`} // namespace`);

  emit(`auto to_string(ASTKind kind) -> std::string_view {`);
  emit(`  return kASTKindNames[int(kind)];`);
  emit(`}`);

  by_bases.forEach((nodes) => {
    nodes.forEach(({ name, base, members: classMembers }) => {

      const members = [...classMembers, ...ast.baseMembers.get(base) ?? []]

      let params: string[] = []
      let paramsNoLoc: string[] = []
      members.forEach((m) => {
        switch (m.kind) {
          case "node":
            params.push(`${m.type}* ${m.name}`)
            paramsNoLoc.push(`${m.type}* ${m.name}`)
            break;
          case "node-list":
            params.push(`List<${m.type}*>* ${m.name}`)
            paramsNoLoc.push(`List<${m.type}*>* ${m.name}`)
            break;
          case "token":
            params.push(`SourceLocation ${m.name}`);
            break;
          case "token-list":
            params.push(`List<SourceLocation>* ${m.name}`);
            break;
          case "attribute": {
            let s = "  ";
            if (m.cv) s = `${s}${m.cv} `;
            s = `${s}${m.type}${m.ptrOps} ${m.name}`;
            params.push(s);
            paramsNoLoc.push(s);
            break;
          }
        }
      });

      emit();
      emit(`auto ${name}::clone(Arena* arena) -> ${name}* {`);
      emit(`  auto node = create(arena);`);
      emit();
      members.forEach((m) => {
        switch (m.kind) {
          case "node":
            emit();
            emit(`  if (${m.name}) node->${m.name} = ${m.name}->clone(arena);`)
            emit();
            break;
          case "node-list":
            emit();
            emit(`  if (${m.name}) {`)
            emit(`    auto it = &node->${m.name};`)
            emit(`    for (auto node : ListView{${m.name}}) {`)
            emit(`      *it = make_list_node<${m.type}>(arena, node->clone(arena));`)
            emit(`      it = &(*it)->next;`)
            emit(`    }`)
            emit(`  }`)
            emit();
            break;
          case "token":
            emit(`  node->${m.name} = ${m.name};`)
            break;
          case "token-list":
            throw new Error("not implemented");
          case "attribute": {
            emit(`  node->${m.name} = ${m.name};`)
            break;
          }
        }
      });
      emit();
      emit(`  return node;`)
      emit(`}`);

      emit();
      emit(`auto ${name}::create(Arena* arena) -> ${name}* {`);
      emit(`  auto node = new (arena) ${name}();`);
      emit(`  return node;`);
      emit(`}`);

      if (params.length > 0) {
        const signature = params.join(", ");
        emit();
        emit(`auto ${name}::create(Arena* arena, ${signature}) -> ${name}* {`);
        emit(`  auto node = new (arena) ${name}();`);
        members.forEach((m) => {
          switch (m.kind) {
            case "node":
            case "node-list":
            case "token":
            case "token-list":
            case "attribute": {
              emit(`  node->${m.name} = ${m.name};`)
              break;
            }
          }
        });
        emit(`  return node;`);
        emit(`}`);
      }

      if (paramsNoLoc.length && paramsNoLoc.length != params.length) {
        const signature = paramsNoLoc.join(", ");
        emit();
        emit(`auto ${name}::create(Arena* arena, ${signature}) -> ${name}* {`);
        emit(`  auto node = new (arena) ${name}();`);
        members.forEach((m) => {
          switch (m.kind) {
            case "node":
            case "node-list":
            case "attribute":
              emit(`  node->${m.name} = ${m.name};`)
              break;
            default:
              break;
          } // switch
        });
        emit(`  return node;`);
        emit(`}`);
      }
    });
  });

  const out = `// Generated file by: gen_ast_cc.ts
${cpy_header}
#include <cxx/ast.h>

namespace cxx {

AST::~AST() = default;

${code.join("\n")}

auto to_string(ValueCategory valueCategory) -> std::string_view {
  switch (valueCategory) {
  case ValueCategory::kNone: return "none";
  case ValueCategory::kLValue: return "lvalue";
  case ValueCategory::kXValue: return "xvalue";
  case ValueCategory::kPrValue: return "prvalue";
  default: cxx_runtime_error("Invalid value category");
  } // switch
}

auto to_string(ImplicitCastKind implicitCastKind) -> std::string_view {
  switch (implicitCastKind) {
  case ImplicitCastKind::kIdentity: return "identity";
  case ImplicitCastKind::kLValueToRValueConversion: return "lvalue-to-rvalue-conversion";
  case ImplicitCastKind::kArrayToPointerConversion: return "array-to-pointer-conversion";
  case ImplicitCastKind::kFunctionToPointerConversion: return "function-to-pointer-conversion";
  case ImplicitCastKind::kIntegralPromotion: return "integral-promotion";
  case ImplicitCastKind::kFloatingPointPromotion: return "floating-point-promotion";
  case ImplicitCastKind::kIntegralConversion: return "integral-conversion";
  case ImplicitCastKind::kFloatingPointConversion: return "floating-point-conversion";
  case ImplicitCastKind::kFloatingIntegralConversion: return "floating-integral-conversion";
  case ImplicitCastKind::kPointerConversion: return "pointer-conversion";
  case ImplicitCastKind::kPointerToMemberConversion: return "pointer-to-member-conversion";
  case ImplicitCastKind::kBooleanConversion: return "boolean-conversion";
  case ImplicitCastKind::kFunctionPointerConversion: return "function-pointer-conversion";
  case ImplicitCastKind::kQualificationConversion: return "qualification-conversion";
  case ImplicitCastKind::kTemporaryMaterializationConversion: return "temporary-materialization-conversion";
  case ImplicitCastKind::kUserDefinedConversion: return "user-defined-conversion";
  default: cxx_runtime_error("Invalid implicit cast kind");
  } // switch
}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
