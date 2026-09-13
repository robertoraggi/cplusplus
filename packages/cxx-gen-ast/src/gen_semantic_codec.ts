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
import { cpy_header } from "./cpy_header.ts";
import type { CodecPlan, EntityPlan, FieldPlan } from "./semanticPlan.ts";
import {
  Names,
  codecName,
  derivesLocalType,
  emitRead,
  emitWrite,
  substitute,
} from "./semanticEmit.ts";

const constAlternatives = [
  {
    tag: 0,
    cpp: "std::intmax_t",
    kind: "scalar",
    writer: "i64",
    reader: "i64",
  },
  { tag: 1, cpp: "const cxx::StringLiteral*", kind: "literal" },
  { tag: 2, cpp: "float", kind: "scalar", writer: "f32", reader: "f32" },
  { tag: 3, cpp: "double", kind: "scalar", writer: "f64", reader: "f64" },
  { tag: 4, cpp: "long double", kind: "scalar", writer: "f80", reader: "f80" },
  { tag: 5, cpp: "cxx::Meta", kind: "shared" },
  { tag: 6, cpp: "cxx::InitializerList", kind: "shared" },
  { tag: 7, cpp: "cxx::ConstObject", kind: "shared" },
  { tag: 8, cpp: "cxx::ConstAddress", kind: "shared" },
  { tag: 9, cpp: "cxx::ConstLabelAddress", kind: "shared" },
  { tag: 10, cpp: "cxx::IndeterminateValue", kind: "indeterminate" },
] as const;

const sharedConstKinds = constAlternatives
  .filter((alternative) => alternative.kind === "shared")
  .map((alternative) => alternative.cpp);

export function gen_semantic_codec({
  plan,
  headerOutput,
  sourceOutput,
}: {
  plan: CodecPlan;
  headerOutput: string;
  sourceOutput: string;
}) {
  fs.writeFileSync(headerOutput, header(plan));
  fs.writeFileSync(sourceOutput, source(plan));
}

function allEntities(plan: CodecPlan): EntityPlan[] {
  return [
    ...plan.names,
    ...plan.types,
    ...plan.symbols,
    ...plan.nodes,
    ...plan.structs,
  ];
}

function header(plan: CodecPlan): string {
  const encoderMembers: string[] = [];
  const decoderMembers: string[] = [];

  for (const entity of plan.names) {
    encoderMembers.push(
      `  void writeName${entity.short}(ByteWriter& out, const ${entity.cpp}* self);`,
    );
    decoderMembers.push(
      `  [[nodiscard]] auto readName${entity.short}(ByteReader& in) -> const cxx::Name*;`,
    );
  }

  for (const entity of plan.types) {
    encoderMembers.push(
      `  void writeType${entity.short}(ByteWriter& out, const ${entity.cpp}* self);`,
    );
    decoderMembers.push(
      `  [[nodiscard]] auto readType${entity.short}(ByteReader& in) -> const cxx::Type*;`,
    );
  }

  for (const entity of plan.symbols) {
    encoderMembers.push(
      `  void writeSymbol${entity.short}(ByteWriter& out, ${entity.cpp}* self);`,
    );
    decoderMembers.push(
      `  void readSymbol${entity.short}(ByteReader& in, ${entity.cpp}* self);`,
    );
  }

  for (const entity of plan.nodes) {
    encoderMembers.push(
      `  void writeAst${entity.short}(ByteWriter& out, ${entity.cpp}* self);`,
    );
    decoderMembers.push(
      `  void readAst${entity.short}(ByteReader& in, ${entity.cpp}* self);`,
    );
  }

  for (const entity of plan.structs) {
    const name = codecName(entity.name);
    encoderMembers.push(
      `  void write${name}(ByteWriter& out, const ${entity.cpp}* self);`,
    );
    decoderMembers.push(
      `  void read${name}(ByteReader& in, ${entity.cpp}* self);`,
    );
  }

  return `// Generated file by: gen_semantic_codec.ts
${cpy_header}
#pragma once

#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/const_value.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/semantic_archive.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <format>
#include <memory>
#include <ranges>
#include <string>
#include <type_traits>
#include <vector>

namespace cxx {

class SemanticEncoder final : public SemanticEncoderBase {
 public:
  explicit SemanticEncoder(TranslationUnit* unit) : SemanticEncoderBase(unit) {}

  [[nodiscard]] auto operator()(const SemanticArchiveRoots& roots,
                                ArchiveWriter& archive) -> bool;

 private:
  [[nodiscard]] auto nameRef(const cxx::Name* name) -> NameRef {
    return NameRef{names_.reference(name)};
  }

  [[nodiscard]] auto typeRef(const cxx::Type* type) -> TypeRef {
    return TypeRef{types_.reference(type)};
  }

  [[nodiscard]] auto symbolRef(const cxx::Symbol* symbol) -> SymbolRef {
    return SymbolRef{symbols_.reference(const_cast<cxx::Symbol*>(symbol))};
  }

  [[nodiscard]] auto astRef(const cxx::AST* ast) -> AstRef {
    return AstRef{nodes_.reference(const_cast<cxx::AST*>(ast))};
  }

  [[nodiscard]] auto identifierRef(const cxx::Identifier* identifier)
      -> StringRef {
    if (!identifier) return StringRef{0};
    return stringRef(identifier->name());
  }

${sharedConstKinds
  .map(
    (cpp, index) => `  [[nodiscard]] auto constRef(
      const std::shared_ptr<${cpp}>& value) -> ConstRef {
    return ConstRef{constNodeRef(value, ${index})};
  }
`,
  )
  .join("\n")}
  template <typename T>
  void writeAstList(ByteWriter& out, cxx::List<T*>* list) {
    std::uint32_t count = 0;
    for (auto node : cxx::ListView{list}) {
      (void)node;
      ++count;
    }
    out.varU32(count);
    for (auto node : cxx::ListView{list})
      out.varU32(static_cast<std::uint32_t>(astRef(node)));
  }

  void writeLiteral(ByteWriter& out, const cxx::Literal* literal);
  void writeAbiTags(ByteWriter& out,
                    const std::vector<const cxx::Identifier*>* tags);
  void writeAttributes(ByteWriter& out, const cxx::AttributeMap* attributes);
  void writeConstValue(ByteWriter& out, const cxx::ConstValue& value);
  void writeTemplateArgument(ByteWriter& out,
                             const cxx::TemplateArgument& argument);

  void writeName(ByteWriter& out, const cxx::Name* name);
  void writeType(ByteWriter& out, const cxx::Type* type);
  void writeSymbol(ByteWriter& out, cxx::Symbol* symbol);
  void writeAst(ByteWriter& out, cxx::AST* ast);
  void writeConstNode(ByteWriter& out, const ConstNode& node);

  void drain();

${encoderMembers.join("\n")}
};

class SemanticDecoder final : public SemanticDecoderBase {
 public:
  explicit SemanticDecoder(TranslationUnit* unit) : SemanticDecoderBase(unit) {}

  [[nodiscard]] auto operator()(const ArchiveReader& archive,
                                SemanticArchiveRoots& roots) -> bool;

 private:
  [[nodiscard]] auto nameAt(NameRef ref) -> const cxx::Name*;
  [[nodiscard]] auto typeAt(TypeRef ref) -> const cxx::Type*;
  [[nodiscard]] auto symbolAt(SymbolRef ref) -> cxx::Symbol*;
  [[nodiscard]] auto astAt(AstRef ref) -> cxx::AST*;
  [[nodiscard]] auto constantAt(ConstRef ref) -> std::shared_ptr<void>;

  [[nodiscard]] auto readEnum(ByteReader& in, std::uint32_t count)
      -> std::uint32_t;
  [[nodiscard]] auto readAbiTags(ByteReader& in)
      -> const std::vector<const cxx::Identifier*>*;
  [[nodiscard]] auto readAttributes(ByteReader& in)
      -> const cxx::AttributeMap*;
  [[nodiscard]] auto readConstValue(ByteReader& in) -> cxx::ConstValue;
  [[nodiscard]] auto readTemplateArgument(ByteReader& in)
      -> cxx::TemplateArgument;

  template <typename T>
  [[nodiscard]] auto readAstList(ByteReader& in) -> cxx::List<T*>* {
    cxx::List<T*>* result = nullptr;
    auto tail = &result;
    const auto count = in.varCount(1);
    if (count > nodeRecords_.size()) {
      fail(std::format("AST list of {} elements exceeds the {} nodes in the archive",
                       count, nodeRecords_.size()));
      return nullptr;
    }
    for (std::uint32_t i = 0; ok() && i < count; ++i) {
      auto element = ast_cast<T>(astAt(AstRef{in.varU32()}));
      *tail = new (arena()) cxx::List<T*>(element);
      tail = &(*tail)->next;
    }
    return result;
  }

  [[nodiscard]] auto allocateSymbol(cxx::SymbolKind kind) -> cxx::Symbol*;
  [[nodiscard]] auto allocateAst(cxx::ASTKind kind) -> cxx::AST*;

  void decodeSymbolFields(ByteReader& in, cxx::Symbol* symbol);
  void decodeAstFields(ByteReader& in, cxx::AST* ast);

${decoderMembers.join("\n")}
};

}  // namespace cxx
`;
}

function source(plan: CodecPlan): string {
  const out: string[] = [];
  const emit = (line = "") => out.push(line);

  emit(encoderEntryPoint(plan));
  emit(encoderDispatch(plan));
  emit(encoderHelpers());
  for (const entity of plan.names) emit(encodeFactoryEntity(entity, "Name"));
  for (const entity of plan.types) emit(encodeFactoryEntity(entity, "Type"));
  for (const entity of plan.symbols) emit(encodeFieldEntity(entity, "Symbol"));
  for (const entity of plan.nodes) emit(encodeFieldEntity(entity, "Ast"));
  for (const entity of plan.structs) emit(encodeStructEntity(entity));

  emit(decoderEntryPoint(plan));
  emit(decoderDispatch(plan));
  emit(decoderHelpers(plan));
  for (const entity of plan.names) emit(decodeFactoryEntity(entity, "Name"));
  for (const entity of plan.types) emit(decodeFactoryEntity(entity, "Type"));
  for (const entity of plan.symbols) emit(decodeFieldEntity(entity, "Symbol"));
  for (const entity of plan.nodes) emit(decodeFieldEntity(entity, "Ast"));
  for (const entity of plan.structs) emit(decodeStructEntity(entity));

  return `// Generated file by: gen_semantic_codec.ts
${cpy_header}
#include <cxx/private/semantic_codec.h>

#include <cxx/arena.h>
#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/time_trace.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>
#include <format>
#include <ranges>

namespace cxx {

namespace {

auto sectionDetail(const ByteReader& section) -> std::string {
  return std::format("{} bytes", section.remaining());
}

auto countDetail(std::size_t count) -> std::string {
  return std::format("{} records", count);
}

}  // namespace

${out.join("\n")}

}  // namespace cxx
`;
}

function encoderEntryPoint(plan: CodecPlan): string {
  const queues = [
    "pendingBodyCompletions",
    "pendingMemberInstantiations",
    "instantiatedMemberClasses",
  ];

  const lines: string[] = [];

  lines.push(
    `auto SemanticEncoder::operator()(const SemanticArchiveRoots& roots,`,
  );
  lines.push(
    `                                 ArchiveWriter& archive) -> bool {`,
  );
  lines.push(`  ByteWriter session;`);
  lines.push(``);
  lines.push(
    `  session.varU32(static_cast<std::uint32_t>(symbolRef(roots.globalScope)));`,
  );
  lines.push(
    `  session.varU32(static_cast<std::uint32_t>(astRef(roots.ast)));`,
  );
  lines.push(`  session.varI32(roots.anonymousIdCount);`);
  lines.push(`  session.varI32(roots.closureNameCount);`);
  lines.push(`  session.varU32(roots.prefixTokenCount);`);
  lines.push(``);
  for (const queue of queues) {
    lines.push(
      `  session.varU32(static_cast<std::uint32_t>(roots.${queue}.size()));`,
    );
    lines.push(`  for (auto entry : roots.${queue})`);
    lines.push(
      `    session.varU32(static_cast<std::uint32_t>(symbolRef(entry)));`,
    );
    lines.push(``);
  }
  lines.push(
    `  session.varU32(static_cast<std::uint32_t>(roots.snippets.size()));`,
  );
  lines.push(`  for (const auto& [key, text] : roots.snippets) {`);
  lines.push(`    session.varU64(key);`);
  lines.push(
    `    session.varU32(static_cast<std::uint32_t>(stringRef(text)));`,
  );
  lines.push(`  }`);
  lines.push(``);
  lines.push(`  drain();`);
  lines.push(`  resolveLocations();`);
  lines.push(``);
  lines.push(`  ByteWriter strings;`);
  lines.push(`  flushStrings(strings);`);
  lines.push(`  ByteWriter sourceMap;`);
  lines.push(`  flushSourceMap(sourceMap);`);
  lines.push(`  ByteWriter names;`);
  lines.push(`  names_.flush(names);`);
  lines.push(`  ByteWriter types;`);
  lines.push(`  types_.flush(types);`);
  lines.push(`  ByteWriter symbols;`);
  lines.push(`  symbols_.flush(symbols);`);
  lines.push(`  ByteWriter nodes;`);
  lines.push(`  nodes_.flush(nodes);`);
  lines.push(`  ByteWriter constants;`);
  lines.push(`  flushConstants(constants);`);
  lines.push(``);
  lines.push(`  archive.addSection(ArchiveSection::kStrings, strings.take());`);
  lines.push(
    `  archive.addSection(ArchiveSection::kSourceMap, sourceMap.take());`,
  );
  lines.push(`  archive.addSection(ArchiveSection::kNames, names.take());`);
  lines.push(`  archive.addSection(ArchiveSection::kTypes, types.take());`);
  lines.push(`  archive.addSection(ArchiveSection::kSymbols, symbols.take());`);
  lines.push(`  archive.addSection(ArchiveSection::kAst, nodes.take());`);
  lines.push(
    `  archive.addSection(ArchiveSection::kConstants, constants.take());`,
  );
  lines.push(`  archive.addSection(ArchiveSection::kSession, session.take());`);
  lines.push(``);
  lines.push(`  return errors().empty();`);
  lines.push(`}`);
  lines.push(``);

  void plan;

  return lines.join("\n");
}

function decoderEntryPoint(plan: CodecPlan): string {
  const queues = [
    { name: "pendingBodyCompletions", cpp: "cxx::FunctionSymbol" },
    { name: "pendingMemberInstantiations", cpp: "cxx::ClassSymbol" },
    { name: "instantiatedMemberClasses", cpp: "cxx::ClassSymbol" },
  ];

  const lines: string[] = [];

  lines.push(`auto SemanticDecoder::operator()(const ArchiveReader& archive,`);
  lines.push(
    `                                 SemanticArchiveRoots& roots) -> bool {`,
  );
  lines.push(`  auto* timeTrace = unit()->timeTrace();`);
  lines.push(``);
  lines.push(`  {`);
  lines.push(`    auto strings = archive.section(ArchiveSection::kStrings);`);
  lines.push(
    `    TimeTrace::Scope trace{timeTrace, "Decode strings", sectionDetail(strings)};`,
  );
  lines.push(`    if (!readStrings(strings)) return false;`);
  lines.push(`  }`);
  lines.push(``);
  lines.push(`  {`);
  lines.push(
    `    auto sourceMap = archive.section(ArchiveSection::kSourceMap);`,
  );
  lines.push(
    `    TimeTrace::Scope trace{timeTrace, "Decode source map", sectionDetail(sourceMap)};`,
  );
  lines.push(`    if (!readSourceMap(sourceMap)) return false;`);
  lines.push(`  }`);
  lines.push(``);
  for (const [section, records] of [
    ["kNames", "nameRecords_"],
    ["kTypes", "typeRecords_"],
    ["kSymbols", "symbolRecords_"],
    ["kAst", "nodeRecords_"],
    ["kConstants", "constRecords_"],
  ]) {
    lines.push(`  {`);
    lines.push(
      `    auto section = archive.section(ArchiveSection::${section});`,
    );
    lines.push(
      `    TimeTrace::Scope trace{timeTrace, "Split ${section} records", sectionDetail(section)};`,
    );
    lines.push(`    if (!readRecords(section, ${records})) {`);
    lines.push(`      fail("${section} section is truncated");`);
    lines.push(`      return false;`);
    lines.push(`    }`);
    lines.push(`  }`);
  }
  lines.push(``);
  lines.push(`  names_.assign(nameRecords_.size(), nullptr);`);
  lines.push(`  nameDecoded_.assign(nameRecords_.size(), false);`);
  lines.push(`  types_.assign(typeRecords_.size(), nullptr);`);
  lines.push(`  typeDecoded_.assign(typeRecords_.size(), false);`);
  lines.push(`  constants_.assign(constRecords_.size(), nullptr);`);
  lines.push(``);
  lines.push(`  {`);
  lines.push(
    `    TimeTrace::Scope trace{timeTrace, "Allocate symbols", countDetail(symbolRecords_.size())};`,
  );
  lines.push(`    symbols_.reserve(symbolRecords_.size());`);
  lines.push(`    for (const auto& record : symbolRecords_) {`);
  lines.push(`      ByteReader in{record.bytes};`);
  lines.push(`      const auto kind =`);
  lines.push(
    `          static_cast<cxx::SymbolKind>(readEnum(in, ${plan.symbols.length}));`,
  );
  lines.push(`      if (!ok()) return false;`);
  lines.push(`      auto symbol = allocateSymbol(kind);`);
  lines.push(`      if (!symbol) {`);
  lines.push(
    `        fail("archive names a symbol kind this compiler cannot allocate");`,
  );
  lines.push(`        return false;`);
  lines.push(`      }`);
  lines.push(`      symbols_.push_back(symbol);`);
  lines.push(`    }`);
  lines.push(`  }`);
  lines.push(``);
  lines.push(`  {`);
  lines.push(
    `    TimeTrace::Scope trace{timeTrace, "Allocate AST nodes", countDetail(nodeRecords_.size())};`,
  );
  lines.push(`    nodes_.reserve(nodeRecords_.size());`);
  lines.push(`    for (const auto& record : nodeRecords_) {`);
  lines.push(`      ByteReader in{record.bytes};`);
  lines.push(`      const auto kind =`);
  lines.push(
    `          static_cast<cxx::ASTKind>(readEnum(in, ${plan.nodes.length}));`,
  );
  lines.push(`      if (!ok()) return false;`);
  lines.push(`      auto node = allocateAst(kind);`);
  lines.push(`      if (!node) {`);
  lines.push(
    `        fail("archive names an AST kind this compiler cannot allocate");`,
  );
  lines.push(`        return false;`);
  lines.push(`      }`);
  lines.push(`      nodes_.push_back(node);`);
  lines.push(`    }`);
  lines.push(`  }`);
  lines.push(``);
  lines.push(`  {`);
  lines.push(
    `    TimeTrace::Scope trace{timeTrace, "Decode symbols", countDetail(symbolRecords_.size())};`,
  );
  lines.push(
    `    for (std::size_t i = 0; ok() && i < symbolRecords_.size(); ++i) {`,
  );
  lines.push(`      ByteReader in{symbolRecords_[i].bytes};`);
  lines.push(`      const auto kind = readEnum(in, ${plan.symbols.length});`);
  lines.push(`      decodeSymbolFields(in, symbols_[i]);`);
  lines.push(`      if (ok() && !in.atEnd()) {`);
  lines.push(
    `        fail(std::format("symbol record {} of kind {} has {} trailing bytes",`,
  );
  lines.push(`                         i, kind, in.remaining()));`);
  lines.push(`      }`);
  lines.push(`    }`);
  lines.push(`  }`);
  lines.push(``);
  lines.push(`  {`);
  lines.push(
    `    TimeTrace::Scope trace{timeTrace, "Decode AST nodes", countDetail(nodeRecords_.size())};`,
  );
  lines.push(
    `    for (std::size_t i = 0; ok() && i < nodeRecords_.size(); ++i) {`,
  );
  lines.push(`      ByteReader in{nodeRecords_[i].bytes};`);
  lines.push(`      const auto kind = readEnum(in, ${plan.nodes.length});`);
  lines.push(`      decodeAstFields(in, nodes_[i]);`);
  lines.push(`      if (ok() && !in.atEnd()) {`);
  lines.push(
    `        fail(std::format("AST record {} of kind {} has {} trailing bytes",`,
  );
  lines.push(`                         i, kind, in.remaining()));`);
  lines.push(`      }`);
  lines.push(`    }`);
  lines.push(`  }`);
  lines.push(``);
  lines.push(`  {`);
  lines.push(`    TimeTrace::Scope trace{timeTrace, "Rebuild lookup tables"};`);
  lines.push(`    for (auto symbol : symbols_) {`);
  lines.push(`      if (auto scope = symbol_cast<ScopeSymbol>(symbol))`);
  lines.push(`        scope->rebuildLookupTable();`);
  lines.push(`    }`);
  lines.push(`  }`);
  lines.push(``);
  lines.push(`  TimeTrace::Scope sessionTrace{timeTrace, "Decode session"};`);
  lines.push(`  auto session = archive.section(ArchiveSection::kSession);`);
  lines.push(``);
  lines.push(`  roots.globalScope =`);
  lines.push(
    `      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{session.varU32()}));`,
  );
  lines.push(
    `  roots.ast = ast_cast<UnitAST>(astAt(AstRef{session.varU32()}));`,
  );
  lines.push(`  roots.anonymousIdCount = session.varI32();`);
  lines.push(`  roots.closureNameCount = session.varI32();`);
  lines.push(`  roots.prefixTokenCount = session.varU32();`);
  lines.push(``);
  for (const queue of queues) {
    lines.push(`  {`);
    lines.push(`    const auto count = session.varCount(1);`);
    lines.push(`    for (std::uint32_t i = 0; ok() && i < count; ++i) {`);
    lines.push(
      `      auto entry = symbol_cast<${queue.cpp.replace(/^cxx::/, "")}>(`,
    );
    lines.push(`          symbolAt(SymbolRef{session.varU32()}));`);
    lines.push(`      if (entry) roots.${queue.name}.push_back(entry);`);
    lines.push(`    }`);
    lines.push(`  }`);
  }
  lines.push(`  {`);
  lines.push(`    const auto count = session.varCount(2);`);
  lines.push(`    for (std::uint32_t i = 0; ok() && i < count; ++i) {`);
  lines.push(`      const auto key = session.varU64();`);
  lines.push(`      roots.snippets.emplace_back(`);
  lines.push(
    `          key, std::string{stringAt(StringRef{session.varU32()})});`,
  );
  lines.push(`    }`);
  lines.push(`  }`);
  lines.push(``);
  lines.push(`  if (!session.ok()) fail("session section is truncated");`);
  lines.push(``);
  lines.push(`  return ok();`);
  lines.push(`}`);
  lines.push(``);

  return lines.join("\n");
}

function encoderDispatch(plan: CodecPlan): string {
  const lines: string[] = [];

  lines.push(`void SemanticEncoder::writeName(ByteWriter& out,`);
  lines.push(`                                const cxx::Name* name) {`);
  lines.push(`  out.varU32(static_cast<std::uint32_t>(name->kind()));`);
  lines.push(`  switch (name->kind()) {`);
  for (const entity of plan.names) {
    lines.push(`    case ${entity.tag}:`);
    lines.push(
      `      writeName${entity.short}(out, static_cast<const ${entity.cpp}*>(name));`,
    );
    lines.push(`      break;`);
  }
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`void SemanticEncoder::writeType(ByteWriter& out,`);
  lines.push(`                                const cxx::Type* type) {`);
  lines.push(`  out.varU32(static_cast<std::uint32_t>(type->kind()));`);
  lines.push(`  switch (type->kind()) {`);
  for (const entity of plan.types) {
    lines.push(`    case ${entity.tag}:`);
    lines.push(
      `      writeType${entity.short}(out, static_cast<const ${entity.cpp}*>(type));`,
    );
    lines.push(`      break;`);
  }
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  lines.push(
    `void SemanticEncoder::writeSymbol(ByteWriter& out, cxx::Symbol* symbol) {`,
  );
  lines.push(`  out.varU32(static_cast<std::uint32_t>(symbol->kind()));`);
  lines.push(`  switch (symbol->kind()) {`);
  for (const entity of plan.symbols) {
    lines.push(`    case ${entity.tag}:`);
    lines.push(
      `      writeSymbol${entity.short}(out, static_cast<${entity.cpp}*>(symbol));`,
    );
    lines.push(`      break;`);
  }
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  lines.push(
    `void SemanticEncoder::writeAst(ByteWriter& out, cxx::AST* ast) {`,
  );
  lines.push(`  out.varU32(static_cast<std::uint32_t>(ast->kind()));`);
  lines.push(`  switch (ast->kind()) {`);
  for (const entity of plan.nodes) {
    lines.push(`    case ${entity.tag}:`);
    lines.push(
      `      writeAst${entity.short}(out, static_cast<${entity.cpp}*>(ast));`,
    );
    lines.push(`      break;`);
  }
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  lines.push(
    `void SemanticEncoder::writeConstNode(ByteWriter& out, const ConstNode& node) {`,
  );
  lines.push(`  out.u8(static_cast<std::uint8_t>(node.kind));`);
  lines.push(`  switch (node.kind) {`);
  sharedConstKinds.forEach((cpp, index) => {
    lines.push(`    case ${index}:`);
    lines.push(
      `      write${codecName(cpp)}(out, static_cast<const ${cpp}*>(node.owner.get()));`,
    );
    lines.push(`      break;`);
  });
  lines.push(`    default:`);
  lines.push(`      reportError("unknown constant node kind");`);
  lines.push(`      break;`);
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`void SemanticEncoder::drain() {`);
  lines.push(`  for (;;) {`);
  lines.push(`    bool progress = false;`);
  lines.push(``);
  lines.push(`    while (names_.hasPending()) {`);
  lines.push(`      auto entity = names_.takePending();`);
  lines.push(`      scratch_.clear();`);
  lines.push(`      writeName(scratch_, entity);`);
  lines.push(`      names_.store(entity, scratch_);`);
  lines.push(`      progress = true;`);
  lines.push(`    }`);
  lines.push(``);
  lines.push(`    while (types_.hasPending()) {`);
  lines.push(`      auto entity = types_.takePending();`);
  lines.push(`      scratch_.clear();`);
  lines.push(`      writeType(scratch_, entity);`);
  lines.push(`      types_.store(entity, scratch_);`);
  lines.push(`      progress = true;`);
  lines.push(`    }`);
  lines.push(``);
  lines.push(`    while (symbols_.hasPending()) {`);
  lines.push(`      auto entity = symbols_.takePending();`);
  lines.push(`      scratch_.clear();`);
  lines.push(`      writeSymbol(scratch_, entity);`);
  lines.push(`      symbols_.store(entity, scratch_);`);
  lines.push(`      progress = true;`);
  lines.push(`    }`);
  lines.push(``);
  lines.push(`    while (nodes_.hasPending()) {`);
  lines.push(`      auto entity = nodes_.takePending();`);
  lines.push(`      scratch_.clear();`);
  lines.push(`      writeAst(scratch_, entity);`);
  lines.push(`      nodes_.store(entity, scratch_);`);
  lines.push(`      progress = true;`);
  lines.push(`    }`);
  lines.push(``);
  lines.push(`    while (constCursor_ < constPending_.size()) {`);
  lines.push(`      const auto index = constCursor_++;`);
  lines.push(`      ByteWriter record;`);
  lines.push(`      writeConstNode(record, constPending_[index]);`);
  lines.push(`      constRecords_[index] = record.take();`);
  lines.push(`      progress = true;`);
  lines.push(`    }`);
  lines.push(``);
  lines.push(`    if (!progress) break;`);
  lines.push(`  }`);
  lines.push(`}`);

  return lines.join("\n");
}

function encoderHelpers(): string {
  const lines: string[] = [];
  const names = new Names();

  lines.push(`void SemanticEncoder::writeLiteral(ByteWriter& out,`);
  lines.push(
    `                                   const cxx::Literal* literal) {`,
  );
  lines.push(`  out.boolean(literal != nullptr);`);
  lines.push(`  if (literal)`);
  lines.push(
    `    out.varU32(static_cast<std::uint32_t>(stringRef(literal->value())));`,
  );
  lines.push(`}`);
  lines.push(``);

  lines.push(`void SemanticEncoder::writeAbiTags(`);
  lines.push(
    `    ByteWriter& out, const std::vector<const cxx::Identifier*>* tags) {`,
  );
  lines.push(`  out.boolean(tags != nullptr);`);
  lines.push(`  if (!tags) return;`);
  lines.push(`  out.varU32(static_cast<std::uint32_t>(tags->size()));`);
  lines.push(`  for (auto tag : *tags)`);
  lines.push(`    out.varU32(static_cast<std::uint32_t>(identifierRef(tag)));`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`void SemanticEncoder::writeAttributes(`);
  lines.push(`    ByteWriter& out, const cxx::AttributeMap* attributes) {`);
  lines.push(`  out.boolean(attributes != nullptr);`);
  lines.push(`  if (!attributes) return;`);
  lines.push(`  out.varU32(static_cast<std::uint32_t>(attributes->size()));`);
  lines.push(`  for (const auto& attribute : *attributes)`);
  lines.push(`    writecxxAttribute(out, &attribute);`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`void SemanticEncoder::writeConstValue(ByteWriter& out,`);
  lines.push(
    `                                      const cxx::ConstValue& value) {`,
  );
  lines.push(`  out.u8(static_cast<std::uint8_t>(value.index()));`);
  lines.push(`  switch (value.index()) {`);
  for (const alternative of constAlternatives) {
    lines.push(`    case ${alternative.tag}: {`);
    if (alternative.kind === "scalar") {
      lines.push(
        `      out.${(alternative as { writer: string }).writer}(std::get<${alternative.tag}>(value));`,
      );
    } else if (alternative.kind === "literal") {
      lines.push(
        `      writeLiteral(out, std::get<${alternative.tag}>(value));`,
      );
    } else if (alternative.kind === "shared") {
      lines.push(
        `      out.varU32(static_cast<std::uint32_t>(constRef(std::get<${alternative.tag}>(value))));`,
      );
    }
    lines.push(`      break;`);
    lines.push(`    }`);
  }
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`void SemanticEncoder::writeTemplateArgument(`);
  lines.push(`    ByteWriter& out, const cxx::TemplateArgument& argument) {`);
  lines.push(`  out.u8(static_cast<std::uint8_t>(argument.index()));`);
  lines.push(`  switch (argument.index()) {`);
  lines.push(`    case 0:`);
  lines.push(
    `      out.varU32(static_cast<std::uint32_t>(typeRef(std::get<0>(argument))));`,
  );
  lines.push(`      break;`);
  lines.push(`    case 1:`);
  lines.push(
    `      out.varU32(static_cast<std::uint32_t>(symbolRef(std::get<1>(argument))));`,
  );
  lines.push(`      break;`);
  lines.push(`    case 2:`);
  lines.push(`      writeConstValue(out, std::get<2>(argument));`);
  lines.push(`      break;`);
  lines.push(`    case 3:`);
  lines.push(
    `      out.varU32(static_cast<std::uint32_t>(astRef(std::get<3>(argument))));`,
  );
  lines.push(`      break;`);
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  void names;

  return lines.join("\n");
}

function fieldWrites(fields: FieldPlan[], self: string): string[] {
  const lines: string[] = [];
  const names = new Names();
  for (const field of fields) {
    if (!field.wire || !field.read) continue;
    lines.push(`  // ${field.owner}::${field.name}`);
    emitWrite(lines, "  ", field.wire, substitute(field.read, self, ""), names);
  }
  return lines;
}

function encodeFactoryEntity(entity: EntityPlan, domain: string): string {
  const lines: string[] = [];
  const names = new Names();

  lines.push(`void SemanticEncoder::write${domain}${entity.short}(`);
  lines.push(
    `    ByteWriter& out, [[maybe_unused]] const ${entity.cpp}* self) {`,
  );

  for (const parameter of entity.factory?.parameters ?? []) {
    if (parameter.read === "{}" || parameter.read === "nullptr") continue;
    lines.push(`  // ${parameter.name}`);
    emitWrite(
      lines,
      "  ",
      parameter.wire,
      substitute(parameter.read, "self", ""),
      names,
    );
  }

  lines.push(`}`);
  lines.push(``);

  return lines.join("\n");
}

function encodeFieldEntity(entity: EntityPlan, domain: string): string {
  const lines: string[] = [];

  lines.push(`void SemanticEncoder::write${domain}${entity.short}(`);
  lines.push(`    ByteWriter& out, [[maybe_unused]] ${entity.cpp}* self) {`);
  lines.push(...fieldWrites(entity.fields, "self"));
  lines.push(`}`);
  lines.push(``);

  return lines.join("\n");
}

function encodeStructEntity(entity: EntityPlan): string {
  const lines: string[] = [];

  lines.push(`void SemanticEncoder::write${codecName(entity.name)}(`);
  lines.push(
    `    ByteWriter& out, [[maybe_unused]] const ${entity.cpp}* self) {`,
  );
  lines.push(...fieldWrites(entity.fields, "self"));
  lines.push(`}`);
  lines.push(``);

  return lines.join("\n");
}

function decoderDispatch(plan: CodecPlan): string {
  const lines: string[] = [];

  lines.push(
    `auto SemanticDecoder::allocateSymbol(cxx::SymbolKind kind) -> cxx::Symbol* {`,
  );
  lines.push(`  switch (kind) {`);
  for (const entity of plan.symbols) {
    const call = entity.factory?.call;
    if (!call) {
      lines.push(`    case ${entity.tag}:`);
      lines.push(`      return nullptr;`);
      continue;
    }
    const args = (entity.factory?.parameters ?? []).map((parameter) =>
      parameter.name === "enclosingScope" ? "nullptr" : "{}",
    );
    lines.push(`    case ${entity.tag}:`);
    lines.push(`      return control()->${call}(${args.join(", ")});`);
  }
  lines.push(`  }`);
  lines.push(`  return nullptr;`);
  lines.push(`}`);
  lines.push(``);

  lines.push(
    `auto SemanticDecoder::allocateAst(cxx::ASTKind kind) -> cxx::AST* {`,
  );
  lines.push(`  switch (kind) {`);
  for (const entity of plan.nodes) {
    lines.push(`    case ${entity.tag}:`);
    lines.push(`      return ${entity.cpp}::create(arena());`);
  }
  lines.push(`  }`);
  lines.push(`  return nullptr;`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`auto SemanticDecoder::nameAt(NameRef ref) -> const cxx::Name* {`);
  lines.push(`  const auto index = static_cast<std::uint32_t>(ref);`);
  lines.push(`  if (index == 0) return nullptr;`);
  lines.push(`  if (index > nameRecords_.size()) {`);
  lines.push(`    fail("name reference is out of range");`);
  lines.push(`    return nullptr;`);
  lines.push(`  }`);
  lines.push(`  if (nameDecoded_[index - 1]) return names_[index - 1];`);
  lines.push(`  nameDecoded_[index - 1] = true;`);
  lines.push(`  ByteReader in{nameRecords_[index - 1].bytes};`);
  lines.push(
    `  const auto kind = static_cast<cxx::NameKind>(readEnum(in, ${plan.names.length}));`,
  );
  lines.push(`  const cxx::Name* name = nullptr;`);
  lines.push(`  switch (kind) {`);
  for (const entity of plan.names) {
    lines.push(`    case ${entity.tag}:`);
    lines.push(`      name = readName${entity.short}(in);`);
    lines.push(`      break;`);
  }
  lines.push(`  }`);
  lines.push(`  names_[index - 1] = name;`);
  lines.push(`  return name;`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`auto SemanticDecoder::typeAt(TypeRef ref) -> const cxx::Type* {`);
  lines.push(`  const auto index = static_cast<std::uint32_t>(ref);`);
  lines.push(`  if (index == 0) return nullptr;`);
  lines.push(`  if (index > typeRecords_.size()) {`);
  lines.push(`    fail("type reference is out of range");`);
  lines.push(`    return nullptr;`);
  lines.push(`  }`);
  lines.push(`  if (typeDecoded_[index - 1]) return types_[index - 1];`);
  lines.push(`  typeDecoded_[index - 1] = true;`);
  lines.push(`  ByteReader in{typeRecords_[index - 1].bytes};`);
  lines.push(
    `  const auto kind = static_cast<cxx::TypeKind>(readEnum(in, ${plan.types.length}));`,
  );
  lines.push(`  const cxx::Type* type = nullptr;`);
  lines.push(`  switch (kind) {`);
  for (const entity of plan.types) {
    lines.push(`    case ${entity.tag}:`);
    lines.push(`      type = readType${entity.short}(in);`);
    lines.push(`      break;`);
  }
  lines.push(`  }`);
  lines.push(`  types_[index - 1] = type;`);
  lines.push(`  return type;`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`auto SemanticDecoder::symbolAt(SymbolRef ref) -> cxx::Symbol* {`);
  lines.push(`  const auto index = static_cast<std::uint32_t>(ref);`);
  lines.push(`  if (index == 0) return nullptr;`);
  lines.push(`  if (index > symbols_.size()) {`);
  lines.push(`    fail("symbol reference is out of range");`);
  lines.push(`    return nullptr;`);
  lines.push(`  }`);
  lines.push(`  return symbols_[index - 1];`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`auto SemanticDecoder::astAt(AstRef ref) -> cxx::AST* {`);
  lines.push(`  const auto index = static_cast<std::uint32_t>(ref);`);
  lines.push(`  if (index == 0) return nullptr;`);
  lines.push(`  if (index > nodes_.size()) {`);
  lines.push(`    fail("AST reference is out of range");`);
  lines.push(`    return nullptr;`);
  lines.push(`  }`);
  lines.push(`  return nodes_[index - 1];`);
  lines.push(`}`);
  lines.push(``);

  lines.push(
    `void SemanticDecoder::decodeSymbolFields(ByteReader& in, cxx::Symbol* symbol) {`,
  );
  lines.push(`  switch (symbol->kind()) {`);
  for (const entity of plan.symbols) {
    lines.push(`    case ${entity.tag}:`);
    lines.push(
      `      readSymbol${entity.short}(in, static_cast<${entity.cpp}*>(symbol));`,
    );
    lines.push(`      break;`);
  }
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  lines.push(
    `void SemanticDecoder::decodeAstFields(ByteReader& in, cxx::AST* ast) {`,
  );
  lines.push(`  switch (ast->kind()) {`);
  for (const entity of plan.nodes) {
    lines.push(`    case ${entity.tag}:`);
    lines.push(
      `      readAst${entity.short}(in, static_cast<${entity.cpp}*>(ast));`,
    );
    lines.push(`      break;`);
  }
  lines.push(`  }`);
  lines.push(`}`);

  return lines.join("\n");
}

function decoderHelpers(plan: CodecPlan): string {
  const lines: string[] = [];

  lines.push(
    `auto SemanticDecoder::readEnum(ByteReader& in, std::uint32_t count)`,
  );
  lines.push(`    -> std::uint32_t {`);
  lines.push(`  const auto value = in.varU32();`);
  lines.push(`  if (value >= count) {`);
  lines.push(`    fail("enumerator is out of range");`);
  lines.push(`    return 0;`);
  lines.push(`  }`);
  lines.push(`  return value;`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`auto SemanticDecoder::readAbiTags(ByteReader& in)`);
  lines.push(`    -> const std::vector<const cxx::Identifier*>* {`);
  lines.push(`  if (!in.boolean()) return nullptr;`);
  lines.push(`  const auto count = in.varCount(1);`);
  lines.push(`  std::vector<const cxx::Identifier*> tags;`);
  lines.push(`  tags.reserve(count);`);
  lines.push(`  for (std::uint32_t i = 0; ok() && i < count; ++i)`);
  lines.push(`    tags.push_back(identifierAt(StringRef{in.varU32()}));`);
  lines.push(`  return control()->getAbiTags(std::move(tags));`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`auto SemanticDecoder::readAttributes(ByteReader& in)`);
  lines.push(`    -> const cxx::AttributeMap* {`);
  lines.push(`  if (!in.boolean()) return nullptr;`);
  lines.push(`  const auto count = in.varCount(1);`);
  lines.push(`  cxx::AttributeMap attributes;`);
  lines.push(`  attributes.reserve(count);`);
  lines.push(`  for (std::uint32_t i = 0; ok() && i < count; ++i) {`);
  lines.push(`    cxx::Attribute attribute;`);
  lines.push(`    readcxxAttribute(in, &attribute);`);
  lines.push(`    attributes.push_back(std::move(attribute));`);
  lines.push(`  }`);
  lines.push(`  return control()->getAttributes(std::move(attributes));`);
  lines.push(`}`);
  lines.push(``);

  lines.push(
    `auto SemanticDecoder::readConstValue(ByteReader& in) -> cxx::ConstValue {`,
  );
  lines.push(`  const auto tag = in.u8();`);
  lines.push(`  switch (tag) {`);
  for (const alternative of constAlternatives) {
    lines.push(`    case ${alternative.tag}:`);
    if (alternative.kind === "scalar") {
      lines.push(
        `      return cxx::ConstValue{static_cast<${alternative.cpp}>(in.${(alternative as { reader: string }).reader}())};`,
      );
    } else if (alternative.kind === "literal") {
      lines.push(`      return cxx::ConstValue{readStringLiteral(in)};`);
    } else if (alternative.kind === "shared") {
      const kindIndex = sharedConstKinds.indexOf(alternative.cpp);
      lines.push(
        `      return cxx::ConstValue{std::static_pointer_cast<${alternative.cpp}>(`,
      );
      lines.push(`          constantAt(ConstRef{in.varU32()}))};`);
      void kindIndex;
    } else {
      lines.push(`      return cxx::ConstValue{cxx::IndeterminateValue{}};`);
    }
  }
  lines.push(`    default:`);
  lines.push(`      fail("unknown constant value alternative");`);
  lines.push(`      return cxx::ConstValue{};`);
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  lines.push(`auto SemanticDecoder::readTemplateArgument(ByteReader& in)`);
  lines.push(`    -> cxx::TemplateArgument {`);
  lines.push(`  const auto tag = in.u8();`);
  lines.push(`  switch (tag) {`);
  lines.push(`    case 0:`);
  lines.push(
    `      return cxx::TemplateArgument{typeAt(TypeRef{in.varU32()})};`,
  );
  lines.push(`    case 1:`);
  lines.push(
    `      return cxx::TemplateArgument{symbolAt(SymbolRef{in.varU32()})};`,
  );
  lines.push(`    case 2:`);
  lines.push(`      return cxx::TemplateArgument{readConstValue(in)};`);
  lines.push(`    case 3:`);
  lines.push(
    `      return cxx::TemplateArgument{ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}))};`,
  );
  lines.push(`    default:`);
  lines.push(`      fail("unknown template argument alternative");`);
  lines.push(`      return cxx::TemplateArgument{};`);
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  lines.push(
    `auto SemanticDecoder::constantAt(ConstRef ref) -> std::shared_ptr<void> {`,
  );
  lines.push(`  const auto index = static_cast<std::uint32_t>(ref);`);
  lines.push(`  if (index == 0) return {};`);
  lines.push(`  if (index > constRecords_.size()) {`);
  lines.push(`    fail("constant reference is out of range");`);
  lines.push(`    return {};`);
  lines.push(`  }`);
  lines.push(`  if (constants_[index - 1]) return constants_[index - 1];`);
  lines.push(`  ByteReader in{constRecords_[index - 1].bytes};`);
  lines.push(`  const auto kind = in.u8();`);
  lines.push(`  switch (kind) {`);
  sharedConstKinds.forEach((cpp, position) => {
    lines.push(`    case ${position}: {`);
    lines.push(`      auto value = std::make_shared<${cpp}>();`);
    lines.push(`      constants_[index - 1] = value;`);
    lines.push(`      read${codecName(cpp)}(in, value.get());`);
    lines.push(`      return value;`);
    lines.push(`    }`);
  });
  lines.push(`    default:`);
  lines.push(`      fail("unknown constant node kind");`);
  lines.push(`      return {};`);
  lines.push(`  }`);
  lines.push(`}`);
  lines.push(``);

  void plan;

  return lines.join("\n");
}

function decodeFactoryEntity(entity: EntityPlan, domain: string): string {
  const lines: string[] = [];
  const names = new Names();
  const arguments_: string[] = [];

  lines.push(
    `auto SemanticDecoder::read${domain}${entity.short}(ByteReader& in)`,
  );
  lines.push(`    -> const cxx::${domain}* {`);

  for (const parameter of entity.factory?.parameters ?? []) {
    if (parameter.read === "{}" || parameter.read === "nullptr") {
      arguments_.push(parameter.read);
      continue;
    }
    const variable = names.fresh("argument");
    emitRead(lines, "  ", parameter.wire, parameter.cppType, variable, names);
    arguments_.push(`std::move(${variable})`);
  }

  const call = entity.factory?.call;
  if (call) {
    lines.push(`  return control()->${call}(${arguments_.join(", ")});`);
  } else {
    lines.push(`  fail("no factory for ${entity.name}");`);
    lines.push(`  return nullptr;`);
  }

  lines.push(`}`);
  lines.push(``);

  return lines.join("\n");
}

function fieldReads(fields: FieldPlan[], self: string): string[] {
  const lines: string[] = [];
  const names = new Names();

  for (const field of fields) {
    if (!field.wire) continue;
    lines.push(`  // ${field.owner}::${field.name}`);

    const variable = names.fresh("value");
    const destination =
      field.typeExpression && derivesLocalType(field.wire)
        ? substitute(field.typeExpression, self, "")
        : undefined;
    const declaredType = destination
      ? destination.endsWith("()")
        ? `std::remove_cvref_t<decltype(${destination})>`
        : `decltype(${destination})`
      : field.cppType;
    emitRead(lines, "  ", field.wire, declaredType, variable, names);

    if (field.writeElement) {
      const element = names.fresh("element");
      lines.push(`  for (auto&& ${element} : ${variable}) {`);
      lines.push(`    ${substitute(field.writeElement, self, element)};`);
      lines.push(`  }`);
    } else if (field.write) {
      lines.push(
        `  ${substitute(field.write, self, `std::move(${variable})`)};`,
      );
    }
  }

  return lines;
}

function decodeFieldEntity(entity: EntityPlan, domain: string): string {
  const lines: string[] = [];

  lines.push(`void SemanticDecoder::read${domain}${entity.short}(`);
  lines.push(
    `    [[maybe_unused]] ByteReader& in, [[maybe_unused]] ${entity.cpp}* self) {`,
  );
  lines.push(...fieldReads(entity.fields, "self"));
  lines.push(`}`);
  lines.push(``);

  return lines.join("\n");
}

function decodeStructEntity(entity: EntityPlan): string {
  const lines: string[] = [];

  lines.push(`void SemanticDecoder::read${codecName(entity.name)}(`);
  lines.push(
    `    [[maybe_unused]] ByteReader& in, [[maybe_unused]] ${entity.cpp}* self) {`,
  );
  lines.push(...fieldReads(entity.fields, "self"));
  lines.push(`}`);
  lines.push(``);

  return lines.join("\n");
}

export { allEntities };
