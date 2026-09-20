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

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/diagnostics_client.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/pch.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace cxx;

namespace {

auto keys() -> PrecompiledHeaderKeys {
  return {precompiledHeaderSerializationAbi(), "wasm32", "c++/c++23", ""};
}

class Prefix {
 public:
  explicit Prefix(std::string source) {
    unit_ = std::make_unique<TranslationUnit>(&diagnostics_);
    unit_->control()->setMemoryLayout(&memoryLayout_);
    // The builtin declarations are written in terms of the target macros a
    // toolchain would supply; these tests run without one.
    for (const auto& [name, body] : {
             std::pair{"__SIZE_TYPE__", "unsigned long"},
             std::pair{"__PTRDIFF_TYPE__", "long"},
             std::pair{"__INT8_TYPE__", "signed char"},
             std::pair{"__INT16_TYPE__", "short"},
             std::pair{"__INT32_TYPE__", "int"},
             std::pair{"__INT64_TYPE__", "long long"},
             std::pair{"__UINT8_TYPE__", "unsigned char"},
             std::pair{"__UINT16_TYPE__", "unsigned short"},
             std::pair{"__UINT32_TYPE__", "unsigned int"},
             std::pair{"__UINT64_TYPE__", "unsigned long long"},
             std::pair{"__INTPTR_TYPE__", "long"},
             std::pair{"__UINTPTR_TYPE__", "unsigned long"},
             std::pair{"__WCHAR_TYPE__", "int"},
             std::pair{"__CHAR16_TYPE__", "unsigned short"},
             std::pair{"__CHAR32_TYPE__", "unsigned int"},
         }) {
      unit_->preprocessor()->defineMacro(name, body);
    }
    unit_->setSource(std::move(source), "prefix.h");
    unit_->parse({.checkTypes = true});
  }

  [[nodiscard]] auto unit() -> TranslationUnit* { return unit_.get(); }

  [[nodiscard]] auto emit() -> std::vector<std::uint8_t> {
    PrecompiledHeaderWriter writer{unit_.get(), keys()};
    writer.setPreprocessorState(unit_->preprocessor()->snapshot());
    auto data = writer();
    errors_ = writer.errors();
    return data;
  }

  [[nodiscard]] auto errors() const -> const std::vector<std::string>& {
    return errors_;
  }

 private:
  DiagnosticsClient diagnostics_;
  MemoryLayout memoryLayout_{32};
  std::unique_ptr<TranslationUnit> unit_;
  std::vector<std::string> errors_;
};

[[nodiscard]] auto findMember(ScopeSymbol* scope, std::string_view name)
    -> Symbol* {
  for (auto member : scope->members()) {
    auto id = name_cast<Identifier>(member->name());
    if (id && id->name() == name) return member;
  }
  return nullptr;
}

}  // namespace

TEST(PrecompiledHeader, RestoresTheGlobalScope) {
  Prefix prefix{R"(
struct Point {
  int x;
  int y;
};

int origin;
)"};

  const auto data = prefix.emit();

  ASSERT_TRUE(prefix.errors().empty());
  ASSERT_FALSE(data.empty());

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{&consumer, keys()};

  ASSERT_TRUE(reader(data)) << reader.error();
  ASSERT_TRUE(consumer.hasAdoptedPrefix());

  auto globalScope = consumer.globalScope();
  ASSERT_TRUE(globalScope);

  auto point = symbol_cast<ClassSymbol>(findMember(globalScope, "Point"));
  ASSERT_TRUE(point);
  ASSERT_TRUE(point->isComplete());

  auto x = symbol_cast<FieldSymbol>(findMember(point, "x"));
  ASSERT_TRUE(x);
  ASSERT_TRUE(type_cast<IntType>(x->type()));

  ASSERT_TRUE(symbol_cast<VariableSymbol>(findMember(globalScope, "origin")));
}

TEST(PrecompiledHeader, RestoresScopeLookup) {
  Prefix prefix{"struct Point { int x; };\nint origin;\n"};

  const auto data = prefix.emit();

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{&consumer, keys()};
  ASSERT_TRUE(reader(data)) << reader.error();

  auto globalScope = consumer.globalScope();

  // Lookup, not just member order: `buckets_` and `Symbol::link_` are rebuilt
  // on load and a name resolves by interned-pointer identity.
  ASSERT_NE(globalScope->find("Point").begin(),
            globalScope->find("Point").end());
  ASSERT_NE(globalScope->find("origin").begin(),
            globalScope->find("origin").end());

  auto point = symbol_cast<ClassSymbol>(findMember(globalScope, "Point"));
  ASSERT_TRUE(point);
  ASSERT_NE(point->find("x").begin(), point->find("x").end());
}

TEST(PrecompiledHeader, RestoresTheMacroEnvironment) {
  Prefix prefix{"#define PREFIX_VALUE 42\nint anchor;\n"};

  const auto data = prefix.emit();
  ASSERT_TRUE(prefix.errors().empty());

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{&consumer, keys()};
  ASSERT_TRUE(reader(data)) << reader.error();

  std::vector<Token> tokens;
  consumer.preprocessor()->preprocess("PREFIX_VALUE", "macro.cc", tokens);

  bool expandedToFortyTwo = false;
  for (const auto& token : tokens) {
    if (token.kind() != TokenKind::T_INTEGER_LITERAL) continue;
    if (token.value().literalValue->value() == "42") expandedToFortyTwo = true;
  }

  ASSERT_TRUE(expandedToFortyTwo);
}

TEST(PrecompiledHeader, RebasesTheConsumerTokenSegment) {
  Prefix prefix{"int anchor;\n"};

  const auto prefixTokenCount = prefix.unit()->tokenCount();
  const auto data = prefix.emit();

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{&consumer, keys()};
  ASSERT_TRUE(reader(data)) << reader.error();

  ASSERT_EQ(consumer.tokenSegmentBase(), prefixTokenCount);
}

TEST(PrecompiledHeader, RendersAPrefixLocationWithoutPrefixTokens) {
  Prefix prefix{"int anchor;\n"};

  const auto data = prefix.emit();

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{&consumer, keys()};
  ASSERT_TRUE(reader(data)) << reader.error();

  auto anchor = findMember(consumer.globalScope(), "anchor");
  ASSERT_TRUE(anchor);

  const auto location = anchor->location();
  ASSERT_FALSE(consumer.ownsLocation(location));

  const auto position = consumer.tokenStartPosition(location);
  ASSERT_EQ(position.fileName, "prefix.h");
  ASSERT_EQ(position.line, 1);
}

TEST(PrecompiledHeader, RendersALocationDerivedFromAPrefixLocation) {
  Prefix prefix{"int anchor;\n"};

  const auto data = prefix.emit();

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{&consumer, keys()};
  ASSERT_TRUE(reader(data)) << reader.error();

  auto anchor = findMember(consumer.globalScope(), "anchor");
  ASSERT_TRUE(anchor);

  const auto pastTheEnd = anchor->location().next();
  ASSERT_FALSE(consumer.ownsLocation(pastTheEnd));

  const auto end = consumer.tokenEndPosition(anchor->location());
  const auto position = consumer.tokenStartPosition(pastTheEnd);
  ASSERT_EQ(position.fileName, "prefix.h");
  ASSERT_EQ(position.line, end.line);
  ASSERT_EQ(position.column, end.column);
}

TEST(PrecompiledHeader, IsDeterministic) {
  Prefix first{"struct S { int a; int b; };\ntemplate <typename T> T id(T);\n"};
  Prefix second{
      "struct S { int a; int b; };\ntemplate <typename T> T id(T);\n"};

  ASSERT_EQ(first.emit(), second.emit());
}

TEST(PrecompiledHeader, RejectsAForeignTarget) {
  Prefix prefix{"int anchor;\n"};
  const auto data = prefix.emit();

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{
      &consumer,
      {precompiledHeaderSerializationAbi(), "x86_64", "c++/c++23", ""}};

  ASSERT_FALSE(reader(data));
  ASSERT_FALSE(consumer.hasAdoptedPrefix());
}

TEST(PrecompiledHeader, RejectsAForeignOptionDigest) {
  Prefix prefix{"int anchor;\n"};
  const auto data = prefix.emit();

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{
      &consumer,
      {precompiledHeaderSerializationAbi(), "wasm32", "c++/c++23", "check=0"}};

  ASSERT_FALSE(reader(data));
  ASSERT_FALSE(consumer.hasAdoptedPrefix());
}

TEST(PrecompiledHeader, RejectsForeignData) {
  std::vector<std::uint8_t> data{1, 2, 3, 4};

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{&consumer, keys()};

  ASSERT_FALSE(reader(data));
  ASSERT_FALSE(reader.error().empty());
  ASSERT_FALSE(consumer.hasAdoptedPrefix());
}

TEST(PrecompiledHeader, RejectsATruncatedArchive) {
  Prefix prefix{"int anchor;\n"};
  auto data = prefix.emit();

  ASSERT_GT(data.size(), 32u);
  data.resize(data.size() / 2);

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");

  PrecompiledHeaderReader reader{&consumer, keys()};

  ASSERT_FALSE(reader(data));
  ASSERT_FALSE(consumer.hasAdoptedPrefix());
}

TEST(PrecompiledHeader, DiscoversIncludePreambleDuringPreprocessing) {
  for (const auto& source : {
           std::string("// heading\n # include /* header */ <header.h>\n\nint "
                       "value;\n"),
           std::string("#include \\\n<header.h>\nint value;\n"),
           std::string("#include <header.h>\nTOKEN value;\n"),
           std::string("#include <header.h>\n/* trailing */ int value;\n"),
           std::string("#include <header.h> // trailing\n\n\nint value;\n"),
           std::string(
               "#include <header.h>\n#include <header.h>\nint value;\n"),
       }) {
    Control control;
    DiagnosticsClient diagnostics;
    Preprocessor pp(&control, &diagnostics);
    pp.setPreambleOnly(true);
    std::vector<Token> tokens;
    pp.beginPreprocessing(source, "main.cc", tokens);
    while (true) {
      auto state = pp.continuePreprocessing(tokens);
      if (std::holds_alternative<ProcessingComplete>(state)) break;
      if (auto include = std::get_if<PendingInclude>(&state)) {
        include->resolveWith("header.h", false);
      } else if (auto content = std::get_if<PendingFileContent>(&state)) {
        content->setContent("#define TOKEN int\nstruct Header {};\n");
      }
    }
    pp.endPreprocessing(tokens);
    const std::string_view header = "<header.h>";
    auto expected = source.rfind(header) + header.size();
    ASSERT_EQ(pp.preambleSize(), expected);
    ASSERT_TRUE(pp.canSnapshot());
    for (const auto& token : tokens) {
      if (token.fileId() != pp.mainSourceFileId()) continue;
      EXPECT_EQ(token.kind(), TokenKind::T_EOF_SYMBOL);
      EXPECT_EQ(token.offset(), expected);
    }
  }
}

TEST(PrecompiledHeader, RejectsOtherDirectivesBeforeFirstMainToken) {
  for (const auto& source : {
           "#define X 1\n#include <header.h>\nint x;",
           "#include <header.h>\n#undef X\nint x;",
           "#ifndef GUARD\n#define GUARD\n#include <header.h>\n#endif\n",
           "#include <header.h>\n#pragma once\nint x;",
           "#include <header.h>\n#\nint x;",
           "int x;\n#include <header.h>\n",
       }) {
    Control control;
    DiagnosticsClient diagnostics;
    Preprocessor pp(&control, &diagnostics);
    pp.setPreambleOnly(true);
    std::vector<Token> tokens;
    pp.beginPreprocessing(source, "main.cc", tokens);
    while (true) {
      auto state = pp.continuePreprocessing(tokens);
      if (std::holds_alternative<ProcessingComplete>(state)) break;
      if (auto include = std::get_if<PendingInclude>(&state)) {
        include->resolveWith("header.h", false);
      } else if (auto content = std::get_if<PendingFileContent>(&state)) {
        content->setContent("struct Header {};\n");
      }
    }
    pp.endPreprocessing(tokens);
    EXPECT_FALSE(pp.preambleSize());
  }
}

TEST(PrecompiledHeader, RejectsPreamblesThatDoNotEndAtAnEmittedToken) {
  for (const auto& source : {
           "#include <header.h>\n",
           "#include <header.h>",
           "#include <header.h>\n/* unterminated\n",
           "#include <",
           "#include \\\n<header.h>",
       }) {
    Control control;
    DiagnosticsClient diagnostics;
    Preprocessor pp(&control, &diagnostics);
    pp.setPreambleOnly(true);
    std::vector<Token> tokens;
    pp.beginPreprocessing(source, "main.cc", tokens);
    while (true) {
      auto state = pp.continuePreprocessing(tokens);
      if (std::holds_alternative<ProcessingComplete>(state)) break;
      if (auto include = std::get_if<PendingInclude>(&state)) {
        include->resolveWith("header.h", false);
      } else if (auto content = std::get_if<PendingFileContent>(&state)) {
        content->setContent("struct Header {};\n");
      }
    }
    pp.endPreprocessing(tokens);
    EXPECT_FALSE(pp.preambleSize());
  }
}

TEST(PrecompiledHeader, ReportsIncludesThatCannotBeRead) {
  Control control;
  std::vector<std::string> messages;
  struct Collector final : DiagnosticsClient {
    std::vector<std::string>& messages;
    explicit Collector(std::vector<std::string>& messages)
        : messages(messages) {}
    void report(const Diagnostic& diag) override {
      messages.push_back(diag.message());
    }
  } collector{messages};

  Preprocessor pp(&control, &collector);
  pp.setPreambleOnly(true);
  std::vector<Token> tokens;
  std::string source = "#include <header.h>\nint x;\n";
  pp.beginPreprocessing(source, "main.cc", tokens);
  while (true) {
    auto state = pp.continuePreprocessing(tokens);
    if (std::holds_alternative<ProcessingComplete>(state)) break;
    if (auto include = std::get_if<PendingInclude>(&state)) {
      include->resolveWith("header.h", false);
    } else if (auto content = std::get_if<PendingFileContent>(&state)) {
      content->setContent(std::nullopt);
    }
  }
  pp.endPreprocessing(tokens);

  ASSERT_EQ(messages.size(), 1);
  EXPECT_EQ(messages[0], "cannot read file 'header.h'");
}

TEST(PrecompiledHeader, RestoresDependentExceptionSpecifications) {
  Prefix prefix{R"(
template<bool B> using Function = int() noexcept(B);
using Nothrow = Function<true>;
using Throwing = Function<false>;
)"};
  const auto data = prefix.emit();
  ASSERT_TRUE(prefix.errors().empty());
  ASSERT_FALSE(data.empty());

  DiagnosticsClient diagnostics;
  TranslationUnit consumer{&diagnostics};
  consumer.setSource("", "consumer.cc");
  PrecompiledHeaderReader reader{&consumer, keys()};
  ASSERT_TRUE(reader(data)) << reader.error();

  auto alias = symbol_cast<TypeAliasSymbol>(
      findMember(consumer.globalScope(), "Function"));
  ASSERT_TRUE(alias);
  auto function = type_cast<FunctionType>(alias->type());
  ASSERT_TRUE(function);
  auto expression = ast_cast<IdExpressionAST>(function->noexceptExpression());
  ASSERT_TRUE(expression);
  auto parameter = symbol_cast<NonTypeParameterSymbol>(expression->symbol);
  ASSERT_TRUE(parameter);
  EXPECT_EQ(parameter->parent(), alias->templateParameters());

  auto nothrow = findMember(consumer.globalScope(), "Nothrow");
  auto throwing = findMember(consumer.globalScope(), "Throwing");
  ASSERT_TRUE(nothrow);
  ASSERT_TRUE(throwing);
  auto nothrowType = type_cast<FunctionType>(nothrow->type());
  auto throwingType = type_cast<FunctionType>(throwing->type());
  ASSERT_TRUE(nothrowType);
  ASSERT_TRUE(throwingType);
  EXPECT_EQ(nothrowType->noexceptExpression(), nullptr);
  EXPECT_EQ(throwingType->noexceptExpression(), nullptr);
  EXPECT_TRUE(nothrowType->isNoexcept());
  EXPECT_FALSE(throwingType->isNoexcept());
}
