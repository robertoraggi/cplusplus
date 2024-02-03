// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/name_printer.h>
#include <cxx/names.h>
#include <cxx/private/format.h>
#include <cxx/scope.h>
#include <cxx/symbol_instantiation.h>
#include <cxx/symbol_printer.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_printer.h>
#include <cxx/types.h>
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

namespace cxx {

auto dump_symbol(Symbol* symbol) -> std::string {
  std::ostringstream out;
  out << symbol;
  return out.str();
}

struct Source {
  Control control;
  DiagnosticsClient diagnosticsClient;
  TranslationUnit unit{&control, &diagnosticsClient};

  explicit Source(std::string_view source) {
    unit.setSource(std::string(source), "<test>");
    unit.parse({
        .checkTypes = true,
        .fuzzyTemplateResolution = false,
        .staticAssert = true,
        .reflect = true,
    });
  }

  auto ast() -> UnitAST* { return unit.ast(); }
  auto scope() -> Scope* { return unit.globalScope(); }

  auto get(std::string_view name) -> Symbol* {
    Symbol* symbol = nullptr;
    auto id = control.getIdentifier(name);
    for (auto candidate : scope()->get(id)) {
      if (symbol) return nullptr;
      symbol = candidate;
    }
    return symbol;
  }

  auto instantiate(std::string_view name,
                   const std::vector<TemplateArgument>& arguments) -> Symbol* {
    auto symbol = get(name);
    return control.instantiate(&unit, symbol, arguments);
  }
};

auto operator""_cxx(const char* source, std::size_t size) -> Source {
  return Source{std::string_view{source, size}};
}

struct LookupMember {
  Source& source;

  auto operator()(Scope* scope, std::string_view name) -> Symbol* {
    auto id = source.control.getIdentifier(name);
    for (auto candidate : scope->get(id)) {
      return candidate;
    }
    return nullptr;
  }
};

}  // namespace cxx

using namespace cxx;

TEST(Substitution, TypeAlias) {
  auto source = R"(
    template <typename T>
    using Ptr = T*;

    template <typename T>
    using RefF = auto (T&, T&&) -> void;

    template <typename T, typename U>
    using F = T (*)(T const, U*, ...);
  )"_cxx;

  auto control = &source.control;

  {
    auto instance = source.instantiate("Ptr", {control->getCharType()});
    ASSERT_TRUE(instance != nullptr);
    ASSERT_TRUE(instance->isTypeAlias());
    ASSERT_EQ(to_string(instance->type()), "char*");
  }

  {
    auto instance = source.instantiate(
        "F", {control->getIntType(), control->getCharType()});

    ASSERT_TRUE(instance != nullptr);
    ASSERT_TRUE(instance->isTypeAlias());
    ASSERT_EQ(to_string(instance->type()), "int (*)(const int, char*...)");
  }

  {
    auto instance = source.instantiate("RefF", {control->getCharType()});
    ASSERT_TRUE(instance != nullptr);
    ASSERT_TRUE(instance->isTypeAlias());
    ASSERT_EQ(to_string(instance->type()), "void (char&, char&&)");
  }
}

TEST(Substitution, Variable) {
  auto source = R"(
    template <typename T>
    constexpr T value = T{};
  )"_cxx;

  auto control = &source.control;

  {
    auto instance = source.instantiate("value", {control->getIntType()});
    ASSERT_TRUE(instance != nullptr);
    ASSERT_TRUE(instance->isVariable());
    ASSERT_EQ(to_string(instance->type()), "int");
  }
}

TEST(Substitution, Class) {
  auto source = R"(
    struct M {};

    template <typename T>
    struct S {
      using type = T;
      T value;

      void foo();
      void foo(int x);
      void foo(T* ptr);

      auto get_value() -> const T&;

      M m;
      int v[10];

      S(T* ptr, void* x);

      enum E {
        A, B, C
      };
    };
  )"_cxx;

  LookupMember getMember{source};

  auto control = &source.control;

  {
    auto instance = source.instantiate("S", {control->getIntType()});
    ASSERT_TRUE(instance != nullptr);
    ASSERT_TRUE(instance->isClass());

    auto S = symbol_cast<ClassSymbol>(instance);
    ASSERT_TRUE(S != nullptr);

    auto S2 = symbol_cast<ClassSymbol>(
        source.instantiate("S", {control->getCharType()}));
    ASSERT_TRUE(S2 != nullptr);

    // test constructors
    ASSERT_EQ(S->constructors().size(), std::size_t(1));
    ASSERT_EQ(S2->constructors().size(), std::size_t(1));
    ASSERT_EQ(to_string(S->constructors()[0]->type()), " (int*, void*)");
    ASSERT_EQ(to_string(S2->constructors()[0]->type()), " (char*, void*)");

    auto type = getMember(S->scope(), "type");
    ASSERT_TRUE(type != nullptr);
    ASSERT_TRUE(type->isTypeAlias());
    ASSERT_EQ(to_string(type->type()), "int");

    auto value = getMember(S->scope(), "value");
    ASSERT_TRUE(value != nullptr);
    ASSERT_TRUE(value->isField());
    ASSERT_EQ(to_string(value->type()), "int");

    auto foo = symbol_cast<OverloadSetSymbol>(getMember(S->scope(), "foo"));
    ASSERT_TRUE(foo != nullptr);
    ASSERT_EQ(foo->functions().size(), std::size_t(3));
    ASSERT_EQ(to_string(foo->functions()[0]->type()), "void ()");
    ASSERT_EQ(to_string(foo->functions()[1]->type()), "void (int)");
    ASSERT_EQ(to_string(foo->functions()[2]->type()), "void (int*)");

    ASSERT_EQ(to_string(getMember(S->scope(), "get_value")->type()),
              "const int& ()");

    ASSERT_EQ(getMember(S->scope(), "m")->type(),
              getMember(S2->scope(), "m")->type());
  }
}
