// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbol_instantiation.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <gtest/gtest.h>

#include <format>
#include <iostream>
#include <sstream>

#include "test_utils.h"

using namespace cxx;

template <typename Node>
auto subst(Source& source, Node* ast, std::vector<TemplateArgument> args) {
  auto control = source.control();
  TypeChecker typeChecker(&source.unit);
  ASTRewriter rewrite{&typeChecker, args};
  return ast_cast<Node>(rewrite(ast));
};

[[nodiscard]] auto getTemplateBody(TemplateDeclarationAST* ast)
    -> DeclarationAST* {
  if (auto nested = ast_cast<TemplateDeclarationAST>(ast->declaration))
    return getTemplateBody(nested);
  return ast->declaration;
}

template <typename Node>
[[nodiscard]] auto getTemplateBodyAs(TemplateDeclarationAST* ast) -> Node* {
  return ast_cast<Node>(getTemplateBody(ast));
}

TEST(Rewriter, TypeAlias) {
  auto source = R"(
template <typename T>
using Ptr = const T*;

template <typename T, typename U>
using Func = void(T, const U&);
  )"_cxx;

  auto control = source.control();

  auto ptrInstance =
      subst(source,
            getTemplateBodyAs<AliasDeclarationAST>(
                source.getAs<TypeAliasSymbol>("Ptr")->templateDeclaration()),
            {control->getIntType()});

  ASSERT_EQ(to_string(ptrInstance->typeId->type), "const int*");

  auto funcInstance =
      subst(source,
            getTemplateBodyAs<AliasDeclarationAST>(
                source.getAs<TypeAliasSymbol>("Func")->templateDeclaration()),
            {control->getIntType(), control->getFloatType()});
  ASSERT_EQ(to_string(funcInstance->typeId->type), "void (int, const float&)");
}

TEST(Rewriter, Var) {
  auto source = R"(
template <int i>
const int c = i + 321;

constexpr int x = 123;

)"_cxx;

  auto control = source.control();

  auto c = source.getAs<VariableSymbol>("c");
  ASSERT_TRUE(c != nullptr);
  auto templateDeclaration = c->templateDeclaration();
  ASSERT_TRUE(templateDeclaration != nullptr);

  auto x = source.getAs<VariableSymbol>("x");
  ASSERT_TRUE(x != nullptr);
  auto xinit = x->initializer();
  ASSERT_TRUE(xinit != nullptr);

  auto instance = subst(
      source, getTemplateBodyAs<SimpleDeclarationAST>(templateDeclaration),
      {xinit});

  auto decl = instance->initDeclaratorList->value;
  ASSERT_TRUE(decl != nullptr);

  auto init = ast_cast<EqualInitializerAST>(decl->initializer);
  ASSERT_TRUE(init);

  ASTInterpreter interp{&source.unit};

  auto value = interp.evaluate(init->expression);

  ASSERT_TRUE(value.has_value());

  ASSERT_EQ(std::visit(ArithmeticCast<int>{}, *value), 123 + 321);
}
