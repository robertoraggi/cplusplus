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

namespace {

[[nodiscard]] auto make_substitution(
    TranslationUnit* unit, TemplateDeclarationAST* templateDecl,
    List<TemplateArgumentAST*>* templateArgumentList)
    -> std::vector<TemplateArgument> {
  auto control = unit->control();
  auto interp = ASTInterpreter{unit};

  std::vector<TemplateArgument> templateArguments;

  for (auto arg : ListView{templateArgumentList}) {
    if (auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg)) {
      auto expr = exprArg->expression;
      // ### need to set scope and location
      auto templArg = control->newVariableSymbol(nullptr, {});
      templArg->setInitializer(expr);
      templArg->setType(control->add_const(expr->type));
      templArg->setConstValue(interp.evaluate(expr));
      if (!templArg->constValue().has_value())
        cxx_runtime_error("template argument is not a constant expression");
      templateArguments.push_back(templArg);
    } else if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
      auto type = typeArg->typeId->type;
      // ### need to set scope and location
      auto templArg = control->newTypeAliasSymbol(nullptr, {});
      templArg->setType(type);
      templateArguments.push_back(templArg);
    }
  }

  return templateArguments;
}

template <typename Node>
[[nodiscard]] auto substitute(Source& source, Node* ast,
                              std::vector<TemplateArgument> args) {
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

}  // namespace

TEST(Rewriter, TypeAlias) {
  auto source = R"(
template <typename T, typename U>
using Func = void(T, const U&);

using Func1 = Func<int, float>;
  )"_cxx_no_templates;

  auto control = source.control();

  auto func1 = source.getAs<TypeAliasSymbol>("Func1");
  ASSERT_TRUE(func1 != nullptr);

  std::cout << "Func1: " << to_string(func1->type()) << "\n";

  auto func1Type = type_cast<UnresolvedNameType>(func1->type());
  ASSERT_TRUE(func1Type != nullptr);

  auto templateId = ast_cast<SimpleTemplateIdAST>(func1Type->unqualifiedId());
  ASSERT_TRUE(templateId != nullptr);
  auto templateSym =
      symbol_cast<TypeAliasSymbol>(templateId->primaryTemplateSymbol);
  ASSERT_TRUE(templateSym != nullptr);

  auto templateArguments =
      make_substitution(&source.unit, templateSym->templateDeclaration(),
                        templateId->templateArgumentList);

  auto funcInstance = substitute(source,
                                 getTemplateBodyAs<AliasDeclarationAST>(
                                     templateSym->templateDeclaration()),
                                 templateArguments);
  ASSERT_EQ(to_string(funcInstance->typeId->type), "void (int, const float&)");
}

TEST(Rewriter, Var) {
  auto source = R"(
template <int i>
const int c = i + 321 + i;

constexpr int y = c<123 * 2>;
)"_cxx_no_templates;

  auto interp = ASTInterpreter{&source.unit};

  auto control = source.control();

  auto y = source.getAs<VariableSymbol>("y");
  ASSERT_TRUE(y != nullptr);
  auto yinit = ast_cast<EqualInitializerAST>(y->initializer())->expression;
  ASSERT_TRUE(yinit != nullptr);

  auto idExpr = ast_cast<IdExpressionAST>(yinit);
  ASSERT_TRUE(idExpr != nullptr);

  ASSERT_TRUE(idExpr->symbol);

  auto templateId = ast_cast<SimpleTemplateIdAST>(idExpr->unqualifiedId);
  ASSERT_TRUE(templateId != nullptr);

  // get the primary template declaration
  auto templateSym =
      symbol_cast<VariableSymbol>(templateId->primaryTemplateSymbol);
  ASSERT_TRUE(templateSym != nullptr);
  auto templateDecl = getTemplateBodyAs<SimpleDeclarationAST>(
      templateSym->templateDeclaration());
  ASSERT_TRUE(templateDecl != nullptr);

  std::vector<TemplateArgument> templateArguments =
      make_substitution(&source.unit, templateSym->templateDeclaration(),
                        templateId->templateArgumentList);

  auto instance = substitute(source, templateDecl, templateArguments);
  ASSERT_TRUE(instance != nullptr);

  auto decl = instance->initDeclaratorList->value;
  ASSERT_TRUE(decl != nullptr);
  auto init = ast_cast<EqualInitializerAST>(decl->initializer);
  ASSERT_TRUE(init != nullptr);
  auto value = interp.evaluate(init->expression);
  ASSERT_TRUE(value.has_value());
  ASSERT_EQ(std::visit(ArithmeticCast<int>{}, *value), 123 * 2 + 321 + 123 * 2);
}
