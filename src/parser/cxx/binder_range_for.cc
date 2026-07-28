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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <vector>

namespace cxx {
namespace {
auto rangeDeclarationVariable(DeclarationAST* rangeDeclaration)
    -> VariableSymbol* {
  auto simpleDecl = ast_cast<SimpleDeclarationAST>(rangeDeclaration);
  if (!simpleDecl || !simpleDecl->initDeclaratorList) return nullptr;

  auto initDeclarator = simpleDecl->initDeclaratorList->value;
  if (!initDeclarator) return nullptr;

  return symbol_cast<VariableSymbol>(initDeclarator->symbol);
}

auto resolveRangeIteration(TranslationUnit* unit, ForRangeStatementAST* ast,
                           const Type* rangeType) -> const Type* {
  auto traits = unit->typeTraits();

  if (auto arrayType = type_cast<BoundedArrayType>(rangeType)) {
    ast->isPointerIterator = true;
    return arrayType->elementType();
  }

  auto classType = type_cast<ClassType>(rangeType);
  if (!classType) return nullptr;

  auto classSymbol = classType->symbol();
  if (classSymbol) classSymbol = classSymbol->resolvedDefinition();
  if (!classSymbol) return nullptr;

  (void)traits.requireCompleteClass(classSymbol);

  auto beginName = unit->control()->getIdentifier("begin");
  auto endName = unit->control()->getIdentifier("end");

  auto beginFunc = views::find_function(classSymbol->find(beginName),
                                        [](FunctionSymbol*) { return true; });
  auto endFunc = views::find_function(classSymbol->find(endName),
                                      [](FunctionSymbol*) { return true; });

  if (!beginFunc || !endFunc) {
    std::vector<const Type*> argTypes = {rangeType};

    auto beginCandidates = argumentDependentLookup(unit, beginName, argTypes);
    auto endCandidates = argumentDependentLookup(unit, endName, argTypes);

    if (!beginCandidates.empty()) beginFunc = beginCandidates.front();
    if (!endCandidates.empty()) endFunc = endCandidates.front();
  }

  if (!beginFunc || !endFunc) return nullptr;

  ast->beginFunction = beginFunc;
  ast->endFunction = endFunc;
  ast->usesMemberBeginEnd = beginFunc->parent() == classSymbol;

  auto beginFuncType = type_cast<FunctionType>(beginFunc->type());
  if (!beginFuncType) return nullptr;

  auto iterType = traits.remove_cvref(beginFuncType->returnType());

  if (traits.is_pointer(iterType)) {
    ast->isPointerIterator = true;
    return traits.get_element_type(iterType);
  }

  auto iterClassType = type_cast<ClassType>(iterType);
  if (!iterClassType) {
    ast->isPointerIterator = true;
    return nullptr;
  }

  auto iterClass = iterClassType->symbol();
  if (iterClass) iterClass = iterClass->resolvedDefinition();
  if (!iterClass) return nullptr;

  (void)traits.requireCompleteClass(iterClass);

  auto definedIterType = iterClass->type();

  auto placeholder = ThisExpressionAST::create(
      unit->arena(), ValueCategory::kLValue, definedIterType);

  OverloadResolution resolution(unit);
  ast->derefFunction = resolution.lookupOperator(
      definedIterType, TokenKind::T_STAR, nullptr, placeholder);
  ast->incrementFunction = resolution.lookupOperator(
      definedIterType, TokenKind::T_PLUS_PLUS, nullptr, placeholder);
  ast->notEqualFunction =
      resolution.lookupOperator(definedIterType, TokenKind::T_EXCLAIM_EQUAL,
                                definedIterType, placeholder, placeholder);

  if (!ast->derefFunction) return nullptr;

  auto derefFuncType = type_cast<FunctionType>(ast->derefFunction->type());
  if (!derefFuncType) return nullptr;

  auto returnType = derefFuncType->returnType();
  if (traits.is_lvalue_reference(returnType) ||
      traits.is_rvalue_reference(returnType)) {
    return traits.remove_reference(returnType);
  }
  return returnType;
}
}  // namespace

void Binder::finishForRangeDeclaration(ForRangeStatementAST* ast) {
  auto rangeInitializer = ast->rangeInitializer;
  auto var = rangeDeclarationVariable(ast->rangeDeclaration);

  TypeChecker check{unit_};
  check.setScope(scope());

  const bool needsDeduction = var && check.hasAutoPlaceholder(var->type());

  if (!rangeInitializer || !rangeInitializer->type ||
      isDependent(unit_, rangeInitializer->type)) {
    if (needsDeduction && isEnclosedInTemplate(scope()))
      var->setType(control()->getTypeParameterType(0, 0, false));
    return;
  }

  auto rangeType = traits.remove_cvref(rangeInitializer->type);

  auto elementType = resolveRangeIteration(unit_, ast, rangeType);
  if (!elementType || !needsDeduction) return;

  auto deduced = check.deduceAutoType(var->type(), elementType);
  if (!deduced) return;

  var->setType(deduced);

  if (auto classType = type_cast<ClassType>(traits.remove_cvref(var->type()))) {
    (void)traits.requireCompleteClass(classType->symbol());
  }
}
}  // namespace cxx
