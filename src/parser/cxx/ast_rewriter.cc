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

#include <cxx/ast_rewriter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>

// c++
#include <map>

// todo remove
#include <cxx/ast_cursor.h>

namespace cxx {

namespace {

struct GetTemplateDeclaration {
  auto operator()(ClassSymbol* symbol) -> TemplateDeclarationAST* {
    return symbol->templateDeclaration();
  }

  auto operator()(VariableSymbol* symbol) -> TemplateDeclarationAST* {
    return symbol->templateDeclaration();
  }

  auto operator()(TypeAliasSymbol* symbol) -> TemplateDeclarationAST* {
    return symbol->templateDeclaration();
  }

  auto operator()(FunctionSymbol* symbol) -> TemplateDeclarationAST* {
    return symbol->templateDeclaration();
  }

  auto operator()(Symbol*) -> TemplateDeclarationAST* { return nullptr; }
};

struct GetDeclaration {
  auto operator()(ClassSymbol* symbol) -> AST* { return symbol->declaration(); }

  auto operator()(VariableSymbol* symbol) -> AST* {
    return symbol->templateDeclaration()->declaration;
  }

  auto operator()(TypeAliasSymbol* symbol) -> AST* {
    return symbol->templateDeclaration()->declaration;
  }

  auto operator()(FunctionSymbol* symbol) -> AST* {
    return symbol->declaration();
  }

  auto operator()(Symbol*) -> AST* { return nullptr; }
};

struct GetSpecialization {
  const std::vector<TemplateArgument>& templateArguments;

  auto operator()(ClassSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(VariableSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(TypeAliasSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(FunctionSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(Symbol*) -> Symbol* { return nullptr; }
};

struct Instantiate {
  ASTRewriter& rewriter;

  auto operator()(ClassSymbol* symbol) -> Symbol* {
    auto classSpecifier = ast_cast<ClassSpecifierAST>(symbol->declaration());
    if (!classSpecifier) return nullptr;

    auto instance =
        ast_cast<ClassSpecifierAST>(rewriter.specifier(classSpecifier));

    if (!instance) return nullptr;

    return instance->symbol;
  }

  auto operator()(VariableSymbol* symbol) -> Symbol* {
    auto declaration = symbol->templateDeclaration()->declaration;
    auto instance = ast_cast<SimpleDeclarationAST>(
        rewriter.declaration(ast_cast<SimpleDeclarationAST>(declaration)));

    if (!instance) return nullptr;

    auto instantiatedSymbol = instance->initDeclaratorList->value->symbol;
    auto instantiatedVariable = symbol_cast<VariableSymbol>(instantiatedSymbol);

    return instantiatedVariable;
  }

  auto operator()(TypeAliasSymbol* symbol) -> Symbol* {
    auto declaration = symbol->templateDeclaration()->declaration;

    auto instance = ast_cast<AliasDeclarationAST>(
        rewriter.declaration(ast_cast<AliasDeclarationAST>(declaration)));

    if (!instance) return nullptr;

    return instance->symbol;
  }

  auto operator()(FunctionSymbol* symbol) -> Symbol* {
    auto funcDef = symbol->declaration();
    if (funcDef) {
      auto instance =
          ast_cast<FunctionDefinitionAST>(rewriter.declaration(funcDef));

      if (!instance) return nullptr;

      return instance->symbol;
    }

    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;

    auto decl = templateDecl->declaration;
    if (!decl) return nullptr;

    auto instance = ast_cast<SimpleDeclarationAST>(
        rewriter.declaration(ast_cast<SimpleDeclarationAST>(decl)));

    if (!instance) return nullptr;
    if (!instance->initDeclaratorList) return nullptr;

    return instance->initDeclaratorList->value->symbol;
  }

  auto operator()(Symbol*) -> Symbol* { return nullptr; }
};

auto isPrimaryTemplate(const std::vector<TemplateArgument>& templateArguments)
    -> bool {
  int expected = 0;
  for (const auto& arg : templateArguments) {
    if (!std::holds_alternative<Symbol*>(arg)) return false;

    auto sym = std::get<Symbol*>(arg);

    if (auto pack = symbol_cast<ParameterPackSymbol>(sym)) {
      if (pack->elements().size() != 1) return false;
      auto ty = getTypeParamInfo(pack->elements()[0]->type());
      if (!ty) return false;
      if (ty->index != expected) return false;
      if (!ty->isPack) return false;
      ++expected;
      continue;
    }

    auto ty = getTypeParamInfo(sym->type());
    if (!ty) return false;
    if (ty->index != expected) return false;
    ++expected;
  }
  return true;
}

struct PackDeducer {
  Control* control;
  std::vector<TemplateArgument>& deducedArgs;
  std::function<int(int depth, int index)> paramPosition;

  void setDeduced(int pos, TemplateArgument arg) {
    if (pos >= 0 && pos < static_cast<int>(deducedArgs.size())) {
      deducedArgs[pos] = std::move(arg);
    }
  }

  auto deducePackElements(const std::vector<Symbol*>& patElems,
                          const std::vector<Symbol*>& concElems) -> bool {
    size_t patIdx = 0;
    size_t concIdx = 0;

    while (patIdx < patElems.size() && concIdx < concElems.size()) {
      auto patElem = patElems[patIdx];
      auto patElemInfo = getTypeParamInfo(patElem->type());

      if (patElemInfo && patElemInfo->isPack) {
        auto deducedPack = control->newParameterPackSymbol(nullptr, {});
        while (concIdx < concElems.size()) {
          deducedPack->addElement(concElems[concIdx]);
          ++concIdx;
        }
        setDeduced(paramPosition(patElemInfo->depth, patElemInfo->index),
                   deducedPack);
        ++patIdx;
        continue;
      }

      if (patElemInfo) {
        setDeduced(paramPosition(patElemInfo->depth, patElemInfo->index),
                   concElems[concIdx]);
        ++patIdx;
        ++concIdx;
        continue;
      }

      if (patElem->type() != concElems[concIdx]->type()) return false;
      ++patIdx;
      ++concIdx;
    }

    while (patIdx < patElems.size()) {
      auto patElem = patElems[patIdx];
      auto patElemInfo = getTypeParamInfo(patElem->type());
      if (patElemInfo && patElemInfo->isPack) {
        auto deducedPack = control->newParameterPackSymbol(nullptr, {});
        setDeduced(paramPosition(patElemInfo->depth, patElemInfo->index),
                   deducedPack);
        ++patIdx;
      } else {
        return false;
      }
    }

    return concIdx == concElems.size();
  }
};

auto findInnerTemplateId(ClassSpecifierAST* specBody, size_t argPos)
    -> SimpleTemplateIdAST* {
  auto outerTemplId = ast_cast<SimpleTemplateIdAST>(specBody->unqualifiedId);
  if (!outerTemplId) return nullptr;

  int idx = 0;
  for (auto arg : ListView{outerTemplId->templateArgumentList}) {
    if (idx == static_cast<int>(argPos)) {
      auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg);
      if (!typeArg || !typeArg->typeId) return nullptr;

      for (auto sp : ListView{typeArg->typeId->typeSpecifierList}) {
        if (auto named = ast_cast<NamedTypeSpecifierAST>(sp)) {
          if (auto templId =
                  ast_cast<SimpleTemplateIdAST>(named->unqualifiedId)) {
            return templId;
          }
        }
      }
      return nullptr;
    }
    ++idx;
  }
  return nullptr;
}

struct PartialSpecMatcher {
  TranslationUnit* unit;
  ClassSpecifierAST* specBody;
  std::vector<TemplateArgument>& deducedArgs;
  std::function<int(int depth, int index)> paramPosition;

  [[nodiscard]] auto control() const -> Control* { return unit->control(); }

  auto matchArg(const TemplateArgument& pat, const TemplateArgument& conc,
                size_t argPos) -> bool {
    auto patSym = std::get_if<Symbol*>(&pat);
    auto concSym = std::get_if<Symbol*>(&conc);
    if (!patSym || !concSym) return false;

    auto patPack = symbol_cast<ParameterPackSymbol>(*patSym);
    auto concPack = symbol_cast<ParameterPackSymbol>(*concSym);
    if (patPack && concPack) {
      PackDeducer deducer{control(), deducedArgs, paramPosition};
      return deducer.deducePackElements(patPack->elements(),
                                        concPack->elements());
    }

    if (auto patParamInfo = getTypeParamInfo((*patSym)->type())) {
      auto pos = paramPosition(patParamInfo->depth, patParamInfo->index);
      if (pos >= 0 && pos < static_cast<int>(deducedArgs.size())) {
        deducedArgs[pos] = conc;
      }
      return true;
    }

    auto patVar = symbol_cast<VariableSymbol>(*patSym);
    auto concVar = symbol_cast<VariableSymbol>(*concSym);
    if (patVar && concVar) {
      if (!patVar->constValue().has_value() ||
          !concVar->constValue().has_value())
        return false;
      return patVar->constValue().value() == concVar->constValue().value();
    }

    if (auto patClassType = type_cast<ClassType>((*patSym)->type())) {
      if (auto concClassType = type_cast<ClassType>((*concSym)->type())) {
        auto patClassSym = patClassType->symbol();
        auto concClassSym = concClassType->symbol();

        if (concClassSym->isSpecialization() &&
            concClassSym->primaryTemplateSymbol() == patClassSym &&
            patClassSym->templateDeclaration()) {
          return matchNestedClassTemplate(patClassSym, concClassSym, argPos);
        }
      }
    }

    return (*patSym)->type() == (*concSym)->type();
  }

 private:
  auto matchNestedClassTemplate(ClassSymbol* patClassSym,
                                ClassSymbol* concClassSym, size_t argPos)
      -> bool {
    auto concInnerArgs = concClassSym->templateArguments();

    auto innerTemplId = findInnerTemplateId(specBody, argPos);
    if (!innerTemplId) return false;

    auto patInnerArgs =
        ASTRewriter::make_substitution(unit, patClassSym->templateDeclaration(),
                                       innerTemplId->templateArgumentList);

    if (patInnerArgs.size() != concInnerArgs.size()) return false;

    std::vector<TemplateArgument> concInnerVec(concInnerArgs.begin(),
                                               concInnerArgs.end());

    for (size_t j = 0; j < patInnerArgs.size(); ++j) {
      auto ipSym = std::get_if<Symbol*>(&patInnerArgs[j]);
      auto icSym = std::get_if<Symbol*>(&concInnerVec[j]);
      if (!ipSym || !icSym) return false;

      auto ipPack = symbol_cast<ParameterPackSymbol>(*ipSym);
      auto icPack = symbol_cast<ParameterPackSymbol>(*icSym);

      if (ipPack && icPack) {
        PackDeducer deducer{control(), deducedArgs, paramPosition};
        if (!deducer.deducePackElements(ipPack->elements(), icPack->elements()))
          return false;
        continue;
      }

      if (auto innerPTInfo = getTypeParamInfo((*ipSym)->type())) {
        auto pos = paramPosition(innerPTInfo->depth, innerPTInfo->index);
        if (pos >= 0 && pos < static_cast<int>(deducedArgs.size())) {
          deducedArgs[pos] = concInnerVec[j];
        }
        continue;
      }

      if ((*ipSym)->type() != (*icSym)->type()) return false;
    }

    return true;
  }
};

}  // namespace

ASTRewriter::ASTRewriter(TranslationUnit* unit, ScopeSymbol* scope,
                         const std::vector<TemplateArgument>& templateArguments)
    : unit_(unit), templateArguments_(templateArguments), binder_(unit_) {
  binder_.setScope(scope);
}

ASTRewriter::~ASTRewriter() {}

auto ASTRewriter::control() const -> Control* { return unit_->control(); }

auto ASTRewriter::arena() const -> Arena* { return unit_->arena(); }

auto ASTRewriter::restrictedToDeclarations() const -> bool {
  return restrictedToDeclarations_;
}

void ASTRewriter::setRestrictedToDeclarations(bool restrictedToDeclarations) {
  restrictedToDeclarations_ = restrictedToDeclarations;
}

void ASTRewriter::warning(SourceLocation loc, std::string message) {
  binder_.warning(loc, std::move(message));
}

void ASTRewriter::error(SourceLocation loc, std::string message) {
  binder_.error(loc, std::move(message));
}

auto ASTRewriter::checkRequiresClause(
    TranslationUnit* unit, Symbol* symbol, RequiresClauseAST* clause,
    const std::vector<TemplateArgument>& templateArguments, int depth) -> bool {
  if (!clause) return true;

  auto parentScope = symbol->enclosingNonTemplateParametersScope();
  auto reqRewriter = ASTRewriter{unit, parentScope, templateArguments};
  reqRewriter.depth_ = depth;
  auto rewrittenClause = reqRewriter.requiresClause(clause);
  if (!rewrittenClause || !rewrittenClause->expression) return true;

  reqRewriter.check(rewrittenClause->expression);
  auto interp = ASTInterpreter{unit};
  auto val = interp.evaluate(rewrittenClause->expression);
  if (!val.has_value()) return true;

  auto boolVal = interp.toBool(*val);
  if (boolVal.has_value() && !*boolVal) return false;

  return true;
}

void ASTRewriter::check(ExpressionAST* ast) {
  auto typeChecker = TypeChecker{unit_};
  typeChecker.setScope(binder_.scope());
  typeChecker.check(ast);
}

auto ASTRewriter::getParameterPack(ExpressionAST* ast) -> ParameterPackSymbol* {
  for (auto cursor = ASTCursor{ast, {}}; cursor; ++cursor) {
    const auto& current = *cursor;
    if (!std::holds_alternative<AST*>(current.node)) continue;

    auto astNode = std::get<AST*>(current.node);

    if (auto id = ast_cast<IdExpressionAST>(astNode)) {
      if (auto param = symbol_cast<NonTypeParameterSymbol>(id->symbol)) {
        if (param->depth() != 0) continue;

        auto arg = templateArguments_[param->index()];
        auto argSymbol = std::get<Symbol*>(arg);

        auto parameterPack = symbol_cast<ParameterPackSymbol>(argSymbol);
        if (parameterPack) return parameterPack;
      }

      if (auto param = symbol_cast<ParameterSymbol>(id->symbol)) {
        auto it = functionParamPacks_.find(param);
        if (it != functionParamPacks_.end()) {
          return it->second;
        }
      }
    }

    if (auto named = ast_cast<NamedTypeSpecifierAST>(astNode)) {
      Symbol* paramSym = symbol_cast<TypeParameterSymbol>(named->symbol);
      if (!paramSym)
        paramSym = symbol_cast<TemplateTypeParameterSymbol>(named->symbol);
      if (!paramSym) continue;

      auto paramInfo = getTypeParamInfo(paramSym->type());
      if (!paramInfo || !paramInfo->isPack) continue;
      if (paramInfo->depth != depth_) continue;

      auto index = paramInfo->index;
      if (index >= static_cast<int>(templateArguments_.size())) continue;

      if (auto sym = std::get_if<Symbol*>(&templateArguments_[index])) {
        if (auto pack = symbol_cast<ParameterPackSymbol>(*sym)) {
          return pack;
        }
      }
    }
  }

  return nullptr;
}

auto ASTRewriter::getTypeParameterPack(SpecifierAST* ast)
    -> ParameterPackSymbol* {
  if (auto named = ast_cast<NamedTypeSpecifierAST>(ast)) {
    Symbol* paramSym = symbol_cast<TypeParameterSymbol>(named->symbol);
    if (!paramSym)
      paramSym = symbol_cast<TemplateTypeParameterSymbol>(named->symbol);
    if (!paramSym) return nullptr;

    auto paramInfo = getTypeParamInfo(paramSym->type());
    if (!paramInfo || !paramInfo->isPack) return nullptr;
    if (paramInfo->depth != depth_) return nullptr;

    auto index = paramInfo->index;
    if (index >= static_cast<int>(templateArguments_.size())) return nullptr;

    if (auto sym = std::get_if<Symbol*>(&templateArguments_[index])) {
      return symbol_cast<ParameterPackSymbol>(*sym);
    }
  }
  return nullptr;
}

auto ASTRewriter::instantiate(TranslationUnit* unit,
                              List<TemplateArgumentAST*>* templateArgumentList,
                              Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;

  auto templateDecl = visit(GetTemplateDeclaration{}, symbol);
  if (!templateDecl) return nullptr;

  auto declaration = visit(GetDeclaration{}, symbol);
  if (!declaration) return nullptr;

  auto templateArguments =
      make_substitution(unit, templateDecl, templateArgumentList);

  if (isPrimaryTemplate(templateArguments)) return symbol;

  if (auto cached = visit(GetSpecialization{templateArguments}, symbol))
    return cached;

  if (!checkRequiresClause(unit, symbol, templateDecl->requiresClause,
                           templateArguments, templateDecl->depth))
    return nullptr;

  if (auto funcDef = ast_cast<FunctionDefinitionAST>(declaration)) {
    if (!checkRequiresClause(unit, symbol, funcDef->requiresClause,
                             templateArguments, templateDecl->depth))
      return nullptr;
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    if (auto result =
            tryPartialSpecialization(unit, classSymbol, templateArguments))
      return result;
  }

  auto parentScope = symbol->enclosingNonTemplateParametersScope();
  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = templateDecl->depth;
  rewriter.binder().setInstantiatingSymbol(symbol);

  return visit(Instantiate{rewriter}, symbol);
}

void ASTRewriter::completePendingBody(TranslationUnit* unit,
                                      FunctionSymbol* func) {
  if (!func || !func->hasPendingBody()) return;

  auto pending = func->pendingBody();

  auto newAst = func->declaration();
  if (!newAst) {
    func->clearPendingBody();
    return;
  }

  auto templateArguments = std::move(pending->templateArguments);
  auto parentScope = pending->parentScope;
  auto depth = pending->depth;
  auto originalDef = pending->originalDefinition;
  func->clearPendingBody();

  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = depth;

  auto functionDeclarator = getFunctionPrototype(newAst->declarator);

  if (functionDeclarator) {
    if (auto params = functionDeclarator->parameterDeclarationClause) {
      rewriter.binder_.setScope(params->functionParametersSymbol);
    } else {
      rewriter.binder_.setScope(func);
    }
  } else {
    rewriter.binder_.setScope(func);
  }

  newAst->functionBody = rewriter.functionBody(originalDef->functionBody);
}

auto ASTRewriter::tryPartialSpecialization(
    TranslationUnit* unit, ClassSymbol* classSymbol,
    const std::vector<TemplateArgument>& templateArguments) -> Symbol* {
  for (const auto& spec : classSymbol->specializations()) {
    auto specClass = symbol_cast<ClassSymbol>(spec.symbol);
    if (!specClass) continue;

    auto specTemplateDecl = specClass->templateDeclaration();
    if (!specTemplateDecl) continue;

    const auto& patternArgs = spec.arguments;
    if (patternArgs.size() != templateArguments.size()) continue;

    auto specBody = ast_cast<ClassSpecifierAST>(specClass->declaration());
    if (!specBody) continue;

    int specParamCount = 0;
    std::map<std::pair<int, int>, int> paramPositionMap;
    for (auto p : ListView{specTemplateDecl->templateParameterList}) {
      paramPositionMap[{p->depth, p->index}] = specParamCount;
      ++specParamCount;
    }

    std::vector<TemplateArgument> deducedArgs(specParamCount);

    auto paramPosition = [&](int depth, int index) -> int {
      auto it = paramPositionMap.find({depth, index});
      if (it != paramPositionMap.end()) return it->second;
      return -1;
    };

    PartialSpecMatcher matcher{unit, specBody, deducedArgs, paramPosition};

    bool matches = true;
    for (size_t i = 0; i < patternArgs.size(); ++i) {
      if (!matcher.matchArg(patternArgs[i], templateArguments[i], i)) {
        matches = false;
        break;
      }
    }
    if (!matches) continue;

    bool allDeduced = true;
    for (const auto& d : deducedArgs) {
      if (!std::holds_alternative<Symbol*>(d)) {
        allDeduced = false;
        break;
      }
    }
    if (!allDeduced) continue;

    if (auto cached = specClass->findSpecialization(deducedArgs)) {
      if (auto cachedClass = symbol_cast<ClassSymbol>(cached)) {
        classSymbol->addSpecialization(templateArguments, cachedClass);
      }
      return cached;
    }

    auto specParentScope = specClass->enclosingNonTemplateParametersScope();
    auto specRewriter = ASTRewriter{unit, specParentScope, deducedArgs};
    specRewriter.depth_ = specTemplateDecl->depth;
    specRewriter.binder().setInstantiatingSymbol(specClass);

    auto instance =
        ast_cast<ClassSpecifierAST>(specRewriter.specifier(specBody));
    if (!instance || !instance->symbol) continue;

    if (auto instanceClass = symbol_cast<ClassSymbol>(instance->symbol)) {
      classSymbol->addSpecialization(templateArguments, instanceClass);
    }

    return instance->symbol;
  }
  return nullptr;
}

auto ASTRewriter::make_substitution(
    TranslationUnit* unit, TemplateDeclarationAST* templateDecl,
    List<TemplateArgumentAST*>* templateArgumentList)
    -> std::vector<TemplateArgument> {
  auto control = unit->control();
  auto interp = ASTInterpreter{unit};

  std::vector<Symbol*> collectedArgs;

  for (auto arg : ListView{templateArgumentList}) {
    if (auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg)) {
      auto expr = exprArg->expression;
      auto value = interp.evaluate(expr);
      if (!value.has_value()) {
#if false
        unit->error(arg->firstSourceLocation(),
                    "template argument is not a constant expression");
#endif
        continue;
      }

      // ### need to set scope and location
      auto templArg = control->newVariableSymbol(nullptr, {});
      templArg->setInitializer(expr);
      auto type = expr->type;
      if (!control->is_scalar(expr->type)) {
        type = control->add_pointer(expr->type);
      }
      templArg->setType(expr->type);
      templArg->setConstexpr(true);
      templArg->setConstValue(value);
      collectedArgs.push_back(templArg);
    } else if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
      auto type = typeArg->typeId->type;
      // ### need to set scope and location
      auto templArg = control->newTypeAliasSymbol(nullptr, {});
      templArg->setType(type);
      collectedArgs.push_back(templArg);
    }
  }

  std::vector<TemplateArgument> templateArguments;

  if (templateDecl) {
    int argIndex = 0;
    for (auto param : ListView{templateDecl->templateParameterList}) {
      bool isPack = false;

      if (auto tyParam = ast_cast<TypenameTypeParameterAST>(param)) {
        isPack = tyParam->isPack;
      } else if (auto ntParam = ast_cast<NonTypeTemplateParameterAST>(param)) {
        isPack = ntParam->declaration && ntParam->declaration->isPack;
      } else if (auto ttParam = ast_cast<TemplateTypeParameterAST>(param)) {
        isPack = ttParam->isPack;
      } else if (auto cParam = ast_cast<ConstraintTypeParameterAST>(param)) {
        isPack = static_cast<bool>(cParam->ellipsisLoc);
      }

      if (isPack) {
        auto pack = control->newParameterPackSymbol(nullptr, {});
        while (argIndex < static_cast<int>(collectedArgs.size())) {
          pack->addElement(collectedArgs[argIndex]);
          ++argIndex;
        }
        templateArguments.push_back(pack);
      } else if (argIndex < static_cast<int>(collectedArgs.size())) {
        templateArguments.push_back(collectedArgs[argIndex]);
        ++argIndex;
      } else {
        if (auto tyParam = ast_cast<TypenameTypeParameterAST>(param)) {
          if (tyParam->typeId && tyParam->typeId->type) {
            auto templArg = control->newTypeAliasSymbol(nullptr, {});
            templArg->setType(tyParam->typeId->type);
            templateArguments.push_back(templArg);
          }
        } else if (auto cParam = ast_cast<ConstraintTypeParameterAST>(param)) {
          if (cParam->typeId && cParam->typeId->type) {
            auto templArg = control->newTypeAliasSymbol(nullptr, {});
            templArg->setType(cParam->typeId->type);
            templateArguments.push_back(templArg);
          }
        }
      }
    }
  } else {
    for (auto sym : collectedArgs) {
      templateArguments.push_back(sym);
    }
  }

  return templateArguments;
}

}  // namespace cxx
