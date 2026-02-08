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
    if (!funcDef) return nullptr;

    auto instance =
        ast_cast<FunctionDefinitionAST>(rewriter.declaration(funcDef));

    if (!instance) return nullptr;

    return instance->symbol;
  }
  auto operator()(Symbol*) -> Symbol* { return nullptr; }
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
      auto param = symbol_cast<NonTypeParameterSymbol>(id->symbol);
      if (!param) continue;

      if (param->depth() != 0) continue;

      auto arg = templateArguments_[param->index()];
      auto argSymbol = std::get<Symbol*>(arg);

      auto parameterPack = symbol_cast<ParameterPackSymbol>(argSymbol);
      if (parameterPack) return parameterPack;
    }

    if (auto named = ast_cast<NamedTypeSpecifierAST>(astNode)) {
      auto typeParam = symbol_cast<TypeParameterSymbol>(named->symbol);
      if (!typeParam) continue;

      auto paramType = type_cast<TypeParameterType>(typeParam->type());
      if (!paramType || !paramType->isParameterPack()) continue;
      if (paramType->depth() != depth_) continue;

      auto index = paramType->index();
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
    auto typeParam = symbol_cast<TypeParameterSymbol>(named->symbol);
    if (!typeParam) return nullptr;

    auto paramType = type_cast<TypeParameterType>(typeParam->type());
    if (!paramType || !paramType->isParameterPack()) return nullptr;
    if (paramType->depth() != depth_) return nullptr;

    auto index = paramType->index();
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

  auto classSymbol = symbol_cast<ClassSymbol>(symbol);
  auto variableSymbol = symbol_cast<VariableSymbol>(symbol);
  auto typeAliasSymbol = symbol_cast<TypeAliasSymbol>(symbol);

  auto templateDecl = visit(GetTemplateDeclaration{}, symbol);
  if (!templateDecl) return nullptr;

  auto declaration = visit(GetDeclaration{}, symbol);
  if (!declaration) return nullptr;

  auto templateArguments =
      make_substitution(unit, templateDecl, templateArgumentList);

  auto is_primary_template = [&]() -> bool {
    int expected = 0;
    for (const auto& arg : templateArguments) {
      if (!std::holds_alternative<Symbol*>(arg)) return false;

      auto sym = std::get<Symbol*>(arg);

      if (auto pack = symbol_cast<ParameterPackSymbol>(sym)) {
        if (pack->elements().size() != 1) return false;
        auto ty = type_cast<TypeParameterType>(pack->elements()[0]->type());
        if (!ty) return false;
        if (ty->index() != expected) return false;
        if (!ty->isParameterPack()) return false;
        ++expected;
        continue;
      }

      auto ty = type_cast<TypeParameterType>(sym->type());
      if (!ty) return false;

      if (ty->index() != expected) return false;
      ++expected;
    }
    return true;
  };

  if (is_primary_template()) {
    // if this is a primary template, we can just return the class symbol
    return symbol;
  }

  auto specialization = visit(GetSpecialization{templateArguments}, symbol);

  if (specialization) return specialization;

  if (templateDecl->requiresClause) {
    auto parentScope = symbol->enclosingNonTemplateParametersScope();
    auto reqRewriter = ASTRewriter{unit, parentScope, templateArguments};
    reqRewriter.depth_ = templateDecl->depth;
    auto rewrittenClause =
        reqRewriter.requiresClause(templateDecl->requiresClause);
    if (rewrittenClause && rewrittenClause->expression) {
      reqRewriter.check(rewrittenClause->expression);
      auto interp = ASTInterpreter{unit};
      auto val = interp.evaluate(rewrittenClause->expression);
      if (val.has_value()) {
        auto boolVal = interp.toBool(*val);
        if (boolVal.has_value() && !*boolVal) {
          return nullptr;
        }
      }
    }
  }

  if (auto funcDef = ast_cast<FunctionDefinitionAST>(declaration)) {
    if (funcDef->requiresClause) {
      auto parentScope = symbol->enclosingNonTemplateParametersScope();
      auto reqRewriter = ASTRewriter{unit, parentScope, templateArguments};
      reqRewriter.depth_ = templateDecl->depth;
      auto rewrittenClause =
          reqRewriter.requiresClause(funcDef->requiresClause);
      if (rewrittenClause && rewrittenClause->expression) {
        reqRewriter.check(rewrittenClause->expression);
        auto interp = ASTInterpreter{unit};
        auto val = interp.evaluate(rewrittenClause->expression);
        if (val.has_value()) {
          auto boolVal = interp.toBool(*val);
          if (boolVal.has_value() && !*boolVal) {
            return nullptr;
          }
        }
      }
    }
  }

  if (classSymbol) {
    auto tryPartialSpecialization = [&]() -> Symbol* {
      for (const auto& spec : classSymbol->specializations()) {
        auto specClass = symbol_cast<ClassSymbol>(spec.symbol);
        if (!specClass) continue;

        auto specTemplateDecl = specClass->templateDeclaration();
        if (!specTemplateDecl) continue;

        const auto& patternArgs = spec.arguments;
        if (patternArgs.size() != templateArguments.size()) continue;

        bool matches = true;
        std::vector<TemplateArgument> deducedArgs;

        int specParamCount = 0;
        std::map<std::pair<int, int>, int> paramPositionMap;
        for (auto p : ListView{specTemplateDecl->templateParameterList}) {
          paramPositionMap[{p->depth, p->index}] = specParamCount;
          ++specParamCount;
        }
        deducedArgs.resize(specParamCount);

        auto paramPosition = [&](const TypeParameterType* ty) -> int {
          auto it = paramPositionMap.find({ty->depth(), ty->index()});
          if (it != paramPositionMap.end()) return it->second;
          return -1;
        };

        for (size_t i = 0; i < patternArgs.size(); ++i) {
          const auto& pat = patternArgs[i];
          const auto& conc = templateArguments[i];

          auto patSym = std::get_if<Symbol*>(&pat);
          auto concSym = std::get_if<Symbol*>(&conc);
          if (!patSym || !concSym) {
            matches = false;
            break;
          }

          auto patPack = symbol_cast<ParameterPackSymbol>(*patSym);
          auto concPack = symbol_cast<ParameterPackSymbol>(*concSym);

          if (patPack && concPack) {
            const auto& patElems = patPack->elements();
            const auto& concElems = concPack->elements();

            size_t patIdx = 0;
            size_t concIdx = 0;

            while (patIdx < patElems.size() && concIdx < concElems.size()) {
              auto patElem = patElems[patIdx];
              auto patElemType = type_cast<TypeParameterType>(patElem->type());

              if (patElemType && patElemType->isParameterPack()) {
                auto deducedPack =
                    unit->control()->newParameterPackSymbol(nullptr, {});
                while (concIdx < concElems.size()) {
                  deducedPack->addElement(concElems[concIdx]);
                  ++concIdx;
                }
                auto pos = paramPosition(patElemType);
                if (pos >= 0 && pos < static_cast<int>(deducedArgs.size())) {
                  deducedArgs[pos] = deducedPack;
                }
                ++patIdx;
                continue;
              }

              if (patElemType) {
                auto pos = paramPosition(patElemType);
                if (pos >= 0 && pos < static_cast<int>(deducedArgs.size())) {
                  deducedArgs[pos] = concElems[concIdx];
                }
                ++patIdx;
                ++concIdx;
                continue;
              }

              if (patElem->type() != concElems[concIdx]->type()) {
                matches = false;
                break;
              }
              ++patIdx;
              ++concIdx;
            }

            while (patIdx < patElems.size() && matches) {
              auto patElem = patElems[patIdx];
              auto patElemType = type_cast<TypeParameterType>(patElem->type());
              if (patElemType && patElemType->isParameterPack()) {
                auto deducedPack =
                    unit->control()->newParameterPackSymbol(nullptr, {});
                auto pos = paramPosition(patElemType);
                if (pos >= 0 && pos < static_cast<int>(deducedArgs.size())) {
                  deducedArgs[pos] = deducedPack;
                }
                ++patIdx;
              } else {
                matches = false;
              }
            }

            if (concIdx != concElems.size()) matches = false;
            if (!matches) break;
            continue;
          }

          auto patParamType = type_cast<TypeParameterType>((*patSym)->type());
          if (patParamType) {
            auto pos = paramPosition(patParamType);
            if (pos >= 0 && pos < static_cast<int>(deducedArgs.size())) {
              deducedArgs[pos] = conc;
            }
            continue;
          }

          auto patVar = symbol_cast<VariableSymbol>(*patSym);
          auto concVar = symbol_cast<VariableSymbol>(*concSym);
          if (patVar && concVar) {
            if (!patVar->constValue().has_value() ||
                !concVar->constValue().has_value()) {
              matches = false;
              break;
            }
            if (patVar->constValue().value() != concVar->constValue().value()) {
              matches = false;
              break;
            }
            continue;
          }

          if ((*patSym)->type() == (*concSym)->type()) continue;

          matches = false;
          break;
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
          auto cachedClass = symbol_cast<ClassSymbol>(cached);
          if (cachedClass) {
            classSymbol->addSpecialization(templateArguments, cachedClass);
          }
          return cached;
        }

        auto specParentScope = specClass->enclosingNonTemplateParametersScope();
        auto specRewriter = ASTRewriter{unit, specParentScope, deducedArgs};
        specRewriter.depth_ = specTemplateDecl->depth;
        specRewriter.binder().setInstantiatingSymbol(specClass);

        auto specBody = ast_cast<ClassSpecifierAST>(specClass->declaration());
        if (!specBody) continue;

        auto instance =
            ast_cast<ClassSpecifierAST>(specRewriter.specifier(specBody));
        if (!instance || !instance->symbol) continue;

        auto instanceClass = symbol_cast<ClassSymbol>(instance->symbol);
        if (instanceClass) {
          classSymbol->addSpecialization(templateArguments, instanceClass);
        }

        return instance->symbol;
      }
      return nullptr;
    };

    if (auto result = tryPartialSpecialization()) return result;
  }

  auto parentScope = symbol->enclosingNonTemplateParametersScope();

  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = templateDecl->depth;

  rewriter.binder().setInstantiatingSymbol(symbol);

  auto instance = visit(Instantiate{rewriter}, symbol);

  return instance;
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
