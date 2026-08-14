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
#include <cxx/ast_rewriter.h>
#include <cxx/class_template_deduction.h>
#include <cxx/control.h>
#include <cxx/diagnostics_client.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

namespace cxx {

namespace {

auto parameterClauseOf(FunctionSymbol* function)
    -> ParameterDeclarationClauseAST* {
  auto declaration = function ? function->declaration() : nullptr;
  if (!declaration || !declaration->declarator) return nullptr;
  for (auto chunk : ListView{declaration->declarator->declaratorChunkList}) {
    if (auto functionChunk = ast_cast<FunctionDeclaratorChunkAST>(chunk))
      return functionChunk->parameterDeclarationClause;
  }
  return nullptr;
}

}  // namespace

ClassTemplateArgumentDeduction::ClassTemplateArgumentDeduction(
    TranslationUnit* unit)
    : unit_(unit), control_(unit->control()), arena_(unit->arena()) {}

auto ClassTemplateArgumentDeduction::placeholderClassTemplate(
    SpecifierAST* typeSpecifier, ScopeSymbol* scope) -> ClassSymbol* {
  auto named = ast_cast<NamedTypeSpecifierAST>(typeSpecifier);
  if (!named) return nullptr;
  if (!ast_cast<NameIdAST>(named->unqualifiedId)) return nullptr;

  if (symbol_cast<InjectedClassNameSymbol>(named->symbol)) return nullptr;

  auto classSymbol = symbol_cast<ClassSymbol>(named->symbol);
  if (!classSymbol) return nullptr;
  if (!classSymbol->templateDeclaration()) return nullptr;
  if (classSymbol->isSpecialization()) return nullptr;

  if (names_current_instantiation(classSymbol, scope)) return nullptr;

  return classSymbol;
}

auto ClassTemplateArgumentDeduction::classTemplateParameterCount(
    ClassSymbol* primaryTemplate) const -> int {
  int count = 0;
  for (auto parameter : ListView{classTemplateParameters(primaryTemplate)}) {
    (void)parameter;
    ++count;
  }
  return count;
}

auto ClassTemplateArgumentDeduction::classTemplateParameters(
    ClassSymbol* primaryTemplate) const -> List<TemplateParameterAST*>* {
  auto templateDeclaration = primaryTemplate->templateDeclaration();
  if (!templateDeclaration) return nullptr;
  return templateDeclaration->templateParameterList;
}

auto ClassTemplateArgumentDeduction::makeGuideTemplateDeclaration(
    ClassSymbol* primaryTemplate,
    TemplateDeclarationAST* ownTemplateDeclaration,
    ParameterDeclarationClauseAST* parameters) -> TemplateDeclarationAST* {
  auto classParameters = classTemplateParameters(primaryTemplate);
  auto classTemplateDeclaration = primaryTemplate->templateDeclaration();

  auto templateDeclaration = TemplateDeclarationAST::create(arena_);
  templateDeclaration->depth = classTemplateDeclaration->depth;

  auto out = &templateDeclaration->templateParameterList;
  for (auto parameter : ListView{classParameters}) {
    *out = make_list_node(arena_, parameter);
    out = &(*out)->next;
  }

  if (ownTemplateDeclaration) {
    for (auto parameter :
         ListView{ownTemplateDeclaration->templateParameterList}) {
      *out = make_list_node(arena_, parameter);
      out = &(*out)->next;
    }

    templateDeclaration->requiresClause =
        ownTemplateDeclaration->requiresClause;
  }

  auto declarator = DeclaratorAST::create(arena_);
  auto chunk = FunctionDeclaratorChunkAST::create(arena_);
  chunk->parameterDeclarationClause = parameters;
  declarator->declaratorChunkList =
      make_list_node<DeclaratorChunkAST>(arena_, chunk);

  auto initDeclarator = InitDeclaratorAST::create(arena_);
  initDeclarator->declarator = declarator;

  auto declaration = SimpleDeclarationAST::create(arena_);
  declaration->initDeclaratorList = make_list_node(arena_, initDeclarator);

  templateDeclaration->declaration = declaration;

  return templateDeclaration;
}

auto ClassTemplateArgumentDeduction::makeInjectedTemplateId(
    ClassSymbol* primaryTemplate) -> SimpleTemplateIdAST* {
  auto templateId = SimpleTemplateIdAST::create(arena_);
  templateId->identifier = name_cast<Identifier>(primaryTemplate->name());
  templateId->symbol = primaryTemplate;

  auto out = &templateId->templateArgumentList;

  for (auto parameter : ListView{classTemplateParameters(primaryTemplate)}) {
    auto symbol = parameter->symbol;
    if (!symbol) return nullptr;

    TemplateArgumentAST* argument = nullptr;

    if (symbol_cast<NonTypeParameterSymbol>(symbol)) {
      auto id = IdExpressionAST::create(arena_);
      id->unqualifiedId =
          NameIdAST::create(arena_, name_cast<Identifier>(symbol->name()));
      id->symbol = symbol;
      id->type = symbol->type();
      id->valueCategory = ValueCategory::kPrValue;

      auto expressionArgument = ExpressionTemplateArgumentAST::create(arena_);
      expressionArgument->expression = id;
      argument = expressionArgument;
    } else {
      auto typeId = TypeIdAST::create(arena_);
      typeId->type = symbol->type();

      auto typeArgument = TypeTemplateArgumentAST::create(arena_);
      typeArgument->typeId = typeId;
      argument = typeArgument;
    }

    *out = make_list_node(arena_, argument);
    out = &(*out)->next;
  }

  return templateId;
}

auto ClassTemplateArgumentDeduction::makeParameterDeclaration(
    const Type* type, SpecifierAST* writtenSpecifier)
    -> ParameterDeclarationAST* {
  auto parameter = ParameterDeclarationAST::create(arena_);
  parameter->type = type;
  if (writtenSpecifier) {
    parameter->typeSpecifierList =
        make_list_node<SpecifierAST>(arena_, writtenSpecifier);
  }
  return parameter;
}

auto ClassTemplateArgumentDeduction::makeParameterClause(
    const std::vector<ParameterDeclarationAST*>& parameters, bool isVariadic)
    -> ParameterDeclarationClauseAST* {
  auto clause = ParameterDeclarationClauseAST::create(arena_);
  clause->isVariadic = isVariadic;

  auto out = &clause->parameterDeclarationList;
  for (auto parameter : parameters) {
    *out = make_list_node(arena_, parameter);
    out = &(*out)->next;
  }

  return clause;
}

void ClassTemplateArgumentDeduction::addConstructorGuide(
    ClassSymbol* primaryTemplate, FunctionSymbol* constructor,
    int constructorIndex, ScopeSymbol* scope) {
  auto constructorType = type_cast<FunctionType>(constructor->type());
  if (!constructorType) return;

  auto parameters = parameterClauseOf(constructor);
  if (!parameters) {
    std::vector<ParameterDeclarationAST*> synthesized;
    for (auto parameterType : constructorType->parameterTypes())
      synthesized.push_back(makeParameterDeclaration(parameterType, nullptr));
    parameters =
        makeParameterClause(synthesized, constructorType->isVariadic());
  }

  auto ownTemplateDeclaration = constructor->templateDeclaration();

  auto templateDeclaration = makeGuideTemplateDeclaration(
      primaryTemplate, ownTemplateDeclaration, parameters);

  auto function = control_->newFunctionSymbol(primaryTemplate->parent(),
                                              constructor->location());
  function->setName(primaryTemplate->name());
  std::vector<const Type*> guideParameterTypes;
  for (auto parameter : ListView{parameters->parameterDeclarationList})
    guideParameterTypes.push_back(parameter->type);

  function->setType(control_->getFunctionType(primaryTemplate->type(),
                                              std::move(guideParameterTypes),
                                              constructorType->isVariadic()));
  function->setTemplateDeclaration(templateDeclaration);

  Guide guide;
  guide.function = function;
  guide.templateDeclaration = templateDeclaration;
  guide.parameters = parameters;
  guide.classParameterCount = classTemplateParameterCount(primaryTemplate);
  guide.constructorIndex = constructorIndex;
  guide.isExplicit = constructor->isExplicit();
  guide.info.fromConstructorTemplate = ownTemplateDeclaration != nullptr;
  guide.info.fromInheritedConstructor =
      constructor->inheritedConstructor() != nullptr;

  guides_.push_back(guide);
}

void ClassTemplateArgumentDeduction::addDefaultConstructorGuide(
    ClassSymbol* primaryTemplate, ScopeSymbol* scope) {
  auto parameters = makeParameterClause({}, /*isVariadic=*/false);

  auto templateDeclaration =
      makeGuideTemplateDeclaration(primaryTemplate, nullptr, parameters);

  auto function = control_->newFunctionSymbol(primaryTemplate->parent(),
                                              primaryTemplate->location());
  function->setName(primaryTemplate->name());
  function->setType(
      control_->getFunctionType(primaryTemplate->type(), {}, false));
  function->setTemplateDeclaration(templateDeclaration);

  Guide guide;
  guide.function = function;
  guide.templateDeclaration = templateDeclaration;
  guide.parameters = parameters;
  guide.classParameterCount = classTemplateParameterCount(primaryTemplate);

  guides_.push_back(guide);
}

void ClassTemplateArgumentDeduction::addCopyDeductionCandidate(
    ClassSymbol* primaryTemplate, ScopeSymbol* scope) {
  auto templateId = makeInjectedTemplateId(primaryTemplate);
  if (!templateId) return;

  auto specifier = NamedTypeSpecifierAST::create(arena_);
  specifier->unqualifiedId = templateId;
  specifier->symbol = primaryTemplate;

  auto parameter = makeParameterDeclaration(primaryTemplate->type(), specifier);

  auto parameters = makeParameterClause({parameter}, /*isVariadic=*/false);

  auto templateDeclaration =
      makeGuideTemplateDeclaration(primaryTemplate, nullptr, parameters);

  auto function = control_->newFunctionSymbol(primaryTemplate->parent(),
                                              primaryTemplate->location());
  function->setName(primaryTemplate->name());
  function->setType(control_->getFunctionType(
      primaryTemplate->type(), {primaryTemplate->type()}, false));
  function->setTemplateDeclaration(templateDeclaration);

  Guide guide;
  guide.function = function;
  guide.templateDeclaration = templateDeclaration;
  guide.parameters = parameters;
  guide.classParameterCount = classTemplateParameterCount(primaryTemplate);
  guide.info.isCopyDeductionCandidate = true;

  guides_.push_back(guide);
}

auto ClassTemplateArgumentDeduction::aggregateElementTypes(
    ClassSymbol* classSymbol, std::size_t argumentCount)
    -> std::vector<const Type*> {
  std::vector<const Type*> elements;

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) return {};
    elements.push_back(baseClass->type());
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    elements.push_back(field->type());
  }

  if (elements.size() < argumentCount) return {};

  elements.resize(argumentCount);

  return elements;
}

void ClassTemplateArgumentDeduction::addAggregateDeductionCandidate(
    ClassSymbol* primaryTemplate, const Initializer& init, ScopeSymbol* scope) {
  if (init.arguments.empty()) return;
  if (!primaryTemplate->deductionGuides().empty()) return;
  if (!TypeTraits{unit_}.is_aggregate(primaryTemplate->type())) return;

  auto elementTypes =
      aggregateElementTypes(primaryTemplate, init.arguments.size());
  if (elementTypes.empty()) return;

  std::vector<ParameterDeclarationAST*> parameterDeclarations;
  for (auto elementType : elementTypes)
    parameterDeclarations.push_back(
        makeParameterDeclaration(elementType, nullptr));

  auto parameters =
      makeParameterClause(parameterDeclarations, /*isVariadic=*/false);

  auto templateDeclaration =
      makeGuideTemplateDeclaration(primaryTemplate, nullptr, parameters);

  auto function = control_->newFunctionSymbol(primaryTemplate->parent(),
                                              primaryTemplate->location());
  function->setName(primaryTemplate->name());
  function->setType(
      control_->getFunctionType(primaryTemplate->type(), elementTypes, false));
  function->setTemplateDeclaration(templateDeclaration);

  Guide guide;
  guide.function = function;
  guide.templateDeclaration = templateDeclaration;
  guide.parameters = parameters;
  guide.isAggregate = true;
  guide.classParameterCount = classTemplateParameterCount(primaryTemplate);

  guides_.push_back(guide);
}

void ClassTemplateArgumentDeduction::addWrittenGuide(
    ClassSymbol* primaryTemplate, DeductionGuideSymbol* guideSymbol) {
  auto declaration = guideSymbol->declaration();
  if (!declaration) return;

  auto guideType = type_cast<FunctionType>(guideSymbol->type());
  if (!guideType) return;

  auto parameters = declaration->parameterDeclarationClause;
  if (!parameters) parameters = makeParameterClause({}, /*isVariadic=*/false);

  auto templateDeclaration = TemplateDeclarationAST::create(arena_);
  if (auto ownTemplateParameters = guideSymbol->templateParameters()) {
    if (auto ownTemplateDeclaration = guideSymbol->templateDeclaration()) {
      templateDeclaration->templateParameterList =
          ownTemplateDeclaration->templateParameterList;
      templateDeclaration->requiresClause =
          ownTemplateDeclaration->requiresClause;
      templateDeclaration->depth = ownTemplateDeclaration->depth;
    }
    (void)ownTemplateParameters;
  }

  auto declarator = DeclaratorAST::create(arena_);
  auto chunk = FunctionDeclaratorChunkAST::create(arena_);
  chunk->parameterDeclarationClause = parameters;
  declarator->declaratorChunkList =
      make_list_node<DeclaratorChunkAST>(arena_, chunk);

  auto initDeclarator = InitDeclaratorAST::create(arena_);
  initDeclarator->declarator = declarator;

  auto simpleDeclaration = SimpleDeclarationAST::create(arena_);
  simpleDeclaration->initDeclaratorList =
      make_list_node(arena_, initDeclarator);
  templateDeclaration->declaration = simpleDeclaration;

  auto function = control_->newFunctionSymbol(primaryTemplate->parent(),
                                              guideSymbol->location());
  function->setName(primaryTemplate->name());
  function->setType(guideType);
  function->setTemplateDeclaration(templateDeclaration);

  Guide guide;
  guide.function = function;
  guide.templateDeclaration = templateDeclaration;
  guide.parameters = parameters;
  guide.returnTemplateId = declaration->templateId;
  guide.isExplicit = guideSymbol->isExplicit();
  guide.info.fromDeductionGuide = true;

  guides_.push_back(guide);
}

void ClassTemplateArgumentDeduction::collectGuides(ClassSymbol* primaryTemplate,
                                                   const Initializer& init,
                                                   ScopeSymbol* scope) {
  auto definition = primaryTemplate->resolvedDefinition();

  bool hasConstructors = false;

  if (definition->isComplete()) {
    int constructorIndex = -1;
    for (auto constructor : definition->constructors()) {
      ++constructorIndex;
      hasConstructors = true;
      addConstructorGuide(primaryTemplate, constructor, constructorIndex,
                          scope);
    }
  }

  if (!hasConstructors) addDefaultConstructorGuide(primaryTemplate, scope);

  addCopyDeductionCandidate(primaryTemplate, scope);

  for (auto guideSymbol : primaryTemplate->deductionGuides())
    addWrittenGuide(primaryTemplate, guideSymbol);

  if (definition->isComplete())
    addAggregateDeductionCandidate(definition, init, scope);
}

auto ClassTemplateArgumentDeduction::guideParameterTypes(
    ClassSymbol* primaryTemplate, const Guide& guide,
    List<TemplateArgumentAST*>* deducedArgs, const Initializer& initializer,
    SourceLocation location, ScopeSymbol* scope)
    -> std::optional<std::vector<const Type*>> {
  if (guide.returnTemplateId) {
    auto substitution =
        Substitution::make(unit_, guide.templateDeclaration, deducedArgs);
    if (!substitution) return std::nullopt;

    return ASTRewriter::substituteParameterTypes(
        unit_, guide.parameters, std::move(*substitution).templateArguments(),
        guide.templateDeclaration->depth, scope);
  }

  auto specialization =
      specializationFor(primaryTemplate, guide, deducedArgs, location, scope);
  if (!specialization) return std::nullopt;

  if (guide.isAggregate) {
    auto elements =
        aggregateElementTypes(specialization, initializer.arguments.size());
    if (elements.empty()) return std::nullopt;
    return elements;
  }

  if (guide.info.isCopyDeductionCandidate)
    return std::vector<const Type*>{specialization->type()};

  if (guide.constructorIndex < 0) return std::vector<const Type*>{};

  auto constructors = specialization->constructors();
  if (guide.constructorIndex >= static_cast<int>(constructors.size()))
    return std::nullopt;

  auto constructor = constructors[guide.constructorIndex];

  if (constructor->templateDeclaration()) {
    List<TemplateArgumentAST*>* ownArgs = nullptr;
    auto out = &ownArgs;
    int skipped = 0;
    for (auto argument : ListView{deducedArgs}) {
      if (skipped++ < guide.classParameterCount) continue;
      *out = make_list_node(arena_, argument);
      out = &(*out)->next;
    }

    auto instantiated = ASTRewriter::instantiateForArgs(
        unit_, ownArgs, constructor, location, /*argsComplete=*/true,
        /*declarationOnly=*/true);
    if (!instantiated) return std::nullopt;
    constructor = instantiated;
  }

  auto constructorType = type_cast<FunctionType>(constructor->type());
  if (!constructorType) return std::nullopt;

  return std::vector<const Type*>(constructorType->parameterTypes().begin(),
                                  constructorType->parameterTypes().end());
}

auto ClassTemplateArgumentDeduction::requiredParameterCount(
    const Guide& guide, std::size_t parameterCount) const -> std::size_t {
  std::size_t required = 0;
  for (auto parameter :
       ListView{guide.parameters ? guide.parameters->parameterDeclarationList
                                 : nullptr}) {
    if (parameter->expression) break;
    if (parameter->isPack) break;
    ++required;
  }
  return std::min(required, parameterCount);
}

auto ClassTemplateArgumentDeduction::argumentList(const Initializer& init)
    -> List<ExpressionAST*>* {
  List<ExpressionAST*>* args = nullptr;
  auto out = &args;
  for (auto argument : init.arguments) {
    *out = make_list_node(arena_, argument);
    out = &(*out)->next;
  }
  return args;
}

auto ClassTemplateArgumentDeduction::specializationFor(
    ClassSymbol* primaryTemplate, const Guide& guide,
    List<TemplateArgumentAST*>* deducedArgs, SourceLocation location,
    ScopeSymbol* scope) -> ClassSymbol* {
  List<TemplateArgumentAST*>* classArgs = nullptr;

  if (guide.returnTemplateId) {
    auto substitution =
        Substitution::make(unit_, guide.templateDeclaration, deducedArgs);
    if (!substitution) return nullptr;

    auto typeId = TypeIdAST::create(arena_);
    auto specifier = NamedTypeSpecifierAST::create(arena_);
    specifier->unqualifiedId = guide.returnTemplateId;
    specifier->symbol = guide.returnTemplateId->symbol;
    typeId->typeSpecifierList = make_list_node<SpecifierAST>(arena_, specifier);

    auto substituted = ASTRewriter::substituteDefaultTypeId(
        unit_, typeId, std::move(*substitution).templateArguments(),
        guide.templateDeclaration->depth, scope);

    if (!substituted || !substituted->type) return nullptr;

    auto deducedClassType = type_cast<ClassType>(substituted->type);
    if (!deducedClassType) return nullptr;

    return deducedClassType->symbol();
  }

  auto out = &classArgs;
  int remaining = guide.classParameterCount;
  for (auto argument : ListView{deducedArgs}) {
    if (remaining-- <= 0) break;
    *out = make_list_node(arena_, argument);
    out = &(*out)->next;
  }

  auto specialization =
      ASTRewriter::instantiate(unit_, classArgs, primaryTemplate, location);

  return symbol_cast<ClassSymbol>(specialization);
}

auto ClassTemplateArgumentDeduction::deduce(ClassSymbol* primaryTemplate,
                                            const Initializer& initializer,
                                            SourceLocation location,
                                            ScopeSymbol* scope)
    -> ClassSymbol* {
  if (!primaryTemplate || !primaryTemplate->templateDeclaration())
    return nullptr;

  guides_.clear();

  collectGuides(primaryTemplate, initializer, scope);
  if (guides_.empty()) return nullptr;

  auto args = argumentList(initializer);

  OverloadResolution overloadResolution{unit_};

  std::vector<Candidate> candidates;
  candidates.reserve(guides_.size());
  std::vector<const Guide*> candidateGuides;
  candidateGuides.reserve(guides_.size());
  bool explicitGuideRejected = false;

  SilentDiagnosticsClient silent;

  for (const auto& guide : guides_) {
    auto guideType = type_cast<FunctionType>(guide.function->type());
    if (!guideType) continue;

    auto saved = unit_->changeDiagnosticsClient(&silent);

    TemplateArgumentDeduction deduction{unit_};
    auto deduced = deduction.deduceForGuide(guide.templateDeclaration,
                                            guideType, guide.parameters, args);

    (void)unit_->changeDiagnosticsClient(saved);

    if (!deduced) continue;

    auto substituted = guideParameterTypes(primaryTemplate, guide, *deduced,
                                           initializer, location, scope);
    if (!substituted) continue;
    auto parameterTypes = std::move(*substituted);

    if (parameterTypes.size() < initializer.arguments.size() &&
        !guideType->isVariadic())
      continue;

    if (initializer.arguments.size() <
        requiredParameterCount(guide, parameterTypes.size()))
      continue;

    if (guide.isExplicit && initializer.isCopyInitialization) {
      explicitGuideRejected = true;
      continue;
    }

    Candidate candidate;
    candidate.symbol = guide.function;
    candidate.viable = true;
    candidate.fromTemplate = true;
    candidate.deducedTemplateArgs = *deduced;
    candidate.deduction = guide.info;

    std::size_t index = 0;
    bool viable = true;

    for (auto argument : initializer.arguments) {
      if (index >= parameterTypes.size()) {
        viable = guideType->isVariadic();
        break;
      }

      auto sequence = overloadResolution.computeImplicitConversionSequence(
          argument, parameterTypes[index]);

      if (sequence.rank == ConversionRank::kNone) {
        viable = false;
        break;
      }

      candidate.conversions.push_back(sequence);
      ++index;
    }

    if (!viable) continue;

    candidates.push_back(std::move(candidate));
    candidateGuides.push_back(&guide);
  }

  if (candidates.empty()) {
    if (explicitGuideRejected) explicitOnly_ = true;
    return nullptr;
  }

  auto result = overloadResolution.selectBestViableFunction(candidates);
  if (!result.best || result.ambiguous) return nullptr;

  auto winnerIndex = static_cast<std::size_t>(result.best - candidates.data());
  if (winnerIndex >= candidateGuides.size()) return nullptr;

  return specializationFor(primaryTemplate, *candidateGuides[winnerIndex],
                           result.best->deducedTemplateArgs, location, scope);
}

}  // namespace cxx
