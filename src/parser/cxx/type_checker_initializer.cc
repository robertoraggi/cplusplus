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

#include <cxx/access_control.h>
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/class_template_deduction.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/initialization.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <cmath>
#include <format>
#include <unordered_set>

namespace cxx {
namespace {

struct TypeDeducer {
  InitContext& ctx;

  template <typename S>
  void deduceArraySize(S* var);

  template <typename S>
  void deduceAutoType(S* symbol);

  template <typename S>
  void deduceClassTemplateArguments(S* var, SpecifierAST* typeSpecifier);

 private:
  template <typename S>
  void deduceArraySizeFromBraced(S* var, const UnboundedArrayType* ty,
                                 BracedInitListAST* braced);
  template <typename S>
  void deduceArraySizeFromExpr(S* var, const UnboundedArrayType* ty,
                               ExpressionAST* initExpr);
};

template <typename S>
void TypeDeducer::deduceArraySize(S* var) {
  auto ty = type_cast<UnboundedArrayType>(var->type());
  if (!ty) return;

  auto initializer = var->initializer();
  if (!initializer) return;

  auto bracedInitList = Initializer{initializer}.bracedInitList();
  if (bracedInitList) {
    deduceArraySizeFromBraced(var, ty, bracedInitList);
    return;
  }

  if (Initializer{initializer}.form() == InitializerForm::kParen) {
    auto elementCount = Initializer{initializer}.arguments().size();
    if (elementCount)
      var->setType(
          ctx.control->getBoundedArrayType(ty->elementType(), elementCount));
    return;
  }

  auto initExpr = Initializer{initializer}.singleExpression();
  if (initExpr) deduceArraySizeFromExpr(var, ty, initExpr);
}

template <typename S>
void TypeDeducer::deduceArraySizeFromBraced(S* var,
                                            const UnboundedArrayType* ty,
                                            BracedInitListAST* braced) {
  if (ctx.traits.is_narrow_char_type(ty->elementType()) &&
      braced->expressionList && !braced->expressionList->next) {
    if (auto strLit = ast_cast<StringLiteralExpressionAST>(
            braced->expressionList->value)) {
      if (auto srcArray = type_cast<BoundedArrayType>(strLit->type)) {
        var->setType(ctx.control->getBoundedArrayType(ty->elementType(),
                                                      srcArray->size()));
        return;
      }
    }
  }

  auto interp = ASTInterpreter{ctx.unit};
  size_t currentIndex = 0;
  size_t maxIndex = 0;
  bool hasElements = false;

  for (auto it = braced->expressionList; it; it = it->next) {
    if (auto desig = ast_cast<DesignatedInitializerClauseAST>(it->value)) {
      if (desig->designatorList) {
        if (auto subscript = ast_cast<SubscriptDesignatorAST>(
                desig->designatorList->value)) {
          if (auto value = interp.evaluate(subscript->expression))
            if (auto idx = interp.toUInt(*value)) currentIndex = *idx;
        }
      }
    }
    if (!hasElements || currentIndex > maxIndex) maxIndex = currentIndex;
    hasElements = true;
    ++currentIndex;
  }

  if (hasElements)
    var->setType(
        ctx.control->getBoundedArrayType(ty->elementType(), maxIndex + 1));
}

template <typename S>
void TypeDeducer::deduceArraySizeFromExpr(S* var, const UnboundedArrayType* ty,
                                          ExpressionAST* initExpr) {
  if (auto bounded = type_cast<BoundedArrayType>(initExpr->type))
    var->setType(
        ctx.control->getBoundedArrayType(ty->elementType(), bounded->size()));
}

static auto deduceAutoReplacement(InitContext& ctx, const Type* P,
                                  const Type* A) -> const Type* {
  if (!A) return nullptr;

  if (type_cast<AutoType>(P)) return A;

  if (auto qual = type_cast<QualType>(P)) {
    auto remaining =
        residual_cv_qualifiers(cv_qualifiers(A), qual->cvQualifiers());
    auto strippedA = ctx.traits.add_cv(unqualified_type(A), remaining);
    return deduceAutoReplacement(ctx, qual->elementType(), strippedA);
  }

  if (auto array = type_cast<BoundedArrayType>(P)) {
    auto aArray = type_cast<BoundedArrayType>(A);
    if (!aArray) return nullptr;
    return deduceAutoReplacement(ctx, array->elementType(),
                                 aArray->elementType());
  }

  if (auto array = type_cast<UnboundedArrayType>(P)) {
    auto aArray = type_cast<UnboundedArrayType>(A);
    if (!aArray) return nullptr;
    return deduceAutoReplacement(ctx, array->elementType(),
                                 aArray->elementType());
  }

  if (auto array = type_cast<UnresolvedBoundedArrayType>(P)) {
    auto aArray = type_cast<BoundedArrayType>(A);
    if (!aArray) return nullptr;
    return deduceAutoReplacement(ctx, array->elementType(),
                                 aArray->elementType());
  }

  if (auto ptr = type_cast<PointerType>(P)) {
    auto aPtr = type_cast<PointerType>(A);
    if (!aPtr) return nullptr;
    return deduceAutoReplacement(ctx, ptr->elementType(), aPtr->elementType());
  }

  if (auto ref = type_cast<LvalueReferenceType>(P)) {
    return deduceAutoReplacement(ctx, ref->elementType(),
                                 ctx.traits.remove_reference(A));
  }

  if (auto ref = type_cast<RvalueReferenceType>(P)) {
    if (type_cast<LvalueReferenceType>(A)) {
      return deduceAutoReplacement(ctx, ref->elementType(), A);
    }
    return deduceAutoReplacement(ctx, ref->elementType(),
                                 ctx.traits.remove_reference(A));
  }

  if (auto function = type_cast<FunctionType>(P)) {
    auto aFunction = type_cast<FunctionType>(A);
    if (!aFunction) return nullptr;

    if (containsPlaceholderType(function->returnType())) {
      return deduceAutoReplacement(ctx, function->returnType(),
                                   aFunction->returnType());
    }

    if (function->parameterTypes().size() != aFunction->parameterTypes().size())
      return nullptr;

    for (std::size_t i = 0; i < function->parameterTypes().size(); ++i) {
      auto parameterType = function->parameterTypes()[i];
      if (!containsPlaceholderType(parameterType)) continue;
      return deduceAutoReplacement(ctx, parameterType,
                                   aFunction->parameterTypes()[i]);
    }
    return nullptr;
  }

  if (auto pointer = type_cast<MemberObjectPointerType>(P)) {
    auto aPointer = type_cast<MemberObjectPointerType>(A);
    if (!aPointer) return nullptr;
    return deduceAutoReplacement(ctx, pointer->elementType(),
                                 aPointer->elementType());
  }

  if (auto pointer = type_cast<MemberFunctionPointerType>(P)) {
    auto aPointer = type_cast<MemberFunctionPointerType>(A);
    if (!aPointer) return nullptr;
    return deduceAutoReplacement(ctx, pointer->functionType(),
                                 aPointer->functionType());
  }

  return nullptr;
}

template <typename S>
void TypeDeducer::deduceClassTemplateArguments(S* var,
                                               SpecifierAST* typeSpecifier) {
  if (!ClassTemplateArgumentDeduction::placeholderClassTemplate(
          typeSpecifier, ctx.checker.scope()))
    return;

  auto initializer = var->initializer();

  const auto initializationKind = Initializer{initializer}.initializationKind();

  auto deduced = ctx.checker.deduceClassTemplateSpecialization(
      typeSpecifier, Initializer{initializer}.arguments(),
      Initializer{initializer}.bracedInitList() != nullptr,
      initializationKind == InitializationKind::kCopyInitialization ||
          initializationKind == InitializationKind::kCopyListInitialization,
      var->location());

  if (!deduced) return;

  var->setType(ctx.traits.add_cv(deduced, cv_qualifiers(var->type())));
}

template <typename S>
void TypeDeducer::deduceAutoType(S* var) {
  auto declType = var->type();
  if (!containsPlaceholderType(declType)) return;

  if (!var->initializer()) {
    ctx.error(var->location(), "variable with 'auto' type must be initialized");
    return;
  }

  auto deducedExpr = Initializer{var->initializer()}.singleExpression();

  const bool inTemplate = isEnclosedInDependentTemplate(
      ctx.unit, ctx.checker.scope(), /*stopAtConcreteSpecialization=*/true);
  if (inTemplate && (!deducedExpr || !deducedExpr->type ||
                     isDependent(ctx.unit, deducedExpr) ||
                     isDependent(ctx.unit, deducedExpr->type))) {
    auto dependentType = ctx.control->getDependentType();
    var->setType(ctx.traits.replace_placeholder_types(declType, dependentType));
    return;
  }

  if (!deducedExpr || !deducedExpr->type) return;

  auto deduced = ctx.checker.deducePlaceholderType(declType, deducedExpr);
  if (!deduced) return;
  var->setType(deduced);

  if (auto classType =
          type_cast<ClassType>(ctx.traits.remove_cvref(var->type()))) {
    ctx.traits.requireCompleteClass(classType->symbol());
  }
}

struct ConstexprEvaluator {
  InitContext& ctx;

  template <typename S>
  auto tryEvaluateConstructor(S* var, ASTInterpreter& interp)
      -> std::optional<ConstValue>;
};

template <typename S>
auto ConstexprEvaluator::tryEvaluateConstructor(S* var, ASTInterpreter& interp)
    -> std::optional<ConstValue> {
  auto classType = unqualified_cast<ClassType>(var->type());
  if (!classType) return std::nullopt;

  auto classSym = classType->symbol();
  if (!classSym) return std::nullopt;

  auto initArgs = Initializer{var->initializer()}.arguments();

  if (initArgs.size() == 1) {
    if (auto typeConstruction = ast_cast<TypeConstructionAST>(initArgs[0])) {
      if (typeConstruction->type == classType ||
          typeConstruction->type == var->type()) {
        initArgs.clear();
        for (auto it = typeConstruction->expressionList; it; it = it->next)
          initArgs.push_back(it->value);
      }
    }
  }

  std::vector<ConstValue> args;
  for (auto argExpr : initArgs) {
    auto argVal = interp.evaluate(argExpr);
    if (!argVal) return std::nullopt;
    args.push_back(std::move(*argVal));
  }

  auto constructor = var->constructor();
  if (!constructor) constructor = classSym->defaultConstructor();
  if (!constructor) return std::nullopt;
  if (!constructor->isConstexpr()) return std::nullopt;
  auto value =
      interp.evaluateConstructor(constructor, classType, std::move(args));
  if (!value || !interp.isFullyInitialized(*value)) return std::nullopt;
  return value;
}

struct InitDeclaratorChecker {
  InitContext ctx;
  TypeDeducer typeDeducer;
  ConstexprEvaluator constexprEval;

  explicit InitDeclaratorChecker(TypeChecker& checker)
      : ctx(checker), typeDeducer{ctx}, constexprEval{ctx} {}

  void checkInitDeclarator(InitDeclaratorAST* ast, SpecifierAST* typeSpecifier);
  template <typename S>
  void checkVariable(S* var, ExpressionAST*& initializer,
                     SourceLocation location,
                     SpecifierAST* typeSpecifier = nullptr);
  void checkFieldInitializer(FieldSymbol* field);
  void evaluateFieldConstValue(FieldSymbol* field);

 private:
  template <typename S>
  void checkInitialization(S* var, ExpressionAST*& initializer,
                           SourceLocation location);
  void evaluateConstValue(VariableSymbol* var, ExpressionAST*& initializer);
};

void InitDeclaratorChecker::checkInitDeclarator(InitDeclaratorAST* ast,
                                                SpecifierAST* typeSpecifier) {
  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && field->isStatic()) {
    checkVariable(field, ast->initializer, field->location(), typeSpecifier);
    return;
  }

  auto var = symbol_cast<VariableSymbol>(ast->symbol);
  if (!var) return;

  checkVariable(var, ast->initializer,
                ctx.checker.getInitDeclaratorLocation(ast, var), typeSpecifier);
}

template <typename S>
void InitDeclaratorChecker::checkVariable(S* var, ExpressionAST*& initializer,
                                          SourceLocation location,
                                          SpecifierAST* typeSpecifier) {
  if (initializer) var->setInitializer(initializer);

  typeDeducer.deduceArraySize(var);
  typeDeducer.deduceClassTemplateArguments(var, typeSpecifier);
  typeDeducer.deduceAutoType(var);

  if (var->isConstexpr()) var->setType(ctx.traits.add_const(var->type()));

  if (var->isExtern() && !initializer) return;

  auto objectType =
      ctx.traits.remove_cv(ctx.traits.remove_all_extents(var->type()));
  if (auto classType = type_cast<ClassType>(objectType)) {
    auto classSymbol = classType->symbol()->resolvedDefinition();
    auto destructor = classSymbol->destructor();
    ASTRewriter::requireFunctionDefinition(ctx.unit, destructor);
    if (destructor && destructor->isDeleted()) {
      ctx.checker.error(location, "attempt to use a deleted destructor");
    } else if (destructor) {
      AccessContext accessContext{ctx.unit, ctx.checker.scope()};
      if (!accessContext.isAccessible(destructor, classSymbol, nullptr)) {
        auto accessKind = std::string_view{"private"};
        if (destructor->accessSpecifier() == AccessSpecifier::kProtected)
          accessKind = "protected";
        ctx.checker.error(
            location, std::format("calling a {} destructor of class '{}'",
                                  accessKind, to_string(classSymbol->type())));
      }
    }
  }

  checkInitialization(var, initializer, location);

  if constexpr (std::is_same_v<S, VariableSymbol>)
    evaluateConstValue(var, initializer);
}

template <typename S>
void InitDeclaratorChecker::checkInitialization(S* var,
                                                ExpressionAST*& initializer,
                                                SourceLocation location) {
  auto entity = InitializedEntity::variable(var->type(), var, location);
  Initializer init{initializer};

  auto sequence = computeInitializationSequence(
      ctx, entity, init.initializationKind(), init);

  if (!sequence) {
    diagnoseInitializationFailure(ctx, sequence, entity, init);
    return;
  }

  auto result = applyInitializationSequence(ctx, sequence, entity, init);

  if (sequence.constructor) var->setConstructor(sequence.constructor);

  if (result) {
    initializer = result;
    var->setInitializer(result);
  }
}

void InitDeclaratorChecker::checkFieldInitializer(FieldSymbol* field) {
  auto targetType = ctx.traits.remove_cv(field->type());
  if (ctx.isTargetTypeUnresolved(targetType)) return;

  if (ctx.traits.is_class(targetType)) {
    auto classType = type_cast<ClassType>(targetType);
    if (!classType || !classType->symbol()) return;
    if (!classType->symbol()->resolvedDefinition()->isComplete()) return;

    auto entity =
        InitializedEntity::member(field->type(), field, field->location());
    Initializer init{field->initializer()};

    auto sequence = computeInitializationSequence(
        ctx, entity, init.initializationKind(), init);

    auto result = applyInitializationSequence(ctx, sequence, entity, init);

    if (sequence.constructor) field->setConstructor(sequence.constructor);
    if (result != field->initializer()) field->setInitializer(result);
    evaluateFieldConstValue(field);
    return;
  }

  auto equal = ast_cast<EqualInitializerAST>(field->initializer());
  auto init = equal ? equal->expression : field->initializer();

  if (auto braced = ast_cast<BracedInitListAST>(init)) {
    if (!braced->type)
      ctx.checker.check_braced_init_list(
          field->type(), braced,
          Initializer{field->initializer()}.initializationKind());
    return;
  }

  if (!equal || !equal->expression) return;
  if (ctx.traits.is_reference(field->type())) return;
  if (isDependent(ctx.unit, equal->expression)) return;

  (void)ctx.checker.implicit_conversion(equal->expression, field->type());
}

void InitDeclaratorChecker::evaluateFieldConstValue(FieldSymbol* field) {
  if (!field->isStatic()) return;
  if (!field->isConstexpr() && !field->isConstinit()) return;
  if (isDependent(ctx.unit, field->type())) return;

  auto interp = ASTInterpreter{ctx.unit};

  std::optional<ConstValue> value;
  if (field->initializer()) value = interp.evaluate(field->initializer());

  if (!value.has_value() || field->constructor()) {
    if (auto ctorValue = constexprEval.tryEvaluateConstructor(field, interp))
      value = std::move(ctorValue);
  }

  field->setConstValue(std::move(value));
}

void InitDeclaratorChecker::evaluateConstValue(VariableSymbol* var,
                                               ExpressionAST*& initializer) {
  auto dependent = var->templateParameters() != nullptr;
  if (!dependent) dependent = isDependent(ctx.unit, var->type());
  if (!dependent && var->initializer())
    dependent = isDependent(ctx.unit, var->initializer());

  if (var->initializer()) {
    auto interp = ASTInterpreter{ctx.unit};
    auto value = interp.evaluate(var->initializer());

    const auto needsConstructor = !value.has_value() || var->constructor();

    if (needsConstructor && var->isConstexpr()) {
      if (auto ctorValue = constexprEval.tryEvaluateConstructor(var, interp))
        value = std::move(ctorValue);
    }

    var->setConstValue(value);
  } else if (var->isConstexpr() && var->constructor()) {
    auto interp = ASTInterpreter{ctx.unit};
    var->setConstValue(constexprEval.tryEvaluateConstructor(var, interp));
  }

  if (var->isConstexpr() && var->constValue().has_value() && !dependent) {
    auto target = constantExpressionTarget(initializer);
    auto needsConstantNode = target != nullptr;
    if (ctx.traits.is_array(var->type())) needsConstantNode = false;
    if (needsConstantNode) needsConstantNode = *target != nullptr;
    if (needsConstantNode)
      needsConstantNode = !ast_cast<ConstExpressionAST>(*target);

    std::optional<ConstValue> constantValue;
    if (needsConstantNode) {
      constantValue = *var->constValue();
      if (is_glvalue(*target)) {
        auto interp = ASTInterpreter{ctx.unit};
        auto address = interp.evaluateAddress(*target);
        if (address.has_value()) {
          constantValue = std::move(address);
          if (ctx.traits.is_reference(var->type())) {
            var->setConstValue(constantValue);
          }
        } else {
          needsConstantNode = false;
          if (ctx.traits.is_reference(var->type())) {
            var->setConstValue(std::nullopt);
          }
        }
      }
    }

    if (needsConstantNode) {
      auto constant = ConstExpressionAST::create(ctx.unit->arena());
      constant->expression = *target;
      constant->constValue =
          ctx.unit->arena()->make<ConstValue>(std::move(*constantValue));
      constant->type = (*target)->type;
      constant->valueCategory = (*target)->valueCategory;
      *target = constant;
      var->setInitializer(initializer);
    }
  }

  if (var->isConstexpr() && !var->constValue().has_value()) {
    if (!dependent) {
      ctx.error(var->location(), "constexpr variable must be initialized");
    }
  }
}
}  // namespace

void TypeChecker::check_init_declarator(InitDeclaratorAST* ast,
                                        SpecifierAST* typeSpecifier) {
  InitDeclaratorChecker{*this}.checkInitDeclarator(ast, typeSpecifier);
}

void TypeChecker::check_condition_declaration(ConditionExpressionAST* ast) {
  auto var = ast->symbol;
  if (!var) return;
  InitDeclaratorChecker{*this}.checkVariable(var, ast->initializer,
                                             var->location());

  ast->type = var->type();
  ast->valueCategory = ValueCategory::kLValue;
}

void TypeChecker::check_field_initializer(FieldSymbol* field) {
  if (!field || !field->initializer()) return;
  InitDeclaratorChecker{*this}.checkFieldInitializer(field);
}

auto TypeChecker::deducePlaceholderType(const Type* declaredType,
                                        ExpressionAST* initializer)
    -> const Type* {
  if (!initializer) return nullptr;
  if (type_cast<DecltypeAutoType>(declaredType))
    return unit_->typeTraits().decltype_of(initializer);
  auto initializerType = initializer->type;
  if (type_cast<RvalueReferenceType>(declaredType) &&
      initializer->valueCategory == ValueCategory::kLValue) {
    initializerType = unit_->typeTraits().add_lvalue_reference(initializerType);
  }
  return deduceAutoType(declaredType, initializerType);
}

auto TypeChecker::deduceAutoType(const Type* declaredType,
                                 const Type* initializerType) -> const Type* {
  if (!initializerType) return nullptr;

  InitContext ctx{*this};

  if (type_cast<AutoType>(declaredType))
    return ctx.traits.decay(initializerType);

  auto replacement = deduceAutoReplacement(ctx, declaredType, initializerType);
  if (!replacement) return nullptr;
  return ctx.traits.replace_placeholder_types(declaredType, replacement);
}

auto TypeChecker::enterAggregateInitialization(ClassSymbol* classSymbol)
    -> bool {
  return aggregatesBeingInitialized_.insert(classSymbol).second;
}

void TypeChecker::leaveAggregateInitialization(ClassSymbol* classSymbol) {
  aggregatesBeingInitialized_.erase(classSymbol);
}

void TypeChecker::checkConstructorAccess(FunctionSymbol* constructor,
                                         SourceLocation location) {
  if (!constructor) return;

  auto declaringClass = declaringClassOf(constructor);
  if (!declaringClass) return;

  AccessContext accessContext{unit_, scope_};
  if (accessContext.isAccessible(constructor, declaringClass, nullptr)) return;

  auto accessKind = std::string_view{"private"};
  if (constructor->accessSpecifier() == AccessSpecifier::kProtected)
    accessKind = "protected";

  error(location, std::format("calling a {} constructor of class '{}'",
                              accessKind, to_string(declaringClass->type())));
}

auto TypeChecker::deduceClassTemplateSpecialization(
    SpecifierAST* typeSpecifier, const std::vector<ExpressionAST*>& arguments,
    bool isListInitialization, bool isCopyInitialization,
    SourceLocation location) -> const Type* {
  auto primaryTemplate =
      ClassTemplateArgumentDeduction::placeholderClassTemplate(typeSpecifier,
                                                               scope_);
  if (!primaryTemplate) return nullptr;

  for (auto argument : arguments) {
    if (!argument || !argument->type) return nullptr;
    if (isDependent(unit_, argument->type)) return nullptr;
  }

  ClassTemplateArgumentDeduction::Initializer initializer{
      .arguments = arguments,
      .isListInitialization = isListInitialization,
      .isCopyInitialization = isCopyInitialization};

  ClassTemplateArgumentDeduction deduction{unit_};
  auto deduced =
      deduction.deduce(primaryTemplate, initializer, location, scope_);

  if (!deduced) {
    if (deduction.selectedExplicitOnly()) {
      error(
          location,
          std::format("class template argument deduction for '{}' selected an "
                      "explicit deduction guide for copy-list-initialization",
                      to_string(primaryTemplate->name())));
    } else {
      error(location,
            std::format("no viable constructor or deduction guide for "
                        "deduction of template arguments of '{}'",
                        to_string(primaryTemplate->name())));
    }
    return nullptr;
  }

  return deduced->type();
}
}  // namespace cxx
