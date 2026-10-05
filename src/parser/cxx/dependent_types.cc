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
#include <cxx/ast_visitor.h>
#include <cxx/dependent_types.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <vector>

namespace cxx {
namespace {
[[nodiscard]] auto enclosingScopeForDependence(FunctionSymbol* function)
    -> ScopeSymbol* {
  auto scope = function->parent();
  if (!function->isFriend()) return scope;
  while (scope && scope->isClass()) scope = scope->parent();
  return scope;
}

[[nodiscard]] auto namesParameterPack(Symbol* symbol) -> bool {
  if (auto parameter = symbol_cast<ParameterSymbol>(symbol))
    return parameter->isParameterPack();
  return is_template_parameter_pack(symbol);
}

struct FindUnexpandedParameterPack final : ASTVisitor {
  AST* root = nullptr;
  bool found = false;

  explicit FindUnexpandedParameterPack(AST* root) : root(root) {}

  auto preVisit(AST*) -> bool override { return !found; }

  void check(Symbol* symbol) {
    if (namesParameterPack(symbol)) found = true;
  }

  void visit(IdExpressionAST* ast) override {
    check(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(NamedTypeSpecifierAST* ast) override {
    check(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(SimpleNestedNameSpecifierAST* ast) override {
    check(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(TypeIdAST* ast) override {
    if (ast != root && isPackExpansion(ast)) return;
    ASTVisitor::visit(ast);
  }

  void visit(PackExpansionExpressionAST*) override {}
  void visit(SizeofPackExpressionAST*) override {}
  void visit(FoldExpressionAST*) override {}
  void visit(LeftFoldExpressionAST*) override {}
  void visit(RightFoldExpressionAST*) override {}
};

[[nodiscard]] auto templateName(UnqualifiedIdAST* id) -> Symbol* {
  if (auto templateId = ast_cast<SimpleTemplateIdAST>(id))
    return templateId->symbol;
  return nullptr;
}

struct NamesTemplateParameter {
  [[nodiscard]] auto operator()(TypeParameterSymbol*) const -> bool {
    return true;
  }
  [[nodiscard]] auto operator()(TemplateTypeParameterSymbol*) const -> bool {
    return true;
  }
  [[nodiscard]] auto operator()(NonTypeParameterSymbol*) const -> bool {
    return true;
  }
  [[nodiscard]] auto operator()(ConstraintTypeParameterSymbol*) const -> bool {
    return true;
  }
  [[nodiscard]] auto operator()(Symbol*) const -> bool { return false; }
};

[[nodiscard]] auto templateParameterDepth(Symbol* symbol)
    -> std::optional<int> {
  if (!symbol || !visit(NamesTemplateParameter{}, symbol)) return std::nullopt;
  auto info = template_parameter_info(symbol);
  if (!info) return std::nullopt;
  return info->depth;
}

[[nodiscard]] auto ownTemplateParameterDepth(FunctionSymbol* function)
    -> std::optional<int> {
  auto parameters = function->templateParameters();
  if (!parameters || parameters->isExplicitTemplateSpecialization())
    return std::nullopt;
  if (parameters->members().empty()) return std::nullopt;
  return templateParameterDepth(parameters->members().front());
}

struct IsDependent {
  TranslationUnit* unit = nullptr;

  struct NonDependentType {
    const Type* type;
    int localTemplateDepth;
  };

  struct NonDependentInitializer {
    ExpressionAST* expression = nullptr;
    std::optional<int> localTemplateDepth;
  };

  NonDependentInitializer firstNonDependentInitializer;
  std::vector<NonDependentInitializer> nonDependentInitializers;

  struct TypeExamination {
    const Type* type;
    TypeExamination* previous;
  };

  TypeExamination* typesUnderExamination = nullptr;
  std::array<NonDependentType, 8> nonDependentTypes;
  std::size_t nonDependentTypeCount = 0;
  std::vector<NonDependentType> dynamicNonDependentTypes;
  std::size_t cycles = 0;
  std::size_t unsettledClasses = 0;
  std::optional<int> localTemplateDepth;

  [[nodiscard]] auto isLocalTemplateDepth(int depth) const -> bool {
    return localTemplateDepth && depth >= *localTemplateDepth;
  }

  [[nodiscard]] auto namesDependentTemplateParameter(Symbol* symbol) const
      -> bool {
    if (!symbol || !visit(NamesTemplateParameter{}, symbol)) return false;
    auto depth = templateParameterDepth(symbol);
    return !depth || !isLocalTemplateDepth(*depth);
  }

  template <typename Operand>
  [[nodiscard]] auto isDependentOperand(Operand operand) -> bool {
    if (!localTemplateDepth) return true;
    return isDependent(operand);
  }

  [[nodiscard]] auto hasDependentImplicitObjectParameter(
      FunctionSymbol* function) -> bool {
    if (!function->isImplicitObjectMemberFunction()) return false;
    return isDependent(function->parent()->type());
  }

  [[nodiscard]] auto isDeclaredWithDependentType(FunctionSymbol* function)
      -> bool {
    if (hasDependentImplicitObjectParameter(function)) return true;
    auto ownDepth = ownTemplateParameterDepth(function);
    if (!ownDepth) return isDependent(function->type());
    if (!enclosedInDependentTemplate(enclosingScopeForDependence(function),
                                     /*stopAtConcreteSpecialization=*/true))
      return false;
    const auto enclosingLocalDepth = localTemplateDepth;
    localTemplateDepth = enclosingLocalDepth
                             ? std::min(*enclosingLocalDepth, *ownDepth)
                             : *ownDepth;
    const auto dependent = isDependent(function->type());
    localTemplateDepth = enclosingLocalDepth;
    return dependent;
  }

  [[nodiscard]] auto hasMemberDeclaredWithDependentType(
      OverloadSetSymbol* overloadSet) -> bool {
    std::vector<FunctionSymbol*> functions;
    addLookupCandidates(functions, overloadSet);
    for (auto function : functions) {
      if (isDeclaredWithDependentType(function)) return true;
    }
    return false;
  }

  struct ReferencedMemberIsDependent {
    IsDependent& self;

    [[nodiscard]] auto operator()(FunctionSymbol* function) -> bool {
      return self.isDeclaredWithDependentType(function);
    }
    [[nodiscard]] auto operator()(OverloadSetSymbol* overloadSet) -> bool {
      return self.hasMemberDeclaredWithDependentType(overloadSet);
    }
    [[nodiscard]] auto operator()(FieldSymbol* field) -> bool {
      return self.isDependent(field->type());
    }
    [[nodiscard]] auto operator()(UsingDeclarationSymbol* symbol) -> bool {
      auto declaration = symbol->declarator();
      return declaration && self.isDependent(declaration->nestedNameSpecifier);
    }
    [[nodiscard]] auto operator()(Symbol*) -> bool { return false; }
  };

  [[nodiscard]] auto isDependent(ExpressionAST* ast) -> bool {
    if (!ast) return false;
    return visit(*this, ast);
  }

  [[nodiscard]] auto isDependent(TypeIdAST* ast) -> bool {
    if (!ast) return false;
    if (bindsAliasTemplate(ast)) return false;
    if (isDependent(ast->type)) return true;
    for (auto typeSpecifier : ListView{ast->typeSpecifierList}) {
      if (isDependent(typeSpecifier)) return true;
    }
    return false;
  }

  [[nodiscard]] static auto bindsAliasTemplate(TypeIdAST* ast) -> bool {
    if (!ast) return false;
    for (auto spec : ListView{ast->typeSpecifierList}) {
      auto named = ast_cast<NamedTypeSpecifierAST>(spec);
      if (!named) continue;
      if (!ast_cast<NameIdAST>(named->unqualifiedId)) return false;
      auto alias = symbol_cast<TypeAliasSymbol>(named->symbol);
      return alias && alias->templateParameters();
    }
    return false;
  }

  [[nodiscard]] auto enclosedInDependentTemplate(
      ScopeSymbol* scope, bool stopAtConcreteSpecialization) -> bool {
    for (; scope; scope = scope->parent()) {
      if (auto tparams = symbol_cast<TemplateParametersSymbol>(scope)) {
        if (!tparams->isExplicitTemplateSpecialization()) return true;
      } else if (auto cls = symbol_cast<ClassSymbol>(scope)) {
        if (stopAtConcreteSpecialization && cls->isSpecialization() &&
            !isDependent(cls->type()))
          return false;
        if (auto tp = cls->templateParameters();
            tp && !tp->isExplicitTemplateSpecialization())
          return true;
      } else if (auto func = symbol_cast<FunctionSymbol>(scope)) {
        if (stopAtConcreteSpecialization && func->isSpecialization()) {
          const auto hasConcreteType = !isDependent(func->type());
          const auto hasDependentParent = enclosedInDependentTemplate(
              enclosingScopeForDependence(func), false);
          if (hasConcreteType && !hasDependentParent) return false;
        }
        if (auto tp = func->templateParameters();
            tp && !tp->isExplicitTemplateSpecialization())
          return true;
      } else if (auto lambda = symbol_cast<LambdaSymbol>(scope)) {
        if (lambda->isTemplate() || lambda->isInTemplate()) return true;
      }
    }
    return false;
  }

  [[nodiscard]] auto isDependentTypeArgument(TypeTemplateArgumentAST* typeArg)
      -> bool {
    if (!typeArg || !typeArg->typeId) return false;
    if (isDependent(typeArg->typeId)) return true;

    for (auto spec : ListView{typeArg->typeId->typeSpecifierList}) {
      auto named = ast_cast<NamedTypeSpecifierAST>(spec);
      if (!named) continue;

      if (namesDependentTemplateParameter(named->symbol)) return true;
      if (auto alias = symbol_cast<TypeAliasSymbol>(named->symbol)) {
        const auto bindsAliasTemplate =
            alias->templateParameters() &&
            ast_cast<NameIdAST>(named->unqualifiedId) != nullptr;
        if (!bindsAliasTemplate) {
          if (!alias->type()) return true;
          if (isDependent(alias->type())) return true;
        }
      }
      if (isDependent(named->nestedNameSpecifier)) {
        return true;
      }

      if (auto innerTemplateId =
              ast_cast<SimpleTemplateIdAST>(named->unqualifiedId)) {
        if (hasDependentTemplateArguments(innerTemplateId)) return true;
      }
    }

    return false;
  }

  [[nodiscard]] auto isDependentTemplateNameArgument(
      TypeTemplateArgumentAST* typeArg) -> bool {
    if (!typeArg || !typeArg->typeId) return true;

    for (auto spec : ListView{typeArg->typeId->typeSpecifierList}) {
      auto named = ast_cast<NamedTypeSpecifierAST>(spec);
      if (!named) continue;
      if (isDependent(named->nestedNameSpecifier)) return true;
      return namesDependentTemplateParameter(named->symbol);
    }

    return true;
  }

  [[nodiscard]] auto isDependentTemplateArgument(TemplateArgumentAST* arg,
                                                 Symbol* parameter = nullptr)
      -> bool {
    if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
      if (symbol_cast<TemplateTypeParameterSymbol>(parameter))
        return isDependentTemplateNameArgument(typeArg);
      if (denotesTemplateName(typeArg))
        return isDependentTemplateNameArgument(typeArg);
      return isDependentTypeArgument(typeArg);
    }
    if (auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg))
      return isDependent(exprArg->expression);
    return false;
  }

  [[nodiscard]] static auto parameterForArgument(
      TemplateParametersSymbol* parameters, std::size_t index) -> Symbol* {
    if (!parameters) return nullptr;

    const auto& members = parameters->members();
    if (members.empty()) return nullptr;
    if (index < members.size()) return members[index];

    auto trailing = members.back();
    if (!is_template_parameter_pack(trailing)) return nullptr;
    return trailing;
  }

  [[nodiscard]] auto hasDependentTemplateArguments(UnqualifiedIdAST* id)
      -> bool {
    auto parameters =
        template_parameters_of(templated_symbol(templateName(id)));

    std::size_t index = 0;
    for (auto arg : ListView{get_template_arguments(id)}) {
      auto parameter = parameterForArgument(parameters, index);
      ++index;
      if (isDependentTemplateArgument(arg, parameter)) return true;
    }
    return false;
  }

  [[nodiscard]] auto isInTemplateScope(Symbol* symbol) -> bool {
    if (auto var = symbol_cast<VariableSymbol>(symbol))
      if (var->templateParameters()) return true;
    return enclosedInDependentTemplate(symbol->parent(),
                                       /*stopAtConcreteSpecialization=*/true);
  }

  [[nodiscard]] auto isPotentiallyConstant(Symbol* symbol, bool isConstexpr)
      -> bool {
    if (isConstexpr) return true;
    if (!symbol || !symbol->type() || !unit) return false;

    auto traits = unit->typeTraits();
    if (traits.is_reference(symbol->type())) return true;

    auto cv = cv_qualifiers(symbol->type());
    if (cv != CvQualifiers::kConst) return false;

    auto type = traits.remove_cv(symbol->type());
    return traits.is_integral(type) || traits.is_enum(type);
  }

  [[nodiscard]] auto hasValueDependentInitializer(Symbol* symbol,
                                                  ExpressionAST* initializer,
                                                  bool isConstexpr) -> bool {
    if (!initializer || !isPotentiallyConstant(symbol, isConstexpr))
      return false;
    return isDependentInitializer(initializer);
  }

  [[nodiscard]] auto isDependentInitializer(ExpressionAST* expression) -> bool {
    if (!expression) return false;
    if (firstNonDependentInitializer.expression == expression &&
        firstNonDependentInitializer.localTemplateDepth == localTemplateDepth)
      return false;
    for (const auto& entry : nonDependentInitializers) {
      if (entry.expression == expression &&
          entry.localTemplateDepth == localTemplateDepth)
        return false;
    }
    const auto cyclesBefore = cycles;
    const auto dependent = isDependent(expression);
    if (dependent || cycles != cyclesBefore) return dependent;
    if (!firstNonDependentInitializer.expression) {
      firstNonDependentInitializer = {expression, localTemplateDepth};
    } else {
      nonDependentInitializers.push_back({expression, localTemplateDepth});
    }
    return false;
  }

  [[nodiscard]] auto isDependent(const Type* type) -> bool {
    if (!type) return false;
    if (unit && unit->isNonDependentType(type)) {
      if (unit->timeTrace())
        unit->timeTrace()->count(TimeTrace::kTypeDependenceCacheHits);
      return false;
    }
    for (auto entry = typesUnderExamination; entry; entry = entry->previous) {
      if (entry->type != type) continue;
      ++cycles;
      return false;
    }
    const auto depth = localTemplateDepth.value_or(-1);
    auto examined = std::span{nonDependentTypes}.first(nonDependentTypeCount);
    if (!dynamicNonDependentTypes.empty()) examined = dynamicNonDependentTypes;
    for (const auto& entry : examined) {
      if (entry.type != type || entry.localTemplateDepth != depth) continue;
      if (unit && unit->timeTrace())
        unit->timeTrace()->count(TimeTrace::kTypeDependenceCacheHits);
      return false;
    }
    const auto cyclesBefore = cycles;
    const auto unsettledClassesBefore = unsettledClasses;
    TypeExamination examination{type, typesUnderExamination};
    typesUnderExamination = &examination;
    if (unit && unit->timeTrace())
      unit->timeTrace()->count(TimeTrace::kTypeDependenceVisits);
    const auto dependent = visit(*this, type);
    typesUnderExamination = examination.previous;
    if (dependent || cycles != cyclesBefore) return dependent;
    if (unit && !localTemplateDepth &&
        unsettledClasses == unsettledClassesBefore)
      unit->addNonDependentType(type);
    if (nonDependentTypeCount < nonDependentTypes.size()) {
      nonDependentTypes[nonDependentTypeCount++] = {type, depth};
      return dependent;
    }
    if (dynamicNonDependentTypes.empty())
      dynamicNonDependentTypes.assign(nonDependentTypes.begin(),
                                      nonDependentTypes.end());
    dynamicNonDependentTypes.push_back({type, depth});
    return dependent;
  }

  auto operator()(const VoidType* type) -> bool { return false; }
  auto operator()(const NullptrType* type) -> bool { return false; }
  auto operator()(const DecltypeAutoType* type) -> bool { return false; }
  auto operator()(const AutoType* type) -> bool { return false; }
  auto operator()(const BoolType* type) -> bool { return false; }
  auto operator()(const SignedCharType* type) -> bool { return false; }
  auto operator()(const ShortIntType* type) -> bool { return false; }
  auto operator()(const IntType* type) -> bool { return false; }
  auto operator()(const LongIntType* type) -> bool { return false; }
  auto operator()(const LongLongIntType* type) -> bool { return false; }
  auto operator()(const Int128Type* type) -> bool { return false; }
  auto operator()(const UnsignedCharType* type) -> bool { return false; }
  auto operator()(const UnsignedShortIntType* type) -> bool { return false; }
  auto operator()(const UnsignedIntType* type) -> bool { return false; }
  auto operator()(const UnsignedLongIntType* type) -> bool { return false; }
  auto operator()(const UnsignedLongLongIntType* type) -> bool { return false; }
  auto operator()(const UnsignedInt128Type* type) -> bool { return false; }
  auto operator()(const CharType* type) -> bool { return false; }
  auto operator()(const Char8Type* type) -> bool { return false; }
  auto operator()(const Char16Type* type) -> bool { return false; }
  auto operator()(const Char32Type* type) -> bool { return false; }
  auto operator()(const WideCharType* type) -> bool { return false; }
  auto operator()(const FloatType* type) -> bool { return false; }
  auto operator()(const DoubleType* type) -> bool { return false; }
  auto operator()(const LongDoubleType* type) -> bool { return false; }
  auto operator()(const Float16Type* type) -> bool { return false; }
  auto operator()(const Float128Type* type) -> bool { return false; }

  auto operator()(const QualType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const BoundedArrayType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const UnboundedArrayType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const PointerType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const LvalueReferenceType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const RvalueReferenceType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const FunctionType* type) -> bool {
    if (type->noexceptExpression() &&
        isDependentOperand(type->noexceptExpression()))
      return true;
    if (isDependent(type->returnType())) return true;
    for (const auto param : type->parameterTypes()) {
      if (isDependent(param)) return true;
    }
    return false;
  }

  auto operator()(const ClassType* type) -> bool {
    auto sym = type->symbol();
    if (!hasSettledDeclaration(sym)) ++unsettledClasses;
    return isDependentClass(sym);
  }

  [[nodiscard]] static auto hasSettledDeclaration(ClassSymbol* sym) -> bool {
    return sym->isSpecialization() || sym->isComplete();
  }

  [[nodiscard]] auto isDependentClass(ClassSymbol* sym) -> bool {
    if (auto ownParameters = sym->templateParameters();
        ownParameters && !ownParameters->members().empty())
      return true;

    if (sym->templateDeclaration() && !sym->primaryTemplateSymbol())
      return true;

    if (enclosedInDependentTemplate(sym->parent(), true)) return true;

    auto parameters = template_parameters_of(sym->primaryTemplateSymbol());

    std::size_t index = 0;
    for (const auto& arg : sym->templateArguments()) {
      auto parameter = parameterForArgument(parameters, index);
      ++index;
      if (isDependentArgument(arg, parameter)) return true;
    }
    return false;
  }

  [[nodiscard]] auto isDependentArgument(const TemplateArgument& arg,
                                         Symbol* parameter = nullptr) -> bool {
    if (const auto typeArg = std::get_if<const Type*>(&arg))
      return isDependent(*typeArg);

    if (std::get_if<ConstValue>(&arg)) return false;

    if (const auto exprArg = std::get_if<ExpressionAST*>(&arg))
      return isDependent(*exprArg);

    const auto symArg = std::get_if<Symbol*>(&arg);
    if (!symArg || !*symArg) return true;

    auto symbol = *symArg;

    if (auto pack = symbol_cast<ParameterPackSymbol>(symbol)) {
      for (auto elem : pack->elements()) {
        if (isDependentArgument(TemplateArgument{elem}, parameter)) return true;
      }
      return false;
    }

    if (symbol_cast<TemplateTypeParameterSymbol>(symbol))
      return namesDependentTemplateParameter(symbol);

    const bool bindsTemplateTemplateParameter =
        !parameter || symbol_cast<TemplateTypeParameterSymbol>(parameter);

    if (bindsTemplateTemplateParameter && template_name_symbol(symbol))
      return false;

    if (auto var = symbol_cast<VariableSymbol>(symbol)) {
      if (!var->constValue().has_value()) {
        if (!var->initializer()) return true;
        if (isDependentInitializer(var->initializer())) return true;
      }
    }

    if (!symbol->type()) return true;

    return isDependent(symbol->type());
  }

  auto operator()(const EnumType* type) -> bool {
    return isInTemplateScope(type->symbol());
  }

  auto operator()(const ScopedEnumType* type) -> bool {
    return isInTemplateScope(type->symbol());
  }

  auto operator()(const MemberObjectPointerType* type) -> bool {
    if (isDependent(type->classType())) return true;
    if (isDependent(type->elementType())) return true;
    return false;
  }

  auto operator()(const MemberFunctionPointerType* type) -> bool {
    if (isDependent(type->classType())) return true;
    if (isDependent(type->functionType())) return true;
    return false;
  }

  auto operator()(const NamespaceType* type) -> bool { return false; }

  auto operator()(const TypeParameterType* type) -> bool {
    return !isLocalTemplateDepth(type->depth());
  }

  auto operator()(const TemplateTypeParameterType* type) -> bool {
    return !isLocalTemplateDepth(type->depth());
  }

  auto operator()(const TemplateTypeParameterSpecializationType* type) -> bool {
    if (isDependent(type->templateParameter())) return true;
    for (const auto& argument : type->templateArguments()) {
      if (isDependentArgument(argument)) return true;
    }
    return false;
  }

  auto operator()(const PackExpansionType* type) -> bool {
    return isDependentOperand(type->pattern());
  }

  auto operator()(const DecltypeType* type) -> bool {
    return isDependentOperand(type->expression());
  }

  auto operator()(const UnresolvedNameType* type) -> bool {
    if (isDependentOperand(type->nestedNameSpecifier())) return true;
    return isDependentOperand(type->unqualifiedId());
  }

  auto operator()(const UnresolvedBoundedArrayType* type) -> bool {
    return isDependent(type->elementType()) || isDependent(type->size());
  }

  auto operator()(const UnresolvedUnderlyingType* type) -> bool {
    return isDependentOperand(type->typeId());
  }

  auto operator()(const UnresolvedBuiltinType* type) -> bool {
    return isDependentOperand(type->typeId());
  }

  auto operator()(const OverloadSetType* type) -> bool { return false; }

  auto operator()(const BuiltinVaListType* type) -> bool { return false; }

  auto operator()(const BuiltinMetaInfoType* type) -> bool { return false; }

  auto operator()(const SveType* type) -> bool { return false; }

  auto operator()(const BitIntType* type) -> bool { return false; }

  auto operator()(const UnsignedBitIntType* type) -> bool { return false; }

  auto operator()(const UnresolvedBitIntType* type) -> bool {
    return isDependentOperand(type->sizeExpression());
  }

  auto operator()(const VectorType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const UnresolvedVectorType* type) -> bool {
    if (isDependentOperand(type->elementType())) return true;
    return isDependentOperand(type->sizeExpression());
  }

  auto operator()(const ComplexType* type) -> bool {
    return isDependent(type->elementType());
  }

  auto operator()(const AtomicType* type) -> bool {
    return isDependent(type->elementType());
  }

  // clang-format off
  [[nodiscard]] auto isDependent(NestedNameSpecifierAST* ast) -> bool;
  auto operator()(GlobalNestedNameSpecifierAST* ast) -> bool;
  auto operator()(SimpleNestedNameSpecifierAST* ast) -> bool;
  auto operator()(DecltypeNestedNameSpecifierAST* ast) -> bool;
  auto operator()(TemplateNestedNameSpecifierAST* ast) -> bool;

  [[nodiscard]] auto isDependent(StatementAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(UnqualifiedIdAST* ast) -> bool {
    auto templateId = ast_cast<SimpleTemplateIdAST>(ast);
    if (!templateId) return false;
    if (namesDependentTemplateParameter(templateId->symbol)) return true;
    return hasDependentTemplateArguments(templateId);
  }
  [[nodiscard]] auto isDependent(LambdaCaptureAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(TemplateParameterAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(AttributeSpecifierAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(RequiresClauseAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(ParameterDeclarationClauseAST* ast) -> bool;
  [[nodiscard]] auto isDependent(LambdaSpecifierAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(ExceptionSpecifierAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(TrailingReturnTypeAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(RequirementAST* ast) -> bool;
  [[nodiscard]] auto isDependent(TypeConstraintAST* ast) -> bool;
  [[nodiscard]] auto isDependent(SplicerAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(DesignatorAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(NewPlacementAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(DeclaratorAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(NewInitializerAST* ast) -> bool { return false; }
  [[nodiscard]] auto isDependent(GenericAssociationAST* ast) -> bool;
  // clang-format on

  auto operator()(CharLiteralExpressionAST* ast) -> bool;
  auto operator()(BoolLiteralExpressionAST* ast) -> bool;
  auto operator()(IntLiteralExpressionAST* ast) -> bool;
  auto operator()(FloatLiteralExpressionAST* ast) -> bool;
  auto operator()(NullptrLiteralExpressionAST* ast) -> bool;
  auto operator()(StringLiteralExpressionAST* ast) -> bool;
  auto operator()(UserDefinedStringLiteralExpressionAST* ast) -> bool;
  auto operator()(ObjectLiteralExpressionAST* ast) -> bool;
  auto operator()(ThisExpressionAST* ast) -> bool;
  auto operator()(PackIndexExpressionAST* ast) -> bool;
  auto operator()(GenericSelectionExpressionAST* ast) -> bool;
  auto operator()(NestedStatementExpressionAST* ast) -> bool;
  auto operator()(DefaultInitializerExpressionAST* ast) -> bool {
    return isDependent(ast->expression);
  }
  auto operator()(NestedExpressionAST* ast) -> bool;
  auto operator()(IdExpressionAST* ast) -> bool;
  auto operator()(LambdaExpressionAST* ast) -> bool;
  auto operator()(FoldExpressionAST* ast) -> bool;
  auto operator()(RightFoldExpressionAST* ast) -> bool;
  auto operator()(LeftFoldExpressionAST* ast) -> bool;
  auto operator()(RequiresExpressionAST* ast) -> bool;
  auto operator()(VaArgExpressionAST* ast) -> bool;
  auto operator()(SubscriptExpressionAST* ast) -> bool;
  auto operator()(CallExpressionAST* ast) -> bool;
  auto operator()(TypeConstructionAST* ast) -> bool;
  auto operator()(BracedTypeConstructionAST* ast) -> bool;
  auto operator()(SpliceMemberExpressionAST* ast) -> bool;
  auto operator()(MemberExpressionAST* ast) -> bool;
  auto operator()(PostIncrExpressionAST* ast) -> bool;
  auto operator()(CppCastExpressionAST* ast) -> bool;
  auto operator()(BuiltinBitCastExpressionAST* ast) -> bool;
  auto operator()(BuiltinConvertVectorExpressionAST* ast) -> bool;
  auto operator()(BuiltinOffsetofExpressionAST* ast) -> bool;
  auto operator()(TypeidExpressionAST* ast) -> bool;
  auto operator()(TypeidOfTypeExpressionAST* ast) -> bool;
  auto operator()(SpliceExpressionAST* ast) -> bool;
  auto operator()(GlobalScopeReflectExpressionAST* ast) -> bool;
  auto operator()(NamespaceReflectExpressionAST* ast) -> bool;
  auto operator()(TypeIdReflectExpressionAST* ast) -> bool;
  auto operator()(ReflectExpressionAST* ast) -> bool;
  auto operator()(LabelAddressExpressionAST* ast) -> bool;
  auto operator()(UnaryExpressionAST* ast) -> bool;
  auto operator()(AwaitExpressionAST* ast) -> bool;
  auto operator()(SizeofExpressionAST* ast) -> bool;
  auto operator()(SizeofTypeExpressionAST* ast) -> bool;
  auto operator()(SizeofPackExpressionAST* ast) -> bool;
  auto operator()(AlignofTypeExpressionAST* ast) -> bool;
  auto operator()(AlignofExpressionAST* ast) -> bool;
  auto operator()(NoexceptExpressionAST* ast) -> bool;
  auto operator()(NewExpressionAST* ast) -> bool;
  auto operator()(DeleteExpressionAST* ast) -> bool;
  auto operator()(CastExpressionAST* ast) -> bool;
  auto operator()(ImplicitCastExpressionAST* ast) -> bool;
  auto operator()(ConstExpressionAST* ast) -> bool;
  auto operator()(BinaryExpressionAST* ast) -> bool;
  auto operator()(ThreeWayComparisonExpressionAST* ast) -> bool;
  auto operator()(ConditionalExpressionAST* ast) -> bool;
  auto operator()(YieldExpressionAST* ast) -> bool;
  auto operator()(ThrowExpressionAST* ast) -> bool;
  auto operator()(AssignmentExpressionAST* ast) -> bool;
  auto operator()(TargetExpressionAST* ast) -> bool;
  auto operator()(RightExpressionAST* ast) -> bool;
  auto operator()(CompoundAssignmentExpressionAST* ast) -> bool;
  auto operator()(PackExpansionExpressionAST* ast) -> bool;
  auto operator()(DesignatedInitializerClauseAST* ast) -> bool;
  auto operator()(TypeTraitExpressionAST* ast) -> bool;
  auto operator()(ConditionExpressionAST* ast) -> bool;
  auto operator()(EqualInitializerAST* ast) -> bool;
  auto operator()(BracedInitListAST* ast) -> bool;
  auto operator()(ParenInitializerAST* ast) -> bool;

  [[nodiscard]] auto isDependent(SpecifierAST* ast) -> bool {
    if (!ast) return false;
    return visit(*this, ast);
  }

  // clang-format off
  auto operator()(TypedefSpecifierAST* ast) -> bool { return false; }
  auto operator()(FriendSpecifierAST* ast) -> bool { return false; }
  auto operator()(ConstevalSpecifierAST* ast) -> bool { return false; }
  auto operator()(ConstinitSpecifierAST* ast) -> bool { return false; }
  auto operator()(ConstexprSpecifierAST* ast) -> bool { return false; }
  auto operator()(InlineSpecifierAST* ast) -> bool { return false; }
  auto operator()(NoreturnSpecifierAST* ast) -> bool { return false; }
  auto operator()(StaticSpecifierAST* ast) -> bool { return false; }
  auto operator()(ExternSpecifierAST* ast) -> bool { return false; }
  auto operator()(RegisterSpecifierAST* ast) -> bool { return false; }
  auto operator()(ThreadLocalSpecifierAST* ast) -> bool { return false; }
  auto operator()(ThreadSpecifierAST* ast) -> bool { return false; }
  auto operator()(MutableSpecifierAST* ast) -> bool { return false; }
  auto operator()(VirtualSpecifierAST* ast) -> bool { return false; }
  auto operator()(ExplicitSpecifierAST* ast) -> bool { return false; }
  auto operator()(AutoTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(VoidTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(SizeTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(SignTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(BuiltinTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(UnaryBuiltinTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(BinaryBuiltinTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(IntegralTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(FloatingPointTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(ComplexTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(NamedTypeSpecifierAST* ast) -> bool {
    if (!ast) return false;
    if (namesDependentTemplateParameter(ast->symbol)) return true;
    if (auto alias = symbol_cast<TypeAliasSymbol>(ast->symbol)) {
      const auto bindsAliasTemplate = alias->templateParameters() &&
                                      ast_cast<NameIdAST>(ast->unqualifiedId);
      if (!bindsAliasTemplate) {
        if (!alias->type()) return true;
        if (isDependent(alias->type())) return true;
      }
    }
    if (isDependent(ast->nestedNameSpecifier)) return true;
    return hasDependentTemplateArguments(
        ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId));
  }
  auto operator()(AtomicTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(BitIntTypeSpecifierAST* ast) -> bool {
    return isDependent(ast->sizeExpression);
  }
  auto operator()(UnderlyingTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(ElaboratedTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(DecltypeAutoSpecifierAST* ast) -> bool { return false; }

  auto operator()(DecltypeSpecifierAST* ast) -> bool {
    if (ast->type && isDependent(ast->type)) return true;
    if (ast->expression && !ast->type) return true;
    if (ast->expression && isDependent(ast->expression)) return true;
    return false;
  }

  auto operator()(PlaceholderTypeSpecifierAST* ast) -> bool { return false; }
  auto operator()(ConstQualifierAST* ast) -> bool { return false; }
  auto operator()(VolatileQualifierAST* ast) -> bool { return false; }
  auto operator()(RestrictQualifierAST* ast) -> bool { return false; }
  auto operator()(AtomicQualifierAST* ast) -> bool { return false; }
  auto operator()(EnumSpecifierAST* ast) -> bool { return false; }
  auto operator()(ClassSpecifierAST* ast) -> bool { return false; }
  auto operator()(TypenameSpecifierAST* ast) -> bool { return false; }
  auto operator()(SplicerTypeSpecifierAST* ast) -> bool { return false; }
  // clang-format on
};
}  // namespace

auto IsDependent::isDependent(NestedNameSpecifierAST* ast) -> bool {
  if (!ast) return false;

  if (ast->symbol) {
    if (namesDependentTemplateParameter(ast->symbol)) return true;
    if (isDependent(ast->symbol->type())) return true;
  }

  return visit(*this, ast);
}

auto IsDependent::operator()(GlobalNestedNameSpecifierAST* ast) -> bool {
  return !ast->symbol;
}

auto IsDependent::operator()(SimpleNestedNameSpecifierAST* ast) -> bool {
  if (isDependent(ast->nestedNameSpecifier)) return true;
  return !ast->symbol && !ast->nestedNameSpecifier;
}

auto IsDependent::operator()(DecltypeNestedNameSpecifierAST* ast) -> bool {
  return !ast->symbol;
}

auto IsDependent::operator()(TemplateNestedNameSpecifierAST* ast) -> bool {
  if (hasDependentTemplateArguments(ast->templateId)) return true;
  if (isDependent(ast->nestedNameSpecifier)) return true;

  return !ast->symbol && !ast->nestedNameSpecifier;
}

auto IsDependent::operator()(CharLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(BoolLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(IntLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(FloatLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(NullptrLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(StringLiteralExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(UserDefinedStringLiteralExpressionAST* ast)
    -> bool {
  return false;
}

auto IsDependent::operator()(ObjectLiteralExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;
  if (isDependent(ast->bracedInitList)) return true;

  return false;
}

auto IsDependent::operator()(ThisExpressionAST* ast) -> bool { return false; }

auto IsDependent::operator()(PackIndexExpressionAST* ast) -> bool {
  if (isDependentOperand(ast->packExpression)) return true;
  return isDependentOperand(ast->indexExpression);
}

auto IsDependent::isDependent(GenericAssociationAST* ast) -> bool {
  if (auto typeAssoc = ast_cast<TypeGenericAssociationAST>(ast)) {
    return isDependent(typeAssoc->typeId);
  }

  return false;
}

auto IsDependent::operator()(GenericSelectionExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  for (auto node : ListView{ast->genericAssociationList}) {
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(NestedStatementExpressionAST* ast) -> bool {
  if (isDependent(ast->statement)) return true;

  return false;
}

auto IsDependent::operator()(NestedExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(IdExpressionAST* ast) -> bool {
  if (auto functionType = type_cast<FunctionType>(ast->type)) {
    if (isDependent(functionType)) return true;
  }
  if (isDependent(ast->nestedNameSpecifier)) return true;
  if (isDependent(ast->unqualifiedId)) return true;
  if (ast->symbol && visit(ReferencedMemberIsDependent{*this}, ast->symbol))
    return true;

  if (namesDependentTemplateParameter(ast->symbol)) return true;

  if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
    if (isDependent(enumerator->type())) return true;
  }

  if (auto field = symbol_cast<FieldSymbol>(ast->symbol)) {
    if (field->isStatic() && !field->initializer() && isInTemplateScope(field))
      return true;
    if (hasValueDependentInitializer(field, field->initializer(),
                                     field->isConstexpr()))
      return true;
    if (isDependent(field->type())) return true;
  }
  if (auto var = symbol_cast<VariableSymbol>(ast->symbol)) {
    if (namesTypeDependentPredefinedVariable(unit, var)) return true;
    if (hasValueDependentInitializer(var, var->initializer(),
                                     var->isConstexpr()))
      return true;
    if (isDependent(var->type())) return true;
  }
  auto hasDependentPlaceholderReturn = [&](FunctionSymbol* function) {
    if (!function) return false;
    auto functionType = type_cast<FunctionType>(function->type());
    if (!functionType) return false;
    if (!containsPlaceholderType(functionType->returnType())) return false;
    return enclosedInDependentTemplate(enclosingScopeForDependence(function),
                                       /*stopAtConcreteSpecialization=*/true);
  };

  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(ast->symbol)) {
    for (auto function : overloadSet->functions()) {
      if (hasDependentPlaceholderReturn(function)) return true;
    }
    if (hasMemberDeclaredWithDependentType(overloadSet)) return true;
  }

  if (auto func = symbol_cast<FunctionSymbol>(ast->symbol)) {
    if (hasDependentPlaceholderReturn(func)) return true;
    if (isDeclaredWithDependentType(func)) return true;
    if (func->isStatic() && isInTemplateScope(func)) return true;
  }
  if (auto param = symbol_cast<ParameterSymbol>(ast->symbol)) {
    if (isDependent(param->type())) return true;
  }

  if (auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId)) {
    if (hasDependentTemplateArguments(templateId)) return true;
  }

  return false;
}

auto IsDependent::operator()(LambdaExpressionAST* ast) -> bool {
  for (auto node : ListView{ast->captureList}) {
    if (isDependent(node)) return true;
  }

  for (auto node : ListView{ast->templateParameterList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->templateRequiresClause)) return true;

  for (auto node : ListView{ast->expressionAttributeList}) {
    if (isDependent(node)) return true;
  }

  auto parameterDeclarationClauseResult =
      isDependent(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->gnuAtributeList}) {
    if (isDependent(node)) return true;
  }

  for (auto node : ListView{ast->lambdaSpecifierList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->exceptionSpecifier)) return true;

  for (auto node : ListView{ast->attributeList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->trailingReturnType)) return true;
  if (isDependent(ast->requiresClause)) return true;
  if (isDependent(ast->statement)) return true;

  return false;
}

auto IsDependent::operator()(FoldExpressionAST* ast) -> bool {
  if (isDependentOperand(ast->leftExpression)) return true;
  return isDependentOperand(ast->rightExpression);
}

auto IsDependent::operator()(RightFoldExpressionAST* ast) -> bool {
  return isDependentOperand(ast->expression);
}

auto IsDependent::operator()(LeftFoldExpressionAST* ast) -> bool {
  return isDependentOperand(ast->expression);
}

auto IsDependent::isDependent(ParameterDeclarationClauseAST* ast) -> bool {
  if (!ast) return false;

  for (auto parameter : ListView{ast->parameterDeclarationList}) {
    if (isDependent(parameter->type)) return true;
    for (auto typeSpecifier : ListView{parameter->typeSpecifierList}) {
      if (isDependent(typeSpecifier)) return true;
    }
  }

  return false;
}

auto IsDependent::isDependent(TypeConstraintAST* ast) -> bool {
  if (!ast) return false;
  if (isDependent(ast->nestedNameSpecifier)) return true;

  for (auto argument : ListView{ast->templateArgumentList}) {
    if (isDependentTemplateArgument(argument)) return true;
  }

  return false;
}

auto IsDependent::isDependent(RequirementAST* ast) -> bool {
  if (!ast) return false;

  if (auto simpleRequirement = ast_cast<SimpleRequirementAST>(ast))
    return isDependent(simpleRequirement->expression);

  if (auto compoundRequirement = ast_cast<CompoundRequirementAST>(ast)) {
    if (isDependent(compoundRequirement->expression)) return true;
    return isDependent(compoundRequirement->typeConstraint);
  }

  if (auto typeRequirement = ast_cast<TypeRequirementAST>(ast))
    return isDependent(typeRequirement->typeId);

  if (auto nestedRequirement = ast_cast<NestedRequirementAST>(ast))
    return isDependent(nestedRequirement->expression);

  return false;
}

auto IsDependent::operator()(RequiresExpressionAST* ast) -> bool {
  if (isDependent(ast->parameterDeclarationClause)) return true;

  for (auto node : ListView{ast->requirementList}) {
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(VaArgExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(SubscriptExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;
  if (isDependent(ast->indexExpression)) return true;

  return false;
}

auto IsDependent::operator()(CallExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;

  for (auto node : ListView{ast->expressionList}) {
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(TypeConstructionAST* ast) -> bool {
  if (isDependent(ast->typeSpecifier)) return true;
  for (auto node : ListView{ast->expressionList}) {
    if (isDependent(node)) return true;
  }
  return false;
}

auto IsDependent::operator()(BracedTypeConstructionAST* ast) -> bool {
  if (isDependent(ast->typeSpecifier)) return true;
  if (isDependent(ast->bracedInitList)) return true;

  return false;
}

auto IsDependent::operator()(SpliceMemberExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;
  if (isDependent(ast->splicer)) return true;

  return false;
}

auto IsDependent::operator()(MemberExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;
  if (isDependent(ast->nestedNameSpecifier)) return true;
  if (isDependent(ast->unqualifiedId)) return true;
  if (!ast->symbol) return false;

  return visit(ReferencedMemberIsDependent{*this}, ast->symbol);
}

auto IsDependent::operator()(PostIncrExpressionAST* ast) -> bool {
  if (isDependent(ast->baseExpression)) return true;

  return false;
}

auto IsDependent::operator()(CppCastExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(BuiltinBitCastExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(BuiltinConvertVectorExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(BuiltinOffsetofExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  for (auto node : ListView{ast->designatorList}) {
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(TypeidExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(TypeidOfTypeExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(SpliceExpressionAST* ast) -> bool {
  if (isDependent(ast->splicer)) return true;

  return false;
}

auto IsDependent::operator()(GlobalScopeReflectExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(NamespaceReflectExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(TypeIdReflectExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(ReflectExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(LabelAddressExpressionAST* ast) -> bool {
  return false;
}

auto IsDependent::operator()(UnaryExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(AwaitExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(SizeofExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(SizeofTypeExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(SizeofPackExpressionAST* ast) -> bool {
  if (!localTemplateDepth) return true;
  if (namesDependentTemplateParameter(ast->symbol)) return true;
  return ast->symbol && isDependent(ast->symbol->type());
}

auto IsDependent::operator()(AlignofTypeExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;

  return false;
}

auto IsDependent::operator()(AlignofExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(NoexceptExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(NewExpressionAST* ast) -> bool {
  if (isDependent(ast->newPlacement)) return true;

  for (auto node : ListView{ast->typeSpecifierList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->declarator)) return true;
  if (isDependent(ast->newInitalizer)) return true;

  return false;
}

auto IsDependent::operator()(DeleteExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(CastExpressionAST* ast) -> bool {
  if (isDependent(ast->typeId)) return true;
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(ImplicitCastExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(ConstExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(BinaryExpressionAST* ast) -> bool {
  if (isDependent(ast->leftExpression)) return true;
  if (isDependent(ast->rightExpression)) return true;

  return false;
}

auto IsDependent::operator()(ThreeWayComparisonExpressionAST* ast) -> bool {
  return isDependent(ast->comparison);
}

auto IsDependent::operator()(ConditionalExpressionAST* ast) -> bool {
  if (isDependent(ast->condition)) return true;
  if (isDependent(ast->iftrueExpression)) return true;
  if (isDependent(ast->iffalseExpression)) return true;

  return false;
}

auto IsDependent::operator()(YieldExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(ThrowExpressionAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(AssignmentExpressionAST* ast) -> bool {
  if (isDependent(ast->leftExpression)) return true;
  if (isDependent(ast->rightExpression)) return true;

  return false;
}

auto IsDependent::operator()(TargetExpressionAST* ast) -> bool { return false; }

auto IsDependent::operator()(RightExpressionAST* ast) -> bool { return false; }

auto IsDependent::operator()(CompoundAssignmentExpressionAST* ast) -> bool {
  if (isDependent(ast->targetExpression)) return true;
  if (isDependent(ast->leftExpression)) return true;
  if (isDependent(ast->rightExpression)) return true;
  if (isDependent(ast->adjustExpression)) return true;

  return false;
}

auto IsDependent::operator()(PackExpansionExpressionAST* ast) -> bool {
  return isDependentOperand(ast->expression);
}

auto IsDependent::operator()(DesignatedInitializerClauseAST* ast) -> bool {
  for (auto node : ListView{ast->designatorList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->initializer)) return true;

  return false;
}

auto IsDependent::operator()(TypeTraitExpressionAST* ast) -> bool {
  if (isDependent(ast->type)) return true;

  for (auto node : ListView{ast->typeIdList}) {
    if (!node->type) return true;
    if (isDependent(node)) return true;
  }

  return false;
}

auto IsDependent::operator()(ConditionExpressionAST* ast) -> bool {
  for (auto node : ListView{ast->attributeList}) {
    if (isDependent(node)) return true;
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    if (isDependent(node)) return true;
  }

  if (isDependent(ast->declarator)) return true;
  if (isDependent(ast->initializer)) return true;

  return false;
}

auto IsDependent::operator()(EqualInitializerAST* ast) -> bool {
  if (isDependent(ast->expression)) return true;

  return false;
}

auto IsDependent::operator()(BracedInitListAST* ast) -> bool {
  for (auto node : ListView{ast->expressionList}) {
    if (isDependent(node)) return true;
  }
  return false;
}

auto IsDependent::operator()(ParenInitializerAST* ast) -> bool {
  for (auto node : ListView{ast->expressionList}) {
    if (isDependent(node)) return true;
  }
  return false;
}

auto isDependent(TranslationUnit* unit, ExpressionAST* ast) -> bool {
  return IsDependent{unit}.isDependent(ast);
}

auto isDependent(TranslationUnit* unit, TypeIdAST* ast) -> bool {
  return IsDependent{unit}.isDependent(ast);
}

auto isDependent(TranslationUnit* unit, SpecifierAST* spec) -> bool {
  return IsDependent{unit}.isDependent(spec);
}

auto isDependent(TranslationUnit* unit, const Type* type) -> bool {
  return IsDependent{unit}.isDependent(type);
}

auto isEnclosedInDependentTemplate(TranslationUnit* unit, ScopeSymbol* scope,
                                   bool stopAtConcreteSpecialization) -> bool {
  return IsDependent{unit}.enclosedInDependentTemplate(
      scope, stopAtConcreteSpecialization);
}

auto namesTypeDependentPredefinedVariable(TranslationUnit* unit, Symbol* symbol)
    -> bool {
  if (!is_function_local_predefined_variable(symbol)) return false;
  return isEnclosedInDependentTemplate(unit, symbol->parent(),
                                       /*stopAtConcreteSpecialization=*/true);
}

auto isDependentTypeParameterSymbol(Symbol* symbol) -> bool {
  return symbol_cast<TypeParameterSymbol>(symbol) ||
         symbol_cast<TemplateTypeParameterSymbol>(symbol);
}

auto isDependentTemplateArgument(TranslationUnit* unit,
                                 TemplateArgumentAST* arg) -> bool {
  return IsDependent{unit}.isDependentTemplateArgument(arg);
}

auto isDependentTemplateArgument(TranslationUnit* unit,
                                 const TemplateArgument& argument) -> bool {
  return IsDependent{unit}.isDependentArgument(argument);
}

auto containsUnexpandedParameterPack(AST* ast) -> bool {
  if (!ast) return false;
  FindUnexpandedParameterPack scan{ast};
  scan.accept(ast);
  return scan.found;
}

auto hasDependentTemplateArguments(TranslationUnit* unit, UnqualifiedIdAST* id)
    -> bool {
  return IsDependent{unit}.hasDependentTemplateArguments(id);
}

auto isDependent(TranslationUnit* unit, NestedNameSpecifierAST* ast) -> bool {
  return IsDependent{unit}.isDependent(ast);
}

auto isCurrentInstantiation(ScopeSymbol* scope, const Type* type) -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return false;

  for (auto current = scope; current; current = current->parent()) {
    if (current == classType->symbol()) return true;
  }

  return false;
}

auto isDependentBaseClass(TranslationUnit* unit, ClassSymbol* classSymbol,
                          BaseClassSymbol* baseClass) -> bool {
  auto base = baseClass->symbol();
  if (!base) return true;

  auto baseType = base->type();
  if (!baseType) return true;

  if (isCurrentInstantiation(classSymbol, baseType)) return false;

  return isDependent(unit, baseType);
}

auto hasDependentBaseClass(TranslationUnit* unit, ClassSymbol* classSymbol)
    -> bool {
  if (!classSymbol) return false;

  std::vector<ClassSymbol*> visited{classSymbol};

  for (std::size_t index = 0; index < visited.size(); ++index) {
    auto current = visited[index];

    for (auto baseClass : current->baseClasses()) {
      if (isDependentBaseClass(unit, current, baseClass)) return true;

      auto baseClassType = type_cast<ClassType>(baseClass->symbol()->type());
      if (!baseClassType) continue;
      if (isCurrentInstantiation(current, baseClassType)) continue;

      auto base = baseClassType->symbol();
      if (std::ranges::contains(visited, base)) continue;
      visited.push_back(base);
    }
  }

  return false;
}
}  // namespace cxx
