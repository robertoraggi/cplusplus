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
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/standard_conversion.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <format>

namespace cxx {
namespace {
struct IsPackParameter {
  auto operator()(TypenameTypeParameterAST* parameter) -> bool {
    return parameter->isPack;
  }

  auto operator()(NonTypeTemplateParameterAST* parameter) -> bool {
    return parameter->declaration && parameter->declaration->isPack;
  }

  auto operator()(TemplateTypeParameterAST* parameter) -> bool {
    return parameter->isPack;
  }

  auto operator()(ConstraintTypeParameterAST* parameter) -> bool {
    return static_cast<bool>(parameter->ellipsisLoc);
  }
};

struct HasDefaultTemplateArgument {
  auto operator()(TypenameTypeParameterAST* parameter) -> bool {
    return parameter->equalLoc && parameter->typeId;
  }

  auto operator()(NonTypeTemplateParameterAST* parameter) -> bool {
    return parameter->declaration && parameter->declaration->equalLoc &&
           parameter->declaration->expression;
  }

  auto operator()(TemplateTypeParameterAST* parameter) -> bool {
    return parameter->idExpression != nullptr;
  }

  auto operator()(ConstraintTypeParameterAST* parameter) -> bool {
    return parameter->equalLoc && parameter->typeId;
  }
};

[[nodiscard]] auto substitutionScope(TemplateDeclarationAST* templateDecl)
    -> ScopeSymbol* {
  if (!templateDecl->symbol) return nullptr;
  return templateDecl->symbol->parent();
}

[[nodiscard]] auto declaredTypeId(Arena* arena,
                                  ParameterDeclarationAST* declaration)
    -> TypeIdAST* {
  auto typeId = TypeIdAST::create(arena);
  typeId->typeSpecifierList = declaration->typeSpecifierList;
  typeId->declarator = declaration->declarator;
  typeId->type = declaration->type;
  return typeId;
}
}  // namespace

auto isPackParameter(TemplateParameterAST* parameter) -> bool {
  if (!parameter) return false;
  return visit(IsPackParameter{}, parameter);
}

auto isPackExpansion(TypeIdAST* typeId) -> bool {
  if (!typeId || !typeId->declarator) return false;
  return ast_cast<ParameterPackAST>(typeId->declarator->coreDeclarator) !=
         nullptr;
}

namespace {

[[nodiscard]] auto injectedTypeArgument(Arena* arena, Symbol* parameter)
    -> TemplateArgumentAST* {
  auto specifier = NamedTypeSpecifierAST::create(arena);
  specifier->unqualifiedId =
      NameIdAST::create(arena, name_cast<Identifier>(parameter->name()));
  specifier->symbol = parameter;

  auto typeId = TypeIdAST::create(arena);
  typeId->typeSpecifierList = make_list_node<SpecifierAST>(arena, specifier);
  typeId->type = parameter->type();

  if (is_template_parameter_pack(parameter)) {
    typeId->declarator = DeclaratorAST::create(arena);
    typeId->declarator->coreDeclarator = ParameterPackAST::create(arena);
  }

  return TypeTemplateArgumentAST::create(arena, typeId);
}

[[nodiscard]] auto injectedValueArgument(Arena* arena,
                                         NonTypeParameterSymbol* parameter)
    -> TemplateArgumentAST* {
  auto id = IdExpressionAST::create(arena);
  id->unqualifiedId =
      NameIdAST::create(arena, name_cast<Identifier>(parameter->name()));
  id->symbol = parameter;
  id->type = parameter->type();
  id->valueCategory = ValueCategory::kPrValue;

  if (!parameter->isParameterPack())
    return ExpressionTemplateArgumentAST::create(arena, id);

  auto expansion = PackExpansionExpressionAST::create(arena);
  expansion->expression = id;
  expansion->type = id->type;
  expansion->valueCategory = id->valueCategory;
  return ExpressionTemplateArgumentAST::create(arena, expansion);
}

[[nodiscard]] auto injectedArgument(Arena* arena, Symbol* parameter)
    -> TemplateArgumentAST* {
  if (auto value = symbol_cast<NonTypeParameterSymbol>(parameter))
    return injectedValueArgument(arena, value);
  return injectedTypeArgument(arena, parameter);
}

}  // namespace

auto injected_template_argument_list(TranslationUnit* unit,
                                     List<TemplateParameterAST*>* parameters)
    -> List<TemplateArgumentAST*>* {
  List<TemplateArgumentAST*>* arguments = nullptr;
  auto out = &arguments;
  for (auto parameter : ListView{parameters}) {
    if (!parameter->symbol) return nullptr;
    *out = make_list_node(unit->arena(),
                          injectedArgument(unit->arena(), parameter->symbol));
    out = &(*out)->next;
  }
  return arguments;
}

auto class_template_arguments(TranslationUnit* unit, ClassSymbol* classSymbol)
    -> std::vector<TemplateArgument> {
  if (!classSymbol) return {};

  if (classSymbol->isSpecialization()) {
    auto arguments = classSymbol->templateArguments();
    return std::vector<TemplateArgument>{arguments.begin(), arguments.end()};
  }

  auto declaration = classSymbol->templateDeclaration();
  if (!declaration) return {};

  auto arguments = Substitution::writtenTemplateArguments(
      unit, injected_template_argument_list(
                unit, declaration->templateParameterList));
  return arguments.value_or(std::vector<TemplateArgument>{});
}

auto hasWrittenDefaultTemplateArgument(TemplateParameterAST* parameter)
    -> bool {
  if (!parameter) return false;
  return visit(HasDefaultTemplateArgument{}, parameter);
}

void recordDefaultTemplateArgument(TemplateParameterAST* parameter,
                                   TemplateParameterAST* pattern) {
  if (!parameter || !parameter->symbol) return;

  if (hasWrittenDefaultTemplateArgument(parameter)) {
    set_default_template_argument(parameter->symbol, parameter);
    return;
  }

  if (!pattern) return;

  set_default_template_argument(parameter->symbol,
                                default_template_argument(pattern->symbol));
}

auto hasDefaultTemplateArgument(TemplateParameterAST* parameter) -> bool {
  if (!parameter) return false;
  return default_template_argument(parameter->symbol) != nullptr;
}

namespace {

[[nodiscard]] auto declaratorIsEmptyOrPackExpansion(TypeIdAST* typeId) -> bool {
  auto declarator = typeId->declarator;
  if (!declarator) return true;
  if (declarator->ptrOpList || declarator->declaratorChunkList) return false;
  return !declarator->coreDeclarator || isPackExpansion(typeId);
}

[[nodiscard]] auto writtenTemplateArgumentName(TemplateArgumentAST* argument)
    -> NamedTypeSpecifierAST* {
  auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument);
  if (!typeArgument || !typeArgument->typeId) return nullptr;
  auto typeId = typeArgument->typeId;
  if (!declaratorIsEmptyOrPackExpansion(typeId)) return nullptr;
  auto specifiers = typeId->typeSpecifierList;
  if (!specifiers || specifiers->next) return nullptr;
  auto named = ast_cast<NamedTypeSpecifierAST>(specifiers->value);
  if (!named) return nullptr;
  if (ast_cast<SimpleTemplateIdAST>(named->unqualifiedId)) return nullptr;
  return named;
}

struct DenotesTemplate {
  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> bool {
    return symbol->templateParameters() || symbol->isSpecialization();
  }

  [[nodiscard]] auto operator()(InjectedClassNameSymbol* symbol) const -> bool {
    return symbol->classSymbol() && (*this)(symbol->classSymbol());
  }

  [[nodiscard]] auto operator()(TypeAliasSymbol* symbol) const -> bool {
    return symbol->templateParameters() != nullptr;
  }

  [[nodiscard]] auto operator()(TemplateTypeParameterSymbol*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(ParameterPackSymbol*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(Symbol*) const -> bool { return false; }
};

struct DenotesTemplateNotType {
  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> bool {
    return symbol->templateParameters() && !symbol->isSpecialization();
  }

  [[nodiscard]] auto operator()(TypeAliasSymbol* symbol) const -> bool {
    return symbol->templateParameters() != nullptr;
  }

  [[nodiscard]] auto operator()(TemplateTypeParameterSymbol*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(Symbol*) const -> bool { return false; }
};

struct DenotesValue {
  [[nodiscard]] auto operator()(VariableSymbol*) const -> bool { return true; }

  [[nodiscard]] auto operator()(NonTypeParameterSymbol*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(ParameterPackSymbol*) const -> bool {
    return true;
  }

  [[nodiscard]] auto operator()(Symbol*) const -> bool { return false; }
};

}  // namespace

auto denotesTemplateName(TemplateArgumentAST* argument) -> bool {
  auto named = writtenTemplateArgumentName(argument);
  if (!named || !named->symbol) return false;
  return visit(DenotesTemplateNotType{}, named->symbol);
}

namespace {

struct MatchesTemplateParameterKind {
  TemplateArgumentAST* argument;

  [[nodiscard]] auto isTypeId() const -> bool {
    auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument);
    if (!typeArgument || !typeArgument->typeId) return false;
    return !denotesTemplateName(argument);
  }

  [[nodiscard]] auto operator()(TypenameTypeParameterAST*) const -> bool {
    return isTypeId();
  }

  [[nodiscard]] auto operator()(ConstraintTypeParameterAST*) const -> bool {
    return isTypeId();
  }

  [[nodiscard]] auto operator()(TemplateTypeParameterAST*) const -> bool {
    auto named = writtenTemplateArgumentName(argument);
    if (!named) return false;
    if (!named->symbol) return true;
    return visit(DenotesTemplate{}, named->symbol);
  }

  [[nodiscard]] auto operator()(NonTypeTemplateParameterAST*) const -> bool {
    if (auto expression = ast_cast<ExpressionTemplateArgumentAST>(argument))
      return expression->expression != nullptr;
    auto named = writtenTemplateArgumentName(argument);
    if (!named || !named->symbol) return false;
    return visit(DenotesValue{}, named->symbol);
  }
};

}  // namespace

auto matchesTemplateParameterKind(TemplateParameterAST* parameter,
                                  TemplateArgumentAST* argument) -> bool {
  if (!parameter || !argument) return false;
  return visit(MatchesTemplateParameterKind{argument}, parameter);
}

auto TemplateArguments::isPackExpansion(TemplateArgumentAST* argument) -> bool {
  if (auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument))
    return cxx::isPackExpansion(typeArgument->typeId);

  if (auto expressionArgument =
          ast_cast<ExpressionTemplateArgumentAST>(argument)) {
    return ast_cast<PackExpansionExpressionAST>(
               expressionArgument->expression) != nullptr;
  }

  return false;
}

auto TemplateArguments::hasPackExpansion(
    List<TemplateArgumentAST*>* templateArgumentList) -> bool {
  for (auto argument : ListView{templateArgumentList}) {
    if (isPackExpansion(argument)) return true;
  }
  return false;
}

auto TemplateArguments::last(List<TemplateArgumentAST*>* templateArgumentList)
    -> TemplateArgumentAST* {
  TemplateArgumentAST* result = nullptr;
  for (auto argument : ListView{templateArgumentList}) result = argument;
  return result;
}

auto TemplateArguments::count(List<TemplateArgumentAST*>* templateArgumentList)
    -> int {
  int count = 0;
  for ([[maybe_unused]] auto argument : ListView{templateArgumentList}) ++count;
  return count;
}

auto TemplateArity::of(TemplateDeclarationAST* templateDecl) -> TemplateArity {
  TemplateArity arity;
  if (!templateDecl) return arity;

  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    ++arity.maxArgs;

    if (isPackParameter(parameter)) {
      ++arity.packCount;
      arity.hasParameterPack = true;
      continue;
    }

    if (!hasDefaultTemplateArgument(parameter)) {
      arity.minArgs = arity.maxArgs;
    }
  }

  return arity;
}

auto TemplateArity::matches(TemplateDeclarationAST* templateDecl,
                            List<TemplateArgumentAST*>* templateArgumentList,
                            bool isFunctionTemplate) -> bool {
  if (!templateDecl) return true;

  auto arity = of(templateDecl);
  auto argc = TemplateArguments::count(templateArgumentList);

  const bool expandsAnUnknownNumberOfArguments =
      TemplateArguments::hasPackExpansion(templateArgumentList);

  if (!isFunctionTemplate && !expandsAnUnknownNumberOfArguments &&
      argc < arity.minArgs)
    return false;
  if (!arity.hasParameterPack && argc > arity.maxArgs) return false;

  return true;
}

auto TemplateArguments::integerLiteralExpression(const ConstInt& value,
                                                 const Type* type) const
    -> ExpressionAST* {
  if (!value.magnitudeFitsInUIntMax()) return nullptr;

  auto arena = unit_->arena();
  auto literal = unit_->control()->integerLiteral(value.toDecimalString());

  ExpressionAST* expression = IntLiteralExpressionAST::create(
      arena, literal, /*literalOperatorCall=*/nullptr, ValueCategory::kPrValue,
      type);

  if (!value.isNegative()) return expression;

  return UnaryExpressionAST::create(arena, SourceLocation{}, expression,
                                    TokenKind::T_MINUS, /*symbol=*/nullptr,
                                    /*isVirtualDispatch=*/false,
                                    ValueCategory::kPrValue, type);
}

auto TemplateArguments::templateName(Symbol* templateSymbol) const
    -> TemplateArgumentAST* {
  if (!templateSymbol) return nullptr;

  auto identifier = name_cast<Identifier>(templateSymbol->name());
  if (!identifier) return nullptr;

  auto arena = unit_->arena();

  auto namedSpecifier = NamedTypeSpecifierAST::create(arena);
  namedSpecifier->unqualifiedId = NameIdAST::create(arena, identifier);
  namedSpecifier->symbol = templateSymbol;

  auto typeId = TypeIdAST::create(arena);
  typeId->typeSpecifierList = make_list_node<SpecifierAST>(
      arena, static_cast<SpecifierAST*>(namedSpecifier));
  typeId->type = templateSymbol->type();

  auto argument = TypeTemplateArgumentAST::create(arena);
  argument->typeId = typeId;
  return argument;
}

auto TemplateArguments::defaultArgument(
    TemplateDeclarationAST* templateDecl, TemplateParameterAST* parameter,
    const std::vector<TemplateArgument>& argumentsSoFar) const
    -> TemplateArgumentAST* {
  auto arena = unit_->arena();

  parameter =
      default_template_argument(parameter ? parameter->symbol : nullptr);
  if (!parameter) return nullptr;

  if (auto nonType = ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    if (!nonType->declaration || !nonType->declaration->expression) {
      return nullptr;
    }
    auto expression = nonType->declaration->expression;
    if (isDependent(unit_, expression)) {
      if (argumentsSoFar.empty() || !templateDecl) return nullptr;
      expression = ASTRewriter::substituteDefaultExpression(
          unit_, expression, argumentsSoFar, templateDecl->depth,
          substitutionScope(templateDecl));
      if (!expression) return nullptr;
    }
    auto argument = ExpressionTemplateArgumentAST::create(arena);
    argument->expression = expression;
    return argument;
  }

  if (auto templateType = ast_cast<TemplateTypeParameterAST>(parameter)) {
    if (!templateType->idExpression) return nullptr;
    return templateName(templateType->idExpression->symbol);
  }

  auto typeId = [&]() -> TypeIdAST* {
    if (auto type = ast_cast<TypenameTypeParameterAST>(parameter))
      return type->typeId;
    if (auto constrained = ast_cast<ConstraintTypeParameterAST>(parameter))
      return constrained->typeId;
    return nullptr;
  }();

  if (!typeId) return nullptr;

  if (!typeId->type || isDependent(unit_, typeId)) {
    if (argumentsSoFar.empty() || !templateDecl) return nullptr;
    auto substituted = ASTRewriter::substituteDefaultTypeId(
        unit_, typeId, argumentsSoFar, templateDecl->depth,
        substitutionScope(templateDecl));
    if (!substituted || !substituted->type ||
        type_cast<UnresolvedNameType>(substituted->type)) {
      return nullptr;
    }
    typeId = substituted;
  }

  auto argument = TypeTemplateArgumentAST::create(arena);
  argument->typeId = typeId;
  return argument;
}

auto TemplateArguments::complete(
    Symbol* templateSymbol, List<TemplateArgumentAST*>* writtenArguments) const
    -> List<TemplateArgumentAST*>* {
  if (!templateSymbol) return writtenArguments;

  auto templateDecl = template_declaration_of(templateSymbol);
  if (!templateDecl) return writtenArguments;

  auto arity = TemplateArity::of(templateDecl);
  if (arity.hasParameterPack) return writtenArguments;
  if (hasPackExpansion(writtenArguments)) return writtenArguments;

  const auto argc = count(writtenArguments);
  if (argc >= arity.maxArgs) return writtenArguments;
  if (argc < arity.minArgs) return writtenArguments;

  auto arena = unit_->arena();
  List<TemplateArgumentAST*>* completed = nullptr;
  auto tail = &completed;

  for (auto argument : ListView{writtenArguments}) {
    *tail = make_list_node<TemplateArgumentAST>(arena, argument);
    tail = &(*tail)->next;
  }

  int index = 0;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    if (index++ < argc) continue;

    auto argument = defaultArgument(templateDecl, parameter, {});
    if (!argument) return writtenArguments;

    *tail = make_list_node<TemplateArgumentAST>(arena, argument);
    tail = &(*tail)->next;
  }

  return completed;
}

struct Substitution::CollectRawTemplateArgument {
  Substitution& subst;

  [[nodiscard]] auto isInTemplateScope(Symbol* symbol) -> bool {
    return isEnclosedInDependentTemplate(subst.unit_, symbol->parent(),
                                         /*stopAtConcreteSpecialization=*/true);
  }

  auto operator()(ExpressionTemplateArgumentAST* ast)
      -> std::optional<CollectedArgument>;

  auto operator()(TypeTemplateArgumentAST* ast)
      -> std::optional<CollectedArgument>;
};

[[nodiscard]] auto injectedClassNameAsTemplate(TemplateArgumentAST* argument)
    -> Symbol* {
  auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument);
  if (!typeArgument || !typeArgument->typeId) return nullptr;

  for (auto spec : ListView{typeArgument->typeId->typeSpecifierList}) {
    auto named = ast_cast<NamedTypeSpecifierAST>(spec);
    if (!named) continue;
    if (!ast_cast<NameIdAST>(named->unqualifiedId)) return nullptr;

    auto classSymbol = symbol_cast<ClassSymbol>(named->symbol);
    if (auto injected = symbol_cast<InjectedClassNameSymbol>(named->symbol))
      classSymbol = injected->classSymbol();
    if (!classSymbol) return nullptr;
    if (classSymbol->isSpecialization())
      return classSymbol->primaryTemplateSymbol();
    if (classSymbol->templateParameters()) return classSymbol;
    return nullptr;
  }

  return nullptr;
}

struct Substitution::MakeDefaultTemplateArgument {
  Substitution& subst;

  [[nodiscard]] auto control() -> Control* { return subst.unit_->control(); }

  auto operator()(TemplateTypeParameterAST* parameter)
      -> std::optional<TemplateArgument>;

  auto operator()(NonTypeTemplateParameterAST* parameter)
      -> std::optional<TemplateArgument>;

  auto operator()(TypenameTypeParameterAST* parameter)
      -> std::optional<TemplateArgument>;

  auto operator()(ConstraintTypeParameterAST* parameter)
      -> std::optional<TemplateArgument>;
};

auto Substitution::MakeDefaultTemplateArgument::operator()(
    TemplateTypeParameterAST* parameter) -> std::optional<TemplateArgument> {
  if (!parameter->idExpression || !parameter->idExpression->symbol) {
    subst.maybeReportMissingTemplateArgument(parameter->firstSourceLocation());
    return std::nullopt;
  }
  return parameter->idExpression->symbol;
}

auto Substitution::MakeDefaultTemplateArgument::operator()(
    NonTypeTemplateParameterAST* parameter) -> std::optional<TemplateArgument> {
  if (!parameter->declaration || !parameter->declaration->expression) {
    subst.maybeReportMissingTemplateArgument(parameter->firstSourceLocation());
    return std::nullopt;
  }

  const Type* declaredType = parameter->declaration->type;
  auto typeId = declaredTypeId(subst.unit_->arena(), parameter->declaration);

  if (declaredType && isDependent(subst.unit_, typeId) &&
      !subst.templateArguments_.empty() && subst.templateDecl_) {
    auto substituted = ASTRewriter::substituteDefaultTypeId(
        subst.unit_, typeId, subst.templateArguments_,
        subst.templateDecl_->depth, substitutionScope(subst.templateDecl_));

    if (!substituted || !substituted->type) {
      subst.maybeReportDefaultArgumentSubstitutionFailure(
          parameter->firstSourceLocation());
      return std::nullopt;
    }

    if (type_cast<UnresolvedNameType>(substituted->type)) return std::nullopt;

    declaredType = substituted->type;
  }

  auto expression = parameter->declaration->expression;

  if (isDependent(subst.unit_, expression) &&
      !subst.templateArguments_.empty() && subst.templateDecl_) {
    expression = ASTRewriter::substituteDefaultExpression(
        subst.unit_, expression, subst.templateArguments_,
        subst.templateDecl_->depth, substitutionScope(subst.templateDecl_));

    if (!expression) {
      subst.maybeReportDefaultArgumentSubstitutionFailure(
          parameter->firstSourceLocation());
      return std::nullopt;
    }
  }

  auto interp = ASTInterpreter{subst.unit_};
  auto value = interp.evaluate(expression);

  if (!value.has_value() && !isDependent(subst.unit_, expression)) {
    subst.maybeReportInvalidConstantExpression(
        parameter->firstSourceLocation());

    return std::nullopt;
  }

  const Type* argumentType = declaredType;
  if (!argumentType && expression) argumentType = expression->type;

  if (!argumentType) {
    if (isDependent(subst.unit_, expression)) return std::nullopt;

    subst.maybeReportMalformedTemplateArgument(
        parameter->firstSourceLocation());

    return std::nullopt;
  }

  return subst.symbolOf({.type = argumentType,
                         .initializer = expression,
                         .isConstexpr = true,
                         .value = value});
}

auto Substitution::MakeDefaultTemplateArgument::operator()(
    TypenameTypeParameterAST* parameter) -> std::optional<TemplateArgument> {
  const auto loc = parameter->firstSourceLocation();

  if (!parameter->typeId) {
    subst.error(loc, "missing default template argument");
    return std::nullopt;
  }

  auto typeId = parameter->typeId;

  if ((!typeId->type || isDependent(subst.unit_, typeId)) &&
      !subst.templateArguments_.empty() && subst.templateDecl_) {
    auto substituted = ASTRewriter::substituteDefaultTypeId(
        subst.unit_, typeId, subst.templateArguments_,
        subst.templateDecl_->depth, substitutionScope(subst.templateDecl_));
    if (!substituted || !substituted->type) {
      subst.maybeReportDefaultArgumentSubstitutionFailure(
          parameter->firstSourceLocation());
      return std::nullopt;
    }
    typeId = substituted;
  }

  if (!typeId->type) {
    if (!isDependent(subst.unit_, parameter->typeId))
      subst.error(loc, "missing default template argument");
    return std::nullopt;
  }

  return control()->getTypeArgumentSymbol(typeId->type);
}

auto Substitution::MakeDefaultTemplateArgument::operator()(
    ConstraintTypeParameterAST* parameter) -> std::optional<TemplateArgument> {
  if (!parameter->typeId || !parameter->typeId->type) {
    subst.maybeReportMissingTemplateArgument(parameter->firstSourceLocation());
    return std::nullopt;
  }

  auto typeId = parameter->typeId;

  if (isDependent(subst.unit_, typeId) && !subst.templateArguments_.empty() &&
      subst.templateDecl_) {
    auto substituted = ASTRewriter::substituteDefaultTypeId(
        subst.unit_, typeId, subst.templateArguments_,
        subst.templateDecl_->depth, substitutionScope(subst.templateDecl_));
    if (!substituted || !substituted->type) {
      subst.maybeReportDefaultArgumentSubstitutionFailure(
          parameter->firstSourceLocation());
      return std::nullopt;
    }
    typeId = substituted;
  }

  return control()->getTypeArgumentSymbol(typeId->type);
}

auto Substitution::CollectRawTemplateArgument::operator()(
    ExpressionTemplateArgumentAST* ast) -> std::optional<CollectedArgument> {
  auto loc = ast->firstSourceLocation();

  auto expression = ast->expression;

  if (!expression) {
    subst.maybeReportMalformedTemplateArgument(loc);
    return std::nullopt;
  }

  auto unit = subst.unit_;
  auto control = unit->control();

  auto interp = ASTInterpreter{unit};

  const auto isDependent = isDependentTemplateArgument(subst.unit_, ast);

  auto value = isDependent ? std::nullopt : interp.evaluate(expression);

  if (!value.has_value()) {
    if (isDependent) {
      if (auto idExpr = ast_cast<IdExpressionAST>(expression)) {
        if (auto nttp = symbol_cast<NonTypeParameterSymbol>(idExpr->symbol)) {
          return nttp;
        }
        if (auto var = symbol_cast<VariableSymbol>(idExpr->symbol);
            var && !var->parent()) {
          return var;
        }
      }
      return NonTypeArgumentValue{.type = expression->type,
                                  .initializer = expression};
    }

    if (subst.valueDependsOnParameterType(expression)) {
      return NonTypeArgumentValue{.type = expression->type,
                                  .initializer = expression};
    }

    subst.maybeReportInvalidConstantExpression(loc);

    return std::nullopt;
  }

  auto argumentType = expression->type;

  if (argumentType && !subst.unit_->typeTraits().is_scalar(argumentType)) {
    argumentType = subst.unit_->typeTraits().add_pointer(expression->type);
  }

  return NonTypeArgumentValue{.type = argumentType,
                              .initializer = expression,
                              .isConstexpr = true,
                              .value = value};
}

auto Substitution::CollectRawTemplateArgument::operator()(
    TypeTemplateArgumentAST* ast) -> std::optional<CollectedArgument> {
  if (!ast->typeId) {
    return std::nullopt;
  }

  auto loc = ast->firstSourceLocation();

  auto unit = subst.unit_;
  auto control = unit->control();

  for (auto spec : ListView{ast->typeId->typeSpecifierList}) {
    auto named = ast_cast<NamedTypeSpecifierAST>(spec);
    if (!named) continue;
    if (auto pack = symbol_cast<ParameterPackSymbol>(named->symbol))
      return pack;
    if (auto variable = symbol_cast<VariableSymbol>(named->symbol);
        variable && variable->isConstexpr()) {
      return variable;
    }
    if (!ast_cast<NameIdAST>(named->unqualifiedId)) break;
    if (auto alias = symbol_cast<TypeAliasSymbol>(named->symbol)) {
      if (alias->templateParameters()) return alias;
    }
    if (auto classSymbol = symbol_cast<ClassSymbol>(named->symbol)) {
      if (classSymbol->templateParameters()) return classSymbol;
    }
    if (auto templateParameter =
            symbol_cast<TemplateTypeParameterSymbol>(named->symbol)) {
      if (!cxx::isPackExpansion(ast->typeId)) return templateParameter;
      auto templateArgument = control->newTypeAliasSymbol(nullptr, {});
      templateArgument->setType(
          control->getPackExpansionType(templateParameter->type()));
      templateArgument->setExpansionTypeId(ast->typeId);
      return templateArgument;
    }
    break;
  }

  if (!ast->typeId->type) {
    if (isDependentTemplateArgument(subst.unit_, ast)) {
      auto templateArgument = control->newTypeAliasSymbol(nullptr, {});
      return templateArgument;
    }
    subst.maybeReportMalformedTemplateArgument(loc);
    return std::nullopt;
  }

  auto type = ast->typeId->type;
  if (cxx::isPackExpansion(ast->typeId))
    type = control->getPackExpansionType(type);

  return control->getTypeArgumentSymbol(type);
}

Substitution::Substitution(TranslationUnit* unit,
                           TemplateDeclarationAST* templateDecl,
                           List<TemplateArgumentAST*>* templateArgumentList,
                           bool argsComplete, bool fillDefaults)
    : unit_(unit),
      templateDecl_(templateDecl),
      templateArgumentList_(templateArgumentList),
      argsComplete_(argsComplete),
      fillDefaults_(fillDefaults) {
  doMake();
}

Substitution::Substitution(TranslationUnit* unit,
                           List<TemplateArgumentAST*>* templateArgumentList)
    : unit_(unit), templateArgumentList_(templateArgumentList) {}

auto Substitution::writtenTemplateArguments(
    TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList)
    -> std::optional<std::vector<TemplateArgument>> {
  Substitution subst{unit, templateArgumentList};
  if (!subst.collectWrittenArguments()) return std::nullopt;
  std::vector<TemplateArgument> arguments;
  arguments.reserve(subst.writtenArguments_.size());
  for (const auto& written : subst.writtenArguments_)
    arguments.push_back(subst.argumentSymbol(written.argument));
  return arguments;
}

auto Substitution::collectWrittenArguments() -> bool {
  const auto view = ListView{templateArgumentList_};
  writtenArguments_.reserve(std::ranges::distance(view));
  for (auto argument : view) {
    auto arg = visit(CollectRawTemplateArgument{*this}, argument);
    if (!arg.has_value()) return false;
    writtenArguments_.push_back(
        {.argument = *arg,
         .node = argument,
         .isPackExpansion = TemplateArguments::isPackExpansion(argument)});
  }
  return true;
}

auto Substitution::make(TranslationUnit* unit,
                        TemplateDeclarationAST* templateDecl,
                        List<TemplateArgumentAST*>* templateArgumentList,
                        bool argsComplete) -> std::optional<Substitution> {
  Substitution subst{unit, templateDecl, templateArgumentList, argsComplete};
  if (subst.hadError_) return std::nullopt;
  return std::optional<Substitution>{std::move(subst)};
}

auto Substitution::makePartial(TranslationUnit* unit,
                               TemplateDeclarationAST* templateDecl,
                               List<TemplateArgumentAST*>* templateArgumentList)
    -> std::optional<Substitution> {
  Substitution subst{unit, templateDecl, templateArgumentList, false, false};
  if (subst.hadError_) return std::nullopt;
  return std::optional<Substitution>{std::move(subst)};
}

void Substitution::doMake() {
  if (!templateDecl_) {
    cxx_runtime_error("no template declaration");
  }

  auto control = unit_->control();

  if (!collectWrittenArguments()) return;

  std::vector<TemplateParameterAST*> parameters;
  for (auto parameter : ListView{templateDecl_->templateParameterList}) {
    parameters.push_back(parameter);
  }

  const int paramCount = static_cast<int>(parameters.size());
  const int argCount = static_cast<int>(writtenArguments_.size());

  int packIndex = -1;
  int packSize = 0;

  for (int i = 0; i < paramCount; ++i) {
    if (!isPackParameter(parameters[i])) continue;
    packIndex = i;

    if (argsComplete_) {
      int nonPackCount = 0;
      for (int j = 0; j < paramCount; ++j)
        if (!isPackParameter(parameters[j])) ++nonPackCount;
      packSize = std::max(0, argCount - nonPackCount);
      break;
    }

    int trailingRequired = 0;
    for (int j = i + 1; j < paramCount; ++j) {
      if (isPackParameter(parameters[j])) continue;
      if (hasDefaultTemplateArgument(parameters[j])) continue;
      ++trailingRequired;
    }

    int availableForPack = argCount - packIndex - trailingRequired;
    packSize = std::max(0, availableForPack);
    break;
  }

  int argumentIndex = 0;
  bool argumentCountIsKnown = true;

  auto deducedPackAt = [&](int index) -> ParameterPackSymbol* {
    if (index >= argCount) return nullptr;
    auto symbol = std::get_if<Symbol*>(&writtenArguments_[index].argument);
    if (!symbol) return nullptr;
    return symbol_cast<ParameterPackSymbol>(*symbol);
  };

  for (int i = 0; i < paramCount; ++i) {
    auto parameter = parameters[i];

    if (isPackParameter(parameter)) {
      if (auto deducedPack = deducedPackAt(argumentIndex)) {
        ++argumentIndex;
        templateArguments_.push_back(deducedPack);
        continue;
      }
      if (argumentIndex >= argCount) {
        templateArguments_.push_back(control->getPackArgumentSymbol({}));
        continue;
      }
    }

    if (i == packIndex) {
      std::vector<Symbol*> elements;
      elements.reserve(packSize);
      auto nonTypeParam = ast_cast<NonTypeTemplateParameterAST>(parameter);

      for (int k = 0; k < packSize && argumentIndex < argCount; ++k) {
        if (!checkArgumentKind(parameter, argumentIndex)) return;
        auto symbol = normalizeNonTypeArgument(
            nonTypeParam, argumentFor(parameter, argumentIndex++));
        if (hadError_) return;
        elements.push_back(symbol);
      }

      templateArguments_.push_back(control->getPackArgumentSymbol(elements));
      continue;
    }

    if (argumentIndex < argCount) {
      if (writtenArguments_[argumentIndex].isPackExpansion)
        argumentCountIsKnown = false;
      if (!checkArgumentKind(parameter, argumentIndex)) return;
      auto nonTypeParam = ast_cast<NonTypeTemplateParameterAST>(parameter);
      auto symbol = normalizeNonTypeArgument(
          nonTypeParam, argumentFor(parameter, argumentIndex++));
      if (hadError_) return;
      templateArguments_.push_back(symbol);
      continue;
    }

    if (!fillDefaults_ || !argumentCountIsKnown) break;

    if (auto defaultArg = getDefaultTemplateArgument(parameter)) {
      templateArguments_.push_back(defaultArg.value());
      continue;
    }

    hadError_ = true;
    return;
  }
}

auto Substitution::argumentFor(TemplateParameterAST* parameter, int index) const
    -> CollectedArgument {
  const auto& collected = writtenArguments_[index].argument;
  if (ast_cast<TemplateTypeParameterAST>(parameter)) {
    if (auto templateName =
            injectedClassNameAsTemplate(writtenArguments_[index].node))
      return templateName;
    return collected;
  }
  auto symbol = std::get_if<Symbol*>(&collected);
  if (!symbol) return collected;
  if (auto classTemplate =
          symbol_cast<ClassSymbol>(template_name_symbol(*symbol)))
    return injectedClassNameAsType(classTemplate);
  return collected;
}

auto Substitution::argumentSymbol(const CollectedArgument& argument) const
    -> Symbol* {
  if (auto symbol = std::get_if<Symbol*>(&argument)) return *symbol;
  return symbolOf(std::get<NonTypeArgumentValue>(argument));
}

auto Substitution::valueOf(VariableSymbol* variable) const
    -> NonTypeArgumentValue {
  NonTypeArgumentValue result{.type = variable->type(),
                              .initializer = variable->initializer(),
                              .isConstexpr = variable->isConstexpr(),
                              .value = variable->constValue()};
  if (!result.initializer) result.initializer = spelledInitializer(result);
  return result;
}

auto Substitution::spelledInitializer(const NonTypeArgumentValue& value) const
    -> ExpressionAST* {
  if (!isSpellableIntegerConstant(value)) return nullptr;
  return TemplateArguments{unit_}.integerLiteralExpression(
      std::get<ConstInt>(*value.value), value.type);
}

auto Substitution::isSpellableIntegerConstant(const NonTypeArgumentValue& value)
    -> bool {
  if (!value.type || !value.value) return false;
  auto integer = std::get_if<ConstInt>(&*value.value);
  return integer && integer->magnitudeFitsInUIntMax();
}

auto Substitution::symbolOf(const NonTypeArgumentValue& value) const
    -> Symbol* {
  auto control = unit_->control();

  if (isSpellableIntegerConstant(value))
    return control->getConstantArgumentSymbol(value.type, *value.value);

  auto symbol = control->newVariableSymbol(nullptr, {});
  symbol->setInitializer(value.initializer);
  symbol->setConstexpr(value.isConstexpr);
  symbol->setConstValue(value.value);
  symbol->setType(value.type);
  return symbol;
}

auto Substitution::checkArgumentKind(TemplateParameterAST* parameter, int index)
    -> bool {
  auto argument = writtenArguments_[index].node;
  if (matchesTemplateParameterKind(parameter, argument)) return true;
  error(argument->firstSourceLocation(),
        "template argument does not match the form of its template "
        "parameter");
  return false;
}

auto Substitution::injectedClassNameAsType(ClassSymbol* classTemplate) const
    -> Symbol* {
  return unit_->control()->getTypeArgumentSymbol(classTemplate->type());
}

void Substitution::maybeReportInvalidConstantExpression(SourceLocation loc) {
  error(loc, "template argument is not a constant expression");
}

void Substitution::maybeReportDefaultArgumentSubstitutionFailure(
    SourceLocation loc) {
  if (hasDependentArguments()) return;
  error(loc, "substitution failure in default template argument");
}

auto Substitution::hasDependentArguments() const -> bool {
  return std::ranges::any_of(
      templateArguments_, [&](const TemplateArgument& argument) {
        return isDependentTemplateArgument(unit_, argument);
      });
}

void Substitution::maybeReportMalformedTemplateArgument(SourceLocation loc) {
  error(loc, "malformed template argument");
}

void Substitution::maybeReportMissingTemplateArgument(SourceLocation loc) {
  error(loc, "missing template argument");
}

void Substitution::error(SourceLocation loc, std::string message) {
  hadError_ = true;
  auto unit = unit_;
  if (!unit->config().checkTypes) return;
  unit->error(loc, std::move(message));
}

void Substitution::warning(SourceLocation loc, std::string message) {
  auto unit = unit_;
  if (!unit->config().checkTypes) return;
  unit->warning(loc, std::move(message));
}

auto Substitution::substitutedNonTypeParameterType(
    NonTypeTemplateParameterAST* parameter) -> std::optional<const Type*> {
  if (!parameter->declaration) return nullptr;

  if (!parameter->declaration->type) return nullptr;
  auto typeId = declaredTypeId(unit_->arena(), parameter->declaration);
  if (!isDependent(unit_, typeId)) return typeId->type;
  if (templateArguments_.empty() || !templateDecl_) return typeId->type;

  auto substituted = ASTRewriter::substituteDefaultTypeId(
      unit_, typeId, templateArguments_, templateDecl_->depth,
      substitutionScope(templateDecl_));

  if (!substituted || !substituted->type) {
    error(parameter->firstSourceLocation(),
          "substitution failure in the type of a non-type template "
          "parameter");
    return std::nullopt;
  }

  return substituted->type;
}

auto Substitution::normalizeNonTypeArgument(
    NonTypeTemplateParameterAST* parameter, const CollectedArgument& argument)
    -> Symbol* {
  if (!parameter) return argumentSymbol(argument);

  auto parameterType = substitutedNonTypeParameterType(parameter);
  if (!parameterType) return argumentSymbol(argument);

  auto unit = unit_;

  NonTypeArgumentValue normalized;

  if (auto written = std::get_if<NonTypeArgumentValue>(&argument)) {
    normalized = *written;
  } else {
    auto symbol = std::get<Symbol*>(argument);
    auto variableArgument = symbol_cast<VariableSymbol>(symbol);

    if (!variableArgument) {
      auto typeAliasArgument = symbol_cast<TypeAliasSymbol>(symbol);
      if (!typeAliasArgument || !typeAliasArgument->type()) return symbol;
      if (typeAliasArgument->templateParameters()) return symbol;
      if (!isDependent(unit, typeAliasArgument->type())) return symbol;
      if (type_cast<ClassType>(typeAliasArgument->type())) return symbol;

      NonTypeArgumentValue placeholder;
      placeholder.type = typeAliasArgument->type();
      if (parameter->declaration && parameter->declaration->type)
        placeholder.type = parameter->declaration->type;
      return symbolOf(placeholder);
    }

    normalized = valueOf(variableArgument);
  }

  const Type* targetType = normalized.type;

  if (!type_cast<TypeParameterType>(targetType) &&
      !type_cast<TemplateTypeParameterType>(targetType)) {
    if (parameter->declaration && parameter->declaration->type) {
      const Type* declaredType = parameter->declaration->type;
      if (containsPlaceholderType(declaredType)) {
        auto checker = TypeChecker{unit};
        if (auto initializer = normalized.initializer) {
          targetType = checker.deducePlaceholderType(declaredType, initializer);
        } else {
          targetType = checker.deduceAutoType(declaredType, targetType);
        }
      } else if (*parameterType &&
                 !type_cast<UnresolvedNameType>(*parameterType) &&
                 !isDependent(unit, *parameterType)) {
        targetType = *parameterType;
      }
    }
  }

  normalized.type = targetType;

  convertNonTypeArgument(normalized, targetType);

  if (normalized.value && !isConstexprRepresentable(*normalized.value)) {
    maybeReportInvalidConstantExpression(
        normalized.initializer->firstSourceLocation());
  }

  if (lacksConvertedValue(normalized)) {
    maybeReportInvalidConstantExpression(
        normalized.initializer->firstSourceLocation());
  }

  return symbolOf(normalized);
}

auto Substitution::valueDependsOnParameterType(ExpressionAST* expression) const
    -> bool {
  if (!expression->type) return false;
  if (is_glvalue(expression)) return true;
  return unit_->typeTraits().is_class(expression->type);
}

namespace {

[[nodiscard]] auto designatesObject(Symbol* referent) -> bool {
  if (symbol_cast<VariableSymbol>(referent)) return true;
  auto field = symbol_cast<FieldSymbol>(referent);
  return field && field->isStatic();
}

}  // namespace

auto Substitution::isConstexprRepresentable(const ConstValue& value) const
    -> bool {
  if (auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value)) {
    if (!*address) return false;
    if ((*address)->stringLiteral() || (*address)->typeInfoFor()) return false;
    auto referent = (*address)->symbol();
    if (!referent || (*address)->owner()) return true;
    if (!designatesObject(referent)) return true;
    return has_static_storage_duration(referent);
  }
  if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
    if (!*object) return false;
    return std::ranges::all_of((*object)->members(), [&](const auto& member) {
      return isConstexprRepresentable(member.value);
    });
  }
  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&value)) {
    if (!*list) return false;
    return std::ranges::all_of((*list)->elements, [&](const auto& element) {
      return isConstexprRepresentable(std::get<0>(element));
    });
  }
  return true;
}

auto Substitution::lacksConvertedValue(
    const NonTypeArgumentValue& argument) const -> bool {
  if (argument.value) return false;
  auto initializer = argument.initializer;
  if (!initializer || !valueDependsOnParameterType(initializer)) return false;
  return !isDependent(unit_, initializer);
}

void Substitution::bindReferenceArgument(NonTypeArgumentValue& argument,
                                         const Type* targetType) {
  argument.value = std::nullopt;

  auto expression = argument.initializer;
  if (!is_glvalue(expression)) return;

  auto traits = unit_->typeTraits();
  if (!traits.is_reference_compatible(traits.remove_reference(targetType),
                                      expression->type))
    return;

  auto converted = expression;
  if (!TypeChecker{unit_}.implicit_conversion(converted, targetType)) return;

  auto address = ASTInterpreter{unit_}.evaluateAddress(converted);
  if (!address.has_value()) return;

  argument.initializer = converted;
  argument.isConstexpr = true;
  argument.value = std::move(address);
}

void Substitution::convertNonTypeArgument(NonTypeArgumentValue& argument,
                                          const Type* targetType) {
  if (!targetType) return;

  auto traits = unit_->typeTraits();
  if (isDependent(unit_, targetType)) return;

  auto expression = argument.initializer;
  if (!expression || !expression->type) return;
  if (isDependent(unit_, expression)) return;

  if (traits.is_reference(targetType)) {
    bindReferenceArgument(argument, targetType);
    return;
  }

  if (traits.is_same(traits.remove_cv(expression->type),
                     traits.remove_cv(targetType)))
    return;

  StandardConversion conversions{unit_};
  auto converted = expression;
  if (!conversions.convertImplicitly(converted, targetType)) {
    error(expression->firstSourceLocation(),
          std::format("template argument of type '{}' cannot be converted to "
                      "'{}'",
                      to_string(expression->type), to_string(targetType)));
    return;
  }

  if (traits.is_narrowing_list_element(expression, targetType)) {
    error(expression->firstSourceLocation(),
          std::format("narrowing conversion from '{}' to '{}' in converted "
                      "constant expression",
                      to_string(expression->type), to_string(targetType)));
    return;
  }

  auto interp = ASTInterpreter{unit_};
  auto value = interp.evaluate(converted);
  if (!value.has_value()) return;

  argument.initializer = converted;
  argument.isConstexpr = true;
  argument.value = value;
}

auto Substitution::getDefaultTemplateArgument(TemplateParameterAST* parameter)
    -> std::optional<TemplateArgument> {
  if (!parameter) return std::nullopt;
  auto provider = default_template_argument(parameter->symbol);
  if (!provider) return std::nullopt;
  return visit(MakeDefaultTemplateArgument{*this}, provider);
}
}  // namespace cxx
