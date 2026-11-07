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
#include <cxx/ast_rewriter.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>

namespace cxx {
namespace {

struct TypeEntity {
  [[nodiscard]] auto operator()(const ClassType* type) const -> Symbol* {
    return type->symbol();
  }
  [[nodiscard]] auto operator()(const EnumType* type) const -> Symbol* {
    return type->symbol();
  }
  [[nodiscard]] auto operator()(const ScopedEnumType* type) const -> Symbol* {
    return type->symbol();
  }
  [[nodiscard]] auto operator()(const QualType* type) const -> Symbol* {
    return visit(*this, type->elementType());
  }
  [[nodiscard]] auto operator()(const Type*) const -> Symbol* {
    return nullptr;
  }
};

struct EnumerationScope {
  [[nodiscard]] auto operator()(EnumSymbol* symbol) const -> ScopeSymbol* {
    return symbol->isDefined() ? symbol : nullptr;
  }
  [[nodiscard]] auto operator()(ScopedEnumSymbol* symbol) const
      -> ScopeSymbol* {
    return symbol->isDefined() ? symbol : nullptr;
  }
  [[nodiscard]] auto operator()(Symbol*) const -> ScopeSymbol* {
    return nullptr;
  }
};

struct DeclarationScope {
  [[nodiscard]] auto operator()(FunctionSymbol* symbol) const -> ScopeSymbol* {
    if (auto scope = namespace_scope_of(symbol)) return scope;
    return symbol->parent();
  }
  [[nodiscard]] auto operator()(Symbol* symbol) const -> ScopeSymbol* {
    return symbol->parent();
  }
};

struct LayoutSubject {
  [[nodiscard]] auto operator()(VariableSymbol* symbol) const -> const Type* {
    if (symbol->isTemplatePattern() || symbol->isStructuredBinding())
      return nullptr;
    return symbol->type();
  }
  [[nodiscard]] auto operator()(ParameterSymbol* symbol) const -> const Type* {
    return symbol->type();
  }
  [[nodiscard]] auto operator()(FieldSymbol* symbol) const -> const Type* {
    return symbol->type();
  }
  [[nodiscard]] auto operator()(BaseClassSymbol* symbol) const -> const Type* {
    auto base = resolved_base_class(symbol);
    return base ? base->type() : nullptr;
  }
  [[nodiscard]] auto operator()(Symbol*) const -> const Type* {
    return nullptr;
  }
};

class MemberAccess {
 public:
  MemberAccess(TranslationUnit* unit, ScopeSymbol* scope,
               ClassSymbol* designating)
      : unit_(unit),
        scope_(scope),
        designating_(designating),
        access_(unit, scope) {}

  [[nodiscard]] auto operator()(Symbol* symbol) -> std::optional<bool> {
    auto base = symbol_cast<BaseClassSymbol>(symbol);
    if (!base && !is_class_member(symbol)) return true;
    auto owner = symbol->enclosingClass();
    if (!owner || !owner->resolvedDefinition()->isComplete())
      return std::nullopt;
    owner = owner->resolvedDefinition();
    auto naming = designating_ ? designating_ : owner;
    auto traits = unit_->typeTraits();
    if (base) {
      if (naming != owner && !traits.is_base_of(owner->type(), naming->type()))
        return false;
    } else if (!traits.is_member_of_object_type(naming->type(), symbol)) {
      return false;
    }
    if (!scope_) return true;
    if (base)
      return access_.isAccessibleBaseClass(naming, resolved_base_class(base));
    return access_.isAccessibleDeclaration(symbol, naming);
  }

 private:
  TranslationUnit* unit_;
  ScopeSymbol* scope_;
  ClassSymbol* designating_;
  AccessContext access_;
};

class CompleteClassContext final : public ASTVisitor {
 public:
  explicit CompleteClassContext(SourceLocation point) : point_(point) {}

  [[nodiscard]] auto contains(AST* ast) const -> bool {
    return ast && ast->firstSourceLocation() <= point_ &&
           point_ < ast->lastSourceLocation();
  }

  [[nodiscard]] auto found() const -> bool { return found_; }

  [[nodiscard]] auto preVisit(AST* ast) -> bool override {
    return !found_ && contains(ast);
  }

  void visit(FunctionDefinitionAST* ast) override {
    found_ = contains(ast->functionBody);
    if (!found_) ASTVisitor::visit(ast);
  }

  void visit(LambdaExpressionAST* ast) override {
    found_ = contains(ast->statement);
    if (!found_) ASTVisitor::visit(ast);
  }

  void visit(ParameterDeclarationAST* ast) override {
    found_ = contains(ast->expression);
    if (!found_) ASTVisitor::visit(ast);
  }

  void visit(NoexceptSpecifierAST* ast) override {
    found_ = contains(ast->expression);
  }

  void visit(InitDeclaratorAST* ast) override {
    auto field = symbol_cast<FieldSymbol>(ast->symbol);
    found_ = field && !field->isStatic() && contains(ast->initializer);
    if (!found_) ASTVisitor::visit(ast);
  }

  void visit(TemplateTypeParameterAST* ast) override {
    found_ = contains(ast->idExpression);
    if (!found_) ASTVisitor::visit(ast);
  }

  void visit(TypenameTypeParameterAST* ast) override {
    found_ = contains(ast->typeId);
    if (!found_) ASTVisitor::visit(ast);
  }

  void visit(ConstraintTypeParameterAST* ast) override {
    found_ = contains(ast->typeId);
    if (!found_) ASTVisitor::visit(ast);
  }

 private:
  SourceLocation point_;
  bool found_ = false;
};

[[nodiscard]] auto membersOfPoint(ScopeSymbol* scope, SourceLocation point)
    -> SourceLocation {
  auto result = point;
  for (; scope; scope = scope->parent()) {
    auto cls = symbol_cast<ClassSymbol>(scope);
    if (!cls) continue;
    auto ast = ast_cast<ClassSpecifierAST>(cls->declaration());
    if (!ast || !ast->rbraceLoc) continue;
    CompleteClassContext context{point};
    context.accept(ast);
    if (context.found()) result = ast->rbraceLoc.next();
  }
  return result;
}

struct MemberRepresentation {
  TranslationUnit* unit;
  FunctionSymbol* destructor;

  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> bool {
    return !symbol->isClosureType() && !symbol->isSpecialization();
  }
  [[nodiscard]] auto operator()(EnumSymbol*) const -> bool { return true; }
  [[nodiscard]] auto operator()(ScopedEnumSymbol*) const -> bool {
    return true;
  }
  [[nodiscard]] auto operator()(TypeAliasSymbol* symbol) const -> bool {
    return !symbol->isSpecialization();
  }
  [[nodiscard]] auto operator()(ConceptSymbol*) const -> bool { return true; }
  [[nodiscard]] auto operator()(NamespaceSymbol*) const -> bool { return true; }
  [[nodiscard]] auto operator()(NamespaceAliasSymbol*) const -> bool {
    return true;
  }
  [[nodiscard]] auto operator()(FieldSymbol*) const -> bool { return true; }
  [[nodiscard]] auto operator()(VariableSymbol* symbol) const -> bool {
    if (symbol->isSpecialization()) return false;
    if (symbol->isTemplatePattern()) return true;
    return !symbol->isStructuredBinding() &&
           !containsPlaceholderType(symbol->type());
  }
  [[nodiscard]] auto operator()(FunctionSymbol* symbol) const -> bool {
    if (symbol->isSpecialization()) return false;
    if (symbol->isTemplatePattern()) return true;
    if (containsPlaceholderType(symbol->type())) return false;
    if (symbol->isDestructor() && symbol->canonical() != destructor)
      return false;
    return ASTRewriter::evaluateAssociatedConstraints(unit, symbol) == true;
  }
  [[nodiscard]] auto operator()(Symbol*) const -> bool { return false; }
};

struct EligibleMemberDeclaration {
  ScopeSymbol* scope;
  SourceLocation point;

  [[nodiscard]] auto operator()(FunctionSymbol* symbol) const
      -> SourceLocation {
    for (auto declaration : symbol->declarations()) {
      if (declaration->hostScope() != scope || declaration->isFriend())
        continue;
      if (declaration->isImplicitlyDeclared() ||
          declaration->location() < point)
        return declaration->location();
    }
    return {};
  }

  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> SourceLocation {
    for (auto declaration : symbol->declarations()) {
      if (declaration->isFriend()) continue;
      if (auto location = declaration->firstNonFriendDeclarationLocation()) {
        if (declaration->parent() == scope && location < point) return location;
      } else if (auto location = (*this)(static_cast<Symbol*>(declaration))) {
        return location;
      }
    }
    return {};
  }

  [[nodiscard]] auto operator()(Symbol* symbol) const -> SourceLocation {
    if (symbol->parent() != scope || symbol->isHidden()) return {};
    auto location = symbol->location();
    return location < point ? location : SourceLocation{};
  }
};

struct MemberCandidate {
  Symbol* symbol;
  SourceLocation location;
  bool implicit = false;
};

class MemberEnumeration {
 public:
  MemberEnumeration(TranslationUnit* unit, ScopeSymbol* scope,
                    SourceLocation point)
      : unit_(unit),
        scope_(scope),
        point_(point),
        cls_(symbol_cast<ClassSymbol>(scope)),
        destructor_(cls_ ? cls_->destructor() : nullptr) {}

  [[nodiscard]] auto operator()() -> std::vector<MemberCandidate> {
    for (auto member : scope_->members()) append(member);
    if (cls_ && !cls_->isClosureType()) append(cls_->constructorOverloadSet());
    if (cls_) {
      std::ranges::stable_sort(result_,
                               [](const auto& first, const auto& second) {
                                 if (first.implicit != second.implicit)
                                   return !first.implicit;
                                 return first.location < second.location;
                               });
    }
    return std::move(result_);
  }

 private:
  void append(Symbol* member) {
    auto functions = views::declared_functions(member);
    if (!functions.empty()) {
      for (auto function : functions) appendFunction(function);
      return;
    }
    if (cls_ && cls_->isClosureType()) return;
    auto location = visit(EligibleMemberDeclaration{scope_, point_}, member);
    if (!location || !visit(MemberRepresentation{unit_, destructor_}, member))
      return;
    result_.push_back({member, location});
  }

  void appendFunction(FunctionSymbol* function) {
    if (function->inheritedConstructor()) return;
    if (cls_ && cls_->isClosureType()) {
      auto name = name_cast<OperatorId>(function->name());
      if (!name || name->op() != TokenKind::T_LPAREN) return;
    }
    auto location = visit(EligibleMemberDeclaration{scope_, point_}, function);
    if (!location || !visit(MemberRepresentation{unit_, destructor_}, function))
      return;
    result_.push_back({function, location, function->isImplicitlyDeclared()});
  }

  TranslationUnit* unit_;
  ScopeSymbol* scope_;
  SourceLocation point_;
  ClassSymbol* cls_;
  FunctionSymbol* destructor_;
  std::vector<MemberCandidate> result_;
};

struct DataMemberKind {
  bool isStatic;
  [[nodiscard]] auto operator()(FieldSymbol* field) const -> bool {
    if (field->isStatic() != isStatic) return false;
    return !field->isBitField() || field->name();
  }
  [[nodiscard]] auto operator()(VariableSymbol* variable) const -> bool {
    return isStatic && !variable->isTemplatePattern() &&
           !variable->isSpecialization() && !variable->isStructuredBinding();
  }
  [[nodiscard]] auto operator()(Symbol*) const -> bool { return false; }
};

[[nodiscard]] auto concreteFunction(Symbol* symbol) -> FunctionSymbol* {
  auto function = symbol_cast<FunctionSymbol>(symbol);
  return function && !function->isTemplatePattern() ? function : nullptr;
}

[[nodiscard]] auto functionTemplate(Symbol* symbol) -> FunctionSymbol* {
  auto function = symbol_cast<FunctionSymbol>(symbol);
  return function && function->isTemplatePattern() ? function : nullptr;
}

[[nodiscard]] auto plainVariable(Symbol* symbol) -> VariableSymbol* {
  auto variable = symbol_cast<VariableSymbol>(symbol);
  if (!variable || variable->isStructuredBinding()) return nullptr;
  return variable->isTemplatePattern() ? nullptr : variable;
}

[[nodiscard]] auto isOperatorNamed(Symbol* symbol, TokenKind op) -> bool {
  auto name = name_cast<OperatorId>(symbol->name());
  return name && name->op() == op;
}

[[nodiscard]] auto hasAccess(Symbol* symbol, AccessSpecifier access) -> bool {
  if (!symbol) return false;
  if (!symbol_cast<BaseClassSymbol>(symbol) && !is_class_member(symbol))
    return false;
  return symbol->accessSpecifier() == access;
}

template <auto Query>
[[nodiscard]] auto memberOfOwnClass(Symbol* symbol) -> bool {
  auto function = concreteFunction(symbol);
  if (!function) return false;
  auto cls = symbol_cast<ClassSymbol>(function->parent());
  return cls && (cls->*Query)(function);
}

}  // namespace

auto TypeTraits::entity_of(const Type* type) -> Symbol* {
  return type ? visit(TypeEntity{}, type) : nullptr;
}

auto TypeTraits::is_namespace(Symbol* symbol) const -> bool {
  return symbol_cast<NamespaceSymbol>(symbol) || is_namespace_alias(symbol);
}

auto TypeTraits::is_namespace_alias(Symbol* symbol) const -> bool {
  return symbol_cast<NamespaceAliasSymbol>(symbol) != nullptr;
}

auto TypeTraits::is_variable(Symbol* symbol) const -> bool {
  if (plainVariable(symbol) || symbol_cast<ParameterSymbol>(symbol))
    return true;
  auto field = symbol_cast<FieldSymbol>(symbol);
  return field && field->isStatic();
}

auto TypeTraits::is_structured_binding(Symbol* symbol) const -> bool {
  auto variable = symbol_cast<VariableSymbol>(symbol);
  return variable && variable->isStructuredBinding();
}

auto TypeTraits::is_function(Symbol* symbol) const -> bool {
  return concreteFunction(symbol) != nullptr;
}

auto TypeTraits::is_enumerator(Symbol* symbol) const -> bool {
  return symbol_cast<EnumeratorSymbol>(symbol) != nullptr;
}

auto TypeTraits::is_template(Symbol* symbol) const -> bool {
  return is_function_template(symbol) || is_variable_template(symbol) ||
         is_class_template(symbol) || is_alias_template(symbol) ||
         is_concept(symbol);
}

auto TypeTraits::is_function_template(Symbol* symbol) const -> bool {
  return functionTemplate(symbol) != nullptr;
}

auto TypeTraits::is_variable_template(Symbol* symbol) const -> bool {
  auto variable = symbol_cast<VariableSymbol>(symbol);
  return variable && !variable->isStructuredBinding() &&
         variable->isTemplatePattern();
}

auto TypeTraits::is_class_template(Symbol* symbol) const -> bool {
  auto cls = symbol_cast<ClassSymbol>(symbol);
  return cls && cls->isTemplatePattern();
}

auto TypeTraits::is_alias_template(Symbol* symbol) const -> bool {
  auto alias = symbol_cast<TypeAliasSymbol>(symbol);
  return alias && alias->isTemplatePattern();
}

auto TypeTraits::is_concept(Symbol* symbol) const -> bool {
  return symbol_cast<ConceptSymbol>(symbol) != nullptr;
}

auto TypeTraits::is_type_alias(Symbol* symbol) const -> bool {
  auto alias = symbol_cast<TypeAliasSymbol>(symbol);
  return alias && !alias->isTemplatePattern();
}

auto TypeTraits::is_nonstatic_data_member(Symbol* symbol) const -> bool {
  auto field = symbol_cast<FieldSymbol>(symbol);
  if (!field || field->isStatic()) return false;
  return !field->isBitField() || field->name();
}

auto TypeTraits::is_bit_field(Symbol* symbol) const -> bool {
  auto field = symbol_cast<FieldSymbol>(symbol);
  return field && field->isBitField();
}

auto TypeTraits::is_mutable_member(Symbol* symbol) const -> bool {
  auto field = symbol_cast<FieldSymbol>(symbol);
  return field && field->isMutable();
}

auto TypeTraits::has_default_member_initializer(Symbol* symbol) const -> bool {
  auto field = symbol_cast<FieldSymbol>(symbol);
  return field && !field->isStatic() && field->hasInitializer();
}

auto TypeTraits::is_virtual(Symbol* symbol) const -> bool {
  if (auto base = symbol_cast<BaseClassSymbol>(symbol))
    return base->isVirtual();
  auto function = concreteFunction(symbol);
  return function && function->isVirtual();
}

auto TypeTraits::is_pure_virtual(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && function->isPure();
}

auto TypeTraits::is_override(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && !function->overriddenFunctions().empty();
}

auto TypeTraits::is_final(Symbol* symbol) const -> bool {
  if (auto function = concreteFunction(symbol)) return function->isFinal();
  auto cls = symbol_cast<ClassSymbol>(symbol);
  if (!cls || cls->isTemplatePattern()) return false;
  (void)ASTRewriter::ensureCompleteClass(unit_, cls);
  return cls->resolvedDefinition()->isFinal();
}

auto TypeTraits::is_deleted(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && is_deleted_function(function->resolvedDefinition());
}

auto TypeTraits::is_defaulted(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && function->resolvedDefinition()->isDefaulted();
}

auto TypeTraits::is_user_provided(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && is_user_provided(function);
}

auto TypeTraits::is_user_declared(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && !function->isImplicitlyDeclared();
}

auto TypeTraits::is_explicit(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && is_class_member(function) && function->isExplicit();
}

auto TypeTraits::has_static_storage_duration(Symbol* symbol) const -> bool {
  if (!plainVariable(symbol) && !symbol_cast<FieldSymbol>(symbol)) return false;
  return cxx::has_static_storage_duration(symbol);
}

auto TypeTraits::has_thread_storage_duration(Symbol* symbol) const -> bool {
  if (!plainVariable(symbol) && !symbol_cast<FieldSymbol>(symbol)) return false;
  return cxx::has_thread_storage_duration(symbol);
}

auto TypeTraits::has_automatic_storage_duration(Symbol* symbol) const -> bool {
  if (symbol_cast<ParameterSymbol>(symbol)) return true;
  if (!plainVariable(symbol) || !is_block_variable(symbol)) return false;
  return !cxx::has_static_storage_duration(symbol) &&
         !cxx::has_thread_storage_duration(symbol);
}

auto TypeTraits::is_conversion_function(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && name_cast<ConversionFunctionId>(function->name());
}

auto TypeTraits::is_operator_function(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && name_cast<OperatorId>(function->name());
}

auto TypeTraits::is_literal_operator(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && name_cast<LiteralOperatorId>(function->name());
}

auto TypeTraits::is_constructor(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && function->isConstructor();
}

auto TypeTraits::is_destructor(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  return function && function->isDestructor();
}

auto TypeTraits::is_default_constructor(Symbol* symbol) const -> bool {
  return memberOfOwnClass<&ClassSymbol::isDefaultConstructor>(symbol);
}

auto TypeTraits::is_copy_constructor(Symbol* symbol) const -> bool {
  return memberOfOwnClass<&ClassSymbol::isCopyConstructor>(symbol);
}

auto TypeTraits::is_move_constructor(Symbol* symbol) const -> bool {
  return memberOfOwnClass<&ClassSymbol::isMoveConstructor>(symbol);
}

auto TypeTraits::is_assignment(Symbol* symbol) const -> bool {
  return concreteFunction(symbol) &&
         isOperatorNamed(symbol, TokenKind::T_EQUAL);
}

auto TypeTraits::is_copy_assignment(Symbol* symbol) const -> bool {
  return memberOfOwnClass<&ClassSymbol::isCopyAssignmentOperator>(symbol);
}

auto TypeTraits::is_move_assignment(Symbol* symbol) const -> bool {
  return memberOfOwnClass<&ClassSymbol::isMoveAssignmentOperator>(symbol);
}

auto TypeTraits::is_special_member_function(Symbol* symbol) const -> bool {
  auto function = concreteFunction(symbol);
  if (!function || !symbol_cast<ClassSymbol>(function->parent())) return false;
  return function->isDestructor() || is_default_constructor(function) ||
         is_copy_constructor(function) || is_move_constructor(function) ||
         is_copy_assignment(function) || is_move_assignment(function);
}

auto TypeTraits::is_conversion_function_template(Symbol* symbol) const -> bool {
  auto function = functionTemplate(symbol);
  return function && name_cast<ConversionFunctionId>(function->name());
}

auto TypeTraits::is_operator_function_template(Symbol* symbol) const -> bool {
  auto function = functionTemplate(symbol);
  return function && name_cast<OperatorId>(function->name());
}

auto TypeTraits::is_literal_operator_template(Symbol* symbol) const -> bool {
  auto function = functionTemplate(symbol);
  return function && name_cast<LiteralOperatorId>(function->name());
}

auto TypeTraits::is_constructor_template(Symbol* symbol) const -> bool {
  auto function = functionTemplate(symbol);
  return function && function->isConstructor();
}

auto TypeTraits::is_static_member(Symbol* symbol) const -> bool {
  if (auto field = symbol_cast<FieldSymbol>(symbol)) return field->isStatic();
  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    return is_class_member(function) && function->isStatic();
  auto variable = symbol_cast<VariableSymbol>(symbol);
  return variable && !variable->isStructuredBinding() &&
         is_class_member(variable);
}

auto TypeTraits::is_public(Symbol* symbol) const -> bool {
  return hasAccess(symbol, AccessSpecifier::kPublic);
}

auto TypeTraits::is_protected(Symbol* symbol) const -> bool {
  return hasAccess(symbol, AccessSpecifier::kProtected);
}

auto TypeTraits::is_private(Symbol* symbol) const -> bool {
  return hasAccess(symbol, AccessSpecifier::kPrivate);
}

auto TypeTraits::is_namespace_member(Symbol* symbol) const -> bool {
  return symbol && namespace_scope_of(symbol);
}

auto TypeTraits::is_base(Symbol* symbol) const -> bool {
  return symbol_cast<BaseClassSymbol>(symbol) != nullptr;
}

auto TypeTraits::has_template_arguments(Symbol* symbol) const -> bool {
  auto primary = primary_template_of(symbol);
  return primary && is_template(primary);
}

auto TypeTraits::has_parent(Symbol* symbol) const -> bool {
  return symbol && symbol != unit_->globalScope() &&
         !has_c_language_linkage(symbol);
}

namespace {

struct EntityLinkage {
  const TypeTraits& traits;
  LanguageKind language;

  using Linkage = TypeTraits::NameLinkage;

  [[nodiscard]] auto ofNamespaceMember(Symbol* symbol) const -> Linkage {
    if (cxx::has_internal_linkage(symbol, language)) return Linkage::kInternal;
    return Linkage::kExternal;
  }

  [[nodiscard]] auto ofScopedEntity(Symbol* symbol) const -> Linkage {
    auto parent = symbol->parent();
    while (symbol_cast<TemplateParametersSymbol>(parent))
      parent = parent->parent();
    if (auto cls = symbol_cast<ClassSymbol>(parent)) return (*this)(cls);
    if (!parent || !parent->isNamespace()) return Linkage::kNone;
    return ofNamespaceMember(symbol);
  }

  [[nodiscard]] auto operator()(NamespaceSymbol* symbol) const -> Linkage {
    if (!symbol->parent()) return Linkage::kNone;
    if (!symbol->name() || is_in_unnamed_namespace(symbol))
      return Linkage::kInternal;
    return Linkage::kExternal;
  }

  [[nodiscard]] auto operator()(VariableSymbol* symbol) const -> Linkage {
    if (symbol->isStructuredBinding()) return Linkage::kNone;
    if (!is_block_variable(symbol)) return ofScopedEntity(symbol);
    return symbol->isExtern() ? Linkage::kExternal : Linkage::kNone;
  }

  [[nodiscard]] auto operator()(FieldSymbol* symbol) const -> Linkage {
    return symbol->isStatic() ? ofScopedEntity(symbol) : Linkage::kNone;
  }

  [[nodiscard]] auto operator()(FunctionSymbol* symbol) const -> Linkage {
    if (auto scope = namespace_scope_of(symbol))
      return scope->isNamespace() ? ofNamespaceMember(symbol) : Linkage::kNone;
    return ofScopedEntity(symbol);
  }

  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> Linkage {
    if (!symbol->name() || symbol->isClosureType()) return Linkage::kNone;
    return ofScopedEntity(symbol);
  }

  [[nodiscard]] auto operator()(EnumSymbol* symbol) const -> Linkage {
    return symbol->name() ? ofScopedEntity(symbol) : Linkage::kNone;
  }

  [[nodiscard]] auto operator()(ScopedEnumSymbol* symbol) const -> Linkage {
    return symbol->name() ? ofScopedEntity(symbol) : Linkage::kNone;
  }

  [[nodiscard]] auto operator()(TypeAliasSymbol* symbol) const -> Linkage {
    return symbol->isTemplatePattern() ? ofScopedEntity(symbol)
                                       : Linkage::kNone;
  }

  [[nodiscard]] auto operator()(ConceptSymbol* symbol) const -> Linkage {
    return ofScopedEntity(symbol);
  }

  [[nodiscard]] auto operator()(Symbol*) const -> Linkage {
    return Linkage::kNone;
  }
};

struct HasIdentifier {
  const TypeTraits& traits;

  [[nodiscard]] auto operator()(FunctionSymbol* symbol) const -> bool {
    if (symbol->isConstructor() || symbol->isDestructor()) return false;
    if (name_cast<OperatorId>(symbol->name())) return false;
    if (name_cast<ConversionFunctionId>(symbol->name())) return false;
    return !traits.has_template_arguments(symbol);
  }

  [[nodiscard]] auto operator()(BaseClassSymbol* symbol) const -> bool {
    auto base = resolved_base_class(symbol);
    return base && traits.has_identifier(base);
  }

  [[nodiscard]] auto operator()(NamespaceSymbol* symbol) const -> bool {
    return symbol->name() != nullptr;
  }

  [[nodiscard]] auto operator()(TypeAliasSymbol* symbol) const -> bool {
    return symbol->name() && !traits.has_template_arguments(symbol);
  }

  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> bool {
    return symbol->name() && !traits.has_template_arguments(symbol);
  }

  [[nodiscard]] auto operator()(VariableSymbol* symbol) const -> bool {
    return symbol->name() && !traits.has_template_arguments(symbol);
  }

  [[nodiscard]] auto operator()(Symbol* symbol) const -> bool {
    return symbol->name() != nullptr;
  }
};

}  // namespace

auto TypeTraits::linkage_of(Symbol* symbol) const -> NameLinkage {
  if (!symbol) return NameLinkage::kNone;
  return visit(EntityLinkage{*this, unit_->language()}, symbol);
}

auto TypeTraits::has_internal_linkage(Symbol* symbol) const -> bool {
  return linkage_of(symbol) == NameLinkage::kInternal;
}

auto TypeTraits::has_module_linkage(Symbol* symbol) const -> bool {
  return linkage_of(symbol) == NameLinkage::kModule;
}

auto TypeTraits::has_external_linkage(Symbol* symbol) const -> bool {
  return linkage_of(symbol) == NameLinkage::kExternal;
}

auto TypeTraits::has_linkage(Symbol* symbol) const -> bool {
  return linkage_of(symbol) != NameLinkage::kNone;
}

auto TypeTraits::has_c_language_linkage(Symbol* symbol) const -> bool {
  if (!concreteFunction(symbol) && !plainVariable(symbol)) return false;
  return cxx::has_c_language_linkage(symbol);
}

auto TypeTraits::has_identifier(Symbol* symbol) const -> bool {
  return symbol && visit(HasIdentifier{*this}, symbol);
}

auto TypeTraits::identifier_of(Symbol* symbol) const -> std::string {
  if (auto base = symbol_cast<BaseClassSymbol>(symbol))
    return identifier_of(resolved_base_class(base));
  if (auto literal = name_cast<LiteralOperatorId>(symbol->name()))
    return literal->name();
  return to_string(symbol->name());
}

auto TypeTraits::parent_of(Symbol* symbol) const -> Symbol* {
  if (!has_parent(symbol)) return nullptr;
  if (auto enumerator = symbol_cast<EnumeratorSymbol>(symbol))
    return enumerator->parent();
  auto scope = visit(DeclarationScope{}, symbol);
  while (scope && !scope->isClassOrNamespace() &&
         !symbol_cast<FunctionSymbol>(scope))
    scope = scope->parent();
  if (is_consteval_block_call_operator(scope))
    return parent_of(parent_of(scope));
  return scope;
}

auto TypeTraits::is_enumerable(const Type* type) -> bool {
  if (!type) return false;
  if (is_class_or_union(type)) return is_complete(type);
  auto entity = entity_of(type);
  return entity && visit(EnumerationScope{}, entity) != nullptr;
}

auto TypeTraits::enumerators_of(const Type* type) const
    -> std::optional<std::vector<Symbol*>> {
  auto entity = entity_of(type);
  auto scope = entity ? visit(EnumerationScope{}, entity) : nullptr;
  if (!scope) return std::nullopt;
  std::vector<Symbol*> result;
  for (auto member : scope->members())
    if (symbol_cast<EnumeratorSymbol>(member)) result.push_back(member);
  return result;
}

auto TypeTraits::subobject_offset(Symbol* symbol) const
    -> std::optional<std::uint64_t> {
  auto owner = symbol->enclosingClass();
  if (!owner) return std::nullopt;
  auto layout = owner->resolvedDefinition()->layout();
  if (!layout) return std::nullopt;
  if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    if (field->isStatic()) return std::nullopt;
    auto info = layout->getFieldInfo(field);
    if (!info) return std::nullopt;
    return info->offset * 8 + info->bitOffset;
  }
  auto base = symbol_cast<BaseClassSymbol>(symbol);
  if (!base || (base->isVirtual() && owner->isAbstract())) return std::nullopt;
  auto cls = resolved_base_class(base);
  if (!cls) return std::nullopt;
  auto info = layout->getBaseInfo(cls, base->isVirtual());
  return info ? std::optional{info->offset * 8} : std::nullopt;
}

auto TypeTraits::layout_type(Symbol* symbol) const -> const Type* {
  return symbol ? visit(LayoutSubject{}, symbol) : nullptr;
}

auto TypeTraits::is_accessible_member(Symbol* member, ScopeSymbol* scope,
                                      ClassSymbol* designating) const
    -> std::optional<bool> {
  return MemberAccess{unit_, scope, designating}(member);
}

auto TypeTraits::members_of(ScopeSymbol* scope, ScopeSymbol* evaluationScope,
                            SourceLocation evaluationPoint) const
    -> std::vector<Symbol*> {
  auto point = membersOfPoint(evaluationScope, evaluationPoint);
  std::vector<Symbol*> result;
  for (const auto& member : MemberEnumeration{unit_, scope, point}())
    result.push_back(member.symbol);
  return result;
}

auto TypeTraits::class_is_complete_at(ClassSymbol* cls,
                                      ScopeSymbol* evaluationScope,
                                      SourceLocation evaluationPoint) const
    -> bool {
  auto point = membersOfPoint(evaluationScope, evaluationPoint);
  auto declaration = ast_cast<ClassSpecifierAST>(cls->declaration());
  if (!cls->isComplete()) return false;
  return !declaration || declaration->rbraceLoc < point;
}

auto TypeTraits::data_members_of(ClassSymbol* cls, bool isStatic) const
    -> std::vector<Symbol*> {
  std::vector<Symbol*> result;
  if (cls->isClosureType()) return result;
  for (auto member : cls->members())
    if (visit(DataMemberKind{isStatic}, member)) result.push_back(member);
  return result;
}

}  // namespace cxx
