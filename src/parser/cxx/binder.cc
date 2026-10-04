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
#include <cxx/ast_visitor.h>
#include <cxx/attributes.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/external_name_encoder.h>
#include <cxx/function_body_warnings.h>
#include <cxx/lambda_captures.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/standard_conversion.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>

namespace cxx {
namespace {
auto redeclarationTypesEquivalent(TranslationUnit* unit,
                                  const Type* existingType,
                                  const Type* incomingType,
                                  bool ignoresArrayBound = true) -> bool;
}

auto Binder::ClosureNumberingKeyHash::operator()(
    const ClosureNumberingKey& key) const -> std::size_t {
  auto seed = std::hash<FunctionSymbol*>{}(key.context);
  seed ^= std::hash<std::string>{}(key.signature) + 0x9e3779b97f4a7c15ull +
          (seed << 6) + (seed >> 2);
  return seed;
}

auto Binder::closureNumberingMark() const -> std::size_t {
  return closureNumberingLog_.size();
}

void Binder::rewindClosureNumbering(std::size_t mark) {
  while (closureNumberingLog_.size() > mark) {
    auto entry = closureNumberingLog_.back();
    closureNumberingLog_.pop_back();
    if (--entry->second == 0) closureCounts_.erase(entry->first);
  }
}

auto Binder::nextClosureNumber(ClassSymbol* closure) -> int {
  auto context = closure->enclosingFunction();
  std::string signature;
  if (closure_mangling_context(closure)) {
    signature = ExternalNameEncoder{unit_}.encodeLambdaSignature(
        closure->functionCallOperator());
  }

  auto [entry, inserted] = closureCounts_.try_emplace(
      ClosureNumberingKey{context, std::move(signature)}, 0);
  closureNumberingLog_.push_back(&*entry);
  return entry->second++;
}

Binder::Binder(TranslationUnit* unit) : unit_(unit), traits(unit) {
  languageLinkage_ = unit_->language();
}

auto Binder::translationUnit() const -> TranslationUnit* { return unit_; }

auto Binder::control() const -> Control* { return unit_->control(); }

auto Binder::isC() const -> bool {
  return unit_->language() == LanguageKind::kC;
}

auto Binder::isCxx() const -> bool {
  return unit_->language() == LanguageKind::kCXX;
}

auto Binder::reportErrors() const -> bool { return reportErrors_; }

void Binder::setReportErrors(bool reportErrors) {
  reportErrors_ = reportErrors;
}

void Binder::error(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  if (!unit_->config().checkTypes) return;
  unit_->error(loc, std::move(message));
}

void Binder::warning(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  if (!unit_->config().checkTypes) return;
  unit_->warning(loc, std::move(message));
}

void Binder::note(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  if (!unit_->config().checkTypes) return;
  unit_->note(loc, std::move(message));
}

auto Binder::inTemplate() const -> bool {
  return inTemplate_ || explicitTemplateHeadDepth_ > 0 ||
         retainsEnclosingTemplateLevels_;
}

void Binder::enterExplicitTemplateHead() { ++explicitTemplateHeadDepth_; }

auto Binder::inDiscardedStatement() const -> bool {
  return discardedStatementDepth_ > 0;
}

void Binder::leaveExplicitTemplateHead() { --explicitTemplateHeadDepth_; }

void Binder::enterExplicitInstantiation(bool isDefinition) {
  ++explicitInstantiationDepth_;
  explicitInstantiationIsDefinition_ = isDefinition;
}

void Binder::leaveExplicitInstantiation() { --explicitInstantiationDepth_; }

auto Binder::inExplicitInstantiation() const -> bool {
  return explicitInstantiationDepth_ > 0;
}

auto Binder::inExplicitInstantiationDefinition() const -> bool {
  return inExplicitInstantiation() && explicitInstantiationIsDefinition_;
}

void Binder::setRetainsEnclosingTemplateLevels(bool value) {
  retainsEnclosingTemplateLevels_ = value;
}

namespace {
struct FindReturnedValue final : ASTVisitor {
  bool found = false;

  auto preVisit(AST*) -> bool override { return !found; }

  void visit(ReturnStatementAST* ast) override {
    if (ast->expression) found = true;
  }

  void visit(LambdaExpressionAST*) override {}
};
}  // namespace

auto Binder::returnsAValue(AST* declaration) -> bool {
  if (!declaration) return false;
  FindReturnedValue scan;
  scan.accept(declaration);
  return scan.found;
}

void Binder::finishAutoReturnType(FunctionSymbol* functionSymbol) {
  if (!functionSymbol) return;
  auto funcType = type_cast<FunctionType>(functionSymbol->type());
  if (!funcType) return;
  if (!isPlaceholderType(funcType->returnType())) return;
  if (returnsAValue(functionSymbol->declaration())) return;

  auto newFuncType = control()->getFunctionType(
      control()->getVoidType(),
      std::vector<const Type*>(funcType->parameterTypes().begin(),
                               funcType->parameterTypes().end()),
      funcType->isVariadic(), funcType->cvQualifiers(),
      funcType->refQualifier(), funcType->exceptionSpecification());
  functionSymbol->setType(newFuncType);
}

auto Binder::currentTemplateParameters() const -> TemplateParametersSymbol* {
  auto templateParameters = symbol_cast<TemplateParametersSymbol>(scope());
  return templateParameters;
}

auto Binder::isInstantiating() const -> bool {
  return instantiatingSymbol_ != nullptr;
}

auto Binder::instantiatingSymbol() const -> Symbol* {
  return instantiatingSymbol_;
}

void Binder::setInstantiatingSymbol(Symbol* symbol) {
  instantiatingSymbol_ = symbol;
}

auto Binder::instantiationLoc() const -> SourceLocation {
  return instantiationLoc_;
}

void Binder::setInstantiationLoc(SourceLocation loc) {
  instantiationLoc_ = loc;
}

auto Binder::declaringScope() const -> ScopeSymbol* {
  auto scope = scope_;
  while (scope && scope->isTemplateParameters()) scope = scope->parent();
  return scope;
}

auto Binder::classBeingDefined() const -> ClassSymbol* {
  if (classBodyStack_.empty()) return nullptr;
  return classBodyStack_.back().classSymbol;
}

auto Binder::currentAccessSpecifier() const -> AccessSpecifier {
  if (classBodyStack_.empty()) return AccessSpecifier::kPublic;
  return classBodyStack_.back().accessSpecifier;
}

auto Binder::defaultAccessSpecifier() const -> AccessSpecifier {
  if (classBodyStack_.empty()) return AccessSpecifier::kPublic;
  return classBodyStack_.back().defaultAccessSpecifier;
}

void Binder::setCurrentAccessSpecifier(AccessSpecifier accessSpecifier) {
  if (classBodyStack_.empty()) return;
  classBodyStack_.back().accessSpecifier = accessSpecifier;
}

void Binder::applyAccessSpecifier(Symbol* symbol) const {
  if (!symbol) return;
  if (symbol_cast<ClassSymbol>(symbol->parent()) != classBeingDefined()) return;
  symbol->setAccessSpecifier(currentAccessSpecifier());
}

namespace {
template <typename S>
void applyTemplateHead(Binder& binder, S* symbol,
                       TemplateDeclarationAST* templateHead) {
  if (!symbol) return;

  if (templateHead) {
    binder.mergeTemplateParameterDefaults(
        symbol->canonical()->templateParameters(), templateHead->symbol);
  }

  symbol->setTemplateDeclaration(templateHead);

  if (!templateHead) return;

  symbol->setTemplateParameters(templateHead->symbol);
  binder.checkDefaultTemplateArgumentOnPack(templateHead->symbol);
}
}  // namespace

void Binder::setTemplateHead(FunctionSymbol* symbol,
                             TemplateDeclarationAST* templateHead) {
  applyTemplateHead(*this, symbol, templateHead);
}

void Binder::setTemplateHead(VariableSymbol* symbol,
                             TemplateDeclarationAST* templateHead) {
  applyTemplateHead(*this, symbol, templateHead);
}

void Binder::setTemplateHead(TypeAliasSymbol* symbol,
                             TemplateDeclarationAST* templateHead) {
  applyTemplateHead(*this, symbol, templateHead);
}

void Binder::checkTemplateParameterDefaultOrder(
    TemplateParametersSymbol* parameters) {
  if (!parameters) return;

  Symbol* defaulted = nullptr;

  for (auto parameter : parameters->members()) {
    if (default_template_argument(parameter)) {
      defaulted = parameter;
      continue;
    }

    if (!defaulted) continue;

    if (is_template_parameter_pack(parameter)) continue;

    error(parameter->location(),
          "template parameter missing a default argument");
    note(defaulted->location(),
         "previous default template argument defined here");
    return;
  }
}

void Binder::checkDefaultTemplateArgumentOnPack(
    TemplateParametersSymbol* parameters) {
  if (!parameters) return;

  for (auto parameter : parameters->members()) {
    if (!is_template_parameter_pack(parameter)) continue;
    if (!default_template_argument(parameter)) continue;

    error(parameter->location(),
          "a template parameter pack cannot have a default argument");
    return;
  }
}

void Binder::rejectDefaultTemplateArguments(
    TemplateDeclarationAST* templateHead, std::string message) {
  if (!templateHead) return;

  for (auto parameter : ListView{templateHead->templateParameterList}) {
    if (!hasWrittenDefaultTemplateArgument(parameter)) continue;
    error(parameter->firstSourceLocation(), std::move(message));
    return;
  }
}

void Binder::copyDefaultArguments(FunctionParametersSymbol* from,
                                  FunctionParametersSymbol* to) {
  if (!from || !to || from == to) return;

  auto sources = from->members() | views::parameters;
  auto targets = to->members() | views::parameters;

  auto sourceIt = sources.begin();
  auto targetIt = targets.begin();

  for (; sourceIt != sources.end() && targetIt != targets.end();
       ++sourceIt, ++targetIt) {
    auto source = *sourceIt;
    auto target = *targetIt;

    if (target->hasDefaultArgument()) continue;
    if (!source->hasDefaultArgument()) continue;

    inheritDefaultArgument(target, source);
  }
}

void Binder::inheritDefaultArgument(ParameterSymbol* target,
                                    ParameterSymbol* source) {
  if (auto expression = source->defaultArgument()) {
    setSpeculativeValue(
        target->defaultArgument(), expression,
        [target](ExpressionAST* value) { target->setDefaultArgument(value); });
    return;
  }

  setSpeculativeValue(target->defaultArgumentSource(), source,
                      [target](ParameterSymbol* value) {
                        target->setDefaultArgumentSource(value);
                      });
}

auto Binder::declaresDefaultTemplateArgument(Symbol* parameter) -> bool {
  auto defaultArgument = default_template_argument(parameter);
  return defaultArgument && defaultArgument->symbol == parameter;
}

void Binder::mergeTemplateParameterDefaults(
    TemplateParametersSymbol* accumulated, TemplateParametersSymbol* incoming) {
  if (!accumulated || !incoming) return;
  if (accumulated == incoming) return;

  const auto& previousParameters = accumulated->members();
  const auto& currentParameters = incoming->members();
  const auto count =
      std::min(previousParameters.size(), currentParameters.size());

  for (std::size_t index = 0; index < count; ++index) {
    auto previous = previousParameters[index];
    auto current = currentParameters[index];
    auto previousDefault = default_template_argument(previous);
    auto currentDefault = default_template_argument(current);

    if (previousDefault == currentDefault) continue;

    if (!declaresDefaultTemplateArgument(current)) {
      if (!previousDefault) continue;
      setSpeculativeValue(currentDefault, previousDefault,
                          [current](TemplateParameterAST* value) {
                            set_default_template_argument(current, value);
                          });
      continue;
    }

    if (!previousDefault) {
      setSpeculativeValue(previousDefault, currentDefault,
                          [previous](TemplateParameterAST* value) {
                            set_default_template_argument(previous, value);
                          });
      continue;
    }

    error(current->location(), "redefinition of default template argument");
    note(previous->location(), "previous definition is here");
  }
}

auto Binder::scopeForBlockDecl(ScopeSymbol* scope) const -> ScopeSymbol* {
  if (scope && scope->isBlock()) {
    if (auto ns = scope->enclosingNamespace()) return ns;
  }
  return scope;
}

void Binder::injectUsing(ScopeSymbol* scope, const Name* name, Symbol* target,
                         SourceLocation loc) {
  auto u = control()->newUsingDeclarationSymbol(scope, loc);
  u->setName(name);
  u->setTarget(target);
  if (target) u->setType(target->type());
  scope->addSymbol(u);
}

auto Binder::scope() const -> ScopeSymbol* { return scope_; }

void Binder::setScope(ScopeSymbol* scope) {
  scope_ = scope;
  inTemplate_ = isEnclosedInDependentTemplate(
      unit_, scope_, /*stopAtConcreteSpecialization=*/true);
}

auto Binder::languageLinkage() const -> LanguageKind {
  return languageLinkage_;
}

void Binder::setLanguageLinkage(LanguageKind linkage) {
  languageLinkage_ = linkage;
}

auto Binder::changeLanguageLinkage(LanguageKind linkage) -> LanguageKind {
  std::swap(languageLinkage_, linkage);
  return linkage;
}

auto Binder::enterBlock(SourceLocation loc) -> BlockSymbol* {
  auto blockSymbol = control()->newBlockSymbol(scope_, loc);
  scope_->addSymbol(blockSymbol);
  setScope(blockSymbol);
  return blockSymbol;
}

namespace {
struct FunctionOfBody {
  [[nodiscard]] auto operator()(FunctionSymbol* function) const
      -> FunctionSymbol* {
    return function;
  }

  [[nodiscard]] auto operator()(FunctionParametersSymbol* parameters) const
      -> FunctionSymbol* {
    return symbol_cast<FunctionSymbol>(parameters->parent());
  }

  [[nodiscard]] auto operator()(ClassSymbol* classSymbol) const
      -> FunctionSymbol* {
    if (!classSymbol->isClosureType()) return nullptr;
    return classSymbol->functionCallOperator();
  }

  [[nodiscard]] auto operator()(LambdaSymbol* lambda) const -> FunctionSymbol* {
    if (!lambda->closureType()) return nullptr;
    return (*this)(lambda->closureType());
  }

  [[nodiscard]] auto operator()(Symbol*) const -> FunctionSymbol* {
    return nullptr;
  }
};
}  // namespace

auto Binder::functionOfBody(ScopeSymbol* scope) -> FunctionSymbol* {
  if (!scope) return nullptr;
  return visit(FunctionOfBody{}, scope);
}

auto Binder::functionBodyBlock(ScopeSymbol* scope) -> BlockSymbol* {
  for (; scope; scope = scope->parent()) {
    auto block = symbol_cast<BlockSymbol>(scope);
    if (block && functionOfBody(block->parent())) return block;
  }
  return nullptr;
}

namespace {
[[nodiscard]] auto isFunctionLocalPredefinedName(WellKnownName name) -> bool {
  return name == WellKnownName::T___FUNC__ ||
         name == WellKnownName::T___FUNCTION__ ||
         name == WellKnownName::T___PRETTY_FUNCTION__;
}

[[nodiscard]] auto functionLocalPredefinedValue(FunctionSymbol* function,
                                                WellKnownName name)
    -> std::string {
  if (name == WellKnownName::T___PRETTY_FUNCTION__)
    return pretty_function_name(function);
  return to_string(function->name());
}
}  // namespace

auto Binder::functionLocalPredefinedVariable(ScopeSymbol* scope,
                                             const Name* name)
    -> VariableSymbol* {
  auto identifier = name_cast<Identifier>(name);
  if (!identifier) return nullptr;
  const auto wellKnownName = identifier->wellKnownName();
  if (!isFunctionLocalPredefinedName(wellKnownName)) return nullptr;

  auto owner = functionParameterScope(scope);
  if (!owner) return nullptr;

  for (auto candidate : owner->find(identifier)) {
    auto variable = symbol_cast<VariableSymbol>(candidate);
    if (variable && variable->isFunctionLocalPredefined()) return variable;
  }

  auto function = functionOfBody(owner);
  return declarePredefinedVariable(
      owner, identifier, functionLocalPredefinedValue(function, wellKnownName));
}

auto Binder::functionParameterScope(ScopeSymbol* scope) const -> ScopeSymbol* {
  if (scope && scope == functionBodyScope_ && functionOfBody(scope))
    return scope;
  if (auto body = functionBodyBlock(scope)) return body->parent();
  return nullptr;
}

auto Binder::declarePredefinedVariable(ScopeSymbol* owner,
                                       const Identifier* name,
                                       std::string_view value)
    -> VariableSymbol* {
  const Type* elementType = control()->getCharType();
  if (isCxx())
    elementType = control()->getQualType(elementType, CvQualifiers::kConst);
  auto type = control()->getBoundedArrayType(elementType, value.size() + 1);

  auto initializer = StringLiteralExpressionAST::create(unit_->arena());
  initializer->literal = control()->stringLiteralFromValue(value);
  initializer->type = type;
  initializer->valueCategory = ValueCategory::kLValue;

  auto variable = control()->newVariableSymbol(owner, owner->location());
  variable->setName(name);
  variable->setType(type);
  variable->setStatic(true);
  variable->setConstexpr(true);
  variable->setFunctionLocalPredefined(true);
  variable->setInitializer(initializer);
  variable->setConstValue(
      ASTInterpreter{unit_}.evaluateInitializer(type, initializer));
  owner->addSymbol(variable);
  return variable;
}

struct Binder::EnumerationHeadOf {
  [[nodiscard]] auto operator()(EnumSymbol* symbol) const
      -> std::optional<EnumerationHead> {
    return EnumerationHead{symbol->underlyingType(), false,
                           symbol->hasFixedUnderlyingType(),
                           symbol->isDefined()};
  }

  [[nodiscard]] auto operator()(ScopedEnumSymbol* symbol) const
      -> std::optional<EnumerationHead> {
    return EnumerationHead{symbol->underlyingType(), true, true,
                           symbol->isDefined()};
  }

  [[nodiscard]] auto operator()(Symbol*) const
      -> std::optional<EnumerationHead> {
    return std::nullopt;
  }
};

namespace {
struct MarkEnumerationDefined {
  void operator()(EnumSymbol* symbol) const { symbol->setDefined(true); }
  void operator()(ScopedEnumSymbol* symbol) const { symbol->setDefined(true); }
  void operator()(Symbol*) const {}
};
}  // namespace

auto Binder::declareEnum(const Name* name, SourceLocation location,
                         const Type* underlyingType, bool scoped,
                         bool fixedUnderlyingType, bool isDefinition,
                         bool isValidDeclaration) -> ScopeSymbol* {
  auto effectiveUnderlyingType = unit_->typeTraits().remove_cv(underlyingType);
  const auto invalidUnderlyingType =
      (scoped || fixedUnderlyingType) &&
      !unit_->typeTraits().is_integral(effectiveUnderlyingType) &&
      !isDependent(unit_, effectiveUnderlyingType);
  if (invalidUnderlyingType) {
    error(location, "enumeration underlying type must be integral");
    effectiveUnderlyingType = control()->getIntType();
  }

  const EnumerationHead head{effectiveUnderlyingType, scoped,
                             scoped || fixedUnderlyingType, isDefinition};

  if (invalidUnderlyingType || !isValidDeclaration)
    return newEnumeration(name, location, head, false);

  if (!name) return newEnumeration(name, location, head, true);

  for (auto candidate : declaringScope()->find(name)) {
    auto existing = visit(EnumerationHeadOf{}, candidate);
    if (!existing) {
      if (!symbol_cast<ClassSymbol>(candidate) &&
          !symbol_cast<TypeAliasSymbol>(candidate))
        continue;
      error(location,
            std::format("conflicting declaration of '{}'", to_string(name)));
      return newEnumeration(name, location, head, false);
    }

    if (!checkEnumerationRedeclaration(*existing, head, name, location))
      return newEnumeration(name, location, head, false);

    if (isDefinition) visit(MarkEnumerationDefined{}, candidate);
    return candidate->asScopeSymbol();
  }

  return newEnumeration(name, location, head, true);
}

auto Binder::newEnumeration(const Name* name, SourceLocation location,
                            const EnumerationHead& head, bool addToScope)
    -> ScopeSymbol* {
  auto enclosingScope = declaringScope();
  if (!addToScope)
    enclosingScope = control()->newBlockSymbol(enclosingScope, location);

  ScopeSymbol* symbol = nullptr;
  if (head.isScoped) {
    auto scopedEnum = control()->newScopedEnumSymbol(enclosingScope, location);
    scopedEnum->setUnderlyingType(head.underlyingType);
    scopedEnum->setDefined(head.isDefined);
    symbol = scopedEnum;
  } else {
    auto enumeration = control()->newEnumSymbol(enclosingScope, location);
    enumeration->setUnderlyingType(head.underlyingType);
    enumeration->setHasFixedUnderlyingType(head.hasFixedUnderlyingType);
    enumeration->setDefined(head.isDefined);
    symbol = enumeration;
  }

  symbol->setName(name);
  if (!addToScope) return symbol;
  scope()->addSymbol(symbol);
  recordStandardLibraryType(symbol);
  return symbol;
}

void Binder::recordStandardLibraryType(Symbol* symbol) {
  if (!traits.is_in_std_namespace(symbol)) return;

  switch (well_known_name(symbol->name())) {
    case WellKnownName::T_ALIGN_VAL_T:
      control()->setAlignValType(symbol->type());
      break;

    case WellKnownName::T_NOTHROW_T:
      control()->setNothrowType(symbol->type());
      break;

    default:
      break;
  }
}

auto Binder::checkEnumerationRedeclaration(const EnumerationHead& existing,
                                           const EnumerationHead& declared,
                                           const Name* name,
                                           SourceLocation location) -> bool {
  if (existing.isScoped != declared.isScoped) {
    error(location,
          std::format("enumeration '{}' redeclared with different scopedness",
                      to_string(name)));
    return false;
  }

  if (existing.hasFixedUnderlyingType != declared.hasFixedUnderlyingType) {
    error(location,
          std::format(
              "enumeration '{}' redeclared with incompatible underlying type",
              to_string(name)));
    return false;
  }

  if (existing.hasFixedUnderlyingType &&
      !traits.is_same(traits.remove_cv(existing.underlyingType),
                      declared.underlyingType)) {
    error(location,
          std::format(
              "enumeration '{}' redeclared with different underlying type",
              to_string(name)));
    return false;
  }

  if (declared.isDefined && existing.isDefined) {
    error(location,
          std::format("redefinition of enumeration '{}'", to_string(name)));
    return false;
  }

  return true;
}

void Binder::bind(EnumSpecifierAST* ast, const DeclSpecs& underlyingTypeSpecs) {
  const auto underlyingType = underlyingTypeSpecs.hasTypeOrSizeSpecifier()
                                  ? underlyingTypeSpecs.type()
                                  : control()->getIntType();
  const auto location = ast->unqualifiedId
                            ? ast->unqualifiedId->firstSourceLocation()
                            : ast->lbraceLoc;
  if (isC() && ast->classLoc)
    error(ast->classLoc, "scoped enums are not allowed in C");
  ast->symbol = declareEnum(get_name(control(), ast->unqualifiedId), location,
                            underlyingType, ast->classLoc && isCxx(),
                            ast->typeSpecifierList != nullptr, true);
  applyAccessSpecifier(ast->symbol);
  applyDeclarationAttributes(ast->symbol, ast->attributeList);
  setScope(ast->symbol->asScopeSymbol());
}

void Binder::bind(OpaqueEnumDeclarationAST* ast,
                  const DeclSpecs& underlyingTypeSpecs) {
  const auto underlyingType = underlyingTypeSpecs.hasTypeOrSizeSpecifier()
                                  ? underlyingTypeSpecs.type()
                                  : control()->getIntType();
  const auto location = ast->unqualifiedId
                            ? ast->unqualifiedId->firstSourceLocation()
                            : ast->enumLoc;
  if (isC() && ast->classLoc)
    error(ast->classLoc, "scoped enums are not allowed in C");
  const auto missingUnscopedBase =
      isCxx() && !ast->classLoc && !ast->typeSpecifierList;
  if (missingUnscopedBase)
    error(location,
          "opaque declaration of an unscoped enumeration requires an "
          "underlying type");
  if (ast->nestedNameSpecifier)
    error(location,
          "opaque enumeration declaration cannot have a qualified name");
  ast->symbol =
      declareEnum(get_name(control(), ast->unqualifiedId), location,
                  underlyingType, ast->classLoc && isCxx(), true, false,
                  !missingUnscopedBase && !ast->nestedNameSpecifier);
}

auto Binder::elaboratedTypeSpecifierTargetScope(bool isFriend,
                                                bool isDeclaration) const
    -> ScopeSymbol* {
  if (!isCxx()) return declaringScope();
  if (isDeclaration && !isFriend) return declaringScope();
  for (auto scope = declaringScope(); scope; scope = scope->parent()) {
    if (scope->isNamespace() || scope->isBlock()) return scope;
  }
  return declaringScope();
}

void Binder::bind(ElaboratedTypeSpecifierAST* ast, DeclSpecs& declSpecs,
                  bool isDeclaration, Symbol* unqualifiedCandidate) {
  const auto _ = ScopeGuard{this};

  if (ast->nestedNameSpecifier) {
    auto parent = ast->nestedNameSpecifier->symbol;

    if (!parent || !parent->isClassOrNamespace()) {
      (void)reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier);
      return;
    }

    setScope(parent->asScopeSymbol());
  }

  auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);

  const Identifier* name = nullptr;
  if (templateId)
    name = templateId->identifier;
  else if (auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId))
    name = nameId->identifier;

  const auto location = ast->unqualifiedId->firstSourceLocation();

  if (ast->classKey == TokenKind::T_CLASS ||
      ast->classKey == TokenKind::T_STRUCT ||
      ast->classKey == TokenKind::T_UNION) {
    auto is_class = [](Symbol* symbol) {
      if (symbol->isClass()) return true;
      return false;
    };

    auto targetScope =
        elaboratedTypeSpecifierTargetScope(declSpecs.isFriend, isDeclaration);

    auto candidate = [&]() -> Symbol* {
      if (declSpecs.isFriend)
        return lookupFriendClass(name, declSpecs.templateHead != nullptr);
      if (ast->nestedNameSpecifier)
        return qualifiedLookup(ast->nestedNameSpecifier->symbol, name,
                               is_class);
      return unqualifiedCandidate;
    }();

    auto classSymbol = symbol_cast<ClassSymbol>(candidate);

    if (classSymbol && isDeclaration && !declSpecs.isFriend &&
        classSymbol->parent() != targetScope) {
      classSymbol = nullptr;
    }

    auto adoptedClassSymbol = static_cast<ClassSymbol*>(nullptr);
    if (!classSymbol && !declSpecs.isFriend) {
      adoptedClassSymbol = adoptFriendDeclaredClass(targetScope, name);
      classSymbol = adoptedClassSymbol;
    }

    if (adoptedClassSymbol) {
      adoptedClassSymbol->setIsUnion(ast->classKey == TokenKind::T_UNION);
      adoptedClassSymbol->setTemplateDeclaration(declSpecs.templateHead);
      if (declSpecs.templateHead) {
        adoptedClassSymbol->setTemplateParameters(
            declSpecs.templateHead->symbol);
      }
      adoptedClassSymbol->setDeclaration(ast);
    }

    if (!classSymbol) {
      const auto isUnion = ast->classKey == TokenKind::T_UNION;
      classSymbol = control()->newClassSymbol(targetScope, location);

      applyAccessSpecifier(classSymbol);
      classSymbol->setIsUnion(isUnion);
      classSymbol->setName(name);
      classSymbol->setTemplateDeclaration(declSpecs.templateHead);
      if (declSpecs.templateHead)
        classSymbol->setTemplateParameters(declSpecs.templateHead->symbol);
      targetScope->addSymbol(classSymbol);
      recordStandardLibraryType(classSymbol);

      if (declSpecs.isFriend) {
        classSymbol->setFriend(true);
        classSymbol->setHidden(true);
      }

      classSymbol->setDeclaration(ast);
    } else if (declSpecs.templateHead && isDeclaration) {
      if (classSymbol->templateParameters()) {
        mergeTemplateParameterDefaults(classSymbol->templateParameters(),
                                       declSpecs.templateHead->symbol);
      } else {
        classSymbol->setTemplateDeclaration(declSpecs.templateHead);
        classSymbol->setTemplateParameters(declSpecs.templateHead->symbol);
      }
    }

    if (declSpecs.templateHead && isDeclaration && declSpecs.isFriend) {
      rejectDefaultTemplateArguments(
          declSpecs.templateHead,
          "a default template argument cannot be specified on a friend "
          "template declaration");
    }

    if (declSpecs.templateHead && isDeclaration) {
      checkDefaultTemplateArgumentOnPack(classSymbol->templateParameters());
    }

    if (declSpecs.templateHead && isDeclaration && !declSpecs.isFriend) {
      checkTemplateParameterDefaultOrder(classSymbol->templateParameters());
    }

    ast->symbol = classSymbol;

    if (auto alignment = explicitAlignment(ast->attributeList, location)) {
      checkRedeclaredAlignment(classSymbol, *alignment, location);
    }

    if (declSpecs.isFriend && !templateId && classBeingDefined()) {
      classSymbol->canonical()->addBefriendingClass(classBeingDefined());
    }
  }

  declSpecs.setTypeSpecifier(ast);

  if (ast->symbol) {
    declSpecs.setType(ast->symbol->type());
  }
}

void Binder::enterSpeculativeDeclarations() { ++speculationDepth_; }

void Binder::leaveSpeculativeDeclarations() {
  if (--speculationDepth_ == 0) speculativeMutations_.clear();
}

void Binder::recordSpeculativeMutation(std::function<void()> undo) {
  if (!speculationDepth_) return;
  speculativeMutations_.push_back(std::move(undo));
}

void Binder::recordSpeculativeOverload(OverloadSetSymbol* overloadSet) {
  if (!speculationDepth_) return;
  auto functionCount = overloadSet->declaredFunctions().size();
  recordSpeculativeMutation([overloadSet, functionCount] {
    overloadSet->truncateFunctions(functionCount);
  });
}

void Binder::undoSpeculativeMutations(std::size_t count) {
  while (speculativeMutations_.size() > count) {
    auto undo = std::move(speculativeMutations_.back());
    speculativeMutations_.pop_back();
    undo();
  }
}

auto Binder::lookupFriendClass(const Identifier* name, bool namesTemplate)
    -> ClassSymbol* {
  auto isClass = [](Symbol* symbol) {
    return symbol->isClass() || symbol_cast<InjectedClassNameSymbol>(symbol);
  };
  for (auto scope = declaringScope(); scope; scope = scope->parent()) {
    if (auto found = qualifiedLookup(scope, name, isClass)) {
      if (namesTemplate)
        return symbol_cast<ClassSymbol>(templated_symbol(found));
      if (auto injected = symbol_cast<InjectedClassNameSymbol>(found))
        return injected->classSymbol();
      return symbol_cast<ClassSymbol>(found);
    }
    for (auto candidate : scope->find(name) | views::classes) {
      if (candidate->isFriend()) return candidate;
    }
    if (scope->isNamespace() || scope->isBlock()) break;
  }
  return nullptr;
}

auto Binder::adoptFriendDeclaredClass(ScopeSymbol* targetScope,
                                      const Identifier* name) -> ClassSymbol* {
  if (!name) return nullptr;

  for (auto candidate : targetScope->find(name)) {
    auto classSymbol = symbol_cast<ClassSymbol>(candidate);
    if (!classSymbol) continue;
    if (!classSymbol->isFriend()) continue;
    if (classSymbol->parent() != targetScope) continue;

    classSymbol->setFriend(false);
    classSymbol->setHidden(false);
    return classSymbol;
  }

  return nullptr;
}

void Binder::disableAccessControlForUnsupportedFriend(
    NestedNameSpecifierAST* nestedNameSpecifier,
    ClassSymbol* befriendingClass) {
  if (!inTemplate() || !isDependent(unit_, nestedNameSpecifier)) return;
  if (!befriendingClass) return;

  befriendingClass->setAccessControlDisabled(true);
}

void Binder::checkExceptionDeclarationType(TypeExceptionDeclarationAST* ast,
                                           const Type* type) {
  auto loc = ast->firstSourceLocation();

  if (traits.is_rvalue_reference(type)) {
    error(loc, std::format("cannot catch an rvalue reference of type '{}'",
                           to_string(type)));
    return;
  }

  const auto isReference = traits.is_reference(type);
  auto declaredType = traits.remove_reference(type);

  if (!traits.is_complete(declaredType)) {
    error(loc, std::format("cannot catch an incomplete type '{}'",
                           to_string(declaredType)));
    return;
  }

  if (!isReference && traits.is_abstract(declaredType)) {
    error(loc, std::format("cannot catch an abstract class type '{}'",
                           to_string(declaredType)));
    return;
  }

  auto pointee = traits.remove_pointer(declaredType);
  if (pointee == declaredType) return;
  if (traits.is_void(traits.remove_cv(pointee))) return;

  if (!traits.is_complete(pointee)) {
    error(loc, std::format("cannot catch a pointer to the incomplete type '{}'",
                           to_string(pointee)));
  }
}

void Binder::checkTrailingRequiresClauseIsTemplated(
    FunctionSymbol* functionSymbol, TemplateDeclarationAST* templateHead) {
  if (!functionSymbol) return;

  auto requiresClause = functionSymbol->trailingRequiresClause();
  if (!requiresClause) return;
  if (templateHead) return;
  if (functionSymbol->templateDeclaration()) return;
  if (isInstantiating()) return;
  if (unit_->isInstantiatingTemplate()) return;

  for (auto s = functionSymbol->parent(); s; s = s->parent()) {
    if (s->isTemplateParameters()) return;
    if (auto classSymbol = symbol_cast<ClassSymbol>(s)) {
      if (classSymbol->templateDeclaration()) return;
      if (classSymbol->isSpecialization()) return;
    }
    if (auto function = symbol_cast<FunctionSymbol>(s)) {
      if (function->templateDeclaration()) return;
    }
  }

  error(requiresClause->firstSourceLocation(),
        "non-templated function cannot have a requires clause");
}

void Binder::bind(TypeExceptionDeclarationAST* ast, const Decl& decl) {
  if (explicitAlignment(ast->attributeList, ast->firstSourceLocation())) {
    error(ast->firstSourceLocation(),
          "'alignas' attribute cannot be applied to an exception declaration");
  }

  auto type = traits.adjusted_parameter_type(
      getDeclaratorType(unit_, ast->declarator, decl.specs.type()));

  auto declaredType = traits.remove_reference(type);

  if (auto classType = unqualified_cast<ClassType>(declaredType)) {
    traits.requireCompleteClass(classType->symbol());
  } else if (auto pointee = traits.remove_pointer(declaredType);
             pointee != declaredType) {
    if (auto classType = unqualified_cast<ClassType>(pointee))
      traits.requireCompleteClass(classType->symbol());
  }

  checkExceptionDeclarationType(ast, type);

  auto location = decl.location();
  if (!location) location = ast->firstSourceLocation();

  auto symbol = control()->newVariableSymbol(scope_, location);
  symbol->setName(decl.getName());
  symbol->setType(type);
  ast->symbol = symbol;

  if (symbol->name()) scope_->addSymbol(symbol);
}

auto Binder::parameterObjectType(ParameterDeclarationAST* ast,
                                 const Type* specifiersType) -> const Type* {
  return traits.adjusted_parameter_type(
      getDeclaratorType(unit_, ast->declarator, specifiersType));
}

void Binder::rebindParameterType(ParameterDeclarationAST* ast,
                                 ParameterSymbol* symbol) {
  DeclSpecs specs{unit_};
  for (auto specifier : ListView{ast->typeSpecifierList})
    specs.accept(specifier);
  specs.finish();

  auto type = parameterObjectType(ast, specs.type());
  ast->type = unqualified_type(type);
  symbol->setType(type);
}

void Binder::bind(ParameterDeclarationAST* ast, const Decl& decl,
                  bool inTemplateParameters) {
  auto parameterObjectType = this->parameterObjectType(ast, decl.specs.type());

  ast->type = unqualified_type(parameterObjectType);

  if (explicitAlignment(ast->attributeList, decl.location())) {
    error(decl.location(),
          "'alignas' attribute cannot be applied to a function parameter");
  }

  if (auto declId = decl.declaratorId; declId && declId->unqualifiedId) {
    auto paramName = get_name(control(), declId->unqualifiedId);
    if (auto identifier = name_cast<Identifier>(paramName)) {
      ast->identifier = identifier;
    } else {
      error(declId->unqualifiedId->firstSourceLocation(),
            "expected an identifier");
    }
  }

  if (!inTemplateParameters) {
    auto parameterLoc = decl.location();
    if (!parameterLoc) parameterLoc = ast->firstSourceLocation();

    const auto isFirstParameter = scope_->members().empty();

    if (ast->isThisIntroduced && !isFirstParameter) {
      error(ast->thisLoc,
            "an explicit object parameter must be the first parameter");
    }

    if (ast->isThisIntroduced && decl.isPack) {
      error(ast->thisLoc,
            "an explicit object parameter cannot be a function parameter pack");
    }

    auto parameterSymbol = control()->newParameterSymbol(scope_, parameterLoc);
    parameterSymbol->setName(ast->identifier);
    parameterSymbol->setType(parameterObjectType);
    parameterSymbol->setDefaultArgument(ast->expression);
    parameterSymbol->setExplicitObject(ast->isThisIntroduced &&
                                       isFirstParameter);
    parameterSymbol->setParameterPack(ast->isPack);
    scope_->addSymbol(parameterSymbol);
    ast->symbol = parameterSymbol;
  }
}

void Binder::bind(DecltypeSpecifierAST* ast) {
  if (isDependent(unit_, ast->expression)) {
    ast->type = control()->getDecltypeType(unit_, ast->expression);
    return;
  }
  if (auto type = traits.decltype_of(ast->expression)) ast->type = type;
}

auto Binder::nextEnumeratorValue(TranslationUnit* unit,
                                 const Type* underlyingType,
                                 const std::optional<ConstValue>& previous)
    -> std::optional<ConstValue> {
  if (!previous.has_value()) return std::intmax_t{0};

  ASTInterpreter interp{unit};

  if (unit->typeTraits().is_unsigned(underlyingType)) {
    if (auto v = interp.toUInt(previous.value()))
      return std::bit_cast<std::intmax_t>(v.value() + 1);
    return std::nullopt;
  }

  if (auto v = interp.toInt(previous.value())) return v.value() + 1;
  return std::nullopt;
}

auto Binder::enumeratorType(EnumeratorAST* ast, const Type* previousType) const
    -> const Type* {
  if (isC()) return scope()->type();
  if (auto scopedEnum = symbol_cast<ScopedEnumSymbol>(scope()))
    return scopedEnum->underlyingType();
  if (auto unscopedEnum = symbol_cast<EnumSymbol>(scope());
      unscopedEnum && unscopedEnum->hasFixedUnderlyingType())
    return unscopedEnum->underlyingType();
  if (!ast->expression)
    return previousType ? previousType : control()->getIntType();
  auto type = traits.remove_cv(ast->expression->type);
  if (auto enumType = type_cast<EnumType>(type))
    return enumType->underlyingType();
  return type;
}

void Binder::bind(EnumeratorAST* ast, const Type* previousType,
                  std::optional<ConstValue> value) {
  auto type = enumeratorType(ast, previousType);
  if (isCxx()) {
    auto symbol = control()->newEnumeratorSymbol(scope(), ast->identifierLoc);
    ast->symbol = symbol;

    symbol->setName(ast->identifier);
    symbol->setType(type);
    ast->symbol->setValue(value);
    if (auto enclosingEnum = symbol_cast<EnumSymbol>(scope()))
      symbol->setAccessSpecifier(enclosingEnum->accessSpecifier());
    if (auto enclosingEnum = symbol_cast<ScopedEnumSymbol>(scope()))
      symbol->setAccessSpecifier(enclosingEnum->accessSpecifier());
    scope()->addSymbol(symbol);

    if (auto enumSymbol = symbol_cast<EnumSymbol>(scope())) {
      auto parentScope = enumSymbol->parent();

      auto u =
          control()->newUsingDeclarationSymbol(parentScope, ast->identifierLoc);
      u->setName(ast->identifier);
      u->setTarget(symbol);
      parentScope->addSymbol(u);
      applyAccessSpecifier(u);
    }

    return;
  }

  if (auto enumSymbol = symbol_cast<EnumSymbol>(scope())) {
    auto parentScope = enumSymbol->parent();

    auto enumeratorSymbol =
        control()->newEnumeratorSymbol(parentScope, ast->identifierLoc);
    ast->symbol = enumeratorSymbol;

    enumeratorSymbol->setName(ast->identifier);
    enumeratorSymbol->setType(type);
    enumeratorSymbol->setValue(value);

    parentScope->addSymbol(enumeratorSymbol);
  }
}

namespace {
struct DeclaredScopeOfType {
  [[nodiscard]] auto operator()(const ClassType* type) const -> ScopeSymbol* {
    return type->symbol();
  }

  [[nodiscard]] auto operator()(const EnumType* type) const -> ScopeSymbol* {
    return type->symbol();
  }

  [[nodiscard]] auto operator()(const ScopedEnumType* type) const
      -> ScopeSymbol* {
    return type->symbol();
  }

  [[nodiscard]] auto operator()(const Type*) const -> ScopeSymbol* {
    return nullptr;
  }
};

[[nodiscard]] auto declaredScopeOfType(const Type* type) -> ScopeSymbol* {
  if (!type) return nullptr;
  return visit(DeclaredScopeOfType{}, type);
}

[[nodiscard]] auto declaresTagName(Symbol* symbol) -> bool {
  return symbol_cast<ClassSymbol>(symbol) || symbol_cast<EnumSymbol>(symbol);
}
}  // namespace

void Binder::addTypeAliasToScope(TypeAliasSymbol* symbol) {
  auto scope = symbol->parent();
  auto name = symbol->name();

  for (auto declaration : scope->find(name)) {
    auto candidate = resolve_using_declaration(declaration);
    if (auto existing = symbol_cast<TypeAliasSymbol>(candidate)) {
      auto equivalent = TemplateEquivalence{unit_}.same(
          existing->templateDeclaration(), symbol->templateDeclaration());
      if (existing->type() && symbol->type()) {
        if (!redeclarationTypesEquivalent(unit_, existing->type(),
                                          symbol->type(), false)) {
          equivalent = false;
        }
      }
      if (equivalent) {
        addRedeclaration(existing->canonical(), symbol);
        return;
      }
    } else if (isC() && declaresTagName(candidate)) {
      continue;
    } else if (declaredScopeOfType(symbol->type()) == candidate) {
      continue;
    }
    error(symbol->location(),
          std::format("conflicting declaration of '{}'", to_string(name)));
    return;
  }
  scope->addSymbol(symbol);
}

namespace {
[[nodiscard]] auto definedUnnamedType(SpecifierAST* specifier) -> Symbol* {
  Symbol* symbol = nullptr;
  if (auto classSpecifier = ast_cast<ClassSpecifierAST>(specifier))
    symbol = classSpecifier->symbol;
  else if (auto enumSpecifier = ast_cast<EnumSpecifierAST>(specifier))
    symbol = enumSpecifier->symbol;
  if (!symbol || symbol->name()) return nullptr;
  return symbol;
}
}  // namespace

void Binder::giveTypedefNameForLinkage(TypeAliasSymbol* alias,
                                       SpecifierAST* specifier) {
  auto unnamed = definedUnnamedType(specifier);
  if (!unnamed || unnamed->type() != alias->type()) return;
  unnamed->setName(alias->name());
}

auto Binder::declareTypeAlias(SourceLocation identifierLoc,
                              const Identifier* identifier, TypeIdAST* typeId,
                              bool addSymbolToParentScope,
                              TemplateDeclarationAST* templateHead)
    -> TypeAliasSymbol* {
  auto symbol = control()->newTypeAliasSymbol(declaringScope(), identifierLoc);
  applyAccessSpecifier(symbol);

  auto name = identifier;
  symbol->setName(name);

  if (typeId) symbol->setType(typeId->type);
  setTemplateHead(symbol, templateHead);
  checkTemplateParameterDefaultOrder(symbol->canonical()->templateParameters());

  if (typeId) {
    for (auto specifier : ListView{typeId->typeSpecifierList})
      giveTypedefNameForLinkage(symbol, specifier);
  }

  if (addSymbolToParentScope) addTypeAliasToScope(symbol);

  return symbol;
}

namespace {

[[nodiscard]] auto joinsFunctionOverloadSet(Symbol* candidate) -> bool {
  if (symbol_cast<OverloadSetSymbol>(candidate)) return true;
  if (symbol_cast<FunctionSymbol>(candidate)) return true;
  auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(candidate);
  if (!usingDeclaration) return false;
  if (usingDeclaration->isUnresolved()) return true;
  return !usingDeclaration->introducedFunctions().empty();
}

struct TerminalNestedNameSpecifierName {
  auto operator()(GlobalNestedNameSpecifierAST*) const -> const Identifier* {
    return nullptr;
  }

  auto operator()(SimpleNestedNameSpecifierAST* ast) const
      -> const Identifier* {
    return ast->identifier;
  }

  auto operator()(DecltypeNestedNameSpecifierAST*) const -> const Identifier* {
    return nullptr;
  }

  auto operator()(TemplateNestedNameSpecifierAST* ast) const
      -> const Identifier* {
    return ast->templateId ? ast->templateId->identifier : nullptr;
  }
};

}  // namespace

auto Binder::usingDeclaratorNamesConstructor(UsingDeclaratorAST* ast) -> bool {
  if (!ast->nestedNameSpecifier || ast->typenameLoc) return false;
  auto terminal =
      visit(TerminalNestedNameSpecifierName{}, ast->nestedNameSpecifier);
  if (!terminal) return false;
  auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId);
  return nameId && nameId->identifier == terminal;
}

auto Binder::bindInheritedConstructors(UsingDeclaratorAST* ast) -> bool {
  auto derived = symbol_cast<ClassSymbol>(scope());
  if (!derived) return false;

  auto lookupContext =
      ast->nestedNameSpecifier ? ast->nestedNameSpecifier->symbol : nullptr;
  auto base = symbol_cast<ClassSymbol>(lookupContext);
  if (!base) return isDependent(unit_, ast->nestedNameSpecifier);
  if (isDependent(unit_, base->type())) return true;
  base = base->resolvedDefinition();

  const auto isDirectBase =
      std::ranges::any_of(derived->baseClasses(), [&](BaseClassSymbol* b) {
        auto candidate = symbol_cast<ClassSymbol>(b->symbol());
        return candidate && candidate->resolvedDefinition() == base;
      });

  if (!isDirectBase) {
    error(ast->unqualifiedId->firstSourceLocation(),
          std::format("'{}' is not a direct base class of '{}'",
                      to_string(base->name()), to_string(derived->name())));
    return true;
  }

  auto symbol = control()->newUsingDeclarationSymbol(
      derived, ast->unqualifiedId->firstSourceLocation());
  ast->symbol = symbol;
  symbol->setName(derived->name());
  symbol->setDeclarator(ast);
  symbol->setTarget(base->constructorOverloadSet());

  derived->constructorOverloadSet()->addUsingDeclaration(symbol);
  return true;
}

void Binder::bind(UsingDeclaratorAST* ast, Symbol* target) {
  auto makeDependentTypeTarget = [&]() -> Symbol* {
    if (!ast->typenameLoc) return nullptr;
    auto alias = control()->newTypeAliasSymbol(
        scope(), ast->unqualifiedId->firstSourceLocation());
    alias->setName(get_name(control(), ast->unqualifiedId));
    alias->setType(control()->getUnresolvedNameType(
        unit_, ast->nestedNameSpecifier, ast->unqualifiedId));
    return alias;
  };

  if (usingDeclaratorNamesConstructor(ast)) {
    if (bindInheritedConstructors(ast)) return;
  }

  if (ast->nestedNameSpecifier && !ast->nestedNameSpecifier->symbol) {
    if (reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier)) return;
  }

  const bool dependentQualifier =
      inTemplate() && isDependent(unit_, ast->nestedNameSpecifier);

  if (dependentQualifier) target = makeDependentTypeTarget();

  target = resolve_using_declaration(target);

  if (!target && !dependentQualifier) {
    if (!inTemplate()) {
      auto missingName = get_name(control(), ast->unqualifiedId);
      error(ast->unqualifiedId->firstSourceLocation(),
            std::format("using declaration refers to unresolved name '{}'",
                        to_string(missingName)));
      return;
    }
    target = makeDependentTypeTarget();
  }

  const auto name = get_name(control(), ast->unqualifiedId);

  auto symbol = control()->newUsingDeclarationSymbol(
      scope(), ast->unqualifiedId->firstSourceLocation());

  ast->symbol = symbol;

  applyAccessSpecifier(symbol);
  symbol->setName(name);
  symbol->setDeclarator(ast);
  symbol->setTarget(target);

  if (!dependentQualifier) checkUsingDeclaratorAccess(ast, symbol);

  if (redeclaresOnlyMembersOf(symbol, scope())) return;

  const auto joinsAnOverloadSet =
      joinsFunctionOverloadSet(symbol) &&
      std::ranges::any_of(scope()->find(name), joinsFunctionOverloadSet);

  if (!joinsAnOverloadSet) {
    scope()->addSymbol(symbol);
    return;
  }

  overloadSetFor(scope(), name, symbol->location())
      ->addUsingDeclaration(symbol);
}

auto Binder::redeclaresOnlyMembersOf(UsingDeclarationSymbol* usingDeclaration,
                                     ScopeSymbol* scope) -> bool {
  auto functions = usingDeclaration->introducedFunctions();
  if (functions.empty()) return false;
  return std::ranges::all_of(functions, [scope](FunctionSymbol* function) {
    return function->parent() == scope;
  });
}

void Binder::checkUsingDeclaratorAccess(UsingDeclaratorAST* ast,
                                        UsingDeclarationSymbol* symbol) {
  if (usingDeclaratorNamesConstructor(ast)) return;
  if (!ast->nestedNameSpecifier) return;

  auto designatingClass =
      symbol_cast<ClassSymbol>(ast->nestedNameSpecifier->symbol);
  if (!designatingClass) return;

  const auto location = ast->unqualifiedId->firstSourceLocation();

  auto reportInaccessible = [&](Symbol* named) {
    if (!named) return;
    (void)checkMemberAccess(unit_, scope(), named, designatingClass, nullptr,
                            location);
  };

  auto introduced = symbol->introducedFunctions();
  if (introduced.empty()) {
    reportInaccessible(symbol->target());
    return;
  }

  for (auto function : introduced) reportInaccessible(function);
}

void Binder::declareBaseClass(BaseSpecifierAST* ast, Symbol* symbol) {
  auto baseClassSymbol = control()->newBaseClassSymbol(
      scope(), ast->unqualifiedId->firstSourceLocation());
  ast->symbol = baseClassSymbol;
  baseClassSymbol->setVirtual(ast->isVirtual);
  baseClassSymbol->setSymbol(symbol);
  baseClassSymbol->setName(symbol->name());
  baseClassSymbol->setAccessSpecifier(
      toAccessSpecifier(ast->accessSpecifier, defaultAccessSpecifier()));
}

void Binder::checkBaseClass(BaseSpecifierAST* ast, ClassSymbol* baseClass) {
  if (!baseClass) return;
  traits.requireCompleteClass(baseClass);

  const auto location = ast->unqualifiedId->firstSourceLocation();

  if (baseClass->isFinal()) {
    error(location, std::format("cannot derive from 'final' class '{}'",
                                to_string(baseClass->name())));
  }

  if (baseClass->resolvedDefinition()->isUnion()) {
    error(location, std::format("union '{}' cannot be used as a base class",
                                to_string(baseClass->name())));
  }

  auto derived = symbol_cast<ClassSymbol>(scope());
  if (derived && derived->isUnion()) {
    error(location, "a union cannot have base classes");
  }
}

void Binder::bind(BaseSpecifierAST* ast, Symbol* resolvedType) {
  const auto checkTemplates = unit_->config().checkTypes;

  if (ast->nestedNameSpecifier && !ast->nestedNameSpecifier->symbol) {
    if (reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier)) return;
  }

  Symbol* symbol = nullptr;

  if (auto decltypeId = ast_cast<DecltypeIdAST>(ast->unqualifiedId)) {
    if (auto classType =
            unqualified_cast<ClassType>(decltypeId->decltypeSpecifier->type)) {
      symbol = classType->symbol();
    }
  } else {
    symbol = resolve(ast->nestedNameSpecifier, ast->unqualifiedId,
                     checkTemplates, resolvedType);
  }

  if (!symbol && isDependent(unit_, ast->nestedNameSpecifier)) {
    auto alias = control()->newTypeAliasSymbol(
        scope(), ast->unqualifiedId->firstSourceLocation());
    alias->setName(get_name(control(), ast->unqualifiedId));
    alias->setType(control()->getUnresolvedNameType(
        unit_, ast->nestedNameSpecifier, ast->unqualifiedId));
    symbol = alias;
  }

  if (auto typeAlias = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType = unqualified_cast<ClassType>(typeAlias->type())) {
      symbol = classType->symbol();
    }
  }

  if (!symbol || !symbol->isClass()) {
    if (!symbol) {
      if (!inTemplate()) {
        auto baseName = get_name(control(), ast->unqualifiedId);
        error(ast->unqualifiedId->firstSourceLocation(),
              std::format("unknown base class '{}'", to_string(baseName)));
      }
      return;
    }

    if (isDependent(unit_, symbol->type())) {
      declareBaseClass(ast, symbol);
      return;
    }
    if (!inTemplate()) {
      error(ast->unqualifiedId->firstSourceLocation(),
            "base class specifier must be a class");
    }
    return;
  }

  checkBaseClass(ast, symbol_cast<ClassSymbol>(symbol));
  declareBaseClass(ast, symbol);
}

void Binder::bind(NonTypeTemplateParameterAST* ast, int index, int depth) {
  auto symbol = control()->newNonTypeParameterSymbol(
      scope(), ast->declaration->firstSourceLocation());
  ast->symbol = symbol;
  ast->index = index;
  ast->depth = depth;

  symbol->setIndex(index);
  symbol->setDepth(depth);
  symbol->setName(ast->declaration->identifier);
  symbol->setParameterPack(ast->declaration->isPack);
  symbol->setObjectType(ast->declaration->type);
  scope()->addSymbol(symbol);
}

void Binder::bind(TypenameTypeParameterAST* ast, int index, int depth) {
  auto location = ast->identifier ? ast->identifierLoc : ast->classKeyLoc;

  auto symbol = control()->newTypeParameterSymbol(scope(), location, index,
                                                  depth, ast->isPack);
  ast->symbol = symbol;
  ast->index = index;
  ast->depth = depth;

  symbol->setName(ast->identifier);
  scope()->addSymbol(symbol);
}

void Binder::bind(ConstraintTypeParameterAST* ast, int index, int depth) {
  const auto isParameterPack = static_cast<bool>(ast->ellipsisLoc);
  auto symbol = control()->newConstraintTypeParameterSymbol(
      scope(), ast->identifierLoc, index, depth, isParameterPack);
  symbol->setName(ast->identifier);
  symbol->setTypeConstraint(ast->typeConstraint);
  ast->symbol = symbol;
  ast->index = index;
  ast->depth = depth;
  scope()->addSymbol(symbol);
}

void Binder::bind(TemplateTypeParameterAST* ast, int index, int depth) {
  std::vector<const Type*> parameters;

  for (auto param : ListView{ast->templateParameterList}) {
    if (param->symbol && param->symbol->type()) {
      parameters.push_back(param->symbol->type());
    }
  }

  auto symbol = control()->newTemplateTypeParameterSymbol(
      scope(), ast->templateLoc, index, depth, ast->isPack,
      std::move(parameters));

  symbol->setName(ast->identifier);

  ast->symbol = symbol;
  ast->index = index;
  ast->depth = depth;

  scope()->addSymbol(symbol);
}

void Binder::bind(ConceptDefinitionAST* ast) {
  auto templateParameters = currentTemplateParameters();

  auto symbol =
      control()->newConceptSymbol(declaringScope(), ast->identifierLoc);
  symbol->setName(ast->identifier);
  if (templateParameters) {
    symbol->setTemplateParameters(templateParameters);
  }
  ast->symbol = symbol;

  declaringScope()->addSymbol(symbol);
}

void Binder::bind(DeductionGuideAST* ast,
                  TemplateDeclarationAST* templateHead) {
  auto templateParameters = currentTemplateParameters();

  auto symbol =
      control()->newDeductionGuideSymbol(declaringScope(), ast->identifierLoc);
  symbol->setName(ast->identifier);
  if (templateParameters) {
    symbol->setTemplateParameters(templateParameters);
  }
  if (ast->explicitSpecifier) {
    symbol->setExplicit(true);
  }
  symbol->setDeclaration(ast);
  if (templateHead) symbol->setTemplateDeclaration(templateHead);
  ast->symbol = symbol;

  auto parameterTypes =
      getParameterTypes(unit_, ast->parameterDeclarationClause);
  bool isVariadic = false;

  if (auto params = ast->parameterDeclarationClause) {
    isVariadic = params->isVariadic;
  }

  auto primaryTemplate = ast->templateId
                             ? symbol_cast<ClassSymbol>(ast->templateId->symbol)
                             : nullptr;
  if (!primaryTemplate) return;

  ClassSymbol* deducedClassSymbol = primaryTemplate;

  if (auto templateDecl = primaryTemplate->templateDeclaration();
      templateDecl && ast->templateId->templateArgumentList) {
    const bool dependent = std::ranges::any_of(
        ListView{ast->templateId->templateArgumentList},
        [&](TemplateArgumentAST* argument) {
          return isDependentTemplateArgument(unit_, argument);
        });
    auto templateArgs =
        Substitution(unit_, templateDecl, ast->templateId->templateArgumentList)
            .templateArguments();

    if (!templateArgs.empty()) {
      if (auto cached =
              primaryTemplate->findSpecialization(unit_, templateArgs)) {
        deducedClassSymbol = symbol_cast<ClassSymbol>(cached);
      } else {
        auto parentScope = primaryTemplate->parent();
        auto spec =
            control()->newClassSymbol(parentScope, primaryTemplate->location());
        spec->setName(primaryTemplate->name());
        spec->setType(control()->getClassType(spec));
        primaryTemplate->addSpecialization(unit_, std::move(templateArgs),
                                           spec);
        primaryTemplate->setPendingInstantiation(
            spec, ast->templateId->templateArgumentList,
            ast->templateId->identifierLoc, !dependent);
        deducedClassSymbol = spec;
      }
    }
  }

  const Type* returnType =
      deducedClassSymbol ? deducedClassSymbol->type() : nullptr;
  if (!returnType) return;

  auto funcType = control()->getFunctionType(
      returnType, std::move(parameterTypes), isVariadic, {}, {}, false);
  symbol->setType(funcType);

  primaryTemplate->addDeductionGuide(symbol);
}

auto Binder::lookupCaptureName(ScopeSymbol* scope, const Name* name)
    -> Symbol* {
  auto enclosingClosureCaptureField = [](ScopeSymbol* enclosingScope,
                                         const Name* name) -> FieldSymbol* {
    auto lambda = symbol_cast<LambdaSymbol>(enclosingScope);
    if (!lambda || !lambda->closureType()) return nullptr;
    for (auto candidate : lambda->closureType()->find(name)) {
      if (auto field = symbol_cast<FieldSymbol>(candidate)) return field;
    }
    return nullptr;
  };

  for (auto current = scope; current; current = current->parent()) {
    if (auto field = enclosingClosureCaptureField(current, name)) return field;
    for (auto candidate : current->find(name)) return candidate;
  }
  return nullptr;
}

auto Binder::isCapturableLocalEntity(Symbol* symbol) -> bool {
  if (!symbol) return false;
  if (symbol_cast<ParameterSymbol>(symbol)) return true;
  if (symbol_cast<ParameterPackSymbol>(symbol)) return true;
  if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    auto closure = symbol_cast<ClassSymbol>(field->parent());
    return closure && closure->isClosureType();
  }
  auto var = symbol_cast<VariableSymbol>(symbol);
  if (!var) return false;
  if (var->isStatic() || var->isExtern() || var->isThreadLocal()) return false;
  return var->enclosingFunction() != nullptr;
}

auto Binder::checkCapturedEntity(Symbol* symbol, const Identifier* identifier,
                                 SourceLocation loc) -> bool {
  if (isCapturableLocalEntity(symbol)) return true;

  if (symbol_cast<FieldSymbol>(symbol)) {
    error(loc, std::format("class member '{}' cannot appear in capture list "
                           "as it is not a variable",
                           identifier->name()));
  } else if (symbol_cast<VariableSymbol>(symbol)) {
    error(loc, std::format("'{}' cannot be captured because it does not have "
                           "automatic storage duration",
                           identifier->name()));
  } else {
    error(loc, std::format("'{}' in capture list does not name a variable",
                           identifier->name()));
  }

  return false;
}

namespace {
[[nodiscard]] auto hasNoThis(FunctionSymbol* function) -> bool {
  return function->isStatic() || function->hasExplicitObjectParameter();
}

[[nodiscard]] auto thisCvQualifiers(FunctionSymbol* function) -> CvQualifiers {
  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return CvQualifiers::kNone;
  return functionType->cvQualifiers();
}
}  // namespace

auto Binder::enclosingThisType(ScopeSymbol* scope) -> const Type* {
  for (auto current = scope; current; current = current->parent()) {
    if (auto parameters = symbol_cast<FunctionParametersSymbol>(current);
        parameters && !symbol_cast<FunctionSymbol>(parameters->parent())) {
      if (auto cls = parameters->enclosingClass())
        return control()->getPointerType(
            traits.add_cv(cls->type(), parameters->cvQualifiers()));
    }

    auto function = symbol_cast<FunctionSymbol>(current);
    auto classSymbol = symbol_cast<ClassSymbol>(
        function ? function->parent() : static_cast<Symbol*>(current));
    if (!classSymbol) {
      if (function) return nullptr;
      continue;
    }

    if (classSymbol->isClosureType()) {
      if (auto capturedThis = classSymbol->capturedThisField())
        return capturedThis->type();
      continue;
    }

    if (!function) return control()->getPointerType(classSymbol->type());

    if (hasNoThis(function)) return nullptr;

    return control()->getPointerType(
        traits.add_cv(classSymbol->type(), thisCvQualifiers(function)));
  }
  return nullptr;
}

namespace {
[[nodiscard]] auto namesNonStaticMember(Symbol* symbol) -> bool {
  if (!symbol) return false;

  if (auto field = symbol_cast<FieldSymbol>(symbol)) return !field->isStatic();

  return std::ranges::any_of(
      views::each_function(symbol), [](FunctionSymbol* function) {
        return function->isImplicitObjectMemberFunction();
      });
}

[[nodiscard]] auto formsPointerToMember(UnaryExpressionAST* ast) -> bool {
  if (ast->op != TokenKind::T_AMP) return false;
  auto id = ast_cast<IdExpressionAST>(ast->expression);
  return id && id->nestedNameSpecifier && namesNonStaticMember(id->symbol);
}

struct ThisUseFinder : ASTVisitor {
  bool found = false;

  void visit(ThisExpressionAST*) override { found = true; }
  void visit(DerefThisLambdaCaptureAST*) override { found = true; }

  void visit(IdExpressionAST* ast) override {
    if (namesNonStaticMember(ast->symbol)) found = true;
    ASTVisitor::visit(ast);
  }

  void visit(UnaryExpressionAST* ast) override {
    if (formsPointerToMember(ast)) return;
    ASTVisitor::visit(ast);
  }

  void visit(LambdaExpressionAST* ast) override {
    for (auto capture : ListView{ast->captureList}) accept(capture);
  }
};

[[nodiscard]] auto isDeclaredInsideClosure(Symbol* symbol, ClassSymbol* closure,
                                           LambdaSymbol* lambda) -> bool {
  for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
    if (scope == closure || scope == lambda) return true;
  }
  return false;
}

struct OdrUsedLocalFinder : ASTVisitor {
  std::vector<IdExpressionAST*> uses;

  void visit(IdExpressionAST* ast) override {
    if (ast->nestedNameSpecifier) return;
    if (ast->symbol) uses.push_back(ast);
  }

  void visit(ImplicitCastExpressionAST* ast) override {
    if (ast->castKind == ImplicitCastKind::kLValueToRValueConversion) {
      auto id = ast_cast<IdExpressionAST>(ast->expression);
      if (id && isUsableInConstantExpressions(id->symbol)) return;
    }
    ASTVisitor::visit(ast);
  }

  void visit(SizeofExpressionAST*) override {}
  void visit(SizeofPackExpressionAST*) override {}
  void visit(AlignofExpressionAST*) override {}
  void visit(NoexceptExpressionAST*) override {}
  void visit(DecltypeSpecifierAST*) override {}
  void visit(RequiresExpressionAST*) override {}
};

[[nodiscard]] auto bitfieldDeclaratorOf(DeclaratorAST* declarator)
    -> BitfieldDeclaratorAST* {
  if (!declarator) return nullptr;
  return ast_cast<BitfieldDeclaratorAST>(declarator->coreDeclarator);
}

[[nodiscard]] auto declaratorIdAttributes(DeclaratorAST* declarator)
    -> List<AttributeSpecifierAST*>* {
  if (auto bitfield = bitfieldDeclaratorOf(declarator))
    return bitfield->attributeList;
  auto declaratorId = getDeclaratorId(declarator);
  if (!declaratorId) return nullptr;
  return declaratorId->attributeList;
}

[[nodiscard]] auto terminatingAttributes(DeclaratorAST* declarator)
    -> List<AttributeSpecifierAST*>* {
  if (!declarator) return nullptr;
  if (auto bitfield = bitfieldDeclaratorOf(declarator))
    return bitfield->trailingAttributeList;
  DeclaratorChunkAST* lastChunk = nullptr;
  for (auto chunk : ListView{declarator->declaratorChunkList})
    lastChunk = chunk;
  if (!lastChunk) return nullptr;
  return visit([](auto chunk) { return chunk->attributeList; }, lastChunk);
}

[[nodiscard]] auto abiTagsOf(Symbol* symbol, const Attribute& abiTag)
    -> std::vector<const Identifier*> {
  if (!abiTag.arguments.empty()) return abiTag.arguments;
  auto ns = symbol_cast<NamespaceSymbol>(symbol);
  if (!ns || !ns->isInline()) return {};
  auto name = name_cast<Identifier>(ns->name());
  if (!name) return {};
  return {name};
}
}  // namespace

void Binder::applyFunctionDefinitionKind(FunctionSymbol* functionSymbol,
                                         FunctionBodyAST* functionBody) {
  if (!functionSymbol) return;

  if (ast_cast<DeleteFunctionBodyAST>(functionBody)) {
    functionSymbol->setDeleted(true);
    return;
  }

  if (!ast_cast<DefaultFunctionBodyAST>(functionBody)) return;

  functionSymbol->setDefaulted(true);

  const auto isFirstDeclaration = functionSymbol->canonical() == functionSymbol;

  if (isFirstDeclaration) functionSymbol->setConstexpr(true);
}

void Binder::applyDeclarationAttributes(
    Symbol* symbol, List<AttributeSpecifierAST*>* attributes,
    DeclaratorAST* declarator) {
  if (!symbol) return;
  auto entityAttributes =
      collectAttributes(unit_, declaratorIdAttributes(declarator));
  auto gnuAttributes =
      collectGnuAttributes(unit_, terminatingAttributes(declarator));
  auto collected = collectAttributes(unit_, attributes);
  collected = mergeAttributes(std::move(collected), &entityAttributes);
  collected = mergeAttributes(std::move(collected), &gnuAttributes);
  applyAttributeMap(symbol, std::move(collected));

  applyAlignedAttribute(symbol, attributes);
  applyAlignedAttribute(symbol, declaratorIdAttributes(declarator));
  applyAlignedAttribute(symbol, terminatingAttributes(declarator));
}

void Binder::inheritDeclarationAttributes(Symbol* symbol, Symbol* pattern) {
  if (!symbol || !pattern) return;
  auto attributes = pattern->attributes();
  if (!attributes) return;
  applyAttributeMap(symbol, *attributes);
}

void Binder::applyAttributeMap(Symbol* symbol, AttributeMap collected) {
  if (!symbol || collected.empty()) return;

  auto canonical = symbol->canonical();
  if (!canonical) canonical = symbol;

  auto merged = control()->getAttributes(
      mergeAttributes(collected, canonical->attributes()));

  symbol->setAttributes(merged);
  canonical->setAttributes(merged);

  if (auto tags = findAttribute(merged, "abi_tag");
      tags && canonical->abiTags().empty()) {
    symbol->setAbiTags(control()->getAbiTags(abiTagsOf(symbol, *tags)));
    canonical->setAbiTags(symbol->abiTagList());
  }

  const auto isNodiscard = findAttribute(merged, "nodiscard") ||
                           findAttribute(merged, "warn_unused_result");
  symbol->setNodiscard(isNodiscard);
  canonical->setNodiscard(isNodiscard);

  if (findAttribute(merged, "used")) {
    symbol->setUsed(true);
    canonical->setUsed(true);
  }

  if (findAttribute(merged, "exclude_from_explicit_instantiation")) {
    symbol->setExcludedFromExplicitInstantiation(true);
    canonical->setExcludedFromExplicitInstantiation(true);
  }

  if (findAttribute(merged, "trivial_abi")) {
    symbol->setTrivialAbi(true);
    canonical->setTrivialAbi(true);
  }

  auto function = symbol_cast<FunctionSymbol>(symbol);
  if (!function) return;

  if (findAttribute(merged, "noreturn")) {
    function->setNoReturn(true);
    function->canonical()->setNoReturn(true);
  }

  if (findAttribute(merged, "nothrow")) {
    function->setExceptionSpecifier(true);
    setFunctionNoexcept(control(), function, true);

    auto canonicalFunction = function->canonical();
    if (canonicalFunction != function) {
      canonicalFunction->setExceptionSpecifier(true);
      setFunctionNoexcept(control(), canonicalFunction, true);
    }
  }

  applyWasmFunctionAttributes(function, merged);
}

void Binder::applyWasmFunctionAttributes(FunctionSymbol* function,
                                         const AttributeMap* attributes) {
  if (!control()->memoryLayout()->isWebAssembly()) return;

  auto canonical = function->canonical();

  auto apply = [&](std::string_view name,
                   void (FunctionSymbol::*setter)(const Identifier*)) {
    auto argument = attributeArgument(attributes, name);
    if (!argument) return;
    (function->*setter)(argument);
    (canonical->*setter)(argument);
  };

  apply("import_module", &FunctionSymbol::setImportModule);
  apply("import_name", &FunctionSymbol::setImportName);
  apply("export_name", &FunctionSymbol::setExportName);

  if (!function->exportName()) return;

  function->setUsed(true);
  canonical->setUsed(true);
}

void Binder::applyDeclarationAttributes(SimpleDeclarationAST* ast) {
  if (!ast) return;

  for (auto initDeclarator : ListView{ast->initDeclaratorList}) {
    applyDeclarationAttributes(initDeclarator->symbol, ast->attributeList,
                               initDeclarator->declarator);
  }
}

auto Binder::usesImplicitThis(StatementAST* stmt) -> bool {
  if (!stmt) return false;
  ThisUseFinder finder;
  finder.accept(stmt);
  return finder.found;
}

void Binder::initializeCapturedField(FieldSymbol* field, ScopeSymbol* scope,
                                     ExpressionAST*& initializer,
                                     InitializationKind kind) {
  TypeChecker check{unit_};
  check.setScope(scope);
  check.setReportErrors(reportErrors());
  if (auto constructor = check.check_member_initialization(
          field, initializer, kind, ArrayCopyPolicy::kElementwiseCopyAllowed))
    field->setConstructor(constructor);
}

auto Binder::declareCaptureField(ClassSymbol* closure, const Identifier* name,
                                 const Type* type, SourceLocation loc)
    -> FieldSymbol* {
  auto field = control()->newFieldSymbol(closure, loc);
  field->setName(name);
  field->setType(type);
  if (auto alignment = control()->memoryLayout()->alignmentOf(type)) {
    field->setAlignment(alignment.value());
  }
  closure->addSymbol(field);
  return field;
}

auto Binder::lookupCapturedEntity(ScopeSymbol* scope,
                                  const Identifier* identifier,
                                  SourceLocation loc) -> Symbol* {
  auto entity = lookupCaptureName(scope, identifier);
  if (!entity) {
    error(loc,
          std::format("use of undeclared identifier '{}'", identifier->name()));
    return nullptr;
  }
  if (!checkCapturedEntity(entity, identifier, loc)) return nullptr;
  return entity;
}

auto Binder::captureEntity(ClassSymbol* closure, Symbol* entity,
                           const Identifier* name, bool byReference,
                           ScopeSymbol* scope, SourceLocation loc)
    -> EntityCapture {
  auto ar = unit_->arena();
  auto entityType = traits.remove_reference(entity->type());

  auto reference = IdExpressionAST::create(ar);
  reference->unqualifiedId = NameIdAST::create(ar, name);
  reference->symbol = entity;
  reference->type = entityType;
  reference->valueCategory = ValueCategory::kLValue;

  auto fieldType =
      byReference ? control()->getLvalueReferenceType(entityType) : entityType;

  EntityCapture capture{declareCaptureField(closure, name, fieldType, loc),
                        reference};
  initializeCapturedField(capture.field, scope, capture.initializer,
                          InitializationKind::kDirectInitialization);
  return capture;
}

auto Binder::captureThis(ClassSymbol* closure, const Type* thisType,
                         ScopeSymbol* scope, SourceLocation loc)
    -> EntityCapture {
  EntityCapture capture{
      declareCaptureField(closure, control()->getIdentifier("__this"), thisType,
                          loc),
      ThisExpressionAST::create(unit_->arena(), loc, ValueCategory::kPrValue,
                                thisType)};
  closure->setCapturedThisField(capture.field);
  initializeCapturedField(capture.field, scope, capture.initializer,
                          InitializationKind::kDirectInitialization);
  return capture;
}

auto Binder::addImplicitThisCapture(ClassSymbol* classSymbol,
                                    const Type* thisType, SourceLocation loc)
    -> ThisLambdaCaptureAST* {
  auto capture = captureThis(classSymbol, thisType, scope(), loc);
  return ThisLambdaCaptureAST::create(unit_->arena(), loc, capture.initializer,
                                      capture.field);
}

void Binder::addImplicitCaptures(LambdaExpressionAST* ast,
                                 ClassSymbol* classSymbol) {
  auto ar = unit_->arena();
  auto loc = ast->lbracketLoc;
  const auto hasCaptureDefault = ast->captureDefault != TokenKind::T_EOF_SYMBOL;
  const auto byReference = ast->captureDefault != TokenKind::T_EQUAL;

  OdrUsedLocalFinder finder;
  finder.accept(ast->statement);

  std::vector<const Identifier*> explicitlyCaptured;
  for (auto captureNode : ListView{ast->captureList}) {
    if (auto identifier = capture_identifier(captureNode))
      explicitlyCaptured.push_back(identifier);
  }

  auto tail = &ast->captureList;
  while (*tail) tail = &(*tail)->next;

  std::vector<std::pair<Symbol*, FieldSymbol*>> captured;
  std::vector<Symbol*> reportedUncapturable;

  for (auto use : finder.uses) {
    auto outerSymbol = use->symbol;

    auto known = std::ranges::find(captured, outerSymbol,
                                   &std::pair<Symbol*, FieldSymbol*>::first);
    if (known != captured.end()) {
      use->symbol = known->second;
      continue;
    }

    if (!isCapturableLocalEntity(outerSymbol)) continue;
    if (isDeclaredInsideClosure(outerSymbol, classSymbol, ast->symbol))
      continue;
    if (!outerSymbol->type()) continue;

    auto identifier = name_cast<Identifier>(outerSymbol->name());
    if (!identifier) continue;
    if (std::ranges::contains(explicitlyCaptured, identifier)) continue;

    if (!hasCaptureDefault) {
      if (std::ranges::contains(reportedUncapturable, outerSymbol)) continue;
      reportedUncapturable.push_back(outerSymbol);
      error(use->firstSourceLocation(),
            std::format("variable '{}' cannot be implicitly captured in a "
                        "lambda with no capture-default specified",
                        identifier->name()));
      continue;
    }

    auto capture = captureEntity(classSymbol, outerSymbol, identifier,
                                 byReference, scope(), loc);
    captured.emplace_back(outerSymbol, capture.field);

    *tail = make_list_node(
        ar, implicitEntityCapture(identifier, capture, byReference, loc));
    tail = &(*tail)->next;

    use->symbol = capture.field;
  }

  if (captured.empty()) return;

  auto status = buildRecordLayout(classSymbol);
  if (!status.has_value()) error(loc, status.error());
}

auto Binder::implicitEntityCapture(const Identifier* identifier,
                                   const EntityCapture& capture,
                                   bool byReference, SourceLocation loc)
    -> LambdaCaptureAST* {
  auto ar = unit_->arena();

  if (!byReference) {
    auto simple = SimpleLambdaCaptureAST::create(ar);
    simple->identifierLoc = loc;
    simple->identifier = identifier;
    simple->initializer = capture.initializer;
    simple->symbol = capture.field;
    return simple;
  }

  auto ref = RefLambdaCaptureAST::create(ar);
  ref->ampLoc = loc;
  ref->identifierLoc = loc;
  ref->identifier = identifier;
  ref->initializer = capture.initializer;
  ref->symbol = capture.field;
  return ref;
}

void Binder::bind(LambdaExpressionAST* ast) {
  auto parentScope = declaringScope();
  auto symbol = control()->newLambdaSymbol(parentScope, ast->lbracketLoc);
  ast->symbol = symbol;

  symbol->setInTemplate(inTemplate());

  setScope(symbol);
}

auto Binder::deducedInitCaptureType(const InitCapture& capture) -> const Type* {
  if (!capture.initializer) return nullptr;
  TypeChecker check{unit_};
  check.setScope(scope_);
  check.setReportErrors(reportErrors_);
  return check.deduceDeclaredPlaceholderType(
      capture.declaredType, capture.initializer, capture.location);
}

auto Binder::declaredInitCapture(ScopeSymbol* lambdaScope,
                                 const Identifier* name) -> VariableSymbol* {
  for (auto candidate : lambdaScope->find(name)) {
    if (auto variable = symbol_cast<VariableSymbol>(candidate)) return variable;
  }
  return nullptr;
}

auto Binder::initCapture(LambdaCaptureAST* captureNode)
    -> std::optional<InitCapture> {
  if (auto initCap = ast_cast<InitLambdaCaptureAST>(captureNode)) {
    return InitCapture{.name = initCap->identifier,
                       .declaredType = control()->getAutoType(),
                       .initializer = initCap->initializer,
                       .location = initCap->identifierLoc,
                       .isPack = static_cast<bool>(initCap->ellipsisLoc)};
  }

  if (auto refInitCap = ast_cast<RefInitLambdaCaptureAST>(captureNode)) {
    return InitCapture{.name = refInitCap->identifier,
                       .declaredType = control()->getLvalueReferenceType(
                           control()->getAutoType()),
                       .initializer = refInitCap->initializer,
                       .location = refInitCap->identifierLoc,
                       .isPack = static_cast<bool>(refInitCap->ellipsisLoc)};
  }

  return std::nullopt;
}

void Binder::declareInitCapturesInLambdaScope(LambdaExpressionAST* ast) {
  for (auto captureNode : ListView{ast->captureList}) {
    auto capture = initCapture(captureNode);
    if (!capture || !capture->name) continue;

    auto variable = control()->newVariableSymbol(
        ast->symbol, captureNode->firstSourceLocation());
    variable->setName(capture->name);
    auto deduced = deducedInitCaptureType(*capture);
    variable->setType(deduced ? deduced : capture->declaredType);
    ast->symbol->addSymbol(variable);
  }
}

namespace {

[[nodiscard]] auto callOperatorQualifiers(LambdaSymbol* lambda,
                                          bool hasExplicitObjectParameter)
    -> CvQualifiers {
  if (lambda->isMutable() || lambda->isStatic()) return CvQualifiers::kNone;
  if (hasExplicitObjectParameter) return CvQualifiers::kNone;
  return CvQualifiers::kConst;
}

}  // namespace

void Binder::complete(LambdaExpressionAST* ast) {
  if (auto params = ast->parameterDeclarationClause) {
    auto lambdaScope = ast->symbol;
    lambdaScope->addSymbol(params->functionParametersSymbol);
    setScope(params->functionParametersSymbol);
  } else {
    setScope(ast->symbol);
  }

  auto parentScope = ast->symbol->parent();
  parentScope->addSymbol(ast->symbol);

  const Type* returnType = control()->getAutoType();
  auto parameterTypes =
      getParameterTypes(unit_, ast->parameterDeclarationClause);
  bool isVariadic = false;

  if (auto params = ast->parameterDeclarationClause) {
    isVariadic = params->isVariadic;
  }

  const bool isNoexcept =
      exceptionSpecifierIsNoexcept(unit_, ast->exceptionSpecifier);

  if (ast->trailingReturnType && ast->trailingReturnType->typeId) {
    returnType = ast->trailingReturnType->typeId->type;
  }

  const auto hasExplicitObjectParameter =
      declaresExplicitObjectParameter(ast->parameterDeclarationClause);

  auto funcType = control()->getFunctionType(
      returnType, std::move(parameterTypes), isVariadic,
      callOperatorQualifiers(ast->symbol, hasExplicitObjectParameter), {},
      isNoexcept);
  ast->symbol->setType(funcType);

  const bool inDependentContext =
      isEnclosedInDependentTemplate(unit_, ast->symbol->parent(),
                                    /*stopAtConcreteSpecialization=*/true) ||
      ast->symbol->isInTemplate();

  if (isCxx()) declareInitCapturesInLambdaScope(ast);

  if (isCxx() && !inDependentContext) {
    auto classSymbol = control()->newClassSymbol(parentScope, ast->lbracketLoc);

    auto operatorCallName = control()->getOperatorId(TokenKind::T_LPAREN);
    auto operatorFunc = declareClosureMemberFunction(
        classSymbol, operatorCallName, funcType, ast->lbracketLoc);
    operatorFunc->setTrailingRequiresClause(ast->requiresClause);
    operatorFunc->setStatic(ast->symbol->isStatic());
    operatorFunc->setExplicitObjectParameter(hasExplicitObjectParameter);

    if (auto lambdaParams = ast->parameterDeclarationClause) {
      if (lambdaParams->functionParametersSymbol) {
        operatorFunc->addSymbol(lambdaParams->functionParametersSymbol);
      }
    }

    if (ast->symbol->isTemplate()) {
      auto ar = unit_->arena();
      auto templateParamsSymbol = control()->newTemplateParametersSymbol(
          operatorFunc, ast->lbracketLoc);
      for (auto p : ListView{ast->templateParameterList}) {
        if (p && p->symbol) templateParamsSymbol->addSymbol(p->symbol);
      }
      int depth = ast->templateParameterList
                      ? ast->templateParameterList->value->depth
                      : 0;
      auto templateDecl = TemplateDeclarationAST::create(
          ar, ast->templateParameterList, ast->templateRequiresClause,
          /*declaration=*/nullptr, templateParamsSymbol, depth);
      operatorFunc->setTemplateParameters(templateParamsSymbol);
      operatorFunc->setTemplateDeclaration(templateDecl);
    }

    classSymbol->setIsClosureType(true);
    ast->symbol->setClosureType(classSymbol);
    classSymbol->setClosureDiscriminator(nextClosureNumber(classSymbol));
    classSymbol->setHasLambdaCapture(ast->captureDefault !=
                                         TokenKind::T_EOF_SYMBOL ||
                                     ast->captureList != nullptr);

    for (auto captureNode : ListView{ast->captureList}) {
      auto captureLoc = captureNode->firstSourceLocation();
      auto ar = unit_->arena();

      if (auto simple = ast_cast<SimpleLambdaCaptureAST>(captureNode)) {
        auto entity = lookupCapturedEntity(parentScope, simple->identifier,
                                           simple->identifierLoc);
        if (!entity) continue;
        auto capture = captureEntity(classSymbol, entity, simple->identifier,
                                     false, parentScope, captureLoc);
        simple->initializer = capture.initializer;
        simple->symbol = capture.field;
      } else if (auto ref = ast_cast<RefLambdaCaptureAST>(captureNode)) {
        auto entity = lookupCapturedEntity(parentScope, ref->identifier,
                                           ref->identifierLoc);
        if (!entity) continue;
        auto capture = captureEntity(classSymbol, entity, ref->identifier, true,
                                     parentScope, captureLoc);
        ref->initializer = capture.initializer;
        ref->symbol = capture.field;
      } else if (auto th = ast_cast<ThisLambdaCaptureAST>(captureNode)) {
        auto thisType = enclosingThisType(parentScope);
        if (!thisType) {
          error(captureLoc, "'this' cannot be captured in this context");
          continue;
        }

        auto capture =
            captureThis(classSymbol, thisType, parentScope, th->thisLoc);
        th->initializer = capture.initializer;
        th->symbol = capture.field;
      } else if (auto deref =
                     ast_cast<DerefThisLambdaCaptureAST>(captureNode)) {
        error(captureLoc, "capture of '*this' is not yet supported");
      } else if (auto capture = initCapture(captureNode)) {
        auto variable = declaredInitCapture(ast->symbol, capture->name);
        if (!variable || containsPlaceholderType(variable->type())) continue;
        auto field = declareCaptureField(classSymbol, capture->name,
                                         variable->type(), captureLoc);
        *capture_field_slot(captureNode) = field;
        if (auto initializer = capture_initializer_slot(captureNode)) {
          initializeCapturedField(
              field, parentScope, *initializer,
              Initializer{*initializer}.initializationKind());
        }
      }
    }

    classSymbol->setComplete(true);
    auto status = buildRecordLayout(classSymbol);
    if (!status.has_value()) {
      error(ast->lbracketLoc, status.error());
    }

    ast->type = classSymbol->type();
    ast->valueCategory = ValueCategory::kPrValue;
  }
}

auto Binder::declareClosureMemberFunction(ClassSymbol* classSymbol,
                                          const Name* name, const Type* type,
                                          SourceLocation loc)
    -> FunctionSymbol* {
  auto function = control()->newFunctionSymbol(classSymbol, loc);
  function->setName(name);
  function->setType(type);
  function->setDefined(true);
  function->setConstexpr(true);
  function->setInline(true);
  function->setLanguageLinkage(LanguageKind::kCXX);
  overloadSetFor(classSymbol, name, loc)->addFunction(function);
  return function;
}

void Binder::declareSynthesizedParameters(FunctionSymbol* function,
                                          const FunctionType* functionType,
                                          SourceLocation loc) {
  auto parameters = control()->newFunctionParametersSymbol(function, loc);
  function->addSymbol(parameters);
  if (!functionType) return;

  int index = 0;
  for (auto parameterType : functionType->parameterTypes()) {
    auto parameter = control()->newParameterSymbol(parameters, loc);
    parameter->setName(control()->getIdentifier(std::format("__p{}", index++)));
    parameter->setType(parameterType);
    parameters->addSymbol(parameter);
  }
}

auto Binder::declareClosureInvoker(ClassSymbol* classSymbol,
                                   FunctionSymbol* operatorFunc,
                                   const FunctionType* operatorType,
                                   SourceLocation loc) -> FunctionSymbol* {
  auto pool = unit_->arena();
  operatorType = traits.remove_function_qualifiers(operatorType);

  auto invoker = declareClosureMemberFunction(
      classSymbol, control()->getIdentifier("__invoke"), operatorType, loc);
  invoker->setStatic(true);

  declareSynthesizedParameters(invoker, operatorType, loc);

  List<ExpressionAST*>* arguments = nullptr;
  auto argumentTail = &arguments;
  for (auto parameter :
       views::members(invoker->functionParameters()) | views::parameters) {
    auto reference = IdExpressionAST::create(pool);
    reference->unqualifiedId =
        NameIdAST::create(pool, name_cast<Identifier>(parameter->name()));
    reference->symbol = parameter;
    reference->type = parameter->type();
    reference->valueCategory = ValueCategory::kLValue;

    ExpressionAST* argument = reference;
    (void)TypeChecker{unit_}.implicit_conversion(argument, parameter->type());

    *argumentTail = make_list_node<ExpressionAST>(pool, argument);
    argumentTail = &(*argumentTail)->next;
  }

  auto closureObject = TypeConstructionAST::create(pool);
  closureObject->type = classSymbol->type();
  closureObject->valueCategory = ValueCategory::kPrValue;
  for (auto constructor : classSymbol->declaredConstructors()) {
    auto constructorType = type_cast<FunctionType>(constructor->type());
    if (constructorType && constructorType->parameterTypes().empty()) {
      closureObject->constructorSymbol = constructor;
      break;
    }
  }

  auto callee = MemberExpressionAST::create(pool);
  callee->baseExpression = closureObject;
  callee->accessOp = TokenKind::T_DOT;
  callee->unqualifiedId =
      OperatorFunctionIdAST::create(pool, TokenKind::T_LPAREN);
  callee->symbol = operatorFunc;
  callee->type = operatorType;
  callee->valueCategory = ValueCategory::kLValue;

  auto call = CallExpressionAST::create(pool);
  call->baseExpression = callee;
  call->expressionList = arguments;
  call->type = operatorType->returnType();
  call->valueCategory = ValueCategory::kPrValue;

  StatementAST* bodyStatement = nullptr;
  if (traits.is_void(operatorType->returnType())) {
    auto expressionStatement = ExpressionStatementAST::create(pool);
    expressionStatement->expression = call;
    bodyStatement = expressionStatement;
  } else {
    auto returnStatement = ReturnStatementAST::create(pool);
    returnStatement->expression = call;
    bodyStatement = returnStatement;
  }

  auto block = CompoundStatementAST::create(pool);
  block->statementList = make_list_node<StatementAST>(pool, bodyStatement);

  attachSynthesizedBody(
      invoker, NameIdAST::create(pool, control()->getIdentifier("__invoke")),
      CompoundStatementFunctionBodyAST::create(
          pool, /*memInitializerList=*/nullptr, block));

  return invoker;
}

auto Binder::materializeClosureFunctionPointerConversion(
    ClassSymbol* closureClass, const FunctionType* targetFunctionType)
    -> FunctionSymbol* {
  if (!closureClass || !targetFunctionType) return nullptr;

  closureClass = closureClass->resolvedDefinition();
  if (!closureClass->isClosureType()) return nullptr;
  if (closureClass->hasLambdaCapture()) return nullptr;

  auto pointerType = control()->getPointerType(targetFunctionType);
  auto conversionName = control()->getConversionFunctionId(pointerType);

  for (auto existing : closureClass->find(conversionName)) {
    if (auto function = symbol_cast<FunctionSymbol>(existing)) return function;
  }

  auto pattern = closureClass->functionCallOperator();
  if (!pattern || !pattern->templateDeclaration()) return nullptr;

  auto patternType = type_cast<FunctionType>(pattern->type());
  if (!patternType) return nullptr;

  auto callOperatorTarget = control()->getFunctionType(
      targetFunctionType->returnType(), targetFunctionType->parameterTypes(),
      targetFunctionType->isVariadic(), patternType->cvQualifiers(),
      patternType->refQualifier(),
      targetFunctionType->exceptionSpecification());

  TemplateArgumentDeduction deduction{unit_};
  auto deducedArguments = deduction.deduceFromTargetType(
      pattern, callOperatorTarget, /*explicitTemplateArgs=*/nullptr,
      /*matchReturnType=*/false);
  if (!deducedArguments.has_value()) return nullptr;

  auto instance = ASTRewriter::instantiateOverloadCandidate(
      unit_, *deducedArguments, pattern, closureClass->location(),
      /*argsComplete=*/true);
  if (!instance) return nullptr;

  ASTRewriter::completeDeducedReturnType(unit_, instance);

  auto instanceType = type_cast<FunctionType>(instance->type());
  if (!instanceType) return nullptr;

  auto invoker = declareClosureInvoker(closureClass, instance, instanceType,
                                       closureClass->location());

  declareClosureFunctionPointerConversion(closureClass, invoker, instanceType,
                                          closureClass->location());

  for (auto existing : closureClass->find(
           control()->getConversionFunctionId(control()->getPointerType(
               traits.remove_function_qualifiers(instanceType))))) {
    if (auto function = symbol_cast<FunctionSymbol>(existing)) return function;
  }

  return nullptr;
}

void Binder::declareClosureFunctionPointerConversion(
    ClassSymbol* classSymbol, FunctionSymbol* invoker,
    const FunctionType* operatorType, SourceLocation loc) {
  auto pool = unit_->arena();
  operatorType = traits.remove_function_qualifiers(operatorType);

  auto pointerType = control()->getPointerType(operatorType);

  auto convFunc = declareClosureMemberFunction(
      classSymbol, control()->getConversionFunctionId(pointerType),
      control()->getFunctionType(pointerType, {}, false, CvQualifiers::kConst),
      loc);

  convFunc->addSymbol(control()->newFunctionParametersSymbol(convFunc, loc));

  auto reference = IdExpressionAST::create(pool);
  reference->unqualifiedId =
      NameIdAST::create(pool, name_cast<Identifier>(invoker->name()));
  reference->symbol = invoker;
  reference->type = operatorType;
  reference->valueCategory = ValueCategory::kLValue;

  auto addressOf = UnaryExpressionAST::create(pool);
  addressOf->op = TokenKind::T_AMP;
  addressOf->expression = reference;
  addressOf->type = pointerType;
  addressOf->valueCategory = ValueCategory::kPrValue;

  auto returnStatement = ReturnStatementAST::create(pool);
  returnStatement->expression = addressOf;

  auto block = CompoundStatementAST::create(pool);
  block->statementList = make_list_node<StatementAST>(pool, returnStatement);

  auto conversionId = ConversionFunctionIdAST::create(pool);

  attachSynthesizedBody(convFunc, conversionId,
                        CompoundStatementFunctionBodyAST::create(
                            pool, /*memInitializerList=*/nullptr, block));
}

void Binder::attachSynthesizedBody(FunctionSymbol* function,
                                   UnqualifiedIdAST* id,
                                   FunctionBodyAST* body) {
  auto pool = unit_->arena();

  auto idDeclarator = IdDeclaratorAST::create(pool);
  idDeclarator->unqualifiedId = id;

  auto functionChunk = FunctionDeclaratorChunkAST::create(pool);

  auto declarator = DeclaratorAST::create(
      pool, /*ptrOpList=*/nullptr, idDeclarator,
      make_list_node<DeclaratorChunkAST>(pool, functionChunk));

  auto definition = FunctionDefinitionAST::create(pool);
  definition->declarator = declarator;
  definition->functionBody = body;
  definition->symbol = function;
  function->setDeclaration(definition);
}

auto Binder::closureCallOperatorDefinition(LambdaExpressionAST* ast,
                                           FunctionSymbol* operatorFunc,
                                           CompoundStatementAST* body)
    -> FunctionDefinitionAST* {
  auto ar = unit_->arena();

  auto idDecl = IdDeclaratorAST::create(ar);
  idDecl->unqualifiedId =
      OperatorFunctionIdAST::create(ar, TokenKind::T_LPAREN);

  auto funcChunk = FunctionDeclaratorChunkAST::create(ar);
  if (ast->parameterDeclarationClause) {
    funcChunk->parameterDeclarationClause =
        ast->parameterDeclarationClause->clone(ar);
  }
  if (ast->trailingReturnType) {
    funcChunk->trailingReturnType = ast->trailingReturnType->clone(ar);
  }
  if (ast->exceptionSpecifier) {
    funcChunk->exceptionSpecifier = ast->exceptionSpecifier->clone(ar);
  }
  if (auto opFuncType = type_cast<FunctionType>(operatorFunc->type());
      opFuncType && has_const(opFuncType->cvQualifiers())) {
    funcChunk->cvQualifierList =
        make_list_node<SpecifierAST>(ar, ConstQualifierAST::create(ar));
  }

  auto funcDef = FunctionDefinitionAST::create(ar);
  funcDef->declarator = DeclaratorAST::create(
      ar, /*ptrOpList=*/nullptr, /*coreDeclarator=*/idDecl,
      /*declaratorChunkList=*/
      make_list_node<DeclaratorChunkAST>(ar, funcChunk));
  funcDef->functionBody = CompoundStatementFunctionBodyAST::create(
      ar, /*memInitializerList=*/nullptr, body);
  funcDef->symbol = operatorFunc;

  auto specifiers = &funcDef->declSpecifierList;
  if (operatorFunc->isStatic()) {
    *specifiers =
        make_list_node<SpecifierAST>(ar, StaticSpecifierAST::create(ar));
    specifiers = &(*specifiers)->next;
  }
  if (!ast->trailingReturnType) {
    *specifiers =
        make_list_node<SpecifierAST>(ar, AutoTypeSpecifierAST::create(ar));
  }

  return funcDef;
}

void Binder::completeLambdaBody(LambdaExpressionAST* ast) {
  auto classType = type_cast<ClassType>(ast->type);
  if (!classType) return;

  auto classSymbol = classType->symbol();
  auto ar = unit_->arena();

  if (!classSymbol->capturedThisField() &&
      (ast->captureDefault == TokenKind::T_AMP ||
       ast->captureDefault == TokenKind::T_EQUAL) &&
      usesImplicitThis(ast->statement)) {
    if (auto thisType = enclosingThisType(ast->symbol->parent())) {
      auto capture =
          addImplicitThisCapture(classSymbol, thisType, ast->lbracketLoc);

      auto tail = &ast->captureList;
      while (*tail) tail = &(*tail)->next;
      *tail = make_list_node<LambdaCaptureAST>(ar, capture);

      auto status = buildRecordLayout(classSymbol);
      if (!status.has_value()) {
        error(ast->lbracketLoc, status.error());
      }
    }
  }

  addImplicitCaptures(ast, classSymbol);

  completeClosureType(classSymbol);

  auto operatorFunc = classSymbol->functionCallOperator();
  if (!operatorFunc) return;

  ScopeSymbol* bodyScope = operatorFunc;
  for (auto member : operatorFunc->members()) {
    if (auto params = symbol_cast<FunctionParametersSymbol>(member)) {
      bodyScope = params;
      break;
    }
  }

  auto reboundBody = ast_cast<CompoundStatementAST>(
      ASTRewriter::paste(unit_, bodyScope, ast->statement));

  auto funcDef = closureCallOperatorDefinition(ast, operatorFunc, reboundBody);
  operatorFunc->setDeclaration(funcDef);

  if (auto templateDecl = operatorFunc->templateDeclaration())
    templateDecl->declaration = funcDef;

  if (!ast->trailingReturnType && !ast->symbol->isTemplate())
    finishAutoReturnType(operatorFunc);

  if (!inTemplate() && !ast->symbol->isTemplate())
    checkLambdaBodyWarnings(unit_, ast);

  if (auto opFuncType = type_cast<FunctionType>(operatorFunc->type());
      opFuncType && !ast->symbol->isTemplate() &&
      ast->captureDefault == TokenKind::T_EOF_SYMBOL && !ast->captureList) {
    auto invoker = declareClosureInvoker(classSymbol, operatorFunc, opFuncType,
                                         ast->lbracketLoc);
    declareClosureFunctionPointerConversion(classSymbol, invoker, opFuncType,
                                            ast->lbracketLoc);
  }

  auto closureName = name_cast<Identifier>(classSymbol->name());
  for (auto ctor : classSymbol->declaredConstructors()) {
    if (ctor->declaration()) continue;
    attachSynthesizedBody(ctor, NameIdAST::create(ar, closureName),
                          DefaultFunctionBodyAST::create(ar));
  }
}

void Binder::bind(ParameterDeclarationClauseAST* ast) {
  ast->functionParametersSymbol =
      control()->newFunctionParametersSymbol(scope(), {});
}

void Binder::bind(UsingDirectiveAST* ast, NamespaceSymbol* resolvedNamespace) {
  auto id = ast->unqualifiedId->identifier;

  NamespaceSymbol* namespaceSymbol = nullptr;
  if (ast->nestedNameSpecifier && ast->nestedNameSpecifier->symbol)
    namespaceSymbol =
        qualifiedLookupNamespace(ast->nestedNameSpecifier->symbol, id);
  else
    namespaceSymbol = resolvedNamespace;

  if (namespaceSymbol) {
    scope()->addUsingDirective(namespaceSymbol);
  } else {
    error(ast->unqualifiedId->firstSourceLocation(),
          std::format("'{}' is not a namespace name", id->name()));
  }
}

void Binder::bind(NamespaceAliasDefinitionAST* ast,
                  NamespaceSymbol* resolvedNamespace) {
  auto id = ast->unqualifiedId->identifier;

  NamespaceSymbol* namespaceSymbol = resolvedNamespace;
  if (ast->nestedNameSpecifier && ast->nestedNameSpecifier->symbol)
    namespaceSymbol =
        qualifiedLookupNamespace(ast->nestedNameSpecifier->symbol, id);

  if (!namespaceSymbol) {
    error(ast->unqualifiedId->firstSourceLocation(),
          std::format("'{}' is not a namespace name", id->name()));
    return;
  }

  auto scope = declaringScope();

  for (auto candidate : scope->find(ast->identifier)) {
    auto previous = symbol_cast<NamespaceAliasSymbol>(candidate);
    auto candidateNamespace = resolve_namespace_alias(candidate);
    if (candidateNamespace == namespaceSymbol) {
      if (previous) {
        ast->symbol = previous;
        return;
      }
      continue;
    }
    error(ast->identifierLoc,
          std::format("redefinition of namespace alias '{}'",
                      ast->identifier->name()));
    return;
  }

  auto symbol = control()->newNamespaceAliasSymbol(scope, ast->identifierLoc);
  symbol->setName(ast->identifier);
  symbol->setNamespaceSymbol(namespaceSymbol);
  ast->symbol = symbol;
  scope->addSymbol(symbol);
}

void Binder::bind(UsingEnumDeclarationAST* ast) {
  if (!ast || !ast->enumTypeSpecifier) return;

  auto spec = ast->enumTypeSpecifier;

  if (spec->nestedNameSpecifier && !spec->nestedNameSpecifier->symbol) {
    if (reportUnresolvedNestedNameSpecifier(spec->nestedNameSpecifier)) return;
  }

  const auto checkTemplates = unit_->config().checkTypes;
  auto symbol = resolve(spec->nestedNameSpecifier, spec->unqualifiedId,
                        checkTemplates, spec->symbol);

  symbol = resolve_using_declaration(symbol);

  ScopeSymbol* enumScope = nullptr;
  if (auto enumSymbol = symbol_cast<EnumSymbol>(symbol)) {
    enumScope = enumSymbol;
  } else if (auto scopedEnumSymbol = symbol_cast<ScopedEnumSymbol>(symbol)) {
    enumScope = scopedEnumSymbol;
  } else if (auto typeAlias = symbol_cast<TypeAliasSymbol>(symbol)) {
    auto unqualType = traits.remove_cv(typeAlias->type());
    if (auto enumType = type_cast<EnumType>(unqualType)) {
      enumScope = enumType->symbol();
    } else if (auto scopedEnumType = type_cast<ScopedEnumType>(unqualType)) {
      enumScope = scopedEnumType->symbol();
    }
  }

  if (!enumScope) {
    if (!inTemplate()) {
      auto missingName = get_name(control(), spec->unqualifiedId);
      error(spec->unqualifiedId->firstSourceLocation(),
            std::format("'{}' does not name an enumeration",
                        to_string(missingName)));
    }
    return;
  }

  spec->symbol = enumScope;

  for (auto member : enumScope->members()) {
    if (auto enumerator = symbol_cast<EnumeratorSymbol>(member)) {
      injectUsing(scope(), enumerator->name(), enumerator, ast->usingLoc);
    }
  }
}

void Binder::bind(TypeIdAST* ast, const Decl& decl) {
  ast->type = getDeclaratorType(unit_, ast->declarator, decl.specs.type());
}

auto Binder::declareTypedef(DeclaratorAST* declarator, const Decl& decl)
    -> TypeAliasSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.type());
  auto targetScope = declaringScope();
  auto symbol = control()->newTypeAliasSymbol(targetScope, decl.location());
  applyAccessSpecifier(symbol);
  symbol->setName(name);
  symbol->setType(type);

  addTypeAliasToScope(symbol);

  giveTypedefNameForLinkage(symbol, decl.specs.typeSpecifier());

  return symbol;
}

namespace {
auto arrayBoundToString(const Type* type) -> std::optional<std::string> {
  if (auto bounded = type_cast<BoundedArrayType>(type)) {
    return std::to_string(bounded->size());
  }
  return std::nullopt;
}

auto isEffectivelyUnboundedArray(TranslationUnit* unit, const Type* type)
    -> bool {
  if (!unit || !type) return false;
  if (unit->typeTraits().is_unbounded_array(type)) return true;

  auto unresolved = type_cast<UnresolvedBoundedArrayType>(type);
  if (!unresolved) return false;
  return !arrayBoundToString(type).has_value();
}

auto unqualifiedIdsStructurallyEquivalentForRedeclaration(TranslationUnit* unit,
                                                          UnqualifiedIdAST* a,
                                                          UnqualifiedIdAST* b)
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;

  if (auto aName = ast_cast<NameIdAST>(a)) {
    auto bName = ast_cast<NameIdAST>(b);
    return bName && aName->identifier == bName->identifier;
  }

  auto aTemplateId = ast_cast<SimpleTemplateIdAST>(a);
  auto bTemplateId = ast_cast<SimpleTemplateIdAST>(b);
  if (!aTemplateId || !bTemplateId) return false;
  if (aTemplateId->identifier != bTemplateId->identifier) return false;
  return TemplateEquivalence{unit}.same(aTemplateId->templateArgumentList,
                                        bTemplateId->templateArgumentList);
}

auto nestedNameSpecifiersStructurallyEquivalent(TranslationUnit* unit,
                                                NestedNameSpecifierAST* a,
                                                NestedNameSpecifierAST* b)
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;
  if (auto ta = ast_cast<TemplateNestedNameSpecifierAST>(a)) {
    auto tb = ast_cast<TemplateNestedNameSpecifierAST>(b);
    if (!tb || !ta->templateId || !tb->templateId) return false;
    if (ta->templateId->identifier != tb->templateId->identifier) return false;
    if (!TemplateEquivalence{unit}.same(ta->templateId->templateArgumentList,
                                        tb->templateId->templateArgumentList)) {
      return false;
    }
    return nestedNameSpecifiersStructurallyEquivalent(
        unit, ta->nestedNameSpecifier, tb->nestedNameSpecifier);
  }

  if (a->symbol && b->symbol) {
    if (a->symbol == b->symbol) return true;
    auto aInfo = template_parameter_info(a->symbol);
    auto bInfo = template_parameter_info(b->symbol);
    return aInfo && bInfo && aInfo->index == bInfo->index &&
           aInfo->depth == bInfo->depth && aInfo->isPack == bInfo->isPack;
  }

  if (ast_cast<GlobalNestedNameSpecifierAST>(a)) {
    return ast_cast<GlobalNestedNameSpecifierAST>(b) != nullptr;
  }

  if (auto sa = ast_cast<SimpleNestedNameSpecifierAST>(a)) {
    auto sb = ast_cast<SimpleNestedNameSpecifierAST>(b);
    if (!sb || sa->identifier != sb->identifier) return false;
    return nestedNameSpecifiersStructurallyEquivalent(
        unit, sa->nestedNameSpecifier, sb->nestedNameSpecifier);
  }

  if (auto da = ast_cast<DecltypeNestedNameSpecifierAST>(a)) {
    auto db = ast_cast<DecltypeNestedNameSpecifierAST>(b);
    if (!db || !da->decltypeSpecifier || !db->decltypeSpecifier) return false;
    auto aType = da->decltypeSpecifier->type;
    auto bType = db->decltypeSpecifier->type;
    if (!aType || !bType) return false;
    return unit->typeTraits().is_same(aType, bType);
  }

  return false;
}

auto unresolvedNameTypesStructurallyEquivalent(TranslationUnit* unit,
                                               const UnresolvedNameType* a,
                                               const UnresolvedNameType* b)
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;
  if (!unqualifiedIdsStructurallyEquivalentForRedeclaration(
          unit, a->unqualifiedId(), b->unqualifiedId())) {
    return false;
  }
  return nestedNameSpecifiersStructurallyEquivalent(
      unit, a->nestedNameSpecifier(), b->nestedNameSpecifier());
}

[[nodiscard]] auto indirectElementType(const Type* type) -> const Type* {
  if (auto pointer = type_cast<PointerType>(type))
    return pointer->elementType();
  if (auto reference = type_cast<LvalueReferenceType>(type))
    return reference->elementType();
  if (auto reference = type_cast<RvalueReferenceType>(type))
    return reference->elementType();
  return nullptr;
}

}  // namespace

namespace {

auto redeclarationTypesEquivalent(TranslationUnit* unit,
                                  const Type* existingType,
                                  const Type* incomingType,
                                  bool ignoresArrayBound) -> bool {
  if (!existingType || !incomingType) return false;

  if (unit->typeTraits().is_same(existingType, incomingType)) return true;

  auto existingQual = type_cast<QualType>(existingType);
  auto incomingQual = type_cast<QualType>(incomingType);
  if (existingQual || incomingQual) {
    if (!existingQual || !incomingQual) return false;
    if (existingQual->cvQualifiers() != incomingQual->cvQualifiers())
      return false;
    return redeclarationTypesEquivalent(unit, existingQual->elementType(),
                                        incomingQual->elementType(),
                                        ignoresArrayBound);
  }

  auto existingReferencedType = indirectElementType(existingType);
  auto incomingReferencedType = indirectElementType(incomingType);
  if (existingReferencedType || incomingReferencedType) {
    if (!existingReferencedType || !incomingReferencedType) return false;
    if (existingType->kind() != incomingType->kind()) return false;
    return redeclarationTypesEquivalent(unit, existingReferencedType,
                                        incomingReferencedType,
                                        /*ignoresArrayBound=*/false);
  }

  auto existingExpansion = type_cast<PackExpansionType>(existingType);
  auto incomingExpansion = type_cast<PackExpansionType>(incomingType);
  if (existingExpansion || incomingExpansion) {
    if (!existingExpansion || !incomingExpansion) return false;
    return redeclarationTypesEquivalent(unit, existingExpansion->pattern(),
                                        incomingExpansion->pattern(),
                                        /*ignoresArrayBound=*/false);
  }

  auto existingUnresolved = type_cast<UnresolvedNameType>(existingType);
  auto incomingUnresolved = type_cast<UnresolvedNameType>(incomingType);
  if (existingUnresolved || incomingUnresolved) {
    if (!existingUnresolved || !incomingUnresolved) return false;
    return unresolvedNameTypesStructurallyEquivalent(unit, existingUnresolved,
                                                     incomingUnresolved);
  }

  if (!unit->typeTraits().is_array(existingType)) return false;
  if (!unit->typeTraits().is_array(incomingType)) return false;

  auto existingElementType = unit->typeTraits().get_element_type(existingType);
  auto incomingElementType = unit->typeTraits().get_element_type(incomingType);
  if (!redeclarationTypesEquivalent(unit, existingElementType,
                                    incomingElementType, ignoresArrayBound)) {
    return false;
  }

  if (ignoresArrayBound) {
    if (isEffectivelyUnboundedArray(unit, existingType)) return true;
    if (isEffectivelyUnboundedArray(unit, incomingType)) return true;
  } else if (isEffectivelyUnboundedArray(unit, existingType) !=
             isEffectivelyUnboundedArray(unit, incomingType)) {
    return false;
  }

  auto existingBound = arrayBoundToString(existingType);
  auto incomingBound = arrayBoundToString(incomingType);
  if (!existingBound || !incomingBound) return true;
  return *existingBound == *incomingBound;
}

}  // namespace

auto areRedeclarationTypesCompatible(TranslationUnit* unit,
                                     const Type* existingType,
                                     const Type* incomingType) -> bool {
  if (!unit || !existingType || !incomingType) return false;

  return redeclarationTypesEquivalent(unit, unqualified_type(existingType),
                                      unqualified_type(incomingType));
}

auto areFunctionSignaturesEquivalentForRedeclaration(
    TranslationUnit* unit, const Type* lhs, const Type* rhs,
    TemplateDeclarationAST* lhsHead, TemplateDeclarationAST* rhsHead,
    bool isOutOfLineDeclaration) -> bool {
  if (!unit || !lhs || !rhs) return false;
  if (unit->typeTraits().is_same(lhs, rhs)) return true;

  auto lhsFn = type_cast<FunctionType>(lhs);
  auto rhsFn = type_cast<FunctionType>(rhs);
  if (!lhsFn || !rhsFn) return false;

  if (lhsHead && rhsHead && lhsHead->depth != rhsHead->depth) {
    int ownParameterCount = 0;
    for ([[maybe_unused]] auto parameter :
         ListView{rhsHead->templateParameterList})
      ++ownParameterCount;

    if (TemplateEquivalence{unit}.corresponds(
            lhs, rhs, {lhsHead->depth, rhsHead->depth, ownParameterCount}))
      return true;
  }

  const bool dependentReturnType =
      isOutOfLineDeclaration && (isDependent(unit, lhsFn->returnType()) ||
                                 isDependent(unit, rhsFn->returnType()));
  if (!dependentReturnType &&
      !areRedeclarationTypesCompatible(unit, lhsFn->returnType(),
                                       rhsFn->returnType()))
    return false;
  if (lhsFn->cvQualifiers() != rhsFn->cvQualifiers()) return false;
  if (lhsFn->refQualifier() != rhsFn->refQualifier()) return false;
  if (lhsFn->isVariadic() != rhsFn->isVariadic()) return false;

  const auto& lhsParams = lhsFn->parameterTypes();
  const auto& rhsParams = rhsFn->parameterTypes();
  if (lhsParams.size() != rhsParams.size()) return false;

  for (std::size_t i = 0; i < lhsParams.size(); ++i) {
    if (!areRedeclarationTypesCompatible(unit, lhsParams[i], rhsParams[i])) {
      return false;
    }
  }

  return true;
}

namespace {
auto preferredRedeclarationType(TranslationUnit* unit, const Type* existingType,
                                const Type* incomingType) -> const Type* {
  if (!unit || !existingType || !incomingType) return existingType;
  if (unit->typeTraits().is_same(existingType, incomingType))
    return existingType;

  if (isEffectivelyUnboundedArray(unit, existingType) &&
      unit->typeTraits().is_array(incomingType) &&
      !isEffectivelyUnboundedArray(unit, incomingType) &&
      areRedeclarationTypesCompatible(
          unit, unit->typeTraits().get_element_type(existingType),
          unit->typeTraits().get_element_type(incomingType))) {
    return incomingType;
  }

  return existingType;
}

}  // namespace

namespace {
[[nodiscard]] auto hasPureFinalOverrider(const VTableLayout::Table& table)
    -> bool {
  return std::ranges::any_of(table.slots, [](const VTableLayout::Slot& slot) {
    return slot.function && slot.function->isPure();
  });
}

[[nodiscard]] auto hasPureFinalOverrider(const VTableLayout* vtable) -> bool {
  if (!vtable) return false;
  return std::ranges::any_of(vtable->main.tables,
                             [](const VTableLayout::Table& table) {
                               return hasPureFinalOverrider(table);
                             });
}
}  // namespace

void Binder::computeClassFlags(ClassSymbol* classSymbol) {
  bool polymorphic =
      views::any_function(classSymbol->members(),
                          [](FunctionSymbol* fn) { return fn->isVirtual(); });

  if (!polymorphic) {
    for (auto base : classSymbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (baseClass && baseClass->isPolymorphic()) {
        polymorphic = true;
        break;
      }
    }
  }
  classSymbol->setPolymorphic(polymorphic);

  classSymbol->setAbstract(hasPureFinalOverrider(classSymbol->vtableLayout()));

  auto dtor = classSymbol->destructor();
  classSymbol->setHasVirtualDestructor(dtor && dtor->isVirtual());
}

void Binder::checkRedeclaredAlignment(ClassSymbol* classSymbol, int requested,
                                      SourceLocation loc) {
  const auto declared = classSymbol->explicitAlignment();

  if (declared && declared != requested) {
    error(loc, std::format("redeclaration has a different alignment "
                           "requirement ({} vs {})",
                           requested, declared));
    return;
  }

  if (!declared && classSymbol->isComplete()) {
    error(loc,
          "'alignas' must be specified on the definition if it is specified "
          "on any declaration");
    return;
  }

  classSymbol->setExplicitAlignment(requested);
}

auto Binder::hasDependentAlignment(
    List<AttributeSpecifierAST*>* attributeList) const -> bool {
  for (auto specifier : ListView{attributeList}) {
    if (auto alignas_ = ast_cast<AlignasAttributeAST>(specifier)) {
      if (isDependent(unit_, alignas_->expression)) return true;
      continue;
    }

    if (auto alignas_ = ast_cast<AlignasTypeAttributeAST>(specifier)) {
      if (!alignas_->typeId) continue;
      if (isDependent(unit_, alignas_->typeId->type)) return true;
    }
  }

  return false;
}

auto Binder::validatedAlignment(std::optional<std::intmax_t> value,
                                SourceLocation loc) -> std::optional<int> {
  if (!value) {
    error(loc, "'aligned' attribute requires integer constant");
    return std::nullopt;
  }
  if (*value == 0) return std::nullopt;
  if (*value < 0 || (*value & (*value - 1)) != 0) {
    error(loc, "requested alignment is not a power of 2");
    return std::nullopt;
  }
  return static_cast<int>(*value);
}

auto Binder::alignedAttribute(List<AttributeSpecifierAST*>* attributeList)
    -> std::optional<int> {
  static constexpr AttributeSpelling kAlignedSpellings[] = {
      {AttributeSyntax::kGnu, "", "aligned"},
      {AttributeSyntax::kCxx, "gnu", "aligned"},
  };

  auto attribute =
      findAttributeBySpelling(unit_, attributeList, kAlignedSpellings);
  if (!attribute) return std::nullopt;

  auto clause = attribute.argumentClause;
  if (!clause || !clause->expressionList)
    return static_cast<int>(
        control()->memoryLayout()->alignedAttributeAlignment());

  auto expression = clause->expressionList->value;
  if (isDependent(unit_, expression)) return std::nullopt;

  ASTInterpreter interp{unit_};
  auto value = interp.evaluate(expression);
  if (!value) return validatedAlignment(std::nullopt, attribute.location);
  return validatedAlignment(interp.toInt(*value), attribute.location);
}

void Binder::applyAlignedAttribute(
    Symbol* symbol, List<AttributeSpecifierAST*>* attributeList) {
  auto requested = alignedAttribute(attributeList);
  if (!requested) return;

  if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    field->setExplicitAlignment(
        std::max(field->explicitAlignment(), *requested));
    return;
  }

  if (auto variable = symbol_cast<VariableSymbol>(symbol)) {
    variable->setExplicitAlignment(
        std::max(variable->explicitAlignment(), *requested));
  }
}

auto Binder::explicitAlignment(List<AttributeSpecifierAST*>* attributeList,
                               SourceLocation loc) -> std::optional<int> {
  std::optional<int> strictest;

  auto require = [&](std::optional<std::intmax_t> value, SourceLocation at) {
    auto alignment = validatedAlignment(value, at);
    if (!alignment) return;
    if (!strictest || *strictest < *alignment) strictest = *alignment;
  };

  for (auto specifier : ListView{attributeList}) {
    if (auto alignas_ = ast_cast<AlignasAttributeAST>(specifier)) {
      auto at = alignas_->alignasLoc ? alignas_->alignasLoc : loc;
      if (isDependent(unit_, alignas_->expression)) continue;
      ASTInterpreter interp{unit_};
      auto value = interp.evaluate(alignas_->expression);
      if (!value) {
        require(std::nullopt, at);
        continue;
      }
      require(interp.toInt(*value), at);
      continue;
    }

    if (auto alignas_ = ast_cast<AlignasTypeAttributeAST>(specifier)) {
      auto at = alignas_->alignasLoc ? alignas_->alignasLoc : loc;
      if (!alignas_->typeId) continue;
      auto type = alignas_->typeId->type;
      if (!type || isDependent(unit_, type)) continue;
      auto alignment = control()->memoryLayout()->alignmentOf(type);
      if (!alignment) {
        require(std::nullopt, at);
        continue;
      }
      require(static_cast<std::intmax_t>(*alignment), at);
    }
  }

  return strictest;
}

auto Binder::checkExplicitAlignment(int requested, const Type* type,
                                    SourceLocation loc) -> bool {
  auto natural = control()->memoryLayout()->alignmentOf(type);
  if (!natural) return true;
  if (static_cast<int>(*natural) <= requested) return true;

  error(loc, std::format("requested alignment is less than minimum alignment "
                         "of {} for type '{}'",
                         *natural, to_string(type)));
  return false;
}

void Binder::applyExplicitAlignment(FieldSymbol* field, const Decl& decl) {
  applyAlignedAttribute(field, decl.specs.attributeList);

  auto requested = explicitAlignment(decl.specs.attributeList, decl.location());
  if (!requested) return;

  if (field->isBitField()) {
    error(decl.location(),
          "'alignas' attribute cannot be applied to a bit-field");
    return;
  }

  if (!checkExplicitAlignment(*requested, field->type(), decl.location()))
    return;

  field->setExplicitAlignment(*requested);
}

void Binder::applyExplicitAlignment(VariableSymbol* variable,
                                    const Decl& decl) {
  applyAlignedAttribute(variable, decl.specs.attributeList);

  auto requested = explicitAlignment(decl.specs.attributeList, decl.location());
  if (!requested) return;

  if (!checkExplicitAlignment(*requested, variable->type(), decl.location()))
    return;

  variable->setExplicitAlignment(*requested);
}

auto Binder::declareField(DeclaratorAST* declarator, const Decl& decl)
    -> FieldSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.type());

  if (name) {
    for (auto candidate : scope()->find(name)) {
      auto existingField = symbol_cast<FieldSymbol>(candidate);
      const bool collides = existingField ||
                            symbol_cast<FunctionSymbol>(candidate) ||
                            symbol_cast<OverloadSetSymbol>(candidate) ||
                            symbol_cast<EnumeratorSymbol>(candidate);
      if (!collides) continue;

      error(decl.location(),
            std::format("duplicate member '{}'", to_string(name)));

      if (existingField) return existingField;
      break;
    }
  }

  auto fieldSymbol = control()->newFieldSymbol(scope(), decl.location());
  applySpecifiers(fieldSymbol, decl.specs);
  fieldSymbol->setName(name);
  fieldSymbol->setType(type);
  fieldSymbol->setMutable(decl.specs.isMutable);
  fieldSymbol->setNoUniqueAddress(decl.specs.isNoUniqueAddress);

  if (auto alignment = control()->memoryLayout()->alignmentOf(type)) {
    fieldSymbol->setAlignment(alignment.value());
  }

  if (decl.isBitField()) {
    fieldSymbol->setBitField(true);

    if (!traits.is_integral(type) && !traits.is_enum(type) && !inTemplate() &&
        !isDependent(unit_, type)) {
      error(decl.location(), "bit-field has non-integral type");
    }

    if (decl.bitfieldDeclarator && decl.bitfieldDeclarator->sizeExpression) {
      ASTInterpreter interp{unit_};
      auto value = interp.evaluate(decl.bitfieldDeclarator->sizeExpression);

      if (value) {
        fieldSymbol->setBitFieldWidth(*value);
        if (auto bitWidth = std::get_if<ConstInt>(&*value)) {
          const auto width = bitWidth->toIntMax();
          if (width < 0) {
            error(decl.location(), "bit-field width is negative");
          } else if (width == 0 && name) {
            error(decl.location(), "zero-width bit-field must be unnamed");
          } else if (!inTemplate()) {
            auto typeSize = control()->memoryLayout()->sizeOf(type);
            if (typeSize && width > static_cast<std::intmax_t>(*typeSize) * 8) {
              error(decl.location(),
                    "width of bit-field exceeds width of its type");
            }
          }
        } else {
          error(decl.location(), "bit-field width is not an integer");
        }
      } else if (!inTemplate() &&
                 !isDependent(unit_, decl.bitfieldDeclarator->sizeExpression)) {
        error(decl.location(), "bit-field width is not a constant expression");
      }
    }
  }

  applyExplicitAlignment(fieldSymbol, decl);

  scope()->addSymbol(fieldSymbol);
  return fieldSymbol;
}

void Binder::declareAnonymousField(ClassSpecifierAST* classSpecifier) {
  auto classSymbol = classSpecifier->symbol;
  if (!classSymbol) return;
  if (classSymbol->name()) return;

  auto fieldSymbol =
      control()->newFieldSymbol(scope(), classSymbol->location());
  fieldSymbol->setName(nullptr);
  fieldSymbol->setType(classSymbol->type());
  if (auto alignment =
          control()->memoryLayout()->alignmentOf(classSymbol->type())) {
    fieldSymbol->setAlignment(alignment.value());
  }
  scope()->addSymbol(fieldSymbol);
}

auto Binder::definedStaticDataMember(VariableSymbol* variable) -> FieldSymbol* {
  auto classSymbol = symbol_cast<ClassSymbol>(variable->parent());
  if (!classSymbol) return nullptr;
  for (auto field : classSymbol->find(variable->name()) | views::fields) {
    if (field->definition() == variable) return field;
  }
  return nullptr;
}

namespace {
[[nodiscard]] auto declaresStaticDataMemberInClass(VariableSymbol* variable)
    -> bool {
  if (variable->canonical() != variable) return false;
  if (!variable->parent() || !variable->parent()->isClass()) return false;
  return !Binder::definedStaticDataMember(variable);
}

[[nodiscard]] auto redeclaresConstexprStaticDataMember(VariableSymbol* variable)
    -> bool {
  auto canonical = variable->canonical();
  if (canonical == variable) return false;
  return declaresStaticDataMemberInClass(canonical) && canonical->isConstexpr();
}

[[nodiscard]] auto definingDeclaration(VariableSymbol* canonical)
    -> VariableSymbol* {
  if (auto definition = canonical->definition()) return definition;
  if (Binder::declaresVariableDefinition(canonical)) return canonical;
  return nullptr;
}
}  // namespace

auto Binder::declaresVariableDefinition(VariableSymbol* variable) -> bool {
  if (declaresStaticDataMemberInClass(variable))
    return variable->isInline() || variable->isConstexpr();
  if (redeclaresConstexprStaticDataMember(variable)) return false;
  if (variable->isExtern()) return variable->initializer() != nullptr;
  return true;
}

auto Binder::isInitializedInClass(VariableSymbol* variable) -> bool {
  if (variable->initializer()) return true;
  auto field = definedStaticDataMember(variable);
  return field && field->hasInitializer();
}

auto Binder::declaresVariableDefinition(FieldSymbol* field) -> bool {
  return field->isInline() || field->isConstexpr();
}

auto Binder::redefinesVariable(VariableSymbol* previous,
                               VariableSymbol* variable) const -> bool {
  if (isCxx()) return true;
  return previous->initializer() && variable->initializer();
}

void Binder::recordVariableDefinition(VariableSymbol* variable) {
  if (!declaresVariableDefinition(variable)) return;

  auto canonical = variable->canonical();
  auto previous = definingDeclaration(canonical);

  if (previous && previous != variable &&
      redefinesVariable(previous, variable)) {
    error(variable->location(),
          std::format("redefinition of '{}'", to_string(variable->name())));
    note(previous->location(), "previous definition is here");
    return;
  }

  if (previous) return;
  setDefinition(canonical, variable);
}

namespace {
[[nodiscard]] auto staticDataMemberDefinition(FieldSymbol* field) -> Symbol* {
  if (auto definition = field->definition()) return definition;
  if (field->isConstexpr()) return nullptr;
  if (field->isInline()) return field;
  return nullptr;
}
}  // namespace

void Binder::recordStaticDataMemberDefinition(FieldSymbol* field,
                                              VariableSymbol* definition) {
  if (inExplicitInstantiation() && !inExplicitInstantiationDefinition()) return;

  auto previous = staticDataMemberDefinition(field);
  if (previous && inExplicitInstantiationDefinition()) return;

  if (previous && previous != definition) {
    error(definition->location(),
          std::format("redefinition of '{}'", to_string(field->name())));
    note(previous->location(), "previous definition is here");
  }

  setSpeculativeValue(
      field->definition(), definition,
      [field](VariableSymbol* value) { field->setDefinition(value); });
}

namespace {
[[nodiscard]] auto definingDeclaration(FunctionSymbol* canonical)
    -> FunctionSymbol* {
  if (auto definition = canonical->definition()) return definition;
  if (canonical->isDefined()) return canonical;
  return nullptr;
}

[[nodiscard]] auto isGnuInlineDefinition(FunctionSymbol* function) -> bool {
  return function->isInline() &&
         findAttribute(function->attributes(), "gnu_inline");
}
}  // namespace

void Binder::recordFunctionDefinition(FunctionSymbol* function) {
  auto canonical = function->canonical();

  if (auto previous = definingDeclaration(canonical);
      previous && previous != function && !isGnuInlineDefinition(previous)) {
    error(function->location(),
          std::format("redefinition of '{}'", to_string(function->name())));
    note(previous->location(), "previous definition is here");
  }

  function->setDefined(true);
  if (canonical != function) setDefinition(canonical, function);
}

auto Binder::staticDataMemberOf(ClassSymbol* classSymbol, const Name* name)
    -> FieldSymbol* {
  if (!classSymbol) return nullptr;
  for (auto candidate : classSymbol->find(name)) {
    auto field = symbol_cast<FieldSymbol>(candidate);
    if (field && field->isStatic()) return field;
  }
  return nullptr;
}

auto Binder::variableMemberOf(ClassSymbol* classSymbol, const Name* name)
    -> VariableSymbol* {
  if (!classSymbol) return nullptr;
  auto variables = classSymbol->find(name) | views::variables;
  if (variables.begin() == variables.end()) return nullptr;
  return *variables.begin();
}

namespace {
[[nodiscard]] auto redeclarationCandidates(ScopeSymbol* scope, const Name* name)
    -> SymbolChainView {
  if (name_cast<TemplateId>(name)) return SymbolChainView{};
  return scope->find(name);
}
}  // namespace

auto Binder::declareVariable(DeclaratorAST* declarator, const Decl& decl,
                             bool addSymbolToParentScope,
                             const Type* declaratorType) -> VariableSymbol* {
  auto name = decl.getName();
  auto currentScope = declaringScope();
  auto qualifiedScope = decl.getScope();
  auto qualifiedClass = symbol_cast<ClassSymbol>(qualifiedScope);
  auto qualifiedNamespace = symbol_cast<NamespaceSymbol>(qualifiedScope);

  auto outOfClassMemberField = staticDataMemberOf(qualifiedClass, name);
  auto outOfClassMemberTemplate = variableMemberOf(qualifiedClass, name);

  const bool isOutOfClassStaticMemberDef = outOfClassMemberField != nullptr;
  const bool isOutOfClassMemberTemplateDef =
      outOfClassMemberTemplate != nullptr;
  const bool isOutOfNamespaceMemberDef = qualifiedNamespace != nullptr;

  auto targetScope =
      decl.specs.isExtern ? scopeForBlockDecl(currentScope) : currentScope;
  if (isOutOfNamespaceMemberDef) targetScope = qualifiedNamespace;
  if (isOutOfClassStaticMemberDef || isOutOfClassMemberTemplateDef)
    targetScope = qualifiedClass;

  auto symbol = control()->newVariableSymbol(targetScope, decl.location());
  auto type = declaratorType;
  if (!type) type = getDeclaratorType(unit_, declarator, decl.specs.type());
  applySpecifiers(symbol, decl.specs);
  symbol->setName(name);
  symbol->setType(type);

  if (isOutOfClassStaticMemberDef) {
    recordStaticDataMemberDefinition(outOfClassMemberField, symbol);
    symbol->setStatic(true);
    symbol->setInitializer(outOfClassMemberField->initializer());
  }

  if (isOutOfClassMemberTemplateDef) symbol->setStatic(true);

  if (auto classType = unqualified_cast<ClassType>(type)) {
    traits.requireCompleteClass(classType->symbol());
  }

  applyExplicitAlignment(symbol, decl);

  if (!addSymbolToParentScope) return symbol;
  if (isOutOfClassStaticMemberDef) return symbol;

  if (isOutOfClassMemberTemplateDef) {
    addRedeclaration(outOfClassMemberTemplate->canonical(), symbol);
    return symbol;
  }

  if (auto block = symbol_cast<BlockSymbol>(targetScope);
      block && block->isOutermostBlockScope()) {
    if (auto parentScope = block->parent()) {
      for (auto candidate : parentScope->find(name)) {
        if (!symbol_cast<VariableSymbol>(candidate) &&
            !symbol_cast<ParameterSymbol>(candidate))
          continue;
        error(symbol->location(),
              std::format("redefinition of '{}'", to_string(name)));
        note(candidate->location(), "previous definition is here");
        break;
      }
    }
  }

  for (auto candidate : redeclarationCandidates(targetScope, name)) {
    if (auto existing = symbol_cast<VariableSymbol>(candidate)) {
      if (targetScope->isBlock()) {
        error(symbol->location(),
              std::format("redefinition of '{}'", to_string(name)));
        note(existing->location(), "previous definition is here");
        break;
      }

      if (!areRedeclarationTypesCompatible(unit_, existing->type(),
                                           symbol->type())) {
        error(symbol->location(),
              std::format("conflicting declaration of '{}'", to_string(name)));
        continue;
      }

      auto canon = existing->canonical();
      auto mergedType =
          preferredRedeclarationType(unit_, canon->type(), symbol->type());
      setSpeculativeValue(
          canon->type(), mergedType,
          [canon](const Type* value) { canon->setType(value); });
      symbol->setType(mergedType);
      addRedeclaration(canon, symbol);
      break;
    }
  }

  targetScope->addSymbol(symbol);

  if (targetScope != currentScope && !isOutOfNamespaceMemberDef) {
    if (symbol->canonical() == symbol) symbol->setHidden(true);
    injectUsing(currentScope, name, symbol->canonical(), decl.location());
  }
  return symbol;
}

void Binder::declareVariableTemplate(VariableSymbol* symbol,
                                     IdDeclaratorAST* declaratorId,
                                     TemplateDeclarationAST* templateHead) {
  setTemplateHead(symbol, templateHead);
  if (!templateHead) return;

  checkTemplateParameterDefaultOrder(symbol->canonical()->templateParameters());

  if (!declaratorId) return;

  auto templateId = ast_cast<SimpleTemplateIdAST>(declaratorId->unqualifiedId);
  if (!templateId) return;

  for (auto candidate :
       declaringScope()->find(templateId->identifier) | views::variables) {
    if (candidate == symbol) continue;
    if (!candidate->templateDeclaration()) continue;

    auto templateArguments =
        Substitution(unit_, candidate->templateDeclaration(),
                     templateId->templateArgumentList)
            .templateArguments();

    candidate->addSpecialization(unit_, std::move(templateArguments), symbol);
    break;
  }
}

auto Binder::declareMemberSymbol(DeclaratorAST* declarator, const Decl& decl,
                                 bool addSymbolToParentScope) -> Symbol* {
  if (decl.specs.isTypedef) return declareTypedef(declarator, decl);

  if (getFunctionPrototype(declarator))
    return declareFunction(declarator, decl, addSymbolToParentScope);

  return declareField(declarator, decl);
}

void Binder::applySpecifiers(FunctionSymbol* symbol, const DeclSpecs& specs) {
  applyAccessSpecifier(symbol);
  symbol->setStatic(specs.isStatic);
  symbol->setExtern(specs.isExtern);
  symbol->setFriend(specs.isFriend);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConsteval(specs.isConsteval);
  symbol->setVirtual(specs.isVirtual);
  symbol->setExplicit(specs.isExplicit);
}

void Binder::applySpecifiers(VariableSymbol* symbol, const DeclSpecs& specs) {
  applyAccessSpecifier(symbol);
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setExtern(specs.isExtern);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

void Binder::applySpecifiers(FieldSymbol* symbol, const DeclSpecs& specs) {
  applyAccessSpecifier(symbol);
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

auto Binder::reportUnresolvedNestedNameSpecifier(NestedNameSpecifierAST* ast)
    -> bool {
  if (inTemplate() || isDependentTypeParameterSymbol(ast->symbol)) return false;

  error(ast->firstSourceLocation(),
        "nested name specifier must be a class or namespace");

  return true;
}

struct Binder::NestedNameSpecifierScope {
  Binder& binder;

  [[nodiscard]] auto operator()(ClassSymbol* symbol) const -> ScopeSymbol* {
    binder.traits.requireCompleteClass(symbol);
    return symbol;
  }

  [[nodiscard]] auto operator()(InjectedClassNameSymbol* symbol) const
      -> ScopeSymbol* {
    return (*this)(symbol->classSymbol());
  }

  [[nodiscard]] auto operator()(NamespaceSymbol* symbol) const -> ScopeSymbol* {
    return symbol;
  }

  [[nodiscard]] auto operator()(NamespaceAliasSymbol* symbol) const
      -> ScopeSymbol* {
    return resolve_namespace_alias(symbol);
  }

  [[nodiscard]] auto operator()(EnumSymbol* symbol) const -> ScopeSymbol* {
    return symbol;
  }

  [[nodiscard]] auto operator()(ScopedEnumSymbol* symbol) const
      -> ScopeSymbol* {
    return symbol;
  }

  [[nodiscard]] auto operator()(TypeAliasSymbol* symbol) const -> ScopeSymbol* {
    return binder.scopeOfType(symbol->type());
  }

  [[nodiscard]] auto operator()(Symbol*) const -> ScopeSymbol* {
    return nullptr;
  }
};

auto Binder::resolveNestedNameSpecifier(Symbol* symbol) -> ScopeSymbol* {
  if (!symbol) return nullptr;
  return visit(NestedNameSpecifierScope{*this}, symbol);
}

auto Binder::scopeOfType(const Type* type) -> ScopeSymbol* {
  auto scope = declaredScopeOfType(unqualified_type(type));
  if (auto classSymbol = symbol_cast<ClassSymbol>(scope))
    traits.requireCompleteClass(classSymbol);
  return scope;
}

namespace {
auto isTemplateArgumentKindMatch(
    TemplateDeclarationAST* templateDecl,
    List<TemplateArgumentAST*>* templateArgumentList) -> bool {
  if (!templateDecl) return true;

  std::vector<TemplateParameterAST*> parameters;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    parameters.push_back(parameter);
  }

  std::vector<TemplateArgumentAST*> arguments;
  for (auto argument : ListView{templateArgumentList}) {
    arguments.push_back(argument);
  }

  int argumentIndex = 0;
  for (int parameterIndex = 0;
       parameterIndex < static_cast<int>(parameters.size()); ++parameterIndex) {
    if (argumentIndex >= static_cast<int>(arguments.size())) break;

    auto parameter = parameters[parameterIndex];

    if (isPackParameter(parameter)) {
      while (argumentIndex < static_cast<int>(arguments.size())) {
        if (!matchesTemplateParameterKind(parameter, arguments[argumentIndex]))
          return false;
        ++argumentIndex;
      }
      break;
    }

    if (!matchesTemplateParameterKind(parameter, arguments[argumentIndex]))
      return false;

    ++argumentIndex;
  }

  return argumentIndex == static_cast<int>(arguments.size());
}
}  // namespace

auto Binder::overloadSetFor(ScopeSymbol* scope, const Name* name,
                            SourceLocation location) -> OverloadSetSymbol* {
  for (auto candidate : scope->find(name)) {
    if (auto overloadSet = symbol_cast<OverloadSetSymbol>(candidate))
      return overloadSet;

    if (!joinsFunctionOverloadSet(candidate)) continue;

    auto function = symbol_cast<FunctionSymbol>(candidate);
    auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(candidate);

    auto overloadSet = control()->newOverloadSetSymbol(scope, location);
    overloadSet->setName(name);
    if (function) overloadSet->addFunction(function);
    if (usingDeclaration) overloadSet->addUsingDeclaration(usingDeclaration);
    if (speculationDepth_) {
      recordSpeculativeMutation([scope, candidate, overloadSet] {
        scope->replaceSymbol(overloadSet, candidate);
      });
    }
    scope->replaceSymbol(candidate, overloadSet);
    return overloadSet;
  }

  auto overloadSet = control()->newOverloadSetSymbol(scope, location);
  overloadSet->setName(name);
  scope->addSymbol(overloadSet);
  return overloadSet;
}

void Binder::declareArgumentDependentCallee(IdExpressionAST* ast) {
  auto name = get_lookup_name(control(), ast->unqualifiedId);
  if (!name_cast<Identifier>(name) && !name_cast<OperatorId>(name)) return;

  auto callee = control()->newOverloadSetSymbol(declaringScope(),
                                                ast->firstSourceLocation());
  callee->setName(name);
  ast->symbol = callee;
}

void Binder::declareBuiltinFunctionCallee(IdExpressionAST* ast) {
  if (ast->nestedNameSpecifier &&
      ast->nestedNameSpecifier->symbol != unit_->globalScope())
    return;

  auto name = get_name(control(), ast->unqualifiedId);
  auto id = name_cast<Identifier>(name);
  if (!id) return;

  const auto kind = id->builtinFunction();
  if (kind == BuiltinFunctionKind::T_NONE) return;
  if (ast->symbol && !namesBuiltinFunction(ast->symbol, kind)) return;

  ast->symbol = resolveBuiltinFunctionSymbol(unit_, id, kind);
}

void Binder::bind(OperatorFunctionIdAST* ast) {
  declareImplicitAllocationFunctions(unit_, ast->op);
}

void Binder::bind(IdExpressionAST* ast, bool mayUseArgumentDependentLookup) {
  if (!ast->unqualifiedId) {
    error(ast->firstSourceLocation(),
          "expected an unqualified identifier in id expression");
    return;
  }

  if (ast->nestedNameSpecifier) {
    if (!ast->nestedNameSpecifier->symbol) {
      (void)reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier);
      return;
    }

    if (!lookupQualifiedIdExpression(ast)) return;
  }

  if (mayUseArgumentDependentLookup) declareBuiltinFunctionCallee(ast);

  if (!ast->symbol && !ast->nestedNameSpecifier &&
      mayUseArgumentDependentLookup) {
    declareArgumentDependentCallee(ast);
  }

  resolveIdExpression(ast, mayUseArgumentDependentLookup);
}

auto Binder::lookupQualifiedIdExpression(IdExpressionAST* ast) -> bool {
  auto qualifier = ast->nestedNameSpecifier->symbol;
  if (isDeferredDependentLookupContext(unit_, qualifier, scope())) return false;

  if (auto classSymbol = symbol_cast<ClassSymbol>(qualifier))
    traits.requireCompleteClass(classSymbol);

  auto componentName = get_lookup_name(control(), ast->unqualifiedId);

  bool ambiguous = false;
  ast->symbol = qualifiedLookupIncludingInlineNamespaces(
      control(), qualifier, componentName, &ambiguous);
  if (!ambiguous) return true;

  error(
      ast->unqualifiedId->firstSourceLocation(),
      std::format("reference to '{}' is ambiguous", to_string(componentName)));
  return false;
}

void Binder::qualifiedLookupIdExpression(IdExpressionAST* ast, bool isCallee) {
  if (!ast->unqualifiedId) return;
  if (!ast->nestedNameSpecifier || !ast->nestedNameSpecifier->symbol) return;
  if (!lookupQualifiedIdExpression(ast)) return;

  resolveIdExpression(ast, isCallee);

  if (auto function = designatedFunction(ast->symbol)) {
    ast->symbol = function;
    ast->type = function->type();
  }
}

void Binder::resolveIdExpression(IdExpressionAST* ast, bool isCallee) {
  if (isArgumentDependentCallee(ast->symbol)) return;

  if (unit_->config().checkTypes) {
    if (auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId)) {
      auto templateIdName = get_name(control(), templateId);
      Symbol* templateSymbol = nullptr;
      bool instantiated = false;
      bool hasTemplateCandidate = false;
      bool hasDeferredFunctionTemplate = false;

      auto needsCallSiteDeduction =
          [&](TemplateDeclarationAST* templateDecl) -> bool {
        if (!templateDecl) return false;
        auto arity = TemplateArity::of(templateDecl);
        auto argc = TemplateArguments::count(templateId->templateArgumentList);
        if (argc < arity.minArgs) return true;
        if (arity.packCount > 0 && (isCallee || argc <= arity.minArgs))
          return true;
        return arity.packCount > 1;
      };

      auto hasDependentArguments = [&]() -> bool {
        return hasDependentTemplateArguments(unit_, templateId);
      };

      if (symbol_cast<ConceptSymbol>(ast->symbol)) return;

      if (auto var = symbol_cast<VariableSymbol>(ast->symbol)) {
        if (var->isSpecialization()) return;
        if (var->templateDeclaration() &&
            (!inTemplate() || !hasDependentArguments())) {
          templateSymbol = var;
        }
      } else if (auto func = symbol_cast<FunctionSymbol>(ast->symbol)) {
        if (func->isSpecialization() && !func->templateDeclaration()) return;
        if (func->templateDeclaration()) {
          hasTemplateCandidate = true;
          if (!inTemplate() &&
              TemplateArity::matches(func->templateDeclaration(),
                                     templateId->templateArgumentList,
                                     /*isFunctionTemplate=*/true) &&
              isTemplateArgumentKindMatch(func->templateDeclaration(),
                                          templateId->templateArgumentList)) {
            if (needsCallSiteDeduction(func->templateDeclaration())) {
              hasDeferredFunctionTemplate = true;
            } else {
              templateSymbol = func;
            }
          }
        }
      } else if (auto ovl = symbol_cast<OverloadSetSymbol>(ast->symbol)) {
        const auto ovlFunctions = ovl->functions();

        int matchingTemplateCount = 0;
        for (auto func : ovlFunctions) {
          if (!func->templateDeclaration()) continue;
          if (TemplateArity::matches(func->templateDeclaration(),
                                     templateId->templateArgumentList,
                                     /*isFunctionTemplate=*/true) &&
              isTemplateArgumentKindMatch(func->templateDeclaration(),
                                          templateId->templateArgumentList)) {
            ++matchingTemplateCount;
          }
        }
        if (matchingTemplateCount > 1) return;

        for (auto func : ovlFunctions) {
          if (!func->templateDeclaration()) continue;
          hasTemplateCandidate = true;
          if (!TemplateArity::matches(func->templateDeclaration(),
                                      templateId->templateArgumentList,
                                      /*isFunctionTemplate=*/true) ||
              !isTemplateArgumentKindMatch(func->templateDeclaration(),
                                           templateId->templateArgumentList)) {
            continue;
          }
          if (needsCallSiteDeduction(func->templateDeclaration())) {
            hasDeferredFunctionTemplate = true;
            continue;
          }
          if (!templateSymbol) templateSymbol = func;
          if (inTemplate()) continue;
          auto instance = ASTRewriter::instantiate(
              unit_, templateId->templateArgumentList, func, {},
              /*sfinaeContext=*/true, /*argsComplete=*/false,
              /*declarationOnly=*/true);
          if (instance) {
            ast->symbol = instance;
            templateSymbol = func;
            instantiated = true;
            break;
          }
        }
        if (instantiated) return;

        if (hasDeferredFunctionTemplate) return;

        if (templateSymbol && !inTemplate()) {
          if (isCallee) return;
          if (reportErrors_) {
            error(templateId->firstSourceLocation(),
                  std::format("invalid template-id '{}'",
                              to_string(templateIdName)));
          }
          return;
        }
      }

      if (hasDeferredFunctionTemplate) return;

      if (!templateSymbol) {
        if (!inTemplate()) {
          if (hasTemplateCandidate) {
            if (isCallee) return;
            error(templateId->firstSourceLocation(),
                  std::format("invalid template-id '{}'",
                              to_string(templateIdName)));
          } else {
            error(templateId->firstSourceLocation(),
                  std::format("not a template"));
          }
        }
      } else {
        if (inTemplate() && hasDependentArguments()) return;

        const bool isFuncTemplate =
            symbol_cast<FunctionSymbol>(templateSymbol) != nullptr;
        auto instance = ASTRewriter::instantiate(
            unit_, templateId->templateArgumentList, templateSymbol, {},
            /*sfinaeContext=*/isFuncTemplate, /*argsComplete=*/false,
            /*declarationOnly=*/isFuncTemplate);
        if (!instance) {
          if (!inTemplate()) {
            error(templateId->firstSourceLocation(),
                  std::format("invalid template-id '{}'",
                              to_string(templateIdName)));
          }
          return;
        }

        ast->symbol = instance;
      }
    }
  }
}

auto Binder::denotesCurrentInstantiation(NestedNameSpecifierAST* nns,
                                         ClassSymbol* currentInstantiation)
    -> bool {
  if (!nns || !currentInstantiation) return false;
  auto qualifier = symbol_cast<ClassSymbol>(nns->symbol);
  if (!qualifier) return false;

  auto enclosesCurrentInstantiation = false;
  for (auto cls = currentInstantiation; cls;
       cls = symbol_cast<ClassSymbol>(cls->parent())) {
    if (cls == qualifier) {
      enclosesCurrentInstantiation = true;
      break;
    }
  }
  if (!enclosesCurrentInstantiation) return false;

  auto templateNns = ast_cast<TemplateNestedNameSpecifierAST>(nns);
  if (!templateNns) return true;

  return names_template_head_parameters(templateNns->templateId, qualifier);
}

auto Binder::currentInstantiationOf(ScopeSymbol* scope) -> ClassSymbol* {
  if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) return classSymbol;
  return scope->enclosingClass();
}

auto Binder::resolveMemberOfCurrentInstantiation(
    NestedNameSpecifierAST* nestedNameSpecifier,
    UnqualifiedIdAST* unqualifiedId, ClassSymbol* currentInstantiation)
    -> Symbol* {
  if (!denotesCurrentInstantiation(nestedNameSpecifier, currentInstantiation))
    return nullptr;
  auto nameId = ast_cast<NameIdAST>(unqualifiedId);
  if (!nameId) return nullptr;
  auto qualifier = symbol_cast<ClassSymbol>(nestedNameSpecifier->symbol);
  return qualifiedLookup(qualifier, nameId->identifier,
                         [](Symbol* s) { return is_type(s); });
}

void Binder::bind(TypenameSpecifierAST* ast) {
  auto nestedNameSpecifier = ast->nestedNameSpecifier;
  if (!nestedNameSpecifier || !nestedNameSpecifier->symbol) return;

  if (isDependent(unit_, nestedNameSpecifier)) {
    ast->symbol = resolveMemberOfCurrentInstantiation(
        nestedNameSpecifier, ast->unqualifiedId,
        currentInstantiationOf(scope()));
    return;
  }

  auto qualifier = nestedNameSpecifier->symbol->asScopeSymbol();
  if (!qualifier) return;

  if (auto classSymbol = symbol_cast<ClassSymbol>(qualifier)) {
    unit_->typeTraits().requireCompleteClass(classSymbol);
    if (auto definition = classSymbol->definition()) qualifier = definition;
  }

  Symbol* symbol = nullptr;
  const Identifier* identifier = nullptr;
  if (auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId)) {
    identifier = nameId->identifier;
    symbol = qualifiedLookup(qualifier, identifier,
                             [](Symbol* s) { return is_type(s); });
  } else if (auto templateId =
                 ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId)) {
    if (hasDependentTemplateArguments(unit_, templateId)) return;
    identifier = templateId->identifier;
    templateId->symbol = qualifiedLookup(qualifier, identifier,
                                         [](Symbol* s) { return is_type(s); });
    if (templateId->symbol)
      symbol =
          resolve(nestedNameSpecifier, templateId, /*checkTemplates=*/true);
  }

  if (!symbol) {
    error(ast->typenameLoc,
          std::format("no type named '{}' in '{}'", to_string(identifier),
                      to_string(qualifier->type())));
  }

  ast->symbol = symbol;
}

struct Binder::ResolveCurrentInstantiationMembers {
  Binder& binder;
  ClassSymbol* currentInstantiation = nullptr;

  [[nodiscard]] auto specifierList(List<SpecifierAST*>* list) -> bool {
    auto changed = false;
    for (auto specifier : ListView{list}) {
      changed |= visit(*this, specifier);
    }
    return changed;
  }

  [[nodiscard]] auto typeId(TypeIdAST* ast) -> bool {
    if (!ast) return false;
    if (!specifierList(ast->typeSpecifierList)) return false;
    DeclSpecs specs{binder.unit_};
    for (auto specifier : ListView{ast->typeSpecifierList}) {
      specs.accept(specifier);
    }
    specs.finish();
    ast->type = getDeclaratorType(binder.unit_, ast->declarator, specs.type());
    return true;
  }

  [[nodiscard]] auto expression(ExpressionAST* ast) -> bool {
    auto recheck = [&](TypeIdAST* operand) {
      if (!typeId(operand)) return false;
      auto typeChecker = TypeChecker{binder.unit_};
      typeChecker.setScope(binder.scope());
      typeChecker.setReportErrors(false);
      typeChecker.check(&ast);
      return true;
    };

    if (auto sizeofType = ast_cast<SizeofTypeExpressionAST>(ast))
      return recheck(sizeofType->typeId);

    if (auto alignofType = ast_cast<AlignofTypeExpressionAST>(ast))
      return recheck(alignofType->typeId);

    return false;
  }

  [[nodiscard]] auto templateId(SimpleTemplateIdAST* ast) -> bool {
    if (!ast) return false;
    auto changed = false;
    for (auto argument : ListView{ast->templateArgumentList}) {
      changed |= visit(*this, argument);
    }
    return changed;
  }

  auto operator()(TypeTemplateArgumentAST* ast) -> bool {
    return typeId(ast->typeId);
  }

  auto operator()(ExpressionTemplateArgumentAST* ast) -> bool {
    return expression(ast->expression);
  }

  auto operator()(TypenameSpecifierAST* ast) -> bool {
    if (ast->symbol) return false;
    if (templateId(ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId))) {
      return true;
    }
    auto member = binder.resolveMemberOfCurrentInstantiation(
        ast->nestedNameSpecifier, ast->unqualifiedId, currentInstantiation);
    if (!member) return false;
    ast->symbol = member;
    return true;
  }

  auto operator()(NamedTypeSpecifierAST* ast) -> bool {
    if (!templateId(ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId)))
      return false;
    auto symbol = binder.resolve(ast->nestedNameSpecifier, ast->unqualifiedId,
                                 /*checkTemplates=*/true);
    if (!symbol) return false;
    ast->symbol = symbol;
    return true;
  }

  auto operator()(SpecifierAST*) -> bool { return false; }
};

auto Binder::resolveMembersOfCurrentInstantiation(
    List<SpecifierAST*>* specifierList, ClassSymbol* currentInstantiation)
    -> bool {
  if (!currentInstantiation) return false;
  return ResolveCurrentInstantiationMembers{*this, currentInstantiation}
      .specifierList(specifierList);
}

struct Binder::MemberOfCurrentInstantiationType {
  Binder& binder;
  ClassSymbol* currentInstantiation;

  [[nodiscard]] auto control() const -> Control* { return binder.control(); }

  [[nodiscard]] auto resolve(const Type* type) const -> const Type* {
    return binder.resolveMemberOfCurrentInstantiation(type,
                                                      currentInstantiation);
  }

  [[nodiscard]] auto operator()(const UnresolvedNameType* type) const
      -> const Type* {
    auto member = binder.resolveMemberOfCurrentInstantiation(
        type->nestedNameSpecifier(), type->unqualifiedId(),
        currentInstantiation);
    if (!member || !member->type()) return type;
    return member->type();
  }

  [[nodiscard]] auto operator()(const QualType* type) const -> const Type* {
    auto elementType = resolve(type->elementType());
    if (elementType == type->elementType()) return type;
    return control()->getQualType(elementType, type->cvQualifiers());
  }

  [[nodiscard]] auto operator()(const PointerType* type) const -> const Type* {
    auto elementType = resolve(type->elementType());
    if (elementType == type->elementType()) return type;
    return control()->getPointerType(elementType);
  }

  [[nodiscard]] auto operator()(const LvalueReferenceType* type) const
      -> const Type* {
    auto elementType = resolve(type->elementType());
    if (elementType == type->elementType()) return type;
    return control()->getLvalueReferenceType(elementType);
  }

  [[nodiscard]] auto operator()(const RvalueReferenceType* type) const
      -> const Type* {
    auto elementType = resolve(type->elementType());
    if (elementType == type->elementType()) return type;
    return control()->getRvalueReferenceType(elementType);
  }

  [[nodiscard]] auto operator()(const BoundedArrayType* type) const
      -> const Type* {
    auto elementType = resolve(type->elementType());
    if (elementType == type->elementType()) return type;
    return control()->getBoundedArrayType(elementType, type->size());
  }

  [[nodiscard]] auto operator()(const UnboundedArrayType* type) const
      -> const Type* {
    auto elementType = resolve(type->elementType());
    if (elementType == type->elementType()) return type;
    return control()->getUnboundedArrayType(elementType);
  }

  [[nodiscard]] auto operator()(const FunctionType* type) const -> const Type* {
    auto returnType = resolve(type->returnType());
    auto changed = returnType != type->returnType();
    std::vector<const Type*> parameterTypes;
    parameterTypes.reserve(type->parameterTypes().size());
    for (auto parameterType : type->parameterTypes()) {
      auto resolved = resolve(parameterType);
      changed = changed || resolved != parameterType;
      parameterTypes.push_back(resolved);
    }
    if (!changed) return type;
    return control()->getFunctionType(returnType, std::move(parameterTypes),
                                      type->isVariadic(), type->cvQualifiers(),
                                      type->refQualifier(),
                                      type->exceptionSpecification());
  }

  [[nodiscard]] auto operator()(const Type* type) const -> const Type* {
    return type;
  }
};

auto Binder::resolveMemberOfCurrentInstantiation(
    const Type* type, ClassSymbol* currentInstantiation) -> const Type* {
  if (!type || !currentInstantiation) return type;
  return visit(MemberOfCurrentInstantiationType{*this, currentInstantiation},
               type);
}

auto isExplicitSpecializationHead(TemplateDeclarationAST* templateHead)
    -> bool {
  return templateHead && !templateHead->templateParameterList;
}

auto Binder::getSpecializedFunctionTemplate(
    ScopeSymbol* scope, const Name* name, TemplateDeclarationAST* templateHead)
    -> FunctionSymbol* {
  if (!scope) return nullptr;
  if (!isExplicitSpecializationHead(templateHead) && !inExplicitInstantiation())
    return nullptr;

  auto templateName = name;
  if (auto templateId = name_cast<TemplateId>(name))
    templateName = templateId->name();

  for (auto candidate : scope->find(templateName)) {
    for (auto function : views::each_function(candidate)) {
      auto templateDeclaration = function->templateDeclaration();
      if (!templateDeclaration || !templateDeclaration->templateParameterList)
        continue;
      return function;
    }
  }

  return nullptr;
}

auto Binder::getFunction(ScopeSymbol* scope, const Name* name, const Type* type,
                         TemplateDeclarationAST* templateHead,
                         RequiresClauseAST* trailingRequiresClause)
    -> FunctionSymbol* {
  auto parentScope = scope;

  while (parentScope && parentScope->isTransparent()) {
    parentScope = parentScope->parent();
  }

  auto matches = [&](FunctionSymbol* function) {
    if (!areFunctionSignaturesEquivalentForRedeclaration(
            unit_, function->type(), type, function->templateDeclaration(),
            templateHead,
            /*isOutOfLineDeclaration=*/true)) {
      return false;
    }
    if (!TemplateEquivalence{unit_}.same(function->trailingRequiresClause(),
                                         trailingRequiresClause))
      return false;
    return areFunctionTemplateHeadsEquivalentForRedeclaration(
        unit_, symbol_cast<ClassSymbol>(parentScope),
        function->templateDeclaration(), templateHead);
  };

  if (auto parentClass = symbol_cast<ClassSymbol>(parentScope);
      parentClass && parentClass->name() == name) {
    for (auto ctor : parentClass->constructors()) {
      if (matches(ctor)) return ctor;
    }
  }

  if (auto namespaceSymbol = symbol_cast<NamespaceSymbol>(parentScope)) {
    for (auto candidateScope : inlineNamespaceSet(namespaceSymbol)) {
      if (auto function =
              views::find_function(candidateScope->find(name), matches))
        return function;
    }
    return nullptr;
  }

  return views::find_function(scope->find(name), matches);
}

auto areFunctionTemplateHeadsEquivalentForRedeclaration(
    TranslationUnit* unit, ClassSymbol* enclosingClass,
    TemplateDeclarationAST* existingHead, TemplateDeclarationAST* newHead)
    -> bool {
  existingHead = TemplateEquivalence{unit}.ownFunctionTemplateHead(
      enclosingClass, existingHead);
  newHead = TemplateEquivalence{unit}.ownFunctionTemplateHead(enclosingClass,
                                                              newHead);
  return TemplateEquivalence{unit}.same(existingHead, newHead);
}
}  // namespace cxx
