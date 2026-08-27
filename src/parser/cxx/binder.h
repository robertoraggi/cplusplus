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

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token_fwd.h>
#include <cxx/type_traits.h>
#include <cxx/types_fwd.h>

#include <expected>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cxx {
class DeclSpecs;
class Decl;

class TranslationUnit;

[[nodiscard]] auto areFunctionTemplateHeadsEquivalentForRedeclaration(
    TranslationUnit* unit, ClassSymbol* enclosingClass,
    TemplateDeclarationAST* existingHead, TemplateDeclarationAST* newHead)
    -> bool;

[[nodiscard]] auto areRedeclarationTypesCompatible(TranslationUnit* unit,
                                                   const Type* existingType,
                                                   const Type* incomingType)
    -> bool;

class Binder {
 public:
  struct DefaultArgumentInfo {
    ExpressionAST* expression = nullptr;
    SourceLocation location = {};
  };

  [[nodiscard]] auto overloadSetFor(ScopeSymbol* scope, const Name* name,
                                    SourceLocation location)
      -> OverloadSetSymbol*;

  explicit Binder(TranslationUnit* unit);

  [[nodiscard]] auto translationUnit() const -> TranslationUnit*;
  [[nodiscard]] auto control() const -> Control*;

  [[nodiscard]] auto reportErrors() const -> bool;
  void setReportErrors(bool reportErrors);

  struct ClosureNamingState {
    int lambdaCount = 0;
    std::unordered_map<FunctionSymbol*, int> lambdaDiscriminators;
  };

  [[nodiscard]] auto closureNamingState() const -> ClosureNamingState;

  void setClosureNamingState(ClosureNamingState state);

  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);
  void note(SourceLocation loc, std::string message);

  [[nodiscard]] auto scope() const -> ScopeSymbol*;
  void setScope(ScopeSymbol* scope);

  [[nodiscard]] auto languageLinkage() const -> LanguageKind;
  void setLanguageLinkage(LanguageKind linkage);

  [[nodiscard]] auto changeLanguageLinkage(LanguageKind linkage)
      -> LanguageKind;

  [[nodiscard]] auto isInstantiating() const -> bool;
  [[nodiscard]] auto instantiatingSymbol() const -> Symbol*;
  void setInstantiatingSymbol(Symbol* symbol);

  [[nodiscard]] auto instantiationLoc() const -> SourceLocation;
  void setInstantiationLoc(SourceLocation loc);

  [[nodiscard]] auto declaringScope() const -> ScopeSymbol*;

  struct ClassBodyState {
    ClassSymbol* classSymbol = nullptr;
    AccessSpecifier defaultAccessSpecifier = AccessSpecifier::kPublic;
    AccessSpecifier accessSpecifier = AccessSpecifier::kPublic;
  };

  struct ClassBodyGuard {
    ClassBodyGuard(const ClassBodyGuard&) = delete;
    auto operator=(const ClassBodyGuard&) -> ClassBodyGuard& = delete;

    ClassBodyGuard(Binder* binder, ClassSymbol* classSymbol,
                   AccessSpecifier defaultAccessSpecifier)
        : binder_(binder) {
      binder_->classBodyStack_.push_back(ClassBodyState{
          .classSymbol = classSymbol,
          .defaultAccessSpecifier = defaultAccessSpecifier,
          .accessSpecifier = defaultAccessSpecifier,
      });
    }

    ~ClassBodyGuard() { binder_->classBodyStack_.pop_back(); }

   private:
    Binder* binder_;
  };

  [[nodiscard]] auto classBeingDefined() const -> ClassSymbol*;
  [[nodiscard]] auto currentAccessSpecifier() const -> AccessSpecifier;
  [[nodiscard]] auto defaultAccessSpecifier() const -> AccessSpecifier;
  void setCurrentAccessSpecifier(AccessSpecifier accessSpecifier);

  void applyAccessSpecifier(Symbol* symbol) const;

  [[nodiscard]] auto currentTemplateParameters() const
      -> TemplateParametersSymbol*;

  [[nodiscard]] auto inTemplate() const -> bool;

  void enterExplicitTemplateHead();
  void leaveExplicitTemplateHead();

  void setRetainsEnclosingTemplateLevels(bool value);

  void finishAutoReturnType(FunctionSymbol* functionSymbol);

  [[nodiscard]] static auto returnsAValue(AST* declaration) -> bool;

  [[nodiscard]] auto enterBlock(SourceLocation loc) -> BlockSymbol*;

  [[nodiscard]] auto declareTypeAlias(SourceLocation identifierLoc,
                                      TypeIdAST* typeId,
                                      bool addSymbolToParentScope = true)
      -> TypeAliasSymbol*;

  [[nodiscard]] auto declareTypedef(DeclaratorAST* declarator, const Decl& decl)
      -> TypeAliasSymbol*;

  [[nodiscard]] auto declareFunction(DeclaratorAST* declarator,
                                     const Decl& decl,
                                     bool addSymbolToParentScope = true)
      -> FunctionSymbol*;

  [[nodiscard]] auto declareField(DeclaratorAST* declarator, const Decl& decl)
      -> FieldSymbol*;

  void declareAnonymousField(ClassSpecifierAST* classSpecifier);

  [[nodiscard]] auto declareVariable(DeclaratorAST* declarator,
                                     const Decl& decl,
                                     bool addSymbolToParentScope)
      -> VariableSymbol*;

  [[nodiscard]] auto declareMemberSymbol(DeclaratorAST* declarator,
                                         const Decl& decl) -> Symbol*;

  void bindStructuredBindings(StructuredBindingDeclarationAST* ast,
                              const DeclSpecs& specs);

  void decomposeStructuredBinding(StructuredBindingDeclarationAST* ast,
                                  VariableSymbol* entity);

  [[nodiscard]] auto declareStructuredBindingEntity(
      SourceLocation loc, const Identifier* name, const DeclSpecs& specs,
      TokenKind refOp, ExpressionAST* initializer, bool addSymbolToParentScope)
      -> InitDeclaratorAST*;

  [[nodiscard]] auto structuredBindingEntityName() -> const Identifier*;

  void finishForRangeDeclaration(ForRangeStatementAST* ast,
                                 const DeclSpecs& specs);

  void applySpecifiers(FunctionSymbol* symbol, const DeclSpecs& specs);
  void applySpecifiers(VariableSymbol* symbol, const DeclSpecs& specs);
  void applySpecifiers(FieldSymbol* symbol, const DeclSpecs& specs);

  void bind(EnumSpecifierAST* ast, const DeclSpecs& underlyingTypeSpec);

  void bind(OpaqueEnumDeclarationAST* ast, const DeclSpecs& underlyingTypeSpec);

  void bind(ElaboratedTypeSpecifierAST* ast, DeclSpecs& declSpecs,
            bool isDeclaration, Symbol* unqualifiedCandidate = nullptr);

  void bind(ClassSpecifierAST* ast, DeclSpecs& declSpecs);

  void complete(ClassSpecifierAST* ast,
                bool deferExceptionSpecificationChecks = false);
  void refreshImplicitExceptionSpecifications(ClassSymbol* classSymbol);
  void finalizeExceptionSpecifications(ClassSymbol* classSymbol);

  void synthesizeCompleteObjectCtor(FunctionSymbol* ctor);

  [[nodiscard]] auto inheritedConstructorFor(ClassSymbol* classSymbol,
                                             FunctionSymbol* baseConstructor)
      -> FunctionSymbol*;

  void synthesizeDefaultedMemberBody(FunctionSymbol* fn);

  void bind(DecltypeSpecifierAST* ast);

  void bind(EnumeratorAST* ast, const Type* type,
            std::optional<ConstValue> value);

  [[nodiscard]] static auto nextEnumeratorValue(
      TranslationUnit* unit, const Type* underlyingType,
      const std::optional<ConstValue>& previous) -> std::optional<ConstValue>;

  void bind(ParameterDeclarationAST* ast, const Decl& decl,
            bool inTemplateParameters);

  void bind(UsingDeclaratorAST* ast, Symbol* target);
  void checkUsingDeclaratorAccess(UsingDeclaratorAST* ast,
                                  UsingDeclarationSymbol* symbol);

  [[nodiscard]] static auto usingDeclaratorNamesConstructor(
      UsingDeclaratorAST* ast) -> bool;

  [[nodiscard]] auto bindInheritedConstructors(UsingDeclaratorAST* ast) -> bool;

  void bind(BaseSpecifierAST* ast, Symbol* resolvedType = nullptr);

  void bind(NonTypeTemplateParameterAST* ast, int index, int depth);

  void bind(TypenameTypeParameterAST* ast, int index, int depth);

  void bind(ConstraintTypeParameterAST* ast, int index, int depth);

  void bind(TemplateTypeParameterAST* ast, int index, int depth);

  void bind(ConceptDefinitionAST* ast);

  void bind(DeductionGuideAST* ast, TemplateDeclarationAST* templateHead);

  void bind(LambdaExpressionAST* ast);

  void complete(LambdaExpressionAST* ast);

  struct InitCapture {
    const Identifier* name = nullptr;
    const Type* type = nullptr;
    bool isPack = false;
  };

  [[nodiscard]] auto initCapture(LambdaCaptureAST* captureNode)
      -> std::optional<InitCapture>;

  void declareInitCapturesInLambdaScope(LambdaExpressionAST* ast);

  void completeLambdaBody(LambdaExpressionAST* ast);
  [[nodiscard]] auto declareClosureInvoker(ClassSymbol* classSymbol,
                                           FunctionSymbol* operatorFunc,
                                           const FunctionType* operatorType,
                                           SourceLocation loc)
      -> FunctionSymbol*;
  void declareClosureFunctionPointerConversion(ClassSymbol* classSymbol,
                                               FunctionSymbol* invoker,
                                               const FunctionType* operatorType,
                                               SourceLocation loc);
  void attachSynthesizedBody(FunctionSymbol* function, UnqualifiedIdAST* id,
                             FunctionBodyAST* body);
  void completeClosureType(ClassSymbol* classSymbol);

  void bind(ParameterDeclarationClauseAST* ast);

  void bind(UsingDirectiveAST* ast,
            NamespaceSymbol* resolvedNamespace = nullptr);

  void bind(UsingEnumDeclarationAST* ast);

  void bind(TypeIdAST* ast, const Decl& decl);

  void bind(IdExpressionAST* ast, bool mayUseArgumentDependentLookup);

  void resolveIdExpression(IdExpressionAST* ast, bool isCallee);

  void qualifiedLookupIdExpression(IdExpressionAST* ast);

  [[nodiscard]] auto resolve(NestedNameSpecifierAST* nestedNameSpecifier,
                             UnqualifiedIdAST* unqualifiedId,
                             bool checkTemplates,
                             Symbol* resolvedType = nullptr) -> Symbol*;

  [[nodiscard]] auto resolveNestedNameSpecifier(Symbol* symbol) -> ScopeSymbol*;

  [[nodiscard]] auto reportUnresolvedNestedNameSpecifier(
      NestedNameSpecifierAST* ast) -> bool;

  [[nodiscard]] auto adoptFriendDeclaredClass(ScopeSymbol* targetScope,
                                              const Identifier* name)
      -> ClassSymbol*;

  void disableAccessControlForUnsupportedFriend(
      NestedNameSpecifierAST* nestedNameSpecifier,
      ClassSymbol* befriendingClass);

  [[nodiscard]] auto getFunction(
      ScopeSymbol* scope, const Name* name, const Type* type,
      TemplateDeclarationAST* templateHead = nullptr,
      RequiresClauseAST* trailingRequiresClause = nullptr) -> FunctionSymbol*;

  class ScopeGuard {
   public:
    Binder* p = nullptr;
    ScopeSymbol* savedScope = nullptr;

    ScopeGuard(const ScopeGuard&) = delete;
    auto operator=(const ScopeGuard&) -> ScopeGuard& = delete;

    ScopeGuard() = default;

    explicit ScopeGuard(Binder* p, ScopeSymbol* scope = nullptr)
        : p(p), savedScope(p->scope_) {
      if (scope) p->setScope(scope);
    }

    ~ScopeGuard() { p->setScope(savedScope); }
  };

  [[nodiscard]] auto isC() const -> bool;
  [[nodiscard]] auto isCxx() const -> bool;

  void mergeDefaultArguments(FunctionSymbol* functionSymbol,
                             DeclaratorAST* declarator);

  void computeClassFlags(ClassSymbol* classSymbol);

  [[nodiscard]] auto buildRecordLayout(ClassSymbol* classSymbol)
      -> std::expected<bool, std::string>;

  [[nodiscard]] auto scopeForBlockDecl(ScopeSymbol* scope) const
      -> ScopeSymbol*;

  void injectUsing(ScopeSymbol* scope, const Name* name, Symbol* target,
                   SourceLocation loc);

  [[nodiscard]] auto lookupCaptureName(ScopeSymbol* scope, const Name* name)
      -> Symbol*;

  [[nodiscard]] auto isCapturableLocalEntity(Symbol* symbol) -> bool;

  [[nodiscard]] auto checkCapturedEntity(Symbol* symbol,
                                         const Identifier* identifier,
                                         SourceLocation loc) -> bool;

  [[nodiscard]] auto enclosingThisType(ScopeSymbol* scope) -> const Type*;

  [[nodiscard]] auto abiTags(List<AttributeSpecifierAST*>* attributes)
      -> std::vector<const Identifier*>;

  void applyAbiTags(Symbol* symbol, List<AttributeSpecifierAST*>* attributes);

  void applyAbiTags(SimpleDeclarationAST* ast);

  [[nodiscard]] auto usesImplicitThis(StatementAST* stmt) -> bool;

  [[nodiscard]] auto addImplicitThisCapture(ClassSymbol* classSymbol,
                                            const Type* thisType,
                                            SourceLocation loc)
      -> ThisLambdaCaptureAST*;

  void addImplicitCaptures(LambdaExpressionAST* ast, ClassSymbol* classSymbol);

  [[nodiscard]] auto denotesCurrentInstantiation(
      NestedNameSpecifierAST* nestedNameSpecifier,
      ClassSymbol* currentInstantiation) -> bool;

  [[nodiscard]] auto currentInstantiationOf(ScopeSymbol* scope) -> ClassSymbol*;

  [[nodiscard]] auto resolveMemberOfCurrentInstantiation(
      const Type* type, ClassSymbol* currentInstantiation) -> const Type*;

  [[nodiscard]] auto resolveMemberOfCurrentInstantiation(
      NestedNameSpecifierAST* nestedNameSpecifier,
      UnqualifiedIdAST* unqualifiedId, ClassSymbol* currentInstantiation)
      -> Symbol*;

  [[nodiscard]] auto resolveMembersOfCurrentInstantiation(
      List<SpecifierAST*>* specifierList, ClassSymbol* currentInstantiation)
      -> bool;

 private:
  struct BindClass;
  struct BuildRecordLayout;
  struct CompleteClass;
  struct DeclareFunction;
  struct ResolveUnqualifiedId;
  struct ResolveCurrentInstantiationMembers;

  [[nodiscard]] auto declareEnum(const Name* name, SourceLocation location,
                                 const Type* underlyingType, bool scoped,
                                 bool fixedUnderlyingType, bool isDefinition,
                                 bool isValidDeclaration = true)
      -> ScopeSymbol*;

  void declareArgumentDependentCallee(IdExpressionAST* ast);

  [[nodiscard]] auto findOverriddenFunctions(ClassSymbol* cls,
                                             FunctionSymbol* fn)
      -> std::vector<FunctionSymbol*>;

  void applyImplicitExceptionSpecification(FunctionSymbol* fn);

  void findOverriddenFunctionsImpl(
      ClassSymbol* cls, FunctionSymbol* fn,
      std::unordered_set<ClassSymbol*>& visited,
      std::vector<FunctionSymbol*>& overriddenFunctions);

 private:
  friend struct ClassBodyGuard;

  TranslationUnit* unit_ = nullptr;
  TypeTraits traits;
  ScopeSymbol* scope_ = nullptr;
  std::vector<ClassBodyState> classBodyStack_;
  Symbol* instantiatingSymbol_ = nullptr;
  SourceLocation instantiationLoc_{};
  LanguageKind languageLinkage_ = LanguageKind::kCXX;
  int explicitTemplateHeadDepth_ = 0;
  bool inTemplate_ = false;
  bool retainsEnclosingTemplateLevels_ = false;
  bool reportErrors_ = true;
  std::unordered_map<FunctionSymbol*, int> lambdaDiscriminators_;
  std::unordered_map<FunctionSymbol*, std::vector<DefaultArgumentInfo>>
      defaultArguments_;
};
}  // namespace cxx
