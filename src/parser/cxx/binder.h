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
#include <cxx/attributes.h>
#include <cxx/const_value.h>
#include <cxx/initialization.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token_fwd.h>
#include <cxx/type_traits.h>
#include <cxx/types_fwd.h>

#include <expected>
#include <functional>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cxx {
class ClassLayout;
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

[[nodiscard]] auto areFunctionSignaturesEquivalentForRedeclaration(
    TranslationUnit* unit, const Type* lhs, const Type* rhs,
    TemplateDeclarationAST* lhsHead, TemplateDeclarationAST* rhsHead,
    bool isOutOfLineDeclaration) -> bool;

[[nodiscard]] auto isExplicitSpecializationHead(
    TemplateDeclarationAST* templateHead) -> bool;

class Binder {
 public:
  [[nodiscard]] auto overloadSetFor(ScopeSymbol* scope, const Name* name,
                                    SourceLocation location)
      -> OverloadSetSymbol*;

  explicit Binder(TranslationUnit* unit);

  [[nodiscard]] auto translationUnit() const -> TranslationUnit*;
  [[nodiscard]] auto control() const -> Control*;

  [[nodiscard]] auto reportErrors() const -> bool;
  void setReportErrors(bool reportErrors);

  struct ClosureNumberingKey {
    FunctionSymbol* context = nullptr;
    std::string signature;

    [[nodiscard]] auto operator==(const ClosureNumberingKey&) const
        -> bool = default;
  };

  struct ClosureNumberingKeyHash {
    [[nodiscard]] auto operator()(const ClosureNumberingKey& key) const
        -> std::size_t;
  };

  [[nodiscard]] auto closureNumberingMark() const -> std::size_t;

  void rewindClosureNumbering(std::size_t mark);

  [[nodiscard]] auto nextClosureNumber(ClassSymbol* closure) -> int;

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

  [[nodiscard]] auto elaboratedTypeSpecifierTargetScope(
      bool isFriend, bool isDeclaration) const -> ScopeSymbol*;

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

  void recordStandardLibraryType(Symbol* symbol);

  [[nodiscard]] auto currentTemplateParameters() const
      -> TemplateParametersSymbol*;

  [[nodiscard]] auto inTemplate() const -> bool;

  void enterExplicitTemplateHead();
  void leaveExplicitTemplateHead();

  void enterExplicitInstantiation(bool isDefinition);
  void leaveExplicitInstantiation();

  [[nodiscard]] auto inExplicitInstantiation() const -> bool;
  [[nodiscard]] auto inExplicitInstantiationDefinition() const -> bool;

  [[nodiscard]] auto inDiscardedStatement() const -> bool;

  class DiscardedStatementGuard {
   public:
    DiscardedStatementGuard(Binder* binder, bool discarded)
        : binder_(discarded ? binder : nullptr) {
      if (binder_) ++binder_->discardedStatementDepth_;
    }

    ~DiscardedStatementGuard() {
      if (binder_) --binder_->discardedStatementDepth_;
    }

    DiscardedStatementGuard(const DiscardedStatementGuard&) = delete;
    auto operator=(const DiscardedStatementGuard&)
        -> DiscardedStatementGuard& = delete;

   private:
    Binder* binder_;
  };

  void setRetainsEnclosingTemplateLevels(bool value);

  void finishAutoReturnType(FunctionSymbol* functionSymbol);

  [[nodiscard]] static auto returnsAValue(AST* declaration) -> bool;

  [[nodiscard]] auto enterBlock(SourceLocation loc) -> BlockSymbol*;

  [[nodiscard]] static auto functionOfBody(ScopeSymbol* scope)
      -> FunctionSymbol*;

  [[nodiscard]] auto functionLocalPredefinedVariable(ScopeSymbol* scope,
                                                     const Name* name)
      -> VariableSymbol*;

  void addTypeAliasToScope(TypeAliasSymbol* symbol);

  void giveTypedefNameForLinkage(TypeAliasSymbol* alias,
                                 SpecifierAST* specifier);

  [[nodiscard]] auto declareTypeAlias(
      SourceLocation identifierLoc, const Identifier* identifier,
      TypeIdAST* typeId, bool addSymbolToParentScope = true,
      TemplateDeclarationAST* templateHead = nullptr) -> TypeAliasSymbol*;

  [[nodiscard]] auto declareTypedef(DeclaratorAST* declarator, const Decl& decl)
      -> TypeAliasSymbol*;

  [[nodiscard]] auto declareFunction(DeclaratorAST* declarator,
                                     const Decl& decl,
                                     bool addSymbolToParentScope = true)
      -> FunctionSymbol*;

  void checkRedeclaredAlignment(ClassSymbol* classSymbol, int requested,
                                SourceLocation loc);

  [[nodiscard]] auto hasDependentAlignment(
      List<AttributeSpecifierAST*>* attributeList) const -> bool;

  [[nodiscard]] auto explicitAlignment(
      List<AttributeSpecifierAST*>* attributeList, SourceLocation loc)
      -> std::optional<int>;

  [[nodiscard]] auto checkExplicitAlignment(int requested, const Type* type,
                                            SourceLocation loc) -> bool;

  [[nodiscard]] auto validatedAlignment(std::optional<std::intmax_t> value,
                                        SourceLocation loc)
      -> std::optional<int>;

  [[nodiscard]] auto alignedAttribute(
      List<AttributeSpecifierAST*>* attributeList) -> std::optional<int>;

  void applyAlignedAttribute(Symbol* symbol,
                             List<AttributeSpecifierAST*>* attributeList);

  void applyExplicitAlignment(FieldSymbol* field, const Decl& decl);
  void applyExplicitAlignment(VariableSymbol* variable, const Decl& decl);

  [[nodiscard]] auto declareField(DeclaratorAST* declarator, const Decl& decl)
      -> FieldSymbol*;

  void declareAnonymousField(ClassSpecifierAST* classSpecifier);

  [[nodiscard]] static auto declaresDefaultTemplateArgument(Symbol* parameter)
      -> bool;

  [[nodiscard]] static auto declaresVariableDefinition(VariableSymbol* variable)
      -> bool;

  [[nodiscard]] static auto declaresVariableDefinition(FieldSymbol* field)
      -> bool;

  [[nodiscard]] static auto definedStaticDataMember(VariableSymbol* variable)
      -> FieldSymbol*;

  [[nodiscard]] static auto isInitializedInClass(VariableSymbol* variable)
      -> bool;

  void recordVariableDefinition(VariableSymbol* variable);

  void recordStaticDataMemberDefinition(FieldSymbol* field,
                                        VariableSymbol* definition);

  void recordFunctionDefinition(FunctionSymbol* function);

  [[nodiscard]] static auto redeclaresOnlyMembersOf(
      UsingDeclarationSymbol* usingDeclaration, ScopeSymbol* scope) -> bool;

  [[nodiscard]] auto staticDataMemberOf(ClassSymbol* classSymbol,
                                        const Name* name) -> FieldSymbol*;

  [[nodiscard]] auto variableMemberOf(ClassSymbol* classSymbol,
                                      const Name* name) -> VariableSymbol*;

  [[nodiscard]] auto declareVariable(DeclaratorAST* declarator,
                                     const Decl& decl,
                                     bool addSymbolToParentScope,
                                     const Type* declaratorType = nullptr)
      -> VariableSymbol*;

  [[nodiscard]] auto declareMemberSymbol(DeclaratorAST* declarator,
                                         const Decl& decl,
                                         bool addSymbolToParentScope = true)
      -> Symbol*;

  void declareVariableTemplate(VariableSymbol* symbol,
                               IdDeclaratorAST* declaratorId,
                               TemplateDeclarationAST* templateHead);

  void bindStructuredBindings(StructuredBindingDeclarationAST* ast,
                              const DeclSpecs& specs);

  void decomposeStructuredBinding(StructuredBindingDeclarationAST* ast,
                                  VariableSymbol* entity);

  [[nodiscard]] auto declareStructuredBindingEntity(
      SourceLocation loc, const Identifier* name, const DeclSpecs& specs,
      TokenKind refOp, ExpressionAST* initializer, bool addSymbolToParentScope,
      const Type* declaredType = nullptr) -> InitDeclaratorAST*;

  [[nodiscard]] auto structuredBindingArrayCopyType(
      StructuredBindingDeclarationAST* ast, const Type* declaredType,
      const Type* initializerType) const -> const Type*;

  [[nodiscard]] auto structuredBindingEntityName() -> const Identifier*;

  [[nodiscard]] auto declareRangeStructuredBindingEntity(
      StructuredBindingDeclarationAST* ast, const DeclSpecs& specs)
      -> VariableSymbol*;

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

  struct DeferredMemberContexts {
    bool exceptionSpecifications;
    bool fieldInitializers;
    bool completedForMemberContexts;
  };

  void complete(ClassSpecifierAST* ast, DeferredMemberContexts deferred = {});
  void completeFieldInitializers(ClassSymbol* classSymbol);
  void refreshImplicitExceptionSpecifications(ClassSymbol* classSymbol);
  void completeDeferredImplicitExceptionSpecification(FunctionSymbol* fn);

  void finalizeExceptionSpecifications(ClassSymbol* classSymbol);

  void synthesizeCompleteObjectCtor(FunctionSymbol* ctor);

  [[nodiscard]] auto inheritedConstructorFor(ClassSymbol* classSymbol,
                                             FunctionSymbol* baseConstructor)
      -> FunctionSymbol*;

  void synthesizeDefaultedMemberBody(FunctionSymbol* fn);

  void bind(DecltypeSpecifierAST* ast);

  void bind(TypenameSpecifierAST* ast);

  void bind(EnumeratorAST* ast, const Type* previousType,
            std::optional<ConstValue> value);

  [[nodiscard]] static auto nextEnumeratorValue(
      TranslationUnit* unit, const Type* underlyingType,
      const std::optional<ConstValue>& previous) -> std::optional<ConstValue>;

  void bind(TypeExceptionDeclarationAST* ast, const Decl& decl);

  void checkExceptionDeclarationType(TypeExceptionDeclarationAST* ast,
                                     const Type* type);

  void checkTrailingRequiresClauseIsTemplated(
      FunctionSymbol* functionSymbol, TemplateDeclarationAST* templateHead);

  [[nodiscard]] auto parameterObjectType(ParameterDeclarationAST* ast,
                                         const Type* specifiersType)
      -> const Type*;

  void rebindParameterType(ParameterDeclarationAST* ast,
                           ParameterSymbol* symbol);

  void bind(ParameterDeclarationAST* ast, const Decl& decl,
            bool inTemplateParameters);

  void bind(UsingDeclaratorAST* ast, Symbol* target);
  void checkUsingDeclaratorAccess(UsingDeclaratorAST* ast,
                                  UsingDeclarationSymbol* symbol);

  [[nodiscard]] static auto usingDeclaratorNamesConstructor(
      UsingDeclaratorAST* ast) -> bool;

  [[nodiscard]] auto bindInheritedConstructors(UsingDeclaratorAST* ast) -> bool;

  void bind(BaseSpecifierAST* ast, Symbol* resolvedType = nullptr);

  void checkBaseClass(BaseSpecifierAST* ast, ClassSymbol* baseClass);

  void declareBaseClass(BaseSpecifierAST* ast, Symbol* symbol);

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
    const Type* declaredType = nullptr;
    ExpressionAST* initializer = nullptr;
    SourceLocation location;
    bool isPack = false;
  };

  void initializeCapturedField(FieldSymbol* field, ScopeSymbol* scope,
                               ExpressionAST*& initializer,
                               InitializationKind kind);

  [[nodiscard]] auto initCapture(LambdaCaptureAST* captureNode)
      -> std::optional<InitCapture>;

  [[nodiscard]] auto deducedInitCaptureType(const InitCapture& capture)
      -> const Type*;

  [[nodiscard]] static auto declaredInitCapture(ScopeSymbol* lambdaScope,
                                                const Identifier* name)
      -> VariableSymbol*;

  void declareInitCapturesInLambdaScope(LambdaExpressionAST* ast);

  [[nodiscard]] auto closureCallOperatorDefinition(LambdaExpressionAST* ast,
                                                   FunctionSymbol* operatorFunc,
                                                   CompoundStatementAST* body)
      -> FunctionDefinitionAST*;
  void completeLambdaBody(LambdaExpressionAST* ast);
  [[nodiscard]] auto declareClosureMemberFunction(ClassSymbol* classSymbol,
                                                  const Name* name,
                                                  const Type* type,
                                                  SourceLocation loc)
      -> FunctionSymbol*;
  void declareSynthesizedParameters(FunctionSymbol* function,
                                    const FunctionType* functionType,
                                    SourceLocation loc);
  [[nodiscard]] auto declareClosureInvoker(ClassSymbol* classSymbol,
                                           FunctionSymbol* operatorFunc,
                                           const FunctionType* operatorType,
                                           SourceLocation loc)
      -> FunctionSymbol*;
  [[nodiscard]] auto materializeClosureFunctionPointerConversion(
      ClassSymbol* closureClass, const FunctionType* targetFunctionType)
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

  void bind(NamespaceAliasDefinitionAST* ast,
            NamespaceSymbol* resolvedNamespace = nullptr);

  void bind(UsingEnumDeclarationAST* ast);

  void bind(TypeIdAST* ast, const Decl& decl);

  void bind(IdExpressionAST* ast, bool mayUseArgumentDependentLookup);

  void bind(OperatorFunctionIdAST* ast);

  void resolveIdExpression(IdExpressionAST* ast, bool isCallee);

  void qualifiedLookupIdExpression(IdExpressionAST* ast, bool isCallee = false);
  [[nodiscard]] auto lookupQualifiedIdExpression(IdExpressionAST* ast) -> bool;

  [[nodiscard]] auto resolve(NestedNameSpecifierAST* nestedNameSpecifier,
                             UnqualifiedIdAST* unqualifiedId,
                             bool checkTemplates,
                             Symbol* resolvedType = nullptr) -> Symbol*;

  [[nodiscard]] auto resolveNestedNameSpecifier(Symbol* symbol) -> ScopeSymbol*;

  [[nodiscard]] auto scopeOfType(const Type* type) -> ScopeSymbol*;

  [[nodiscard]] auto reportUnresolvedNestedNameSpecifier(
      NestedNameSpecifierAST* ast) -> bool;

  [[nodiscard]] auto lookupFriendClass(const Identifier* name,
                                       bool namesTemplate) -> ClassSymbol*;

  [[nodiscard]] auto adoptFriendDeclaredClass(ScopeSymbol* targetScope,
                                              const Identifier* name)
      -> ClassSymbol*;

  void disableAccessControlForUnsupportedFriend(
      NestedNameSpecifierAST* nestedNameSpecifier,
      ClassSymbol* befriendingClass);

  [[nodiscard]] auto getSpecializedFunctionTemplate(
      ScopeSymbol* scope, const Name* name,
      TemplateDeclarationAST* templateHead) -> FunctionSymbol*;

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

  class FunctionBodyGuard {
   public:
    Binder* p = nullptr;
    ScopeSymbol* savedScope = nullptr;

    FunctionBodyGuard(const FunctionBodyGuard&) = delete;
    auto operator=(const FunctionBodyGuard&) -> FunctionBodyGuard& = delete;

    FunctionBodyGuard(Binder* p, ScopeSymbol* parameterScope)
        : p(p), savedScope(p->functionBodyScope_) {
      p->functionBodyScope_ = parameterScope;
    }

    ~FunctionBodyGuard() { p->functionBodyScope_ = savedScope; }
  };

  [[nodiscard]] auto isC() const -> bool;
  [[nodiscard]] auto isCxx() const -> bool;

  void computeClassFlags(ClassSymbol* classSymbol);

  void completeForMemberContexts(ClassSpecifierAST* ast);

  void buildVTableLayout(ClassSymbol* classSymbol);

  [[nodiscard]] auto buildRecordLayout(ClassSymbol* classSymbol)
      -> std::expected<bool, std::string>;

  void mergeTemplateParameterDefaults(TemplateParametersSymbol* accumulated,
                                      TemplateParametersSymbol* incoming);

  void copyDefaultArguments(FunctionParametersSymbol* from,
                            FunctionParametersSymbol* to);

  void checkTemplateParameterDefaultOrder(TemplateParametersSymbol* parameters);

  void checkDefaultTemplateArgumentOnPack(TemplateParametersSymbol* parameters);

  void rejectDefaultTemplateArguments(TemplateDeclarationAST* templateHead,
                                      std::string message);

  void setTemplateHead(FunctionSymbol* symbol,
                       TemplateDeclarationAST* templateHead);

  void setTemplateHead(VariableSymbol* symbol,
                       TemplateDeclarationAST* templateHead);

  void setTemplateHead(TypeAliasSymbol* symbol,
                       TemplateDeclarationAST* templateHead);

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

  void applyFunctionDefinitionKind(FunctionSymbol* functionSymbol,
                                   FunctionBodyAST* functionBody);

  void applyDeclarationAttributes(Symbol* symbol,
                                  List<AttributeSpecifierAST*>* attributes,
                                  DeclaratorAST* declarator = nullptr);

  void applyDeclarationAttributes(SimpleDeclarationAST* ast);

  void inheritDeclarationAttributes(Symbol* symbol, Symbol* pattern);

  void applyAttributeMap(Symbol* symbol, AttributeMap collected);

  void applyWasmFunctionAttributes(FunctionSymbol* function,
                                   const AttributeMap* attributes);

  [[nodiscard]] auto usesImplicitThis(StatementAST* stmt) -> bool;

  struct EntityCapture {
    FieldSymbol* field = nullptr;
    ExpressionAST* initializer = nullptr;
  };

  [[nodiscard]] auto declareCaptureField(ClassSymbol* closure,
                                         const Identifier* name,
                                         const Type* type, SourceLocation loc)
      -> FieldSymbol*;

  [[nodiscard]] auto lookupCapturedEntity(ScopeSymbol* scope,
                                          const Identifier* identifier,
                                          SourceLocation loc) -> Symbol*;

  [[nodiscard]] auto captureEntity(ClassSymbol* closure, Symbol* entity,
                                   const Identifier* name, bool byReference,
                                   ScopeSymbol* scope, SourceLocation loc)
      -> EntityCapture;

  [[nodiscard]] auto captureThis(ClassSymbol* closure, const Type* thisType,
                                 ScopeSymbol* scope, SourceLocation loc)
      -> EntityCapture;

  [[nodiscard]] auto implicitEntityCapture(const Identifier* identifier,
                                           const EntityCapture& capture,
                                           bool byReference, SourceLocation loc)
      -> LambdaCaptureAST*;

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

  void enterSpeculativeDeclarations();
  void leaveSpeculativeDeclarations();

  [[nodiscard]] auto speculativeMutationCount() const -> std::size_t {
    return speculativeMutations_.size();
  }

  void undoSpeculativeMutations(std::size_t count);

  template <typename S>
  void addRedeclaration(S* canonical, S* redeclaration) {
    if (speculationDepth_) {
      const auto count = canonical->redeclarations().size();
      recordSpeculativeMutation(
          [canonical, count] { canonical->truncateRedeclarations(count); });
    }
    canonical->addRedeclaration(redeclaration);
  }

  template <typename S>
  void setDefinition(S* canonical, S* definition) {
    if (speculationDepth_) {
      auto previous = canonical->definition();
      recordSpeculativeMutation(
          [canonical, previous] { canonical->setDefinition(previous); });
    }
    canonical->setDefinition(definition);
  }

  template <typename T, typename Setter>
  void setSpeculativeValue(T previous, T value, Setter setter) {
    if (speculationDepth_) {
      recordSpeculativeMutation(
          [previous, setter]() mutable { setter(previous); });
    }
    setter(value);
  }

 private:
  [[nodiscard]] auto enumeratorType(EnumeratorAST* ast,
                                    const Type* previousType) const
      -> const Type*;
  void inheritDefaultArgument(ParameterSymbol* target, ParameterSymbol* source);

  [[nodiscard]] static auto functionBodyBlock(ScopeSymbol* scope)
      -> BlockSymbol*;

  [[nodiscard]] auto functionParameterScope(ScopeSymbol* scope) const
      -> ScopeSymbol*;

  [[nodiscard]] auto declarePredefinedVariable(ScopeSymbol* owner,
                                               const Identifier* name,
                                               std::string_view value)
      -> VariableSymbol*;

  struct ClassSubobjectExtent {
    std::uint64_t stride = 0;
    std::uint64_t count = 1;
  };

  struct ClassSubobject {
    ClassSymbol* symbol = nullptr;
    std::uint64_t offset = 0;
    std::vector<ClassSubobjectExtent> extents;
  };

  using ClassSubobjectList = std::vector<ClassSubobject>;

  struct CachedClassSubobjects {
    const ClassLayout* layout = nullptr;
    ClassSubobjectList subobjects;
  };

  [[nodiscard]] auto emptyClassSubobjects(ClassSymbol* classSymbol)
      -> const ClassSubobjectList&;

  [[nodiscard]] auto fieldElementClass(FieldSymbol* field) -> ClassSymbol*;

  [[nodiscard]] auto fieldArrayExtent(FieldSymbol* field)
      -> ClassSubobjectExtent;

  void appendClassSubobjects(ClassSubobjectList& subobjects,
                             ClassSymbol* classSymbol, std::uint64_t offset,
                             ClassSubobjectExtent extent);

  std::unordered_map<ClassSymbol*, CachedClassSubobjects> emptyClassSubobjects_;

  struct BindClass;
  struct BuildRecordLayout;
  struct BuildVTableLayout;
  struct CompleteClass;
  struct DeclareFunction;
  struct ResolveUnqualifiedId;
  struct ResolveCurrentInstantiationMembers;

  [[nodiscard]] auto redefinesVariable(VariableSymbol* previous,
                                       VariableSymbol* variable) const -> bool;

  [[nodiscard]] auto declareEnum(const Name* name, SourceLocation location,
                                 const Type* underlyingType, bool scoped,
                                 bool fixedUnderlyingType, bool isDefinition,
                                 bool isValidDeclaration = true)
      -> ScopeSymbol*;

  struct EnumerationHead {
    const Type* underlyingType = nullptr;
    bool isScoped = false;
    bool hasFixedUnderlyingType = false;
    bool isDefined = false;
  };

  struct EnumerationHeadOf;
  struct ClassRangeRewrite;
  struct DecomposeStructuredBinding;
  struct ImplicitExceptionSpecification;
  struct NestedNameSpecifierScope;
  struct MemberOfCurrentInstantiationType;

  [[nodiscard]] auto newEnumeration(const Name* name, SourceLocation location,
                                    const EnumerationHead& head,
                                    bool addToScope) -> ScopeSymbol*;

  [[nodiscard]] auto checkEnumerationRedeclaration(
      const EnumerationHead& existing, const EnumerationHead& declared,
      const Name* name, SourceLocation location) -> bool;

  void declareArgumentDependentCallee(IdExpressionAST* ast);
  void declareBuiltinFunctionCallee(IdExpressionAST* ast);

  [[nodiscard]] auto findOverriddenFunctions(ClassSymbol* cls,
                                             FunctionSymbol* fn)
      -> std::vector<FunctionSymbol*>;

  void applyImplicitExceptionSpecification(FunctionSymbol* fn);

  void findOverriddenFunctionsImpl(
      ClassSymbol* cls, FunctionSymbol* fn, std::vector<ClassSymbol*>& visited,
      std::vector<FunctionSymbol*>& overriddenFunctions);

 private:
  friend struct ClassBodyGuard;

  TranslationUnit* unit_ = nullptr;
  TypeTraits traits;
  ScopeSymbol* scope_ = nullptr;
  ScopeSymbol* functionBodyScope_ = nullptr;
  std::vector<ClassBodyState> classBodyStack_;
  Symbol* instantiatingSymbol_ = nullptr;
  SourceLocation instantiationLoc_{};
  LanguageKind languageLinkage_ = LanguageKind::kCXX;
  int explicitTemplateHeadDepth_ = 0;
  int explicitInstantiationDepth_ = 0;
  bool explicitInstantiationIsDefinition_ = false;
  int discardedStatementDepth_ = 0;
  bool inTemplate_ = false;
  bool retainsEnclosingTemplateLevels_ = false;
  bool reportErrors_ = true;
  std::unordered_map<ClosureNumberingKey, int, ClosureNumberingKeyHash>
      closureCounts_;
  std::vector<std::pair<const ClosureNumberingKey, int>*> closureNumberingLog_;
  std::vector<std::function<void()>> speculativeMutations_;
  int speculationDepth_ = 0;

  void recordSpeculativeMutation(std::function<void()> undo);
  void recordSpeculativeOverload(OverloadSetSymbol* overloadSet);
};
}  // namespace cxx
