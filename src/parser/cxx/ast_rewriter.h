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
#include <cxx/binder.h>
#include <cxx/diagnostic.h>
#include <cxx/diagnostics_client.h>
#include <cxx/names_fwd.h>
#include <cxx/token_fwd.h>

#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace cxx {
class TranslationUnit;
class Control;
class Arena;
class FieldSymbol;

class [[nodiscard]] ASTRewriter {
  explicit ASTRewriter(TranslationUnit* unit, ScopeSymbol* scope,
                       std::vector<TemplateArgument> templateArguments);

 public:
  ~ASTRewriter();

  static auto paste(TranslationUnit* unit, ScopeSymbol* scope,
                    StatementAST* ast) -> StatementAST*;

  static auto instantiate(TranslationUnit* unit,
                          List<TemplateArgumentAST*>* templateArgumentList,
                          Symbol* symbol, SourceLocation instantiationLoc = {},
                          bool sfinaeContext = false, bool argsComplete = false,
                          bool declarationOnly = false,
                          bool retainEnclosingTemplateLevels = false)
      -> Symbol*;

  static auto instantiateForArgs(
      TranslationUnit* unit, List<TemplateArgumentAST*>* deducedArguments,
      FunctionSymbol* function, SourceLocation instantiationLoc,
      bool argsComplete, bool declarationOnly = false) -> FunctionSymbol*;

  static auto ensureCompleteClass(TranslationUnit* unit,
                                  ClassSymbol* classSymbol) -> bool;

  [[nodiscard]] static auto evaluateConcept(
      TranslationUnit* unit, ConceptSymbol* conceptSymbol,
      List<TemplateArgumentAST*>* templateArgumentList) -> std::optional<bool>;

  static void markExplicitInstantiationDeclared(
      TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
      Symbol* symbol);

  static void reportPendingInstantiationErrors(TranslationUnit* unit,
                                               Symbol* primaryTemplate,
                                               Symbol* instantiated,
                                               SourceLocation instantiationLoc);

  static auto substituteDefaultTypeId(
      TranslationUnit* unit, TypeIdAST* typeId,
      const std::vector<TemplateArgument>& templateArguments, int depth,
      ScopeSymbol* scope) -> TypeIdAST*;

  static auto substituteDefaultExpression(
      TranslationUnit* unit, ExpressionAST* expression,
      const std::vector<TemplateArgument>& templateArguments, int depth,
      ScopeSymbol* scope) -> ExpressionAST*;

  static auto substituteParameterClause(
      TranslationUnit* unit, ParameterDeclarationClauseAST* parameters,
      const std::vector<TemplateArgument>& templateArguments, int depth,
      ScopeSymbol* scope) -> ParameterDeclarationClauseAST*;

  static auto substituteParameterTypes(
      TranslationUnit* unit, ParameterDeclarationClauseAST* parameters,
      const std::vector<TemplateArgument>& templateArguments, int depth,
      ScopeSymbol* scope) -> std::optional<std::vector<const Type*>>;

  static auto findPartialSpecializationPattern(
      TranslationUnit* unit, ClassSymbol* primary,
      List<TemplateArgumentAST*>* templateArgumentList) -> ClassSymbol*;

  auto translationUnit() const -> TranslationUnit* { return unit_; }

  auto templateArguments() const -> const std::vector<TemplateArgument>& {
    return templateArguments_;
  }

  [[nodiscard]] auto templateArgumentAt(int depth, int index) const
      -> const TemplateArgument*;

  [[nodiscard]] auto templateArgumentFor(Symbol* templateParameter) const
      -> const TemplateArgument*;

  [[nodiscard]] auto writtenArgumentForAliasedParameter(
      TypeIdAST* patternTypeId) const -> TypeIdAST*;

  [[nodiscard]] auto writtenTypeArgumentSpecifierFor(
      Symbol* templateParameter) const -> NamedTypeSpecifierAST*;

  [[nodiscard]] auto retainsEnclosingTemplateLevels() const -> bool {
    return retainsEnclosingTemplateLevels_;
  }

  void setRetainsEnclosingTemplateLevels(bool value) {
    retainsEnclosingTemplateLevels_ = value;
    binder_.setRetainsEnclosingTemplateLevels(value);
  }

  [[nodiscard]] auto substitutedSymbol(Symbol* templateParameter) const
      -> Symbol*;

  [[nodiscard]] auto hasUnresolvedParameterPack(AST* ast) const -> bool;

  void inheritEnclosingTemplateArguments(Symbol* symbol);

  auto declaration(DeclarationAST* ast,
                   TemplateDeclarationAST* templateHead = nullptr)
      -> DeclarationAST*;

  auto specifier(SpecifierAST* ast,
                 TemplateDeclarationAST* templateHead = nullptr)
      -> SpecifierAST*;

  auto statement(StatementAST* ast) -> StatementAST*;

  auto completePendingBody(FunctionSymbol* func, bool captureBodyErrors = false)
      -> std::vector<Diagnostic>;

  static void completePendingMemberInstantiations(TranslationUnit* unit);

  static auto completePendingBodyFor(TranslationUnit* unit,
                                     FunctionSymbol* function,
                                     bool captureBodyErrors = false)
      -> std::vector<Diagnostic>;

  static void requireFunctionDefinition(TranslationUnit* unit,
                                        FunctionSymbol* function);

  static void completeDeducedReturnType(TranslationUnit* unit, Symbol* symbol);

  void setInstantiatingFunctionTemplateSpecialization(bool value) {
    instantiatingFunctionTemplateSpecialization_ = value;
  }

  void setDepth(int depth) { depth_ = depth; }
  auto depth() const -> int { return depth_; }

  void instantiateOutOfClassMemberDefinitions(ClassSymbol* pattern);

  void retryPendingMemberTemplateAttachment(FunctionSymbol* member);

  [[nodiscard]] static auto associatedConstraints(TranslationUnit* unit,
                                                  Symbol* symbol)
      -> std::vector<ExpressionAST*>;

  [[nodiscard]] static auto evaluateAssociatedConstraints(TranslationUnit* unit,
                                                          Symbol* symbol)
      -> std::optional<bool>;

  [[nodiscard]] static auto checkAssociatedConstraints(
      TranslationUnit* unit, Symbol* symbol,
      const std::vector<TemplateArgument>& templateArguments, int depth)
      -> bool;

  [[nodiscard]] static auto typeConstraintExpression(
      TranslationUnit* unit, ConstraintTypeParameterSymbol* parameter)
      -> ExpressionAST*;

  [[nodiscard]] static auto isMoreConstrained(TranslationUnit* unit,
                                              Symbol* symbol, Symbol* other)
      -> bool;

 private:
  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);
  void note(SourceLocation loc, std::string message);

  void check(ExpressionAST* ast);

  struct RewritePartialSpecialization;
  struct ConstraintSubsumption;

  [[nodiscard]] static auto substituteTemplateArgumentList(
      TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
      const std::vector<TemplateArgument>& templateArguments, int depth,
      ScopeSymbol* scope) -> List<TemplateArgumentAST*>*;

  struct PartialSpecializationResult {
    Symbol* symbol = nullptr;
    bool resolutionFailed = false;

    [[nodiscard]] auto handled() const -> bool {
      return symbol || resolutionFailed;
    }
  };

  static auto tryPartialSpecialization(
      TranslationUnit* unit, ClassSymbol* classSymbol,
      const std::vector<TemplateArgument>& templateArguments)
      -> PartialSpecializationResult;

  static auto tryPartialSpecialization(
      TranslationUnit* unit, VariableSymbol* variableSymbol,
      const std::vector<TemplateArgument>& templateArguments)
      -> PartialSpecializationResult;

  static auto checkConstraintExpression(
      TranslationUnit* unit, Symbol* symbol, ExpressionAST* constraint,
      const std::vector<TemplateArgument>& templateArguments, int depth)
      -> bool;

  static auto evaluateConstraintExpression(
      TranslationUnit* unit, ScopeSymbol* parentScope,
      ExpressionAST* expression,
      const std::vector<TemplateArgument>& templateArguments, int depth)
      -> std::optional<bool>;

  auto control() const -> Control*;
  auto arena() const -> Arena*;
  auto binder() -> Binder& { return binder_; }

  auto takeBodyErrors() -> std::vector<Diagnostic> {
    return std::move(bodyErrors_);
  }

  auto restrictedToDeclarations() const -> bool;
  void setRestrictedToDeclarations(bool restrictedToDeclarations);

  void setClassInstanceToComplete(ClassSymbol* classSymbol) {
    classInstanceToComplete_ = classSymbol;
  }

  [[nodiscard]] auto substitutionFailed() const -> bool {
    return substitutionFailed_;
  }

  class ImmediateContextGuard {
   public:
    explicit ImmediateContextGuard(ASTRewriter& rewrite);
    ~ImmediateContextGuard();

    ImmediateContextGuard(const ImmediateContextGuard&) = delete;
    auto operator=(const ImmediateContextGuard&)
        -> ImmediateContextGuard& = delete;

   private:
    ASTRewriter& rewrite_;
    SilentDiagnosticsScope silent_;
    bool substitutionFailed_;
  };

  void markSubstitutionFailure() {
    if (!shouldCaptureBodyErrors()) substitutionFailed_ = true;
  }

  auto unit(UnitAST* ast) -> UnitAST*;
  auto expression(ExpressionAST* ast) -> ExpressionAST*;
  auto unevaluatedExpression(ExpressionAST* ast) -> ExpressionAST*;
  void checkUnevaluated(ExpressionAST* ast);

  auto rewriteExpressionList(List<ExpressionAST*>* source)
      -> List<ExpressionAST*>*;
  [[nodiscard]] auto rewriteMemInitializerList(List<MemInitializerAST*>* source)
      -> List<MemInitializerAST*>*;
  auto genericAssociation(GenericAssociationAST* ast) -> GenericAssociationAST*;
  auto designator(DesignatorAST* ast) -> DesignatorAST*;
  auto templateParameter(TemplateParameterAST* ast) -> TemplateParameterAST*;
  auto ptrOperator(PtrOperatorAST* ast) -> PtrOperatorAST*;
  auto coreDeclarator(CoreDeclaratorAST* ast) -> CoreDeclaratorAST*;
  auto declaratorChunk(DeclaratorChunkAST* ast) -> DeclaratorChunkAST*;
  auto unqualifiedId(UnqualifiedIdAST* ast) -> UnqualifiedIdAST*;
  auto nestedNameSpecifier(NestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;
  auto functionBody(FunctionBodyAST* ast) -> FunctionBodyAST*;
  auto lambdaBody(StatementAST* ast) -> CompoundStatementAST*;
  auto templateArgument(TemplateArgumentAST* ast) -> TemplateArgumentAST*;
  auto exceptionSpecifier(ExceptionSpecifierAST* ast) -> ExceptionSpecifierAST*;
  auto requirement(RequirementAST* ast) -> RequirementAST*;
  auto newInitializer(NewInitializerAST* ast) -> NewInitializerAST*;
  auto memInitializer(MemInitializerAST* ast) -> MemInitializerAST*;
  auto lambdaCapture(LambdaCaptureAST* ast) -> LambdaCaptureAST*;
  auto exceptionDeclaration(ExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;
  auto attributeSpecifier(AttributeSpecifierAST* ast) -> AttributeSpecifierAST*;
  auto attributeToken(AttributeTokenAST* ast) -> AttributeTokenAST*;

  auto splicer(SplicerAST* ast) -> SplicerAST*;
  auto globalModuleFragment(GlobalModuleFragmentAST* ast)
      -> GlobalModuleFragmentAST*;
  auto privateModuleFragment(PrivateModuleFragmentAST* ast)
      -> PrivateModuleFragmentAST*;
  auto moduleDeclaration(ModuleDeclarationAST* ast) -> ModuleDeclarationAST*;
  auto moduleName(ModuleNameAST* ast) -> ModuleNameAST*;
  auto moduleQualifier(ModuleQualifierAST* ast) -> ModuleQualifierAST*;
  auto modulePartition(ModulePartitionAST* ast) -> ModulePartitionAST*;
  auto importName(ImportNameAST* ast) -> ImportNameAST*;
  auto initDeclarator(InitDeclaratorAST* ast, const DeclSpecs& declSpecs)
      -> InitDeclaratorAST*;
  auto declarator(DeclaratorAST* ast) -> DeclaratorAST*;
  auto usingDeclarator(UsingDeclaratorAST* ast) -> UsingDeclaratorAST*;
  auto enumerator(EnumeratorAST* ast) -> EnumeratorAST*;
  auto typeId(TypeIdAST* ast) -> TypeIdAST*;
  auto handler(HandlerAST* ast) -> HandlerAST*;
  auto baseSpecifier(BaseSpecifierAST* ast) -> BaseSpecifierAST*;
  auto requiresClause(RequiresClauseAST* ast) -> RequiresClauseAST*;
  auto parameterDeclarationClause(ParameterDeclarationClauseAST* ast)
      -> ParameterDeclarationClauseAST*;
  auto trailingReturnType(TrailingReturnTypeAST* ast) -> TrailingReturnTypeAST*;
  auto lambdaSpecifier(LambdaSpecifierAST* ast) -> LambdaSpecifierAST*;
  auto typeConstraint(TypeConstraintAST* ast) -> TypeConstraintAST*;
  auto attributeArgumentClause(AttributeArgumentClauseAST* ast)
      -> AttributeArgumentClauseAST*;
  auto attribute(AttributeAST* ast) -> AttributeAST*;
  auto attributeUsingPrefix(AttributeUsingPrefixAST* ast)
      -> AttributeUsingPrefixAST*;
  auto newPlacement(NewPlacementAST* ast) -> NewPlacementAST*;
  auto nestedNamespaceSpecifier(NestedNamespaceSpecifierAST* ast)
      -> NestedNamespaceSpecifierAST*;
  auto asmOperand(AsmOperandAST* ast) -> AsmOperandAST*;
  auto asmQualifier(AsmQualifierAST* ast) -> AsmQualifierAST*;
  auto asmClobber(AsmClobberAST* ast) -> AsmClobberAST*;
  auto asmGotoLabel(AsmGotoLabelAST* ast) -> AsmGotoLabelAST*;

 private:
  struct UnitVisitor;
  struct DeclarationVisitor;
  struct StatementVisitor;
  struct ExpressionVisitor;
  struct GenericAssociationVisitor;
  struct DesignatorVisitor;
  struct TemplateParameterVisitor;
  struct SpecifierVisitor;
  struct PtrOperatorVisitor;
  struct CoreDeclaratorVisitor;
  struct DeclaratorChunkVisitor;
  struct UnqualifiedIdVisitor;
  struct NestedNameSpecifierVisitor;
  struct FunctionBodyVisitor;
  struct TemplateArgumentVisitor;
  struct ExceptionSpecifierVisitor;
  struct RequirementVisitor;
  struct NewInitializerVisitor;
  struct MemInitializerVisitor;
  struct LambdaCaptureVisitor;
  struct ExceptionDeclarationVisitor;
  struct AttributeSpecifierVisitor;
  struct AttributeTokenVisitor;

 private:
  auto rewriter() -> ASTRewriter* { return this; }

  auto shouldReportCheckErrors() const -> bool;
  auto shouldCaptureBodyErrors() const -> bool;
  void typeCheckAndCapture(std::function<void()> checkFn);

  void addEnclosingTemplateArguments(int depth,
                                     std::vector<TemplateArgument> arguments);

  [[nodiscard]] auto findReferencedParameterPack(AST* ast) const
      -> ParameterPackSymbol*;

  [[nodiscard]] auto expandedParameterPack(TypeIdAST* typeId) const
      -> ParameterPackSymbol*;

  [[nodiscard]] auto parameterPackAt(int depth, int index, bool isPack) const
      -> ParameterPackSymbol*;

  [[nodiscard]] auto parameterPackFor(Symbol* symbol) const
      -> ParameterPackSymbol*;

  [[nodiscard]] auto functionParameterPackFor(Symbol* symbol) const
      -> ParameterPackSymbol*;

  [[nodiscard]] auto substitutedTemplateParameterClass(Symbol* symbol) const
      -> Symbol*;

  [[nodiscard]] auto packElementCount(ParameterPackSymbol* pack) const -> int;

  [[nodiscard]] auto packElementAt(ParameterPackSymbol* pack) const -> Symbol*;

  template <typename Expand>
  void forEachPackElement(ParameterPackSymbol* pack, Expand expand) {
    const int elementCount = packElementCount(pack);
    std::swap(parameterPack_, pack);
    for (int i = 0; i < elementCount; ++i) expandPackElement(i, expand);
    std::swap(parameterPack_, pack);
  }

  template <typename Expand>
  void forEachPackElementReversed(ParameterPackSymbol* pack, Expand expand) {
    const int elementCount = packElementCount(pack);
    std::swap(parameterPack_, pack);
    for (int i = elementCount - 1; i >= 0; --i) expandPackElement(i, expand);
    std::swap(parameterPack_, pack);
  }

  template <typename Expand>
  void expandPackElement(int i, Expand&& expand) {
    std::optional<int> index{i};
    std::swap(elementIndex_, index);
    expand();
    std::swap(elementIndex_, index);
  }

  friend struct FindReferencedParameterPack;
  friend struct FindUnresolvedParameterPack;

  auto emptyFoldIdentity(TokenKind op) -> ExpressionAST*;

  void addSymbolRemap(Symbol* oldSym, Symbol* newSym);

  void remapStructuredBindingSymbols(StructuredBindingDeclarationAST* from,
                                     StructuredBindingDeclarationAST* to);

  void remapScopeMembers(ScopeSymbol* oldScope, ScopeSymbol* newScope);

  void checkMemInitializers(FunctionSymbol* function,
                            CompoundStatementFunctionBodyAST* body);

  void remapFunctionParameters(FunctionDeclaratorChunkAST* patternPrototype,
                               FunctionDeclaratorChunkAST* instancePrototype,
                               FunctionParametersSymbol* patternParameters,
                               FunctionParametersSymbol* instanceParameters);

  [[nodiscard]] auto remapSymbol(Symbol* sym) const -> Symbol*;

  void pushLambdaCaptureFields(
      std::unordered_map<Symbol*, FieldSymbol*> fields);
  void popLambdaCaptureFields();
  [[nodiscard]] auto lambdaCaptureField(Symbol* sym) const -> FieldSymbol*;

  TranslationUnit* unit_ = nullptr;
  std::vector<TemplateArgument> templateArguments_;
  List<TemplateArgumentAST*>* writtenTemplateArgumentList_ = nullptr;
  std::unordered_map<int, std::vector<TemplateArgument>>
      enclosingTemplateArguments_;
  std::vector<Diagnostic> bodyErrors_;
  bool rewritingFunctionBody_ = false;
  int unevaluatedOperandDepth_ = 0;
  int immediateContextDepth_ = 0;
  ParameterPackSymbol* parameterPack_ = nullptr;
  std::optional<int> elementIndex_;
  Binder binder_;
  std::unordered_map<Symbol*, ParameterPackSymbol*> functionParamPacks_;
  std::unordered_map<Symbol*, Symbol*> symbolRemap_;
  std::vector<std::unordered_map<Symbol*, FieldSymbol*>> lambdaCaptureFields_;
  TemplateDeclarationAST* currentTemplateHead_ = nullptr;
  int depth_ = 0;
  ClassSymbol* classInstanceToComplete_ = nullptr;
  bool restrictedToDeclarations_ = false;
  bool retainsEnclosingTemplateLevels_ = false;
  bool substitutionFailed_ = false;

  bool instantiatingFunctionTemplateSpecialization_ = false;

  int classBodyDepth_ = 0;

  struct PendingFieldInitializer {
    InitDeclaratorAST* pattern = nullptr;
    InitDeclaratorAST* instance = nullptr;
    ScopeSymbol* scope = nullptr;
  };

  std::vector<PendingFieldInitializer> pendingFieldInitializers_;

 public:
  [[nodiscard]] auto pendingFieldInitializerMark() const -> std::size_t {
    return pendingFieldInitializers_.size();
  }

  void completePendingFieldInitializers(std::size_t mark);

 private:
  std::vector<FunctionSymbol*> pendingBodyCompletions_;
  std::vector<ClassSymbol*> pendingOutOfClassMemberDefClasses_;
};
}  // namespace cxx
