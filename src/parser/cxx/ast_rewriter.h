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
#include <cxx/names_fwd.h>
#include <cxx/token_fwd.h>

#include <unordered_map>
#include <vector>

namespace cxx {

class TranslationUnit;
class Control;
class Arena;

class [[nodiscard]] ASTRewriter {
  explicit ASTRewriter(TranslationUnit* unit, ScopeSymbol* scope,
                       std::vector<TemplateArgument> templateArguments);

 public:
  ~ASTRewriter();

  static auto paste(TranslationUnit* unit, ScopeSymbol* scope,
                    StatementAST* ast) -> StatementAST*;

  static auto instantiate(TranslationUnit* unit,
                          List<TemplateArgumentAST*>* templateArgumentList,
                          Symbol* symbol) -> Symbol*;

  auto translationUnit() const -> TranslationUnit* { return unit_; }

  auto declaration(DeclarationAST* ast,
                   TemplateDeclarationAST* templateHead = nullptr)
      -> DeclarationAST*;

  auto specifier(SpecifierAST* ast,
                 TemplateDeclarationAST* templateHead = nullptr)
      -> SpecifierAST*;

  auto statement(StatementAST* ast) -> StatementAST*;

 private:
  void completePendingBody(FunctionSymbol* func);

  auto templateArguments() const -> const std::vector<TemplateArgument>& {
    return templateArguments_;
  }

  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);

  void check(ExpressionAST* ast);

  static auto tryPartialSpecialization(
      TranslationUnit* unit, ClassSymbol* classSymbol,
      const std::vector<TemplateArgument>& templateArguments) -> Symbol*;

  static auto tryPartialSpecialization(
      TranslationUnit* unit, VariableSymbol* variableSymbol,
      const std::vector<TemplateArgument>& templateArguments) -> Symbol*;

  static auto checkRequiresClause(
      TranslationUnit* unit, Symbol* symbol, RequiresClauseAST* clause,
      const std::vector<TemplateArgument>& templateArguments, int depth)
      -> bool;

  auto control() const -> Control*;
  auto arena() const -> Arena*;
  auto binder() -> Binder& { return binder_; }

  auto restrictedToDeclarations() const -> bool;
  void setRestrictedToDeclarations(bool restrictedToDeclarations);

  // run on the base nodes
  auto unit(UnitAST* ast) -> UnitAST*;
  auto expression(ExpressionAST* ast) -> ExpressionAST*;
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

  // run on the misc nodes
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

  auto getParameterPack(ExpressionAST* ast) -> ParameterPackSymbol*;

  auto getTypeParameterPack(SpecifierAST* ast) -> ParameterPackSymbol*;

  auto emptyFoldIdentity(TokenKind op) -> ExpressionAST*;

  TranslationUnit* unit_ = nullptr;
  std::vector<TemplateArgument> templateArguments_;
  ParameterPackSymbol* parameterPack_ = nullptr;
  std::optional<int> elementIndex_;
  Binder binder_;
  int depth_ = 0;
  bool restrictedToDeclarations_ = false;
  std::unordered_map<Symbol*, ParameterPackSymbol*> functionParamPacks_;
};

}  // namespace cxx
