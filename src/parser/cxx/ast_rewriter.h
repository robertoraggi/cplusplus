// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <vector>

namespace cxx {

class TranslationUnit;
class Control;
class Arena;

class ASTRewriter {
 public:
  [[nodiscard]] static auto instantiateClassTemplate(
      TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
      ClassSymbol* symbol) -> ClassSymbol*;

  [[nodiscard]] static auto instantiateTypeAliasTemplate(
      TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
      TypeAliasSymbol* typeAliasSymbol) -> TypeAliasSymbol*;

  [[nodiscard]] static auto make_substitution(
      TranslationUnit* unit, TemplateDeclarationAST* templateDecl,
      List<TemplateArgumentAST*>* templateArgumentList)
      -> std::vector<TemplateArgument>;

  explicit ASTRewriter(TranslationUnit* unit, Scope* scope,
                       const std::vector<TemplateArgument>& templateArguments);
  ~ASTRewriter();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto declaration(DeclarationAST* ast) -> DeclarationAST*;

 private:
  [[nodiscard]] auto templateArguments() const
      -> const std::vector<TemplateArgument>& {
    return templateArguments_;
  }

  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);

  void check(ExpressionAST* ast);

  [[nodiscard]] auto binder() -> Binder& { return binder_; }

  [[nodiscard]] auto control() const -> Control*;
  [[nodiscard]] auto arena() const -> Arena*;

  [[nodiscard]] auto restrictedToDeclarations() const -> bool;
  void setRestrictedToDeclarations(bool restrictedToDeclarations);

  // run on the base nodes
  [[nodiscard]] auto unit(UnitAST* ast) -> UnitAST*;
  [[nodiscard]] auto statement(StatementAST* ast) -> StatementAST*;
  [[nodiscard]] auto expression(ExpressionAST* ast) -> ExpressionAST*;
  [[nodiscard]] auto genericAssociation(GenericAssociationAST* ast)
      -> GenericAssociationAST*;
  [[nodiscard]] auto designator(DesignatorAST* ast) -> DesignatorAST*;
  [[nodiscard]] auto templateParameter(TemplateParameterAST* ast)
      -> TemplateParameterAST*;
  [[nodiscard]] auto specifier(SpecifierAST* ast) -> SpecifierAST*;
  [[nodiscard]] auto ptrOperator(PtrOperatorAST* ast) -> PtrOperatorAST*;
  [[nodiscard]] auto coreDeclarator(CoreDeclaratorAST* ast)
      -> CoreDeclaratorAST*;
  [[nodiscard]] auto declaratorChunk(DeclaratorChunkAST* ast)
      -> DeclaratorChunkAST*;
  [[nodiscard]] auto unqualifiedId(UnqualifiedIdAST* ast) -> UnqualifiedIdAST*;
  [[nodiscard]] auto nestedNameSpecifier(NestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;
  [[nodiscard]] auto functionBody(FunctionBodyAST* ast) -> FunctionBodyAST*;
  [[nodiscard]] auto templateArgument(TemplateArgumentAST* ast)
      -> TemplateArgumentAST*;
  [[nodiscard]] auto exceptionSpecifier(ExceptionSpecifierAST* ast)
      -> ExceptionSpecifierAST*;
  [[nodiscard]] auto requirement(RequirementAST* ast) -> RequirementAST*;
  [[nodiscard]] auto newInitializer(NewInitializerAST* ast)
      -> NewInitializerAST*;
  [[nodiscard]] auto memInitializer(MemInitializerAST* ast)
      -> MemInitializerAST*;
  [[nodiscard]] auto lambdaCapture(LambdaCaptureAST* ast) -> LambdaCaptureAST*;
  [[nodiscard]] auto exceptionDeclaration(ExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;
  [[nodiscard]] auto attributeSpecifier(AttributeSpecifierAST* ast)
      -> AttributeSpecifierAST*;
  [[nodiscard]] auto attributeToken(AttributeTokenAST* ast)
      -> AttributeTokenAST*;

  // run on the misc nodes
  [[nodiscard]] auto splicer(SplicerAST* ast) -> SplicerAST*;
  [[nodiscard]] auto globalModuleFragment(GlobalModuleFragmentAST* ast)
      -> GlobalModuleFragmentAST*;
  [[nodiscard]] auto privateModuleFragment(PrivateModuleFragmentAST* ast)
      -> PrivateModuleFragmentAST*;
  [[nodiscard]] auto moduleDeclaration(ModuleDeclarationAST* ast)
      -> ModuleDeclarationAST*;
  [[nodiscard]] auto moduleName(ModuleNameAST* ast) -> ModuleNameAST*;
  [[nodiscard]] auto moduleQualifier(ModuleQualifierAST* ast)
      -> ModuleQualifierAST*;
  [[nodiscard]] auto modulePartition(ModulePartitionAST* ast)
      -> ModulePartitionAST*;
  [[nodiscard]] auto importName(ImportNameAST* ast) -> ImportNameAST*;
  [[nodiscard]] auto initDeclarator(InitDeclaratorAST* ast,
                                    const DeclSpecs& declSpecs)
      -> InitDeclaratorAST*;
  [[nodiscard]] auto declarator(DeclaratorAST* ast) -> DeclaratorAST*;
  [[nodiscard]] auto usingDeclarator(UsingDeclaratorAST* ast)
      -> UsingDeclaratorAST*;
  [[nodiscard]] auto enumerator(EnumeratorAST* ast) -> EnumeratorAST*;
  [[nodiscard]] auto typeId(TypeIdAST* ast) -> TypeIdAST*;
  [[nodiscard]] auto handler(HandlerAST* ast) -> HandlerAST*;
  [[nodiscard]] auto baseSpecifier(BaseSpecifierAST* ast) -> BaseSpecifierAST*;
  [[nodiscard]] auto requiresClause(RequiresClauseAST* ast)
      -> RequiresClauseAST*;
  [[nodiscard]] auto parameterDeclarationClause(
      ParameterDeclarationClauseAST* ast) -> ParameterDeclarationClauseAST*;
  [[nodiscard]] auto trailingReturnType(TrailingReturnTypeAST* ast)
      -> TrailingReturnTypeAST*;
  [[nodiscard]] auto lambdaSpecifier(LambdaSpecifierAST* ast)
      -> LambdaSpecifierAST*;
  [[nodiscard]] auto typeConstraint(TypeConstraintAST* ast)
      -> TypeConstraintAST*;
  [[nodiscard]] auto attributeArgumentClause(AttributeArgumentClauseAST* ast)
      -> AttributeArgumentClauseAST*;
  [[nodiscard]] auto attribute(AttributeAST* ast) -> AttributeAST*;
  [[nodiscard]] auto attributeUsingPrefix(AttributeUsingPrefixAST* ast)
      -> AttributeUsingPrefixAST*;
  [[nodiscard]] auto newPlacement(NewPlacementAST* ast) -> NewPlacementAST*;
  [[nodiscard]] auto nestedNamespaceSpecifier(NestedNamespaceSpecifierAST* ast)
      -> NestedNamespaceSpecifierAST*;
  [[nodiscard]] auto asmOperand(AsmOperandAST* ast) -> AsmOperandAST*;
  [[nodiscard]] auto asmQualifier(AsmQualifierAST* ast) -> AsmQualifierAST*;
  [[nodiscard]] auto asmClobber(AsmClobberAST* ast) -> AsmClobberAST*;
  [[nodiscard]] auto asmGotoLabel(AsmGotoLabelAST* ast) -> AsmGotoLabelAST*;

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
  [[nodiscard]] auto rewriter() -> ASTRewriter* { return this; }

  [[nodiscard]] auto getParameterPack(ExpressionAST* ast)
      -> ParameterPackSymbol*;

  std::vector<TemplateArgument> templateArguments_;
  ParameterPackSymbol* parameterPack_ = nullptr;
  std::optional<int> elementIndex_;
  TranslationUnit* unit_ = nullptr;
  Binder binder_;
  bool restrictedToDeclarations_ = false;
};

}  // namespace cxx
