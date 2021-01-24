
// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast_visitor.h>

namespace cxx {

struct RecursiveASTVisitor : ASTVisitor {
  virtual bool preVisit(TypeIdAST*) { return true; }
  virtual void postVisit(TypeIdAST*) {}

  virtual bool preVisit(NestedNameSpecifierAST*) { return true; }
  virtual void postVisit(NestedNameSpecifierAST*) {}

  virtual bool preVisit(UsingDeclaratorAST*) { return true; }
  virtual void postVisit(UsingDeclaratorAST*) {}

  virtual bool preVisit(HandlerAST*) { return true; }
  virtual void postVisit(HandlerAST*) {}

  virtual bool preVisit(TemplateArgumentAST*) { return true; }
  virtual void postVisit(TemplateArgumentAST*) {}

  virtual bool preVisit(EnumBaseAST*) { return true; }
  virtual void postVisit(EnumBaseAST*) {}

  virtual bool preVisit(EnumeratorAST*) { return true; }
  virtual void postVisit(EnumeratorAST*) {}

  virtual bool preVisit(DeclaratorAST*) { return true; }
  virtual void postVisit(DeclaratorAST*) {}

  virtual bool preVisit(EllipsisExceptionDeclarationAST*) { return true; }
  virtual void postVisit(EllipsisExceptionDeclarationAST*) {}

  virtual bool preVisit(TypeExceptionDeclarationAST*) { return true; }
  virtual void postVisit(TypeExceptionDeclarationAST*) {}

  virtual bool preVisit(TranslationUnitAST*) { return true; }
  virtual void postVisit(TranslationUnitAST*) {}

  virtual bool preVisit(ModuleUnitAST*) { return true; }
  virtual void postVisit(ModuleUnitAST*) {}

  virtual bool preVisit(ThisExpressionAST*) { return true; }
  virtual void postVisit(ThisExpressionAST*) {}

  virtual bool preVisit(CharLiteralExpressionAST*) { return true; }
  virtual void postVisit(CharLiteralExpressionAST*) {}

  virtual bool preVisit(BoolLiteralExpressionAST*) { return true; }
  virtual void postVisit(BoolLiteralExpressionAST*) {}

  virtual bool preVisit(IntLiteralExpressionAST*) { return true; }
  virtual void postVisit(IntLiteralExpressionAST*) {}

  virtual bool preVisit(FloatLiteralExpressionAST*) { return true; }
  virtual void postVisit(FloatLiteralExpressionAST*) {}

  virtual bool preVisit(NullptrLiteralExpressionAST*) { return true; }
  virtual void postVisit(NullptrLiteralExpressionAST*) {}

  virtual bool preVisit(StringLiteralExpressionAST*) { return true; }
  virtual void postVisit(StringLiteralExpressionAST*) {}

  virtual bool preVisit(UserDefinedStringLiteralExpressionAST*) { return true; }
  virtual void postVisit(UserDefinedStringLiteralExpressionAST*) {}

  virtual bool preVisit(IdExpressionAST*) { return true; }
  virtual void postVisit(IdExpressionAST*) {}

  virtual bool preVisit(NestedExpressionAST*) { return true; }
  virtual void postVisit(NestedExpressionAST*) {}

  virtual bool preVisit(BinaryExpressionAST*) { return true; }
  virtual void postVisit(BinaryExpressionAST*) {}

  virtual bool preVisit(AssignmentExpressionAST*) { return true; }
  virtual void postVisit(AssignmentExpressionAST*) {}

  virtual bool preVisit(LabeledStatementAST*) { return true; }
  virtual void postVisit(LabeledStatementAST*) {}

  virtual bool preVisit(CaseStatementAST*) { return true; }
  virtual void postVisit(CaseStatementAST*) {}

  virtual bool preVisit(DefaultStatementAST*) { return true; }
  virtual void postVisit(DefaultStatementAST*) {}

  virtual bool preVisit(ExpressionStatementAST*) { return true; }
  virtual void postVisit(ExpressionStatementAST*) {}

  virtual bool preVisit(CompoundStatementAST*) { return true; }
  virtual void postVisit(CompoundStatementAST*) {}

  virtual bool preVisit(IfStatementAST*) { return true; }
  virtual void postVisit(IfStatementAST*) {}

  virtual bool preVisit(SwitchStatementAST*) { return true; }
  virtual void postVisit(SwitchStatementAST*) {}

  virtual bool preVisit(WhileStatementAST*) { return true; }
  virtual void postVisit(WhileStatementAST*) {}

  virtual bool preVisit(DoStatementAST*) { return true; }
  virtual void postVisit(DoStatementAST*) {}

  virtual bool preVisit(ForRangeStatementAST*) { return true; }
  virtual void postVisit(ForRangeStatementAST*) {}

  virtual bool preVisit(ForStatementAST*) { return true; }
  virtual void postVisit(ForStatementAST*) {}

  virtual bool preVisit(BreakStatementAST*) { return true; }
  virtual void postVisit(BreakStatementAST*) {}

  virtual bool preVisit(ContinueStatementAST*) { return true; }
  virtual void postVisit(ContinueStatementAST*) {}

  virtual bool preVisit(ReturnStatementAST*) { return true; }
  virtual void postVisit(ReturnStatementAST*) {}

  virtual bool preVisit(GotoStatementAST*) { return true; }
  virtual void postVisit(GotoStatementAST*) {}

  virtual bool preVisit(CoroutineReturnStatementAST*) { return true; }
  virtual void postVisit(CoroutineReturnStatementAST*) {}

  virtual bool preVisit(DeclarationStatementAST*) { return true; }
  virtual void postVisit(DeclarationStatementAST*) {}

  virtual bool preVisit(TryBlockStatementAST*) { return true; }
  virtual void postVisit(TryBlockStatementAST*) {}

  virtual bool preVisit(FunctionDefinitionAST*) { return true; }
  virtual void postVisit(FunctionDefinitionAST*) {}

  virtual bool preVisit(ConceptDefinitionAST*) { return true; }
  virtual void postVisit(ConceptDefinitionAST*) {}

  virtual bool preVisit(ForRangeDeclarationAST*) { return true; }
  virtual void postVisit(ForRangeDeclarationAST*) {}

  virtual bool preVisit(AliasDeclarationAST*) { return true; }
  virtual void postVisit(AliasDeclarationAST*) {}

  virtual bool preVisit(SimpleDeclarationAST*) { return true; }
  virtual void postVisit(SimpleDeclarationAST*) {}

  virtual bool preVisit(StaticAssertDeclarationAST*) { return true; }
  virtual void postVisit(StaticAssertDeclarationAST*) {}

  virtual bool preVisit(EmptyDeclarationAST*) { return true; }
  virtual void postVisit(EmptyDeclarationAST*) {}

  virtual bool preVisit(AttributeDeclarationAST*) { return true; }
  virtual void postVisit(AttributeDeclarationAST*) {}

  virtual bool preVisit(OpaqueEnumDeclarationAST*) { return true; }
  virtual void postVisit(OpaqueEnumDeclarationAST*) {}

  virtual bool preVisit(UsingEnumDeclarationAST*) { return true; }
  virtual void postVisit(UsingEnumDeclarationAST*) {}

  virtual bool preVisit(NamespaceDefinitionAST*) { return true; }
  virtual void postVisit(NamespaceDefinitionAST*) {}

  virtual bool preVisit(NamespaceAliasDefinitionAST*) { return true; }
  virtual void postVisit(NamespaceAliasDefinitionAST*) {}

  virtual bool preVisit(UsingDirectiveAST*) { return true; }
  virtual void postVisit(UsingDirectiveAST*) {}

  virtual bool preVisit(UsingDeclarationAST*) { return true; }
  virtual void postVisit(UsingDeclarationAST*) {}

  virtual bool preVisit(AsmDeclarationAST*) { return true; }
  virtual void postVisit(AsmDeclarationAST*) {}

  virtual bool preVisit(LinkageSpecificationAST*) { return true; }
  virtual void postVisit(LinkageSpecificationAST*) {}

  virtual bool preVisit(ExportDeclarationAST*) { return true; }
  virtual void postVisit(ExportDeclarationAST*) {}

  virtual bool preVisit(ModuleImportDeclarationAST*) { return true; }
  virtual void postVisit(ModuleImportDeclarationAST*) {}

  virtual bool preVisit(TemplateDeclarationAST*) { return true; }
  virtual void postVisit(TemplateDeclarationAST*) {}

  virtual bool preVisit(DeductionGuideAST*) { return true; }
  virtual void postVisit(DeductionGuideAST*) {}

  virtual bool preVisit(ExplicitInstantiationAST*) { return true; }
  virtual void postVisit(ExplicitInstantiationAST*) {}

  virtual bool preVisit(SimpleNameAST*) { return true; }
  virtual void postVisit(SimpleNameAST*) {}

  virtual bool preVisit(DestructorNameAST*) { return true; }
  virtual void postVisit(DestructorNameAST*) {}

  virtual bool preVisit(DecltypeNameAST*) { return true; }
  virtual void postVisit(DecltypeNameAST*) {}

  virtual bool preVisit(OperatorNameAST*) { return true; }
  virtual void postVisit(OperatorNameAST*) {}

  virtual bool preVisit(TemplateNameAST*) { return true; }
  virtual void postVisit(TemplateNameAST*) {}

  virtual bool preVisit(SimpleSpecifierAST*) { return true; }
  virtual void postVisit(SimpleSpecifierAST*) {}

  virtual bool preVisit(ExplicitSpecifierAST*) { return true; }
  virtual void postVisit(ExplicitSpecifierAST*) {}

  virtual bool preVisit(NamedTypeSpecifierAST*) { return true; }
  virtual void postVisit(NamedTypeSpecifierAST*) {}

  virtual bool preVisit(PlaceholderTypeSpecifierHelperAST*) { return true; }
  virtual void postVisit(PlaceholderTypeSpecifierHelperAST*) {}

  virtual bool preVisit(DecltypeSpecifierTypeSpecifierAST*) { return true; }
  virtual void postVisit(DecltypeSpecifierTypeSpecifierAST*) {}

  virtual bool preVisit(UnderlyingTypeSpecifierAST*) { return true; }
  virtual void postVisit(UnderlyingTypeSpecifierAST*) {}

  virtual bool preVisit(AtomicTypeSpecifierAST*) { return true; }
  virtual void postVisit(AtomicTypeSpecifierAST*) {}

  virtual bool preVisit(ElaboratedTypeSpecifierAST*) { return true; }
  virtual void postVisit(ElaboratedTypeSpecifierAST*) {}

  virtual bool preVisit(DecltypeSpecifierAST*) { return true; }
  virtual void postVisit(DecltypeSpecifierAST*) {}

  virtual bool preVisit(PlaceholderTypeSpecifierAST*) { return true; }
  virtual void postVisit(PlaceholderTypeSpecifierAST*) {}

  virtual bool preVisit(CvQualifierAST*) { return true; }
  virtual void postVisit(CvQualifierAST*) {}

  virtual bool preVisit(EnumSpecifierAST*) { return true; }
  virtual void postVisit(EnumSpecifierAST*) {}

  virtual bool preVisit(ClassSpecifierAST*) { return true; }
  virtual void postVisit(ClassSpecifierAST*) {}

  virtual bool preVisit(TypenameSpecifierAST*) { return true; }
  virtual void postVisit(TypenameSpecifierAST*) {}

  virtual bool preVisit(IdDeclaratorAST*) { return true; }
  virtual void postVisit(IdDeclaratorAST*) {}

  virtual bool preVisit(NestedDeclaratorAST*) { return true; }
  virtual void postVisit(NestedDeclaratorAST*) {}

  virtual bool preVisit(PointerOperatorAST*) { return true; }
  virtual void postVisit(PointerOperatorAST*) {}

  virtual bool preVisit(ReferenceOperatorAST*) { return true; }
  virtual void postVisit(ReferenceOperatorAST*) {}

  virtual bool preVisit(PtrToMemberOperatorAST*) { return true; }
  virtual void postVisit(PtrToMemberOperatorAST*) {}

  virtual bool preVisit(FunctionDeclaratorAST*) { return true; }
  virtual void postVisit(FunctionDeclaratorAST*) {}

  virtual bool preVisit(ArrayDeclaratorAST*) { return true; }
  virtual void postVisit(ArrayDeclaratorAST*) {}

  void visit(TypeIdAST* ast) override;
  void visit(NestedNameSpecifierAST* ast) override;
  void visit(UsingDeclaratorAST* ast) override;
  void visit(HandlerAST* ast) override;
  void visit(TemplateArgumentAST* ast) override;
  void visit(EnumBaseAST* ast) override;
  void visit(EnumeratorAST* ast) override;
  void visit(DeclaratorAST* ast) override;

  void visit(EllipsisExceptionDeclarationAST* ast) override;
  void visit(TypeExceptionDeclarationAST* ast) override;

  void visit(TranslationUnitAST* ast) override;
  void visit(ModuleUnitAST* ast) override;

  void visit(ThisExpressionAST* ast) override;
  void visit(CharLiteralExpressionAST* ast) override;
  void visit(BoolLiteralExpressionAST* ast) override;
  void visit(IntLiteralExpressionAST* ast) override;
  void visit(FloatLiteralExpressionAST* ast) override;
  void visit(NullptrLiteralExpressionAST* ast) override;
  void visit(StringLiteralExpressionAST* ast) override;
  void visit(UserDefinedStringLiteralExpressionAST* ast) override;
  void visit(IdExpressionAST* ast) override;
  void visit(NestedExpressionAST* ast) override;
  void visit(BinaryExpressionAST* ast) override;
  void visit(AssignmentExpressionAST* ast) override;

  void visit(LabeledStatementAST* ast) override;
  void visit(CaseStatementAST* ast) override;
  void visit(DefaultStatementAST* ast) override;
  void visit(ExpressionStatementAST* ast) override;
  void visit(CompoundStatementAST* ast) override;
  void visit(IfStatementAST* ast) override;
  void visit(SwitchStatementAST* ast) override;
  void visit(WhileStatementAST* ast) override;
  void visit(DoStatementAST* ast) override;
  void visit(ForRangeStatementAST* ast) override;
  void visit(ForStatementAST* ast) override;
  void visit(BreakStatementAST* ast) override;
  void visit(ContinueStatementAST* ast) override;
  void visit(ReturnStatementAST* ast) override;
  void visit(GotoStatementAST* ast) override;
  void visit(CoroutineReturnStatementAST* ast) override;
  void visit(DeclarationStatementAST* ast) override;
  void visit(TryBlockStatementAST* ast) override;

  void visit(FunctionDefinitionAST* ast) override;
  void visit(ConceptDefinitionAST* ast) override;
  void visit(ForRangeDeclarationAST* ast) override;
  void visit(AliasDeclarationAST* ast) override;
  void visit(SimpleDeclarationAST* ast) override;
  void visit(StaticAssertDeclarationAST* ast) override;
  void visit(EmptyDeclarationAST* ast) override;
  void visit(AttributeDeclarationAST* ast) override;
  void visit(OpaqueEnumDeclarationAST* ast) override;
  void visit(UsingEnumDeclarationAST* ast) override;
  void visit(NamespaceDefinitionAST* ast) override;
  void visit(NamespaceAliasDefinitionAST* ast) override;
  void visit(UsingDirectiveAST* ast) override;
  void visit(UsingDeclarationAST* ast) override;
  void visit(AsmDeclarationAST* ast) override;
  void visit(LinkageSpecificationAST* ast) override;
  void visit(ExportDeclarationAST* ast) override;
  void visit(ModuleImportDeclarationAST* ast) override;
  void visit(TemplateDeclarationAST* ast) override;
  void visit(DeductionGuideAST* ast) override;
  void visit(ExplicitInstantiationAST* ast) override;

  void visit(SimpleNameAST* ast) override;
  void visit(DestructorNameAST* ast) override;
  void visit(DecltypeNameAST* ast) override;
  void visit(OperatorNameAST* ast) override;
  void visit(TemplateNameAST* ast) override;

  void visit(SimpleSpecifierAST* ast) override;
  void visit(ExplicitSpecifierAST* ast) override;
  void visit(NamedTypeSpecifierAST* ast) override;
  void visit(PlaceholderTypeSpecifierHelperAST* ast) override;
  void visit(DecltypeSpecifierTypeSpecifierAST* ast) override;
  void visit(UnderlyingTypeSpecifierAST* ast) override;
  void visit(AtomicTypeSpecifierAST* ast) override;
  void visit(ElaboratedTypeSpecifierAST* ast) override;
  void visit(DecltypeSpecifierAST* ast) override;
  void visit(PlaceholderTypeSpecifierAST* ast) override;
  void visit(CvQualifierAST* ast) override;
  void visit(EnumSpecifierAST* ast) override;
  void visit(ClassSpecifierAST* ast) override;
  void visit(TypenameSpecifierAST* ast) override;

  void visit(IdDeclaratorAST* ast) override;
  void visit(NestedDeclaratorAST* ast) override;

  void visit(PointerOperatorAST* ast) override;
  void visit(ReferenceOperatorAST* ast) override;
  void visit(PtrToMemberOperatorAST* ast) override;

  void visit(FunctionDeclaratorAST* ast) override;
  void visit(ArrayDeclaratorAST* ast) override;
};

}  // namespace cxx
