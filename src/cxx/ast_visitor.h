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

#include <cxx/ast_fwd.h>

namespace cxx {

struct ASTVisitor {
  virtual ~ASTVisitor() = default;

  virtual void visit(ThisExpressionAST*) {}
  virtual void visit(NestedExpressionAST*) {}
  virtual void visit(StringLiteralExpressionAST*) {}
  virtual void visit(UserDefinedStringLiteralExpressionAST*) {}
  virtual void visit(CharLiteralExpressionAST*) {}
  virtual void visit(BoolLiteralExpressionAST*) {}
  virtual void visit(IntLiteralExpressionAST*) {}
  virtual void visit(FloatLiteralExpressionAST*) {}
  virtual void visit(NullptrLiteralExpressionAST*) {}
  virtual void visit(IdExpressionAST*) {}
  virtual void visit(BinaryExpressionAST*) {}
  virtual void visit(AssignmentExpressionAST*) {}

  virtual void visit(AliasDeclarationAST*) {}
  virtual void visit(ArrayDeclaratorAST*) {}
  virtual void visit(AsmDeclarationAST*) {}
  virtual void visit(AtomicTypeSpecifierAST*) {}
  virtual void visit(AttributeDeclarationAST*) {}
  virtual void visit(BreakStatementAST*) {}
  virtual void visit(CaseStatementAST*) {}
  virtual void visit(ClassSpecifierAST*) {}
  virtual void visit(CompoundStatementAST*) {}
  virtual void visit(ConceptDefinitionAST*) {}
  virtual void visit(ContinueStatementAST*) {}
  virtual void visit(CoroutineReturnStatementAST*) {}
  virtual void visit(CvQualifierAST*) {}
  virtual void visit(DeclarationStatementAST*) {}
  virtual void visit(DeclaratorAST*) {}
  virtual void visit(DecltypeNameAST*) {}
  virtual void visit(DecltypeSpecifierAST*) {}
  virtual void visit(DecltypeSpecifierTypeSpecifierAST*) {}
  virtual void visit(DeductionGuideAST*) {}
  virtual void visit(DefaultStatementAST*) {}
  virtual void visit(DestructorNameAST*) {}
  virtual void visit(DoStatementAST*) {}
  virtual void visit(ElaboratedTypeSpecifierAST*) {}
  virtual void visit(EllipsisExceptionDeclarationAST*) {}
  virtual void visit(EmptyDeclarationAST*) {}
  virtual void visit(EnumBaseAST*) {}
  virtual void visit(EnumeratorAST*) {}
  virtual void visit(EnumSpecifierAST*) {}
  virtual void visit(ExplicitInstantiationAST*) {}
  virtual void visit(ExplicitSpecifierAST*) {}
  virtual void visit(ExportDeclarationAST*) {}
  virtual void visit(ExpressionStatementAST*) {}
  virtual void visit(ForRangeDeclarationAST*) {}
  virtual void visit(ForRangeStatementAST*) {}
  virtual void visit(ForStatementAST*) {}
  virtual void visit(FunctionDeclaratorAST*) {}
  virtual void visit(FunctionDefinitionAST*) {}
  virtual void visit(FunctionSpecifierAST*) {}
  virtual void visit(GotoStatementAST*) {}
  virtual void visit(HandlerAST*) {}
  virtual void visit(IdDeclaratorAST*) {}
  virtual void visit(IfStatementAST*) {}
  virtual void visit(LabeledStatementAST*) {}
  virtual void visit(LinkageSpecificationAST*) {}
  virtual void visit(ModuleImportDeclarationAST*) {}
  virtual void visit(ModuleUnitAST*) {}
  virtual void visit(NamedTypeSpecifierAST*) {}
  virtual void visit(NamespaceAliasDefinitionAST*) {}
  virtual void visit(NamespaceDefinitionAST*) {}
  virtual void visit(NestedDeclaratorAST*) {}
  virtual void visit(NestedNameSpecifierAST*) {}
  virtual void visit(OpaqueEnumDeclarationAST*) {}
  virtual void visit(OperatorNameAST*) {}
  virtual void visit(PlaceholderTypeSpecifierAST*) {}
  virtual void visit(PlaceholderTypeSpecifierHelperAST*) {}
  virtual void visit(PointerOperatorAST*) {}
  virtual void visit(PrimitiveTypeSpecifierAST*) {}
  virtual void visit(PtrToMemberOperatorAST*) {}
  virtual void visit(ReferenceOperatorAST*) {}
  virtual void visit(ReturnStatementAST*) {}
  virtual void visit(SimpleDeclarationAST*) {}
  virtual void visit(SimpleNameAST*) {}
  virtual void visit(SimpleTypeSpecifierAST*) {}
  virtual void visit(StaticAssertDeclarationAST*) {}
  virtual void visit(StorageClassSpecifierAST*) {}
  virtual void visit(SwitchStatementAST*) {}
  virtual void visit(TemplateArgumentAST*) {}
  virtual void visit(TemplateDeclarationAST*) {}
  virtual void visit(TemplateNameAST*) {}
  virtual void visit(TranslationUnitAST*) {}
  virtual void visit(TryBlockStatementAST*) {}
  virtual void visit(TypeExceptionDeclarationAST*) {}
  virtual void visit(TypeIdAST*) {}
  virtual void visit(TypenameSpecifierAST*) {}
  virtual void visit(UnderlyingTypeSpecifierAST*) {}
  virtual void visit(UsingDeclarationAST*) {}
  virtual void visit(UsingDeclaratorAST*) {}
  virtual void visit(UsingDirectiveAST*) {}
  virtual void visit(UsingEnumDeclarationAST*) {}
  virtual void visit(WhileStatementAST*) {}
};

}  // namespace cxx