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

  // AST
  virtual void visit(TypeIdAST* ast) = 0;
  virtual void visit(NestedNameSpecifierAST* ast) = 0;
  virtual void visit(UsingDeclaratorAST* ast) = 0;
  virtual void visit(HandlerAST* ast) = 0;
  virtual void visit(TemplateArgumentAST* ast) = 0;
  virtual void visit(EnumBaseAST* ast) = 0;
  virtual void visit(EnumeratorAST* ast) = 0;
  virtual void visit(DeclaratorAST* ast) = 0;
  virtual void visit(InitDeclaratorAST* ast) = 0;
  virtual void visit(BaseSpecifierAST* ast) = 0;
  virtual void visit(BaseClauseAST* ast) = 0;
  virtual void visit(NewTypeIdAST* ast) = 0;
  virtual void visit(ParameterDeclarationClauseAST* ast) = 0;
  virtual void visit(ParametersAndQualifiersAST* ast) = 0;
  virtual void visit(LambdaIntroducerAST* ast) = 0;
  virtual void visit(LambdaDeclaratorAST* ast) = 0;
  virtual void visit(TrailingReturnTypeAST* ast) = 0;

  // InitializerAST
  virtual void visit(EqualInitializerAST* ast) = 0;
  virtual void visit(BracedInitListAST* ast) = 0;
  virtual void visit(ParenInitializerAST* ast) = 0;

  // NewInitializerAST
  virtual void visit(NewParenInitializerAST* ast) = 0;
  virtual void visit(NewBracedInitializerAST* ast) = 0;

  // ExceptionDeclarationAST
  virtual void visit(EllipsisExceptionDeclarationAST* ast) = 0;
  virtual void visit(TypeExceptionDeclarationAST* ast) = 0;

  // UnitAST
  virtual void visit(TranslationUnitAST* ast) = 0;
  virtual void visit(ModuleUnitAST* ast) = 0;

  // ExpressionAST
  virtual void visit(ThisExpressionAST* ast) = 0;
  virtual void visit(CharLiteralExpressionAST* ast) = 0;
  virtual void visit(BoolLiteralExpressionAST* ast) = 0;
  virtual void visit(IntLiteralExpressionAST* ast) = 0;
  virtual void visit(FloatLiteralExpressionAST* ast) = 0;
  virtual void visit(NullptrLiteralExpressionAST* ast) = 0;
  virtual void visit(StringLiteralExpressionAST* ast) = 0;
  virtual void visit(UserDefinedStringLiteralExpressionAST* ast) = 0;
  virtual void visit(IdExpressionAST* ast) = 0;
  virtual void visit(NestedExpressionAST* ast) = 0;
  virtual void visit(LambdaExpressionAST* ast) = 0;
  virtual void visit(UnaryExpressionAST* ast) = 0;
  virtual void visit(BinaryExpressionAST* ast) = 0;
  virtual void visit(AssignmentExpressionAST* ast) = 0;
  virtual void visit(CallExpressionAST* ast) = 0;
  virtual void visit(SubscriptExpressionAST* ast) = 0;
  virtual void visit(MemberExpressionAST* ast) = 0;
  virtual void visit(ConditionalExpressionAST* ast) = 0;
  virtual void visit(CppCastExpressionAST* ast) = 0;
  virtual void visit(NewExpressionAST* ast) = 0;
  virtual void visit(DeleteExpressionAST* ast) = 0;

  // StatementAST
  virtual void visit(LabeledStatementAST* ast) = 0;
  virtual void visit(CaseStatementAST* ast) = 0;
  virtual void visit(DefaultStatementAST* ast) = 0;
  virtual void visit(ExpressionStatementAST* ast) = 0;
  virtual void visit(CompoundStatementAST* ast) = 0;
  virtual void visit(IfStatementAST* ast) = 0;
  virtual void visit(SwitchStatementAST* ast) = 0;
  virtual void visit(WhileStatementAST* ast) = 0;
  virtual void visit(DoStatementAST* ast) = 0;
  virtual void visit(ForRangeStatementAST* ast) = 0;
  virtual void visit(ForStatementAST* ast) = 0;
  virtual void visit(BreakStatementAST* ast) = 0;
  virtual void visit(ContinueStatementAST* ast) = 0;
  virtual void visit(ReturnStatementAST* ast) = 0;
  virtual void visit(GotoStatementAST* ast) = 0;
  virtual void visit(CoroutineReturnStatementAST* ast) = 0;
  virtual void visit(DeclarationStatementAST* ast) = 0;
  virtual void visit(TryBlockStatementAST* ast) = 0;

  // DeclarationAST
  virtual void visit(FunctionDefinitionAST* ast) = 0;
  virtual void visit(ConceptDefinitionAST* ast) = 0;
  virtual void visit(ForRangeDeclarationAST* ast) = 0;
  virtual void visit(AliasDeclarationAST* ast) = 0;
  virtual void visit(SimpleDeclarationAST* ast) = 0;
  virtual void visit(StaticAssertDeclarationAST* ast) = 0;
  virtual void visit(EmptyDeclarationAST* ast) = 0;
  virtual void visit(AttributeDeclarationAST* ast) = 0;
  virtual void visit(OpaqueEnumDeclarationAST* ast) = 0;
  virtual void visit(UsingEnumDeclarationAST* ast) = 0;
  virtual void visit(NamespaceDefinitionAST* ast) = 0;
  virtual void visit(NamespaceAliasDefinitionAST* ast) = 0;
  virtual void visit(UsingDirectiveAST* ast) = 0;
  virtual void visit(UsingDeclarationAST* ast) = 0;
  virtual void visit(AsmDeclarationAST* ast) = 0;
  virtual void visit(ExportDeclarationAST* ast) = 0;
  virtual void visit(ModuleImportDeclarationAST* ast) = 0;
  virtual void visit(TemplateDeclarationAST* ast) = 0;
  virtual void visit(DeductionGuideAST* ast) = 0;
  virtual void visit(ExplicitInstantiationAST* ast) = 0;
  virtual void visit(ParameterDeclarationAST* ast) = 0;
  virtual void visit(LinkageSpecificationAST* ast) = 0;

  // NameAST
  virtual void visit(SimpleNameAST* ast) = 0;
  virtual void visit(DestructorNameAST* ast) = 0;
  virtual void visit(DecltypeNameAST* ast) = 0;
  virtual void visit(OperatorNameAST* ast) = 0;
  virtual void visit(TemplateNameAST* ast) = 0;
  virtual void visit(QualifiedNameAST* ast) = 0;

  // SpecifierAST
  virtual void visit(TypedefSpecifierAST* ast) = 0;
  virtual void visit(FriendSpecifierAST* ast) = 0;
  virtual void visit(ConstevalSpecifierAST* ast) = 0;
  virtual void visit(ConstinitSpecifierAST* ast) = 0;
  virtual void visit(ConstexprSpecifierAST* ast) = 0;
  virtual void visit(InlineSpecifierAST* ast) = 0;
  virtual void visit(StaticSpecifierAST* ast) = 0;
  virtual void visit(ExternSpecifierAST* ast) = 0;
  virtual void visit(ThreadLocalSpecifierAST* ast) = 0;
  virtual void visit(ThreadSpecifierAST* ast) = 0;
  virtual void visit(MutableSpecifierAST* ast) = 0;
  virtual void visit(VirtualSpecifierAST* ast) = 0;
  virtual void visit(ExplicitSpecifierAST* ast) = 0;
  virtual void visit(AutoTypeSpecifierAST* ast) = 0;
  virtual void visit(VoidTypeSpecifierAST* ast) = 0;
  virtual void visit(VaListTypeSpecifierAST* ast) = 0;
  virtual void visit(IntegralTypeSpecifierAST* ast) = 0;
  virtual void visit(FloatingPointTypeSpecifierAST* ast) = 0;
  virtual void visit(ComplexTypeSpecifierAST* ast) = 0;
  virtual void visit(NamedTypeSpecifierAST* ast) = 0;
  virtual void visit(AtomicTypeSpecifierAST* ast) = 0;
  virtual void visit(UnderlyingTypeSpecifierAST* ast) = 0;
  virtual void visit(ElaboratedTypeSpecifierAST* ast) = 0;
  virtual void visit(DecltypeAutoSpecifierAST* ast) = 0;
  virtual void visit(DecltypeSpecifierAST* ast) = 0;
  virtual void visit(TypeofSpecifierAST* ast) = 0;
  virtual void visit(PlaceholderTypeSpecifierAST* ast) = 0;
  virtual void visit(ConstQualifierAST* ast) = 0;
  virtual void visit(VolatileQualifierAST* ast) = 0;
  virtual void visit(RestrictQualifierAST* ast) = 0;
  virtual void visit(EnumSpecifierAST* ast) = 0;
  virtual void visit(ClassSpecifierAST* ast) = 0;
  virtual void visit(TypenameSpecifierAST* ast) = 0;

  // CoreDeclaratorAST
  virtual void visit(IdDeclaratorAST* ast) = 0;
  virtual void visit(NestedDeclaratorAST* ast) = 0;

  // PtrOperatorAST
  virtual void visit(PointerOperatorAST* ast) = 0;
  virtual void visit(ReferenceOperatorAST* ast) = 0;
  virtual void visit(PtrToMemberOperatorAST* ast) = 0;

  // DeclaratorModifierAST
  virtual void visit(FunctionDeclaratorAST* ast) = 0;
  virtual void visit(ArrayDeclaratorAST* ast) = 0;
};

}  // namespace cxx
