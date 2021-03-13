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

#include <cxx/arena.h>
#include <cxx/ast.h>
#include <cxx/ast_visitor.h>

namespace cxx {

struct ASTCloner : ASTVisitor {
  virtual AST* clone(AST* ast, Arena* arena);

  void visit(TypeIdAST* ast) override;
  void visit(NestedNameSpecifierAST* ast) override;
  void visit(UsingDeclaratorAST* ast) override;
  void visit(HandlerAST* ast) override;
  void visit(TemplateArgumentAST* ast) override;
  void visit(EnumBaseAST* ast) override;
  void visit(EnumeratorAST* ast) override;
  void visit(DeclaratorAST* ast) override;
  void visit(InitDeclaratorAST* ast) override;
  void visit(BaseSpecifierAST* ast) override;
  void visit(BaseClauseAST* ast) override;
  void visit(NewTypeIdAST* ast) override;
  void visit(ParameterDeclarationClauseAST* ast) override;
  void visit(ParametersAndQualifiersAST* ast) override;
  void visit(LambdaIntroducerAST* ast) override;
  void visit(LambdaDeclaratorAST* ast) override;
  void visit(TrailingReturnTypeAST* ast) override;

  void visit(EqualInitializerAST* ast) override;
  void visit(BracedInitListAST* ast) override;
  void visit(ParenInitializerAST* ast) override;

  void visit(NewParenInitializerAST* ast) override;
  void visit(NewBracedInitializerAST* ast) override;

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
  void visit(LambdaExpressionAST* ast) override;
  void visit(TypeidExpressionAST* ast) override;
  void visit(TypeidOfTypeExpressionAST* ast) override;
  void visit(UnaryExpressionAST* ast) override;
  void visit(BinaryExpressionAST* ast) override;
  void visit(AssignmentExpressionAST* ast) override;
  void visit(BracedTypeConstructionAST* ast) override;
  void visit(TypeConstructionAST* ast) override;
  void visit(CallExpressionAST* ast) override;
  void visit(SubscriptExpressionAST* ast) override;
  void visit(MemberExpressionAST* ast) override;
  void visit(ConditionalExpressionAST* ast) override;
  void visit(CastExpressionAST* ast) override;
  void visit(CppCastExpressionAST* ast) override;
  void visit(NewExpressionAST* ast) override;
  void visit(DeleteExpressionAST* ast) override;
  void visit(ThrowExpressionAST* ast) override;

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
  void visit(ExportDeclarationAST* ast) override;
  void visit(ModuleImportDeclarationAST* ast) override;
  void visit(TemplateDeclarationAST* ast) override;
  void visit(DeductionGuideAST* ast) override;
  void visit(ExplicitInstantiationAST* ast) override;
  void visit(ParameterDeclarationAST* ast) override;
  void visit(LinkageSpecificationAST* ast) override;

  void visit(SimpleNameAST* ast) override;
  void visit(DestructorNameAST* ast) override;
  void visit(DecltypeNameAST* ast) override;
  void visit(OperatorNameAST* ast) override;
  void visit(TemplateNameAST* ast) override;
  void visit(QualifiedNameAST* ast) override;

  void visit(TypedefSpecifierAST* ast) override;
  void visit(FriendSpecifierAST* ast) override;
  void visit(ConstevalSpecifierAST* ast) override;
  void visit(ConstinitSpecifierAST* ast) override;
  void visit(ConstexprSpecifierAST* ast) override;
  void visit(InlineSpecifierAST* ast) override;
  void visit(StaticSpecifierAST* ast) override;
  void visit(ExternSpecifierAST* ast) override;
  void visit(ThreadLocalSpecifierAST* ast) override;
  void visit(ThreadSpecifierAST* ast) override;
  void visit(MutableSpecifierAST* ast) override;
  void visit(VirtualSpecifierAST* ast) override;
  void visit(ExplicitSpecifierAST* ast) override;
  void visit(AutoTypeSpecifierAST* ast) override;
  void visit(VoidTypeSpecifierAST* ast) override;
  void visit(VaListTypeSpecifierAST* ast) override;
  void visit(IntegralTypeSpecifierAST* ast) override;
  void visit(FloatingPointTypeSpecifierAST* ast) override;
  void visit(ComplexTypeSpecifierAST* ast) override;
  void visit(NamedTypeSpecifierAST* ast) override;
  void visit(AtomicTypeSpecifierAST* ast) override;
  void visit(UnderlyingTypeSpecifierAST* ast) override;
  void visit(ElaboratedTypeSpecifierAST* ast) override;
  void visit(DecltypeAutoSpecifierAST* ast) override;
  void visit(DecltypeSpecifierAST* ast) override;
  void visit(TypeofSpecifierAST* ast) override;
  void visit(PlaceholderTypeSpecifierAST* ast) override;
  void visit(ConstQualifierAST* ast) override;
  void visit(VolatileQualifierAST* ast) override;
  void visit(RestrictQualifierAST* ast) override;
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

 protected:
  template <typename T>
  T accept(T ast) {
    if (!ast) return nullptr;
    AST* copy = nullptr;
    std::swap(copy_, copy);
    ast->accept(this);
    std::swap(copy_, copy);
    return static_cast<T>(copy);
  }

  Arena* arena_ = nullptr;
  AST* copy_ = nullptr;
};

}  // namespace cxx
