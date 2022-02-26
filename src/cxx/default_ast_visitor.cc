// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast.h>
#include <cxx/default_ast_visitor.h>

#include <stdexcept>

namespace cxx {

// AST
void DefaultASTVisitor::visit(TypeIdAST* ast) {
  throw std::runtime_error("visit(TypeIdAST): not implemented");
}

void DefaultASTVisitor::visit(NestedNameSpecifierAST* ast) {
  throw std::runtime_error("visit(NestedNameSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(UsingDeclaratorAST* ast) {
  throw std::runtime_error("visit(UsingDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(HandlerAST* ast) {
  throw std::runtime_error("visit(HandlerAST): not implemented");
}

void DefaultASTVisitor::visit(EnumBaseAST* ast) {
  throw std::runtime_error("visit(EnumBaseAST): not implemented");
}

void DefaultASTVisitor::visit(EnumeratorAST* ast) {
  throw std::runtime_error("visit(EnumeratorAST): not implemented");
}

void DefaultASTVisitor::visit(DeclaratorAST* ast) {
  throw std::runtime_error("visit(DeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(InitDeclaratorAST* ast) {
  throw std::runtime_error("visit(InitDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(BaseSpecifierAST* ast) {
  throw std::runtime_error("visit(BaseSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(BaseClauseAST* ast) {
  throw std::runtime_error("visit(BaseClauseAST): not implemented");
}

void DefaultASTVisitor::visit(NewTypeIdAST* ast) {
  throw std::runtime_error("visit(NewTypeIdAST): not implemented");
}

void DefaultASTVisitor::visit(RequiresClauseAST* ast) {
  throw std::runtime_error("visit(RequiresClauseAST): not implemented");
}

void DefaultASTVisitor::visit(ParameterDeclarationClauseAST* ast) {
  throw std::runtime_error(
      "visit(ParameterDeclarationClauseAST): not implemented");
}

void DefaultASTVisitor::visit(ParametersAndQualifiersAST* ast) {
  throw std::runtime_error(
      "visit(ParametersAndQualifiersAST): not implemented");
}

void DefaultASTVisitor::visit(LambdaIntroducerAST* ast) {
  throw std::runtime_error("visit(LambdaIntroducerAST): not implemented");
}

void DefaultASTVisitor::visit(LambdaDeclaratorAST* ast) {
  throw std::runtime_error("visit(LambdaDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(TrailingReturnTypeAST* ast) {
  throw std::runtime_error("visit(TrailingReturnTypeAST): not implemented");
}

void DefaultASTVisitor::visit(CtorInitializerAST* ast) {
  throw std::runtime_error("visit(CtorInitializerAST): not implemented");
}

void DefaultASTVisitor::visit(RequirementBodyAST* ast) {
  throw std::runtime_error("visit(RequirementBodyAST): not implemented");
}

void DefaultASTVisitor::visit(TypeConstraintAST* ast) {
  throw std::runtime_error("visit(TypeConstraintAST): not implemented");
}

void DefaultASTVisitor::visit(GlobalModuleFragmentAST* ast) {
  throw std::runtime_error("visit(GlobalModuleFragmentAST): not implemented");
}

void DefaultASTVisitor::visit(PrivateModuleFragmentAST* ast) {
  throw std::runtime_error("visit(PrivateModuleFragmentAST): not implemented");
}

void DefaultASTVisitor::visit(ModuleDeclarationAST* ast) {
  throw std::runtime_error("visit(ModuleDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(ModuleNameAST* ast) {
  throw std::runtime_error("visit(ModuleNameAST): not implemented");
}

void DefaultASTVisitor::visit(ImportNameAST* ast) {
  throw std::runtime_error("visit(ImportNameAST): not implemented");
}

void DefaultASTVisitor::visit(ModulePartitionAST* ast) {
  throw std::runtime_error("visit(ModulePartitionAST): not implemented");
}

// RequirementAST
void DefaultASTVisitor::visit(SimpleRequirementAST* ast) {
  throw std::runtime_error("visit(SimpleRequirementAST): not implemented");
}

void DefaultASTVisitor::visit(CompoundRequirementAST* ast) {
  throw std::runtime_error("visit(CompoundRequirementAST): not implemented");
}

void DefaultASTVisitor::visit(TypeRequirementAST* ast) {
  throw std::runtime_error("visit(TypeRequirementAST): not implemented");
}

void DefaultASTVisitor::visit(NestedRequirementAST* ast) {
  throw std::runtime_error("visit(NestedRequirementAST): not implemented");
}

// TemplateArgumentAST
void DefaultASTVisitor::visit(TypeTemplateArgumentAST* ast) {
  throw std::runtime_error("visit(TypeTemplateArgumentAST): not implemented");
}

void DefaultASTVisitor::visit(ExpressionTemplateArgumentAST* ast) {
  throw std::runtime_error(
      "visit(ExpressionTemplateArgumentAST): not implemented");
}

// MemInitializerAST
void DefaultASTVisitor::visit(ParenMemInitializerAST* ast) {
  throw std::runtime_error("visit(ParenMemInitializerAST): not implemented");
}

void DefaultASTVisitor::visit(BracedMemInitializerAST* ast) {
  throw std::runtime_error("visit(BracedMemInitializerAST): not implemented");
}

// LambdaCaptureAST
void DefaultASTVisitor::visit(ThisLambdaCaptureAST* ast) {
  throw std::runtime_error("visit(ThisLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(DerefThisLambdaCaptureAST* ast) {
  throw std::runtime_error("visit(DerefThisLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(SimpleLambdaCaptureAST* ast) {
  throw std::runtime_error("visit(SimpleLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(RefLambdaCaptureAST* ast) {
  throw std::runtime_error("visit(RefLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(RefInitLambdaCaptureAST* ast) {
  throw std::runtime_error("visit(RefInitLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(InitLambdaCaptureAST* ast) {
  throw std::runtime_error("visit(InitLambdaCaptureAST): not implemented");
}

// InitializerAST
void DefaultASTVisitor::visit(EqualInitializerAST* ast) {
  throw std::runtime_error("visit(EqualInitializerAST): not implemented");
}

void DefaultASTVisitor::visit(BracedInitListAST* ast) {
  throw std::runtime_error("visit(BracedInitListAST): not implemented");
}

void DefaultASTVisitor::visit(ParenInitializerAST* ast) {
  throw std::runtime_error("visit(ParenInitializerAST): not implemented");
}

// NewInitializerAST
void DefaultASTVisitor::visit(NewParenInitializerAST* ast) {
  throw std::runtime_error("visit(NewParenInitializerAST): not implemented");
}

void DefaultASTVisitor::visit(NewBracedInitializerAST* ast) {
  throw std::runtime_error("visit(NewBracedInitializerAST): not implemented");
}

// ExceptionDeclarationAST
void DefaultASTVisitor::visit(EllipsisExceptionDeclarationAST* ast) {
  throw std::runtime_error(
      "visit(EllipsisExceptionDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(TypeExceptionDeclarationAST* ast) {
  throw std::runtime_error(
      "visit(TypeExceptionDeclarationAST): not implemented");
}

// FunctionBodyAST
void DefaultASTVisitor::visit(DefaultFunctionBodyAST* ast) {
  throw std::runtime_error("visit(DefaultFunctionBodyAST): not implemented");
}

void DefaultASTVisitor::visit(CompoundStatementFunctionBodyAST* ast) {
  throw std::runtime_error(
      "visit(CompoundStatementFunctionBodyAST): not implemented");
}

void DefaultASTVisitor::visit(TryStatementFunctionBodyAST* ast) {
  throw std::runtime_error(
      "visit(TryStatementFunctionBodyAST): not implemented");
}

void DefaultASTVisitor::visit(DeleteFunctionBodyAST* ast) {
  throw std::runtime_error("visit(DeleteFunctionBodyAST): not implemented");
}

// UnitAST
void DefaultASTVisitor::visit(TranslationUnitAST* ast) {
  throw std::runtime_error("visit(TranslationUnitAST): not implemented");
}

void DefaultASTVisitor::visit(ModuleUnitAST* ast) {
  throw std::runtime_error("visit(ModuleUnitAST): not implemented");
}

// ExpressionAST
void DefaultASTVisitor::visit(ThisExpressionAST* ast) {
  throw std::runtime_error("visit(ThisExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(CharLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(CharLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(BoolLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(BoolLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(IntLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(IntLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(FloatLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(FloatLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(NullptrLiteralExpressionAST* ast) {
  throw std::runtime_error(
      "visit(NullptrLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(StringLiteralExpressionAST* ast) {
  throw std::runtime_error(
      "visit(StringLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(UserDefinedStringLiteralExpressionAST* ast) {
  throw std::runtime_error(
      "visit(UserDefinedStringLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(IdExpressionAST* ast) {
  throw std::runtime_error("visit(IdExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(RequiresExpressionAST* ast) {
  throw std::runtime_error("visit(RequiresExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(NestedExpressionAST* ast) {
  throw std::runtime_error("visit(NestedExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(RightFoldExpressionAST* ast) {
  throw std::runtime_error("visit(RightFoldExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(LeftFoldExpressionAST* ast) {
  throw std::runtime_error("visit(LeftFoldExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(FoldExpressionAST* ast) {
  throw std::runtime_error("visit(FoldExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(LambdaExpressionAST* ast) {
  throw std::runtime_error("visit(LambdaExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(SizeofExpressionAST* ast) {
  throw std::runtime_error("visit(SizeofExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(SizeofTypeExpressionAST* ast) {
  throw std::runtime_error("visit(SizeofTypeExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(SizeofPackExpressionAST* ast) {
  throw std::runtime_error("visit(SizeofPackExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(TypeidExpressionAST* ast) {
  throw std::runtime_error("visit(TypeidExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(TypeidOfTypeExpressionAST* ast) {
  throw std::runtime_error("visit(TypeidOfTypeExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(AlignofExpressionAST* ast) {
  throw std::runtime_error("visit(AlignofExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(TypeTraitsExpressionAST* ast) {
  throw std::runtime_error("visit(TypeTraitsExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(UnaryExpressionAST* ast) {
  throw std::runtime_error("visit(UnaryExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(BinaryExpressionAST* ast) {
  throw std::runtime_error("visit(BinaryExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(AssignmentExpressionAST* ast) {
  throw std::runtime_error("visit(AssignmentExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(BracedTypeConstructionAST* ast) {
  throw std::runtime_error("visit(BracedTypeConstructionAST): not implemented");
}

void DefaultASTVisitor::visit(TypeConstructionAST* ast) {
  throw std::runtime_error("visit(TypeConstructionAST): not implemented");
}

void DefaultASTVisitor::visit(CallExpressionAST* ast) {
  throw std::runtime_error("visit(CallExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(SubscriptExpressionAST* ast) {
  throw std::runtime_error("visit(SubscriptExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(MemberExpressionAST* ast) {
  throw std::runtime_error("visit(MemberExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(PostIncrExpressionAST* ast) {
  throw std::runtime_error("visit(PostIncrExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(ConditionalExpressionAST* ast) {
  throw std::runtime_error("visit(ConditionalExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(ImplicitCastExpressionAST* ast) {
  throw std::runtime_error("visit(ImplicitCastExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(CastExpressionAST* ast) {
  throw std::runtime_error("visit(CastExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(CppCastExpressionAST* ast) {
  throw std::runtime_error("visit(CppCastExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(NewExpressionAST* ast) {
  throw std::runtime_error("visit(NewExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(DeleteExpressionAST* ast) {
  throw std::runtime_error("visit(DeleteExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(ThrowExpressionAST* ast) {
  throw std::runtime_error("visit(ThrowExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(NoexceptExpressionAST* ast) {
  throw std::runtime_error("visit(NoexceptExpressionAST): not implemented");
}

// StatementAST
void DefaultASTVisitor::visit(LabeledStatementAST* ast) {
  throw std::runtime_error("visit(LabeledStatementAST): not implemented");
}

void DefaultASTVisitor::visit(CaseStatementAST* ast) {
  throw std::runtime_error("visit(CaseStatementAST): not implemented");
}

void DefaultASTVisitor::visit(DefaultStatementAST* ast) {
  throw std::runtime_error("visit(DefaultStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ExpressionStatementAST* ast) {
  throw std::runtime_error("visit(ExpressionStatementAST): not implemented");
}

void DefaultASTVisitor::visit(CompoundStatementAST* ast) {
  throw std::runtime_error("visit(CompoundStatementAST): not implemented");
}

void DefaultASTVisitor::visit(IfStatementAST* ast) {
  throw std::runtime_error("visit(IfStatementAST): not implemented");
}

void DefaultASTVisitor::visit(SwitchStatementAST* ast) {
  throw std::runtime_error("visit(SwitchStatementAST): not implemented");
}

void DefaultASTVisitor::visit(WhileStatementAST* ast) {
  throw std::runtime_error("visit(WhileStatementAST): not implemented");
}

void DefaultASTVisitor::visit(DoStatementAST* ast) {
  throw std::runtime_error("visit(DoStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ForRangeStatementAST* ast) {
  throw std::runtime_error("visit(ForRangeStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ForStatementAST* ast) {
  throw std::runtime_error("visit(ForStatementAST): not implemented");
}

void DefaultASTVisitor::visit(BreakStatementAST* ast) {
  throw std::runtime_error("visit(BreakStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ContinueStatementAST* ast) {
  throw std::runtime_error("visit(ContinueStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ReturnStatementAST* ast) {
  throw std::runtime_error("visit(ReturnStatementAST): not implemented");
}

void DefaultASTVisitor::visit(GotoStatementAST* ast) {
  throw std::runtime_error("visit(GotoStatementAST): not implemented");
}

void DefaultASTVisitor::visit(CoroutineReturnStatementAST* ast) {
  throw std::runtime_error(
      "visit(CoroutineReturnStatementAST): not implemented");
}

void DefaultASTVisitor::visit(DeclarationStatementAST* ast) {
  throw std::runtime_error("visit(DeclarationStatementAST): not implemented");
}

void DefaultASTVisitor::visit(TryBlockStatementAST* ast) {
  throw std::runtime_error("visit(TryBlockStatementAST): not implemented");
}

// DeclarationAST
void DefaultASTVisitor::visit(AccessDeclarationAST* ast) {
  throw std::runtime_error("visit(AccessDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(FunctionDefinitionAST* ast) {
  throw std::runtime_error("visit(FunctionDefinitionAST): not implemented");
}

void DefaultASTVisitor::visit(ConceptDefinitionAST* ast) {
  throw std::runtime_error("visit(ConceptDefinitionAST): not implemented");
}

void DefaultASTVisitor::visit(ForRangeDeclarationAST* ast) {
  throw std::runtime_error("visit(ForRangeDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(AliasDeclarationAST* ast) {
  throw std::runtime_error("visit(AliasDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(SimpleDeclarationAST* ast) {
  throw std::runtime_error("visit(SimpleDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(StaticAssertDeclarationAST* ast) {
  throw std::runtime_error(
      "visit(StaticAssertDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(EmptyDeclarationAST* ast) {
  throw std::runtime_error("visit(EmptyDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(AttributeDeclarationAST* ast) {
  throw std::runtime_error("visit(AttributeDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(OpaqueEnumDeclarationAST* ast) {
  throw std::runtime_error("visit(OpaqueEnumDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(UsingEnumDeclarationAST* ast) {
  throw std::runtime_error("visit(UsingEnumDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(NamespaceDefinitionAST* ast) {
  throw std::runtime_error("visit(NamespaceDefinitionAST): not implemented");
}

void DefaultASTVisitor::visit(NamespaceAliasDefinitionAST* ast) {
  throw std::runtime_error(
      "visit(NamespaceAliasDefinitionAST): not implemented");
}

void DefaultASTVisitor::visit(UsingDirectiveAST* ast) {
  throw std::runtime_error("visit(UsingDirectiveAST): not implemented");
}

void DefaultASTVisitor::visit(UsingDeclarationAST* ast) {
  throw std::runtime_error("visit(UsingDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(AsmDeclarationAST* ast) {
  throw std::runtime_error("visit(AsmDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(ExportDeclarationAST* ast) {
  throw std::runtime_error("visit(ExportDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(ExportCompoundDeclarationAST* ast) {
  throw std::runtime_error(
      "visit(ExportCompoundDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(ModuleImportDeclarationAST* ast) {
  throw std::runtime_error(
      "visit(ModuleImportDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(TemplateDeclarationAST* ast) {
  throw std::runtime_error("visit(TemplateDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(TypenameTypeParameterAST* ast) {
  throw std::runtime_error("visit(TypenameTypeParameterAST): not implemented");
}

void DefaultASTVisitor::visit(TypenamePackTypeParameterAST* ast) {
  throw std::runtime_error(
      "visit(TypenamePackTypeParameterAST): not implemented");
}

void DefaultASTVisitor::visit(TemplateTypeParameterAST* ast) {
  throw std::runtime_error("visit(TemplateTypeParameterAST): not implemented");
}

void DefaultASTVisitor::visit(TemplatePackTypeParameterAST* ast) {
  throw std::runtime_error(
      "visit(TemplatePackTypeParameterAST): not implemented");
}

void DefaultASTVisitor::visit(DeductionGuideAST* ast) {
  throw std::runtime_error("visit(DeductionGuideAST): not implemented");
}

void DefaultASTVisitor::visit(ExplicitInstantiationAST* ast) {
  throw std::runtime_error("visit(ExplicitInstantiationAST): not implemented");
}

void DefaultASTVisitor::visit(ParameterDeclarationAST* ast) {
  throw std::runtime_error("visit(ParameterDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(LinkageSpecificationAST* ast) {
  throw std::runtime_error("visit(LinkageSpecificationAST): not implemented");
}

// NameAST
void DefaultASTVisitor::visit(SimpleNameAST* ast) {
  throw std::runtime_error("visit(SimpleNameAST): not implemented");
}

void DefaultASTVisitor::visit(DestructorNameAST* ast) {
  throw std::runtime_error("visit(DestructorNameAST): not implemented");
}

void DefaultASTVisitor::visit(DecltypeNameAST* ast) {
  throw std::runtime_error("visit(DecltypeNameAST): not implemented");
}

void DefaultASTVisitor::visit(OperatorNameAST* ast) {
  throw std::runtime_error("visit(OperatorNameAST): not implemented");
}

void DefaultASTVisitor::visit(ConversionNameAST* ast) {
  throw std::runtime_error("visit(ConversionNameAST): not implemented");
}

void DefaultASTVisitor::visit(TemplateNameAST* ast) {
  throw std::runtime_error("visit(TemplateNameAST): not implemented");
}

void DefaultASTVisitor::visit(QualifiedNameAST* ast) {
  throw std::runtime_error("visit(QualifiedNameAST): not implemented");
}

// SpecifierAST
void DefaultASTVisitor::visit(TypedefSpecifierAST* ast) {
  throw std::runtime_error("visit(TypedefSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(FriendSpecifierAST* ast) {
  throw std::runtime_error("visit(FriendSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ConstevalSpecifierAST* ast) {
  throw std::runtime_error("visit(ConstevalSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ConstinitSpecifierAST* ast) {
  throw std::runtime_error("visit(ConstinitSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ConstexprSpecifierAST* ast) {
  throw std::runtime_error("visit(ConstexprSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(InlineSpecifierAST* ast) {
  throw std::runtime_error("visit(InlineSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(StaticSpecifierAST* ast) {
  throw std::runtime_error("visit(StaticSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ExternSpecifierAST* ast) {
  throw std::runtime_error("visit(ExternSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ThreadLocalSpecifierAST* ast) {
  throw std::runtime_error("visit(ThreadLocalSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ThreadSpecifierAST* ast) {
  throw std::runtime_error("visit(ThreadSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(MutableSpecifierAST* ast) {
  throw std::runtime_error("visit(MutableSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(VirtualSpecifierAST* ast) {
  throw std::runtime_error("visit(VirtualSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ExplicitSpecifierAST* ast) {
  throw std::runtime_error("visit(ExplicitSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(AutoTypeSpecifierAST* ast) {
  throw std::runtime_error("visit(AutoTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(VoidTypeSpecifierAST* ast) {
  throw std::runtime_error("visit(VoidTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(VaListTypeSpecifierAST* ast) {
  throw std::runtime_error("visit(VaListTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(IntegralTypeSpecifierAST* ast) {
  throw std::runtime_error("visit(IntegralTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(FloatingPointTypeSpecifierAST* ast) {
  throw std::runtime_error(
      "visit(FloatingPointTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ComplexTypeSpecifierAST* ast) {
  throw std::runtime_error("visit(ComplexTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(NamedTypeSpecifierAST* ast) {
  throw std::runtime_error("visit(NamedTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(AtomicTypeSpecifierAST* ast) {
  throw std::runtime_error("visit(AtomicTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(UnderlyingTypeSpecifierAST* ast) {
  throw std::runtime_error(
      "visit(UnderlyingTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ElaboratedTypeSpecifierAST* ast) {
  throw std::runtime_error(
      "visit(ElaboratedTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(DecltypeAutoSpecifierAST* ast) {
  throw std::runtime_error("visit(DecltypeAutoSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(DecltypeSpecifierAST* ast) {
  throw std::runtime_error("visit(DecltypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(PlaceholderTypeSpecifierAST* ast) {
  throw std::runtime_error(
      "visit(PlaceholderTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ConstQualifierAST* ast) {
  throw std::runtime_error("visit(ConstQualifierAST): not implemented");
}

void DefaultASTVisitor::visit(VolatileQualifierAST* ast) {
  throw std::runtime_error("visit(VolatileQualifierAST): not implemented");
}

void DefaultASTVisitor::visit(RestrictQualifierAST* ast) {
  throw std::runtime_error("visit(RestrictQualifierAST): not implemented");
}

void DefaultASTVisitor::visit(EnumSpecifierAST* ast) {
  throw std::runtime_error("visit(EnumSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ClassSpecifierAST* ast) {
  throw std::runtime_error("visit(ClassSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(TypenameSpecifierAST* ast) {
  throw std::runtime_error("visit(TypenameSpecifierAST): not implemented");
}

// CoreDeclaratorAST
void DefaultASTVisitor::visit(IdDeclaratorAST* ast) {
  throw std::runtime_error("visit(IdDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(NestedDeclaratorAST* ast) {
  throw std::runtime_error("visit(NestedDeclaratorAST): not implemented");
}

// PtrOperatorAST
void DefaultASTVisitor::visit(PointerOperatorAST* ast) {
  throw std::runtime_error("visit(PointerOperatorAST): not implemented");
}

void DefaultASTVisitor::visit(ReferenceOperatorAST* ast) {
  throw std::runtime_error("visit(ReferenceOperatorAST): not implemented");
}

void DefaultASTVisitor::visit(PtrToMemberOperatorAST* ast) {
  throw std::runtime_error("visit(PtrToMemberOperatorAST): not implemented");
}

// DeclaratorModifierAST
void DefaultASTVisitor::visit(FunctionDeclaratorAST* ast) {
  throw std::runtime_error("visit(FunctionDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(ArrayDeclaratorAST* ast) {
  throw std::runtime_error("visit(ArrayDeclaratorAST): not implemented");
}

}  // namespace cxx
