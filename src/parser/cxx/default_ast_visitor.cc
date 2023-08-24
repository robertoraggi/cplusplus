// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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
  cxx_runtime_error("visit(TypeIdAST): not implemented");
}

void DefaultASTVisitor::visit(NestedNameSpecifierAST* ast) {
  cxx_runtime_error("visit(NestedNameSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(UsingDeclaratorAST* ast) {
  cxx_runtime_error("visit(UsingDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(HandlerAST* ast) {
  cxx_runtime_error("visit(HandlerAST): not implemented");
}

void DefaultASTVisitor::visit(EnumBaseAST* ast) {
  cxx_runtime_error("visit(EnumBaseAST): not implemented");
}

void DefaultASTVisitor::visit(EnumeratorAST* ast) {
  cxx_runtime_error("visit(EnumeratorAST): not implemented");
}

void DefaultASTVisitor::visit(DeclaratorAST* ast) {
  cxx_runtime_error("visit(DeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(InitDeclaratorAST* ast) {
  cxx_runtime_error("visit(InitDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(BaseSpecifierAST* ast) {
  cxx_runtime_error("visit(BaseSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(BaseClauseAST* ast) {
  cxx_runtime_error("visit(BaseClauseAST): not implemented");
}

void DefaultASTVisitor::visit(NewTypeIdAST* ast) {
  cxx_runtime_error("visit(NewTypeIdAST): not implemented");
}

void DefaultASTVisitor::visit(RequiresClauseAST* ast) {
  cxx_runtime_error("visit(RequiresClauseAST): not implemented");
}

void DefaultASTVisitor::visit(ParameterDeclarationClauseAST* ast) {
  cxx_runtime_error("visit(ParameterDeclarationClauseAST): not implemented");
}

void DefaultASTVisitor::visit(ParametersAndQualifiersAST* ast) {
  cxx_runtime_error("visit(ParametersAndQualifiersAST): not implemented");
}

void DefaultASTVisitor::visit(LambdaIntroducerAST* ast) {
  cxx_runtime_error("visit(LambdaIntroducerAST): not implemented");
}

void DefaultASTVisitor::visit(LambdaDeclaratorAST* ast) {
  cxx_runtime_error("visit(LambdaDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(TrailingReturnTypeAST* ast) {
  cxx_runtime_error("visit(TrailingReturnTypeAST): not implemented");
}

void DefaultASTVisitor::visit(CtorInitializerAST* ast) {
  cxx_runtime_error("visit(CtorInitializerAST): not implemented");
}

void DefaultASTVisitor::visit(RequirementBodyAST* ast) {
  cxx_runtime_error("visit(RequirementBodyAST): not implemented");
}

void DefaultASTVisitor::visit(TypeConstraintAST* ast) {
  cxx_runtime_error("visit(TypeConstraintAST): not implemented");
}

void DefaultASTVisitor::visit(GlobalModuleFragmentAST* ast) {
  cxx_runtime_error("visit(GlobalModuleFragmentAST): not implemented");
}

void DefaultASTVisitor::visit(PrivateModuleFragmentAST* ast) {
  cxx_runtime_error("visit(PrivateModuleFragmentAST): not implemented");
}

void DefaultASTVisitor::visit(ModuleDeclarationAST* ast) {
  cxx_runtime_error("visit(ModuleDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(ModuleNameAST* ast) {
  cxx_runtime_error("visit(ModuleNameAST): not implemented");
}

void DefaultASTVisitor::visit(ImportNameAST* ast) {
  cxx_runtime_error("visit(ImportNameAST): not implemented");
}

void DefaultASTVisitor::visit(ModulePartitionAST* ast) {
  cxx_runtime_error("visit(ModulePartitionAST): not implemented");
}

void DefaultASTVisitor::visit(AttributeArgumentClauseAST* ast) {
  cxx_runtime_error("visit(AttributeArgumentClauseAST): not implemented");
}

void DefaultASTVisitor::visit(AttributeAST* ast) {
  cxx_runtime_error("visit(AttributeAST): not implemented");
}

void DefaultASTVisitor::visit(AttributeUsingPrefixAST* ast) {
  cxx_runtime_error("visit(AttributeUsingPrefixAST): not implemented");
}

void DefaultASTVisitor::visit(DesignatorAST* ast) {
  cxx_runtime_error("visit(DesignatorAST): not implemented");
}

// ExceptionSpecifierAST
void DefaultASTVisitor::visit(ThrowExceptionSpecifierAST* ast) {
  cxx_runtime_error("visit(ThrowExceptionSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(NoexceptSpecifierAST* ast) {
  cxx_runtime_error("visit(NoexceptSpecifierAST): not implemented");
}

// ExpressionAST
void DefaultASTVisitor::visit(DesignatedInitializerClauseAST* ast) {
  cxx_runtime_error("visit(DesignatedInitializerClauseAST): not implemented");
}

void DefaultASTVisitor::visit(ThisExpressionAST* ast) {
  cxx_runtime_error("visit(ThisExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(CharLiteralExpressionAST* ast) {
  cxx_runtime_error("visit(CharLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(BoolLiteralExpressionAST* ast) {
  cxx_runtime_error("visit(BoolLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(IntLiteralExpressionAST* ast) {
  cxx_runtime_error("visit(IntLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(FloatLiteralExpressionAST* ast) {
  cxx_runtime_error("visit(FloatLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(NullptrLiteralExpressionAST* ast) {
  cxx_runtime_error("visit(NullptrLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(StringLiteralExpressionAST* ast) {
  cxx_runtime_error("visit(StringLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(UserDefinedStringLiteralExpressionAST* ast) {
  cxx_runtime_error(
      "visit(UserDefinedStringLiteralExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(IdExpressionAST* ast) {
  cxx_runtime_error("visit(IdExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(RequiresExpressionAST* ast) {
  cxx_runtime_error("visit(RequiresExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(NestedExpressionAST* ast) {
  cxx_runtime_error("visit(NestedExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(RightFoldExpressionAST* ast) {
  cxx_runtime_error("visit(RightFoldExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(LeftFoldExpressionAST* ast) {
  cxx_runtime_error("visit(LeftFoldExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(FoldExpressionAST* ast) {
  cxx_runtime_error("visit(FoldExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(LambdaExpressionAST* ast) {
  cxx_runtime_error("visit(LambdaExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(SizeofExpressionAST* ast) {
  cxx_runtime_error("visit(SizeofExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(SizeofTypeExpressionAST* ast) {
  cxx_runtime_error("visit(SizeofTypeExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(SizeofPackExpressionAST* ast) {
  cxx_runtime_error("visit(SizeofPackExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(TypeidExpressionAST* ast) {
  cxx_runtime_error("visit(TypeidExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(TypeidOfTypeExpressionAST* ast) {
  cxx_runtime_error("visit(TypeidOfTypeExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(AlignofExpressionAST* ast) {
  cxx_runtime_error("visit(AlignofExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(TypeTraitsExpressionAST* ast) {
  cxx_runtime_error("visit(TypeTraitsExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(UnaryExpressionAST* ast) {
  cxx_runtime_error("visit(UnaryExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(BinaryExpressionAST* ast) {
  cxx_runtime_error("visit(BinaryExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(AssignmentExpressionAST* ast) {
  cxx_runtime_error("visit(AssignmentExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(BracedTypeConstructionAST* ast) {
  cxx_runtime_error("visit(BracedTypeConstructionAST): not implemented");
}

void DefaultASTVisitor::visit(TypeConstructionAST* ast) {
  cxx_runtime_error("visit(TypeConstructionAST): not implemented");
}

void DefaultASTVisitor::visit(CallExpressionAST* ast) {
  cxx_runtime_error("visit(CallExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(SubscriptExpressionAST* ast) {
  cxx_runtime_error("visit(SubscriptExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(MemberExpressionAST* ast) {
  cxx_runtime_error("visit(MemberExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(PostIncrExpressionAST* ast) {
  cxx_runtime_error("visit(PostIncrExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(ConditionalExpressionAST* ast) {
  cxx_runtime_error("visit(ConditionalExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(ImplicitCastExpressionAST* ast) {
  cxx_runtime_error("visit(ImplicitCastExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(CastExpressionAST* ast) {
  cxx_runtime_error("visit(CastExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(CppCastExpressionAST* ast) {
  cxx_runtime_error("visit(CppCastExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(NewExpressionAST* ast) {
  cxx_runtime_error("visit(NewExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(DeleteExpressionAST* ast) {
  cxx_runtime_error("visit(DeleteExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(ThrowExpressionAST* ast) {
  cxx_runtime_error("visit(ThrowExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(NoexceptExpressionAST* ast) {
  cxx_runtime_error("visit(NoexceptExpressionAST): not implemented");
}

void DefaultASTVisitor::visit(EqualInitializerAST* ast) {
  cxx_runtime_error("visit(EqualInitializerAST): not implemented");
}

void DefaultASTVisitor::visit(BracedInitListAST* ast) {
  cxx_runtime_error("visit(BracedInitListAST): not implemented");
}

void DefaultASTVisitor::visit(ParenInitializerAST* ast) {
  cxx_runtime_error("visit(ParenInitializerAST): not implemented");
}

// RequirementAST
void DefaultASTVisitor::visit(SimpleRequirementAST* ast) {
  cxx_runtime_error("visit(SimpleRequirementAST): not implemented");
}

void DefaultASTVisitor::visit(CompoundRequirementAST* ast) {
  cxx_runtime_error("visit(CompoundRequirementAST): not implemented");
}

void DefaultASTVisitor::visit(TypeRequirementAST* ast) {
  cxx_runtime_error("visit(TypeRequirementAST): not implemented");
}

void DefaultASTVisitor::visit(NestedRequirementAST* ast) {
  cxx_runtime_error("visit(NestedRequirementAST): not implemented");
}

// TemplateArgumentAST
void DefaultASTVisitor::visit(TypeTemplateArgumentAST* ast) {
  cxx_runtime_error("visit(TypeTemplateArgumentAST): not implemented");
}

void DefaultASTVisitor::visit(ExpressionTemplateArgumentAST* ast) {
  cxx_runtime_error("visit(ExpressionTemplateArgumentAST): not implemented");
}

// MemInitializerAST
void DefaultASTVisitor::visit(ParenMemInitializerAST* ast) {
  cxx_runtime_error("visit(ParenMemInitializerAST): not implemented");
}

void DefaultASTVisitor::visit(BracedMemInitializerAST* ast) {
  cxx_runtime_error("visit(BracedMemInitializerAST): not implemented");
}

// LambdaCaptureAST
void DefaultASTVisitor::visit(ThisLambdaCaptureAST* ast) {
  cxx_runtime_error("visit(ThisLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(DerefThisLambdaCaptureAST* ast) {
  cxx_runtime_error("visit(DerefThisLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(SimpleLambdaCaptureAST* ast) {
  cxx_runtime_error("visit(SimpleLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(RefLambdaCaptureAST* ast) {
  cxx_runtime_error("visit(RefLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(RefInitLambdaCaptureAST* ast) {
  cxx_runtime_error("visit(RefInitLambdaCaptureAST): not implemented");
}

void DefaultASTVisitor::visit(InitLambdaCaptureAST* ast) {
  cxx_runtime_error("visit(InitLambdaCaptureAST): not implemented");
}

// NewInitializerAST
void DefaultASTVisitor::visit(NewParenInitializerAST* ast) {
  cxx_runtime_error("visit(NewParenInitializerAST): not implemented");
}

void DefaultASTVisitor::visit(NewBracedInitializerAST* ast) {
  cxx_runtime_error("visit(NewBracedInitializerAST): not implemented");
}

// ExceptionDeclarationAST
void DefaultASTVisitor::visit(EllipsisExceptionDeclarationAST* ast) {
  cxx_runtime_error("visit(EllipsisExceptionDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(TypeExceptionDeclarationAST* ast) {
  cxx_runtime_error("visit(TypeExceptionDeclarationAST): not implemented");
}

// FunctionBodyAST
void DefaultASTVisitor::visit(DefaultFunctionBodyAST* ast) {
  cxx_runtime_error("visit(DefaultFunctionBodyAST): not implemented");
}

void DefaultASTVisitor::visit(CompoundStatementFunctionBodyAST* ast) {
  cxx_runtime_error("visit(CompoundStatementFunctionBodyAST): not implemented");
}

void DefaultASTVisitor::visit(TryStatementFunctionBodyAST* ast) {
  cxx_runtime_error("visit(TryStatementFunctionBodyAST): not implemented");
}

void DefaultASTVisitor::visit(DeleteFunctionBodyAST* ast) {
  cxx_runtime_error("visit(DeleteFunctionBodyAST): not implemented");
}

// UnitAST
void DefaultASTVisitor::visit(TranslationUnitAST* ast) {
  cxx_runtime_error("visit(TranslationUnitAST): not implemented");
}

void DefaultASTVisitor::visit(ModuleUnitAST* ast) {
  cxx_runtime_error("visit(ModuleUnitAST): not implemented");
}

// StatementAST
void DefaultASTVisitor::visit(LabeledStatementAST* ast) {
  cxx_runtime_error("visit(LabeledStatementAST): not implemented");
}

void DefaultASTVisitor::visit(CaseStatementAST* ast) {
  cxx_runtime_error("visit(CaseStatementAST): not implemented");
}

void DefaultASTVisitor::visit(DefaultStatementAST* ast) {
  cxx_runtime_error("visit(DefaultStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ExpressionStatementAST* ast) {
  cxx_runtime_error("visit(ExpressionStatementAST): not implemented");
}

void DefaultASTVisitor::visit(CompoundStatementAST* ast) {
  cxx_runtime_error("visit(CompoundStatementAST): not implemented");
}

void DefaultASTVisitor::visit(IfStatementAST* ast) {
  cxx_runtime_error("visit(IfStatementAST): not implemented");
}

void DefaultASTVisitor::visit(SwitchStatementAST* ast) {
  cxx_runtime_error("visit(SwitchStatementAST): not implemented");
}

void DefaultASTVisitor::visit(WhileStatementAST* ast) {
  cxx_runtime_error("visit(WhileStatementAST): not implemented");
}

void DefaultASTVisitor::visit(DoStatementAST* ast) {
  cxx_runtime_error("visit(DoStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ForRangeStatementAST* ast) {
  cxx_runtime_error("visit(ForRangeStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ForStatementAST* ast) {
  cxx_runtime_error("visit(ForStatementAST): not implemented");
}

void DefaultASTVisitor::visit(BreakStatementAST* ast) {
  cxx_runtime_error("visit(BreakStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ContinueStatementAST* ast) {
  cxx_runtime_error("visit(ContinueStatementAST): not implemented");
}

void DefaultASTVisitor::visit(ReturnStatementAST* ast) {
  cxx_runtime_error("visit(ReturnStatementAST): not implemented");
}

void DefaultASTVisitor::visit(GotoStatementAST* ast) {
  cxx_runtime_error("visit(GotoStatementAST): not implemented");
}

void DefaultASTVisitor::visit(CoroutineReturnStatementAST* ast) {
  cxx_runtime_error("visit(CoroutineReturnStatementAST): not implemented");
}

void DefaultASTVisitor::visit(DeclarationStatementAST* ast) {
  cxx_runtime_error("visit(DeclarationStatementAST): not implemented");
}

void DefaultASTVisitor::visit(TryBlockStatementAST* ast) {
  cxx_runtime_error("visit(TryBlockStatementAST): not implemented");
}

// DeclarationAST
void DefaultASTVisitor::visit(AccessDeclarationAST* ast) {
  cxx_runtime_error("visit(AccessDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(FunctionDefinitionAST* ast) {
  cxx_runtime_error("visit(FunctionDefinitionAST): not implemented");
}

void DefaultASTVisitor::visit(ConceptDefinitionAST* ast) {
  cxx_runtime_error("visit(ConceptDefinitionAST): not implemented");
}

void DefaultASTVisitor::visit(ForRangeDeclarationAST* ast) {
  cxx_runtime_error("visit(ForRangeDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(AliasDeclarationAST* ast) {
  cxx_runtime_error("visit(AliasDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(SimpleDeclarationAST* ast) {
  cxx_runtime_error("visit(SimpleDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(StructuredBindingDeclarationAST* ast) {
  cxx_runtime_error("visit(StructuredBindingDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(StaticAssertDeclarationAST* ast) {
  cxx_runtime_error("visit(StaticAssertDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(EmptyDeclarationAST* ast) {
  cxx_runtime_error("visit(EmptyDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(AttributeDeclarationAST* ast) {
  cxx_runtime_error("visit(AttributeDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(OpaqueEnumDeclarationAST* ast) {
  cxx_runtime_error("visit(OpaqueEnumDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(NestedNamespaceSpecifierAST* ast) {
  cxx_runtime_error("visit(NestedNamespaceSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(NamespaceDefinitionAST* ast) {
  cxx_runtime_error("visit(NamespaceDefinitionAST): not implemented");
}

void DefaultASTVisitor::visit(NamespaceAliasDefinitionAST* ast) {
  cxx_runtime_error("visit(NamespaceAliasDefinitionAST): not implemented");
}

void DefaultASTVisitor::visit(UsingDirectiveAST* ast) {
  cxx_runtime_error("visit(UsingDirectiveAST): not implemented");
}

void DefaultASTVisitor::visit(UsingDeclarationAST* ast) {
  cxx_runtime_error("visit(UsingDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(UsingEnumDeclarationAST* ast) {
  cxx_runtime_error("visit(UsingEnumDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(AsmDeclarationAST* ast) {
  cxx_runtime_error("visit(AsmDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(ExportDeclarationAST* ast) {
  cxx_runtime_error("visit(ExportDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(ExportCompoundDeclarationAST* ast) {
  cxx_runtime_error("visit(ExportCompoundDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(ModuleImportDeclarationAST* ast) {
  cxx_runtime_error("visit(ModuleImportDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(TemplateDeclarationAST* ast) {
  cxx_runtime_error("visit(TemplateDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(TypenameTypeParameterAST* ast) {
  cxx_runtime_error("visit(TypenameTypeParameterAST): not implemented");
}

void DefaultASTVisitor::visit(TemplateTypeParameterAST* ast) {
  cxx_runtime_error("visit(TemplateTypeParameterAST): not implemented");
}

void DefaultASTVisitor::visit(TemplatePackTypeParameterAST* ast) {
  cxx_runtime_error("visit(TemplatePackTypeParameterAST): not implemented");
}

void DefaultASTVisitor::visit(DeductionGuideAST* ast) {
  cxx_runtime_error("visit(DeductionGuideAST): not implemented");
}

void DefaultASTVisitor::visit(ExplicitInstantiationAST* ast) {
  cxx_runtime_error("visit(ExplicitInstantiationAST): not implemented");
}

void DefaultASTVisitor::visit(ParameterDeclarationAST* ast) {
  cxx_runtime_error("visit(ParameterDeclarationAST): not implemented");
}

void DefaultASTVisitor::visit(LinkageSpecificationAST* ast) {
  cxx_runtime_error("visit(LinkageSpecificationAST): not implemented");
}

// NameAST
void DefaultASTVisitor::visit(SimpleNameAST* ast) {
  cxx_runtime_error("visit(SimpleNameAST): not implemented");
}

void DefaultASTVisitor::visit(DestructorNameAST* ast) {
  cxx_runtime_error("visit(DestructorNameAST): not implemented");
}

void DefaultASTVisitor::visit(DecltypeNameAST* ast) {
  cxx_runtime_error("visit(DecltypeNameAST): not implemented");
}

void DefaultASTVisitor::visit(OperatorNameAST* ast) {
  cxx_runtime_error("visit(OperatorNameAST): not implemented");
}

void DefaultASTVisitor::visit(ConversionNameAST* ast) {
  cxx_runtime_error("visit(ConversionNameAST): not implemented");
}

void DefaultASTVisitor::visit(TemplateNameAST* ast) {
  cxx_runtime_error("visit(TemplateNameAST): not implemented");
}

void DefaultASTVisitor::visit(QualifiedNameAST* ast) {
  cxx_runtime_error("visit(QualifiedNameAST): not implemented");
}

// SpecifierAST
void DefaultASTVisitor::visit(TypedefSpecifierAST* ast) {
  cxx_runtime_error("visit(TypedefSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(FriendSpecifierAST* ast) {
  cxx_runtime_error("visit(FriendSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ConstevalSpecifierAST* ast) {
  cxx_runtime_error("visit(ConstevalSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ConstinitSpecifierAST* ast) {
  cxx_runtime_error("visit(ConstinitSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ConstexprSpecifierAST* ast) {
  cxx_runtime_error("visit(ConstexprSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(InlineSpecifierAST* ast) {
  cxx_runtime_error("visit(InlineSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(StaticSpecifierAST* ast) {
  cxx_runtime_error("visit(StaticSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ExternSpecifierAST* ast) {
  cxx_runtime_error("visit(ExternSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ThreadLocalSpecifierAST* ast) {
  cxx_runtime_error("visit(ThreadLocalSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ThreadSpecifierAST* ast) {
  cxx_runtime_error("visit(ThreadSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(MutableSpecifierAST* ast) {
  cxx_runtime_error("visit(MutableSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(VirtualSpecifierAST* ast) {
  cxx_runtime_error("visit(VirtualSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ExplicitSpecifierAST* ast) {
  cxx_runtime_error("visit(ExplicitSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(AutoTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(AutoTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(VoidTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(VoidTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(VaListTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(VaListTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(IntegralTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(IntegralTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(FloatingPointTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(FloatingPointTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ComplexTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(ComplexTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(NamedTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(NamedTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(AtomicTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(AtomicTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(UnderlyingTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(UnderlyingTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ElaboratedTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(ElaboratedTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(DecltypeAutoSpecifierAST* ast) {
  cxx_runtime_error("visit(DecltypeAutoSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(DecltypeSpecifierAST* ast) {
  cxx_runtime_error("visit(DecltypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(PlaceholderTypeSpecifierAST* ast) {
  cxx_runtime_error("visit(PlaceholderTypeSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ConstQualifierAST* ast) {
  cxx_runtime_error("visit(ConstQualifierAST): not implemented");
}

void DefaultASTVisitor::visit(VolatileQualifierAST* ast) {
  cxx_runtime_error("visit(VolatileQualifierAST): not implemented");
}

void DefaultASTVisitor::visit(RestrictQualifierAST* ast) {
  cxx_runtime_error("visit(RestrictQualifierAST): not implemented");
}

void DefaultASTVisitor::visit(EnumSpecifierAST* ast) {
  cxx_runtime_error("visit(EnumSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(ClassSpecifierAST* ast) {
  cxx_runtime_error("visit(ClassSpecifierAST): not implemented");
}

void DefaultASTVisitor::visit(TypenameSpecifierAST* ast) {
  cxx_runtime_error("visit(TypenameSpecifierAST): not implemented");
}

// CoreDeclaratorAST
void DefaultASTVisitor::visit(BitfieldDeclaratorAST* ast) {
  cxx_runtime_error("visit(BitfieldDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(ParameterPackAST* ast) {
  cxx_runtime_error("visit(ParameterPackAST): not implemented");
}

void DefaultASTVisitor::visit(IdDeclaratorAST* ast) {
  cxx_runtime_error("visit(IdDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(NestedDeclaratorAST* ast) {
  cxx_runtime_error("visit(NestedDeclaratorAST): not implemented");
}

// PtrOperatorAST
void DefaultASTVisitor::visit(PointerOperatorAST* ast) {
  cxx_runtime_error("visit(PointerOperatorAST): not implemented");
}

void DefaultASTVisitor::visit(ReferenceOperatorAST* ast) {
  cxx_runtime_error("visit(ReferenceOperatorAST): not implemented");
}

void DefaultASTVisitor::visit(PtrToMemberOperatorAST* ast) {
  cxx_runtime_error("visit(PtrToMemberOperatorAST): not implemented");
}

// DeclaratorModifierAST
void DefaultASTVisitor::visit(FunctionDeclaratorAST* ast) {
  cxx_runtime_error("visit(FunctionDeclaratorAST): not implemented");
}

void DefaultASTVisitor::visit(ArrayDeclaratorAST* ast) {
  cxx_runtime_error("visit(ArrayDeclaratorAST): not implemented");
}

// AttributeSpecifierAST
void DefaultASTVisitor::visit(CxxAttributeAST* ast) {
  cxx_runtime_error("visit(CxxAttributeAST): not implemented");
}

void DefaultASTVisitor::visit(GCCAttributeAST* ast) {
  cxx_runtime_error("visit(GCCAttributeAST): not implemented");
}

void DefaultASTVisitor::visit(AlignasAttributeAST* ast) {
  cxx_runtime_error("visit(AlignasAttributeAST): not implemented");
}

void DefaultASTVisitor::visit(AsmAttributeAST* ast) {
  cxx_runtime_error("visit(AsmAttributeAST): not implemented");
}

// AttributeTokenAST
void DefaultASTVisitor::visit(ScopedAttributeTokenAST* ast) {
  cxx_runtime_error("visit(ScopedAttributeTokenAST): not implemented");
}

void DefaultASTVisitor::visit(SimpleAttributeTokenAST* ast) {
  cxx_runtime_error("visit(SimpleAttributeTokenAST): not implemented");
}

}  // namespace cxx
