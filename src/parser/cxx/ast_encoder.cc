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

#include "ast_encoder.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/translation_unit.h>

#include <algorithm>

namespace cxx {

void ASTEncoder::accept(AST* ast) {
  if (!ast) return;
  ast->accept(this);
}

void ASTEncoder::visit(TypeIdAST* ast) {}

void ASTEncoder::visit(NestedNameSpecifierAST* ast) {}

void ASTEncoder::visit(UsingDeclaratorAST* ast) {}

void ASTEncoder::visit(HandlerAST* ast) {}

void ASTEncoder::visit(EnumBaseAST* ast) {}

void ASTEncoder::visit(EnumeratorAST* ast) {}

void ASTEncoder::visit(DeclaratorAST* ast) {}

void ASTEncoder::visit(InitDeclaratorAST* ast) {}

void ASTEncoder::visit(BaseSpecifierAST* ast) {}

void ASTEncoder::visit(BaseClauseAST* ast) {}

void ASTEncoder::visit(NewTypeIdAST* ast) {}

void ASTEncoder::visit(RequiresClauseAST* ast) {}

void ASTEncoder::visit(ParameterDeclarationClauseAST* ast) {}

void ASTEncoder::visit(ParametersAndQualifiersAST* ast) {}

void ASTEncoder::visit(LambdaIntroducerAST* ast) {}

void ASTEncoder::visit(LambdaDeclaratorAST* ast) {}

void ASTEncoder::visit(TrailingReturnTypeAST* ast) {}

void ASTEncoder::visit(CtorInitializerAST* ast) {}

void ASTEncoder::visit(RequirementBodyAST* ast) {}

void ASTEncoder::visit(TypeConstraintAST* ast) {}

void ASTEncoder::visit(GlobalModuleFragmentAST* ast) {}

void ASTEncoder::visit(PrivateModuleFragmentAST* ast) {}

void ASTEncoder::visit(ModuleDeclarationAST* ast) {}

void ASTEncoder::visit(ModuleNameAST* ast) {}

void ASTEncoder::visit(ImportNameAST* ast) {}

void ASTEncoder::visit(ModulePartitionAST* ast) {}

void ASTEncoder::visit(SimpleRequirementAST* ast) {}

void ASTEncoder::visit(CompoundRequirementAST* ast) {}

void ASTEncoder::visit(TypeRequirementAST* ast) {}

void ASTEncoder::visit(NestedRequirementAST* ast) {}

void ASTEncoder::visit(TypeTemplateArgumentAST* ast) {}

void ASTEncoder::visit(ExpressionTemplateArgumentAST* ast) {}

void ASTEncoder::visit(ParenMemInitializerAST* ast) {}

void ASTEncoder::visit(BracedMemInitializerAST* ast) {}

void ASTEncoder::visit(ThisLambdaCaptureAST* ast) {}

void ASTEncoder::visit(DerefThisLambdaCaptureAST* ast) {}

void ASTEncoder::visit(SimpleLambdaCaptureAST* ast) {}

void ASTEncoder::visit(RefLambdaCaptureAST* ast) {}

void ASTEncoder::visit(RefInitLambdaCaptureAST* ast) {}

void ASTEncoder::visit(InitLambdaCaptureAST* ast) {}

void ASTEncoder::visit(EqualInitializerAST* ast) {}

void ASTEncoder::visit(BracedInitListAST* ast) {}

void ASTEncoder::visit(ParenInitializerAST* ast) {}

void ASTEncoder::visit(NewParenInitializerAST* ast) {}

void ASTEncoder::visit(NewBracedInitializerAST* ast) {}

void ASTEncoder::visit(EllipsisExceptionDeclarationAST* ast) {}

void ASTEncoder::visit(TypeExceptionDeclarationAST* ast) {}

void ASTEncoder::visit(DefaultFunctionBodyAST* ast) {}

void ASTEncoder::visit(CompoundStatementFunctionBodyAST* ast) {}

void ASTEncoder::visit(TryStatementFunctionBodyAST* ast) {}

void ASTEncoder::visit(DeleteFunctionBodyAST* ast) {}

void ASTEncoder::visit(TranslationUnitAST* ast) {}

void ASTEncoder::visit(ModuleUnitAST* ast) {}

void ASTEncoder::visit(ThisExpressionAST* ast) {}

void ASTEncoder::visit(CharLiteralExpressionAST* ast) {}

void ASTEncoder::visit(BoolLiteralExpressionAST* ast) {}

void ASTEncoder::visit(IntLiteralExpressionAST* ast) {}

void ASTEncoder::visit(FloatLiteralExpressionAST* ast) {}

void ASTEncoder::visit(NullptrLiteralExpressionAST* ast) {}

void ASTEncoder::visit(StringLiteralExpressionAST* ast) {}

void ASTEncoder::visit(UserDefinedStringLiteralExpressionAST* ast) {}

void ASTEncoder::visit(IdExpressionAST* ast) {}

void ASTEncoder::visit(RequiresExpressionAST* ast) {}

void ASTEncoder::visit(NestedExpressionAST* ast) {}

void ASTEncoder::visit(RightFoldExpressionAST* ast) {}

void ASTEncoder::visit(LeftFoldExpressionAST* ast) {}

void ASTEncoder::visit(FoldExpressionAST* ast) {}

void ASTEncoder::visit(LambdaExpressionAST* ast) {}

void ASTEncoder::visit(SizeofExpressionAST* ast) {}

void ASTEncoder::visit(SizeofTypeExpressionAST* ast) {}

void ASTEncoder::visit(SizeofPackExpressionAST* ast) {}

void ASTEncoder::visit(TypeidExpressionAST* ast) {}

void ASTEncoder::visit(TypeidOfTypeExpressionAST* ast) {}

void ASTEncoder::visit(AlignofExpressionAST* ast) {}

void ASTEncoder::visit(TypeTraitsExpressionAST* ast) {}

void ASTEncoder::visit(UnaryExpressionAST* ast) {}

void ASTEncoder::visit(BinaryExpressionAST* ast) {}

void ASTEncoder::visit(AssignmentExpressionAST* ast) {}

void ASTEncoder::visit(BracedTypeConstructionAST* ast) {}

void ASTEncoder::visit(TypeConstructionAST* ast) {}

void ASTEncoder::visit(CallExpressionAST* ast) {}

void ASTEncoder::visit(SubscriptExpressionAST* ast) {}

void ASTEncoder::visit(MemberExpressionAST* ast) {}

void ASTEncoder::visit(PostIncrExpressionAST* ast) {}

void ASTEncoder::visit(ConditionalExpressionAST* ast) {}

void ASTEncoder::visit(ImplicitCastExpressionAST* ast) {}

void ASTEncoder::visit(CastExpressionAST* ast) {}

void ASTEncoder::visit(CppCastExpressionAST* ast) {}

void ASTEncoder::visit(NewExpressionAST* ast) {}

void ASTEncoder::visit(DeleteExpressionAST* ast) {}

void ASTEncoder::visit(ThrowExpressionAST* ast) {}

void ASTEncoder::visit(NoexceptExpressionAST* ast) {}

void ASTEncoder::visit(LabeledStatementAST* ast) {}

void ASTEncoder::visit(CaseStatementAST* ast) {}

void ASTEncoder::visit(DefaultStatementAST* ast) {}

void ASTEncoder::visit(ExpressionStatementAST* ast) {}

void ASTEncoder::visit(CompoundStatementAST* ast) {}

void ASTEncoder::visit(IfStatementAST* ast) {}

void ASTEncoder::visit(SwitchStatementAST* ast) {}

void ASTEncoder::visit(WhileStatementAST* ast) {}

void ASTEncoder::visit(DoStatementAST* ast) {}

void ASTEncoder::visit(ForRangeStatementAST* ast) {}

void ASTEncoder::visit(ForStatementAST* ast) {}

void ASTEncoder::visit(BreakStatementAST* ast) {}

void ASTEncoder::visit(ContinueStatementAST* ast) {}

void ASTEncoder::visit(ReturnStatementAST* ast) {}

void ASTEncoder::visit(GotoStatementAST* ast) {}

void ASTEncoder::visit(CoroutineReturnStatementAST* ast) {}

void ASTEncoder::visit(DeclarationStatementAST* ast) {}

void ASTEncoder::visit(TryBlockStatementAST* ast) {}

void ASTEncoder::visit(AccessDeclarationAST* ast) {}

void ASTEncoder::visit(FunctionDefinitionAST* ast) {}

void ASTEncoder::visit(ConceptDefinitionAST* ast) {}

void ASTEncoder::visit(ForRangeDeclarationAST* ast) {}

void ASTEncoder::visit(AliasDeclarationAST* ast) {}

void ASTEncoder::visit(SimpleDeclarationAST* ast) {}

void ASTEncoder::visit(StaticAssertDeclarationAST* ast) {}

void ASTEncoder::visit(EmptyDeclarationAST* ast) {}

void ASTEncoder::visit(AttributeDeclarationAST* ast) {}

void ASTEncoder::visit(OpaqueEnumDeclarationAST* ast) {}

void ASTEncoder::visit(UsingEnumDeclarationAST* ast) {}

void ASTEncoder::visit(NamespaceDefinitionAST* ast) {}

void ASTEncoder::visit(NamespaceAliasDefinitionAST* ast) {}

void ASTEncoder::visit(UsingDirectiveAST* ast) {}

void ASTEncoder::visit(UsingDeclarationAST* ast) {}

void ASTEncoder::visit(AsmDeclarationAST* ast) {}

void ASTEncoder::visit(ExportDeclarationAST* ast) {}

void ASTEncoder::visit(ExportCompoundDeclarationAST* ast) {}

void ASTEncoder::visit(ModuleImportDeclarationAST* ast) {}

void ASTEncoder::visit(TemplateDeclarationAST* ast) {}

void ASTEncoder::visit(TypenameTypeParameterAST* ast) {}

void ASTEncoder::visit(TemplateTypeParameterAST* ast) {}

void ASTEncoder::visit(TemplatePackTypeParameterAST* ast) {}

void ASTEncoder::visit(DeductionGuideAST* ast) {}

void ASTEncoder::visit(ExplicitInstantiationAST* ast) {}

void ASTEncoder::visit(ParameterDeclarationAST* ast) {}

void ASTEncoder::visit(LinkageSpecificationAST* ast) {}

void ASTEncoder::visit(SimpleNameAST* ast) {}

void ASTEncoder::visit(DestructorNameAST* ast) {}

void ASTEncoder::visit(DecltypeNameAST* ast) {}

void ASTEncoder::visit(OperatorNameAST* ast) {}

void ASTEncoder::visit(ConversionNameAST* ast) {}

void ASTEncoder::visit(TemplateNameAST* ast) {}

void ASTEncoder::visit(QualifiedNameAST* ast) {}

void ASTEncoder::visit(TypedefSpecifierAST* ast) {}

void ASTEncoder::visit(FriendSpecifierAST* ast) {}

void ASTEncoder::visit(ConstevalSpecifierAST* ast) {}

void ASTEncoder::visit(ConstinitSpecifierAST* ast) {}

void ASTEncoder::visit(ConstexprSpecifierAST* ast) {}

void ASTEncoder::visit(InlineSpecifierAST* ast) {}

void ASTEncoder::visit(StaticSpecifierAST* ast) {}

void ASTEncoder::visit(ExternSpecifierAST* ast) {}

void ASTEncoder::visit(ThreadLocalSpecifierAST* ast) {}

void ASTEncoder::visit(ThreadSpecifierAST* ast) {}

void ASTEncoder::visit(MutableSpecifierAST* ast) {}

void ASTEncoder::visit(VirtualSpecifierAST* ast) {}

void ASTEncoder::visit(ExplicitSpecifierAST* ast) {}

void ASTEncoder::visit(AutoTypeSpecifierAST* ast) {}

void ASTEncoder::visit(VoidTypeSpecifierAST* ast) {}

void ASTEncoder::visit(VaListTypeSpecifierAST* ast) {}

void ASTEncoder::visit(IntegralTypeSpecifierAST* ast) {}

void ASTEncoder::visit(FloatingPointTypeSpecifierAST* ast) {}

void ASTEncoder::visit(ComplexTypeSpecifierAST* ast) {}

void ASTEncoder::visit(NamedTypeSpecifierAST* ast) {}

void ASTEncoder::visit(AtomicTypeSpecifierAST* ast) {}

void ASTEncoder::visit(UnderlyingTypeSpecifierAST* ast) {}

void ASTEncoder::visit(ElaboratedTypeSpecifierAST* ast) {}

void ASTEncoder::visit(DecltypeAutoSpecifierAST* ast) {}

void ASTEncoder::visit(DecltypeSpecifierAST* ast) {}

void ASTEncoder::visit(PlaceholderTypeSpecifierAST* ast) {}

void ASTEncoder::visit(ConstQualifierAST* ast) {}

void ASTEncoder::visit(VolatileQualifierAST* ast) {}

void ASTEncoder::visit(RestrictQualifierAST* ast) {}

void ASTEncoder::visit(EnumSpecifierAST* ast) {}

void ASTEncoder::visit(ClassSpecifierAST* ast) {}

void ASTEncoder::visit(TypenameSpecifierAST* ast) {}

void ASTEncoder::visit(IdDeclaratorAST* ast) {}

void ASTEncoder::visit(NestedDeclaratorAST* ast) {}

void ASTEncoder::visit(PointerOperatorAST* ast) {}

void ASTEncoder::visit(ReferenceOperatorAST* ast) {}

void ASTEncoder::visit(PtrToMemberOperatorAST* ast) {}

void ASTEncoder::visit(FunctionDeclaratorAST* ast) {}

void ASTEncoder::visit(ArrayDeclaratorAST* ast) {}

}  // namespace cxx
