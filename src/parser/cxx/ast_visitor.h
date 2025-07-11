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

namespace cxx {

class ASTVisitor {
 public:
  virtual ~ASTVisitor() = default;

  void accept(AST* ast);

  [[nodiscard]] virtual bool preVisit(AST* ast);
  virtual void postVisit(AST* ast);

  // UnitAST
  virtual void visit(TranslationUnitAST* ast);
  virtual void visit(ModuleUnitAST* ast);

  // DeclarationAST
  virtual void visit(SimpleDeclarationAST* ast);
  virtual void visit(AsmDeclarationAST* ast);
  virtual void visit(NamespaceAliasDefinitionAST* ast);
  virtual void visit(UsingDeclarationAST* ast);
  virtual void visit(UsingEnumDeclarationAST* ast);
  virtual void visit(UsingDirectiveAST* ast);
  virtual void visit(StaticAssertDeclarationAST* ast);
  virtual void visit(AliasDeclarationAST* ast);
  virtual void visit(OpaqueEnumDeclarationAST* ast);
  virtual void visit(FunctionDefinitionAST* ast);
  virtual void visit(TemplateDeclarationAST* ast);
  virtual void visit(ConceptDefinitionAST* ast);
  virtual void visit(DeductionGuideAST* ast);
  virtual void visit(ExplicitInstantiationAST* ast);
  virtual void visit(ExportDeclarationAST* ast);
  virtual void visit(ExportCompoundDeclarationAST* ast);
  virtual void visit(LinkageSpecificationAST* ast);
  virtual void visit(NamespaceDefinitionAST* ast);
  virtual void visit(EmptyDeclarationAST* ast);
  virtual void visit(AttributeDeclarationAST* ast);
  virtual void visit(ModuleImportDeclarationAST* ast);
  virtual void visit(ParameterDeclarationAST* ast);
  virtual void visit(AccessDeclarationAST* ast);
  virtual void visit(ForRangeDeclarationAST* ast);
  virtual void visit(StructuredBindingDeclarationAST* ast);
  virtual void visit(AsmOperandAST* ast);
  virtual void visit(AsmQualifierAST* ast);
  virtual void visit(AsmClobberAST* ast);
  virtual void visit(AsmGotoLabelAST* ast);

  // StatementAST
  virtual void visit(LabeledStatementAST* ast);
  virtual void visit(CaseStatementAST* ast);
  virtual void visit(DefaultStatementAST* ast);
  virtual void visit(ExpressionStatementAST* ast);
  virtual void visit(CompoundStatementAST* ast);
  virtual void visit(IfStatementAST* ast);
  virtual void visit(ConstevalIfStatementAST* ast);
  virtual void visit(SwitchStatementAST* ast);
  virtual void visit(WhileStatementAST* ast);
  virtual void visit(DoStatementAST* ast);
  virtual void visit(ForRangeStatementAST* ast);
  virtual void visit(ForStatementAST* ast);
  virtual void visit(BreakStatementAST* ast);
  virtual void visit(ContinueStatementAST* ast);
  virtual void visit(ReturnStatementAST* ast);
  virtual void visit(CoroutineReturnStatementAST* ast);
  virtual void visit(GotoStatementAST* ast);
  virtual void visit(DeclarationStatementAST* ast);
  virtual void visit(TryBlockStatementAST* ast);

  // ExpressionAST
  virtual void visit(GeneratedLiteralExpressionAST* ast);
  virtual void visit(CharLiteralExpressionAST* ast);
  virtual void visit(BoolLiteralExpressionAST* ast);
  virtual void visit(IntLiteralExpressionAST* ast);
  virtual void visit(FloatLiteralExpressionAST* ast);
  virtual void visit(NullptrLiteralExpressionAST* ast);
  virtual void visit(StringLiteralExpressionAST* ast);
  virtual void visit(UserDefinedStringLiteralExpressionAST* ast);
  virtual void visit(ObjectLiteralExpressionAST* ast);
  virtual void visit(ThisExpressionAST* ast);
  virtual void visit(NestedStatementExpressionAST* ast);
  virtual void visit(NestedExpressionAST* ast);
  virtual void visit(IdExpressionAST* ast);
  virtual void visit(LambdaExpressionAST* ast);
  virtual void visit(FoldExpressionAST* ast);
  virtual void visit(RightFoldExpressionAST* ast);
  virtual void visit(LeftFoldExpressionAST* ast);
  virtual void visit(RequiresExpressionAST* ast);
  virtual void visit(VaArgExpressionAST* ast);
  virtual void visit(SubscriptExpressionAST* ast);
  virtual void visit(CallExpressionAST* ast);
  virtual void visit(TypeConstructionAST* ast);
  virtual void visit(BracedTypeConstructionAST* ast);
  virtual void visit(SpliceMemberExpressionAST* ast);
  virtual void visit(MemberExpressionAST* ast);
  virtual void visit(PostIncrExpressionAST* ast);
  virtual void visit(CppCastExpressionAST* ast);
  virtual void visit(BuiltinBitCastExpressionAST* ast);
  virtual void visit(BuiltinOffsetofExpressionAST* ast);
  virtual void visit(TypeidExpressionAST* ast);
  virtual void visit(TypeidOfTypeExpressionAST* ast);
  virtual void visit(SpliceExpressionAST* ast);
  virtual void visit(GlobalScopeReflectExpressionAST* ast);
  virtual void visit(NamespaceReflectExpressionAST* ast);
  virtual void visit(TypeIdReflectExpressionAST* ast);
  virtual void visit(ReflectExpressionAST* ast);
  virtual void visit(UnaryExpressionAST* ast);
  virtual void visit(AwaitExpressionAST* ast);
  virtual void visit(SizeofExpressionAST* ast);
  virtual void visit(SizeofTypeExpressionAST* ast);
  virtual void visit(SizeofPackExpressionAST* ast);
  virtual void visit(AlignofTypeExpressionAST* ast);
  virtual void visit(AlignofExpressionAST* ast);
  virtual void visit(NoexceptExpressionAST* ast);
  virtual void visit(NewExpressionAST* ast);
  virtual void visit(DeleteExpressionAST* ast);
  virtual void visit(CastExpressionAST* ast);
  virtual void visit(ImplicitCastExpressionAST* ast);
  virtual void visit(BinaryExpressionAST* ast);
  virtual void visit(ConditionalExpressionAST* ast);
  virtual void visit(YieldExpressionAST* ast);
  virtual void visit(ThrowExpressionAST* ast);
  virtual void visit(AssignmentExpressionAST* ast);
  virtual void visit(PackExpansionExpressionAST* ast);
  virtual void visit(DesignatedInitializerClauseAST* ast);
  virtual void visit(TypeTraitExpressionAST* ast);
  virtual void visit(ConditionExpressionAST* ast);
  virtual void visit(EqualInitializerAST* ast);
  virtual void visit(BracedInitListAST* ast);
  virtual void visit(ParenInitializerAST* ast);

  // DesignatorAST
  virtual void visit(DotDesignatorAST* ast);
  virtual void visit(SubscriptDesignatorAST* ast);

  // AST
  virtual void visit(SplicerAST* ast);
  virtual void visit(GlobalModuleFragmentAST* ast);
  virtual void visit(PrivateModuleFragmentAST* ast);
  virtual void visit(ModuleDeclarationAST* ast);
  virtual void visit(ModuleNameAST* ast);
  virtual void visit(ModuleQualifierAST* ast);
  virtual void visit(ModulePartitionAST* ast);
  virtual void visit(ImportNameAST* ast);
  virtual void visit(InitDeclaratorAST* ast);
  virtual void visit(DeclaratorAST* ast);
  virtual void visit(UsingDeclaratorAST* ast);
  virtual void visit(EnumeratorAST* ast);
  virtual void visit(TypeIdAST* ast);
  virtual void visit(HandlerAST* ast);
  virtual void visit(BaseSpecifierAST* ast);
  virtual void visit(RequiresClauseAST* ast);
  virtual void visit(ParameterDeclarationClauseAST* ast);
  virtual void visit(TrailingReturnTypeAST* ast);
  virtual void visit(LambdaSpecifierAST* ast);
  virtual void visit(TypeConstraintAST* ast);
  virtual void visit(AttributeArgumentClauseAST* ast);
  virtual void visit(AttributeAST* ast);
  virtual void visit(AttributeUsingPrefixAST* ast);
  virtual void visit(NewPlacementAST* ast);
  virtual void visit(NestedNamespaceSpecifierAST* ast);

  // TemplateParameterAST
  virtual void visit(TemplateTypeParameterAST* ast);
  virtual void visit(NonTypeTemplateParameterAST* ast);
  virtual void visit(TypenameTypeParameterAST* ast);
  virtual void visit(ConstraintTypeParameterAST* ast);

  // SpecifierAST
  virtual void visit(GeneratedTypeSpecifierAST* ast);
  virtual void visit(TypedefSpecifierAST* ast);
  virtual void visit(FriendSpecifierAST* ast);
  virtual void visit(ConstevalSpecifierAST* ast);
  virtual void visit(ConstinitSpecifierAST* ast);
  virtual void visit(ConstexprSpecifierAST* ast);
  virtual void visit(InlineSpecifierAST* ast);
  virtual void visit(NoreturnSpecifierAST* ast);
  virtual void visit(StaticSpecifierAST* ast);
  virtual void visit(ExternSpecifierAST* ast);
  virtual void visit(RegisterSpecifierAST* ast);
  virtual void visit(ThreadLocalSpecifierAST* ast);
  virtual void visit(ThreadSpecifierAST* ast);
  virtual void visit(MutableSpecifierAST* ast);
  virtual void visit(VirtualSpecifierAST* ast);
  virtual void visit(ExplicitSpecifierAST* ast);
  virtual void visit(AutoTypeSpecifierAST* ast);
  virtual void visit(VoidTypeSpecifierAST* ast);
  virtual void visit(SizeTypeSpecifierAST* ast);
  virtual void visit(SignTypeSpecifierAST* ast);
  virtual void visit(VaListTypeSpecifierAST* ast);
  virtual void visit(IntegralTypeSpecifierAST* ast);
  virtual void visit(FloatingPointTypeSpecifierAST* ast);
  virtual void visit(ComplexTypeSpecifierAST* ast);
  virtual void visit(NamedTypeSpecifierAST* ast);
  virtual void visit(AtomicTypeSpecifierAST* ast);
  virtual void visit(UnderlyingTypeSpecifierAST* ast);
  virtual void visit(ElaboratedTypeSpecifierAST* ast);
  virtual void visit(DecltypeAutoSpecifierAST* ast);
  virtual void visit(DecltypeSpecifierAST* ast);
  virtual void visit(PlaceholderTypeSpecifierAST* ast);
  virtual void visit(ConstQualifierAST* ast);
  virtual void visit(VolatileQualifierAST* ast);
  virtual void visit(RestrictQualifierAST* ast);
  virtual void visit(EnumSpecifierAST* ast);
  virtual void visit(ClassSpecifierAST* ast);
  virtual void visit(TypenameSpecifierAST* ast);
  virtual void visit(SplicerTypeSpecifierAST* ast);

  // PtrOperatorAST
  virtual void visit(PointerOperatorAST* ast);
  virtual void visit(ReferenceOperatorAST* ast);
  virtual void visit(PtrToMemberOperatorAST* ast);

  // CoreDeclaratorAST
  virtual void visit(BitfieldDeclaratorAST* ast);
  virtual void visit(ParameterPackAST* ast);
  virtual void visit(IdDeclaratorAST* ast);
  virtual void visit(NestedDeclaratorAST* ast);

  // DeclaratorChunkAST
  virtual void visit(FunctionDeclaratorChunkAST* ast);
  virtual void visit(ArrayDeclaratorChunkAST* ast);

  // UnqualifiedIdAST
  virtual void visit(NameIdAST* ast);
  virtual void visit(DestructorIdAST* ast);
  virtual void visit(DecltypeIdAST* ast);
  virtual void visit(OperatorFunctionIdAST* ast);
  virtual void visit(LiteralOperatorIdAST* ast);
  virtual void visit(ConversionFunctionIdAST* ast);
  virtual void visit(SimpleTemplateIdAST* ast);
  virtual void visit(LiteralOperatorTemplateIdAST* ast);
  virtual void visit(OperatorFunctionTemplateIdAST* ast);

  // NestedNameSpecifierAST
  virtual void visit(GlobalNestedNameSpecifierAST* ast);
  virtual void visit(SimpleNestedNameSpecifierAST* ast);
  virtual void visit(DecltypeNestedNameSpecifierAST* ast);
  virtual void visit(TemplateNestedNameSpecifierAST* ast);

  // FunctionBodyAST
  virtual void visit(DefaultFunctionBodyAST* ast);
  virtual void visit(CompoundStatementFunctionBodyAST* ast);
  virtual void visit(TryStatementFunctionBodyAST* ast);
  virtual void visit(DeleteFunctionBodyAST* ast);

  // TemplateArgumentAST
  virtual void visit(TypeTemplateArgumentAST* ast);
  virtual void visit(ExpressionTemplateArgumentAST* ast);

  // ExceptionSpecifierAST
  virtual void visit(ThrowExceptionSpecifierAST* ast);
  virtual void visit(NoexceptSpecifierAST* ast);

  // RequirementAST
  virtual void visit(SimpleRequirementAST* ast);
  virtual void visit(CompoundRequirementAST* ast);
  virtual void visit(TypeRequirementAST* ast);
  virtual void visit(NestedRequirementAST* ast);

  // NewInitializerAST
  virtual void visit(NewParenInitializerAST* ast);
  virtual void visit(NewBracedInitializerAST* ast);

  // MemInitializerAST
  virtual void visit(ParenMemInitializerAST* ast);
  virtual void visit(BracedMemInitializerAST* ast);

  // LambdaCaptureAST
  virtual void visit(ThisLambdaCaptureAST* ast);
  virtual void visit(DerefThisLambdaCaptureAST* ast);
  virtual void visit(SimpleLambdaCaptureAST* ast);
  virtual void visit(RefLambdaCaptureAST* ast);
  virtual void visit(RefInitLambdaCaptureAST* ast);
  virtual void visit(InitLambdaCaptureAST* ast);

  // ExceptionDeclarationAST
  virtual void visit(EllipsisExceptionDeclarationAST* ast);
  virtual void visit(TypeExceptionDeclarationAST* ast);

  // AttributeSpecifierAST
  virtual void visit(CxxAttributeAST* ast);
  virtual void visit(GccAttributeAST* ast);
  virtual void visit(AlignasAttributeAST* ast);
  virtual void visit(AlignasTypeAttributeAST* ast);
  virtual void visit(AsmAttributeAST* ast);

  // AttributeTokenAST
  virtual void visit(ScopedAttributeTokenAST* ast);
  virtual void visit(SimpleAttributeTokenAST* ast);
};

}  // namespace cxx
