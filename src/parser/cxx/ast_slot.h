// Generated file by: gen_ast_slot_h.ts
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

#include <cxx/ast_visitor.h>

#include <cstdint>
#include <tuple>

namespace cxx {

enum ASTSlotKind {
  kInvalid,
  kToken,
  kNode,
  kTokenList,
  kNodeList,
  kIdentifierAttribute,
  kLiteralAttribute,
  kBoolAttribute,
  kIntAttribute,
};

enum class SlotNameIndex : int {};

class ASTSlot final : ASTVisitor {
 public:
  struct SlotInfo {
    std::intptr_t handle;
    ASTSlotKind kind;
    SlotNameIndex nameIndex;
    int slotCount;
  };

  auto operator()(AST* ast, int slot) -> SlotInfo;

 private:
  void visit(TranslationUnitAST* ast) override;
  void visit(ModuleUnitAST* ast) override;

  void visit(SimpleDeclarationAST* ast) override;
  void visit(AsmDeclarationAST* ast) override;
  void visit(NamespaceAliasDefinitionAST* ast) override;
  void visit(UsingDeclarationAST* ast) override;
  void visit(UsingEnumDeclarationAST* ast) override;
  void visit(UsingDirectiveAST* ast) override;
  void visit(StaticAssertDeclarationAST* ast) override;
  void visit(AliasDeclarationAST* ast) override;
  void visit(OpaqueEnumDeclarationAST* ast) override;
  void visit(FunctionDefinitionAST* ast) override;
  void visit(TemplateDeclarationAST* ast) override;
  void visit(ConceptDefinitionAST* ast) override;
  void visit(DeductionGuideAST* ast) override;
  void visit(ExplicitInstantiationAST* ast) override;
  void visit(ExportDeclarationAST* ast) override;
  void visit(ExportCompoundDeclarationAST* ast) override;
  void visit(LinkageSpecificationAST* ast) override;
  void visit(NamespaceDefinitionAST* ast) override;
  void visit(EmptyDeclarationAST* ast) override;
  void visit(AttributeDeclarationAST* ast) override;
  void visit(ModuleImportDeclarationAST* ast) override;
  void visit(ParameterDeclarationAST* ast) override;
  void visit(AccessDeclarationAST* ast) override;
  void visit(ForRangeDeclarationAST* ast) override;
  void visit(StructuredBindingDeclarationAST* ast) override;

  void visit(AsmOperandAST* ast) override;
  void visit(AsmQualifierAST* ast) override;
  void visit(AsmClobberAST* ast) override;
  void visit(AsmGotoLabelAST* ast) override;
  void visit(SplicerAST* ast) override;
  void visit(GlobalModuleFragmentAST* ast) override;
  void visit(PrivateModuleFragmentAST* ast) override;
  void visit(ModuleDeclarationAST* ast) override;
  void visit(ModuleNameAST* ast) override;
  void visit(ModuleQualifierAST* ast) override;
  void visit(ModulePartitionAST* ast) override;
  void visit(ImportNameAST* ast) override;
  void visit(InitDeclaratorAST* ast) override;
  void visit(DeclaratorAST* ast) override;
  void visit(UsingDeclaratorAST* ast) override;
  void visit(EnumeratorAST* ast) override;
  void visit(TypeIdAST* ast) override;
  void visit(HandlerAST* ast) override;
  void visit(BaseSpecifierAST* ast) override;
  void visit(RequiresClauseAST* ast) override;
  void visit(ParameterDeclarationClauseAST* ast) override;
  void visit(TrailingReturnTypeAST* ast) override;
  void visit(LambdaSpecifierAST* ast) override;
  void visit(TypeConstraintAST* ast) override;
  void visit(AttributeArgumentClauseAST* ast) override;
  void visit(AttributeAST* ast) override;
  void visit(AttributeUsingPrefixAST* ast) override;
  void visit(NewPlacementAST* ast) override;
  void visit(NestedNamespaceSpecifierAST* ast) override;

  void visit(LabeledStatementAST* ast) override;
  void visit(CaseStatementAST* ast) override;
  void visit(DefaultStatementAST* ast) override;
  void visit(ExpressionStatementAST* ast) override;
  void visit(CompoundStatementAST* ast) override;
  void visit(IfStatementAST* ast) override;
  void visit(ConstevalIfStatementAST* ast) override;
  void visit(SwitchStatementAST* ast) override;
  void visit(WhileStatementAST* ast) override;
  void visit(DoStatementAST* ast) override;
  void visit(ForRangeStatementAST* ast) override;
  void visit(ForStatementAST* ast) override;
  void visit(BreakStatementAST* ast) override;
  void visit(ContinueStatementAST* ast) override;
  void visit(ReturnStatementAST* ast) override;
  void visit(CoroutineReturnStatementAST* ast) override;
  void visit(GotoStatementAST* ast) override;
  void visit(DeclarationStatementAST* ast) override;
  void visit(TryBlockStatementAST* ast) override;

  void visit(GeneratedLiteralExpressionAST* ast) override;
  void visit(CharLiteralExpressionAST* ast) override;
  void visit(BoolLiteralExpressionAST* ast) override;
  void visit(IntLiteralExpressionAST* ast) override;
  void visit(FloatLiteralExpressionAST* ast) override;
  void visit(NullptrLiteralExpressionAST* ast) override;
  void visit(StringLiteralExpressionAST* ast) override;
  void visit(UserDefinedStringLiteralExpressionAST* ast) override;
  void visit(ObjectLiteralExpressionAST* ast) override;
  void visit(ThisExpressionAST* ast) override;
  void visit(GenericSelectionExpressionAST* ast) override;
  void visit(NestedStatementExpressionAST* ast) override;
  void visit(NestedExpressionAST* ast) override;
  void visit(IdExpressionAST* ast) override;
  void visit(LambdaExpressionAST* ast) override;
  void visit(FoldExpressionAST* ast) override;
  void visit(RightFoldExpressionAST* ast) override;
  void visit(LeftFoldExpressionAST* ast) override;
  void visit(RequiresExpressionAST* ast) override;
  void visit(VaArgExpressionAST* ast) override;
  void visit(SubscriptExpressionAST* ast) override;
  void visit(CallExpressionAST* ast) override;
  void visit(TypeConstructionAST* ast) override;
  void visit(BracedTypeConstructionAST* ast) override;
  void visit(SpliceMemberExpressionAST* ast) override;
  void visit(MemberExpressionAST* ast) override;
  void visit(PostIncrExpressionAST* ast) override;
  void visit(CppCastExpressionAST* ast) override;
  void visit(BuiltinBitCastExpressionAST* ast) override;
  void visit(BuiltinOffsetofExpressionAST* ast) override;
  void visit(TypeidExpressionAST* ast) override;
  void visit(TypeidOfTypeExpressionAST* ast) override;
  void visit(SpliceExpressionAST* ast) override;
  void visit(GlobalScopeReflectExpressionAST* ast) override;
  void visit(NamespaceReflectExpressionAST* ast) override;
  void visit(TypeIdReflectExpressionAST* ast) override;
  void visit(ReflectExpressionAST* ast) override;
  void visit(LabelAddressExpressionAST* ast) override;
  void visit(UnaryExpressionAST* ast) override;
  void visit(AwaitExpressionAST* ast) override;
  void visit(SizeofExpressionAST* ast) override;
  void visit(SizeofTypeExpressionAST* ast) override;
  void visit(SizeofPackExpressionAST* ast) override;
  void visit(AlignofTypeExpressionAST* ast) override;
  void visit(AlignofExpressionAST* ast) override;
  void visit(NoexceptExpressionAST* ast) override;
  void visit(NewExpressionAST* ast) override;
  void visit(DeleteExpressionAST* ast) override;
  void visit(CastExpressionAST* ast) override;
  void visit(ImplicitCastExpressionAST* ast) override;
  void visit(BinaryExpressionAST* ast) override;
  void visit(ConditionalExpressionAST* ast) override;
  void visit(YieldExpressionAST* ast) override;
  void visit(ThrowExpressionAST* ast) override;
  void visit(AssignmentExpressionAST* ast) override;
  void visit(CompoundAssignmentExpressionAST* ast) override;
  void visit(PackExpansionExpressionAST* ast) override;
  void visit(DesignatedInitializerClauseAST* ast) override;
  void visit(TypeTraitExpressionAST* ast) override;
  void visit(ConditionExpressionAST* ast) override;
  void visit(EqualInitializerAST* ast) override;
  void visit(BracedInitListAST* ast) override;
  void visit(ParenInitializerAST* ast) override;

  void visit(DefaultGenericAssociationAST* ast) override;
  void visit(TypeGenericAssociationAST* ast) override;

  void visit(DotDesignatorAST* ast) override;
  void visit(SubscriptDesignatorAST* ast) override;

  void visit(TemplateTypeParameterAST* ast) override;
  void visit(NonTypeTemplateParameterAST* ast) override;
  void visit(TypenameTypeParameterAST* ast) override;
  void visit(ConstraintTypeParameterAST* ast) override;

  void visit(GeneratedTypeSpecifierAST* ast) override;
  void visit(TypedefSpecifierAST* ast) override;
  void visit(FriendSpecifierAST* ast) override;
  void visit(ConstevalSpecifierAST* ast) override;
  void visit(ConstinitSpecifierAST* ast) override;
  void visit(ConstexprSpecifierAST* ast) override;
  void visit(InlineSpecifierAST* ast) override;
  void visit(NoreturnSpecifierAST* ast) override;
  void visit(StaticSpecifierAST* ast) override;
  void visit(ExternSpecifierAST* ast) override;
  void visit(RegisterSpecifierAST* ast) override;
  void visit(ThreadLocalSpecifierAST* ast) override;
  void visit(ThreadSpecifierAST* ast) override;
  void visit(MutableSpecifierAST* ast) override;
  void visit(VirtualSpecifierAST* ast) override;
  void visit(ExplicitSpecifierAST* ast) override;
  void visit(AutoTypeSpecifierAST* ast) override;
  void visit(VoidTypeSpecifierAST* ast) override;
  void visit(SizeTypeSpecifierAST* ast) override;
  void visit(SignTypeSpecifierAST* ast) override;
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
  void visit(PlaceholderTypeSpecifierAST* ast) override;
  void visit(ConstQualifierAST* ast) override;
  void visit(VolatileQualifierAST* ast) override;
  void visit(AtomicQualifierAST* ast) override;
  void visit(RestrictQualifierAST* ast) override;
  void visit(EnumSpecifierAST* ast) override;
  void visit(ClassSpecifierAST* ast) override;
  void visit(TypenameSpecifierAST* ast) override;
  void visit(SplicerTypeSpecifierAST* ast) override;

  void visit(PointerOperatorAST* ast) override;
  void visit(ReferenceOperatorAST* ast) override;
  void visit(PtrToMemberOperatorAST* ast) override;

  void visit(BitfieldDeclaratorAST* ast) override;
  void visit(ParameterPackAST* ast) override;
  void visit(IdDeclaratorAST* ast) override;
  void visit(NestedDeclaratorAST* ast) override;

  void visit(FunctionDeclaratorChunkAST* ast) override;
  void visit(ArrayDeclaratorChunkAST* ast) override;

  void visit(NameIdAST* ast) override;
  void visit(DestructorIdAST* ast) override;
  void visit(DecltypeIdAST* ast) override;
  void visit(OperatorFunctionIdAST* ast) override;
  void visit(LiteralOperatorIdAST* ast) override;
  void visit(ConversionFunctionIdAST* ast) override;
  void visit(SimpleTemplateIdAST* ast) override;
  void visit(LiteralOperatorTemplateIdAST* ast) override;
  void visit(OperatorFunctionTemplateIdAST* ast) override;

  void visit(GlobalNestedNameSpecifierAST* ast) override;
  void visit(SimpleNestedNameSpecifierAST* ast) override;
  void visit(DecltypeNestedNameSpecifierAST* ast) override;
  void visit(TemplateNestedNameSpecifierAST* ast) override;

  void visit(DefaultFunctionBodyAST* ast) override;
  void visit(CompoundStatementFunctionBodyAST* ast) override;
  void visit(TryStatementFunctionBodyAST* ast) override;
  void visit(DeleteFunctionBodyAST* ast) override;

  void visit(TypeTemplateArgumentAST* ast) override;
  void visit(ExpressionTemplateArgumentAST* ast) override;

  void visit(ThrowExceptionSpecifierAST* ast) override;
  void visit(NoexceptSpecifierAST* ast) override;

  void visit(SimpleRequirementAST* ast) override;
  void visit(CompoundRequirementAST* ast) override;
  void visit(TypeRequirementAST* ast) override;
  void visit(NestedRequirementAST* ast) override;

  void visit(NewParenInitializerAST* ast) override;
  void visit(NewBracedInitializerAST* ast) override;

  void visit(ParenMemInitializerAST* ast) override;
  void visit(BracedMemInitializerAST* ast) override;

  void visit(ThisLambdaCaptureAST* ast) override;
  void visit(DerefThisLambdaCaptureAST* ast) override;
  void visit(SimpleLambdaCaptureAST* ast) override;
  void visit(RefLambdaCaptureAST* ast) override;
  void visit(RefInitLambdaCaptureAST* ast) override;
  void visit(InitLambdaCaptureAST* ast) override;

  void visit(EllipsisExceptionDeclarationAST* ast) override;
  void visit(TypeExceptionDeclarationAST* ast) override;

  void visit(CxxAttributeAST* ast) override;
  void visit(GccAttributeAST* ast) override;
  void visit(AlignasAttributeAST* ast) override;
  void visit(AlignasTypeAttributeAST* ast) override;
  void visit(AsmAttributeAST* ast) override;

  void visit(ScopedAttributeTokenAST* ast) override;
  void visit(SimpleAttributeTokenAST* ast) override;

 private:
  std::intptr_t value_ = 0;
  int slot_ = 0;
  ASTSlotKind slotKind_ = ASTSlotKind::kInvalid;
  SlotNameIndex slotNameIndex_{};
  int slotCount_ = 0;
};

std::string_view to_string(SlotNameIndex index);

}  // namespace cxx
