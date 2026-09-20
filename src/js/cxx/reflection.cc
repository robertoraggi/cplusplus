// Generated file by: gen_reflection.ts
// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/private/model_inputs.h>
#include <cxx/translation_unit.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>

namespace cxx::js {
namespace {
using emscripten::val;

constexpr int ConstComplexSlotBase = 0;

constexpr int ConstObjectSlotBase = ConstComplexSlotBase + 2;

constexpr int ConstAddressSlotBase = ConstObjectSlotBase + 3;

constexpr int ConstLabelAddressSlotBase = ConstAddressSlotBase + 5;

constexpr int ASTSlotBase = 0;

constexpr int AttributeSpecifierASTSlotBase = ASTSlotBase + 3;

constexpr int ExpressionASTSlotBase = AttributeSpecifierASTSlotBase + 1;

constexpr int MemInitializerASTSlotBase = ExpressionASTSlotBase + 2;

constexpr int NestedNameSpecifierASTSlotBase = MemInitializerASTSlotBase + 2;

constexpr int TemplateParameterASTSlotBase = NestedNameSpecifierASTSlotBase + 1;

constexpr int UnitASTSlotBase = TemplateParameterASTSlotBase + 3;

constexpr int TranslationUnitASTSlotBase = UnitASTSlotBase + 1;

constexpr int ModuleUnitASTSlotBase = TranslationUnitASTSlotBase + 1;

constexpr int SimpleDeclarationASTSlotBase = ModuleUnitASTSlotBase + 4;

constexpr int AsmDeclarationASTSlotBase = SimpleDeclarationASTSlotBase + 5;

constexpr int NamespaceAliasDefinitionASTSlotBase =
    AsmDeclarationASTSlotBase + 12;

constexpr int UsingDeclarationASTSlotBase =
    NamespaceAliasDefinitionASTSlotBase + 8;

constexpr int UsingEnumDeclarationASTSlotBase = UsingDeclarationASTSlotBase + 3;

constexpr int UsingDirectiveASTSlotBase = UsingEnumDeclarationASTSlotBase + 3;

constexpr int StaticAssertDeclarationASTSlotBase =
    UsingDirectiveASTSlotBase + 6;

constexpr int AliasDeclarationASTSlotBase =
    StaticAssertDeclarationASTSlotBase + 9;

constexpr int OpaqueEnumDeclarationASTSlotBase =
    AliasDeclarationASTSlotBase + 9;

constexpr int FunctionDefinitionASTSlotBase =
    OpaqueEnumDeclarationASTSlotBase + 9;

constexpr int TemplateDeclarationASTSlotBase =
    FunctionDefinitionASTSlotBase + 6;

constexpr int ConceptDefinitionASTSlotBase = TemplateDeclarationASTSlotBase + 8;

constexpr int DeductionGuideASTSlotBase = ConceptDefinitionASTSlotBase + 7;

constexpr int ExplicitInstantiationASTSlotBase = DeductionGuideASTSlotBase + 11;

constexpr int ExportDeclarationASTSlotBase =
    ExplicitInstantiationASTSlotBase + 3;

constexpr int ExportCompoundDeclarationASTSlotBase =
    ExportDeclarationASTSlotBase + 2;

constexpr int LinkageSpecificationASTSlotBase =
    ExportCompoundDeclarationASTSlotBase + 4;

constexpr int NamespaceDefinitionASTSlotBase =
    LinkageSpecificationASTSlotBase + 6;

constexpr int EmptyDeclarationASTSlotBase = NamespaceDefinitionASTSlotBase + 12;

constexpr int AttributeDeclarationASTSlotBase = EmptyDeclarationASTSlotBase + 1;

constexpr int ModuleImportDeclarationASTSlotBase =
    AttributeDeclarationASTSlotBase + 2;

constexpr int ParameterDeclarationASTSlotBase =
    ModuleImportDeclarationASTSlotBase + 4;

constexpr int AccessDeclarationASTSlotBase =
    ParameterDeclarationASTSlotBase + 11;

constexpr int StructuredBindingDeclarationASTSlotBase =
    AccessDeclarationASTSlotBase + 3;

constexpr int AsmOperandASTSlotBase =
    StructuredBindingDeclarationASTSlotBase + 10;

constexpr int AsmQualifierASTSlotBase = AsmOperandASTSlotBase + 9;

constexpr int AsmClobberASTSlotBase = AsmQualifierASTSlotBase + 2;

constexpr int AsmGotoLabelASTSlotBase = AsmClobberASTSlotBase + 2;

constexpr int SplicerASTSlotBase = AsmGotoLabelASTSlotBase + 2;

constexpr int GlobalModuleFragmentASTSlotBase = SplicerASTSlotBase + 6;

constexpr int PrivateModuleFragmentASTSlotBase =
    GlobalModuleFragmentASTSlotBase + 3;

constexpr int ModuleDeclarationASTSlotBase =
    PrivateModuleFragmentASTSlotBase + 5;

constexpr int ModuleNameASTSlotBase = ModuleDeclarationASTSlotBase + 6;

constexpr int ModuleQualifierASTSlotBase = ModuleNameASTSlotBase + 3;

constexpr int ModulePartitionASTSlotBase = ModuleQualifierASTSlotBase + 4;

constexpr int ImportNameASTSlotBase = ModulePartitionASTSlotBase + 2;

constexpr int InitDeclaratorASTSlotBase = ImportNameASTSlotBase + 3;

constexpr int DeclaratorASTSlotBase = InitDeclaratorASTSlotBase + 4;

constexpr int UsingDeclaratorASTSlotBase = DeclaratorASTSlotBase + 3;

constexpr int EnumeratorASTSlotBase = UsingDeclaratorASTSlotBase + 6;

constexpr int TypeIdASTSlotBase = EnumeratorASTSlotBase + 6;

constexpr int HandlerASTSlotBase = TypeIdASTSlotBase + 4;

constexpr int BaseSpecifierASTSlotBase = HandlerASTSlotBase + 6;

constexpr int RequiresClauseASTSlotBase = BaseSpecifierASTSlotBase + 12;

constexpr int ParameterDeclarationClauseASTSlotBase =
    RequiresClauseASTSlotBase + 2;

constexpr int TrailingReturnTypeASTSlotBase =
    ParameterDeclarationClauseASTSlotBase + 5;

constexpr int LambdaSpecifierASTSlotBase = TrailingReturnTypeASTSlotBase + 2;

constexpr int TypeConstraintASTSlotBase = LambdaSpecifierASTSlotBase + 2;

constexpr int AttributeArgumentClauseASTSlotBase =
    TypeConstraintASTSlotBase + 7;

constexpr int AttributeASTSlotBase = AttributeArgumentClauseASTSlotBase + 3;

constexpr int AttributeUsingPrefixASTSlotBase = AttributeASTSlotBase + 3;

constexpr int NewPlacementASTSlotBase = AttributeUsingPrefixASTSlotBase + 3;

constexpr int NestedNamespaceSpecifierASTSlotBase = NewPlacementASTSlotBase + 3;

constexpr int LabeledStatementASTSlotBase =
    NestedNamespaceSpecifierASTSlotBase + 6;

constexpr int CaseStatementASTSlotBase = LabeledStatementASTSlotBase + 4;

constexpr int DefaultStatementASTSlotBase = CaseStatementASTSlotBase + 4;

constexpr int ExpressionStatementASTSlotBase = DefaultStatementASTSlotBase + 2;

constexpr int CompoundStatementASTSlotBase = ExpressionStatementASTSlotBase + 3;

constexpr int IfStatementASTSlotBase = CompoundStatementASTSlotBase + 5;

constexpr int ConstevalIfStatementASTSlotBase = IfStatementASTSlotBase + 11;

constexpr int SwitchStatementASTSlotBase = ConstevalIfStatementASTSlotBase + 8;

constexpr int WhileStatementASTSlotBase = SwitchStatementASTSlotBase + 8;

constexpr int DoStatementASTSlotBase = WhileStatementASTSlotBase + 7;

constexpr int ForRangeStatementASTSlotBase = DoStatementASTSlotBase + 8;

constexpr int ForStatementASTSlotBase = ForRangeStatementASTSlotBase + 27;

constexpr int BreakStatementASTSlotBase = ForStatementASTSlotBase + 10;

constexpr int ContinueStatementASTSlotBase = BreakStatementASTSlotBase + 3;

constexpr int ReturnStatementASTSlotBase = ContinueStatementASTSlotBase + 3;

constexpr int CoroutineReturnStatementASTSlotBase =
    ReturnStatementASTSlotBase + 4;

constexpr int GotoStatementASTSlotBase =
    CoroutineReturnStatementASTSlotBase + 4;

constexpr int DeclarationStatementASTSlotBase = GotoStatementASTSlotBase + 8;

constexpr int TryBlockStatementASTSlotBase =
    DeclarationStatementASTSlotBase + 1;

constexpr int CharLiteralExpressionASTSlotBase =
    TryBlockStatementASTSlotBase + 4;

constexpr int BoolLiteralExpressionASTSlotBase =
    CharLiteralExpressionASTSlotBase + 3;

constexpr int IntLiteralExpressionASTSlotBase =
    BoolLiteralExpressionASTSlotBase + 2;

constexpr int FloatLiteralExpressionASTSlotBase =
    IntLiteralExpressionASTSlotBase + 3;

constexpr int NullptrLiteralExpressionASTSlotBase =
    FloatLiteralExpressionASTSlotBase + 3;

constexpr int StringLiteralExpressionASTSlotBase =
    NullptrLiteralExpressionASTSlotBase + 2;

constexpr int UserDefinedStringLiteralExpressionASTSlotBase =
    StringLiteralExpressionASTSlotBase + 3;

constexpr int ObjectLiteralExpressionASTSlotBase =
    UserDefinedStringLiteralExpressionASTSlotBase + 4;

constexpr int ThisExpressionASTSlotBase =
    ObjectLiteralExpressionASTSlotBase + 5;

constexpr int PackIndexExpressionASTSlotBase = ThisExpressionASTSlotBase + 1;

constexpr int GenericSelectionExpressionASTSlotBase =
    PackIndexExpressionASTSlotBase + 5;

constexpr int NestedStatementExpressionASTSlotBase =
    GenericSelectionExpressionASTSlotBase + 7;

constexpr int DefaultInitializerExpressionASTSlotBase =
    NestedStatementExpressionASTSlotBase + 3;

constexpr int NestedExpressionASTSlotBase =
    DefaultInitializerExpressionASTSlotBase + 2;

constexpr int IdExpressionASTSlotBase = NestedExpressionASTSlotBase + 3;

constexpr int LambdaExpressionASTSlotBase = IdExpressionASTSlotBase + 5;

constexpr int FoldExpressionASTSlotBase = LambdaExpressionASTSlotBase + 22;

constexpr int RightFoldExpressionASTSlotBase = FoldExpressionASTSlotBase + 9;

constexpr int LeftFoldExpressionASTSlotBase =
    RightFoldExpressionASTSlotBase + 6;

constexpr int RequiresExpressionASTSlotBase = LeftFoldExpressionASTSlotBase + 6;

constexpr int VaArgExpressionASTSlotBase = RequiresExpressionASTSlotBase + 7;

constexpr int SubscriptExpressionASTSlotBase = VaArgExpressionASTSlotBase + 6;

constexpr int CallExpressionASTSlotBase = SubscriptExpressionASTSlotBase + 6;

constexpr int TypeConstructionASTSlotBase = CallExpressionASTSlotBase + 6;

constexpr int BracedTypeConstructionASTSlotBase =
    TypeConstructionASTSlotBase + 5;

constexpr int SpliceMemberExpressionASTSlotBase =
    BracedTypeConstructionASTSlotBase + 3;

constexpr int MemberExpressionASTSlotBase =
    SpliceMemberExpressionASTSlotBase + 7;

constexpr int PostIncrExpressionASTSlotBase = MemberExpressionASTSlotBase + 8;

constexpr int CppCastExpressionASTSlotBase = PostIncrExpressionASTSlotBase + 5;

constexpr int BuiltinBitCastExpressionASTSlotBase =
    CppCastExpressionASTSlotBase + 8;

constexpr int BuiltinOffsetofExpressionASTSlotBase =
    BuiltinBitCastExpressionASTSlotBase + 6;

constexpr int TypeidExpressionASTSlotBase =
    BuiltinOffsetofExpressionASTSlotBase + 9;

constexpr int TypeidOfTypeExpressionASTSlotBase =
    TypeidExpressionASTSlotBase + 4;

constexpr int SpliceExpressionASTSlotBase =
    TypeidOfTypeExpressionASTSlotBase + 4;

constexpr int GlobalScopeReflectExpressionASTSlotBase =
    SpliceExpressionASTSlotBase + 1;

constexpr int NamespaceReflectExpressionASTSlotBase =
    GlobalScopeReflectExpressionASTSlotBase + 2;

constexpr int TypeIdReflectExpressionASTSlotBase =
    NamespaceReflectExpressionASTSlotBase + 4;

constexpr int ReflectExpressionASTSlotBase =
    TypeIdReflectExpressionASTSlotBase + 2;

constexpr int LabelAddressExpressionASTSlotBase =
    ReflectExpressionASTSlotBase + 2;

constexpr int UnaryExpressionASTSlotBase =
    LabelAddressExpressionASTSlotBase + 3;

constexpr int AwaitExpressionASTSlotBase = UnaryExpressionASTSlotBase + 5;

constexpr int SizeofExpressionASTSlotBase = AwaitExpressionASTSlotBase + 2;

constexpr int SizeofTypeExpressionASTSlotBase = SizeofExpressionASTSlotBase + 3;

constexpr int SizeofPackExpressionASTSlotBase =
    SizeofTypeExpressionASTSlotBase + 5;

constexpr int AlignofTypeExpressionASTSlotBase =
    SizeofPackExpressionASTSlotBase + 7;

constexpr int AlignofExpressionASTSlotBase =
    AlignofTypeExpressionASTSlotBase + 4;

constexpr int NoexceptExpressionASTSlotBase = AlignofExpressionASTSlotBase + 2;

constexpr int NewExpressionASTSlotBase = NoexceptExpressionASTSlotBase + 5;

constexpr int DeleteExpressionASTSlotBase = NewExpressionASTSlotBase + 11;

constexpr int CastExpressionASTSlotBase = DeleteExpressionASTSlotBase + 6;

constexpr int ImplicitCastExpressionASTSlotBase = CastExpressionASTSlotBase + 4;

constexpr int ConstExpressionASTSlotBase =
    ImplicitCastExpressionASTSlotBase + 4;

constexpr int BinaryExpressionASTSlotBase = ConstExpressionASTSlotBase + 2;

constexpr int ConditionalExpressionASTSlotBase =
    BinaryExpressionASTSlotBase + 6;

constexpr int YieldExpressionASTSlotBase = ConditionalExpressionASTSlotBase + 5;

constexpr int ThrowExpressionASTSlotBase = YieldExpressionASTSlotBase + 2;

constexpr int AssignmentExpressionASTSlotBase = ThrowExpressionASTSlotBase + 2;

constexpr int CompoundAssignmentExpressionASTSlotBase =
    AssignmentExpressionASTSlotBase + 6;

constexpr int PackExpansionExpressionASTSlotBase =
    CompoundAssignmentExpressionASTSlotBase + 8;

constexpr int DesignatedInitializerClauseASTSlotBase =
    PackExpansionExpressionASTSlotBase + 2;

constexpr int TypeTraitExpressionASTSlotBase =
    DesignatedInitializerClauseASTSlotBase + 3;

constexpr int ConditionExpressionASTSlotBase =
    TypeTraitExpressionASTSlotBase + 6;

constexpr int EqualInitializerASTSlotBase = ConditionExpressionASTSlotBase + 5;

constexpr int BracedInitListASTSlotBase = EqualInitializerASTSlotBase + 2;

constexpr int ParenInitializerASTSlotBase = BracedInitListASTSlotBase + 4;

constexpr int ThreeWayComparisonExpressionASTSlotBase =
    ParenInitializerASTSlotBase + 3;

constexpr int DefaultGenericAssociationASTSlotBase =
    ThreeWayComparisonExpressionASTSlotBase + 5;

constexpr int TypeGenericAssociationASTSlotBase =
    DefaultGenericAssociationASTSlotBase + 3;

constexpr int DotDesignatorASTSlotBase = TypeGenericAssociationASTSlotBase + 3;

constexpr int SubscriptDesignatorASTSlotBase = DotDesignatorASTSlotBase + 4;

constexpr int TemplateTypeParameterASTSlotBase =
    SubscriptDesignatorASTSlotBase + 3;

constexpr int NonTypeTemplateParameterASTSlotBase =
    TemplateTypeParameterASTSlotBase + 12;

constexpr int TypenameTypeParameterASTSlotBase =
    NonTypeTemplateParameterASTSlotBase + 1;

constexpr int ConstraintTypeParameterASTSlotBase =
    TypenameTypeParameterASTSlotBase + 7;

constexpr int TypedefSpecifierASTSlotBase =
    ConstraintTypeParameterASTSlotBase + 6;

constexpr int FriendSpecifierASTSlotBase = TypedefSpecifierASTSlotBase + 1;

constexpr int ConstevalSpecifierASTSlotBase = FriendSpecifierASTSlotBase + 1;

constexpr int ConstinitSpecifierASTSlotBase = ConstevalSpecifierASTSlotBase + 1;

constexpr int ConstexprSpecifierASTSlotBase = ConstinitSpecifierASTSlotBase + 1;

constexpr int InlineSpecifierASTSlotBase = ConstexprSpecifierASTSlotBase + 1;

constexpr int NoreturnSpecifierASTSlotBase = InlineSpecifierASTSlotBase + 1;

constexpr int StaticSpecifierASTSlotBase = NoreturnSpecifierASTSlotBase + 1;

constexpr int ExternSpecifierASTSlotBase = StaticSpecifierASTSlotBase + 1;

constexpr int RegisterSpecifierASTSlotBase = ExternSpecifierASTSlotBase + 1;

constexpr int ThreadLocalSpecifierASTSlotBase =
    RegisterSpecifierASTSlotBase + 1;

constexpr int ThreadSpecifierASTSlotBase = ThreadLocalSpecifierASTSlotBase + 1;

constexpr int MutableSpecifierASTSlotBase = ThreadSpecifierASTSlotBase + 1;

constexpr int VirtualSpecifierASTSlotBase = MutableSpecifierASTSlotBase + 1;

constexpr int ExplicitSpecifierASTSlotBase = VirtualSpecifierASTSlotBase + 1;

constexpr int AutoTypeSpecifierASTSlotBase = ExplicitSpecifierASTSlotBase + 4;

constexpr int VoidTypeSpecifierASTSlotBase = AutoTypeSpecifierASTSlotBase + 1;

constexpr int SizeTypeSpecifierASTSlotBase = VoidTypeSpecifierASTSlotBase + 1;

constexpr int SignTypeSpecifierASTSlotBase = SizeTypeSpecifierASTSlotBase + 2;

constexpr int BuiltinTypeSpecifierASTSlotBase =
    SignTypeSpecifierASTSlotBase + 2;

constexpr int UnaryBuiltinTypeSpecifierASTSlotBase =
    BuiltinTypeSpecifierASTSlotBase + 2;

constexpr int BinaryBuiltinTypeSpecifierASTSlotBase =
    UnaryBuiltinTypeSpecifierASTSlotBase + 5;

constexpr int IntegralTypeSpecifierASTSlotBase =
    BinaryBuiltinTypeSpecifierASTSlotBase + 7;

constexpr int FloatingPointTypeSpecifierASTSlotBase =
    IntegralTypeSpecifierASTSlotBase + 2;

constexpr int ComplexTypeSpecifierASTSlotBase =
    FloatingPointTypeSpecifierASTSlotBase + 2;

constexpr int NamedTypeSpecifierASTSlotBase =
    ComplexTypeSpecifierASTSlotBase + 1;

constexpr int AtomicTypeSpecifierASTSlotBase =
    NamedTypeSpecifierASTSlotBase + 5;

constexpr int BitIntTypeSpecifierASTSlotBase =
    AtomicTypeSpecifierASTSlotBase + 4;

constexpr int UnderlyingTypeSpecifierASTSlotBase =
    BitIntTypeSpecifierASTSlotBase + 5;

constexpr int ElaboratedTypeSpecifierASTSlotBase =
    UnderlyingTypeSpecifierASTSlotBase + 4;

constexpr int DecltypeAutoSpecifierASTSlotBase =
    ElaboratedTypeSpecifierASTSlotBase + 8;

constexpr int DecltypeSpecifierASTSlotBase =
    DecltypeAutoSpecifierASTSlotBase + 4;

constexpr int PlaceholderTypeSpecifierASTSlotBase =
    DecltypeSpecifierASTSlotBase + 5;

constexpr int ConstQualifierASTSlotBase =
    PlaceholderTypeSpecifierASTSlotBase + 2;

constexpr int VolatileQualifierASTSlotBase = ConstQualifierASTSlotBase + 1;

constexpr int AtomicQualifierASTSlotBase = VolatileQualifierASTSlotBase + 1;

constexpr int RestrictQualifierASTSlotBase = AtomicQualifierASTSlotBase + 1;

constexpr int EnumSpecifierASTSlotBase = RestrictQualifierASTSlotBase + 1;

constexpr int ClassSpecifierASTSlotBase = EnumSpecifierASTSlotBase + 12;

constexpr int TypenameSpecifierASTSlotBase = ClassSpecifierASTSlotBase + 13;

constexpr int SplicerTypeSpecifierASTSlotBase =
    TypenameSpecifierASTSlotBase + 6;

constexpr int PointerOperatorASTSlotBase = SplicerTypeSpecifierASTSlotBase + 2;

constexpr int ReferenceOperatorASTSlotBase = PointerOperatorASTSlotBase + 3;

constexpr int PtrToMemberOperatorASTSlotBase = ReferenceOperatorASTSlotBase + 3;

constexpr int BitfieldDeclaratorASTSlotBase =
    PtrToMemberOperatorASTSlotBase + 4;

constexpr int ParameterPackASTSlotBase = BitfieldDeclaratorASTSlotBase + 3;

constexpr int IdDeclaratorASTSlotBase = ParameterPackASTSlotBase + 2;

constexpr int NestedDeclaratorASTSlotBase = IdDeclaratorASTSlotBase + 5;

constexpr int FunctionDeclaratorChunkASTSlotBase =
    NestedDeclaratorASTSlotBase + 3;

constexpr int ArrayDeclaratorChunkASTSlotBase =
    FunctionDeclaratorChunkASTSlotBase + 12;

constexpr int NameIdASTSlotBase = ArrayDeclaratorChunkASTSlotBase + 5;

constexpr int DestructorIdASTSlotBase = NameIdASTSlotBase + 2;

constexpr int DecltypeIdASTSlotBase = DestructorIdASTSlotBase + 2;

constexpr int OperatorFunctionIdASTSlotBase = DecltypeIdASTSlotBase + 1;

constexpr int LiteralOperatorIdASTSlotBase = OperatorFunctionIdASTSlotBase + 5;

constexpr int ConversionFunctionIdASTSlotBase =
    LiteralOperatorIdASTSlotBase + 5;

constexpr int SimpleTemplateIdASTSlotBase = ConversionFunctionIdASTSlotBase + 2;

constexpr int LiteralOperatorTemplateIdASTSlotBase =
    SimpleTemplateIdASTSlotBase + 6;

constexpr int OperatorFunctionTemplateIdASTSlotBase =
    LiteralOperatorTemplateIdASTSlotBase + 4;

constexpr int GlobalNestedNameSpecifierASTSlotBase =
    OperatorFunctionTemplateIdASTSlotBase + 4;

constexpr int SimpleNestedNameSpecifierASTSlotBase =
    GlobalNestedNameSpecifierASTSlotBase + 1;

constexpr int DecltypeNestedNameSpecifierASTSlotBase =
    SimpleNestedNameSpecifierASTSlotBase + 4;

constexpr int TemplateNestedNameSpecifierASTSlotBase =
    DecltypeNestedNameSpecifierASTSlotBase + 2;

constexpr int DefaultFunctionBodyASTSlotBase =
    TemplateNestedNameSpecifierASTSlotBase + 5;

constexpr int CompoundStatementFunctionBodyASTSlotBase =
    DefaultFunctionBodyASTSlotBase + 3;

constexpr int TryStatementFunctionBodyASTSlotBase =
    CompoundStatementFunctionBodyASTSlotBase + 3;

constexpr int DeleteFunctionBodyASTSlotBase =
    TryStatementFunctionBodyASTSlotBase + 5;

constexpr int TypeTemplateArgumentASTSlotBase =
    DeleteFunctionBodyASTSlotBase + 3;

constexpr int ExpressionTemplateArgumentASTSlotBase =
    TypeTemplateArgumentASTSlotBase + 1;

constexpr int ThrowExceptionSpecifierASTSlotBase =
    ExpressionTemplateArgumentASTSlotBase + 1;

constexpr int NoexceptSpecifierASTSlotBase =
    ThrowExceptionSpecifierASTSlotBase + 3;

constexpr int SimpleRequirementASTSlotBase = NoexceptSpecifierASTSlotBase + 4;

constexpr int CompoundRequirementASTSlotBase = SimpleRequirementASTSlotBase + 2;

constexpr int TypeRequirementASTSlotBase = CompoundRequirementASTSlotBase + 7;

constexpr int NestedRequirementASTSlotBase = TypeRequirementASTSlotBase + 6;

constexpr int NewParenInitializerASTSlotBase = NestedRequirementASTSlotBase + 3;

constexpr int NewBracedInitializerASTSlotBase =
    NewParenInitializerASTSlotBase + 3;

constexpr int ParenMemInitializerASTSlotBase =
    NewBracedInitializerASTSlotBase + 1;

constexpr int BracedMemInitializerASTSlotBase =
    ParenMemInitializerASTSlotBase + 6;

constexpr int ThisLambdaCaptureASTSlotBase =
    BracedMemInitializerASTSlotBase + 4;

constexpr int DerefThisLambdaCaptureASTSlotBase =
    ThisLambdaCaptureASTSlotBase + 3;

constexpr int SimpleLambdaCaptureASTSlotBase =
    DerefThisLambdaCaptureASTSlotBase + 3;

constexpr int RefLambdaCaptureASTSlotBase = SimpleLambdaCaptureASTSlotBase + 5;

constexpr int RefInitLambdaCaptureASTSlotBase = RefLambdaCaptureASTSlotBase + 6;

constexpr int InitLambdaCaptureASTSlotBase =
    RefInitLambdaCaptureASTSlotBase + 6;

constexpr int EllipsisExceptionDeclarationASTSlotBase =
    InitLambdaCaptureASTSlotBase + 5;

constexpr int TypeExceptionDeclarationASTSlotBase =
    EllipsisExceptionDeclarationASTSlotBase + 1;

constexpr int CxxAttributeASTSlotBase = TypeExceptionDeclarationASTSlotBase + 4;

constexpr int GccAttributeASTSlotBase = CxxAttributeASTSlotBase + 6;

constexpr int AlignasAttributeASTSlotBase = GccAttributeASTSlotBase + 6;

constexpr int AlignasTypeAttributeASTSlotBase = AlignasAttributeASTSlotBase + 6;

constexpr int AsmAttributeASTSlotBase = AlignasTypeAttributeASTSlotBase + 6;

constexpr int ScopedAttributeTokenASTSlotBase = AsmAttributeASTSlotBase + 5;

constexpr int SimpleAttributeTokenASTSlotBase =
    ScopedAttributeTokenASTSlotBase + 5;

constexpr int LiteralSlotBase = 0;

constexpr int IntegerLiteralSlotBase = LiteralSlotBase + 2;

constexpr int FloatLiteralSlotBase = IntegerLiteralSlotBase + 2;

constexpr int StringLiteralSlotBase = FloatLiteralSlotBase + 2;

constexpr int CharLiteralSlotBase = StringLiteralSlotBase + 5;

constexpr int NameSlotBase = 0;

constexpr int IdentifierSlotBase = NameSlotBase + 2;

constexpr int OperatorIdSlotBase = IdentifierSlotBase + 8;

constexpr int DestructorIdSlotBase = OperatorIdSlotBase + 1;

constexpr int LiteralOperatorIdSlotBase = DestructorIdSlotBase + 1;

constexpr int ConversionFunctionIdSlotBase = LiteralOperatorIdSlotBase + 1;

constexpr int TemplateIdSlotBase = ConversionFunctionIdSlotBase + 1;

constexpr int SymbolSlotBase = 0;

constexpr int ScopeSymbolSlotBase = SymbolSlotBase + 54;

constexpr int NamespaceSymbolSlotBase = ScopeSymbolSlotBase + 4;

constexpr int ConceptSymbolSlotBase = NamespaceSymbolSlotBase + 4;

constexpr int DeductionGuideSymbolSlotBase = ConceptSymbolSlotBase + 9;

constexpr int BaseClassSymbolSlotBase = DeductionGuideSymbolSlotBase + 10;

constexpr int InjectedClassNameSymbolSlotBase = BaseClassSymbolSlotBase + 2;

constexpr int ClassSymbolSlotBase = InjectedClassNameSymbolSlotBase + 1;

constexpr int EnumSymbolSlotBase = ClassSymbolSlotBase + 60;

constexpr int ScopedEnumSymbolSlotBase = EnumSymbolSlotBase + 3;

constexpr int FunctionSymbolSlotBase = ScopedEnumSymbolSlotBase + 2;

constexpr int OverloadSetSymbolSlotBase = FunctionSymbolSlotBase + 69;

constexpr int LambdaSymbolSlotBase = OverloadSetSymbolSlotBase + 3;

constexpr int TemplateParametersSymbolSlotBase = LambdaSymbolSlotBase + 7;

constexpr int BlockSymbolSlotBase = TemplateParametersSymbolSlotBase + 1;

constexpr int TypeAliasSymbolSlotBase = BlockSymbolSlotBase + 1;

constexpr int VariableSymbolSlotBase = TypeAliasSymbolSlotBase + 16;

constexpr int FieldSymbolSlotBase = VariableSymbolSlotBase + 25;

constexpr int ParameterSymbolSlotBase = FieldSymbolSlotBase + 21;

constexpr int ParameterPackSymbolSlotBase = ParameterSymbolSlotBase + 2;

constexpr int TypeParameterSymbolSlotBase = ParameterPackSymbolSlotBase + 1;

constexpr int NonTypeParameterSymbolSlotBase = TypeParameterSymbolSlotBase + 1;

constexpr int TemplateTypeParameterSymbolSlotBase =
    NonTypeParameterSymbolSlotBase + 5;

constexpr int ConstraintTypeParameterSymbolSlotBase =
    TemplateTypeParameterSymbolSlotBase + 1;

constexpr int EnumeratorSymbolSlotBase =
    ConstraintTypeParameterSymbolSlotBase + 6;

constexpr int NamespaceAliasSymbolSlotBase = EnumeratorSymbolSlotBase + 1;

constexpr int UsingDeclarationSymbolSlotBase = NamespaceAliasSymbolSlotBase + 1;

constexpr int TypeSlotBase = 0;

constexpr int QualTypeSlotBase = TypeSlotBase + 1;

constexpr int BoundedArrayTypeSlotBase = QualTypeSlotBase + 4;

constexpr int UnboundedArrayTypeSlotBase = BoundedArrayTypeSlotBase + 2;

constexpr int PointerTypeSlotBase = UnboundedArrayTypeSlotBase + 1;

constexpr int LvalueReferenceTypeSlotBase = PointerTypeSlotBase + 1;

constexpr int RvalueReferenceTypeSlotBase = LvalueReferenceTypeSlotBase + 1;

constexpr int OverloadSetTypeSlotBase = RvalueReferenceTypeSlotBase + 1;

constexpr int FunctionTypeSlotBase = OverloadSetTypeSlotBase + 1;

constexpr int ClassTypeSlotBase = FunctionTypeSlotBase + 6;

constexpr int EnumTypeSlotBase = ClassTypeSlotBase + 4;

constexpr int ScopedEnumTypeSlotBase = EnumTypeSlotBase + 2;

constexpr int MemberObjectPointerTypeSlotBase = ScopedEnumTypeSlotBase + 2;

constexpr int MemberFunctionPointerTypeSlotBase =
    MemberObjectPointerTypeSlotBase + 2;

constexpr int NamespaceTypeSlotBase = MemberFunctionPointerTypeSlotBase + 2;

constexpr int TypeParameterTypeSlotBase = NamespaceTypeSlotBase + 1;

constexpr int TemplateTypeParameterTypeSlotBase = TypeParameterTypeSlotBase + 3;

constexpr int UnresolvedNameTypeSlotBase =
    TemplateTypeParameterTypeSlotBase + 4;

constexpr int UnresolvedBoundedArrayTypeSlotBase =
    UnresolvedNameTypeSlotBase + 3;

constexpr int UnresolvedUnderlyingTypeSlotBase =
    UnresolvedBoundedArrayTypeSlotBase + 2;

constexpr int UnresolvedBuiltinTypeSlotBase =
    UnresolvedUnderlyingTypeSlotBase + 1;

constexpr int BitIntTypeSlotBase = UnresolvedBuiltinTypeSlotBase + 2;

constexpr int UnsignedBitIntTypeSlotBase = BitIntTypeSlotBase + 1;

constexpr int UnresolvedBitIntTypeSlotBase = UnsignedBitIntTypeSlotBase + 1;

constexpr int VectorTypeSlotBase = UnresolvedBitIntTypeSlotBase + 2;

constexpr int UnresolvedVectorTypeSlotBase = VectorTypeSlotBase + 3;

constexpr int ComplexTypeSlotBase = UnresolvedVectorTypeSlotBase + 4;

constexpr int AtomicTypeSlotBase = ComplexTypeSlotBase + 1;

template <typename T, typename F>
auto optionalValue(const T& value, F convert) -> val {
  if (!value) return val::undefined();
  return convert(*value);
}

template <typename T, typename F>
auto arrayValue(const T& values, F convert) -> val {
  auto result = val::array();
  for (const auto& item : values) result.call<void>("push", convert(item));
  return result;
}

auto readAST(std::intptr_t handle, int slot) -> double {
  switch (slot) {
    case ASTSlotBase + 0: {
      auto self = reinterpret_cast<const ::cxx::AST*>(handle);
      return static_cast<double>(self->internalId());
    }
    case ASTSlotBase + 1: {
      auto self = reinterpret_cast<const ::cxx::AST*>(handle);
      return static_cast<double>(
          const_cast<::cxx::AST*>(self)->firstSourceLocation().index());
    }
    case ASTSlotBase + 2: {
      auto self = reinterpret_cast<const ::cxx::AST*>(handle);
      return static_cast<double>(
          const_cast<::cxx::AST*>(self)->lastSourceLocation().index());
    }
    case ExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->valueCategory);
    }
    case ExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->type)));
    }
    case MemInitializerASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::MemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case MemInitializerASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::MemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructor)));
    }
    case NestedNameSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case TemplateParameterASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TemplateParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case TemplateParameterASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TemplateParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->depth);
    }
    case TemplateParameterASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TemplateParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->index);
    }
    case UnitASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnitAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case TranslationUnitASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TranslationUnitAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declarationList));
    }
    case ModuleUnitASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ModuleUnitAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->globalModuleFragment)));
    }
    case ModuleUnitASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ModuleUnitAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->moduleDeclaration)));
    }
    case ModuleUnitASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ModuleUnitAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declarationList));
    }
    case ModuleUnitASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ModuleUnitAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->privateModuleFragment)));
    }
    case SimpleDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SimpleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case SimpleDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SimpleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declSpecifierList));
    }
    case SimpleDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SimpleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->initDeclaratorList));
    }
    case SimpleDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SimpleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->requiresClause)));
    }
    case SimpleDeclarationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::SimpleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case AsmDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case AsmDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->asmQualifierList));
    }
    case AsmDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->asmLoc.index());
    }
    case AsmDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case AsmDeclarationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case AsmDeclarationASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->outputOperandList));
    }
    case AsmDeclarationASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->inputOperandList));
    }
    case AsmDeclarationASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->clobberList));
    }
    case AsmDeclarationASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->gotoLabelList));
    }
    case AsmDeclarationASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case AsmDeclarationASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case AsmDeclarationASTSlotBase + 11: {
      auto self = static_cast<const ::cxx::AsmDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case NamespaceAliasDefinitionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NamespaceAliasDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->namespaceLoc.index());
    }
    case NamespaceAliasDefinitionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NamespaceAliasDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case NamespaceAliasDefinitionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NamespaceAliasDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case NamespaceAliasDefinitionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::NamespaceAliasDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case NamespaceAliasDefinitionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::NamespaceAliasDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case NamespaceAliasDefinitionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::NamespaceAliasDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case NamespaceAliasDefinitionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::NamespaceAliasDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case NamespaceAliasDefinitionASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::NamespaceAliasDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case UsingDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::UsingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->usingLoc.index());
    }
    case UsingDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::UsingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->usingDeclaratorList));
    }
    case UsingDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::UsingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case UsingEnumDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::UsingEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->usingLoc.index());
    }
    case UsingEnumDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::UsingEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->enumTypeSpecifier)));
    }
    case UsingEnumDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::UsingEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case UsingDirectiveASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::UsingDirectiveAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case UsingDirectiveASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::UsingDirectiveAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->usingLoc.index());
    }
    case UsingDirectiveASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::UsingDirectiveAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->namespaceLoc.index());
    }
    case UsingDirectiveASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::UsingDirectiveAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case UsingDirectiveASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::UsingDirectiveAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case UsingDirectiveASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::UsingDirectiveAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case StaticAssertDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::StaticAssertDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->staticAssertLoc.index());
    }
    case StaticAssertDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::StaticAssertDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case StaticAssertDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::StaticAssertDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case StaticAssertDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::StaticAssertDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->commaLoc.index());
    }
    case StaticAssertDeclarationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::StaticAssertDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case StaticAssertDeclarationASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::StaticAssertDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case StaticAssertDeclarationASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::StaticAssertDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case StaticAssertDeclarationASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::StaticAssertDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case AliasDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AliasDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->usingLoc.index());
    }
    case AliasDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AliasDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case AliasDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AliasDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case AliasDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::AliasDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case AliasDeclarationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::AliasDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->gnuAttributeList));
    }
    case AliasDeclarationASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::AliasDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case AliasDeclarationASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::AliasDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case AliasDeclarationASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::AliasDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case AliasDeclarationASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::AliasDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case OpaqueEnumDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::OpaqueEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->enumLoc.index());
    }
    case OpaqueEnumDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::OpaqueEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->classLoc.index());
    }
    case OpaqueEnumDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::OpaqueEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case OpaqueEnumDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::OpaqueEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case OpaqueEnumDeclarationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::OpaqueEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case OpaqueEnumDeclarationASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::OpaqueEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case OpaqueEnumDeclarationASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::OpaqueEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->typeSpecifierList));
    }
    case OpaqueEnumDeclarationASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::OpaqueEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->emicolonLoc.index());
    }
    case OpaqueEnumDeclarationASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::OpaqueEnumDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case FunctionDefinitionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::FunctionDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case FunctionDefinitionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::FunctionDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declSpecifierList));
    }
    case FunctionDefinitionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::FunctionDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declarator)));
    }
    case FunctionDefinitionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::FunctionDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->requiresClause)));
    }
    case FunctionDefinitionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::FunctionDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->functionBody)));
    }
    case FunctionDefinitionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::FunctionDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case TemplateDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TemplateDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case TemplateDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TemplateDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lessLoc.index());
    }
    case TemplateDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TemplateDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->templateParameterList));
    }
    case TemplateDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TemplateDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->greaterLoc.index());
    }
    case TemplateDeclarationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TemplateDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->requiresClause)));
    }
    case TemplateDeclarationASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::TemplateDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration)));
    }
    case TemplateDeclarationASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::TemplateDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case TemplateDeclarationASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::TemplateDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->depth);
    }
    case ConceptDefinitionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConceptDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->conceptLoc.index());
    }
    case ConceptDefinitionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ConceptDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case ConceptDefinitionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ConceptDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case ConceptDefinitionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ConceptDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ConceptDefinitionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ConceptDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case ConceptDefinitionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ConceptDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case ConceptDefinitionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::ConceptDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case DeductionGuideASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case DeductionGuideASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->explicitSpecifier)));
    }
    case DeductionGuideASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case DeductionGuideASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case DeductionGuideASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->parameterDeclarationClause)));
    }
    case DeductionGuideASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case DeductionGuideASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->arrowLoc.index());
    }
    case DeductionGuideASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->templateId)));
    }
    case DeductionGuideASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case DeductionGuideASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case DeductionGuideASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::DeductionGuideAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case ExplicitInstantiationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ExplicitInstantiationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->externLoc.index());
    }
    case ExplicitInstantiationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ExplicitInstantiationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case ExplicitInstantiationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ExplicitInstantiationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration)));
    }
    case ExportDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ExportDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->exportLoc.index());
    }
    case ExportDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ExportDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration)));
    }
    case ExportCompoundDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ExportCompoundDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->exportLoc.index());
    }
    case ExportCompoundDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ExportCompoundDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbraceLoc.index());
    }
    case ExportCompoundDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ExportCompoundDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declarationList));
    }
    case ExportCompoundDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ExportCompoundDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbraceLoc.index());
    }
    case LinkageSpecificationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::LinkageSpecificationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->externLoc.index());
    }
    case LinkageSpecificationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::LinkageSpecificationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->stringliteralLoc.index());
    }
    case LinkageSpecificationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::LinkageSpecificationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbraceLoc.index());
    }
    case LinkageSpecificationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::LinkageSpecificationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declarationList));
    }
    case LinkageSpecificationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::LinkageSpecificationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbraceLoc.index());
    }
    case LinkageSpecificationASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::LinkageSpecificationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->stringLiteral)));
    }
    case NamespaceDefinitionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->inlineLoc.index());
    }
    case NamespaceDefinitionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->namespaceLoc.index());
    }
    case NamespaceDefinitionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case NamespaceDefinitionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->nestedNamespaceSpecifierList));
    }
    case NamespaceDefinitionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case NamespaceDefinitionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->extraAttributeList));
    }
    case NamespaceDefinitionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbraceLoc.index());
    }
    case NamespaceDefinitionASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declarationList));
    }
    case NamespaceDefinitionASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbraceLoc.index());
    }
    case NamespaceDefinitionASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case NamespaceDefinitionASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case NamespaceDefinitionASTSlotBase + 11: {
      auto self = static_cast<const ::cxx::NamespaceDefinitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isInline);
    }
    case EmptyDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::EmptyDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case AttributeDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AttributeDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case AttributeDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AttributeDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case ModuleImportDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ModuleImportDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->importLoc.index());
    }
    case ModuleImportDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ModuleImportDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->importName)));
    }
    case ModuleImportDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ModuleImportDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ModuleImportDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ModuleImportDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case ParameterDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ParameterDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->thisLoc.index());
    }
    case ParameterDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->typeSpecifierList));
    }
    case ParameterDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declarator)));
    }
    case ParameterDeclarationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case ParameterDeclarationASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ParameterDeclarationASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->type)));
    }
    case ParameterDeclarationASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case ParameterDeclarationASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case ParameterDeclarationASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isThisIntroduced);
    }
    case ParameterDeclarationASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::ParameterDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isPack);
    }
    case AccessDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AccessDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->accessLoc.index());
    }
    case AccessDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AccessDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case AccessDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AccessDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->accessSpecifier);
    }
    case StructuredBindingDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case StructuredBindingDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declSpecifierList));
    }
    case StructuredBindingDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->refQualifierLoc.index());
    }
    case StructuredBindingDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case StructuredBindingDeclarationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->bindingList));
    }
    case StructuredBindingDeclarationASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case StructuredBindingDeclarationASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case StructuredBindingDeclarationASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case StructuredBindingDeclarationASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->hiddenVariable)));
    }
    case StructuredBindingDeclarationASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::StructuredBindingDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->bindingDeclaratorList));
    }
    case AsmOperandASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AsmOperandAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case AsmOperandASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AsmOperandAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->symbolicNameLoc.index());
    }
    case AsmOperandASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AsmOperandAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case AsmOperandASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::AsmOperandAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->constraintLiteralLoc.index());
    }
    case AsmOperandASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::AsmOperandAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case AsmOperandASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::AsmOperandAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case AsmOperandASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::AsmOperandAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case AsmOperandASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::AsmOperandAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->symbolicName)));
    }
    case AsmOperandASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::AsmOperandAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->constraintLiteral)));
    }
    case AsmQualifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AsmQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->qualifierLoc.index());
    }
    case AsmQualifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AsmQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->qualifier);
    }
    case AsmClobberASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AsmClobberAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case AsmClobberASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AsmClobberAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case AsmGotoLabelASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AsmGotoLabelAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case AsmGotoLabelASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AsmGotoLabelAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case SplicerASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SplicerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case SplicerASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SplicerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case SplicerASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SplicerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case SplicerASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SplicerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case SplicerASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::SplicerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->secondColonLoc.index());
    }
    case SplicerASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::SplicerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case GlobalModuleFragmentASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::GlobalModuleFragmentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->moduleLoc.index());
    }
    case GlobalModuleFragmentASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::GlobalModuleFragmentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case GlobalModuleFragmentASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::GlobalModuleFragmentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declarationList));
    }
    case PrivateModuleFragmentASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::PrivateModuleFragmentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->moduleLoc.index());
    }
    case PrivateModuleFragmentASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::PrivateModuleFragmentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case PrivateModuleFragmentASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::PrivateModuleFragmentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->privateLoc.index());
    }
    case PrivateModuleFragmentASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::PrivateModuleFragmentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case PrivateModuleFragmentASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::PrivateModuleFragmentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declarationList));
    }
    case ModuleDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ModuleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->exportLoc.index());
    }
    case ModuleDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ModuleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->moduleLoc.index());
    }
    case ModuleDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ModuleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->moduleName)));
    }
    case ModuleDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ModuleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->modulePartition)));
    }
    case ModuleDeclarationASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ModuleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ModuleDeclarationASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ModuleDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case ModuleNameASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ModuleNameAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->moduleQualifier)));
    }
    case ModuleNameASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ModuleNameAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case ModuleNameASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ModuleNameAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case ModuleQualifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ModuleQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->moduleQualifier)));
    }
    case ModuleQualifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ModuleQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case ModuleQualifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ModuleQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->dotLoc.index());
    }
    case ModuleQualifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ModuleQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case ModulePartitionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ModulePartitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case ModulePartitionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ModulePartitionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->moduleName)));
    }
    case ImportNameASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ImportNameAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->headerLoc.index());
    }
    case ImportNameASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ImportNameAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->modulePartition)));
    }
    case ImportNameASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ImportNameAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->moduleName)));
    }
    case InitDeclaratorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::InitDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declarator)));
    }
    case InitDeclaratorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::InitDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->requiresClause)));
    }
    case InitDeclaratorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::InitDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case InitDeclaratorASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::InitDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case DeclaratorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->ptrOpList));
    }
    case DeclaratorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->coreDeclarator)));
    }
    case DeclaratorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declaratorChunkList));
    }
    case UsingDeclaratorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::UsingDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->typenameLoc.index());
    }
    case UsingDeclaratorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::UsingDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case UsingDeclaratorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::UsingDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case UsingDeclaratorASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::UsingDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case UsingDeclaratorASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::UsingDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case UsingDeclaratorASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::UsingDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isPack);
    }
    case EnumeratorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::EnumeratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case EnumeratorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::EnumeratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case EnumeratorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::EnumeratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case EnumeratorASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::EnumeratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case EnumeratorASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::EnumeratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case EnumeratorASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::EnumeratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case TypeIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->typeSpecifierList));
    }
    case TypeIdASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case TypeIdASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declarator)));
    }
    case TypeIdASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypeIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->type)));
    }
    case HandlerASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::HandlerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->catchLoc.index());
    }
    case HandlerASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::HandlerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case HandlerASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::HandlerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->exceptionDeclaration)));
    }
    case HandlerASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::HandlerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case HandlerASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::HandlerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case HandlerASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::HandlerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case BaseSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case BaseSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->virtualOrAccessLoc.index());
    }
    case BaseSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->otherVirtualOrAccessLoc.index());
    }
    case BaseSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case BaseSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case BaseSpecifierASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case BaseSpecifierASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case BaseSpecifierASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case BaseSpecifierASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVirtual);
    }
    case BaseSpecifierASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVariadic);
    }
    case BaseSpecifierASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->accessSpecifier);
    }
    case BaseSpecifierASTSlotBase + 11: {
      auto self = static_cast<const ::cxx::BaseSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case RequiresClauseASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::RequiresClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->requiresLoc.index());
    }
    case RequiresClauseASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::RequiresClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ParameterDeclarationClauseASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ParameterDeclarationClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->parameterDeclarationList));
    }
    case ParameterDeclarationClauseASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ParameterDeclarationClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->commaLoc.index());
    }
    case ParameterDeclarationClauseASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ParameterDeclarationClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case ParameterDeclarationClauseASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ParameterDeclarationClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->functionParametersSymbol)));
    }
    case ParameterDeclarationClauseASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ParameterDeclarationClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVariadic);
    }
    case TrailingReturnTypeASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TrailingReturnTypeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->minusGreaterLoc.index());
    }
    case TrailingReturnTypeASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TrailingReturnTypeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case LambdaSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::LambdaSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifierLoc.index());
    }
    case LambdaSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::LambdaSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifier);
    }
    case TypeConstraintASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeConstraintAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case TypeConstraintASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeConstraintAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case TypeConstraintASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeConstraintAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lessLoc.index());
    }
    case TypeConstraintASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypeConstraintAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->templateArgumentList));
    }
    case TypeConstraintASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TypeConstraintAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->greaterLoc.index());
    }
    case TypeConstraintASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::TypeConstraintAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case TypeConstraintASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::TypeConstraintAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case AttributeArgumentClauseASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AttributeArgumentClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case AttributeArgumentClauseASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AttributeArgumentClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->expressionList));
    }
    case AttributeArgumentClauseASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AttributeArgumentClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case AttributeASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->attributeToken)));
    }
    case AttributeASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->attributeArgumentClause)));
    }
    case AttributeASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case AttributeUsingPrefixASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AttributeUsingPrefixAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->usingLoc.index());
    }
    case AttributeUsingPrefixASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AttributeUsingPrefixAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->attributeNamespaceLoc.index());
    }
    case AttributeUsingPrefixASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AttributeUsingPrefixAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case NewPlacementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NewPlacementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case NewPlacementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NewPlacementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->expressionList));
    }
    case NewPlacementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NewPlacementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case NestedNamespaceSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NestedNamespaceSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->inlineLoc.index());
    }
    case NestedNamespaceSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NestedNamespaceSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case NestedNamespaceSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NestedNamespaceSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->scopeLoc.index());
    }
    case NestedNamespaceSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::NestedNamespaceSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case NestedNamespaceSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::NestedNamespaceSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case NestedNamespaceSpecifierASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::NestedNamespaceSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isInline);
    }
    case LabeledStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::LabeledStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case LabeledStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::LabeledStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case LabeledStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::LabeledStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case LabeledStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::LabeledStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case CaseStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CaseStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->caseLoc.index());
    }
    case CaseStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CaseStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case CaseStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CaseStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case DefaultStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DefaultStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->defaultLoc.index());
    }
    case DefaultStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DefaultStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case ExpressionStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ExpressionStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ExpressionStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ExpressionStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ExpressionStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ExpressionStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case CompoundStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CompoundStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case CompoundStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CompoundStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbraceLoc.index());
    }
    case CompoundStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CompoundStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->statementList));
    }
    case CompoundStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::CompoundStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbraceLoc.index());
    }
    case CompoundStatementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::CompoundStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case IfStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case IfStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ifLoc.index());
    }
    case IfStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->constexprLoc.index());
    }
    case IfStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case IfStatementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case IfStatementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->condition)));
    }
    case IfStatementASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case IfStatementASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case IfStatementASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->elseLoc.index());
    }
    case IfStatementASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->elseStatement)));
    }
    case IfStatementASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::IfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case ConstevalIfStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConstevalIfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ConstevalIfStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ConstevalIfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ifLoc.index());
    }
    case ConstevalIfStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ConstevalIfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->exclaimLoc.index());
    }
    case ConstevalIfStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ConstevalIfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->constvalLoc.index());
    }
    case ConstevalIfStatementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ConstevalIfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case ConstevalIfStatementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ConstevalIfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->elseLoc.index());
    }
    case ConstevalIfStatementASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::ConstevalIfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->elseStatement)));
    }
    case ConstevalIfStatementASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::ConstevalIfStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isNot);
    }
    case SwitchStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SwitchStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case SwitchStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SwitchStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->switchLoc.index());
    }
    case SwitchStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SwitchStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case SwitchStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SwitchStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case SwitchStatementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::SwitchStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->condition)));
    }
    case SwitchStatementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::SwitchStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case SwitchStatementASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::SwitchStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case SwitchStatementASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::SwitchStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case WhileStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::WhileStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case WhileStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::WhileStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->whileLoc.index());
    }
    case WhileStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::WhileStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case WhileStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::WhileStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->condition)));
    }
    case WhileStatementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::WhileStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case WhileStatementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::WhileStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case WhileStatementASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::WhileStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case DoStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case DoStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->doLoc.index());
    }
    case DoStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case DoStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::DoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->whileLoc.index());
    }
    case DoStatementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::DoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case DoStatementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::DoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case DoStatementASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::DoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case DoStatementASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::DoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case ForRangeStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ForRangeStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->forLoc.index());
    }
    case ForRangeStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case ForRangeStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case ForRangeStatementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->rangeDeclaration)));
    }
    case ForRangeStatementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case ForRangeStatementASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->rangeInitializer)));
    }
    case ForRangeStatementASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case ForRangeStatementASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case ForRangeStatementASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->beginInitializer)));
    }
    case ForRangeStatementASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->endInitializer)));
    }
    case ForRangeStatementASTSlotBase + 11: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->condition)));
    }
    case ForRangeStatementASTSlotBase + 12: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->increment)));
    }
    case ForRangeStatementASTSlotBase + 13: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->element)));
    }
    case ForRangeStatementASTSlotBase + 14: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case ForRangeStatementASTSlotBase + 15: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->rangeVariable)));
    }
    case ForRangeStatementASTSlotBase + 16: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->beginVariable)));
    }
    case ForRangeStatementASTSlotBase + 17: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->endVariable)));
    }
    case ForRangeStatementASTSlotBase + 18: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->beginFunction)));
    }
    case ForRangeStatementASTSlotBase + 19: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->endFunction)));
    }
    case ForRangeStatementASTSlotBase + 20: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->derefFunction)));
    }
    case ForRangeStatementASTSlotBase + 21: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->incrementFunction)));
    }
    case ForRangeStatementASTSlotBase + 22: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->notEqualFunction)));
    }
    case ForRangeStatementASTSlotBase + 23: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->usesMemberBeginEnd);
    }
    case ForRangeStatementASTSlotBase + 24: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isPointerIterator);
    }
    case ForRangeStatementASTSlotBase + 25: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->notEqualRewritten);
    }
    case ForRangeStatementASTSlotBase + 26: {
      auto self = static_cast<const ::cxx::ForRangeStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->notEqualReversed);
    }
    case ForStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ForStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->forLoc.index());
    }
    case ForStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case ForStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case ForStatementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->condition)));
    }
    case ForStatementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case ForStatementASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ForStatementASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case ForStatementASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case ForStatementASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::ForStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case BreakStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BreakStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case BreakStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BreakStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->breakLoc.index());
    }
    case BreakStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BreakStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case ContinueStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ContinueStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ContinueStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ContinueStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->continueLoc.index());
    }
    case ContinueStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ContinueStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case ReturnStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ReturnStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ReturnStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ReturnStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->returnLoc.index());
    }
    case ReturnStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ReturnStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ReturnStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ReturnStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case CoroutineReturnStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CoroutineReturnStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case CoroutineReturnStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CoroutineReturnStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->coreturnLoc.index());
    }
    case CoroutineReturnStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CoroutineReturnStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case CoroutineReturnStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::CoroutineReturnStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case GotoStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::GotoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case GotoStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::GotoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case GotoStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::GotoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->gotoLoc.index());
    }
    case GotoStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::GotoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->starLoc.index());
    }
    case GotoStatementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::GotoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case GotoStatementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::GotoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case GotoStatementASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::GotoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case GotoStatementASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::GotoStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isIndirect);
    }
    case DeclarationStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DeclarationStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration)));
    }
    case TryBlockStatementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TryBlockStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case TryBlockStatementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TryBlockStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->tryLoc.index());
    }
    case TryBlockStatementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TryBlockStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case TryBlockStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TryBlockStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->handlerList));
    }
    case CharLiteralExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CharLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case CharLiteralExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CharLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case CharLiteralExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CharLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->literalOperatorCall)));
    }
    case BoolLiteralExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BoolLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case BoolLiteralExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BoolLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTrue);
    }
    case IntLiteralExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::IntLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case IntLiteralExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::IntLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case IntLiteralExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::IntLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->literalOperatorCall)));
    }
    case FloatLiteralExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::FloatLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case FloatLiteralExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::FloatLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case FloatLiteralExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::FloatLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->literalOperatorCall)));
    }
    case NullptrLiteralExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NullptrLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case NullptrLiteralExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NullptrLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literal);
    }
    case StringLiteralExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::StringLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case StringLiteralExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::StringLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case StringLiteralExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::StringLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->encoding);
    }
    case UserDefinedStringLiteralExpressionASTSlotBase + 0: {
      auto self =
          static_cast<const ::cxx::UserDefinedStringLiteralExpressionAST*>(
              reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case UserDefinedStringLiteralExpressionASTSlotBase + 1: {
      auto self =
          static_cast<const ::cxx::UserDefinedStringLiteralExpressionAST*>(
              reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case UserDefinedStringLiteralExpressionASTSlotBase + 2: {
      auto self =
          static_cast<const ::cxx::UserDefinedStringLiteralExpressionAST*>(
              reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->literalOperatorCall)));
    }
    case UserDefinedStringLiteralExpressionASTSlotBase + 3: {
      auto self =
          static_cast<const ::cxx::UserDefinedStringLiteralExpressionAST*>(
              reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->encoding);
    }
    case ObjectLiteralExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ObjectLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case ObjectLiteralExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ObjectLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case ObjectLiteralExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ObjectLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case ObjectLiteralExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ObjectLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->bracedInitList)));
    }
    case ObjectLiteralExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ObjectLiteralExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case ThisExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ThisExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->thisLoc.index());
    }
    case PackIndexExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::PackIndexExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->packExpression)));
    }
    case PackIndexExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::PackIndexExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case PackIndexExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::PackIndexExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case PackIndexExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::PackIndexExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->indexExpression)));
    }
    case PackIndexExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::PackIndexExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case GenericSelectionExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::GenericSelectionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->genericLoc.index());
    }
    case GenericSelectionExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::GenericSelectionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case GenericSelectionExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::GenericSelectionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case GenericSelectionExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::GenericSelectionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->commaLoc.index());
    }
    case GenericSelectionExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::GenericSelectionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->genericAssociationList));
    }
    case GenericSelectionExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::GenericSelectionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case GenericSelectionExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::GenericSelectionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->matchedAssocIndex);
    }
    case NestedStatementExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NestedStatementExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case NestedStatementExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NestedStatementExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case NestedStatementExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NestedStatementExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case DefaultInitializerExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DefaultInitializerExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case DefaultInitializerExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DefaultInitializerExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(&(self->context)));
    }
    case NestedExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NestedExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case NestedExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NestedExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case NestedExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NestedExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case IdExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::IdExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case IdExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::IdExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case IdExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::IdExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case IdExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::IdExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case IdExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::IdExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case LambdaExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case LambdaExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->captureDefaultLoc.index());
    }
    case LambdaExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->captureList));
    }
    case LambdaExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case LambdaExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lessLoc.index());
    }
    case LambdaExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->templateParameterList));
    }
    case LambdaExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->greaterLoc.index());
    }
    case LambdaExpressionASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->templateRequiresClause)));
    }
    case LambdaExpressionASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->expressionAttributeList));
    }
    case LambdaExpressionASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case LambdaExpressionASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->parameterDeclarationClause)));
    }
    case LambdaExpressionASTSlotBase + 11: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case LambdaExpressionASTSlotBase + 12: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->gnuAtributeList));
    }
    case LambdaExpressionASTSlotBase + 13: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->lambdaSpecifierList));
    }
    case LambdaExpressionASTSlotBase + 14: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->exceptionSpecifier)));
    }
    case LambdaExpressionASTSlotBase + 15: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case LambdaExpressionASTSlotBase + 16: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->trailingReturnType)));
    }
    case LambdaExpressionASTSlotBase + 17: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->requiresClause)));
    }
    case LambdaExpressionASTSlotBase + 18: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case LambdaExpressionASTSlotBase + 19: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->captureDefault);
    }
    case LambdaExpressionASTSlotBase + 20: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case LambdaExpressionASTSlotBase + 21: {
      auto self = static_cast<const ::cxx::LambdaExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructorSymbol)));
    }
    case FoldExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::FoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case FoldExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::FoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->leftExpression)));
    }
    case FoldExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::FoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->opLoc.index());
    }
    case FoldExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::FoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case FoldExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::FoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->foldOpLoc.index());
    }
    case FoldExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::FoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->rightExpression)));
    }
    case FoldExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::FoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case FoldExpressionASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::FoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->op);
    }
    case FoldExpressionASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::FoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->foldOp);
    }
    case RightFoldExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::RightFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case RightFoldExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::RightFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case RightFoldExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::RightFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->opLoc.index());
    }
    case RightFoldExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::RightFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case RightFoldExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::RightFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case RightFoldExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::RightFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->op);
    }
    case LeftFoldExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::LeftFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case LeftFoldExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::LeftFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case LeftFoldExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::LeftFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->opLoc.index());
    }
    case LeftFoldExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::LeftFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case LeftFoldExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::LeftFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case LeftFoldExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::LeftFoldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->op);
    }
    case RequiresExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::RequiresExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->requiresLoc.index());
    }
    case RequiresExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::RequiresExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case RequiresExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::RequiresExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->parameterDeclarationClause)));
    }
    case RequiresExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::RequiresExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case RequiresExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::RequiresExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbraceLoc.index());
    }
    case RequiresExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::RequiresExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->requirementList));
    }
    case RequiresExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::RequiresExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbraceLoc.index());
    }
    case VaArgExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::VaArgExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->vaArgLoc.index());
    }
    case VaArgExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::VaArgExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case VaArgExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::VaArgExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case VaArgExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::VaArgExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->commaLoc.index());
    }
    case VaArgExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::VaArgExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case VaArgExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::VaArgExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case SubscriptExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SubscriptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->baseExpression)));
    }
    case SubscriptExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SubscriptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case SubscriptExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SubscriptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->indexExpression)));
    }
    case SubscriptExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SubscriptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case SubscriptExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::SubscriptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case SubscriptExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::SubscriptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVirtualDispatch);
    }
    case CallExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CallExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->baseExpression)));
    }
    case CallExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CallExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case CallExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CallExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->expressionList));
    }
    case CallExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::CallExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case CallExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::CallExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVirtualDispatch);
    }
    case CallExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::CallExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructorSymbol)));
    }
    case TypeConstructionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeConstructionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeSpecifier)));
    }
    case TypeConstructionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeConstructionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case TypeConstructionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeConstructionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->expressionList));
    }
    case TypeConstructionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypeConstructionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case TypeConstructionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TypeConstructionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructorSymbol)));
    }
    case BracedTypeConstructionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BracedTypeConstructionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeSpecifier)));
    }
    case BracedTypeConstructionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BracedTypeConstructionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->bracedInitList)));
    }
    case BracedTypeConstructionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BracedTypeConstructionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructorSymbol)));
    }
    case SpliceMemberExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SpliceMemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->baseExpression)));
    }
    case SpliceMemberExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SpliceMemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->accessLoc.index());
    }
    case SpliceMemberExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SpliceMemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case SpliceMemberExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SpliceMemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->splicer)));
    }
    case SpliceMemberExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::SpliceMemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case SpliceMemberExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::SpliceMemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->accessOp);
    }
    case SpliceMemberExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::SpliceMemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case MemberExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::MemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->baseExpression)));
    }
    case MemberExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::MemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->accessLoc.index());
    }
    case MemberExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::MemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case MemberExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::MemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case MemberExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::MemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case MemberExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::MemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case MemberExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::MemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->accessOp);
    }
    case MemberExpressionASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::MemberExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case PostIncrExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::PostIncrExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->baseExpression)));
    }
    case PostIncrExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::PostIncrExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->opLoc.index());
    }
    case PostIncrExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::PostIncrExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->op);
    }
    case PostIncrExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::PostIncrExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case PostIncrExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::PostIncrExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVirtualDispatch);
    }
    case CppCastExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CppCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->castLoc.index());
    }
    case CppCastExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CppCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lessLoc.index());
    }
    case CppCastExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CppCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case CppCastExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::CppCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->greaterLoc.index());
    }
    case CppCastExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::CppCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case CppCastExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::CppCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case CppCastExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::CppCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case CppCastExpressionASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::CppCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->castOp);
    }
    case BuiltinBitCastExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BuiltinBitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->castLoc.index());
    }
    case BuiltinBitCastExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BuiltinBitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case BuiltinBitCastExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BuiltinBitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case BuiltinBitCastExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::BuiltinBitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->commaLoc.index());
    }
    case BuiltinBitCastExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::BuiltinBitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case BuiltinBitCastExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::BuiltinBitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case BuiltinOffsetofExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BuiltinOffsetofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->offsetofLoc.index());
    }
    case BuiltinOffsetofExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BuiltinOffsetofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case BuiltinOffsetofExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BuiltinOffsetofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case BuiltinOffsetofExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::BuiltinOffsetofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->commaLoc.index());
    }
    case BuiltinOffsetofExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::BuiltinOffsetofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case BuiltinOffsetofExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::BuiltinOffsetofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->designatorList));
    }
    case BuiltinOffsetofExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::BuiltinOffsetofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case BuiltinOffsetofExpressionASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::BuiltinOffsetofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case BuiltinOffsetofExpressionASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::BuiltinOffsetofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case TypeidExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeidExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->typeidLoc.index());
    }
    case TypeidExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeidExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case TypeidExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeidExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case TypeidExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypeidExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case TypeidOfTypeExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeidOfTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->typeidLoc.index());
    }
    case TypeidOfTypeExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeidOfTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case TypeidOfTypeExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeidOfTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case TypeidOfTypeExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypeidOfTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case SpliceExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SpliceExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->splicer)));
    }
    case GlobalScopeReflectExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::GlobalScopeReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->caretCaretLoc.index());
    }
    case GlobalScopeReflectExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::GlobalScopeReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->scopeLoc.index());
    }
    case NamespaceReflectExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NamespaceReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->caretCaretLoc.index());
    }
    case NamespaceReflectExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NamespaceReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case NamespaceReflectExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NamespaceReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case NamespaceReflectExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::NamespaceReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case TypeIdReflectExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeIdReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->caretCaretLoc.index());
    }
    case TypeIdReflectExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeIdReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case ReflectExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->caretCaretLoc.index());
    }
    case ReflectExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ReflectExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case LabelAddressExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::LabelAddressExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ampAmpLoc.index());
    }
    case LabelAddressExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::LabelAddressExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case LabelAddressExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::LabelAddressExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case UnaryExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->opLoc.index());
    }
    case UnaryExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::UnaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case UnaryExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::UnaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->op);
    }
    case UnaryExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::UnaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case UnaryExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::UnaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVirtualDispatch);
    }
    case AwaitExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AwaitExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->awaitLoc.index());
    }
    case AwaitExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AwaitExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case SizeofExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SizeofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->sizeofLoc.index());
    }
    case SizeofExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SizeofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case SizeofTypeExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SizeofTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->sizeofLoc.index());
    }
    case SizeofTypeExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SizeofTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case SizeofTypeExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SizeofTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case SizeofTypeExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SizeofTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case SizeofPackExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SizeofPackExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->sizeofLoc.index());
    }
    case SizeofPackExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SizeofPackExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case SizeofPackExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SizeofPackExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case SizeofPackExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SizeofPackExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case SizeofPackExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::SizeofPackExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case SizeofPackExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::SizeofPackExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case SizeofPackExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::SizeofPackExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case AlignofTypeExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AlignofTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->alignofLoc.index());
    }
    case AlignofTypeExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AlignofTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case AlignofTypeExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AlignofTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case AlignofTypeExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::AlignofTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case AlignofExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AlignofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->alignofLoc.index());
    }
    case AlignofExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AlignofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case NoexceptExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NoexceptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->noexceptLoc.index());
    }
    case NoexceptExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NoexceptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case NoexceptExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NoexceptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case NoexceptExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::NoexceptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case NewExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->scopeLoc.index());
    }
    case NewExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->newLoc.index());
    }
    case NewExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->newPlacement)));
    }
    case NewExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case NewExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->typeSpecifierList));
    }
    case NewExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declarator)));
    }
    case NewExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case NewExpressionASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->newInitalizer)));
    }
    case NewExpressionASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->objectType)));
    }
    case NewExpressionASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructorSymbol)));
    }
    case NewExpressionASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::NewExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case DeleteExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DeleteExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->scopeLoc.index());
    }
    case DeleteExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DeleteExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->deleteLoc.index());
    }
    case DeleteExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DeleteExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case DeleteExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::DeleteExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case DeleteExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::DeleteExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case DeleteExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::DeleteExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case CastExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case CastExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case CastExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case CastExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::CastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ImplicitCastExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ImplicitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ImplicitCastExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ImplicitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->castKind);
    }
    case ImplicitCastExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ImplicitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->conversionFunction)));
    }
    case ImplicitCastExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ImplicitCastExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVirtualDispatch);
    }
    case ConstExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConstExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case BinaryExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BinaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->leftExpression)));
    }
    case BinaryExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BinaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->opLoc.index());
    }
    case BinaryExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BinaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->rightExpression)));
    }
    case BinaryExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::BinaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->op);
    }
    case BinaryExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::BinaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case BinaryExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::BinaryExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVirtualDispatch);
    }
    case ConditionalExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConditionalExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->condition)));
    }
    case ConditionalExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ConditionalExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->questionLoc.index());
    }
    case ConditionalExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ConditionalExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->iftrueExpression)));
    }
    case ConditionalExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ConditionalExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case ConditionalExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ConditionalExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->iffalseExpression)));
    }
    case YieldExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::YieldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->yieldLoc.index());
    }
    case YieldExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::YieldExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ThrowExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ThrowExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->throwLoc.index());
    }
    case ThrowExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ThrowExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case AssignmentExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->leftExpression)));
    }
    case AssignmentExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->opLoc.index());
    }
    case AssignmentExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->rightExpression)));
    }
    case AssignmentExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::AssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->op);
    }
    case AssignmentExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::AssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case AssignmentExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::AssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVirtualDispatch);
    }
    case CompoundAssignmentExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CompoundAssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->targetExpression)));
    }
    case CompoundAssignmentExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CompoundAssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->opLoc.index());
    }
    case CompoundAssignmentExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CompoundAssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->leftExpression)));
    }
    case CompoundAssignmentExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::CompoundAssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->rightExpression)));
    }
    case CompoundAssignmentExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::CompoundAssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->adjustExpression)));
    }
    case CompoundAssignmentExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::CompoundAssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->op);
    }
    case CompoundAssignmentExpressionASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::CompoundAssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case CompoundAssignmentExpressionASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::CompoundAssignmentExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isVirtualDispatch);
    }
    case PackExpansionExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::PackExpansionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case PackExpansionExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::PackExpansionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case DesignatedInitializerClauseASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DesignatedInitializerClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->designatorList));
    }
    case DesignatedInitializerClauseASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DesignatedInitializerClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case DesignatedInitializerClauseASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DesignatedInitializerClauseAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructorSymbol)));
    }
    case TypeTraitExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeTraitExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->typeTraitLoc.index());
    }
    case TypeTraitExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeTraitExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case TypeTraitExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeTraitExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->typeIdList));
    }
    case TypeTraitExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypeTraitExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case TypeTraitExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TypeTraitExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->typeTrait);
    }
    case ConditionExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConditionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ConditionExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ConditionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declSpecifierList));
    }
    case ConditionExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ConditionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declarator)));
    }
    case ConditionExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ConditionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case ConditionExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ConditionExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case EqualInitializerASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::EqualInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case EqualInitializerASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::EqualInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case BracedInitListASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BracedInitListAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbraceLoc.index());
    }
    case BracedInitListASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BracedInitListAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->expressionList));
    }
    case BracedInitListASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BracedInitListAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->commaLoc.index());
    }
    case BracedInitListASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::BracedInitListAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbraceLoc.index());
    }
    case ParenInitializerASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ParenInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case ParenInitializerASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ParenInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->expressionList));
    }
    case ParenInitializerASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ParenInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case ThreeWayComparisonExpressionASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ThreeWayComparisonExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->comparison)));
    }
    case ThreeWayComparisonExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ThreeWayComparisonExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->lessResult)));
    }
    case ThreeWayComparisonExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ThreeWayComparisonExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->equalResult)));
    }
    case ThreeWayComparisonExpressionASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ThreeWayComparisonExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->greaterResult)));
    }
    case ThreeWayComparisonExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ThreeWayComparisonExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->unorderedResult)));
    }
    case DefaultGenericAssociationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DefaultGenericAssociationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->defaultLoc.index());
    }
    case DefaultGenericAssociationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DefaultGenericAssociationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case DefaultGenericAssociationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DefaultGenericAssociationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case TypeGenericAssociationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeGenericAssociationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case TypeGenericAssociationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeGenericAssociationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case TypeGenericAssociationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeGenericAssociationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case DotDesignatorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DotDesignatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->dotLoc.index());
    }
    case DotDesignatorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DotDesignatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case DotDesignatorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DotDesignatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case DotDesignatorASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::DotDesignatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case SubscriptDesignatorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SubscriptDesignatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case SubscriptDesignatorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SubscriptDesignatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case SubscriptDesignatorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SubscriptDesignatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case TemplateTypeParameterASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case TemplateTypeParameterASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lessLoc.index());
    }
    case TemplateTypeParameterASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->templateParameterList));
    }
    case TemplateTypeParameterASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->greaterLoc.index());
    }
    case TemplateTypeParameterASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->requiresClause)));
    }
    case TemplateTypeParameterASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->classKeyLoc.index());
    }
    case TemplateTypeParameterASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case TemplateTypeParameterASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case TemplateTypeParameterASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case TemplateTypeParameterASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->idExpression)));
    }
    case TemplateTypeParameterASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case TemplateTypeParameterASTSlotBase + 11: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isPack);
    }
    case NonTypeTemplateParameterASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NonTypeTemplateParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration)));
    }
    case TypenameTypeParameterASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypenameTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->classKeyLoc.index());
    }
    case TypenameTypeParameterASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypenameTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case TypenameTypeParameterASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypenameTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case TypenameTypeParameterASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypenameTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case TypenameTypeParameterASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TypenameTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case TypenameTypeParameterASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::TypenameTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case TypenameTypeParameterASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::TypenameTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isPack);
    }
    case ConstraintTypeParameterASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeConstraint)));
    }
    case ConstraintTypeParameterASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case ConstraintTypeParameterASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case ConstraintTypeParameterASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case ConstraintTypeParameterASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case ConstraintTypeParameterASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case TypedefSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypedefSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->typedefLoc.index());
    }
    case FriendSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::FriendSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->friendLoc.index());
    }
    case ConstevalSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConstevalSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->constevalLoc.index());
    }
    case ConstinitSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConstinitSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->constinitLoc.index());
    }
    case ConstexprSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConstexprSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->constexprLoc.index());
    }
    case InlineSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::InlineSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->inlineLoc.index());
    }
    case NoreturnSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NoreturnSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->noreturnLoc.index());
    }
    case StaticSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::StaticSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->staticLoc.index());
    }
    case ExternSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ExternSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->externLoc.index());
    }
    case RegisterSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::RegisterSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->registerLoc.index());
    }
    case ThreadLocalSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ThreadLocalSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->threadLocalLoc.index());
    }
    case ThreadSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ThreadSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->threadLoc.index());
    }
    case MutableSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::MutableSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->mutableLoc.index());
    }
    case VirtualSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::VirtualSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->virtualLoc.index());
    }
    case ExplicitSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ExplicitSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->explicitLoc.index());
    }
    case ExplicitSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ExplicitSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case ExplicitSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ExplicitSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ExplicitSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ExplicitSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case AutoTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AutoTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->autoLoc.index());
    }
    case VoidTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::VoidTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->voidLoc.index());
    }
    case SizeTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SizeTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifierLoc.index());
    }
    case SizeTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SizeTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifier);
    }
    case SignTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SignTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifierLoc.index());
    }
    case SignTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SignTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifier);
    }
    case BuiltinTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifierLoc.index());
    }
    case BuiltinTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifier);
    }
    case UnaryBuiltinTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->builtinLoc.index());
    }
    case UnaryBuiltinTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::UnaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case UnaryBuiltinTypeSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::UnaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case UnaryBuiltinTypeSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::UnaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case UnaryBuiltinTypeSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::UnaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->builtinKind);
    }
    case BinaryBuiltinTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BinaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->builtinLoc.index());
    }
    case BinaryBuiltinTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BinaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case BinaryBuiltinTypeSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BinaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->leftTypeId)));
    }
    case BinaryBuiltinTypeSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::BinaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->commaLoc.index());
    }
    case BinaryBuiltinTypeSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::BinaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->rightTypeId)));
    }
    case BinaryBuiltinTypeSpecifierASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::BinaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case BinaryBuiltinTypeSpecifierASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::BinaryBuiltinTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->builtinKind);
    }
    case IntegralTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::IntegralTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifierLoc.index());
    }
    case IntegralTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::IntegralTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifier);
    }
    case FloatingPointTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::FloatingPointTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifierLoc.index());
    }
    case FloatingPointTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::FloatingPointTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->specifier);
    }
    case ComplexTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ComplexTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->complexLoc.index());
    }
    case NamedTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NamedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case NamedTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NamedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case NamedTypeSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NamedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case NamedTypeSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::NamedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case NamedTypeSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::NamedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case AtomicTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AtomicTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->atomicLoc.index());
    }
    case AtomicTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AtomicTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case AtomicTypeSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AtomicTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case AtomicTypeSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::AtomicTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case BitIntTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BitIntTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->bitintLoc.index());
    }
    case BitIntTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BitIntTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case BitIntTypeSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BitIntTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->sizeExpression)));
    }
    case BitIntTypeSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::BitIntTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case BitIntTypeSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::BitIntTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->bitCount);
    }
    case UnderlyingTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnderlyingTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->underlyingTypeLoc.index());
    }
    case UnderlyingTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::UnderlyingTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case UnderlyingTypeSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::UnderlyingTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case UnderlyingTypeSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::UnderlyingTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case ElaboratedTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ElaboratedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->classLoc.index());
    }
    case ElaboratedTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ElaboratedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ElaboratedTypeSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ElaboratedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case ElaboratedTypeSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ElaboratedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case ElaboratedTypeSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ElaboratedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case ElaboratedTypeSpecifierASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ElaboratedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->classKey);
    }
    case ElaboratedTypeSpecifierASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::ElaboratedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case ElaboratedTypeSpecifierASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::ElaboratedTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case DecltypeAutoSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DecltypeAutoSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->decltypeLoc.index());
    }
    case DecltypeAutoSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DecltypeAutoSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case DecltypeAutoSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DecltypeAutoSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->autoLoc.index());
    }
    case DecltypeAutoSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::DecltypeAutoSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case DecltypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DecltypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->decltypeLoc.index());
    }
    case DecltypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DecltypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case DecltypeSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DecltypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case DecltypeSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::DecltypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case DecltypeSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::DecltypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->type)));
    }
    case PlaceholderTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::PlaceholderTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeConstraint)));
    }
    case PlaceholderTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::PlaceholderTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->specifier)));
    }
    case ConstQualifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConstQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->constLoc.index());
    }
    case VolatileQualifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::VolatileQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->volatileLoc.index());
    }
    case AtomicQualifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AtomicQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->atomicLoc.index());
    }
    case RestrictQualifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::RestrictQualifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->restrictLoc.index());
    }
    case EnumSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->enumLoc.index());
    }
    case EnumSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->classLoc.index());
    }
    case EnumSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case EnumSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case EnumSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case EnumSpecifierASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case EnumSpecifierASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->typeSpecifierList));
    }
    case EnumSpecifierASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbraceLoc.index());
    }
    case EnumSpecifierASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->enumeratorList));
    }
    case EnumSpecifierASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->commaLoc.index());
    }
    case EnumSpecifierASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbraceLoc.index());
    }
    case EnumSpecifierASTSlotBase + 11: {
      auto self = static_cast<const ::cxx::EnumSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case ClassSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->classLoc.index());
    }
    case ClassSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ClassSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case ClassSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case ClassSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->finalLoc.index());
    }
    case ClassSpecifierASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case ClassSpecifierASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->baseSpecifierList));
    }
    case ClassSpecifierASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbraceLoc.index());
    }
    case ClassSpecifierASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->declarationList));
    }
    case ClassSpecifierASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbraceLoc.index());
    }
    case ClassSpecifierASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->classKey);
    }
    case ClassSpecifierASTSlotBase + 11: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case ClassSpecifierASTSlotBase + 12: {
      auto self = static_cast<const ::cxx::ClassSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isFinal);
    }
    case TypenameSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypenameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->typenameLoc.index());
    }
    case TypenameSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypenameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case TypenameSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypenameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case TypenameSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypenameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case TypenameSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TypenameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case TypenameSpecifierASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::TypenameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case SplicerTypeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SplicerTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->typenameLoc.index());
    }
    case SplicerTypeSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SplicerTypeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->splicer)));
    }
    case PointerOperatorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::PointerOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->starLoc.index());
    }
    case PointerOperatorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::PointerOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case PointerOperatorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::PointerOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->cvQualifierList));
    }
    case ReferenceOperatorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ReferenceOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->refLoc.index());
    }
    case ReferenceOperatorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ReferenceOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case ReferenceOperatorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ReferenceOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->refOp);
    }
    case PtrToMemberOperatorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::PtrToMemberOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case PtrToMemberOperatorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::PtrToMemberOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->starLoc.index());
    }
    case PtrToMemberOperatorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::PtrToMemberOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case PtrToMemberOperatorASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::PtrToMemberOperatorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->cvQualifierList));
    }
    case BitfieldDeclaratorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BitfieldDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case BitfieldDeclaratorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BitfieldDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case BitfieldDeclaratorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BitfieldDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->sizeExpression)));
    }
    case ParameterPackASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ParameterPackAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case ParameterPackASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ParameterPackAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->coreDeclarator)));
    }
    case IdDeclaratorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::IdDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case IdDeclaratorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::IdDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case IdDeclaratorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::IdDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case IdDeclaratorASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::IdDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case IdDeclaratorASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::IdDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case NestedDeclaratorASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NestedDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case NestedDeclaratorASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NestedDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declarator)));
    }
    case NestedDeclaratorASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NestedDeclaratorAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case FunctionDeclaratorChunkASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case FunctionDeclaratorChunkASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->parameterDeclarationClause)));
    }
    case FunctionDeclaratorChunkASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case FunctionDeclaratorChunkASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->cvQualifierList));
    }
    case FunctionDeclaratorChunkASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->refLoc.index());
    }
    case FunctionDeclaratorChunkASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->exceptionSpecifier)));
    }
    case FunctionDeclaratorChunkASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case FunctionDeclaratorChunkASTSlotBase + 7: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->trailingReturnType)));
    }
    case FunctionDeclaratorChunkASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->refOp);
    }
    case FunctionDeclaratorChunkASTSlotBase + 9: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isFinal);
    }
    case FunctionDeclaratorChunkASTSlotBase + 10: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isOverride);
    }
    case FunctionDeclaratorChunkASTSlotBase + 11: {
      auto self = static_cast<const ::cxx::FunctionDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isPure);
    }
    case ArrayDeclaratorChunkASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ArrayDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case ArrayDeclaratorChunkASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ArrayDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->typeQualifierList));
    }
    case ArrayDeclaratorChunkASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ArrayDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ArrayDeclaratorChunkASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ArrayDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case ArrayDeclaratorChunkASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ArrayDeclaratorChunkAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case NameIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NameIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case NameIdASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NameIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case DestructorIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DestructorIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->tildeLoc.index());
    }
    case DestructorIdASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DestructorIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->id)));
    }
    case DecltypeIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DecltypeIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->decltypeSpecifier)));
    }
    case OperatorFunctionIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::OperatorFunctionIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->operatorLoc.index());
    }
    case OperatorFunctionIdASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::OperatorFunctionIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->opLoc.index());
    }
    case OperatorFunctionIdASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::OperatorFunctionIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->openLoc.index());
    }
    case OperatorFunctionIdASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::OperatorFunctionIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->closeLoc.index());
    }
    case OperatorFunctionIdASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::OperatorFunctionIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->op);
    }
    case LiteralOperatorIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::LiteralOperatorIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->operatorLoc.index());
    }
    case LiteralOperatorIdASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::LiteralOperatorIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case LiteralOperatorIdASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::LiteralOperatorIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case LiteralOperatorIdASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::LiteralOperatorIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case LiteralOperatorIdASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::LiteralOperatorIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case ConversionFunctionIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConversionFunctionIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->operatorLoc.index());
    }
    case ConversionFunctionIdASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ConversionFunctionIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case SimpleTemplateIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SimpleTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case SimpleTemplateIdASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SimpleTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lessLoc.index());
    }
    case SimpleTemplateIdASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SimpleTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->templateArgumentList));
    }
    case SimpleTemplateIdASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SimpleTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->greaterLoc.index());
    }
    case SimpleTemplateIdASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::SimpleTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case SimpleTemplateIdASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::SimpleTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case LiteralOperatorTemplateIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::LiteralOperatorTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->literalOperatorId)));
    }
    case LiteralOperatorTemplateIdASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::LiteralOperatorTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lessLoc.index());
    }
    case LiteralOperatorTemplateIdASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::LiteralOperatorTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->templateArgumentList));
    }
    case LiteralOperatorTemplateIdASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::LiteralOperatorTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->greaterLoc.index());
    }
    case OperatorFunctionTemplateIdASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::OperatorFunctionTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->operatorFunctionId)));
    }
    case OperatorFunctionTemplateIdASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::OperatorFunctionTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lessLoc.index());
    }
    case OperatorFunctionTemplateIdASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::OperatorFunctionTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->templateArgumentList));
    }
    case OperatorFunctionTemplateIdASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::OperatorFunctionTemplateIdAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->greaterLoc.index());
    }
    case GlobalNestedNameSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::GlobalNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->scopeLoc.index());
    }
    case SimpleNestedNameSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SimpleNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case SimpleNestedNameSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SimpleNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case SimpleNestedNameSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SimpleNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case SimpleNestedNameSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SimpleNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->scopeLoc.index());
    }
    case DecltypeNestedNameSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DecltypeNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->decltypeSpecifier)));
    }
    case DecltypeNestedNameSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DecltypeNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->scopeLoc.index());
    }
    case TemplateNestedNameSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TemplateNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case TemplateNestedNameSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TemplateNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case TemplateNestedNameSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TemplateNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->templateId)));
    }
    case TemplateNestedNameSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TemplateNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->scopeLoc.index());
    }
    case TemplateNestedNameSpecifierASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TemplateNestedNameSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case DefaultFunctionBodyASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DefaultFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case DefaultFunctionBodyASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DefaultFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->defaultLoc.index());
    }
    case DefaultFunctionBodyASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DefaultFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case CompoundStatementFunctionBodyASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CompoundStatementFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case CompoundStatementFunctionBodyASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CompoundStatementFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->memInitializerList));
    }
    case CompoundStatementFunctionBodyASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CompoundStatementFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case TryStatementFunctionBodyASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TryStatementFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->tryLoc.index());
    }
    case TryStatementFunctionBodyASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TryStatementFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->colonLoc.index());
    }
    case TryStatementFunctionBodyASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TryStatementFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->memInitializerList));
    }
    case TryStatementFunctionBodyASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TryStatementFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->statement)));
    }
    case TryStatementFunctionBodyASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TryStatementFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->handlerList));
    }
    case DeleteFunctionBodyASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DeleteFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->equalLoc.index());
    }
    case DeleteFunctionBodyASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DeleteFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->deleteLoc.index());
    }
    case DeleteFunctionBodyASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DeleteFunctionBodyAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case TypeTemplateArgumentASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeTemplateArgumentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case ExpressionTemplateArgumentASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ExpressionTemplateArgumentAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case ThrowExceptionSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ThrowExceptionSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->throwLoc.index());
    }
    case ThrowExceptionSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ThrowExceptionSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case ThrowExceptionSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ThrowExceptionSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case NoexceptSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NoexceptSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->noexceptLoc.index());
    }
    case NoexceptSpecifierASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NoexceptSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case NoexceptSpecifierASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NoexceptSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case NoexceptSpecifierASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::NoexceptSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case SimpleRequirementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SimpleRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case SimpleRequirementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SimpleRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case CompoundRequirementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CompoundRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbraceLoc.index());
    }
    case CompoundRequirementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CompoundRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case CompoundRequirementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CompoundRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbraceLoc.index());
    }
    case CompoundRequirementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::CompoundRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->noexceptLoc.index());
    }
    case CompoundRequirementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::CompoundRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->minusGreaterLoc.index());
    }
    case CompoundRequirementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::CompoundRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeConstraint)));
    }
    case CompoundRequirementASTSlotBase + 6: {
      auto self = static_cast<const ::cxx::CompoundRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case TypeRequirementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->typenameLoc.index());
    }
    case TypeRequirementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case TypeRequirementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->templateLoc.index());
    }
    case TypeRequirementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypeRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case TypeRequirementASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::TypeRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case TypeRequirementASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::TypeRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isTemplateIntroduced);
    }
    case NestedRequirementASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NestedRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->requiresLoc.index());
    }
    case NestedRequirementASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NestedRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case NestedRequirementASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NestedRequirementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->semicolonLoc.index());
    }
    case NewParenInitializerASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NewParenInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case NewParenInitializerASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::NewParenInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->expressionList));
    }
    case NewParenInitializerASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::NewParenInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case NewBracedInitializerASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::NewBracedInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->bracedInitList)));
    }
    case ParenMemInitializerASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ParenMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case ParenMemInitializerASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ParenMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case ParenMemInitializerASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ParenMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case ParenMemInitializerASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ParenMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->expressionList));
    }
    case ParenMemInitializerASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ParenMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case ParenMemInitializerASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::ParenMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case BracedMemInitializerASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::BracedMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier)));
    }
    case BracedMemInitializerASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::BracedMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId)));
    }
    case BracedMemInitializerASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::BracedMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->bracedInitList)));
    }
    case BracedMemInitializerASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::BracedMemInitializerAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case ThisLambdaCaptureASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ThisLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->thisLoc.index());
    }
    case ThisLambdaCaptureASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ThisLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case ThisLambdaCaptureASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ThisLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case DerefThisLambdaCaptureASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::DerefThisLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->starLoc.index());
    }
    case DerefThisLambdaCaptureASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::DerefThisLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->thisLoc.index());
    }
    case DerefThisLambdaCaptureASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::DerefThisLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case SimpleLambdaCaptureASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SimpleLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case SimpleLambdaCaptureASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SimpleLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case SimpleLambdaCaptureASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SimpleLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case SimpleLambdaCaptureASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::SimpleLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case SimpleLambdaCaptureASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::SimpleLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case RefLambdaCaptureASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::RefLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ampLoc.index());
    }
    case RefLambdaCaptureASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::RefLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case RefLambdaCaptureASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::RefLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case RefLambdaCaptureASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::RefLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case RefLambdaCaptureASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::RefLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case RefLambdaCaptureASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::RefLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case RefInitLambdaCaptureASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::RefInitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ampLoc.index());
    }
    case RefInitLambdaCaptureASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::RefInitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case RefInitLambdaCaptureASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::RefInitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case RefInitLambdaCaptureASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::RefInitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case RefInitLambdaCaptureASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::RefInitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case RefInitLambdaCaptureASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::RefInitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case InitLambdaCaptureASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::InitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case InitLambdaCaptureASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::InitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case InitLambdaCaptureASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::InitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer)));
    }
    case InitLambdaCaptureASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::InitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case InitLambdaCaptureASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::InitLambdaCaptureAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case EllipsisExceptionDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::EllipsisExceptionDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case TypeExceptionDeclarationASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeExceptionDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case TypeExceptionDeclarationASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeExceptionDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->typeSpecifierList));
    }
    case TypeExceptionDeclarationASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeExceptionDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declarator)));
    }
    case TypeExceptionDeclarationASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypeExceptionDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol)));
    }
    case CxxAttributeASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::CxxAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracketLoc.index());
    }
    case CxxAttributeASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::CxxAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lbracket2Loc.index());
    }
    case CxxAttributeASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::CxxAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->attributeUsingPrefix)));
    }
    case CxxAttributeASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::CxxAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case CxxAttributeASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::CxxAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracketLoc.index());
    }
    case CxxAttributeASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::CxxAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rbracket2Loc.index());
    }
    case GccAttributeASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::GccAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->attributeLoc.index());
    }
    case GccAttributeASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::GccAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case GccAttributeASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::GccAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparen2Loc.index());
    }
    case GccAttributeASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::GccAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->attributeList));
    }
    case GccAttributeASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::GccAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case GccAttributeASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::GccAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparen2Loc.index());
    }
    case AlignasAttributeASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AlignasAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->alignasLoc.index());
    }
    case AlignasAttributeASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AlignasAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case AlignasAttributeASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AlignasAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expression)));
    }
    case AlignasAttributeASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::AlignasAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case AlignasAttributeASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::AlignasAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case AlignasAttributeASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::AlignasAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isPack);
    }
    case AlignasTypeAttributeASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AlignasTypeAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->alignasLoc.index());
    }
    case AlignasTypeAttributeASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AlignasTypeAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case AlignasTypeAttributeASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AlignasTypeAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId)));
    }
    case AlignasTypeAttributeASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::AlignasTypeAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->ellipsisLoc.index());
    }
    case AlignasTypeAttributeASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::AlignasTypeAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case AlignasTypeAttributeASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::AlignasTypeAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->isPack);
    }
    case AsmAttributeASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AsmAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->asmLoc.index());
    }
    case AsmAttributeASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::AsmAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->lparenLoc.index());
    }
    case AsmAttributeASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::AsmAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->literalLoc.index());
    }
    case AsmAttributeASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::AsmAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->rparenLoc.index());
    }
    case AsmAttributeASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::AsmAttributeAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->literal)));
    }
    case ScopedAttributeTokenASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::ScopedAttributeTokenAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->attributeNamespaceLoc.index());
    }
    case ScopedAttributeTokenASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ScopedAttributeTokenAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->scopeLoc.index());
    }
    case ScopedAttributeTokenASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::ScopedAttributeTokenAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case ScopedAttributeTokenASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::ScopedAttributeTokenAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->attributeNamespace)));
    }
    case ScopedAttributeTokenASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::ScopedAttributeTokenAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
    case SimpleAttributeTokenASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::SimpleAttributeTokenAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(self->identifierLoc.index());
    }
    case SimpleAttributeTokenASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::SimpleAttributeTokenAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->identifier)));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readASTBigInt(std::intptr_t handle, int slot) -> std::int64_t {
  switch (slot) {
    case CaseStatementASTSlotBase + 3: {
      auto self = static_cast<const ::cxx::CaseStatementAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return static_cast<std::int64_t>(self->caseValue);
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readASTVal(std::intptr_t handle, int slot) -> val {
  switch (slot) {
    case AttributeSpecifierASTSlotBase + 0: {
      auto self = static_cast<const ::cxx::AttributeSpecifierAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return optionalValue(self->attributes, [&](const auto& item) {
        return arrayValue(item, [&](const auto& item) {
          return [&]() -> val {
            auto result = val::object();
            result.set("attributeNamespace",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           static_cast<const ::cxx::Name*>(
                               (item).attributeNamespace)))));
            result.set("name",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           static_cast<const ::cxx::Name*>((item).name)))));
            result.set("arguments",
                       arrayValue((item).arguments, [&](const auto& item) {
                         return val(static_cast<double>(
                             reinterpret_cast<std::intptr_t>(
                                 static_cast<const ::cxx::Name*>(item))));
                       }));
            return result;
          }();
        });
      });
    }
    case StaticAssertDeclarationASTSlotBase + 8: {
      auto self = static_cast<const ::cxx::StaticAssertDeclarationAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return optionalValue(self->value, [&](const auto& item) {
        return val(static_cast<double>(item));
      });
    }
    case SizeofExpressionASTSlotBase + 2: {
      auto self = static_cast<const ::cxx::SizeofExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return optionalValue(self->value, [&](const auto& item) {
        return val(static_cast<std::int64_t>(item));
      });
    }
    case SizeofTypeExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::SizeofTypeExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return optionalValue(self->value, [&](const auto& item) {
        return val(static_cast<std::int64_t>(item));
      });
    }
    case NoexceptExpressionASTSlotBase + 4: {
      auto self = static_cast<const ::cxx::NoexceptExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return optionalValue(self->value, [&](const auto& item) {
        return val(static_cast<double>(item));
      });
    }
    case ConstExpressionASTSlotBase + 1: {
      auto self = static_cast<const ::cxx::ConstExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return optionalValue(self->constValue, [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set("value", val(std::get<0>(item).toString()));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Literal*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", val(static_cast<double>(std::get<2>(item))));
              break;
            case 3:
              result.set("value", val(static_cast<double>(std::get<3>(item))));
              break;
            case 4:
              result.set("value", val(static_cast<double>(std::get<4>(item))));
              break;
            case 5:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<5>(item).get()))));
              break;
            case 6:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<6>(item).get()))));
              break;
            case 7:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<7>(item).get()))));
              break;
            case 8:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<8>(item).get()))));
              break;
            case 9:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<9>(item).get()))));
              break;
            case 10:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<10>(item).get()))));
              break;
            case 11:
              result.set("value", val::undefined());
              break;
          }
          return result;
        }();
      });
    }
    case TypeTraitExpressionASTSlotBase + 5: {
      auto self = static_cast<const ::cxx::TypeTraitExpressionAST*>(
          reinterpret_cast<const ::cxx::AST*>(handle));
      return optionalValue(self->value, [&](const auto& item) {
        return val(static_cast<double>(item));
      });
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readSymbol(std::intptr_t handle, int slot) -> double {
  switch (slot) {
    case SymbolSlotBase + 0: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->name())));
    }
    case SymbolSlotBase + 1: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->type())));
    }
    case SymbolSlotBase + 2: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->location().index());
    }
    case SymbolSlotBase + 3: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->parent())));
    }
    case SymbolSlotBase + 4: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->enclosingNamespace())));
    }
    case SymbolSlotBase + 5: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->enclosingClass())));
    }
    case SymbolSlotBase + 6: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->enclosingFunction())));
    }
    case SymbolSlotBase + 7: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->enclosingFunctionOrSelf())));
    }
    case SymbolSlotBase + 8: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->next())));
    }
    case SymbolSlotBase + 9: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isHidden());
    }
    case SymbolSlotBase + 10: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->accessSpecifier());
    }
    case SymbolSlotBase + 14: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isNodiscard());
    }
    case SymbolSlotBase + 15: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isUsed());
    }
    case SymbolSlotBase + 16: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isExcludedFromExplicitInstantiation());
    }
    case SymbolSlotBase + 17: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isTrivialAbi());
    }
    case SymbolSlotBase + 18: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->hasDeducedReturnType());
    }
    case SymbolSlotBase + 19: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->canonical())));
    }
    case SymbolSlotBase + 20: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->definition())));
    }
    case SymbolSlotBase + 21: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isNamespace());
    }
    case SymbolSlotBase + 22: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isNamespaceAlias());
    }
    case SymbolSlotBase + 23: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isConcept());
    }
    case SymbolSlotBase + 24: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isDeductionGuide());
    }
    case SymbolSlotBase + 25: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isClass());
    }
    case SymbolSlotBase + 26: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isEnum());
    }
    case SymbolSlotBase + 27: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isScopedEnum());
    }
    case SymbolSlotBase + 28: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isFunction());
    }
    case SymbolSlotBase + 29: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isTypeAlias());
    }
    case SymbolSlotBase + 30: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isVariable());
    }
    case SymbolSlotBase + 31: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isField());
    }
    case SymbolSlotBase + 32: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isParameter());
    }
    case SymbolSlotBase + 33: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isParameterPack());
    }
    case SymbolSlotBase + 34: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isEnumerator());
    }
    case SymbolSlotBase + 35: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isFunctionParameters());
    }
    case SymbolSlotBase + 36: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isTemplateParameters());
    }
    case SymbolSlotBase + 37: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isBlock());
    }
    case SymbolSlotBase + 38: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isLambda());
    }
    case SymbolSlotBase + 39: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isTypeParameter());
    }
    case SymbolSlotBase + 40: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isNonTypeParameter());
    }
    case SymbolSlotBase + 41: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isTemplateTypeParameter());
    }
    case SymbolSlotBase + 42: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isConstraintTypeParameter());
    }
    case SymbolSlotBase + 43: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isOverloadSet());
    }
    case SymbolSlotBase + 44: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isBaseClass());
    }
    case SymbolSlotBase + 45: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isInjectedClassName());
    }
    case SymbolSlotBase + 46: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isUnresolved());
    }
    case SymbolSlotBase + 47: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isUsingDeclaration());
    }
    case SymbolSlotBase + 48: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isClassOrNamespace());
    }
    case SymbolSlotBase + 49: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isNamespaceName());
    }
    case SymbolSlotBase + 50: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->isEnumOrScopedEnum());
    }
    case SymbolSlotBase + 51: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(self->internalId());
    }
    case SymbolSlotBase + 53: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<double>(
          is_type(const_cast<Symbol*>(static_cast<const Symbol*>(self))));
    }
    case ScopeSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::ScopeSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->empty());
    }
    case ScopeSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::ScopeSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isTransparent());
    }
    case NamespaceSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::NamespaceSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isInline());
    }
    case NamespaceSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::NamespaceSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasInlineNamespaces());
    }
    case NamespaceSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::NamespaceSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->unnamedNamespace())));
    }
    case ConceptSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->templateDeclaration())));
    }
    case ConceptSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->templateParameters())));
    }
    case ConceptSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isSpecialization());
    }
    case ConceptSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isTemplatePattern());
    }
    case ConceptSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration())));
    }
    case ConceptSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->primaryTemplateSymbol())));
    }
    case ConceptSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->templateSpecializationIndex());
    }
    case DeductionGuideSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->templateDeclaration())));
    }
    case DeductionGuideSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->templateParameters())));
    }
    case DeductionGuideSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isSpecialization());
    }
    case DeductionGuideSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isTemplatePattern());
    }
    case DeductionGuideSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration())));
    }
    case DeductionGuideSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->primaryTemplateSymbol())));
    }
    case DeductionGuideSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->templateSpecializationIndex());
    }
    case DeductionGuideSymbolSlotBase + 9: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isExplicit());
    }
    case BaseClassSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::BaseClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isVirtual());
    }
    case BaseClassSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::BaseClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol())));
    }
    case InjectedClassNameSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::InjectedClassNameSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->classSymbol())));
    }
    case ClassSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->canonical())));
    }
    case ClassSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->definition())));
    }
    case ClassSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->templateDeclaration())));
    }
    case ClassSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->templateParameters())));
    }
    case ClassSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isSpecialization());
    }
    case ClassSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isTemplatePattern());
    }
    case ClassSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration())));
    }
    case ClassSymbolSlotBase + 9: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->primaryTemplateSymbol())));
    }
    case ClassSymbolSlotBase + 10: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->templateSpecializationIndex());
    }
    case ClassSymbolSlotBase + 11: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->canonicalOrNull())));
    }
    case ClassSymbolSlotBase + 12: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->resolvedDefinition())));
    }
    case ClassSymbolSlotBase + 15: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->instantiationSubstitutionDepth());
    }
    case ClassSymbolSlotBase + 17: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isUnion());
    }
    case ClassSymbolSlotBase + 21: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructorOverloadSet())));
    }
    case ClassSymbolSlotBase + 26: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->destructor())));
    }
    case ClassSymbolSlotBase + 27: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->defaultConstructor())));
    }
    case ClassSymbolSlotBase + 28: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->copyConstructor())));
    }
    case ClassSymbolSlotBase + 29: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->moveConstructor())));
    }
    case ClassSymbolSlotBase + 30: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->copyAssignmentOperator())));
    }
    case ClassSymbolSlotBase + 31: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->moveAssignmentOperator())));
    }
    case ClassSymbolSlotBase + 32: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasUserDeclaredConstructors());
    }
    case ClassSymbolSlotBase + 33: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasInheritedConstructors());
    }
    case ClassSymbolSlotBase + 34: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasVirtualFunctions());
    }
    case ClassSymbolSlotBase + 35: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasVirtualBaseClasses());
    }
    case ClassSymbolSlotBase + 37: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isFinal());
    }
    case ClassSymbolSlotBase + 38: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isComplete());
    }
    case ClassSymbolSlotBase + 39: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isFriend());
    }
    case ClassSymbolSlotBase + 40: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isPolymorphic());
    }
    case ClassSymbolSlotBase + 41: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isAbstract());
    }
    case ClassSymbolSlotBase + 42: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasVirtualDestructor());
    }
    case ClassSymbolSlotBase + 43: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isAccessControlDisabled());
    }
    case ClassSymbolSlotBase + 44: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->sizeInBytes());
    }
    case ClassSymbolSlotBase + 45: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->alignment());
    }
    case ClassSymbolSlotBase + 46: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->explicitAlignment());
    }
    case ClassSymbolSlotBase + 47: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->packAlignment());
    }
    case ClassSymbolSlotBase + 51: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->flags());
    }
    case ClassSymbolSlotBase + 52: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isClosureType());
    }
    case ClassSymbolSlotBase + 53: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasLambdaCapture());
    }
    case ClassSymbolSlotBase + 54: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->capturedThisField())));
    }
    case ClassSymbolSlotBase + 55: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->closureDiscriminator());
    }
    case ClassSymbolSlotBase + 56: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->instantiationPattern())));
    }
    case ClassSymbolSlotBase + 57: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(static_cast<const ::cxx::Symbol*>(
              class_template_of(const_cast<ClassSymbol*>(self)))));
    }
    case EnumSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::EnumSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasFixedUnderlyingType());
    }
    case EnumSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::EnumSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isDefined());
    }
    case EnumSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::EnumSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->underlyingType())));
    }
    case ScopedEnumSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::ScopedEnumSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->underlyingType())));
    }
    case ScopedEnumSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::ScopedEnumSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isDefined());
    }
    case FunctionSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->canonical())));
    }
    case FunctionSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->definition())));
    }
    case FunctionSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->templateDeclaration())));
    }
    case FunctionSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->templateParameters())));
    }
    case FunctionSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isSpecialization());
    }
    case FunctionSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isTemplatePattern());
    }
    case FunctionSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration())));
    }
    case FunctionSymbolSlotBase + 9: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->primaryTemplateSymbol())));
    }
    case FunctionSymbolSlotBase + 10: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->templateSpecializationIndex());
    }
    case FunctionSymbolSlotBase + 11: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->canonicalOrNull())));
    }
    case FunctionSymbolSlotBase + 12: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->resolvedDefinition())));
    }
    case FunctionSymbolSlotBase + 15: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->functionParameters())));
    }
    case FunctionSymbolSlotBase + 16: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isDefined());
    }
    case FunctionSymbolSlotBase + 17: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isStatic());
    }
    case FunctionSymbolSlotBase + 18: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isExtern());
    }
    case FunctionSymbolSlotBase + 19: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isFriend());
    }
    case FunctionSymbolSlotBase + 20: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isImplicitObjectMemberFunction());
    }
    case FunctionSymbolSlotBase + 21: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasExplicitObjectParameter());
    }
    case FunctionSymbolSlotBase + 22: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->explicitObjectParameter())));
    }
    case FunctionSymbolSlotBase + 24: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isConstexpr());
    }
    case FunctionSymbolSlotBase + 25: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isConsteval());
    }
    case FunctionSymbolSlotBase + 26: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isInline());
    }
    case FunctionSymbolSlotBase + 27: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isVirtual());
    }
    case FunctionSymbolSlotBase + 28: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isExplicit());
    }
    case FunctionSymbolSlotBase + 29: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isDeleted());
    }
    case FunctionSymbolSlotBase + 30: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isDefaulted());
    }
    case FunctionSymbolSlotBase + 31: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isPure());
    }
    case FunctionSymbolSlotBase + 32: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isOverride());
    }
    case FunctionSymbolSlotBase + 33: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isFinal());
    }
    case FunctionSymbolSlotBase + 34: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasNoPrototype());
    }
    case FunctionSymbolSlotBase + 35: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasExceptionSpecifier());
    }
    case FunctionSymbolSlotBase + 36: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isDefinitionRequired());
    }
    case FunctionSymbolSlotBase + 37: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isNoReturn());
    }
    case FunctionSymbolSlotBase + 38: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->builtinKind());
    }
    case FunctionSymbolSlotBase + 39: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->trailingRequiresClause())));
    }
    case FunctionSymbolSlotBase + 40: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isConstructor());
    }
    case FunctionSymbolSlotBase + 41: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isDestructor());
    }
    case FunctionSymbolSlotBase + 42: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->languageLinkage());
    }
    case FunctionSymbolSlotBase + 43: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasCLinkage());
    }
    case FunctionSymbolSlotBase + 44: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->externalName())));
    }
    case FunctionSymbolSlotBase + 45: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->aliasName())));
    }
    case FunctionSymbolSlotBase + 46: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasHiddenVisibility());
    }
    case FunctionSymbolSlotBase + 47: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->importModule())));
    }
    case FunctionSymbolSlotBase + 48: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->importName())));
    }
    case FunctionSymbolSlotBase + 49: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->exportName())));
    }
    case FunctionSymbolSlotBase + 50: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasPendingBody());
    }
    case FunctionSymbolSlotBase + 51: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasUninstantiatedBody());
    }
    case FunctionSymbolSlotBase + 54: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->vtableSlotIndex());
    }
    case FunctionSymbolSlotBase + 58: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->delegatingConstructor())));
    }
    case FunctionSymbolSlotBase + 59: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->completeObjectVariant())));
    }
    case FunctionSymbolSlotBase + 60: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->deletingDtorVariant())));
    }
    case FunctionSymbolSlotBase + 61: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->structorPrincipal())));
    }
    case FunctionSymbolSlotBase + 62: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isStructorVariant());
    }
    case FunctionSymbolSlotBase + 63: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isStructor());
    }
    case FunctionSymbolSlotBase + 64: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasBaseObjectVariant());
    }
    case FunctionSymbolSlotBase + 65: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->inheritedConstructor())));
    }
    case FunctionSymbolSlotBase + 66: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(static_cast<const ::cxx::Symbol*>(
              self->inheritedConstructorOrigin())));
    }
    case FunctionSymbolSlotBase + 67: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isDeletingDtorVariant());
    }
    case FunctionSymbolSlotBase + 68: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->hostScope())));
    }
    case LambdaSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::LambdaSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isConstexpr());
    }
    case LambdaSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::LambdaSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isConsteval());
    }
    case LambdaSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::LambdaSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isMutable());
    }
    case LambdaSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::LambdaSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isStatic());
    }
    case LambdaSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::LambdaSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isTemplate());
    }
    case LambdaSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::LambdaSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isInTemplate());
    }
    case LambdaSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::LambdaSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->closureType())));
    }
    case TemplateParametersSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::TemplateParametersSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isExplicitTemplateSpecialization());
    }
    case BlockSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::BlockSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isOutermostBlockScope());
    }
    case TypeAliasSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->canonical())));
    }
    case TypeAliasSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->definition())));
    }
    case TypeAliasSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->templateDeclaration())));
    }
    case TypeAliasSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->templateParameters())));
    }
    case TypeAliasSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isSpecialization());
    }
    case TypeAliasSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isTemplatePattern());
    }
    case TypeAliasSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration())));
    }
    case TypeAliasSymbolSlotBase + 9: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->primaryTemplateSymbol())));
    }
    case TypeAliasSymbolSlotBase + 10: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->templateSpecializationIndex());
    }
    case TypeAliasSymbolSlotBase + 11: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->canonicalOrNull())));
    }
    case TypeAliasSymbolSlotBase + 12: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->resolvedDefinition())));
    }
    case TypeAliasSymbolSlotBase + 15: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->expansionTypeId())));
    }
    case VariableSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->canonical())));
    }
    case VariableSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->definition())));
    }
    case VariableSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->templateDeclaration())));
    }
    case VariableSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->templateParameters())));
    }
    case VariableSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isSpecialization());
    }
    case VariableSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isTemplatePattern());
    }
    case VariableSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declaration())));
    }
    case VariableSymbolSlotBase + 9: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->primaryTemplateSymbol())));
    }
    case VariableSymbolSlotBase + 10: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->templateSpecializationIndex());
    }
    case VariableSymbolSlotBase + 11: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->canonicalOrNull())));
    }
    case VariableSymbolSlotBase + 12: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->resolvedDefinition())));
    }
    case VariableSymbolSlotBase + 15: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isStatic());
    }
    case VariableSymbolSlotBase + 16: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isThreadLocal());
    }
    case VariableSymbolSlotBase + 17: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isExtern());
    }
    case VariableSymbolSlotBase + 18: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isConstexpr());
    }
    case VariableSymbolSlotBase + 19: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isConstinit());
    }
    case VariableSymbolSlotBase + 20: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isInline());
    }
    case VariableSymbolSlotBase + 21: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer())));
    }
    case VariableSymbolSlotBase + 22: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructor())));
    }
    case VariableSymbolSlotBase + 24: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->explicitAlignment());
    }
    case FieldSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->definition())));
    }
    case FieldSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isBitField());
    }
    case FieldSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->bitFieldOffset());
    }
    case FieldSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isExtern());
    }
    case FieldSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isStatic());
    }
    case FieldSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isThreadLocal());
    }
    case FieldSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isConstexpr());
    }
    case FieldSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isConstinit());
    }
    case FieldSymbolSlotBase + 9: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isInline());
    }
    case FieldSymbolSlotBase + 10: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isMutable());
    }
    case FieldSymbolSlotBase + 11: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isNoUniqueAddress());
    }
    case FieldSymbolSlotBase + 13: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->localOffset());
    }
    case FieldSymbolSlotBase + 14: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->alignment());
    }
    case FieldSymbolSlotBase + 15: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->initializer())));
    }
    case FieldSymbolSlotBase + 16: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->constructor())));
    }
    case FieldSymbolSlotBase + 18: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isDefinitionRequired());
    }
    case FieldSymbolSlotBase + 19: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasPendingInitializer());
    }
    case FieldSymbolSlotBase + 20: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->hasInitializer());
    }
    case ParameterSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::ParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->defaultArgument())));
    }
    case ParameterSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::ParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isExplicitObject());
    }
    case TypeParameterSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->defaultArgument())));
    }
    case NonTypeParameterSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::NonTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isParameterPack());
    }
    case NonTypeParameterSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::NonTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->defaultArgument())));
    }
    case NonTypeParameterSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::NonTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->index());
    }
    case NonTypeParameterSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::NonTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->depth());
    }
    case NonTypeParameterSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::NonTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->objectType())));
    }
    case TemplateTypeParameterSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->defaultArgument())));
    }
    case ConstraintTypeParameterSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->isParameterPack());
    }
    case ConstraintTypeParameterSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->defaultArgument())));
    }
    case ConstraintTypeParameterSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->index());
    }
    case ConstraintTypeParameterSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(self->depth());
    }
    case ConstraintTypeParameterSymbolSlotBase + 4: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeConstraint())));
    }
    case ConstraintTypeParameterSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::ConstraintTypeParameterSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->constraintExpression())));
    }
    case NamespaceAliasSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::NamespaceAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->namespaceSymbol())));
    }
    case UsingDeclarationSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::UsingDeclarationSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->declarator())));
    }
    case UsingDeclarationSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::UsingDeclarationSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->target())));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readSymbolString(std::intptr_t handle, int slot) -> std::string {
  switch (slot) {
    case SymbolSlotBase + 52: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return std::string(to_string(self->name()));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readSymbolVal(std::intptr_t handle, int slot) -> val {
  switch (slot) {
    case SymbolSlotBase + 12: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return optionalValue(self->abiTagList(), [&](const auto& item) {
        return arrayValue(item, [&](const auto& item) {
          return val(static_cast<double>(reinterpret_cast<std::intptr_t>(
              static_cast<const ::cxx::Name*>(item))));
        });
      });
    }
    case SymbolSlotBase + 13: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return optionalValue(self->attributes(), [&](const auto& item) {
        return arrayValue(item, [&](const auto& item) {
          return [&]() -> val {
            auto result = val::object();
            result.set("attributeNamespace",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           static_cast<const ::cxx::Name*>(
                               (item).attributeNamespace)))));
            result.set("name",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           static_cast<const ::cxx::Name*>((item).name)))));
            result.set("arguments",
                       arrayValue((item).arguments, [&](const auto& item) {
                         return val(static_cast<double>(
                             reinterpret_cast<std::intptr_t>(
                                 static_cast<const ::cxx::Name*>(item))));
                       }));
            return result;
          }();
        });
      });
    }
    case ScopeSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::ScopeSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return arrayValue(self->usingDirectives(), [&](const auto& item) {
        return val(static_cast<double>(reinterpret_cast<std::intptr_t>(
            static_cast<const ::cxx::Symbol*>(item))));
      });
    }
    case NamespaceSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::NamespaceSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return optionalValue(self->anonNamespaceIndex(), [&](const auto& item) {
        return val(static_cast<double>(item));
      });
    }
    case ClassSymbolSlotBase + 50: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return [&]() -> val {
        auto result = val::object();
        result.set("nonDiamondRepeat",
                   val(static_cast<double>(
                       (self->baseClassRepetition()).nonDiamondRepeat)));
        result.set("diamondShaped",
                   val(static_cast<double>(
                       (self->baseClassRepetition()).diamondShaped)));
        return result;
      }();
    }
    case FunctionSymbolSlotBase + 52: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return optionalValue(self->pendingBody(), [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set(
              "originalDefinition",
              val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                  static_cast<const ::cxx::AST*>((item).originalDefinition)))));
          result.set(
              "templateArguments",
              arrayValue((item).templateArguments, [&](const auto& item) {
                return [&]() -> val {
                  auto result = val::object();
                  result.set("index", item.index());
                  switch (item.index()) {
                    case 0:
                      result.set("value",
                                 val(static_cast<double>(
                                     reinterpret_cast<std::intptr_t>(
                                         static_cast<const ::cxx::Type*>(
                                             std::get<0>(item))))));
                      break;
                    case 1:
                      result.set("value",
                                 val(static_cast<double>(
                                     reinterpret_cast<std::intptr_t>(
                                         static_cast<const ::cxx::Symbol*>(
                                             std::get<1>(item))))));
                      break;
                    case 2:
                      result.set("value", [&]() -> val {
                        auto result = val::object();
                        result.set("index", std::get<2>(item).index());
                        switch (std::get<2>(item).index()) {
                          case 0:
                            result.set(
                                "value",
                                val(std::get<0>(std::get<2>(item)).toString()));
                            break;
                          case 1:
                            result.set(
                                "value",
                                val(static_cast<double>(
                                    reinterpret_cast<std::intptr_t>(
                                        static_cast<const ::cxx::Literal*>(
                                            std::get<1>(std::get<2>(item)))))));
                            break;
                          case 2:
                            result.set("value",
                                       val(static_cast<double>(
                                           std::get<2>(std::get<2>(item)))));
                            break;
                          case 3:
                            result.set("value",
                                       val(static_cast<double>(
                                           std::get<3>(std::get<2>(item)))));
                            break;
                          case 4:
                            result.set("value",
                                       val(static_cast<double>(
                                           std::get<4>(std::get<2>(item)))));
                            break;
                          case 5:
                            result.set("value",
                                       val(static_cast<double>(
                                           reinterpret_cast<std::intptr_t>(
                                               std::get<5>(std::get<2>(item))
                                                   .get()))));
                            break;
                          case 6:
                            result.set("value",
                                       val(static_cast<double>(
                                           reinterpret_cast<std::intptr_t>(
                                               std::get<6>(std::get<2>(item))
                                                   .get()))));
                            break;
                          case 7:
                            result.set("value",
                                       val(static_cast<double>(
                                           reinterpret_cast<std::intptr_t>(
                                               std::get<7>(std::get<2>(item))
                                                   .get()))));
                            break;
                          case 8:
                            result.set("value",
                                       val(static_cast<double>(
                                           reinterpret_cast<std::intptr_t>(
                                               std::get<8>(std::get<2>(item))
                                                   .get()))));
                            break;
                          case 9:
                            result.set("value",
                                       val(static_cast<double>(
                                           reinterpret_cast<std::intptr_t>(
                                               std::get<9>(std::get<2>(item))
                                                   .get()))));
                            break;
                          case 10:
                            result.set("value",
                                       val(static_cast<double>(
                                           reinterpret_cast<std::intptr_t>(
                                               std::get<10>(std::get<2>(item))
                                                   .get()))));
                            break;
                          case 11:
                            result.set("value", val::undefined());
                            break;
                        }
                        return result;
                      }());
                      break;
                    case 3:
                      result.set("value",
                                 val(static_cast<double>(
                                     reinterpret_cast<std::intptr_t>(
                                         static_cast<const ::cxx::AST*>(
                                             std::get<3>(item))))));
                      break;
                  }
                  return result;
                }();
              }));
          result.set(
              "parentScope",
              val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                  static_cast<const ::cxx::Symbol*>((item).parentScope)))));
          result.set("depth", val(static_cast<double>((item).depth)));
          return result;
        }();
      });
    }
    case FunctionSymbolSlotBase + 53: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return optionalValue(
          self->pendingExceptionSpecification(), [&](const auto& item) {
            return [&]() -> val {
              auto result = val::object();
              result.set(
                  "original",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::AST*>((item).original)))));
              result.set(
                  "instance",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::AST*>((item).instance)))));
              result.set(
                  "originalFunction",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Symbol*>(
                          (item).originalFunction)))));
              result.set(
                  "templateArguments",
                  arrayValue((item).templateArguments, [&](const auto& item) {
                    return [&]() -> val {
                      auto result = val::object();
                      result.set("index", item.index());
                      switch (item.index()) {
                        case 0:
                          result.set("value",
                                     val(static_cast<double>(
                                         reinterpret_cast<std::intptr_t>(
                                             static_cast<const ::cxx::Type*>(
                                                 std::get<0>(item))))));
                          break;
                        case 1:
                          result.set("value",
                                     val(static_cast<double>(
                                         reinterpret_cast<std::intptr_t>(
                                             static_cast<const ::cxx::Symbol*>(
                                                 std::get<1>(item))))));
                          break;
                        case 2:
                          result.set("value", [&]() -> val {
                            auto result = val::object();
                            result.set("index", std::get<2>(item).index());
                            switch (std::get<2>(item).index()) {
                              case 0:
                                result.set("value",
                                           val(std::get<0>(std::get<2>(item))
                                                   .toString()));
                                break;
                              case 1:
                                result.set(
                                    "value",
                                    val(static_cast<double>(
                                        reinterpret_cast<std::intptr_t>(
                                            static_cast<const ::cxx::Literal*>(
                                                std::get<1>(
                                                    std::get<2>(item)))))));
                                break;
                              case 2:
                                result.set("value",
                                           val(static_cast<double>(std::get<2>(
                                               std::get<2>(item)))));
                                break;
                              case 3:
                                result.set("value",
                                           val(static_cast<double>(std::get<3>(
                                               std::get<2>(item)))));
                                break;
                              case 4:
                                result.set("value",
                                           val(static_cast<double>(std::get<4>(
                                               std::get<2>(item)))));
                                break;
                              case 5:
                                result.set(
                                    "value",
                                    val(static_cast<double>(
                                        reinterpret_cast<std::intptr_t>(
                                            std::get<5>(std::get<2>(item))
                                                .get()))));
                                break;
                              case 6:
                                result.set(
                                    "value",
                                    val(static_cast<double>(
                                        reinterpret_cast<std::intptr_t>(
                                            std::get<6>(std::get<2>(item))
                                                .get()))));
                                break;
                              case 7:
                                result.set(
                                    "value",
                                    val(static_cast<double>(
                                        reinterpret_cast<std::intptr_t>(
                                            std::get<7>(std::get<2>(item))
                                                .get()))));
                                break;
                              case 8:
                                result.set(
                                    "value",
                                    val(static_cast<double>(
                                        reinterpret_cast<std::intptr_t>(
                                            std::get<8>(std::get<2>(item))
                                                .get()))));
                                break;
                              case 9:
                                result.set(
                                    "value",
                                    val(static_cast<double>(
                                        reinterpret_cast<std::intptr_t>(
                                            std::get<9>(std::get<2>(item))
                                                .get()))));
                                break;
                              case 10:
                                result.set(
                                    "value",
                                    val(static_cast<double>(
                                        reinterpret_cast<std::intptr_t>(
                                            std::get<10>(std::get<2>(item))
                                                .get()))));
                                break;
                              case 11:
                                result.set("value", val::undefined());
                                break;
                            }
                            return result;
                          }());
                          break;
                        case 3:
                          result.set("value",
                                     val(static_cast<double>(
                                         reinterpret_cast<std::intptr_t>(
                                             static_cast<const ::cxx::AST*>(
                                                 std::get<3>(item))))));
                          break;
                      }
                      return result;
                    }();
                  }));
              result.set(
                  "parentScope",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Symbol*>((item).parentScope)))));
              result.set("depth", val(static_cast<double>((item).depth)));
              result.set("state", val(static_cast<double>((item).state)));
              result.set("recursionDiagnosed",
                         val(static_cast<double>((item).recursionDiagnosed)));
              return result;
            }();
          });
    }
    case VariableSymbolSlotBase + 23: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return optionalValue(self->constValue(), [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set("value", val(std::get<0>(item).toString()));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Literal*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", val(static_cast<double>(std::get<2>(item))));
              break;
            case 3:
              result.set("value", val(static_cast<double>(std::get<3>(item))));
              break;
            case 4:
              result.set("value", val(static_cast<double>(std::get<4>(item))));
              break;
            case 5:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<5>(item).get()))));
              break;
            case 6:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<6>(item).get()))));
              break;
            case 7:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<7>(item).get()))));
              break;
            case 8:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<8>(item).get()))));
              break;
            case 9:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<9>(item).get()))));
              break;
            case 10:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<10>(item).get()))));
              break;
            case 11:
              result.set("value", val::undefined());
              break;
          }
          return result;
        }();
      });
    }
    case FieldSymbolSlotBase + 3: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return optionalValue(self->bitFieldWidth(), [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set("value", val(std::get<0>(item).toString()));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Literal*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", val(static_cast<double>(std::get<2>(item))));
              break;
            case 3:
              result.set("value", val(static_cast<double>(std::get<3>(item))));
              break;
            case 4:
              result.set("value", val(static_cast<double>(std::get<4>(item))));
              break;
            case 5:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<5>(item).get()))));
              break;
            case 6:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<6>(item).get()))));
              break;
            case 7:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<7>(item).get()))));
              break;
            case 8:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<8>(item).get()))));
              break;
            case 9:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<9>(item).get()))));
              break;
            case 10:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<10>(item).get()))));
              break;
            case 11:
              result.set("value", val::undefined());
              break;
          }
          return result;
        }();
      });
    }
    case FieldSymbolSlotBase + 12: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return optionalValue(self->offsetInClass(), [&](const auto& item) {
        return val(static_cast<std::uint64_t>(item));
      });
    }
    case FieldSymbolSlotBase + 17: {
      auto self = static_cast<const ::cxx::FieldSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return optionalValue(self->constValue(), [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set("value", val(std::get<0>(item).toString()));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Literal*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", val(static_cast<double>(std::get<2>(item))));
              break;
            case 3:
              result.set("value", val(static_cast<double>(std::get<3>(item))));
              break;
            case 4:
              result.set("value", val(static_cast<double>(std::get<4>(item))));
              break;
            case 5:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<5>(item).get()))));
              break;
            case 6:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<6>(item).get()))));
              break;
            case 7:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<7>(item).get()))));
              break;
            case 8:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<8>(item).get()))));
              break;
            case 9:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<9>(item).get()))));
              break;
            case 10:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<10>(item).get()))));
              break;
            case 11:
              result.set("value", val::undefined());
              break;
          }
          return result;
        }();
      });
    }
    case EnumeratorSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::EnumeratorSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return optionalValue(self->value(), [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set("value", val(std::get<0>(item).toString()));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Literal*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", val(static_cast<double>(std::get<2>(item))));
              break;
            case 3:
              result.set("value", val(static_cast<double>(std::get<3>(item))));
              break;
            case 4:
              result.set("value", val(static_cast<double>(std::get<4>(item))));
              break;
            case 5:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<5>(item).get()))));
              break;
            case 6:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<6>(item).get()))));
              break;
            case 7:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<7>(item).get()))));
              break;
            case 8:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<8>(item).get()))));
              break;
            case 9:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<9>(item).get()))));
              break;
            case 10:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<10>(item).get()))));
              break;
            case 11:
              result.set("value", val::undefined());
              break;
          }
          return result;
        }();
      });
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readSymbolSize(std::intptr_t handle, int slot) -> int {
  switch (slot) {
    case SymbolSlotBase + 11: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      return static_cast<int>(std::size(self->abiTags()));
    }
    case ScopeSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::ScopeSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->members()));
    }
    case ConceptSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->templateArguments()));
    }
    case ConceptSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(
          std::size(self->externInstantiationDeclarations()));
    }
    case DeductionGuideSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->templateArguments()));
    }
    case DeductionGuideSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(
          std::size(self->externInstantiationDeclarations()));
    }
    case ClassSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->templateArguments()));
    }
    case ClassSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(
          std::size(self->externInstantiationDeclarations()));
    }
    case ClassSymbolSlotBase + 13: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->redeclarations()));
    }
    case ClassSymbolSlotBase + 14: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->declarations()));
    }
    case ClassSymbolSlotBase + 16: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(
          std::size(self->instantiationSubstitutionArguments()));
    }
    case ClassSymbolSlotBase + 18: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->baseClasses()));
    }
    case ClassSymbolSlotBase + 19: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->constructors()));
    }
    case ClassSymbolSlotBase + 20: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->declaredConstructors()));
    }
    case ClassSymbolSlotBase + 22: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->deductionGuides()));
    }
    case ClassSymbolSlotBase + 23: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->conversionFunctions()));
    }
    case ClassSymbolSlotBase + 24: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->implicitConversionFunctions()));
    }
    case ClassSymbolSlotBase + 25: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->visibleConversionFunctions()));
    }
    case ClassSymbolSlotBase + 36: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->convertingConstructors()));
    }
    case ClassSymbolSlotBase + 48: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->befriendingClasses()));
    }
    case ClassSymbolSlotBase + 49: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->templateFriendships()));
    }
    case ClassSymbolSlotBase + 58: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(expand_template_arguments(
          class_template_arguments(const_cast<ClassSymbol*>(self)))));
    }
    case ClassSymbolSlotBase + 59: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size([&] {
        std::vector<std::string> result;
        for (const auto& argument : expand_template_arguments(
                 class_template_arguments(const_cast<ClassSymbol*>(self))))
          result.push_back(to_string(argument));
        return result;
      }()));
    }
    case FunctionSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->templateArguments()));
    }
    case FunctionSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(
          std::size(self->externInstantiationDeclarations()));
    }
    case FunctionSymbolSlotBase + 13: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->redeclarations()));
    }
    case FunctionSymbolSlotBase + 14: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->declarations()));
    }
    case FunctionSymbolSlotBase + 23: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->parameters()));
    }
    case FunctionSymbolSlotBase + 55: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->overriddenFunctions()));
    }
    case FunctionSymbolSlotBase + 56: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->befriendingClasses()));
    }
    case FunctionSymbolSlotBase + 57: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->templateFriendships()));
    }
    case OverloadSetSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::OverloadSetSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->functions()));
    }
    case OverloadSetSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::OverloadSetSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->declaredFunctions()));
    }
    case OverloadSetSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::OverloadSetSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->usingDeclarations()));
    }
    case TypeAliasSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->templateArguments()));
    }
    case TypeAliasSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(
          std::size(self->externInstantiationDeclarations()));
    }
    case TypeAliasSymbolSlotBase + 13: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->redeclarations()));
    }
    case TypeAliasSymbolSlotBase + 14: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->declarations()));
    }
    case VariableSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->templateArguments()));
    }
    case VariableSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(
          std::size(self->externInstantiationDeclarations()));
    }
    case VariableSymbolSlotBase + 13: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->redeclarations()));
    }
    case VariableSymbolSlotBase + 14: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->declarations()));
    }
    case ParameterPackSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::ParameterPackSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->elements()));
    }
    case UsingDeclarationSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::UsingDeclarationSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      return static_cast<int>(std::size(self->introducedFunctions()));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readSymbolItem(std::intptr_t handle, int slot, int index) -> double {
  switch (slot) {
    case SymbolSlotBase + 11: {
      auto self = reinterpret_cast<const ::cxx::Symbol*>(handle);
      const auto& container = self->abiTags();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(item)));
    }
    case ScopeSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::ScopeSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->members();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 13: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->redeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 14: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->declarations();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 18: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->baseClasses();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 19: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->constructors();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 20: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->declaredConstructors();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 22: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->deductionGuides();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 23: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->conversionFunctions();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 24: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->implicitConversionFunctions();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 25: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->visibleConversionFunctions();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 36: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->convertingConstructors();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ClassSymbolSlotBase + 48: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->befriendingClasses();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case FunctionSymbolSlotBase + 13: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->redeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case FunctionSymbolSlotBase + 14: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->declarations();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case FunctionSymbolSlotBase + 23: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->parameters();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case FunctionSymbolSlotBase + 55: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->overriddenFunctions();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case FunctionSymbolSlotBase + 56: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->befriendingClasses();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case OverloadSetSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::OverloadSetSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->functions();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case OverloadSetSymbolSlotBase + 1: {
      auto self = static_cast<const ::cxx::OverloadSetSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->declaredFunctions();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case OverloadSetSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::OverloadSetSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->usingDeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case TypeAliasSymbolSlotBase + 13: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->redeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case TypeAliasSymbolSlotBase + 14: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->declarations();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case VariableSymbolSlotBase + 13: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->redeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case VariableSymbolSlotBase + 14: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->declarations();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case ParameterPackSymbolSlotBase + 0: {
      auto self = static_cast<const ::cxx::ParameterPackSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->elements();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
    case UsingDeclarationSymbolSlotBase + 2: {
      auto self = static_cast<const ::cxx::UsingDeclarationSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->introducedFunctions();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(item)));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readSymbolItemString(std::intptr_t handle, int slot, int index)
    -> std::string {
  switch (slot) {
    case ClassSymbolSlotBase + 59: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = [&] {
        std::vector<std::string> result;
        for (const auto& argument : expand_template_arguments(
                 class_template_arguments(const_cast<ClassSymbol*>(self))))
          result.push_back(to_string(argument));
        return result;
      }();
      const auto& item = *std::next(std::begin(container), index);
      return std::string(item);
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readSymbolItemVal(std::intptr_t handle, int slot, int index) -> val {
  switch (slot) {
    case ConceptSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->templateArguments();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", item.index());
        switch (item.index()) {
          case 0:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Type*>(std::get<0>(item))))));
            break;
          case 1:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
            break;
          case 2:
            result.set("value", [&]() -> val {
              auto result = val::object();
              result.set("index", std::get<2>(item).index());
              switch (std::get<2>(item).index()) {
                case 0:
                  result.set("value",
                             val(std::get<0>(std::get<2>(item)).toString()));
                  break;
                case 1:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          static_cast<const ::cxx::Literal*>(
                              std::get<1>(std::get<2>(item)))))));
                  break;
                case 2:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<2>(std::get<2>(item)))));
                  break;
                case 3:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<3>(std::get<2>(item)))));
                  break;
                case 4:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<4>(std::get<2>(item)))));
                  break;
                case 5:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<5>(std::get<2>(item)).get()))));
                  break;
                case 6:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<6>(std::get<2>(item)).get()))));
                  break;
                case 7:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<7>(std::get<2>(item)).get()))));
                  break;
                case 8:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<8>(std::get<2>(item)).get()))));
                  break;
                case 9:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<9>(std::get<2>(item)).get()))));
                  break;
                case 10:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<10>(std::get<2>(item)).get()))));
                  break;
                case 11:
                  result.set("value", val::undefined());
                  break;
              }
              return result;
            }());
            break;
          case 3:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::AST*>(std::get<3>(item))))));
            break;
        }
        return result;
      }();
    }
    case ConceptSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::ConceptSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->externInstantiationDeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return arrayValue(item, [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Type*>(std::get<0>(item))))));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", [&]() -> val {
                auto result = val::object();
                result.set("index", std::get<2>(item).index());
                switch (std::get<2>(item).index()) {
                  case 0:
                    result.set("value",
                               val(std::get<0>(std::get<2>(item)).toString()));
                    break;
                  case 1:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Literal*>(
                                std::get<1>(std::get<2>(item)))))));
                    break;
                  case 2:
                    result.set("value", val(static_cast<double>(
                                            std::get<2>(std::get<2>(item)))));
                    break;
                  case 3:
                    result.set("value", val(static_cast<double>(
                                            std::get<3>(std::get<2>(item)))));
                    break;
                  case 4:
                    result.set("value", val(static_cast<double>(
                                            std::get<4>(std::get<2>(item)))));
                    break;
                  case 5:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<5>(std::get<2>(item)).get()))));
                    break;
                  case 6:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<6>(std::get<2>(item)).get()))));
                    break;
                  case 7:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<7>(std::get<2>(item)).get()))));
                    break;
                  case 8:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<8>(std::get<2>(item)).get()))));
                    break;
                  case 9:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<9>(std::get<2>(item)).get()))));
                    break;
                  case 10:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<10>(std::get<2>(item)).get()))));
                    break;
                  case 11:
                    result.set("value", val::undefined());
                    break;
                }
                return result;
              }());
              break;
            case 3:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::AST*>(std::get<3>(item))))));
              break;
          }
          return result;
        }();
      });
    }
    case DeductionGuideSymbolSlotBase + 5: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->templateArguments();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", item.index());
        switch (item.index()) {
          case 0:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Type*>(std::get<0>(item))))));
            break;
          case 1:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
            break;
          case 2:
            result.set("value", [&]() -> val {
              auto result = val::object();
              result.set("index", std::get<2>(item).index());
              switch (std::get<2>(item).index()) {
                case 0:
                  result.set("value",
                             val(std::get<0>(std::get<2>(item)).toString()));
                  break;
                case 1:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          static_cast<const ::cxx::Literal*>(
                              std::get<1>(std::get<2>(item)))))));
                  break;
                case 2:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<2>(std::get<2>(item)))));
                  break;
                case 3:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<3>(std::get<2>(item)))));
                  break;
                case 4:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<4>(std::get<2>(item)))));
                  break;
                case 5:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<5>(std::get<2>(item)).get()))));
                  break;
                case 6:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<6>(std::get<2>(item)).get()))));
                  break;
                case 7:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<7>(std::get<2>(item)).get()))));
                  break;
                case 8:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<8>(std::get<2>(item)).get()))));
                  break;
                case 9:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<9>(std::get<2>(item)).get()))));
                  break;
                case 10:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<10>(std::get<2>(item)).get()))));
                  break;
                case 11:
                  result.set("value", val::undefined());
                  break;
              }
              return result;
            }());
            break;
          case 3:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::AST*>(std::get<3>(item))))));
            break;
        }
        return result;
      }();
    }
    case DeductionGuideSymbolSlotBase + 6: {
      auto self = static_cast<const ::cxx::DeductionGuideSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->externInstantiationDeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return arrayValue(item, [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Type*>(std::get<0>(item))))));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", [&]() -> val {
                auto result = val::object();
                result.set("index", std::get<2>(item).index());
                switch (std::get<2>(item).index()) {
                  case 0:
                    result.set("value",
                               val(std::get<0>(std::get<2>(item)).toString()));
                    break;
                  case 1:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Literal*>(
                                std::get<1>(std::get<2>(item)))))));
                    break;
                  case 2:
                    result.set("value", val(static_cast<double>(
                                            std::get<2>(std::get<2>(item)))));
                    break;
                  case 3:
                    result.set("value", val(static_cast<double>(
                                            std::get<3>(std::get<2>(item)))));
                    break;
                  case 4:
                    result.set("value", val(static_cast<double>(
                                            std::get<4>(std::get<2>(item)))));
                    break;
                  case 5:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<5>(std::get<2>(item)).get()))));
                    break;
                  case 6:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<6>(std::get<2>(item)).get()))));
                    break;
                  case 7:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<7>(std::get<2>(item)).get()))));
                    break;
                  case 8:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<8>(std::get<2>(item)).get()))));
                    break;
                  case 9:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<9>(std::get<2>(item)).get()))));
                    break;
                  case 10:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<10>(std::get<2>(item)).get()))));
                    break;
                  case 11:
                    result.set("value", val::undefined());
                    break;
                }
                return result;
              }());
              break;
            case 3:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::AST*>(std::get<3>(item))))));
              break;
          }
          return result;
        }();
      });
    }
    case ClassSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->templateArguments();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", item.index());
        switch (item.index()) {
          case 0:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Type*>(std::get<0>(item))))));
            break;
          case 1:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
            break;
          case 2:
            result.set("value", [&]() -> val {
              auto result = val::object();
              result.set("index", std::get<2>(item).index());
              switch (std::get<2>(item).index()) {
                case 0:
                  result.set("value",
                             val(std::get<0>(std::get<2>(item)).toString()));
                  break;
                case 1:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          static_cast<const ::cxx::Literal*>(
                              std::get<1>(std::get<2>(item)))))));
                  break;
                case 2:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<2>(std::get<2>(item)))));
                  break;
                case 3:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<3>(std::get<2>(item)))));
                  break;
                case 4:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<4>(std::get<2>(item)))));
                  break;
                case 5:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<5>(std::get<2>(item)).get()))));
                  break;
                case 6:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<6>(std::get<2>(item)).get()))));
                  break;
                case 7:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<7>(std::get<2>(item)).get()))));
                  break;
                case 8:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<8>(std::get<2>(item)).get()))));
                  break;
                case 9:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<9>(std::get<2>(item)).get()))));
                  break;
                case 10:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<10>(std::get<2>(item)).get()))));
                  break;
                case 11:
                  result.set("value", val::undefined());
                  break;
              }
              return result;
            }());
            break;
          case 3:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::AST*>(std::get<3>(item))))));
            break;
        }
        return result;
      }();
    }
    case ClassSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->externInstantiationDeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return arrayValue(item, [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Type*>(std::get<0>(item))))));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", [&]() -> val {
                auto result = val::object();
                result.set("index", std::get<2>(item).index());
                switch (std::get<2>(item).index()) {
                  case 0:
                    result.set("value",
                               val(std::get<0>(std::get<2>(item)).toString()));
                    break;
                  case 1:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Literal*>(
                                std::get<1>(std::get<2>(item)))))));
                    break;
                  case 2:
                    result.set("value", val(static_cast<double>(
                                            std::get<2>(std::get<2>(item)))));
                    break;
                  case 3:
                    result.set("value", val(static_cast<double>(
                                            std::get<3>(std::get<2>(item)))));
                    break;
                  case 4:
                    result.set("value", val(static_cast<double>(
                                            std::get<4>(std::get<2>(item)))));
                    break;
                  case 5:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<5>(std::get<2>(item)).get()))));
                    break;
                  case 6:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<6>(std::get<2>(item)).get()))));
                    break;
                  case 7:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<7>(std::get<2>(item)).get()))));
                    break;
                  case 8:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<8>(std::get<2>(item)).get()))));
                    break;
                  case 9:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<9>(std::get<2>(item)).get()))));
                    break;
                  case 10:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<10>(std::get<2>(item)).get()))));
                    break;
                  case 11:
                    result.set("value", val::undefined());
                    break;
                }
                return result;
              }());
              break;
            case 3:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::AST*>(std::get<3>(item))))));
              break;
          }
          return result;
        }();
      });
    }
    case ClassSymbolSlotBase + 16: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->instantiationSubstitutionArguments();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", item.index());
        switch (item.index()) {
          case 0:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Type*>(std::get<0>(item))))));
            break;
          case 1:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
            break;
          case 2:
            result.set("value", [&]() -> val {
              auto result = val::object();
              result.set("index", std::get<2>(item).index());
              switch (std::get<2>(item).index()) {
                case 0:
                  result.set("value",
                             val(std::get<0>(std::get<2>(item)).toString()));
                  break;
                case 1:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          static_cast<const ::cxx::Literal*>(
                              std::get<1>(std::get<2>(item)))))));
                  break;
                case 2:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<2>(std::get<2>(item)))));
                  break;
                case 3:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<3>(std::get<2>(item)))));
                  break;
                case 4:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<4>(std::get<2>(item)))));
                  break;
                case 5:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<5>(std::get<2>(item)).get()))));
                  break;
                case 6:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<6>(std::get<2>(item)).get()))));
                  break;
                case 7:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<7>(std::get<2>(item)).get()))));
                  break;
                case 8:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<8>(std::get<2>(item)).get()))));
                  break;
                case 9:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<9>(std::get<2>(item)).get()))));
                  break;
                case 10:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<10>(std::get<2>(item)).get()))));
                  break;
                case 11:
                  result.set("value", val::undefined());
                  break;
              }
              return result;
            }());
            break;
          case 3:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::AST*>(std::get<3>(item))))));
            break;
        }
        return result;
      }();
    }
    case ClassSymbolSlotBase + 49: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->templateFriendships();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set(
            "arguments", arrayValue((item).arguments, [&](const auto& item) {
              return [&]() -> val {
                auto result = val::object();
                result.set("index", item.index());
                switch (item.index()) {
                  case 0:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Type*>(
                                std::get<0>(item))))));
                    break;
                  case 1:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Symbol*>(
                                std::get<1>(item))))));
                    break;
                  case 2:
                    result.set("value", [&]() -> val {
                      auto result = val::object();
                      result.set("index", std::get<2>(item).index());
                      switch (std::get<2>(item).index()) {
                        case 0:
                          result.set(
                              "value",
                              val(std::get<0>(std::get<2>(item)).toString()));
                          break;
                        case 1:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      static_cast<const ::cxx::Literal*>(
                                          std::get<1>(std::get<2>(item)))))));
                          break;
                        case 2:
                          result.set("value",
                                     val(static_cast<double>(
                                         std::get<2>(std::get<2>(item)))));
                          break;
                        case 3:
                          result.set("value",
                                     val(static_cast<double>(
                                         std::get<3>(std::get<2>(item)))));
                          break;
                        case 4:
                          result.set("value",
                                     val(static_cast<double>(
                                         std::get<4>(std::get<2>(item)))));
                          break;
                        case 5:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<5>(std::get<2>(item)).get()))));
                          break;
                        case 6:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<6>(std::get<2>(item)).get()))));
                          break;
                        case 7:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<7>(std::get<2>(item)).get()))));
                          break;
                        case 8:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<8>(std::get<2>(item)).get()))));
                          break;
                        case 9:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<9>(std::get<2>(item)).get()))));
                          break;
                        case 10:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<10>(std::get<2>(item)).get()))));
                          break;
                        case 11:
                          result.set("value", val::undefined());
                          break;
                      }
                      return result;
                    }());
                    break;
                  case 3:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::AST*>(
                                std::get<3>(item))))));
                    break;
                }
                return result;
              }();
            }));
        result.set(
            "befriendingClass",
            val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                static_cast<const ::cxx::Symbol*>((item).befriendingClass)))));
        return result;
      }();
    }
    case ClassSymbolSlotBase + 58: {
      auto self = static_cast<const ::cxx::ClassSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = expand_template_arguments(
          class_template_arguments(const_cast<ClassSymbol*>(self)));
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", item.index());
        switch (item.index()) {
          case 0:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Type*>(std::get<0>(item))))));
            break;
          case 1:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
            break;
          case 2:
            result.set("value", [&]() -> val {
              auto result = val::object();
              result.set("index", std::get<2>(item).index());
              switch (std::get<2>(item).index()) {
                case 0:
                  result.set("value",
                             val(std::get<0>(std::get<2>(item)).toString()));
                  break;
                case 1:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          static_cast<const ::cxx::Literal*>(
                              std::get<1>(std::get<2>(item)))))));
                  break;
                case 2:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<2>(std::get<2>(item)))));
                  break;
                case 3:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<3>(std::get<2>(item)))));
                  break;
                case 4:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<4>(std::get<2>(item)))));
                  break;
                case 5:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<5>(std::get<2>(item)).get()))));
                  break;
                case 6:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<6>(std::get<2>(item)).get()))));
                  break;
                case 7:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<7>(std::get<2>(item)).get()))));
                  break;
                case 8:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<8>(std::get<2>(item)).get()))));
                  break;
                case 9:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<9>(std::get<2>(item)).get()))));
                  break;
                case 10:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<10>(std::get<2>(item)).get()))));
                  break;
                case 11:
                  result.set("value", val::undefined());
                  break;
              }
              return result;
            }());
            break;
          case 3:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::AST*>(std::get<3>(item))))));
            break;
        }
        return result;
      }();
    }
    case FunctionSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->templateArguments();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", item.index());
        switch (item.index()) {
          case 0:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Type*>(std::get<0>(item))))));
            break;
          case 1:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
            break;
          case 2:
            result.set("value", [&]() -> val {
              auto result = val::object();
              result.set("index", std::get<2>(item).index());
              switch (std::get<2>(item).index()) {
                case 0:
                  result.set("value",
                             val(std::get<0>(std::get<2>(item)).toString()));
                  break;
                case 1:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          static_cast<const ::cxx::Literal*>(
                              std::get<1>(std::get<2>(item)))))));
                  break;
                case 2:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<2>(std::get<2>(item)))));
                  break;
                case 3:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<3>(std::get<2>(item)))));
                  break;
                case 4:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<4>(std::get<2>(item)))));
                  break;
                case 5:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<5>(std::get<2>(item)).get()))));
                  break;
                case 6:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<6>(std::get<2>(item)).get()))));
                  break;
                case 7:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<7>(std::get<2>(item)).get()))));
                  break;
                case 8:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<8>(std::get<2>(item)).get()))));
                  break;
                case 9:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<9>(std::get<2>(item)).get()))));
                  break;
                case 10:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<10>(std::get<2>(item)).get()))));
                  break;
                case 11:
                  result.set("value", val::undefined());
                  break;
              }
              return result;
            }());
            break;
          case 3:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::AST*>(std::get<3>(item))))));
            break;
        }
        return result;
      }();
    }
    case FunctionSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->externInstantiationDeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return arrayValue(item, [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Type*>(std::get<0>(item))))));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", [&]() -> val {
                auto result = val::object();
                result.set("index", std::get<2>(item).index());
                switch (std::get<2>(item).index()) {
                  case 0:
                    result.set("value",
                               val(std::get<0>(std::get<2>(item)).toString()));
                    break;
                  case 1:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Literal*>(
                                std::get<1>(std::get<2>(item)))))));
                    break;
                  case 2:
                    result.set("value", val(static_cast<double>(
                                            std::get<2>(std::get<2>(item)))));
                    break;
                  case 3:
                    result.set("value", val(static_cast<double>(
                                            std::get<3>(std::get<2>(item)))));
                    break;
                  case 4:
                    result.set("value", val(static_cast<double>(
                                            std::get<4>(std::get<2>(item)))));
                    break;
                  case 5:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<5>(std::get<2>(item)).get()))));
                    break;
                  case 6:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<6>(std::get<2>(item)).get()))));
                    break;
                  case 7:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<7>(std::get<2>(item)).get()))));
                    break;
                  case 8:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<8>(std::get<2>(item)).get()))));
                    break;
                  case 9:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<9>(std::get<2>(item)).get()))));
                    break;
                  case 10:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<10>(std::get<2>(item)).get()))));
                    break;
                  case 11:
                    result.set("value", val::undefined());
                    break;
                }
                return result;
              }());
              break;
            case 3:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::AST*>(std::get<3>(item))))));
              break;
          }
          return result;
        }();
      });
    }
    case FunctionSymbolSlotBase + 57: {
      auto self = static_cast<const ::cxx::FunctionSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->templateFriendships();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set(
            "arguments", arrayValue((item).arguments, [&](const auto& item) {
              return [&]() -> val {
                auto result = val::object();
                result.set("index", item.index());
                switch (item.index()) {
                  case 0:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Type*>(
                                std::get<0>(item))))));
                    break;
                  case 1:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Symbol*>(
                                std::get<1>(item))))));
                    break;
                  case 2:
                    result.set("value", [&]() -> val {
                      auto result = val::object();
                      result.set("index", std::get<2>(item).index());
                      switch (std::get<2>(item).index()) {
                        case 0:
                          result.set(
                              "value",
                              val(std::get<0>(std::get<2>(item)).toString()));
                          break;
                        case 1:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      static_cast<const ::cxx::Literal*>(
                                          std::get<1>(std::get<2>(item)))))));
                          break;
                        case 2:
                          result.set("value",
                                     val(static_cast<double>(
                                         std::get<2>(std::get<2>(item)))));
                          break;
                        case 3:
                          result.set("value",
                                     val(static_cast<double>(
                                         std::get<3>(std::get<2>(item)))));
                          break;
                        case 4:
                          result.set("value",
                                     val(static_cast<double>(
                                         std::get<4>(std::get<2>(item)))));
                          break;
                        case 5:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<5>(std::get<2>(item)).get()))));
                          break;
                        case 6:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<6>(std::get<2>(item)).get()))));
                          break;
                        case 7:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<7>(std::get<2>(item)).get()))));
                          break;
                        case 8:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<8>(std::get<2>(item)).get()))));
                          break;
                        case 9:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<9>(std::get<2>(item)).get()))));
                          break;
                        case 10:
                          result.set(
                              "value",
                              val(static_cast<double>(
                                  reinterpret_cast<std::intptr_t>(
                                      std::get<10>(std::get<2>(item)).get()))));
                          break;
                        case 11:
                          result.set("value", val::undefined());
                          break;
                      }
                      return result;
                    }());
                    break;
                  case 3:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::AST*>(
                                std::get<3>(item))))));
                    break;
                }
                return result;
              }();
            }));
        result.set(
            "befriendingClass",
            val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                static_cast<const ::cxx::Symbol*>((item).befriendingClass)))));
        return result;
      }();
    }
    case TypeAliasSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->templateArguments();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", item.index());
        switch (item.index()) {
          case 0:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Type*>(std::get<0>(item))))));
            break;
          case 1:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
            break;
          case 2:
            result.set("value", [&]() -> val {
              auto result = val::object();
              result.set("index", std::get<2>(item).index());
              switch (std::get<2>(item).index()) {
                case 0:
                  result.set("value",
                             val(std::get<0>(std::get<2>(item)).toString()));
                  break;
                case 1:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          static_cast<const ::cxx::Literal*>(
                              std::get<1>(std::get<2>(item)))))));
                  break;
                case 2:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<2>(std::get<2>(item)))));
                  break;
                case 3:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<3>(std::get<2>(item)))));
                  break;
                case 4:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<4>(std::get<2>(item)))));
                  break;
                case 5:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<5>(std::get<2>(item)).get()))));
                  break;
                case 6:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<6>(std::get<2>(item)).get()))));
                  break;
                case 7:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<7>(std::get<2>(item)).get()))));
                  break;
                case 8:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<8>(std::get<2>(item)).get()))));
                  break;
                case 9:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<9>(std::get<2>(item)).get()))));
                  break;
                case 10:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<10>(std::get<2>(item)).get()))));
                  break;
                case 11:
                  result.set("value", val::undefined());
                  break;
              }
              return result;
            }());
            break;
          case 3:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::AST*>(std::get<3>(item))))));
            break;
        }
        return result;
      }();
    }
    case TypeAliasSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::TypeAliasSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->externInstantiationDeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return arrayValue(item, [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Type*>(std::get<0>(item))))));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", [&]() -> val {
                auto result = val::object();
                result.set("index", std::get<2>(item).index());
                switch (std::get<2>(item).index()) {
                  case 0:
                    result.set("value",
                               val(std::get<0>(std::get<2>(item)).toString()));
                    break;
                  case 1:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Literal*>(
                                std::get<1>(std::get<2>(item)))))));
                    break;
                  case 2:
                    result.set("value", val(static_cast<double>(
                                            std::get<2>(std::get<2>(item)))));
                    break;
                  case 3:
                    result.set("value", val(static_cast<double>(
                                            std::get<3>(std::get<2>(item)))));
                    break;
                  case 4:
                    result.set("value", val(static_cast<double>(
                                            std::get<4>(std::get<2>(item)))));
                    break;
                  case 5:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<5>(std::get<2>(item)).get()))));
                    break;
                  case 6:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<6>(std::get<2>(item)).get()))));
                    break;
                  case 7:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<7>(std::get<2>(item)).get()))));
                    break;
                  case 8:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<8>(std::get<2>(item)).get()))));
                    break;
                  case 9:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<9>(std::get<2>(item)).get()))));
                    break;
                  case 10:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<10>(std::get<2>(item)).get()))));
                    break;
                  case 11:
                    result.set("value", val::undefined());
                    break;
                }
                return result;
              }());
              break;
            case 3:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::AST*>(std::get<3>(item))))));
              break;
          }
          return result;
        }();
      });
    }
    case VariableSymbolSlotBase + 7: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->templateArguments();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", item.index());
        switch (item.index()) {
          case 0:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Type*>(std::get<0>(item))))));
            break;
          case 1:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
            break;
          case 2:
            result.set("value", [&]() -> val {
              auto result = val::object();
              result.set("index", std::get<2>(item).index());
              switch (std::get<2>(item).index()) {
                case 0:
                  result.set("value",
                             val(std::get<0>(std::get<2>(item)).toString()));
                  break;
                case 1:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          static_cast<const ::cxx::Literal*>(
                              std::get<1>(std::get<2>(item)))))));
                  break;
                case 2:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<2>(std::get<2>(item)))));
                  break;
                case 3:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<3>(std::get<2>(item)))));
                  break;
                case 4:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<4>(std::get<2>(item)))));
                  break;
                case 5:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<5>(std::get<2>(item)).get()))));
                  break;
                case 6:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<6>(std::get<2>(item)).get()))));
                  break;
                case 7:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<7>(std::get<2>(item)).get()))));
                  break;
                case 8:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<8>(std::get<2>(item)).get()))));
                  break;
                case 9:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<9>(std::get<2>(item)).get()))));
                  break;
                case 10:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<10>(std::get<2>(item)).get()))));
                  break;
                case 11:
                  result.set("value", val::undefined());
                  break;
              }
              return result;
            }());
            break;
          case 3:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::AST*>(std::get<3>(item))))));
            break;
        }
        return result;
      }();
    }
    case VariableSymbolSlotBase + 8: {
      auto self = static_cast<const ::cxx::VariableSymbol*>(
          reinterpret_cast<const ::cxx::Symbol*>(handle));
      const auto& container = self->externInstantiationDeclarations();
      const auto& item = *std::next(std::begin(container), index);
      return arrayValue(item, [&](const auto& item) {
        return [&]() -> val {
          auto result = val::object();
          result.set("index", item.index());
          switch (item.index()) {
            case 0:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Type*>(std::get<0>(item))))));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
              break;
            case 2:
              result.set("value", [&]() -> val {
                auto result = val::object();
                result.set("index", std::get<2>(item).index());
                switch (std::get<2>(item).index()) {
                  case 0:
                    result.set("value",
                               val(std::get<0>(std::get<2>(item)).toString()));
                    break;
                  case 1:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            static_cast<const ::cxx::Literal*>(
                                std::get<1>(std::get<2>(item)))))));
                    break;
                  case 2:
                    result.set("value", val(static_cast<double>(
                                            std::get<2>(std::get<2>(item)))));
                    break;
                  case 3:
                    result.set("value", val(static_cast<double>(
                                            std::get<3>(std::get<2>(item)))));
                    break;
                  case 4:
                    result.set("value", val(static_cast<double>(
                                            std::get<4>(std::get<2>(item)))));
                    break;
                  case 5:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<5>(std::get<2>(item)).get()))));
                    break;
                  case 6:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<6>(std::get<2>(item)).get()))));
                    break;
                  case 7:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<7>(std::get<2>(item)).get()))));
                    break;
                  case 8:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<8>(std::get<2>(item)).get()))));
                    break;
                  case 9:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<9>(std::get<2>(item)).get()))));
                    break;
                  case 10:
                    result.set(
                        "value",
                        val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                            std::get<10>(std::get<2>(item)).get()))));
                    break;
                  case 11:
                    result.set("value", val::undefined());
                    break;
                }
                return result;
              }());
              break;
            case 3:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::AST*>(std::get<3>(item))))));
              break;
          }
          return result;
        }();
      });
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readType(std::intptr_t handle, int slot) -> double {
  switch (slot) {
    case QualTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::QualType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case QualTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::QualType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->cvQualifiers());
    }
    case QualTypeSlotBase + 2: {
      auto self = static_cast<const ::cxx::QualType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->isConst());
    }
    case QualTypeSlotBase + 3: {
      auto self = static_cast<const ::cxx::QualType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->isVolatile());
    }
    case BoundedArrayTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::BoundedArrayType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case BoundedArrayTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::BoundedArrayType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->size());
    }
    case UnboundedArrayTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnboundedArrayType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case PointerTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::PointerType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case LvalueReferenceTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::LvalueReferenceType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case RvalueReferenceTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::RvalueReferenceType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case OverloadSetTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::OverloadSetType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol())));
    }
    case FunctionTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::FunctionType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->returnType())));
    }
    case FunctionTypeSlotBase + 2: {
      auto self = static_cast<const ::cxx::FunctionType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->isVariadic());
    }
    case FunctionTypeSlotBase + 3: {
      auto self = static_cast<const ::cxx::FunctionType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->cvQualifiers());
    }
    case FunctionTypeSlotBase + 4: {
      auto self = static_cast<const ::cxx::FunctionType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->refQualifier());
    }
    case FunctionTypeSlotBase + 5: {
      auto self = static_cast<const ::cxx::FunctionType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->isNoexcept());
    }
    case ClassTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::ClassType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol())));
    }
    case ClassTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::ClassType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->definition())));
    }
    case ClassTypeSlotBase + 2: {
      auto self = static_cast<const ::cxx::ClassType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->isComplete());
    }
    case ClassTypeSlotBase + 3: {
      auto self = static_cast<const ::cxx::ClassType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->isUnion());
    }
    case EnumTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::EnumType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol())));
    }
    case EnumTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::EnumType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->underlyingType())));
    }
    case ScopedEnumTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::ScopedEnumType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol())));
    }
    case ScopedEnumTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::ScopedEnumType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->underlyingType())));
    }
    case MemberObjectPointerTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::MemberObjectPointerType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->classType())));
    }
    case MemberObjectPointerTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::MemberObjectPointerType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case MemberFunctionPointerTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::MemberFunctionPointerType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->classType())));
    }
    case MemberFunctionPointerTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::MemberFunctionPointerType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->functionType())));
    }
    case NamespaceTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::NamespaceType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol())));
    }
    case TypeParameterTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::TypeParameterType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->index());
    }
    case TypeParameterTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::TypeParameterType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->depth());
    }
    case TypeParameterTypeSlotBase + 2: {
      auto self = static_cast<const ::cxx::TypeParameterType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->isParameterPack());
    }
    case TemplateTypeParameterTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->index());
    }
    case TemplateTypeParameterTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->depth());
    }
    case TemplateTypeParameterTypeSlotBase + 2: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->isParameterPack());
    }
    case UnresolvedNameTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnresolvedNameType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->nestedNameSpecifier())));
    }
    case UnresolvedNameTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::UnresolvedNameType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->unqualifiedId())));
    }
    case UnresolvedBoundedArrayTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnresolvedBoundedArrayType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case UnresolvedBoundedArrayTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::UnresolvedBoundedArrayType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->size())));
    }
    case UnresolvedUnderlyingTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnresolvedUnderlyingType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId())));
    }
    case UnresolvedBuiltinTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnresolvedBuiltinType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->builtinKind());
    }
    case UnresolvedBuiltinTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::UnresolvedBuiltinType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->typeId())));
    }
    case BitIntTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::BitIntType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->numBits());
    }
    case UnsignedBitIntTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnsignedBitIntType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->numBits());
    }
    case UnresolvedBitIntTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnresolvedBitIntType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->sizeExpression())));
    }
    case UnresolvedBitIntTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::UnresolvedBitIntType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->isUnsigned());
    }
    case VectorTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::VectorType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case VectorTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::VectorType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->elementCount());
    }
    case VectorTypeSlotBase + 2: {
      auto self = static_cast<const ::cxx::VectorType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->vectorKind());
    }
    case UnresolvedVectorTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::UnresolvedVectorType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case UnresolvedVectorTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::UnresolvedVectorType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::AST*>(self->sizeExpression())));
    }
    case UnresolvedVectorTypeSlotBase + 2: {
      auto self = static_cast<const ::cxx::UnresolvedVectorType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->vectorKind());
    }
    case UnresolvedVectorTypeSlotBase + 3: {
      auto self = static_cast<const ::cxx::UnresolvedVectorType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(self->sizeKind());
    }
    case ComplexTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::ComplexType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
    case AtomicTypeSlotBase + 0: {
      auto self = static_cast<const ::cxx::AtomicType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->elementType())));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readTypeString(std::intptr_t handle, int slot) -> std::string {
  switch (slot) {
    case TypeSlotBase + 0: {
      auto self = reinterpret_cast<const ::cxx::Type*>(handle);
      return std::string(to_string(self));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readTypeVal(std::intptr_t handle, int slot) -> val {
  switch (slot) {
    case UnresolvedNameTypeSlotBase + 2: {
      auto self = static_cast<const ::cxx::UnresolvedNameType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return [&]() -> val {
        auto result = val::array();
        result.call<void>(
            "push", val(static_cast<double>(
                        std::get<0>(self->sourceLocationRange()).index())));
        result.call<void>(
            "push", val(static_cast<double>(
                        std::get<1>(self->sourceLocationRange()).index())));
        return result;
      }();
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readTypeSize(std::intptr_t handle, int slot) -> int {
  switch (slot) {
    case FunctionTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::FunctionType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<int>(std::size(self->parameterTypes()));
    }
    case TemplateTypeParameterTypeSlotBase + 3: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      return static_cast<int>(std::size(self->templateParameters()));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readTypeItem(std::intptr_t handle, int slot, int index) -> double {
  switch (slot) {
    case FunctionTypeSlotBase + 1: {
      auto self = static_cast<const ::cxx::FunctionType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      const auto& container = self->parameterTypes();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(item)));
    }
    case TemplateTypeParameterTypeSlotBase + 3: {
      auto self = static_cast<const ::cxx::TemplateTypeParameterType*>(
          reinterpret_cast<const ::cxx::Type*>(handle));
      const auto& container = self->templateParameters();
      const auto& item = *std::next(std::begin(container), index);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(item)));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readName(std::intptr_t handle, int slot) -> double {
  switch (slot) {
    case NameSlotBase + 0: {
      auto self = reinterpret_cast<const ::cxx::Name*>(handle);
      return static_cast<double>(self->hashValue());
    }
    case IdentifierSlotBase + 0: {
      auto self = static_cast<const ::cxx::Identifier*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(self->isAnonymous());
    }
    case IdentifierSlotBase + 3: {
      auto self = static_cast<const ::cxx::Identifier*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(self->isBuiltinTypeTrait());
    }
    case IdentifierSlotBase + 4: {
      auto self = static_cast<const ::cxx::Identifier*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(self->builtinTypeTrait());
    }
    case IdentifierSlotBase + 5: {
      auto self = static_cast<const ::cxx::Identifier*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(self->builtinFunction());
    }
    case IdentifierSlotBase + 6: {
      auto self = static_cast<const ::cxx::Identifier*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(self->builtinTemplate());
    }
    case IdentifierSlotBase + 7: {
      auto self = static_cast<const ::cxx::Identifier*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(self->wellKnownName());
    }
    case OperatorIdSlotBase + 0: {
      auto self = static_cast<const ::cxx::OperatorId*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(self->op());
    }
    case DestructorIdSlotBase + 0: {
      auto self = static_cast<const ::cxx::DestructorId*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->name())));
    }
    case ConversionFunctionIdSlotBase + 0: {
      auto self = static_cast<const ::cxx::ConversionFunctionId*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->type())));
    }
    case TemplateIdSlotBase + 0: {
      auto self = static_cast<const ::cxx::TemplateId*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Name*>(self->name())));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readNameString(std::intptr_t handle, int slot) -> std::string {
  switch (slot) {
    case NameSlotBase + 1: {
      auto self = reinterpret_cast<const ::cxx::Name*>(handle);
      return std::string(to_string(self));
    }
    case IdentifierSlotBase + 1: {
      auto self = static_cast<const ::cxx::Identifier*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return std::string(self->name());
    }
    case IdentifierSlotBase + 2: {
      auto self = static_cast<const ::cxx::Identifier*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return std::string(self->value());
    }
    case LiteralOperatorIdSlotBase + 0: {
      auto self = static_cast<const ::cxx::LiteralOperatorId*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return std::string(self->name());
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readNameSize(std::intptr_t handle, int slot) -> int {
  switch (slot) {
    case TemplateIdSlotBase + 1: {
      auto self = static_cast<const ::cxx::TemplateId*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      return static_cast<int>(std::size(self->arguments()));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readNameItemVal(std::intptr_t handle, int slot, int index) -> val {
  switch (slot) {
    case TemplateIdSlotBase + 1: {
      auto self = static_cast<const ::cxx::TemplateId*>(
          reinterpret_cast<const ::cxx::Name*>(handle));
      const auto& container = self->arguments();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", item.index());
        switch (item.index()) {
          case 0:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Type*>(std::get<0>(item))))));
            break;
          case 1:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::Symbol*>(std::get<1>(item))))));
            break;
          case 2:
            result.set("value", [&]() -> val {
              auto result = val::object();
              result.set("index", std::get<2>(item).index());
              switch (std::get<2>(item).index()) {
                case 0:
                  result.set("value",
                             val(std::get<0>(std::get<2>(item)).toString()));
                  break;
                case 1:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          static_cast<const ::cxx::Literal*>(
                              std::get<1>(std::get<2>(item)))))));
                  break;
                case 2:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<2>(std::get<2>(item)))));
                  break;
                case 3:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<3>(std::get<2>(item)))));
                  break;
                case 4:
                  result.set(
                      "value",
                      val(static_cast<double>(std::get<4>(std::get<2>(item)))));
                  break;
                case 5:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<5>(std::get<2>(item)).get()))));
                  break;
                case 6:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<6>(std::get<2>(item)).get()))));
                  break;
                case 7:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<7>(std::get<2>(item)).get()))));
                  break;
                case 8:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<8>(std::get<2>(item)).get()))));
                  break;
                case 9:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<9>(std::get<2>(item)).get()))));
                  break;
                case 10:
                  result.set(
                      "value",
                      val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                          std::get<10>(std::get<2>(item)).get()))));
                  break;
                case 11:
                  result.set("value", val::undefined());
                  break;
              }
              return result;
            }());
            break;
          case 3:
            result.set(
                "value",
                val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                    static_cast<const ::cxx::AST*>(std::get<3>(item))))));
            break;
        }
        return result;
      }();
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readLiteral(std::intptr_t handle, int slot) -> double {
  switch (slot) {
    case LiteralSlotBase + 1: {
      auto self = reinterpret_cast<const ::cxx::Literal*>(handle);
      return static_cast<double>(self->hashCode());
    }
    case FloatLiteralSlotBase + 0: {
      auto self = static_cast<const ::cxx::FloatLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return static_cast<double>(self->floatValue());
    }
    case StringLiteralSlotBase + 0: {
      auto self = static_cast<const ::cxx::StringLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return static_cast<double>(self->encoding());
    }
    case StringLiteralSlotBase + 1: {
      auto self = static_cast<const ::cxx::StringLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return static_cast<double>(self->isRaw());
    }
    case StringLiteralSlotBase + 3: {
      auto self = static_cast<const ::cxx::StringLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return static_cast<double>(self->charCount());
    }
    case CharLiteralSlotBase + 0: {
      auto self = static_cast<const ::cxx::CharLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return static_cast<double>(self->charValue());
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readLiteralBigInt(std::intptr_t handle, int slot) -> std::int64_t {
  switch (slot) {
    case IntegerLiteralSlotBase + 0: {
      auto self = static_cast<const ::cxx::IntegerLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return static_cast<std::uint64_t>(self->integerValue());
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readLiteralString(std::intptr_t handle, int slot) -> std::string {
  switch (slot) {
    case LiteralSlotBase + 0: {
      auto self = reinterpret_cast<const ::cxx::Literal*>(handle);
      return std::string(self->value());
    }
    case StringLiteralSlotBase + 2: {
      auto self = static_cast<const ::cxx::StringLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return std::string(self->stringValue());
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readLiteralVal(std::intptr_t handle, int slot) -> val {
  switch (slot) {
    case IntegerLiteralSlotBase + 1: {
      auto self = static_cast<const ::cxx::IntegerLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return [&]() -> val {
        auto result = val::object();
        result.set("value",
                   val(static_cast<std::uint64_t>((self->components()).value)));
        result.set("integerPart",
                   val(std::string((self->components()).integerPart)));
        result.set("userSuffix",
                   val(std::string((self->components()).userSuffix)));
        result.set("radix",
                   val(static_cast<double>((self->components()).radix)));
        result.set("isUnsigned",
                   val(static_cast<double>((self->components()).isUnsigned)));
        result.set("isLongLong",
                   val(static_cast<double>((self->components()).isLongLong)));
        result.set("isLong",
                   val(static_cast<double>((self->components()).isLong)));
        result.set(
            "hasSizeSuffix",
            val(static_cast<double>((self->components()).hasSizeSuffix)));
        result.set("isWB", val(static_cast<double>((self->components()).isWB)));
        result.set("bitIntWidth",
                   val(static_cast<double>((self->components()).bitIntWidth)));
        return result;
      }();
    }
    case FloatLiteralSlotBase + 1: {
      auto self = static_cast<const ::cxx::FloatLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return [&]() -> val {
        auto result = val::object();
        result.set("value",
                   val(static_cast<double>((self->components()).value)));
        result.set("literalPart",
                   val(std::string((self->components()).literalPart)));
        result.set("userSuffix",
                   val(std::string((self->components()).userSuffix)));
        result.set("suffix",
                   val(static_cast<double>((self->components()).suffix)));
        result.set("isDouble",
                   val(static_cast<double>((self->components()).isDouble)));
        result.set("isFloat",
                   val(static_cast<double>((self->components()).isFloat)));
        result.set("isLongDouble",
                   val(static_cast<double>((self->components()).isLongDouble)));
        return result;
      }();
    }
    case StringLiteralSlotBase + 4: {
      auto self = static_cast<const ::cxx::StringLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return [&]() -> val {
        auto result = val::object();
        result.set("value", val(std::string((self->components()).value)));
        result.set("userSuffix",
                   val(std::string((self->components()).userSuffix)));
        result.set("encoding",
                   val(static_cast<double>((self->components()).encoding)));
        result.set("isRaw",
                   val(static_cast<double>((self->components()).isRaw)));
        return result;
      }();
    }
    case CharLiteralSlotBase + 1: {
      auto self = static_cast<const ::cxx::CharLiteral*>(
          reinterpret_cast<const ::cxx::Literal*>(handle));
      return [&]() -> val {
        auto result = val::object();
        result.set("value",
                   val(static_cast<double>((self->components()).value)));
        result.set("prefix", val(std::string((self->components()).prefix)));
        result.set("userSuffix",
                   val(std::string((self->components()).userSuffix)));
        return result;
      }();
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readMisc(std::intptr_t handle, int slot) -> double {
  switch (slot) {
    case ConstObjectSlotBase + 0: {
      auto self = reinterpret_cast<const ::cxx::ConstObject*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->type())));
    }
    case ConstObjectSlotBase + 2: {
      auto self = reinterpret_cast<const ::cxx::ConstObject*>(handle);
      return static_cast<double>(self->isUnion());
    }
    case ConstAddressSlotBase + 0: {
      auto self = reinterpret_cast<const ::cxx::ConstAddress*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Symbol*>(self->symbol())));
    }
    case ConstAddressSlotBase + 1: {
      auto self = reinterpret_cast<const ::cxx::ConstAddress*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Type*>(self->typeInfoFor())));
    }
    case ConstAddressSlotBase + 2: {
      auto self = reinterpret_cast<const ::cxx::ConstAddress*>(handle);
      return static_cast<double>(
          reinterpret_cast<std::intptr_t>(self->owner().get()));
    }
    case ConstAddressSlotBase + 3: {
      auto self = reinterpret_cast<const ::cxx::ConstAddress*>(handle);
      return static_cast<double>(reinterpret_cast<std::intptr_t>(
          static_cast<const ::cxx::Literal*>(self->stringLiteral())));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readMiscBigInt(std::intptr_t handle, int slot) -> std::int64_t {
  switch (slot) {
    case ConstAddressSlotBase + 4: {
      auto self = reinterpret_cast<const ::cxx::ConstAddress*>(handle);
      return static_cast<std::int64_t>(self->offset());
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readMiscString(std::intptr_t handle, int slot) -> std::string {
  switch (slot) {
    case ConstLabelAddressSlotBase + 0: {
      auto self = reinterpret_cast<const ::cxx::ConstLabelAddress*>(handle);
      return std::string(self->name());
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readMiscVal(std::intptr_t handle, int slot) -> val {
  switch (slot) {
    case ConstComplexSlotBase + 0: {
      auto self = reinterpret_cast<const ::cxx::ConstComplex*>(handle);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", self->real().index());
        switch (self->real().index()) {
          case 0:
            result.set("value", val(std::get<0>(self->real()).toString()));
            break;
          case 1:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           static_cast<const ::cxx::Literal*>(
                               std::get<1>(self->real()))))));
            break;
          case 2:
            result.set("value",
                       val(static_cast<double>(std::get<2>(self->real()))));
            break;
          case 3:
            result.set("value",
                       val(static_cast<double>(std::get<3>(self->real()))));
            break;
          case 4:
            result.set("value",
                       val(static_cast<double>(std::get<4>(self->real()))));
            break;
          case 5:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<5>(self->real()).get()))));
            break;
          case 6:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<6>(self->real()).get()))));
            break;
          case 7:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<7>(self->real()).get()))));
            break;
          case 8:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<8>(self->real()).get()))));
            break;
          case 9:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<9>(self->real()).get()))));
            break;
          case 10:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<10>(self->real()).get()))));
            break;
          case 11:
            result.set("value", val::undefined());
            break;
        }
        return result;
      }();
    }
    case ConstComplexSlotBase + 1: {
      auto self = reinterpret_cast<const ::cxx::ConstComplex*>(handle);
      return [&]() -> val {
        auto result = val::object();
        result.set("index", self->imag().index());
        switch (self->imag().index()) {
          case 0:
            result.set("value", val(std::get<0>(self->imag()).toString()));
            break;
          case 1:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           static_cast<const ::cxx::Literal*>(
                               std::get<1>(self->imag()))))));
            break;
          case 2:
            result.set("value",
                       val(static_cast<double>(std::get<2>(self->imag()))));
            break;
          case 3:
            result.set("value",
                       val(static_cast<double>(std::get<3>(self->imag()))));
            break;
          case 4:
            result.set("value",
                       val(static_cast<double>(std::get<4>(self->imag()))));
            break;
          case 5:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<5>(self->imag()).get()))));
            break;
          case 6:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<6>(self->imag()).get()))));
            break;
          case 7:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<7>(self->imag()).get()))));
            break;
          case 8:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<8>(self->imag()).get()))));
            break;
          case 9:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<9>(self->imag()).get()))));
            break;
          case 10:
            result.set("value",
                       val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                           std::get<10>(self->imag()).get()))));
            break;
          case 11:
            result.set("value", val::undefined());
            break;
        }
        return result;
      }();
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readMiscSize(std::intptr_t handle, int slot) -> int {
  switch (slot) {
    case ConstObjectSlotBase + 1: {
      auto self = reinterpret_cast<const ::cxx::ConstObject*>(handle);
      return static_cast<int>(std::size(self->members()));
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto readMiscItemVal(std::intptr_t handle, int slot, int index) -> val {
  switch (slot) {
    case ConstObjectSlotBase + 1: {
      auto self = reinterpret_cast<const ::cxx::ConstObject*>(handle);
      const auto& container = self->members();
      const auto& item = *std::next(std::begin(container), index);
      return [&]() -> val {
        auto result = val::object();
        result.set("symbol",
                   val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                       static_cast<const ::cxx::Symbol*>((item).symbol)))));
        result.set("value", [&]() -> val {
          auto result = val::object();
          result.set("index", (item).value.index());
          switch ((item).value.index()) {
            case 0:
              result.set("value", val(std::get<0>((item).value).toString()));
              break;
            case 1:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      static_cast<const ::cxx::Literal*>(
                          std::get<1>((item).value))))));
              break;
            case 2:
              result.set("value",
                         val(static_cast<double>(std::get<2>((item).value))));
              break;
            case 3:
              result.set("value",
                         val(static_cast<double>(std::get<3>((item).value))));
              break;
            case 4:
              result.set("value",
                         val(static_cast<double>(std::get<4>((item).value))));
              break;
            case 5:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<5>((item).value).get()))));
              break;
            case 6:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<6>((item).value).get()))));
              break;
            case 7:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<7>((item).value).get()))));
              break;
            case 8:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<8>((item).value).get()))));
              break;
            case 9:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<9>((item).value).get()))));
              break;
            case 10:
              result.set(
                  "value",
                  val(static_cast<double>(reinterpret_cast<std::intptr_t>(
                      std::get<10>((item).value).get()))));
              break;
            case 11:
              result.set("value", val::undefined());
              break;
          }
          return result;
        }());
        return result;
      }();
    }
  }
  cxx_runtime_error("unknown model slot");
}

auto getASTKind(std::intptr_t handle) -> int {
  return static_cast<int>(reinterpret_cast<const ::cxx::AST*>(handle)->kind());
}

auto getSymbolKind(std::intptr_t handle) -> int {
  return static_cast<int>(
      reinterpret_cast<const ::cxx::Symbol*>(handle)->kind());
}

auto getTypeKind(std::intptr_t handle) -> int {
  return static_cast<int>(reinterpret_cast<const ::cxx::Type*>(handle)->kind());
}

auto getNameKind(std::intptr_t handle) -> int {
  return static_cast<int>(reinterpret_cast<const ::cxx::Name*>(handle)->kind());
}

auto getListValue(std::intptr_t handle) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(
      reinterpret_cast<const List<AST*>*>(handle)->value);
}

auto getListNext(std::intptr_t handle) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(
      reinterpret_cast<const List<AST*>*>(handle)->next);
}

auto getUnitAST(std::intptr_t handle) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(static_cast<const ::cxx::AST*>(
      reinterpret_cast<TranslationUnit*>(handle)->ast()));
}

auto getGlobalScope(std::intptr_t handle) -> std::intptr_t {
  return reinterpret_cast<std::intptr_t>(static_cast<const ::cxx::Symbol*>(
      reinterpret_cast<TranslationUnit*>(handle)->globalScope()));
}

}  // namespace
}  // namespace cxx::js

EMSCRIPTEN_BINDINGS(cxx_reflection) {
  emscripten::function("readAST", &cxx::js::readAST);
  emscripten::function("readASTBigInt", &cxx::js::readASTBigInt);
  emscripten::function("readASTVal", &cxx::js::readASTVal);
  emscripten::function("readSymbol", &cxx::js::readSymbol);
  emscripten::function("readSymbolString", &cxx::js::readSymbolString);
  emscripten::function("readSymbolVal", &cxx::js::readSymbolVal);
  emscripten::function("readSymbolSize", &cxx::js::readSymbolSize);
  emscripten::function("readSymbolItem", &cxx::js::readSymbolItem);
  emscripten::function("readSymbolItemString", &cxx::js::readSymbolItemString);
  emscripten::function("readSymbolItemVal", &cxx::js::readSymbolItemVal);
  emscripten::function("readType", &cxx::js::readType);
  emscripten::function("readTypeString", &cxx::js::readTypeString);
  emscripten::function("readTypeVal", &cxx::js::readTypeVal);
  emscripten::function("readTypeSize", &cxx::js::readTypeSize);
  emscripten::function("readTypeItem", &cxx::js::readTypeItem);
  emscripten::function("readName", &cxx::js::readName);
  emscripten::function("readNameString", &cxx::js::readNameString);
  emscripten::function("readNameSize", &cxx::js::readNameSize);
  emscripten::function("readNameItemVal", &cxx::js::readNameItemVal);
  emscripten::function("readLiteral", &cxx::js::readLiteral);
  emscripten::function("readLiteralBigInt", &cxx::js::readLiteralBigInt);
  emscripten::function("readLiteralString", &cxx::js::readLiteralString);
  emscripten::function("readLiteralVal", &cxx::js::readLiteralVal);
  emscripten::function("readMisc", &cxx::js::readMisc);
  emscripten::function("readMiscBigInt", &cxx::js::readMiscBigInt);
  emscripten::function("readMiscString", &cxx::js::readMiscString);
  emscripten::function("readMiscVal", &cxx::js::readMiscVal);
  emscripten::function("readMiscSize", &cxx::js::readMiscSize);
  emscripten::function("readMiscItemVal", &cxx::js::readMiscItemVal);
  emscripten::function("getASTKind", &cxx::js::getASTKind);
  emscripten::function("getSymbolKind", &cxx::js::getSymbolKind);
  emscripten::function("getTypeKind", &cxx::js::getTypeKind);
  emscripten::function("getNameKind", &cxx::js::getNameKind);
  emscripten::function("getListValue", &cxx::js::getListValue);
  emscripten::function("getListNext", &cxx::js::getListNext);
  emscripten::function("getUnitAST", &cxx::js::getUnitAST);
  emscripten::function("getGlobalScope", &cxx::js::getGlobalScope);
}