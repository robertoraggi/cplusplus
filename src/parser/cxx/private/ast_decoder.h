// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx-ast-flatbuffers/ast_generated.h>
#include <cxx/ast_fwd.h>

#include <span>

namespace cxx {

class TranslationUnit;
class Control;
class Arena;

class ASTDecoder {
 public:
  explicit ASTDecoder(TranslationUnit* unit);

  auto operator()(std::span<const std::uint8_t> data) -> bool;

 private:
  auto decodeUnit(const void* ptr, io::Unit type) -> UnitAST*;
  auto decodeDeclaration(const void* ptr,
                         io::Declaration type) -> DeclarationAST*;
  auto decodeStatement(const void* ptr, io::Statement type) -> StatementAST*;
  auto decodeExpression(const void* ptr, io::Expression type) -> ExpressionAST*;
  auto decodeTemplateParameter(const void* ptr, io::TemplateParameter type)
      -> TemplateParameterAST*;
  auto decodeSpecifier(const void* ptr, io::Specifier type) -> SpecifierAST*;
  auto decodePtrOperator(const void* ptr,
                         io::PtrOperator type) -> PtrOperatorAST*;
  auto decodeCoreDeclarator(const void* ptr,
                            io::CoreDeclarator type) -> CoreDeclaratorAST*;
  auto decodeDeclaratorChunk(const void* ptr,
                             io::DeclaratorChunk type) -> DeclaratorChunkAST*;
  auto decodeUnqualifiedId(const void* ptr,
                           io::UnqualifiedId type) -> UnqualifiedIdAST*;
  auto decodeNestedNameSpecifier(const void* ptr, io::NestedNameSpecifier type)
      -> NestedNameSpecifierAST*;
  auto decodeFunctionBody(const void* ptr,
                          io::FunctionBody type) -> FunctionBodyAST*;
  auto decodeTemplateArgument(const void* ptr, io::TemplateArgument type)
      -> TemplateArgumentAST*;
  auto decodeExceptionSpecifier(const void* ptr, io::ExceptionSpecifier type)
      -> ExceptionSpecifierAST*;
  auto decodeRequirement(const void* ptr,
                         io::Requirement type) -> RequirementAST*;
  auto decodeNewInitializer(const void* ptr,
                            io::NewInitializer type) -> NewInitializerAST*;
  auto decodeMemInitializer(const void* ptr,
                            io::MemInitializer type) -> MemInitializerAST*;
  auto decodeLambdaCapture(const void* ptr,
                           io::LambdaCapture type) -> LambdaCaptureAST*;
  auto decodeExceptionDeclaration(const void* ptr,
                                  io::ExceptionDeclaration type)
      -> ExceptionDeclarationAST*;
  auto decodeAttributeSpecifier(const void* ptr, io::AttributeSpecifier type)
      -> AttributeSpecifierAST*;
  auto decodeAttributeToken(const void* ptr,
                            io::AttributeToken type) -> AttributeTokenAST*;

  auto decodeTranslationUnit(const io::TranslationUnit* node)
      -> TranslationUnitAST*;
  auto decodeModuleUnit(const io::ModuleUnit* node) -> ModuleUnitAST*;

  auto decodeSimpleDeclaration(const io::SimpleDeclaration* node)
      -> SimpleDeclarationAST*;
  auto decodeAsmDeclaration(const io::AsmDeclaration* node)
      -> AsmDeclarationAST*;
  auto decodeNamespaceAliasDefinition(const io::NamespaceAliasDefinition* node)
      -> NamespaceAliasDefinitionAST*;
  auto decodeUsingDeclaration(const io::UsingDeclaration* node)
      -> UsingDeclarationAST*;
  auto decodeUsingEnumDeclaration(const io::UsingEnumDeclaration* node)
      -> UsingEnumDeclarationAST*;
  auto decodeUsingDirective(const io::UsingDirective* node)
      -> UsingDirectiveAST*;
  auto decodeStaticAssertDeclaration(const io::StaticAssertDeclaration* node)
      -> StaticAssertDeclarationAST*;
  auto decodeAliasDeclaration(const io::AliasDeclaration* node)
      -> AliasDeclarationAST*;
  auto decodeOpaqueEnumDeclaration(const io::OpaqueEnumDeclaration* node)
      -> OpaqueEnumDeclarationAST*;
  auto decodeFunctionDefinition(const io::FunctionDefinition* node)
      -> FunctionDefinitionAST*;
  auto decodeTemplateDeclaration(const io::TemplateDeclaration* node)
      -> TemplateDeclarationAST*;
  auto decodeConceptDefinition(const io::ConceptDefinition* node)
      -> ConceptDefinitionAST*;
  auto decodeDeductionGuide(const io::DeductionGuide* node)
      -> DeductionGuideAST*;
  auto decodeExplicitInstantiation(const io::ExplicitInstantiation* node)
      -> ExplicitInstantiationAST*;
  auto decodeExportDeclaration(const io::ExportDeclaration* node)
      -> ExportDeclarationAST*;
  auto decodeExportCompoundDeclaration(
      const io::ExportCompoundDeclaration* node)
      -> ExportCompoundDeclarationAST*;
  auto decodeLinkageSpecification(const io::LinkageSpecification* node)
      -> LinkageSpecificationAST*;
  auto decodeNamespaceDefinition(const io::NamespaceDefinition* node)
      -> NamespaceDefinitionAST*;
  auto decodeEmptyDeclaration(const io::EmptyDeclaration* node)
      -> EmptyDeclarationAST*;
  auto decodeAttributeDeclaration(const io::AttributeDeclaration* node)
      -> AttributeDeclarationAST*;
  auto decodeModuleImportDeclaration(const io::ModuleImportDeclaration* node)
      -> ModuleImportDeclarationAST*;
  auto decodeParameterDeclaration(const io::ParameterDeclaration* node)
      -> ParameterDeclarationAST*;
  auto decodeAccessDeclaration(const io::AccessDeclaration* node)
      -> AccessDeclarationAST*;
  auto decodeForRangeDeclaration(const io::ForRangeDeclaration* node)
      -> ForRangeDeclarationAST*;
  auto decodeStructuredBindingDeclaration(
      const io::StructuredBindingDeclaration* node)
      -> StructuredBindingDeclarationAST*;
  auto decodeAsmOperand(const io::AsmOperand* node) -> AsmOperandAST*;
  auto decodeAsmQualifier(const io::AsmQualifier* node) -> AsmQualifierAST*;
  auto decodeAsmClobber(const io::AsmClobber* node) -> AsmClobberAST*;
  auto decodeAsmGotoLabel(const io::AsmGotoLabel* node) -> AsmGotoLabelAST*;

  auto decodeLabeledStatement(const io::LabeledStatement* node)
      -> LabeledStatementAST*;
  auto decodeCaseStatement(const io::CaseStatement* node) -> CaseStatementAST*;
  auto decodeDefaultStatement(const io::DefaultStatement* node)
      -> DefaultStatementAST*;
  auto decodeExpressionStatement(const io::ExpressionStatement* node)
      -> ExpressionStatementAST*;
  auto decodeCompoundStatement(const io::CompoundStatement* node)
      -> CompoundStatementAST*;
  auto decodeIfStatement(const io::IfStatement* node) -> IfStatementAST*;
  auto decodeConstevalIfStatement(const io::ConstevalIfStatement* node)
      -> ConstevalIfStatementAST*;
  auto decodeSwitchStatement(const io::SwitchStatement* node)
      -> SwitchStatementAST*;
  auto decodeWhileStatement(const io::WhileStatement* node)
      -> WhileStatementAST*;
  auto decodeDoStatement(const io::DoStatement* node) -> DoStatementAST*;
  auto decodeForRangeStatement(const io::ForRangeStatement* node)
      -> ForRangeStatementAST*;
  auto decodeForStatement(const io::ForStatement* node) -> ForStatementAST*;
  auto decodeBreakStatement(const io::BreakStatement* node)
      -> BreakStatementAST*;
  auto decodeContinueStatement(const io::ContinueStatement* node)
      -> ContinueStatementAST*;
  auto decodeReturnStatement(const io::ReturnStatement* node)
      -> ReturnStatementAST*;
  auto decodeCoroutineReturnStatement(const io::CoroutineReturnStatement* node)
      -> CoroutineReturnStatementAST*;
  auto decodeGotoStatement(const io::GotoStatement* node) -> GotoStatementAST*;
  auto decodeDeclarationStatement(const io::DeclarationStatement* node)
      -> DeclarationStatementAST*;
  auto decodeTryBlockStatement(const io::TryBlockStatement* node)
      -> TryBlockStatementAST*;

  auto decodeGeneratedLiteralExpression(
      const io::GeneratedLiteralExpression* node)
      -> GeneratedLiteralExpressionAST*;
  auto decodeCharLiteralExpression(const io::CharLiteralExpression* node)
      -> CharLiteralExpressionAST*;
  auto decodeBoolLiteralExpression(const io::BoolLiteralExpression* node)
      -> BoolLiteralExpressionAST*;
  auto decodeIntLiteralExpression(const io::IntLiteralExpression* node)
      -> IntLiteralExpressionAST*;
  auto decodeFloatLiteralExpression(const io::FloatLiteralExpression* node)
      -> FloatLiteralExpressionAST*;
  auto decodeNullptrLiteralExpression(const io::NullptrLiteralExpression* node)
      -> NullptrLiteralExpressionAST*;
  auto decodeStringLiteralExpression(const io::StringLiteralExpression* node)
      -> StringLiteralExpressionAST*;
  auto decodeUserDefinedStringLiteralExpression(
      const io::UserDefinedStringLiteralExpression* node)
      -> UserDefinedStringLiteralExpressionAST*;
  auto decodeThisExpression(const io::ThisExpression* node)
      -> ThisExpressionAST*;
  auto decodeNestedExpression(const io::NestedExpression* node)
      -> NestedExpressionAST*;
  auto decodeIdExpression(const io::IdExpression* node) -> IdExpressionAST*;
  auto decodeLambdaExpression(const io::LambdaExpression* node)
      -> LambdaExpressionAST*;
  auto decodeFoldExpression(const io::FoldExpression* node)
      -> FoldExpressionAST*;
  auto decodeRightFoldExpression(const io::RightFoldExpression* node)
      -> RightFoldExpressionAST*;
  auto decodeLeftFoldExpression(const io::LeftFoldExpression* node)
      -> LeftFoldExpressionAST*;
  auto decodeRequiresExpression(const io::RequiresExpression* node)
      -> RequiresExpressionAST*;
  auto decodeVaArgExpression(const io::VaArgExpression* node)
      -> VaArgExpressionAST*;
  auto decodeSubscriptExpression(const io::SubscriptExpression* node)
      -> SubscriptExpressionAST*;
  auto decodeCallExpression(const io::CallExpression* node)
      -> CallExpressionAST*;
  auto decodeTypeConstruction(const io::TypeConstruction* node)
      -> TypeConstructionAST*;
  auto decodeBracedTypeConstruction(const io::BracedTypeConstruction* node)
      -> BracedTypeConstructionAST*;
  auto decodeSpliceMemberExpression(const io::SpliceMemberExpression* node)
      -> SpliceMemberExpressionAST*;
  auto decodeMemberExpression(const io::MemberExpression* node)
      -> MemberExpressionAST*;
  auto decodePostIncrExpression(const io::PostIncrExpression* node)
      -> PostIncrExpressionAST*;
  auto decodeCppCastExpression(const io::CppCastExpression* node)
      -> CppCastExpressionAST*;
  auto decodeBuiltinBitCastExpression(const io::BuiltinBitCastExpression* node)
      -> BuiltinBitCastExpressionAST*;
  auto decodeTypeidExpression(const io::TypeidExpression* node)
      -> TypeidExpressionAST*;
  auto decodeTypeidOfTypeExpression(const io::TypeidOfTypeExpression* node)
      -> TypeidOfTypeExpressionAST*;
  auto decodeSpliceExpression(const io::SpliceExpression* node)
      -> SpliceExpressionAST*;
  auto decodeGlobalScopeReflectExpression(
      const io::GlobalScopeReflectExpression* node)
      -> GlobalScopeReflectExpressionAST*;
  auto decodeNamespaceReflectExpression(
      const io::NamespaceReflectExpression* node)
      -> NamespaceReflectExpressionAST*;
  auto decodeTypeIdReflectExpression(const io::TypeIdReflectExpression* node)
      -> TypeIdReflectExpressionAST*;
  auto decodeReflectExpression(const io::ReflectExpression* node)
      -> ReflectExpressionAST*;
  auto decodeUnaryExpression(const io::UnaryExpression* node)
      -> UnaryExpressionAST*;
  auto decodeAwaitExpression(const io::AwaitExpression* node)
      -> AwaitExpressionAST*;
  auto decodeSizeofExpression(const io::SizeofExpression* node)
      -> SizeofExpressionAST*;
  auto decodeSizeofTypeExpression(const io::SizeofTypeExpression* node)
      -> SizeofTypeExpressionAST*;
  auto decodeSizeofPackExpression(const io::SizeofPackExpression* node)
      -> SizeofPackExpressionAST*;
  auto decodeAlignofTypeExpression(const io::AlignofTypeExpression* node)
      -> AlignofTypeExpressionAST*;
  auto decodeAlignofExpression(const io::AlignofExpression* node)
      -> AlignofExpressionAST*;
  auto decodeNoexceptExpression(const io::NoexceptExpression* node)
      -> NoexceptExpressionAST*;
  auto decodeNewExpression(const io::NewExpression* node) -> NewExpressionAST*;
  auto decodeDeleteExpression(const io::DeleteExpression* node)
      -> DeleteExpressionAST*;
  auto decodeCastExpression(const io::CastExpression* node)
      -> CastExpressionAST*;
  auto decodeImplicitCastExpression(const io::ImplicitCastExpression* node)
      -> ImplicitCastExpressionAST*;
  auto decodeBinaryExpression(const io::BinaryExpression* node)
      -> BinaryExpressionAST*;
  auto decodeConditionalExpression(const io::ConditionalExpression* node)
      -> ConditionalExpressionAST*;
  auto decodeYieldExpression(const io::YieldExpression* node)
      -> YieldExpressionAST*;
  auto decodeThrowExpression(const io::ThrowExpression* node)
      -> ThrowExpressionAST*;
  auto decodeAssignmentExpression(const io::AssignmentExpression* node)
      -> AssignmentExpressionAST*;
  auto decodePackExpansionExpression(const io::PackExpansionExpression* node)
      -> PackExpansionExpressionAST*;
  auto decodeDesignatedInitializerClause(
      const io::DesignatedInitializerClause* node)
      -> DesignatedInitializerClauseAST*;
  auto decodeTypeTraitExpression(const io::TypeTraitExpression* node)
      -> TypeTraitExpressionAST*;
  auto decodeConditionExpression(const io::ConditionExpression* node)
      -> ConditionExpressionAST*;
  auto decodeEqualInitializer(const io::EqualInitializer* node)
      -> EqualInitializerAST*;
  auto decodeBracedInitList(const io::BracedInitList* node)
      -> BracedInitListAST*;
  auto decodeParenInitializer(const io::ParenInitializer* node)
      -> ParenInitializerAST*;

  auto decodeSplicer(const io::Splicer* node) -> SplicerAST*;
  auto decodeGlobalModuleFragment(const io::GlobalModuleFragment* node)
      -> GlobalModuleFragmentAST*;
  auto decodePrivateModuleFragment(const io::PrivateModuleFragment* node)
      -> PrivateModuleFragmentAST*;
  auto decodeModuleDeclaration(const io::ModuleDeclaration* node)
      -> ModuleDeclarationAST*;
  auto decodeModuleName(const io::ModuleName* node) -> ModuleNameAST*;
  auto decodeModuleQualifier(const io::ModuleQualifier* node)
      -> ModuleQualifierAST*;
  auto decodeModulePartition(const io::ModulePartition* node)
      -> ModulePartitionAST*;
  auto decodeImportName(const io::ImportName* node) -> ImportNameAST*;
  auto decodeInitDeclarator(const io::InitDeclarator* node)
      -> InitDeclaratorAST*;
  auto decodeDeclarator(const io::Declarator* node) -> DeclaratorAST*;
  auto decodeUsingDeclarator(const io::UsingDeclarator* node)
      -> UsingDeclaratorAST*;
  auto decodeEnumerator(const io::Enumerator* node) -> EnumeratorAST*;
  auto decodeTypeId(const io::TypeId* node) -> TypeIdAST*;
  auto decodeHandler(const io::Handler* node) -> HandlerAST*;
  auto decodeBaseSpecifier(const io::BaseSpecifier* node) -> BaseSpecifierAST*;
  auto decodeRequiresClause(const io::RequiresClause* node)
      -> RequiresClauseAST*;
  auto decodeParameterDeclarationClause(
      const io::ParameterDeclarationClause* node)
      -> ParameterDeclarationClauseAST*;
  auto decodeTrailingReturnType(const io::TrailingReturnType* node)
      -> TrailingReturnTypeAST*;
  auto decodeLambdaSpecifier(const io::LambdaSpecifier* node)
      -> LambdaSpecifierAST*;
  auto decodeTypeConstraint(const io::TypeConstraint* node)
      -> TypeConstraintAST*;
  auto decodeAttributeArgumentClause(const io::AttributeArgumentClause* node)
      -> AttributeArgumentClauseAST*;
  auto decodeAttribute(const io::Attribute* node) -> AttributeAST*;
  auto decodeAttributeUsingPrefix(const io::AttributeUsingPrefix* node)
      -> AttributeUsingPrefixAST*;
  auto decodeNewPlacement(const io::NewPlacement* node) -> NewPlacementAST*;
  auto decodeNestedNamespaceSpecifier(const io::NestedNamespaceSpecifier* node)
      -> NestedNamespaceSpecifierAST*;

  auto decodeTemplateTypeParameter(const io::TemplateTypeParameter* node)
      -> TemplateTypeParameterAST*;
  auto decodeNonTypeTemplateParameter(const io::NonTypeTemplateParameter* node)
      -> NonTypeTemplateParameterAST*;
  auto decodeTypenameTypeParameter(const io::TypenameTypeParameter* node)
      -> TypenameTypeParameterAST*;
  auto decodeConstraintTypeParameter(const io::ConstraintTypeParameter* node)
      -> ConstraintTypeParameterAST*;

  auto decodeGeneratedTypeSpecifier(const io::GeneratedTypeSpecifier* node)
      -> GeneratedTypeSpecifierAST*;
  auto decodeTypedefSpecifier(const io::TypedefSpecifier* node)
      -> TypedefSpecifierAST*;
  auto decodeFriendSpecifier(const io::FriendSpecifier* node)
      -> FriendSpecifierAST*;
  auto decodeConstevalSpecifier(const io::ConstevalSpecifier* node)
      -> ConstevalSpecifierAST*;
  auto decodeConstinitSpecifier(const io::ConstinitSpecifier* node)
      -> ConstinitSpecifierAST*;
  auto decodeConstexprSpecifier(const io::ConstexprSpecifier* node)
      -> ConstexprSpecifierAST*;
  auto decodeInlineSpecifier(const io::InlineSpecifier* node)
      -> InlineSpecifierAST*;
  auto decodeStaticSpecifier(const io::StaticSpecifier* node)
      -> StaticSpecifierAST*;
  auto decodeExternSpecifier(const io::ExternSpecifier* node)
      -> ExternSpecifierAST*;
  auto decodeThreadLocalSpecifier(const io::ThreadLocalSpecifier* node)
      -> ThreadLocalSpecifierAST*;
  auto decodeThreadSpecifier(const io::ThreadSpecifier* node)
      -> ThreadSpecifierAST*;
  auto decodeMutableSpecifier(const io::MutableSpecifier* node)
      -> MutableSpecifierAST*;
  auto decodeVirtualSpecifier(const io::VirtualSpecifier* node)
      -> VirtualSpecifierAST*;
  auto decodeExplicitSpecifier(const io::ExplicitSpecifier* node)
      -> ExplicitSpecifierAST*;
  auto decodeAutoTypeSpecifier(const io::AutoTypeSpecifier* node)
      -> AutoTypeSpecifierAST*;
  auto decodeVoidTypeSpecifier(const io::VoidTypeSpecifier* node)
      -> VoidTypeSpecifierAST*;
  auto decodeSizeTypeSpecifier(const io::SizeTypeSpecifier* node)
      -> SizeTypeSpecifierAST*;
  auto decodeSignTypeSpecifier(const io::SignTypeSpecifier* node)
      -> SignTypeSpecifierAST*;
  auto decodeVaListTypeSpecifier(const io::VaListTypeSpecifier* node)
      -> VaListTypeSpecifierAST*;
  auto decodeIntegralTypeSpecifier(const io::IntegralTypeSpecifier* node)
      -> IntegralTypeSpecifierAST*;
  auto decodeFloatingPointTypeSpecifier(
      const io::FloatingPointTypeSpecifier* node)
      -> FloatingPointTypeSpecifierAST*;
  auto decodeComplexTypeSpecifier(const io::ComplexTypeSpecifier* node)
      -> ComplexTypeSpecifierAST*;
  auto decodeNamedTypeSpecifier(const io::NamedTypeSpecifier* node)
      -> NamedTypeSpecifierAST*;
  auto decodeAtomicTypeSpecifier(const io::AtomicTypeSpecifier* node)
      -> AtomicTypeSpecifierAST*;
  auto decodeUnderlyingTypeSpecifier(const io::UnderlyingTypeSpecifier* node)
      -> UnderlyingTypeSpecifierAST*;
  auto decodeElaboratedTypeSpecifier(const io::ElaboratedTypeSpecifier* node)
      -> ElaboratedTypeSpecifierAST*;
  auto decodeDecltypeAutoSpecifier(const io::DecltypeAutoSpecifier* node)
      -> DecltypeAutoSpecifierAST*;
  auto decodeDecltypeSpecifier(const io::DecltypeSpecifier* node)
      -> DecltypeSpecifierAST*;
  auto decodePlaceholderTypeSpecifier(const io::PlaceholderTypeSpecifier* node)
      -> PlaceholderTypeSpecifierAST*;
  auto decodeConstQualifier(const io::ConstQualifier* node)
      -> ConstQualifierAST*;
  auto decodeVolatileQualifier(const io::VolatileQualifier* node)
      -> VolatileQualifierAST*;
  auto decodeRestrictQualifier(const io::RestrictQualifier* node)
      -> RestrictQualifierAST*;
  auto decodeEnumSpecifier(const io::EnumSpecifier* node) -> EnumSpecifierAST*;
  auto decodeClassSpecifier(const io::ClassSpecifier* node)
      -> ClassSpecifierAST*;
  auto decodeTypenameSpecifier(const io::TypenameSpecifier* node)
      -> TypenameSpecifierAST*;
  auto decodeSplicerTypeSpecifier(const io::SplicerTypeSpecifier* node)
      -> SplicerTypeSpecifierAST*;

  auto decodePointerOperator(const io::PointerOperator* node)
      -> PointerOperatorAST*;
  auto decodeReferenceOperator(const io::ReferenceOperator* node)
      -> ReferenceOperatorAST*;
  auto decodePtrToMemberOperator(const io::PtrToMemberOperator* node)
      -> PtrToMemberOperatorAST*;

  auto decodeBitfieldDeclarator(const io::BitfieldDeclarator* node)
      -> BitfieldDeclaratorAST*;
  auto decodeParameterPack(const io::ParameterPack* node) -> ParameterPackAST*;
  auto decodeIdDeclarator(const io::IdDeclarator* node) -> IdDeclaratorAST*;
  auto decodeNestedDeclarator(const io::NestedDeclarator* node)
      -> NestedDeclaratorAST*;

  auto decodeFunctionDeclaratorChunk(const io::FunctionDeclaratorChunk* node)
      -> FunctionDeclaratorChunkAST*;
  auto decodeArrayDeclaratorChunk(const io::ArrayDeclaratorChunk* node)
      -> ArrayDeclaratorChunkAST*;

  auto decodeNameId(const io::NameId* node) -> NameIdAST*;
  auto decodeDestructorId(const io::DestructorId* node) -> DestructorIdAST*;
  auto decodeDecltypeId(const io::DecltypeId* node) -> DecltypeIdAST*;
  auto decodeOperatorFunctionId(const io::OperatorFunctionId* node)
      -> OperatorFunctionIdAST*;
  auto decodeLiteralOperatorId(const io::LiteralOperatorId* node)
      -> LiteralOperatorIdAST*;
  auto decodeConversionFunctionId(const io::ConversionFunctionId* node)
      -> ConversionFunctionIdAST*;
  auto decodeSimpleTemplateId(const io::SimpleTemplateId* node)
      -> SimpleTemplateIdAST*;
  auto decodeLiteralOperatorTemplateId(
      const io::LiteralOperatorTemplateId* node)
      -> LiteralOperatorTemplateIdAST*;
  auto decodeOperatorFunctionTemplateId(
      const io::OperatorFunctionTemplateId* node)
      -> OperatorFunctionTemplateIdAST*;

  auto decodeGlobalNestedNameSpecifier(
      const io::GlobalNestedNameSpecifier* node)
      -> GlobalNestedNameSpecifierAST*;
  auto decodeSimpleNestedNameSpecifier(
      const io::SimpleNestedNameSpecifier* node)
      -> SimpleNestedNameSpecifierAST*;
  auto decodeDecltypeNestedNameSpecifier(
      const io::DecltypeNestedNameSpecifier* node)
      -> DecltypeNestedNameSpecifierAST*;
  auto decodeTemplateNestedNameSpecifier(
      const io::TemplateNestedNameSpecifier* node)
      -> TemplateNestedNameSpecifierAST*;

  auto decodeDefaultFunctionBody(const io::DefaultFunctionBody* node)
      -> DefaultFunctionBodyAST*;
  auto decodeCompoundStatementFunctionBody(
      const io::CompoundStatementFunctionBody* node)
      -> CompoundStatementFunctionBodyAST*;
  auto decodeTryStatementFunctionBody(const io::TryStatementFunctionBody* node)
      -> TryStatementFunctionBodyAST*;
  auto decodeDeleteFunctionBody(const io::DeleteFunctionBody* node)
      -> DeleteFunctionBodyAST*;

  auto decodeTypeTemplateArgument(const io::TypeTemplateArgument* node)
      -> TypeTemplateArgumentAST*;
  auto decodeExpressionTemplateArgument(
      const io::ExpressionTemplateArgument* node)
      -> ExpressionTemplateArgumentAST*;

  auto decodeThrowExceptionSpecifier(const io::ThrowExceptionSpecifier* node)
      -> ThrowExceptionSpecifierAST*;
  auto decodeNoexceptSpecifier(const io::NoexceptSpecifier* node)
      -> NoexceptSpecifierAST*;

  auto decodeSimpleRequirement(const io::SimpleRequirement* node)
      -> SimpleRequirementAST*;
  auto decodeCompoundRequirement(const io::CompoundRequirement* node)
      -> CompoundRequirementAST*;
  auto decodeTypeRequirement(const io::TypeRequirement* node)
      -> TypeRequirementAST*;
  auto decodeNestedRequirement(const io::NestedRequirement* node)
      -> NestedRequirementAST*;

  auto decodeNewParenInitializer(const io::NewParenInitializer* node)
      -> NewParenInitializerAST*;
  auto decodeNewBracedInitializer(const io::NewBracedInitializer* node)
      -> NewBracedInitializerAST*;

  auto decodeParenMemInitializer(const io::ParenMemInitializer* node)
      -> ParenMemInitializerAST*;
  auto decodeBracedMemInitializer(const io::BracedMemInitializer* node)
      -> BracedMemInitializerAST*;

  auto decodeThisLambdaCapture(const io::ThisLambdaCapture* node)
      -> ThisLambdaCaptureAST*;
  auto decodeDerefThisLambdaCapture(const io::DerefThisLambdaCapture* node)
      -> DerefThisLambdaCaptureAST*;
  auto decodeSimpleLambdaCapture(const io::SimpleLambdaCapture* node)
      -> SimpleLambdaCaptureAST*;
  auto decodeRefLambdaCapture(const io::RefLambdaCapture* node)
      -> RefLambdaCaptureAST*;
  auto decodeRefInitLambdaCapture(const io::RefInitLambdaCapture* node)
      -> RefInitLambdaCaptureAST*;
  auto decodeInitLambdaCapture(const io::InitLambdaCapture* node)
      -> InitLambdaCaptureAST*;

  auto decodeEllipsisExceptionDeclaration(
      const io::EllipsisExceptionDeclaration* node)
      -> EllipsisExceptionDeclarationAST*;
  auto decodeTypeExceptionDeclaration(const io::TypeExceptionDeclaration* node)
      -> TypeExceptionDeclarationAST*;

  auto decodeCxxAttribute(const io::CxxAttribute* node) -> CxxAttributeAST*;
  auto decodeGccAttribute(const io::GccAttribute* node) -> GccAttributeAST*;
  auto decodeAlignasAttribute(const io::AlignasAttribute* node)
      -> AlignasAttributeAST*;
  auto decodeAlignasTypeAttribute(const io::AlignasTypeAttribute* node)
      -> AlignasTypeAttributeAST*;
  auto decodeAsmAttribute(const io::AsmAttribute* node) -> AsmAttributeAST*;

  auto decodeScopedAttributeToken(const io::ScopedAttributeToken* node)
      -> ScopedAttributeTokenAST*;
  auto decodeSimpleAttributeToken(const io::SimpleAttributeToken* node)
      -> SimpleAttributeTokenAST*;

 private:
  TranslationUnit* unit_ = nullptr;
  Arena* pool_ = nullptr;
};

}  // namespace cxx
