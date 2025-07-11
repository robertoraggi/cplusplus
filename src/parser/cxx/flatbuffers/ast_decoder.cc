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

#include <cxx/private/ast_decoder.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/translation_unit.h>

#include <algorithm>

namespace cxx {

ASTDecoder::ASTDecoder(TranslationUnit* unit)
    : unit_(unit), pool_(unit->arena()) {}

auto ASTDecoder::operator()(std::span<const std::uint8_t> bytes) -> bool {
  auto serializedUnit = io::GetSerializedUnit(bytes.data());

  if (auto file_name = serializedUnit->file_name()) {
    unit_->setSource(std::string(), file_name->str());
  }

  auto ast = decodeUnit(serializedUnit->unit(), serializedUnit->unit_type());
  unit_->setAST(ast);

  return true;
}

auto ASTDecoder::decodeUnit(const void* ptr, io::Unit type) -> UnitAST* {
  switch (type) {
    case io::Unit_TranslationUnit:
      return decodeTranslationUnit(
          reinterpret_cast<const io::TranslationUnit*>(ptr));
    case io::Unit_ModuleUnit:
      return decodeModuleUnit(reinterpret_cast<const io::ModuleUnit*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeDeclaration(const void* ptr, io::Declaration type)
    -> DeclarationAST* {
  switch (type) {
    case io::Declaration_SimpleDeclaration:
      return decodeSimpleDeclaration(
          reinterpret_cast<const io::SimpleDeclaration*>(ptr));
    case io::Declaration_AsmDeclaration:
      return decodeAsmDeclaration(
          reinterpret_cast<const io::AsmDeclaration*>(ptr));
    case io::Declaration_NamespaceAliasDefinition:
      return decodeNamespaceAliasDefinition(
          reinterpret_cast<const io::NamespaceAliasDefinition*>(ptr));
    case io::Declaration_UsingDeclaration:
      return decodeUsingDeclaration(
          reinterpret_cast<const io::UsingDeclaration*>(ptr));
    case io::Declaration_UsingEnumDeclaration:
      return decodeUsingEnumDeclaration(
          reinterpret_cast<const io::UsingEnumDeclaration*>(ptr));
    case io::Declaration_UsingDirective:
      return decodeUsingDirective(
          reinterpret_cast<const io::UsingDirective*>(ptr));
    case io::Declaration_StaticAssertDeclaration:
      return decodeStaticAssertDeclaration(
          reinterpret_cast<const io::StaticAssertDeclaration*>(ptr));
    case io::Declaration_AliasDeclaration:
      return decodeAliasDeclaration(
          reinterpret_cast<const io::AliasDeclaration*>(ptr));
    case io::Declaration_OpaqueEnumDeclaration:
      return decodeOpaqueEnumDeclaration(
          reinterpret_cast<const io::OpaqueEnumDeclaration*>(ptr));
    case io::Declaration_FunctionDefinition:
      return decodeFunctionDefinition(
          reinterpret_cast<const io::FunctionDefinition*>(ptr));
    case io::Declaration_TemplateDeclaration:
      return decodeTemplateDeclaration(
          reinterpret_cast<const io::TemplateDeclaration*>(ptr));
    case io::Declaration_ConceptDefinition:
      return decodeConceptDefinition(
          reinterpret_cast<const io::ConceptDefinition*>(ptr));
    case io::Declaration_DeductionGuide:
      return decodeDeductionGuide(
          reinterpret_cast<const io::DeductionGuide*>(ptr));
    case io::Declaration_ExplicitInstantiation:
      return decodeExplicitInstantiation(
          reinterpret_cast<const io::ExplicitInstantiation*>(ptr));
    case io::Declaration_ExportDeclaration:
      return decodeExportDeclaration(
          reinterpret_cast<const io::ExportDeclaration*>(ptr));
    case io::Declaration_ExportCompoundDeclaration:
      return decodeExportCompoundDeclaration(
          reinterpret_cast<const io::ExportCompoundDeclaration*>(ptr));
    case io::Declaration_LinkageSpecification:
      return decodeLinkageSpecification(
          reinterpret_cast<const io::LinkageSpecification*>(ptr));
    case io::Declaration_NamespaceDefinition:
      return decodeNamespaceDefinition(
          reinterpret_cast<const io::NamespaceDefinition*>(ptr));
    case io::Declaration_EmptyDeclaration:
      return decodeEmptyDeclaration(
          reinterpret_cast<const io::EmptyDeclaration*>(ptr));
    case io::Declaration_AttributeDeclaration:
      return decodeAttributeDeclaration(
          reinterpret_cast<const io::AttributeDeclaration*>(ptr));
    case io::Declaration_ModuleImportDeclaration:
      return decodeModuleImportDeclaration(
          reinterpret_cast<const io::ModuleImportDeclaration*>(ptr));
    case io::Declaration_ParameterDeclaration:
      return decodeParameterDeclaration(
          reinterpret_cast<const io::ParameterDeclaration*>(ptr));
    case io::Declaration_AccessDeclaration:
      return decodeAccessDeclaration(
          reinterpret_cast<const io::AccessDeclaration*>(ptr));
    case io::Declaration_ForRangeDeclaration:
      return decodeForRangeDeclaration(
          reinterpret_cast<const io::ForRangeDeclaration*>(ptr));
    case io::Declaration_StructuredBindingDeclaration:
      return decodeStructuredBindingDeclaration(
          reinterpret_cast<const io::StructuredBindingDeclaration*>(ptr));
    case io::Declaration_AsmOperand:
      return decodeAsmOperand(reinterpret_cast<const io::AsmOperand*>(ptr));
    case io::Declaration_AsmQualifier:
      return decodeAsmQualifier(reinterpret_cast<const io::AsmQualifier*>(ptr));
    case io::Declaration_AsmClobber:
      return decodeAsmClobber(reinterpret_cast<const io::AsmClobber*>(ptr));
    case io::Declaration_AsmGotoLabel:
      return decodeAsmGotoLabel(reinterpret_cast<const io::AsmGotoLabel*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeStatement(const void* ptr, io::Statement type)
    -> StatementAST* {
  switch (type) {
    case io::Statement_LabeledStatement:
      return decodeLabeledStatement(
          reinterpret_cast<const io::LabeledStatement*>(ptr));
    case io::Statement_CaseStatement:
      return decodeCaseStatement(
          reinterpret_cast<const io::CaseStatement*>(ptr));
    case io::Statement_DefaultStatement:
      return decodeDefaultStatement(
          reinterpret_cast<const io::DefaultStatement*>(ptr));
    case io::Statement_ExpressionStatement:
      return decodeExpressionStatement(
          reinterpret_cast<const io::ExpressionStatement*>(ptr));
    case io::Statement_CompoundStatement:
      return decodeCompoundStatement(
          reinterpret_cast<const io::CompoundStatement*>(ptr));
    case io::Statement_IfStatement:
      return decodeIfStatement(reinterpret_cast<const io::IfStatement*>(ptr));
    case io::Statement_ConstevalIfStatement:
      return decodeConstevalIfStatement(
          reinterpret_cast<const io::ConstevalIfStatement*>(ptr));
    case io::Statement_SwitchStatement:
      return decodeSwitchStatement(
          reinterpret_cast<const io::SwitchStatement*>(ptr));
    case io::Statement_WhileStatement:
      return decodeWhileStatement(
          reinterpret_cast<const io::WhileStatement*>(ptr));
    case io::Statement_DoStatement:
      return decodeDoStatement(reinterpret_cast<const io::DoStatement*>(ptr));
    case io::Statement_ForRangeStatement:
      return decodeForRangeStatement(
          reinterpret_cast<const io::ForRangeStatement*>(ptr));
    case io::Statement_ForStatement:
      return decodeForStatement(reinterpret_cast<const io::ForStatement*>(ptr));
    case io::Statement_BreakStatement:
      return decodeBreakStatement(
          reinterpret_cast<const io::BreakStatement*>(ptr));
    case io::Statement_ContinueStatement:
      return decodeContinueStatement(
          reinterpret_cast<const io::ContinueStatement*>(ptr));
    case io::Statement_ReturnStatement:
      return decodeReturnStatement(
          reinterpret_cast<const io::ReturnStatement*>(ptr));
    case io::Statement_CoroutineReturnStatement:
      return decodeCoroutineReturnStatement(
          reinterpret_cast<const io::CoroutineReturnStatement*>(ptr));
    case io::Statement_GotoStatement:
      return decodeGotoStatement(
          reinterpret_cast<const io::GotoStatement*>(ptr));
    case io::Statement_DeclarationStatement:
      return decodeDeclarationStatement(
          reinterpret_cast<const io::DeclarationStatement*>(ptr));
    case io::Statement_TryBlockStatement:
      return decodeTryBlockStatement(
          reinterpret_cast<const io::TryBlockStatement*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeExpression(const void* ptr, io::Expression type)
    -> ExpressionAST* {
  switch (type) {
    case io::Expression_GeneratedLiteralExpression:
      return decodeGeneratedLiteralExpression(
          reinterpret_cast<const io::GeneratedLiteralExpression*>(ptr));
    case io::Expression_CharLiteralExpression:
      return decodeCharLiteralExpression(
          reinterpret_cast<const io::CharLiteralExpression*>(ptr));
    case io::Expression_BoolLiteralExpression:
      return decodeBoolLiteralExpression(
          reinterpret_cast<const io::BoolLiteralExpression*>(ptr));
    case io::Expression_IntLiteralExpression:
      return decodeIntLiteralExpression(
          reinterpret_cast<const io::IntLiteralExpression*>(ptr));
    case io::Expression_FloatLiteralExpression:
      return decodeFloatLiteralExpression(
          reinterpret_cast<const io::FloatLiteralExpression*>(ptr));
    case io::Expression_NullptrLiteralExpression:
      return decodeNullptrLiteralExpression(
          reinterpret_cast<const io::NullptrLiteralExpression*>(ptr));
    case io::Expression_StringLiteralExpression:
      return decodeStringLiteralExpression(
          reinterpret_cast<const io::StringLiteralExpression*>(ptr));
    case io::Expression_UserDefinedStringLiteralExpression:
      return decodeUserDefinedStringLiteralExpression(
          reinterpret_cast<const io::UserDefinedStringLiteralExpression*>(ptr));
    case io::Expression_ObjectLiteralExpression:
      return decodeObjectLiteralExpression(
          reinterpret_cast<const io::ObjectLiteralExpression*>(ptr));
    case io::Expression_ThisExpression:
      return decodeThisExpression(
          reinterpret_cast<const io::ThisExpression*>(ptr));
    case io::Expression_NestedStatementExpression:
      return decodeNestedStatementExpression(
          reinterpret_cast<const io::NestedStatementExpression*>(ptr));
    case io::Expression_NestedExpression:
      return decodeNestedExpression(
          reinterpret_cast<const io::NestedExpression*>(ptr));
    case io::Expression_IdExpression:
      return decodeIdExpression(reinterpret_cast<const io::IdExpression*>(ptr));
    case io::Expression_LambdaExpression:
      return decodeLambdaExpression(
          reinterpret_cast<const io::LambdaExpression*>(ptr));
    case io::Expression_FoldExpression:
      return decodeFoldExpression(
          reinterpret_cast<const io::FoldExpression*>(ptr));
    case io::Expression_RightFoldExpression:
      return decodeRightFoldExpression(
          reinterpret_cast<const io::RightFoldExpression*>(ptr));
    case io::Expression_LeftFoldExpression:
      return decodeLeftFoldExpression(
          reinterpret_cast<const io::LeftFoldExpression*>(ptr));
    case io::Expression_RequiresExpression:
      return decodeRequiresExpression(
          reinterpret_cast<const io::RequiresExpression*>(ptr));
    case io::Expression_VaArgExpression:
      return decodeVaArgExpression(
          reinterpret_cast<const io::VaArgExpression*>(ptr));
    case io::Expression_SubscriptExpression:
      return decodeSubscriptExpression(
          reinterpret_cast<const io::SubscriptExpression*>(ptr));
    case io::Expression_CallExpression:
      return decodeCallExpression(
          reinterpret_cast<const io::CallExpression*>(ptr));
    case io::Expression_TypeConstruction:
      return decodeTypeConstruction(
          reinterpret_cast<const io::TypeConstruction*>(ptr));
    case io::Expression_BracedTypeConstruction:
      return decodeBracedTypeConstruction(
          reinterpret_cast<const io::BracedTypeConstruction*>(ptr));
    case io::Expression_SpliceMemberExpression:
      return decodeSpliceMemberExpression(
          reinterpret_cast<const io::SpliceMemberExpression*>(ptr));
    case io::Expression_MemberExpression:
      return decodeMemberExpression(
          reinterpret_cast<const io::MemberExpression*>(ptr));
    case io::Expression_PostIncrExpression:
      return decodePostIncrExpression(
          reinterpret_cast<const io::PostIncrExpression*>(ptr));
    case io::Expression_CppCastExpression:
      return decodeCppCastExpression(
          reinterpret_cast<const io::CppCastExpression*>(ptr));
    case io::Expression_BuiltinBitCastExpression:
      return decodeBuiltinBitCastExpression(
          reinterpret_cast<const io::BuiltinBitCastExpression*>(ptr));
    case io::Expression_BuiltinOffsetofExpression:
      return decodeBuiltinOffsetofExpression(
          reinterpret_cast<const io::BuiltinOffsetofExpression*>(ptr));
    case io::Expression_TypeidExpression:
      return decodeTypeidExpression(
          reinterpret_cast<const io::TypeidExpression*>(ptr));
    case io::Expression_TypeidOfTypeExpression:
      return decodeTypeidOfTypeExpression(
          reinterpret_cast<const io::TypeidOfTypeExpression*>(ptr));
    case io::Expression_SpliceExpression:
      return decodeSpliceExpression(
          reinterpret_cast<const io::SpliceExpression*>(ptr));
    case io::Expression_GlobalScopeReflectExpression:
      return decodeGlobalScopeReflectExpression(
          reinterpret_cast<const io::GlobalScopeReflectExpression*>(ptr));
    case io::Expression_NamespaceReflectExpression:
      return decodeNamespaceReflectExpression(
          reinterpret_cast<const io::NamespaceReflectExpression*>(ptr));
    case io::Expression_TypeIdReflectExpression:
      return decodeTypeIdReflectExpression(
          reinterpret_cast<const io::TypeIdReflectExpression*>(ptr));
    case io::Expression_ReflectExpression:
      return decodeReflectExpression(
          reinterpret_cast<const io::ReflectExpression*>(ptr));
    case io::Expression_UnaryExpression:
      return decodeUnaryExpression(
          reinterpret_cast<const io::UnaryExpression*>(ptr));
    case io::Expression_AwaitExpression:
      return decodeAwaitExpression(
          reinterpret_cast<const io::AwaitExpression*>(ptr));
    case io::Expression_SizeofExpression:
      return decodeSizeofExpression(
          reinterpret_cast<const io::SizeofExpression*>(ptr));
    case io::Expression_SizeofTypeExpression:
      return decodeSizeofTypeExpression(
          reinterpret_cast<const io::SizeofTypeExpression*>(ptr));
    case io::Expression_SizeofPackExpression:
      return decodeSizeofPackExpression(
          reinterpret_cast<const io::SizeofPackExpression*>(ptr));
    case io::Expression_AlignofTypeExpression:
      return decodeAlignofTypeExpression(
          reinterpret_cast<const io::AlignofTypeExpression*>(ptr));
    case io::Expression_AlignofExpression:
      return decodeAlignofExpression(
          reinterpret_cast<const io::AlignofExpression*>(ptr));
    case io::Expression_NoexceptExpression:
      return decodeNoexceptExpression(
          reinterpret_cast<const io::NoexceptExpression*>(ptr));
    case io::Expression_NewExpression:
      return decodeNewExpression(
          reinterpret_cast<const io::NewExpression*>(ptr));
    case io::Expression_DeleteExpression:
      return decodeDeleteExpression(
          reinterpret_cast<const io::DeleteExpression*>(ptr));
    case io::Expression_CastExpression:
      return decodeCastExpression(
          reinterpret_cast<const io::CastExpression*>(ptr));
    case io::Expression_ImplicitCastExpression:
      return decodeImplicitCastExpression(
          reinterpret_cast<const io::ImplicitCastExpression*>(ptr));
    case io::Expression_BinaryExpression:
      return decodeBinaryExpression(
          reinterpret_cast<const io::BinaryExpression*>(ptr));
    case io::Expression_ConditionalExpression:
      return decodeConditionalExpression(
          reinterpret_cast<const io::ConditionalExpression*>(ptr));
    case io::Expression_YieldExpression:
      return decodeYieldExpression(
          reinterpret_cast<const io::YieldExpression*>(ptr));
    case io::Expression_ThrowExpression:
      return decodeThrowExpression(
          reinterpret_cast<const io::ThrowExpression*>(ptr));
    case io::Expression_AssignmentExpression:
      return decodeAssignmentExpression(
          reinterpret_cast<const io::AssignmentExpression*>(ptr));
    case io::Expression_PackExpansionExpression:
      return decodePackExpansionExpression(
          reinterpret_cast<const io::PackExpansionExpression*>(ptr));
    case io::Expression_DesignatedInitializerClause:
      return decodeDesignatedInitializerClause(
          reinterpret_cast<const io::DesignatedInitializerClause*>(ptr));
    case io::Expression_TypeTraitExpression:
      return decodeTypeTraitExpression(
          reinterpret_cast<const io::TypeTraitExpression*>(ptr));
    case io::Expression_ConditionExpression:
      return decodeConditionExpression(
          reinterpret_cast<const io::ConditionExpression*>(ptr));
    case io::Expression_EqualInitializer:
      return decodeEqualInitializer(
          reinterpret_cast<const io::EqualInitializer*>(ptr));
    case io::Expression_BracedInitList:
      return decodeBracedInitList(
          reinterpret_cast<const io::BracedInitList*>(ptr));
    case io::Expression_ParenInitializer:
      return decodeParenInitializer(
          reinterpret_cast<const io::ParenInitializer*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeDesignator(const void* ptr, io::Designator type)
    -> DesignatorAST* {
  switch (type) {
    case io::Designator_DotDesignator:
      return decodeDotDesignator(
          reinterpret_cast<const io::DotDesignator*>(ptr));
    case io::Designator_SubscriptDesignator:
      return decodeSubscriptDesignator(
          reinterpret_cast<const io::SubscriptDesignator*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeTemplateParameter(const void* ptr,
                                         io::TemplateParameter type)
    -> TemplateParameterAST* {
  switch (type) {
    case io::TemplateParameter_TemplateTypeParameter:
      return decodeTemplateTypeParameter(
          reinterpret_cast<const io::TemplateTypeParameter*>(ptr));
    case io::TemplateParameter_NonTypeTemplateParameter:
      return decodeNonTypeTemplateParameter(
          reinterpret_cast<const io::NonTypeTemplateParameter*>(ptr));
    case io::TemplateParameter_TypenameTypeParameter:
      return decodeTypenameTypeParameter(
          reinterpret_cast<const io::TypenameTypeParameter*>(ptr));
    case io::TemplateParameter_ConstraintTypeParameter:
      return decodeConstraintTypeParameter(
          reinterpret_cast<const io::ConstraintTypeParameter*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeSpecifier(const void* ptr, io::Specifier type)
    -> SpecifierAST* {
  switch (type) {
    case io::Specifier_GeneratedTypeSpecifier:
      return decodeGeneratedTypeSpecifier(
          reinterpret_cast<const io::GeneratedTypeSpecifier*>(ptr));
    case io::Specifier_TypedefSpecifier:
      return decodeTypedefSpecifier(
          reinterpret_cast<const io::TypedefSpecifier*>(ptr));
    case io::Specifier_FriendSpecifier:
      return decodeFriendSpecifier(
          reinterpret_cast<const io::FriendSpecifier*>(ptr));
    case io::Specifier_ConstevalSpecifier:
      return decodeConstevalSpecifier(
          reinterpret_cast<const io::ConstevalSpecifier*>(ptr));
    case io::Specifier_ConstinitSpecifier:
      return decodeConstinitSpecifier(
          reinterpret_cast<const io::ConstinitSpecifier*>(ptr));
    case io::Specifier_ConstexprSpecifier:
      return decodeConstexprSpecifier(
          reinterpret_cast<const io::ConstexprSpecifier*>(ptr));
    case io::Specifier_InlineSpecifier:
      return decodeInlineSpecifier(
          reinterpret_cast<const io::InlineSpecifier*>(ptr));
    case io::Specifier_NoreturnSpecifier:
      return decodeNoreturnSpecifier(
          reinterpret_cast<const io::NoreturnSpecifier*>(ptr));
    case io::Specifier_StaticSpecifier:
      return decodeStaticSpecifier(
          reinterpret_cast<const io::StaticSpecifier*>(ptr));
    case io::Specifier_ExternSpecifier:
      return decodeExternSpecifier(
          reinterpret_cast<const io::ExternSpecifier*>(ptr));
    case io::Specifier_RegisterSpecifier:
      return decodeRegisterSpecifier(
          reinterpret_cast<const io::RegisterSpecifier*>(ptr));
    case io::Specifier_ThreadLocalSpecifier:
      return decodeThreadLocalSpecifier(
          reinterpret_cast<const io::ThreadLocalSpecifier*>(ptr));
    case io::Specifier_ThreadSpecifier:
      return decodeThreadSpecifier(
          reinterpret_cast<const io::ThreadSpecifier*>(ptr));
    case io::Specifier_MutableSpecifier:
      return decodeMutableSpecifier(
          reinterpret_cast<const io::MutableSpecifier*>(ptr));
    case io::Specifier_VirtualSpecifier:
      return decodeVirtualSpecifier(
          reinterpret_cast<const io::VirtualSpecifier*>(ptr));
    case io::Specifier_ExplicitSpecifier:
      return decodeExplicitSpecifier(
          reinterpret_cast<const io::ExplicitSpecifier*>(ptr));
    case io::Specifier_AutoTypeSpecifier:
      return decodeAutoTypeSpecifier(
          reinterpret_cast<const io::AutoTypeSpecifier*>(ptr));
    case io::Specifier_VoidTypeSpecifier:
      return decodeVoidTypeSpecifier(
          reinterpret_cast<const io::VoidTypeSpecifier*>(ptr));
    case io::Specifier_SizeTypeSpecifier:
      return decodeSizeTypeSpecifier(
          reinterpret_cast<const io::SizeTypeSpecifier*>(ptr));
    case io::Specifier_SignTypeSpecifier:
      return decodeSignTypeSpecifier(
          reinterpret_cast<const io::SignTypeSpecifier*>(ptr));
    case io::Specifier_VaListTypeSpecifier:
      return decodeVaListTypeSpecifier(
          reinterpret_cast<const io::VaListTypeSpecifier*>(ptr));
    case io::Specifier_IntegralTypeSpecifier:
      return decodeIntegralTypeSpecifier(
          reinterpret_cast<const io::IntegralTypeSpecifier*>(ptr));
    case io::Specifier_FloatingPointTypeSpecifier:
      return decodeFloatingPointTypeSpecifier(
          reinterpret_cast<const io::FloatingPointTypeSpecifier*>(ptr));
    case io::Specifier_ComplexTypeSpecifier:
      return decodeComplexTypeSpecifier(
          reinterpret_cast<const io::ComplexTypeSpecifier*>(ptr));
    case io::Specifier_NamedTypeSpecifier:
      return decodeNamedTypeSpecifier(
          reinterpret_cast<const io::NamedTypeSpecifier*>(ptr));
    case io::Specifier_AtomicTypeSpecifier:
      return decodeAtomicTypeSpecifier(
          reinterpret_cast<const io::AtomicTypeSpecifier*>(ptr));
    case io::Specifier_UnderlyingTypeSpecifier:
      return decodeUnderlyingTypeSpecifier(
          reinterpret_cast<const io::UnderlyingTypeSpecifier*>(ptr));
    case io::Specifier_ElaboratedTypeSpecifier:
      return decodeElaboratedTypeSpecifier(
          reinterpret_cast<const io::ElaboratedTypeSpecifier*>(ptr));
    case io::Specifier_DecltypeAutoSpecifier:
      return decodeDecltypeAutoSpecifier(
          reinterpret_cast<const io::DecltypeAutoSpecifier*>(ptr));
    case io::Specifier_DecltypeSpecifier:
      return decodeDecltypeSpecifier(
          reinterpret_cast<const io::DecltypeSpecifier*>(ptr));
    case io::Specifier_PlaceholderTypeSpecifier:
      return decodePlaceholderTypeSpecifier(
          reinterpret_cast<const io::PlaceholderTypeSpecifier*>(ptr));
    case io::Specifier_ConstQualifier:
      return decodeConstQualifier(
          reinterpret_cast<const io::ConstQualifier*>(ptr));
    case io::Specifier_VolatileQualifier:
      return decodeVolatileQualifier(
          reinterpret_cast<const io::VolatileQualifier*>(ptr));
    case io::Specifier_RestrictQualifier:
      return decodeRestrictQualifier(
          reinterpret_cast<const io::RestrictQualifier*>(ptr));
    case io::Specifier_EnumSpecifier:
      return decodeEnumSpecifier(
          reinterpret_cast<const io::EnumSpecifier*>(ptr));
    case io::Specifier_ClassSpecifier:
      return decodeClassSpecifier(
          reinterpret_cast<const io::ClassSpecifier*>(ptr));
    case io::Specifier_TypenameSpecifier:
      return decodeTypenameSpecifier(
          reinterpret_cast<const io::TypenameSpecifier*>(ptr));
    case io::Specifier_SplicerTypeSpecifier:
      return decodeSplicerTypeSpecifier(
          reinterpret_cast<const io::SplicerTypeSpecifier*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodePtrOperator(const void* ptr, io::PtrOperator type)
    -> PtrOperatorAST* {
  switch (type) {
    case io::PtrOperator_PointerOperator:
      return decodePointerOperator(
          reinterpret_cast<const io::PointerOperator*>(ptr));
    case io::PtrOperator_ReferenceOperator:
      return decodeReferenceOperator(
          reinterpret_cast<const io::ReferenceOperator*>(ptr));
    case io::PtrOperator_PtrToMemberOperator:
      return decodePtrToMemberOperator(
          reinterpret_cast<const io::PtrToMemberOperator*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeCoreDeclarator(const void* ptr, io::CoreDeclarator type)
    -> CoreDeclaratorAST* {
  switch (type) {
    case io::CoreDeclarator_BitfieldDeclarator:
      return decodeBitfieldDeclarator(
          reinterpret_cast<const io::BitfieldDeclarator*>(ptr));
    case io::CoreDeclarator_ParameterPack:
      return decodeParameterPack(
          reinterpret_cast<const io::ParameterPack*>(ptr));
    case io::CoreDeclarator_IdDeclarator:
      return decodeIdDeclarator(reinterpret_cast<const io::IdDeclarator*>(ptr));
    case io::CoreDeclarator_NestedDeclarator:
      return decodeNestedDeclarator(
          reinterpret_cast<const io::NestedDeclarator*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeDeclaratorChunk(const void* ptr,
                                       io::DeclaratorChunk type)
    -> DeclaratorChunkAST* {
  switch (type) {
    case io::DeclaratorChunk_FunctionDeclaratorChunk:
      return decodeFunctionDeclaratorChunk(
          reinterpret_cast<const io::FunctionDeclaratorChunk*>(ptr));
    case io::DeclaratorChunk_ArrayDeclaratorChunk:
      return decodeArrayDeclaratorChunk(
          reinterpret_cast<const io::ArrayDeclaratorChunk*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeUnqualifiedId(const void* ptr, io::UnqualifiedId type)
    -> UnqualifiedIdAST* {
  switch (type) {
    case io::UnqualifiedId_NameId:
      return decodeNameId(reinterpret_cast<const io::NameId*>(ptr));
    case io::UnqualifiedId_DestructorId:
      return decodeDestructorId(reinterpret_cast<const io::DestructorId*>(ptr));
    case io::UnqualifiedId_DecltypeId:
      return decodeDecltypeId(reinterpret_cast<const io::DecltypeId*>(ptr));
    case io::UnqualifiedId_OperatorFunctionId:
      return decodeOperatorFunctionId(
          reinterpret_cast<const io::OperatorFunctionId*>(ptr));
    case io::UnqualifiedId_LiteralOperatorId:
      return decodeLiteralOperatorId(
          reinterpret_cast<const io::LiteralOperatorId*>(ptr));
    case io::UnqualifiedId_ConversionFunctionId:
      return decodeConversionFunctionId(
          reinterpret_cast<const io::ConversionFunctionId*>(ptr));
    case io::UnqualifiedId_SimpleTemplateId:
      return decodeSimpleTemplateId(
          reinterpret_cast<const io::SimpleTemplateId*>(ptr));
    case io::UnqualifiedId_LiteralOperatorTemplateId:
      return decodeLiteralOperatorTemplateId(
          reinterpret_cast<const io::LiteralOperatorTemplateId*>(ptr));
    case io::UnqualifiedId_OperatorFunctionTemplateId:
      return decodeOperatorFunctionTemplateId(
          reinterpret_cast<const io::OperatorFunctionTemplateId*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeNestedNameSpecifier(const void* ptr,
                                           io::NestedNameSpecifier type)
    -> NestedNameSpecifierAST* {
  switch (type) {
    case io::NestedNameSpecifier_GlobalNestedNameSpecifier:
      return decodeGlobalNestedNameSpecifier(
          reinterpret_cast<const io::GlobalNestedNameSpecifier*>(ptr));
    case io::NestedNameSpecifier_SimpleNestedNameSpecifier:
      return decodeSimpleNestedNameSpecifier(
          reinterpret_cast<const io::SimpleNestedNameSpecifier*>(ptr));
    case io::NestedNameSpecifier_DecltypeNestedNameSpecifier:
      return decodeDecltypeNestedNameSpecifier(
          reinterpret_cast<const io::DecltypeNestedNameSpecifier*>(ptr));
    case io::NestedNameSpecifier_TemplateNestedNameSpecifier:
      return decodeTemplateNestedNameSpecifier(
          reinterpret_cast<const io::TemplateNestedNameSpecifier*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeFunctionBody(const void* ptr, io::FunctionBody type)
    -> FunctionBodyAST* {
  switch (type) {
    case io::FunctionBody_DefaultFunctionBody:
      return decodeDefaultFunctionBody(
          reinterpret_cast<const io::DefaultFunctionBody*>(ptr));
    case io::FunctionBody_CompoundStatementFunctionBody:
      return decodeCompoundStatementFunctionBody(
          reinterpret_cast<const io::CompoundStatementFunctionBody*>(ptr));
    case io::FunctionBody_TryStatementFunctionBody:
      return decodeTryStatementFunctionBody(
          reinterpret_cast<const io::TryStatementFunctionBody*>(ptr));
    case io::FunctionBody_DeleteFunctionBody:
      return decodeDeleteFunctionBody(
          reinterpret_cast<const io::DeleteFunctionBody*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeTemplateArgument(const void* ptr,
                                        io::TemplateArgument type)
    -> TemplateArgumentAST* {
  switch (type) {
    case io::TemplateArgument_TypeTemplateArgument:
      return decodeTypeTemplateArgument(
          reinterpret_cast<const io::TypeTemplateArgument*>(ptr));
    case io::TemplateArgument_ExpressionTemplateArgument:
      return decodeExpressionTemplateArgument(
          reinterpret_cast<const io::ExpressionTemplateArgument*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeExceptionSpecifier(const void* ptr,
                                          io::ExceptionSpecifier type)
    -> ExceptionSpecifierAST* {
  switch (type) {
    case io::ExceptionSpecifier_ThrowExceptionSpecifier:
      return decodeThrowExceptionSpecifier(
          reinterpret_cast<const io::ThrowExceptionSpecifier*>(ptr));
    case io::ExceptionSpecifier_NoexceptSpecifier:
      return decodeNoexceptSpecifier(
          reinterpret_cast<const io::NoexceptSpecifier*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeRequirement(const void* ptr, io::Requirement type)
    -> RequirementAST* {
  switch (type) {
    case io::Requirement_SimpleRequirement:
      return decodeSimpleRequirement(
          reinterpret_cast<const io::SimpleRequirement*>(ptr));
    case io::Requirement_CompoundRequirement:
      return decodeCompoundRequirement(
          reinterpret_cast<const io::CompoundRequirement*>(ptr));
    case io::Requirement_TypeRequirement:
      return decodeTypeRequirement(
          reinterpret_cast<const io::TypeRequirement*>(ptr));
    case io::Requirement_NestedRequirement:
      return decodeNestedRequirement(
          reinterpret_cast<const io::NestedRequirement*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeNewInitializer(const void* ptr, io::NewInitializer type)
    -> NewInitializerAST* {
  switch (type) {
    case io::NewInitializer_NewParenInitializer:
      return decodeNewParenInitializer(
          reinterpret_cast<const io::NewParenInitializer*>(ptr));
    case io::NewInitializer_NewBracedInitializer:
      return decodeNewBracedInitializer(
          reinterpret_cast<const io::NewBracedInitializer*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeMemInitializer(const void* ptr, io::MemInitializer type)
    -> MemInitializerAST* {
  switch (type) {
    case io::MemInitializer_ParenMemInitializer:
      return decodeParenMemInitializer(
          reinterpret_cast<const io::ParenMemInitializer*>(ptr));
    case io::MemInitializer_BracedMemInitializer:
      return decodeBracedMemInitializer(
          reinterpret_cast<const io::BracedMemInitializer*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeLambdaCapture(const void* ptr, io::LambdaCapture type)
    -> LambdaCaptureAST* {
  switch (type) {
    case io::LambdaCapture_ThisLambdaCapture:
      return decodeThisLambdaCapture(
          reinterpret_cast<const io::ThisLambdaCapture*>(ptr));
    case io::LambdaCapture_DerefThisLambdaCapture:
      return decodeDerefThisLambdaCapture(
          reinterpret_cast<const io::DerefThisLambdaCapture*>(ptr));
    case io::LambdaCapture_SimpleLambdaCapture:
      return decodeSimpleLambdaCapture(
          reinterpret_cast<const io::SimpleLambdaCapture*>(ptr));
    case io::LambdaCapture_RefLambdaCapture:
      return decodeRefLambdaCapture(
          reinterpret_cast<const io::RefLambdaCapture*>(ptr));
    case io::LambdaCapture_RefInitLambdaCapture:
      return decodeRefInitLambdaCapture(
          reinterpret_cast<const io::RefInitLambdaCapture*>(ptr));
    case io::LambdaCapture_InitLambdaCapture:
      return decodeInitLambdaCapture(
          reinterpret_cast<const io::InitLambdaCapture*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeExceptionDeclaration(const void* ptr,
                                            io::ExceptionDeclaration type)
    -> ExceptionDeclarationAST* {
  switch (type) {
    case io::ExceptionDeclaration_EllipsisExceptionDeclaration:
      return decodeEllipsisExceptionDeclaration(
          reinterpret_cast<const io::EllipsisExceptionDeclaration*>(ptr));
    case io::ExceptionDeclaration_TypeExceptionDeclaration:
      return decodeTypeExceptionDeclaration(
          reinterpret_cast<const io::TypeExceptionDeclaration*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeAttributeSpecifier(const void* ptr,
                                          io::AttributeSpecifier type)
    -> AttributeSpecifierAST* {
  switch (type) {
    case io::AttributeSpecifier_CxxAttribute:
      return decodeCxxAttribute(reinterpret_cast<const io::CxxAttribute*>(ptr));
    case io::AttributeSpecifier_GccAttribute:
      return decodeGccAttribute(reinterpret_cast<const io::GccAttribute*>(ptr));
    case io::AttributeSpecifier_AlignasAttribute:
      return decodeAlignasAttribute(
          reinterpret_cast<const io::AlignasAttribute*>(ptr));
    case io::AttributeSpecifier_AlignasTypeAttribute:
      return decodeAlignasTypeAttribute(
          reinterpret_cast<const io::AlignasTypeAttribute*>(ptr));
    case io::AttributeSpecifier_AsmAttribute:
      return decodeAsmAttribute(reinterpret_cast<const io::AsmAttribute*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeAttributeToken(const void* ptr, io::AttributeToken type)
    -> AttributeTokenAST* {
  switch (type) {
    case io::AttributeToken_ScopedAttributeToken:
      return decodeScopedAttributeToken(
          reinterpret_cast<const io::ScopedAttributeToken*>(ptr));
    case io::AttributeToken_SimpleAttributeToken:
      return decodeSimpleAttributeToken(
          reinterpret_cast<const io::SimpleAttributeToken*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeTranslationUnit(const io::TranslationUnit* node)
    -> TranslationUnitAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TranslationUnitAST();
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::uint32_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeModuleUnit(const io::ModuleUnit* node)
    -> ModuleUnitAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ModuleUnitAST();
  ast->globalModuleFragment =
      decodeGlobalModuleFragment(node->global_module_fragment());
  ast->moduleDeclaration = decodeModuleDeclaration(node->module_declaration());
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::uint32_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->privateModuleFragment =
      decodePrivateModuleFragment(node->private_module_fragment());
  return ast;
}

auto ASTDecoder::decodeSimpleDeclaration(const io::SimpleDeclaration* node)
    -> SimpleDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->decl_specifier_list()) {
    auto* inserter = &ast->declSpecifierList;
    for (std::uint32_t i = 0; i < node->decl_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->decl_specifier_list()->Get(i),
          io::Specifier(node->decl_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->init_declarator_list()) {
    auto* inserter = &ast->initDeclaratorList;
    for (std::uint32_t i = 0; i < node->init_declarator_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeInitDeclarator(node->init_declarator_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeAsmDeclaration(const io::AsmDeclaration* node)
    -> AsmDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AsmDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->asm_qualifier_list()) {
    auto* inserter = &ast->asmQualifierList;
    for (std::uint32_t i = 0; i < node->asm_qualifier_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeAsmQualifier(node->asm_qualifier_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->asmLoc = SourceLocation(node->asm_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->literalLoc = SourceLocation(node->literal_loc());
  if (node->output_operand_list()) {
    auto* inserter = &ast->outputOperandList;
    for (std::uint32_t i = 0; i < node->output_operand_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeAsmOperand(node->output_operand_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  if (node->input_operand_list()) {
    auto* inserter = &ast->inputOperandList;
    for (std::uint32_t i = 0; i < node->input_operand_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeAsmOperand(node->input_operand_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  if (node->clobber_list()) {
    auto* inserter = &ast->clobberList;
    for (std::uint32_t i = 0; i < node->clobber_list()->size(); ++i) {
      *inserter =
          new (pool_) List(decodeAsmClobber(node->clobber_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  if (node->goto_label_list()) {
    auto* inserter = &ast->gotoLabelList;
    for (std::uint32_t i = 0; i < node->goto_label_list()->size(); ++i) {
      *inserter =
          new (pool_) List(decodeAsmGotoLabel(node->goto_label_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeNamespaceAliasDefinition(
    const io::NamespaceAliasDefinition* node) -> NamespaceAliasDefinitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NamespaceAliasDefinitionAST();
  ast->namespaceLoc = SourceLocation(node->namespace_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->unqualifiedId = decodeNameId(node->unqualified_id());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeUsingDeclaration(const io::UsingDeclaration* node)
    -> UsingDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UsingDeclarationAST();
  ast->usingLoc = SourceLocation(node->using_loc());
  if (node->using_declarator_list()) {
    auto* inserter = &ast->usingDeclaratorList;
    for (std::uint32_t i = 0; i < node->using_declarator_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeUsingDeclarator(node->using_declarator_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeUsingEnumDeclaration(
    const io::UsingEnumDeclaration* node) -> UsingEnumDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UsingEnumDeclarationAST();
  ast->usingLoc = SourceLocation(node->using_loc());
  ast->enumTypeSpecifier =
      decodeElaboratedTypeSpecifier(node->enum_type_specifier());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeUsingDirective(const io::UsingDirective* node)
    -> UsingDirectiveAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UsingDirectiveAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->usingLoc = SourceLocation(node->using_loc());
  ast->namespaceLoc = SourceLocation(node->namespace_loc());
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->unqualifiedId = decodeNameId(node->unqualified_id());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeStaticAssertDeclaration(
    const io::StaticAssertDeclaration* node) -> StaticAssertDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) StaticAssertDeclarationAST();
  ast->staticAssertLoc = SourceLocation(node->static_assert_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->commaLoc = SourceLocation(node->comma_loc());
  ast->literalLoc = SourceLocation(node->literal_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeAliasDeclaration(const io::AliasDeclaration* node)
    -> AliasDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AliasDeclarationAST();
  ast->usingLoc = SourceLocation(node->using_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->equalLoc = SourceLocation(node->equal_loc());
  if (node->gnu_attribute_list()) {
    auto* inserter = &ast->gnuAttributeList;
    for (std::uint32_t i = 0; i < node->gnu_attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->gnu_attribute_list()->Get(i),
          io::AttributeSpecifier(node->gnu_attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->typeId = decodeTypeId(node->type_id());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeOpaqueEnumDeclaration(
    const io::OpaqueEnumDeclaration* node) -> OpaqueEnumDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) OpaqueEnumDeclarationAST();
  ast->enumLoc = SourceLocation(node->enum_loc());
  ast->classLoc = SourceLocation(node->class_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->unqualifiedId = decodeNameId(node->unqualified_id());
  ast->colonLoc = SourceLocation(node->colon_loc());
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::uint32_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->emicolonLoc = SourceLocation(node->emicolon_loc());
  return ast;
}

auto ASTDecoder::decodeFunctionDefinition(const io::FunctionDefinition* node)
    -> FunctionDefinitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FunctionDefinitionAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->decl_specifier_list()) {
    auto* inserter = &ast->declSpecifierList;
    for (std::uint32_t i = 0; i < node->decl_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->decl_specifier_list()->Get(i),
          io::Specifier(node->decl_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->declarator = decodeDeclarator(node->declarator());
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->functionBody =
      decodeFunctionBody(node->function_body(), node->function_body_type());
  return ast;
}

auto ASTDecoder::decodeTemplateDeclaration(const io::TemplateDeclaration* node)
    -> TemplateDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TemplateDeclarationAST();
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->lessLoc = SourceLocation(node->less_loc());
  if (node->template_parameter_list()) {
    auto* inserter = &ast->templateParameterList;
    for (std::uint32_t i = 0; i < node->template_parameter_list()->size();
         ++i) {
      *inserter = new (pool_) List(decodeTemplateParameter(
          node->template_parameter_list()->Get(i),
          io::TemplateParameter(node->template_parameter_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->greaterLoc = SourceLocation(node->greater_loc());
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->declaration =
      decodeDeclaration(node->declaration(), node->declaration_type());
  return ast;
}

auto ASTDecoder::decodeConceptDefinition(const io::ConceptDefinition* node)
    -> ConceptDefinitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConceptDefinitionAST();
  ast->conceptLoc = SourceLocation(node->concept_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeDeductionGuide(const io::DeductionGuide* node)
    -> DeductionGuideAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DeductionGuideAST();
  ast->explicitSpecifier = decodeSpecifier(node->explicit_specifier(),
                                           node->explicit_specifier_type());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->parameterDeclarationClause =
      decodeParameterDeclarationClause(node->parameter_declaration_clause());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->arrowLoc = SourceLocation(node->arrow_loc());
  ast->templateId = decodeSimpleTemplateId(node->template_id());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeExplicitInstantiation(
    const io::ExplicitInstantiation* node) -> ExplicitInstantiationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExplicitInstantiationAST();
  ast->externLoc = SourceLocation(node->extern_loc());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->declaration =
      decodeDeclaration(node->declaration(), node->declaration_type());
  return ast;
}

auto ASTDecoder::decodeExportDeclaration(const io::ExportDeclaration* node)
    -> ExportDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExportDeclarationAST();
  ast->exportLoc = SourceLocation(node->export_loc());
  ast->declaration =
      decodeDeclaration(node->declaration(), node->declaration_type());
  return ast;
}

auto ASTDecoder::decodeExportCompoundDeclaration(
    const io::ExportCompoundDeclaration* node)
    -> ExportCompoundDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExportCompoundDeclarationAST();
  ast->exportLoc = SourceLocation(node->export_loc());
  ast->lbraceLoc = SourceLocation(node->lbrace_loc());
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::uint32_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rbraceLoc = SourceLocation(node->rbrace_loc());
  return ast;
}

auto ASTDecoder::decodeLinkageSpecification(
    const io::LinkageSpecification* node) -> LinkageSpecificationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LinkageSpecificationAST();
  ast->externLoc = SourceLocation(node->extern_loc());
  ast->stringliteralLoc = SourceLocation(node->stringliteral_loc());
  ast->lbraceLoc = SourceLocation(node->lbrace_loc());
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::uint32_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rbraceLoc = SourceLocation(node->rbrace_loc());
  if (node->string_literal()) {
    ast->stringLiteral =
        unit_->control()->stringLiteral(node->string_literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeNamespaceDefinition(const io::NamespaceDefinition* node)
    -> NamespaceDefinitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NamespaceDefinitionAST();
  ast->inlineLoc = SourceLocation(node->inline_loc());
  ast->namespaceLoc = SourceLocation(node->namespace_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->nested_namespace_specifier_list()) {
    auto* inserter = &ast->nestedNamespaceSpecifierList;
    for (std::uint32_t i = 0;
         i < node->nested_namespace_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeNestedNamespaceSpecifier(
          node->nested_namespace_specifier_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->extra_attribute_list()) {
    auto* inserter = &ast->extraAttributeList;
    for (std::uint32_t i = 0; i < node->extra_attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->extra_attribute_list()->Get(i),
          io::AttributeSpecifier(node->extra_attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->lbraceLoc = SourceLocation(node->lbrace_loc());
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::uint32_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rbraceLoc = SourceLocation(node->rbrace_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeEmptyDeclaration(const io::EmptyDeclaration* node)
    -> EmptyDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EmptyDeclarationAST();
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeAttributeDeclaration(
    const io::AttributeDeclaration* node) -> AttributeDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AttributeDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeModuleImportDeclaration(
    const io::ModuleImportDeclaration* node) -> ModuleImportDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ModuleImportDeclarationAST();
  ast->importLoc = SourceLocation(node->import_loc());
  ast->importName = decodeImportName(node->import_name());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeParameterDeclaration(
    const io::ParameterDeclaration* node) -> ParameterDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ParameterDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->thisLoc = SourceLocation(node->this_loc());
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::uint32_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->declarator = decodeDeclarator(node->declarator());
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeAccessDeclaration(const io::AccessDeclaration* node)
    -> AccessDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AccessDeclarationAST();
  ast->accessLoc = SourceLocation(node->access_loc());
  ast->colonLoc = SourceLocation(node->colon_loc());
  ast->accessSpecifier = static_cast<TokenKind>(node->access_specifier());
  return ast;
}

auto ASTDecoder::decodeForRangeDeclaration(const io::ForRangeDeclaration* node)
    -> ForRangeDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ForRangeDeclarationAST();
  return ast;
}

auto ASTDecoder::decodeStructuredBindingDeclaration(
    const io::StructuredBindingDeclaration* node)
    -> StructuredBindingDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) StructuredBindingDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->decl_specifier_list()) {
    auto* inserter = &ast->declSpecifierList;
    for (std::uint32_t i = 0; i < node->decl_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->decl_specifier_list()->Get(i),
          io::Specifier(node->decl_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->refQualifierLoc = SourceLocation(node->ref_qualifier_loc());
  ast->lbracketLoc = SourceLocation(node->lbracket_loc());
  if (node->binding_list()) {
    auto* inserter = &ast->bindingList;
    for (std::uint32_t i = 0; i < node->binding_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeNameId(node->binding_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->rbracketLoc = SourceLocation(node->rbracket_loc());
  ast->initializer =
      decodeExpression(node->initializer(), node->initializer_type());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeAsmOperand(const io::AsmOperand* node)
    -> AsmOperandAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AsmOperandAST();
  ast->lbracketLoc = SourceLocation(node->lbracket_loc());
  ast->symbolicNameLoc = SourceLocation(node->symbolic_name_loc());
  ast->rbracketLoc = SourceLocation(node->rbracket_loc());
  ast->constraintLiteralLoc = SourceLocation(node->constraint_literal_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  if (node->symbolic_name()) {
    ast->symbolicName =
        unit_->control()->getIdentifier(node->symbolic_name()->str());
  }
  return ast;
}

auto ASTDecoder::decodeAsmQualifier(const io::AsmQualifier* node)
    -> AsmQualifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AsmQualifierAST();
  ast->qualifierLoc = SourceLocation(node->qualifier_loc());
  ast->qualifier = static_cast<TokenKind>(node->qualifier());
  return ast;
}

auto ASTDecoder::decodeAsmClobber(const io::AsmClobber* node)
    -> AsmClobberAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AsmClobberAST();
  ast->literalLoc = SourceLocation(node->literal_loc());
  if (node->literal()) {
    ast->literal = unit_->control()->stringLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeAsmGotoLabel(const io::AsmGotoLabel* node)
    -> AsmGotoLabelAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AsmGotoLabelAST();
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeLabeledStatement(const io::LabeledStatement* node)
    -> LabeledStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LabeledStatementAST();
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->colonLoc = SourceLocation(node->colon_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeCaseStatement(const io::CaseStatement* node)
    -> CaseStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CaseStatementAST();
  ast->caseLoc = SourceLocation(node->case_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->colonLoc = SourceLocation(node->colon_loc());
  return ast;
}

auto ASTDecoder::decodeDefaultStatement(const io::DefaultStatement* node)
    -> DefaultStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DefaultStatementAST();
  ast->defaultLoc = SourceLocation(node->default_loc());
  ast->colonLoc = SourceLocation(node->colon_loc());
  return ast;
}

auto ASTDecoder::decodeExpressionStatement(const io::ExpressionStatement* node)
    -> ExpressionStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExpressionStatementAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeCompoundStatement(const io::CompoundStatement* node)
    -> CompoundStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CompoundStatementAST();
  ast->lbraceLoc = SourceLocation(node->lbrace_loc());
  if (node->statement_list()) {
    auto* inserter = &ast->statementList;
    for (std::uint32_t i = 0; i < node->statement_list()->size(); ++i) {
      *inserter = new (pool_) List(
          decodeStatement(node->statement_list()->Get(i),
                          io::Statement(node->statement_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rbraceLoc = SourceLocation(node->rbrace_loc());
  return ast;
}

auto ASTDecoder::decodeIfStatement(const io::IfStatement* node)
    -> IfStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IfStatementAST();
  ast->ifLoc = SourceLocation(node->if_loc());
  ast->constexprLoc = SourceLocation(node->constexpr_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->initializer =
      decodeStatement(node->initializer(), node->initializer_type());
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  ast->elseLoc = SourceLocation(node->else_loc());
  ast->elseStatement =
      decodeStatement(node->else_statement(), node->else_statement_type());
  return ast;
}

auto ASTDecoder::decodeConstevalIfStatement(
    const io::ConstevalIfStatement* node) -> ConstevalIfStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConstevalIfStatementAST();
  ast->ifLoc = SourceLocation(node->if_loc());
  ast->exclaimLoc = SourceLocation(node->exclaim_loc());
  ast->constvalLoc = SourceLocation(node->constval_loc());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  ast->elseLoc = SourceLocation(node->else_loc());
  ast->elseStatement =
      decodeStatement(node->else_statement(), node->else_statement_type());
  return ast;
}

auto ASTDecoder::decodeSwitchStatement(const io::SwitchStatement* node)
    -> SwitchStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SwitchStatementAST();
  ast->switchLoc = SourceLocation(node->switch_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->initializer =
      decodeStatement(node->initializer(), node->initializer_type());
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeWhileStatement(const io::WhileStatement* node)
    -> WhileStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) WhileStatementAST();
  ast->whileLoc = SourceLocation(node->while_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeDoStatement(const io::DoStatement* node)
    -> DoStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DoStatementAST();
  ast->doLoc = SourceLocation(node->do_loc());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  ast->whileLoc = SourceLocation(node->while_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeForRangeStatement(const io::ForRangeStatement* node)
    -> ForRangeStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ForRangeStatementAST();
  ast->forLoc = SourceLocation(node->for_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->initializer =
      decodeStatement(node->initializer(), node->initializer_type());
  ast->rangeDeclaration = decodeDeclaration(node->range_declaration(),
                                            node->range_declaration_type());
  ast->colonLoc = SourceLocation(node->colon_loc());
  ast->rangeInitializer = decodeExpression(node->range_initializer(),
                                           node->range_initializer_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeForStatement(const io::ForStatement* node)
    -> ForStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ForStatementAST();
  ast->forLoc = SourceLocation(node->for_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->initializer =
      decodeStatement(node->initializer(), node->initializer_type());
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeBreakStatement(const io::BreakStatement* node)
    -> BreakStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BreakStatementAST();
  ast->breakLoc = SourceLocation(node->break_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeContinueStatement(const io::ContinueStatement* node)
    -> ContinueStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ContinueStatementAST();
  ast->continueLoc = SourceLocation(node->continue_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeReturnStatement(const io::ReturnStatement* node)
    -> ReturnStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ReturnStatementAST();
  ast->returnLoc = SourceLocation(node->return_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeCoroutineReturnStatement(
    const io::CoroutineReturnStatement* node) -> CoroutineReturnStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CoroutineReturnStatementAST();
  ast->coreturnLoc = SourceLocation(node->coreturn_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeGotoStatement(const io::GotoStatement* node)
    -> GotoStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GotoStatementAST();
  ast->gotoLoc = SourceLocation(node->goto_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeDeclarationStatement(
    const io::DeclarationStatement* node) -> DeclarationStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DeclarationStatementAST();
  ast->declaration =
      decodeDeclaration(node->declaration(), node->declaration_type());
  return ast;
}

auto ASTDecoder::decodeTryBlockStatement(const io::TryBlockStatement* node)
    -> TryBlockStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TryBlockStatementAST();
  ast->tryLoc = SourceLocation(node->try_loc());
  ast->statement = decodeCompoundStatement(node->statement());
  if (node->handler_list()) {
    auto* inserter = &ast->handlerList;
    for (std::uint32_t i = 0; i < node->handler_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeHandler(node->handler_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeGeneratedLiteralExpression(
    const io::GeneratedLiteralExpression* node)
    -> GeneratedLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GeneratedLiteralExpressionAST();
  ast->literalLoc = SourceLocation(node->literal_loc());
  return ast;
}

auto ASTDecoder::decodeCharLiteralExpression(
    const io::CharLiteralExpression* node) -> CharLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CharLiteralExpressionAST();
  ast->literalLoc = SourceLocation(node->literal_loc());
  if (node->literal()) {
    ast->literal = unit_->control()->charLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeBoolLiteralExpression(
    const io::BoolLiteralExpression* node) -> BoolLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BoolLiteralExpressionAST();
  ast->literalLoc = SourceLocation(node->literal_loc());
  return ast;
}

auto ASTDecoder::decodeIntLiteralExpression(
    const io::IntLiteralExpression* node) -> IntLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IntLiteralExpressionAST();
  ast->literalLoc = SourceLocation(node->literal_loc());
  if (node->literal()) {
    ast->literal = unit_->control()->integerLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeFloatLiteralExpression(
    const io::FloatLiteralExpression* node) -> FloatLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FloatLiteralExpressionAST();
  ast->literalLoc = SourceLocation(node->literal_loc());
  if (node->literal()) {
    ast->literal = unit_->control()->floatLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeNullptrLiteralExpression(
    const io::NullptrLiteralExpression* node) -> NullptrLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NullptrLiteralExpressionAST();
  ast->literalLoc = SourceLocation(node->literal_loc());
  ast->literal = static_cast<TokenKind>(node->literal());
  return ast;
}

auto ASTDecoder::decodeStringLiteralExpression(
    const io::StringLiteralExpression* node) -> StringLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) StringLiteralExpressionAST();
  ast->literalLoc = SourceLocation(node->literal_loc());
  if (node->literal()) {
    ast->literal = unit_->control()->stringLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeUserDefinedStringLiteralExpression(
    const io::UserDefinedStringLiteralExpression* node)
    -> UserDefinedStringLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UserDefinedStringLiteralExpressionAST();
  ast->literalLoc = SourceLocation(node->literal_loc());
  if (node->literal()) {
    ast->literal = unit_->control()->stringLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeObjectLiteralExpression(
    const io::ObjectLiteralExpression* node) -> ObjectLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ObjectLiteralExpressionAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->bracedInitList = decodeBracedInitList(node->braced_init_list());
  return ast;
}

auto ASTDecoder::decodeThisExpression(const io::ThisExpression* node)
    -> ThisExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThisExpressionAST();
  ast->thisLoc = SourceLocation(node->this_loc());
  return ast;
}

auto ASTDecoder::decodeNestedStatementExpression(
    const io::NestedStatementExpression* node)
    -> NestedStatementExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NestedStatementExpressionAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->statement = decodeCompoundStatement(node->statement());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeNestedExpression(const io::NestedExpression* node)
    -> NestedExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NestedExpressionAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeIdExpression(const io::IdExpression* node)
    -> IdExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IdExpressionAST();
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  return ast;
}

auto ASTDecoder::decodeLambdaExpression(const io::LambdaExpression* node)
    -> LambdaExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LambdaExpressionAST();
  ast->lbracketLoc = SourceLocation(node->lbracket_loc());
  ast->captureDefaultLoc = SourceLocation(node->capture_default_loc());
  if (node->capture_list()) {
    auto* inserter = &ast->captureList;
    for (std::uint32_t i = 0; i < node->capture_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeLambdaCapture(
          node->capture_list()->Get(i),
          io::LambdaCapture(node->capture_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rbracketLoc = SourceLocation(node->rbracket_loc());
  ast->lessLoc = SourceLocation(node->less_loc());
  if (node->template_parameter_list()) {
    auto* inserter = &ast->templateParameterList;
    for (std::uint32_t i = 0; i < node->template_parameter_list()->size();
         ++i) {
      *inserter = new (pool_) List(decodeTemplateParameter(
          node->template_parameter_list()->Get(i),
          io::TemplateParameter(node->template_parameter_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->greaterLoc = SourceLocation(node->greater_loc());
  ast->templateRequiresClause =
      decodeRequiresClause(node->template_requires_clause());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->parameterDeclarationClause =
      decodeParameterDeclarationClause(node->parameter_declaration_clause());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  if (node->gnu_atribute_list()) {
    auto* inserter = &ast->gnuAtributeList;
    for (std::uint32_t i = 0; i < node->gnu_atribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->gnu_atribute_list()->Get(i),
          io::AttributeSpecifier(node->gnu_atribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->lambda_specifier_list()) {
    auto* inserter = &ast->lambdaSpecifierList;
    for (std::uint32_t i = 0; i < node->lambda_specifier_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeLambdaSpecifier(node->lambda_specifier_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->exceptionSpecifier = decodeExceptionSpecifier(
      node->exception_specifier(), node->exception_specifier_type());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->trailingReturnType =
      decodeTrailingReturnType(node->trailing_return_type());
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->statement = decodeCompoundStatement(node->statement());
  ast->captureDefault = static_cast<TokenKind>(node->capture_default());
  return ast;
}

auto ASTDecoder::decodeFoldExpression(const io::FoldExpression* node)
    -> FoldExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FoldExpressionAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->leftExpression =
      decodeExpression(node->left_expression(), node->left_expression_type());
  ast->opLoc = SourceLocation(node->op_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->foldOpLoc = SourceLocation(node->fold_op_loc());
  ast->rightExpression =
      decodeExpression(node->right_expression(), node->right_expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->op = static_cast<TokenKind>(node->op());
  ast->foldOp = static_cast<TokenKind>(node->fold_op());
  return ast;
}

auto ASTDecoder::decodeRightFoldExpression(const io::RightFoldExpression* node)
    -> RightFoldExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RightFoldExpressionAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->opLoc = SourceLocation(node->op_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeLeftFoldExpression(const io::LeftFoldExpression* node)
    -> LeftFoldExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LeftFoldExpressionAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->opLoc = SourceLocation(node->op_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeRequiresExpression(const io::RequiresExpression* node)
    -> RequiresExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RequiresExpressionAST();
  ast->requiresLoc = SourceLocation(node->requires_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->parameterDeclarationClause =
      decodeParameterDeclarationClause(node->parameter_declaration_clause());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->lbraceLoc = SourceLocation(node->lbrace_loc());
  if (node->requirement_list()) {
    auto* inserter = &ast->requirementList;
    for (std::uint32_t i = 0; i < node->requirement_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeRequirement(
          node->requirement_list()->Get(i),
          io::Requirement(node->requirement_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rbraceLoc = SourceLocation(node->rbrace_loc());
  return ast;
}

auto ASTDecoder::decodeVaArgExpression(const io::VaArgExpression* node)
    -> VaArgExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) VaArgExpressionAST();
  ast->vaArgLoc = SourceLocation(node->va_arg_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->commaLoc = SourceLocation(node->comma_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeSubscriptExpression(const io::SubscriptExpression* node)
    -> SubscriptExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SubscriptExpressionAST();
  ast->baseExpression =
      decodeExpression(node->base_expression(), node->base_expression_type());
  ast->lbracketLoc = SourceLocation(node->lbracket_loc());
  ast->indexExpression =
      decodeExpression(node->index_expression(), node->index_expression_type());
  ast->rbracketLoc = SourceLocation(node->rbracket_loc());
  return ast;
}

auto ASTDecoder::decodeCallExpression(const io::CallExpression* node)
    -> CallExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CallExpressionAST();
  ast->baseExpression =
      decodeExpression(node->base_expression(), node->base_expression_type());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::uint32_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeTypeConstruction(const io::TypeConstruction* node)
    -> TypeConstructionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeConstructionAST();
  ast->typeSpecifier =
      decodeSpecifier(node->type_specifier(), node->type_specifier_type());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::uint32_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeBracedTypeConstruction(
    const io::BracedTypeConstruction* node) -> BracedTypeConstructionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BracedTypeConstructionAST();
  ast->typeSpecifier =
      decodeSpecifier(node->type_specifier(), node->type_specifier_type());
  ast->bracedInitList = decodeBracedInitList(node->braced_init_list());
  return ast;
}

auto ASTDecoder::decodeSpliceMemberExpression(
    const io::SpliceMemberExpression* node) -> SpliceMemberExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SpliceMemberExpressionAST();
  ast->baseExpression =
      decodeExpression(node->base_expression(), node->base_expression_type());
  ast->accessLoc = SourceLocation(node->access_loc());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->splicer = decodeSplicer(node->splicer());
  ast->accessOp = static_cast<TokenKind>(node->access_op());
  return ast;
}

auto ASTDecoder::decodeMemberExpression(const io::MemberExpression* node)
    -> MemberExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) MemberExpressionAST();
  ast->baseExpression =
      decodeExpression(node->base_expression(), node->base_expression_type());
  ast->accessLoc = SourceLocation(node->access_loc());
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  ast->accessOp = static_cast<TokenKind>(node->access_op());
  return ast;
}

auto ASTDecoder::decodePostIncrExpression(const io::PostIncrExpression* node)
    -> PostIncrExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) PostIncrExpressionAST();
  ast->baseExpression =
      decodeExpression(node->base_expression(), node->base_expression_type());
  ast->opLoc = SourceLocation(node->op_loc());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeCppCastExpression(const io::CppCastExpression* node)
    -> CppCastExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CppCastExpressionAST();
  ast->castLoc = SourceLocation(node->cast_loc());
  ast->lessLoc = SourceLocation(node->less_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->greaterLoc = SourceLocation(node->greater_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeBuiltinBitCastExpression(
    const io::BuiltinBitCastExpression* node) -> BuiltinBitCastExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BuiltinBitCastExpressionAST();
  ast->castLoc = SourceLocation(node->cast_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->commaLoc = SourceLocation(node->comma_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeBuiltinOffsetofExpression(
    const io::BuiltinOffsetofExpression* node)
    -> BuiltinOffsetofExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BuiltinOffsetofExpressionAST();
  ast->offsetofLoc = SourceLocation(node->offsetof_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->commaLoc = SourceLocation(node->comma_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeTypeidExpression(const io::TypeidExpression* node)
    -> TypeidExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeidExpressionAST();
  ast->typeidLoc = SourceLocation(node->typeid_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeTypeidOfTypeExpression(
    const io::TypeidOfTypeExpression* node) -> TypeidOfTypeExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeidOfTypeExpressionAST();
  ast->typeidLoc = SourceLocation(node->typeid_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeSpliceExpression(const io::SpliceExpression* node)
    -> SpliceExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SpliceExpressionAST();
  ast->splicer = decodeSplicer(node->splicer());
  return ast;
}

auto ASTDecoder::decodeGlobalScopeReflectExpression(
    const io::GlobalScopeReflectExpression* node)
    -> GlobalScopeReflectExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GlobalScopeReflectExpressionAST();
  ast->caretLoc = SourceLocation(node->caret_loc());
  ast->scopeLoc = SourceLocation(node->scope_loc());
  return ast;
}

auto ASTDecoder::decodeNamespaceReflectExpression(
    const io::NamespaceReflectExpression* node)
    -> NamespaceReflectExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NamespaceReflectExpressionAST();
  ast->caretLoc = SourceLocation(node->caret_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeTypeIdReflectExpression(
    const io::TypeIdReflectExpression* node) -> TypeIdReflectExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeIdReflectExpressionAST();
  ast->caretLoc = SourceLocation(node->caret_loc());
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeReflectExpression(const io::ReflectExpression* node)
    -> ReflectExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ReflectExpressionAST();
  ast->caretLoc = SourceLocation(node->caret_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeUnaryExpression(const io::UnaryExpression* node)
    -> UnaryExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UnaryExpressionAST();
  ast->opLoc = SourceLocation(node->op_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeAwaitExpression(const io::AwaitExpression* node)
    -> AwaitExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AwaitExpressionAST();
  ast->awaitLoc = SourceLocation(node->await_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeSizeofExpression(const io::SizeofExpression* node)
    -> SizeofExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SizeofExpressionAST();
  ast->sizeofLoc = SourceLocation(node->sizeof_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeSizeofTypeExpression(
    const io::SizeofTypeExpression* node) -> SizeofTypeExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SizeofTypeExpressionAST();
  ast->sizeofLoc = SourceLocation(node->sizeof_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeSizeofPackExpression(
    const io::SizeofPackExpression* node) -> SizeofPackExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SizeofPackExpressionAST();
  ast->sizeofLoc = SourceLocation(node->sizeof_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeAlignofTypeExpression(
    const io::AlignofTypeExpression* node) -> AlignofTypeExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AlignofTypeExpressionAST();
  ast->alignofLoc = SourceLocation(node->alignof_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeAlignofExpression(const io::AlignofExpression* node)
    -> AlignofExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AlignofExpressionAST();
  ast->alignofLoc = SourceLocation(node->alignof_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeNoexceptExpression(const io::NoexceptExpression* node)
    -> NoexceptExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NoexceptExpressionAST();
  ast->noexceptLoc = SourceLocation(node->noexcept_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeNewExpression(const io::NewExpression* node)
    -> NewExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NewExpressionAST();
  ast->scopeLoc = SourceLocation(node->scope_loc());
  ast->newLoc = SourceLocation(node->new_loc());
  ast->newPlacement = decodeNewPlacement(node->new_placement());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::uint32_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->declarator = decodeDeclarator(node->declarator());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->newInitalizer =
      decodeNewInitializer(node->new_initalizer(), node->new_initalizer_type());
  return ast;
}

auto ASTDecoder::decodeDeleteExpression(const io::DeleteExpression* node)
    -> DeleteExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DeleteExpressionAST();
  ast->scopeLoc = SourceLocation(node->scope_loc());
  ast->deleteLoc = SourceLocation(node->delete_loc());
  ast->lbracketLoc = SourceLocation(node->lbracket_loc());
  ast->rbracketLoc = SourceLocation(node->rbracket_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeCastExpression(const io::CastExpression* node)
    -> CastExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CastExpressionAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeImplicitCastExpression(
    const io::ImplicitCastExpression* node) -> ImplicitCastExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ImplicitCastExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeBinaryExpression(const io::BinaryExpression* node)
    -> BinaryExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BinaryExpressionAST();
  ast->leftExpression =
      decodeExpression(node->left_expression(), node->left_expression_type());
  ast->opLoc = SourceLocation(node->op_loc());
  ast->rightExpression =
      decodeExpression(node->right_expression(), node->right_expression_type());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeConditionalExpression(
    const io::ConditionalExpression* node) -> ConditionalExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConditionalExpressionAST();
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->questionLoc = SourceLocation(node->question_loc());
  ast->iftrueExpression = decodeExpression(node->iftrue_expression(),
                                           node->iftrue_expression_type());
  ast->colonLoc = SourceLocation(node->colon_loc());
  ast->iffalseExpression = decodeExpression(node->iffalse_expression(),
                                            node->iffalse_expression_type());
  return ast;
}

auto ASTDecoder::decodeYieldExpression(const io::YieldExpression* node)
    -> YieldExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) YieldExpressionAST();
  ast->yieldLoc = SourceLocation(node->yield_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeThrowExpression(const io::ThrowExpression* node)
    -> ThrowExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThrowExpressionAST();
  ast->throwLoc = SourceLocation(node->throw_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeAssignmentExpression(
    const io::AssignmentExpression* node) -> AssignmentExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AssignmentExpressionAST();
  ast->leftExpression =
      decodeExpression(node->left_expression(), node->left_expression_type());
  ast->opLoc = SourceLocation(node->op_loc());
  ast->rightExpression =
      decodeExpression(node->right_expression(), node->right_expression_type());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodePackExpansionExpression(
    const io::PackExpansionExpression* node) -> PackExpansionExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) PackExpansionExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  return ast;
}

auto ASTDecoder::decodeDesignatedInitializerClause(
    const io::DesignatedInitializerClause* node)
    -> DesignatedInitializerClauseAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DesignatedInitializerClauseAST();
  if (node->designator_list()) {
    auto* inserter = &ast->designatorList;
    for (std::uint32_t i = 0; i < node->designator_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDesignator(
          node->designator_list()->Get(i),
          io::Designator(node->designator_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->initializer =
      decodeExpression(node->initializer(), node->initializer_type());
  return ast;
}

auto ASTDecoder::decodeTypeTraitExpression(const io::TypeTraitExpression* node)
    -> TypeTraitExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeTraitExpressionAST();
  ast->typeTraitLoc = SourceLocation(node->type_trait_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  if (node->type_id_list()) {
    auto* inserter = &ast->typeIdList;
    for (std::uint32_t i = 0; i < node->type_id_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeTypeId(node->type_id_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeConditionExpression(const io::ConditionExpression* node)
    -> ConditionExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConditionExpressionAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->decl_specifier_list()) {
    auto* inserter = &ast->declSpecifierList;
    for (std::uint32_t i = 0; i < node->decl_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->decl_specifier_list()->Get(i),
          io::Specifier(node->decl_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->declarator = decodeDeclarator(node->declarator());
  ast->initializer =
      decodeExpression(node->initializer(), node->initializer_type());
  return ast;
}

auto ASTDecoder::decodeEqualInitializer(const io::EqualInitializer* node)
    -> EqualInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EqualInitializerAST();
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeBracedInitList(const io::BracedInitList* node)
    -> BracedInitListAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BracedInitListAST();
  ast->lbraceLoc = SourceLocation(node->lbrace_loc());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::uint32_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->commaLoc = SourceLocation(node->comma_loc());
  ast->rbraceLoc = SourceLocation(node->rbrace_loc());
  return ast;
}

auto ASTDecoder::decodeParenInitializer(const io::ParenInitializer* node)
    -> ParenInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ParenInitializerAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::uint32_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeDotDesignator(const io::DotDesignator* node)
    -> DotDesignatorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DotDesignatorAST();
  ast->dotLoc = SourceLocation(node->dot_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeSubscriptDesignator(const io::SubscriptDesignator* node)
    -> SubscriptDesignatorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SubscriptDesignatorAST();
  ast->lbracketLoc = SourceLocation(node->lbracket_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rbracketLoc = SourceLocation(node->rbracket_loc());
  return ast;
}

auto ASTDecoder::decodeSplicer(const io::Splicer* node) -> SplicerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SplicerAST();
  ast->lbracketLoc = SourceLocation(node->lbracket_loc());
  ast->colonLoc = SourceLocation(node->colon_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->secondColonLoc = SourceLocation(node->second_colon_loc());
  ast->rbracketLoc = SourceLocation(node->rbracket_loc());
  return ast;
}

auto ASTDecoder::decodeGlobalModuleFragment(
    const io::GlobalModuleFragment* node) -> GlobalModuleFragmentAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GlobalModuleFragmentAST();
  ast->moduleLoc = SourceLocation(node->module_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::uint32_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodePrivateModuleFragment(
    const io::PrivateModuleFragment* node) -> PrivateModuleFragmentAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) PrivateModuleFragmentAST();
  ast->moduleLoc = SourceLocation(node->module_loc());
  ast->colonLoc = SourceLocation(node->colon_loc());
  ast->privateLoc = SourceLocation(node->private_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::uint32_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeModuleDeclaration(const io::ModuleDeclaration* node)
    -> ModuleDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ModuleDeclarationAST();
  ast->exportLoc = SourceLocation(node->export_loc());
  ast->moduleLoc = SourceLocation(node->module_loc());
  ast->moduleName = decodeModuleName(node->module_name());
  ast->modulePartition = decodeModulePartition(node->module_partition());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeModuleName(const io::ModuleName* node)
    -> ModuleNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ModuleNameAST();
  ast->moduleQualifier = decodeModuleQualifier(node->module_qualifier());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeModuleQualifier(const io::ModuleQualifier* node)
    -> ModuleQualifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ModuleQualifierAST();
  ast->moduleQualifier = decodeModuleQualifier(node->module_qualifier());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->dotLoc = SourceLocation(node->dot_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeModulePartition(const io::ModulePartition* node)
    -> ModulePartitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ModulePartitionAST();
  ast->colonLoc = SourceLocation(node->colon_loc());
  ast->moduleName = decodeModuleName(node->module_name());
  return ast;
}

auto ASTDecoder::decodeImportName(const io::ImportName* node)
    -> ImportNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ImportNameAST();
  ast->headerLoc = SourceLocation(node->header_loc());
  ast->modulePartition = decodeModulePartition(node->module_partition());
  ast->moduleName = decodeModuleName(node->module_name());
  return ast;
}

auto ASTDecoder::decodeInitDeclarator(const io::InitDeclarator* node)
    -> InitDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) InitDeclaratorAST();
  ast->declarator = decodeDeclarator(node->declarator());
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->initializer =
      decodeExpression(node->initializer(), node->initializer_type());
  return ast;
}

auto ASTDecoder::decodeDeclarator(const io::Declarator* node)
    -> DeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DeclaratorAST();
  if (node->ptr_op_list()) {
    auto* inserter = &ast->ptrOpList;
    for (std::uint32_t i = 0; i < node->ptr_op_list()->size(); ++i) {
      *inserter = new (pool_) List(
          decodePtrOperator(node->ptr_op_list()->Get(i),
                            io::PtrOperator(node->ptr_op_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->coreDeclarator = decodeCoreDeclarator(node->core_declarator(),
                                             node->core_declarator_type());
  if (node->declarator_chunk_list()) {
    auto* inserter = &ast->declaratorChunkList;
    for (std::uint32_t i = 0; i < node->declarator_chunk_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaratorChunk(
          node->declarator_chunk_list()->Get(i),
          io::DeclaratorChunk(node->declarator_chunk_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeUsingDeclarator(const io::UsingDeclarator* node)
    -> UsingDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UsingDeclaratorAST();
  ast->typenameLoc = SourceLocation(node->typename_loc());
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  return ast;
}

auto ASTDecoder::decodeEnumerator(const io::Enumerator* node)
    -> EnumeratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EnumeratorAST();
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeTypeId(const io::TypeId* node) -> TypeIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeIdAST();
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::uint32_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->declarator = decodeDeclarator(node->declarator());
  return ast;
}

auto ASTDecoder::decodeHandler(const io::Handler* node) -> HandlerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) HandlerAST();
  ast->catchLoc = SourceLocation(node->catch_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->exceptionDeclaration = decodeExceptionDeclaration(
      node->exception_declaration(), node->exception_declaration_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->statement = decodeCompoundStatement(node->statement());
  return ast;
}

auto ASTDecoder::decodeBaseSpecifier(const io::BaseSpecifier* node)
    -> BaseSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BaseSpecifierAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->virtualOrAccessLoc = SourceLocation(node->virtual_or_access_loc());
  ast->otherVirtualOrAccessLoc =
      SourceLocation(node->other_virtual_or_access_loc());
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->accessSpecifier = static_cast<TokenKind>(node->access_specifier());
  return ast;
}

auto ASTDecoder::decodeRequiresClause(const io::RequiresClause* node)
    -> RequiresClauseAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RequiresClauseAST();
  ast->requiresLoc = SourceLocation(node->requires_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeParameterDeclarationClause(
    const io::ParameterDeclarationClause* node)
    -> ParameterDeclarationClauseAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ParameterDeclarationClauseAST();
  if (node->parameter_declaration_list()) {
    auto* inserter = &ast->parameterDeclarationList;
    for (std::uint32_t i = 0; i < node->parameter_declaration_list()->size();
         ++i) {
      *inserter = new (pool_) List(decodeParameterDeclaration(
          node->parameter_declaration_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->commaLoc = SourceLocation(node->comma_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  return ast;
}

auto ASTDecoder::decodeTrailingReturnType(const io::TrailingReturnType* node)
    -> TrailingReturnTypeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TrailingReturnTypeAST();
  ast->minusGreaterLoc = SourceLocation(node->minus_greater_loc());
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeLambdaSpecifier(const io::LambdaSpecifier* node)
    -> LambdaSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LambdaSpecifierAST();
  ast->specifierLoc = SourceLocation(node->specifier_loc());
  ast->specifier = static_cast<TokenKind>(node->specifier());
  return ast;
}

auto ASTDecoder::decodeTypeConstraint(const io::TypeConstraint* node)
    -> TypeConstraintAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeConstraintAST();
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->lessLoc = SourceLocation(node->less_loc());
  if (node->template_argument_list()) {
    auto* inserter = &ast->templateArgumentList;
    for (std::uint32_t i = 0; i < node->template_argument_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeTemplateArgument(
          node->template_argument_list()->Get(i),
          io::TemplateArgument(node->template_argument_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->greaterLoc = SourceLocation(node->greater_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeAttributeArgumentClause(
    const io::AttributeArgumentClause* node) -> AttributeArgumentClauseAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AttributeArgumentClauseAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeAttribute(const io::Attribute* node) -> AttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AttributeAST();
  ast->attributeToken = decodeAttributeToken(node->attribute_token(),
                                             node->attribute_token_type());
  ast->attributeArgumentClause =
      decodeAttributeArgumentClause(node->attribute_argument_clause());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  return ast;
}

auto ASTDecoder::decodeAttributeUsingPrefix(
    const io::AttributeUsingPrefix* node) -> AttributeUsingPrefixAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AttributeUsingPrefixAST();
  ast->usingLoc = SourceLocation(node->using_loc());
  ast->attributeNamespaceLoc = SourceLocation(node->attribute_namespace_loc());
  ast->colonLoc = SourceLocation(node->colon_loc());
  return ast;
}

auto ASTDecoder::decodeNewPlacement(const io::NewPlacement* node)
    -> NewPlacementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NewPlacementAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::uint32_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeNestedNamespaceSpecifier(
    const io::NestedNamespaceSpecifier* node) -> NestedNamespaceSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NestedNamespaceSpecifierAST();
  ast->inlineLoc = SourceLocation(node->inline_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->scopeLoc = SourceLocation(node->scope_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeTemplateTypeParameter(
    const io::TemplateTypeParameter* node) -> TemplateTypeParameterAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TemplateTypeParameterAST();
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->lessLoc = SourceLocation(node->less_loc());
  if (node->template_parameter_list()) {
    auto* inserter = &ast->templateParameterList;
    for (std::uint32_t i = 0; i < node->template_parameter_list()->size();
         ++i) {
      *inserter = new (pool_) List(decodeTemplateParameter(
          node->template_parameter_list()->Get(i),
          io::TemplateParameter(node->template_parameter_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->greaterLoc = SourceLocation(node->greater_loc());
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->classKeyLoc = SourceLocation(node->class_key_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->idExpression = decodeIdExpression(node->id_expression());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeNonTypeTemplateParameter(
    const io::NonTypeTemplateParameter* node) -> NonTypeTemplateParameterAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NonTypeTemplateParameterAST();
  ast->declaration = decodeParameterDeclaration(node->declaration());
  return ast;
}

auto ASTDecoder::decodeTypenameTypeParameter(
    const io::TypenameTypeParameter* node) -> TypenameTypeParameterAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypenameTypeParameterAST();
  ast->classKeyLoc = SourceLocation(node->class_key_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->typeId = decodeTypeId(node->type_id());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeConstraintTypeParameter(
    const io::ConstraintTypeParameter* node) -> ConstraintTypeParameterAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConstraintTypeParameterAST();
  ast->typeConstraint = decodeTypeConstraint(node->type_constraint());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->typeId = decodeTypeId(node->type_id());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeGeneratedTypeSpecifier(
    const io::GeneratedTypeSpecifier* node) -> GeneratedTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GeneratedTypeSpecifierAST();
  ast->typeLoc = SourceLocation(node->type_loc());
  return ast;
}

auto ASTDecoder::decodeTypedefSpecifier(const io::TypedefSpecifier* node)
    -> TypedefSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypedefSpecifierAST();
  ast->typedefLoc = SourceLocation(node->typedef_loc());
  return ast;
}

auto ASTDecoder::decodeFriendSpecifier(const io::FriendSpecifier* node)
    -> FriendSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FriendSpecifierAST();
  ast->friendLoc = SourceLocation(node->friend_loc());
  return ast;
}

auto ASTDecoder::decodeConstevalSpecifier(const io::ConstevalSpecifier* node)
    -> ConstevalSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConstevalSpecifierAST();
  ast->constevalLoc = SourceLocation(node->consteval_loc());
  return ast;
}

auto ASTDecoder::decodeConstinitSpecifier(const io::ConstinitSpecifier* node)
    -> ConstinitSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConstinitSpecifierAST();
  ast->constinitLoc = SourceLocation(node->constinit_loc());
  return ast;
}

auto ASTDecoder::decodeConstexprSpecifier(const io::ConstexprSpecifier* node)
    -> ConstexprSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConstexprSpecifierAST();
  ast->constexprLoc = SourceLocation(node->constexpr_loc());
  return ast;
}

auto ASTDecoder::decodeInlineSpecifier(const io::InlineSpecifier* node)
    -> InlineSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) InlineSpecifierAST();
  ast->inlineLoc = SourceLocation(node->inline_loc());
  return ast;
}

auto ASTDecoder::decodeNoreturnSpecifier(const io::NoreturnSpecifier* node)
    -> NoreturnSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NoreturnSpecifierAST();
  ast->noreturnLoc = SourceLocation(node->noreturn_loc());
  return ast;
}

auto ASTDecoder::decodeStaticSpecifier(const io::StaticSpecifier* node)
    -> StaticSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) StaticSpecifierAST();
  ast->staticLoc = SourceLocation(node->static_loc());
  return ast;
}

auto ASTDecoder::decodeExternSpecifier(const io::ExternSpecifier* node)
    -> ExternSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExternSpecifierAST();
  ast->externLoc = SourceLocation(node->extern_loc());
  return ast;
}

auto ASTDecoder::decodeRegisterSpecifier(const io::RegisterSpecifier* node)
    -> RegisterSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RegisterSpecifierAST();
  ast->registerLoc = SourceLocation(node->register_loc());
  return ast;
}

auto ASTDecoder::decodeThreadLocalSpecifier(
    const io::ThreadLocalSpecifier* node) -> ThreadLocalSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThreadLocalSpecifierAST();
  ast->threadLocalLoc = SourceLocation(node->thread_local_loc());
  return ast;
}

auto ASTDecoder::decodeThreadSpecifier(const io::ThreadSpecifier* node)
    -> ThreadSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThreadSpecifierAST();
  ast->threadLoc = SourceLocation(node->thread_loc());
  return ast;
}

auto ASTDecoder::decodeMutableSpecifier(const io::MutableSpecifier* node)
    -> MutableSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) MutableSpecifierAST();
  ast->mutableLoc = SourceLocation(node->mutable_loc());
  return ast;
}

auto ASTDecoder::decodeVirtualSpecifier(const io::VirtualSpecifier* node)
    -> VirtualSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) VirtualSpecifierAST();
  ast->virtualLoc = SourceLocation(node->virtual_loc());
  return ast;
}

auto ASTDecoder::decodeExplicitSpecifier(const io::ExplicitSpecifier* node)
    -> ExplicitSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExplicitSpecifierAST();
  ast->explicitLoc = SourceLocation(node->explicit_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeAutoTypeSpecifier(const io::AutoTypeSpecifier* node)
    -> AutoTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AutoTypeSpecifierAST();
  ast->autoLoc = SourceLocation(node->auto_loc());
  return ast;
}

auto ASTDecoder::decodeVoidTypeSpecifier(const io::VoidTypeSpecifier* node)
    -> VoidTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) VoidTypeSpecifierAST();
  ast->voidLoc = SourceLocation(node->void_loc());
  return ast;
}

auto ASTDecoder::decodeSizeTypeSpecifier(const io::SizeTypeSpecifier* node)
    -> SizeTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SizeTypeSpecifierAST();
  ast->specifierLoc = SourceLocation(node->specifier_loc());
  ast->specifier = static_cast<TokenKind>(node->specifier());
  return ast;
}

auto ASTDecoder::decodeSignTypeSpecifier(const io::SignTypeSpecifier* node)
    -> SignTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SignTypeSpecifierAST();
  ast->specifierLoc = SourceLocation(node->specifier_loc());
  ast->specifier = static_cast<TokenKind>(node->specifier());
  return ast;
}

auto ASTDecoder::decodeVaListTypeSpecifier(const io::VaListTypeSpecifier* node)
    -> VaListTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) VaListTypeSpecifierAST();
  ast->specifierLoc = SourceLocation(node->specifier_loc());
  ast->specifier = static_cast<TokenKind>(node->specifier());
  return ast;
}

auto ASTDecoder::decodeIntegralTypeSpecifier(
    const io::IntegralTypeSpecifier* node) -> IntegralTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IntegralTypeSpecifierAST();
  ast->specifierLoc = SourceLocation(node->specifier_loc());
  ast->specifier = static_cast<TokenKind>(node->specifier());
  return ast;
}

auto ASTDecoder::decodeFloatingPointTypeSpecifier(
    const io::FloatingPointTypeSpecifier* node)
    -> FloatingPointTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FloatingPointTypeSpecifierAST();
  ast->specifierLoc = SourceLocation(node->specifier_loc());
  ast->specifier = static_cast<TokenKind>(node->specifier());
  return ast;
}

auto ASTDecoder::decodeComplexTypeSpecifier(
    const io::ComplexTypeSpecifier* node) -> ComplexTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ComplexTypeSpecifierAST();
  ast->complexLoc = SourceLocation(node->complex_loc());
  return ast;
}

auto ASTDecoder::decodeNamedTypeSpecifier(const io::NamedTypeSpecifier* node)
    -> NamedTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NamedTypeSpecifierAST();
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  return ast;
}

auto ASTDecoder::decodeAtomicTypeSpecifier(const io::AtomicTypeSpecifier* node)
    -> AtomicTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AtomicTypeSpecifierAST();
  ast->atomicLoc = SourceLocation(node->atomic_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeUnderlyingTypeSpecifier(
    const io::UnderlyingTypeSpecifier* node) -> UnderlyingTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UnderlyingTypeSpecifierAST();
  ast->underlyingTypeLoc = SourceLocation(node->underlying_type_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeElaboratedTypeSpecifier(
    const io::ElaboratedTypeSpecifier* node) -> ElaboratedTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ElaboratedTypeSpecifierAST();
  ast->classLoc = SourceLocation(node->class_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  ast->classKey = static_cast<TokenKind>(node->class_key());
  return ast;
}

auto ASTDecoder::decodeDecltypeAutoSpecifier(
    const io::DecltypeAutoSpecifier* node) -> DecltypeAutoSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DecltypeAutoSpecifierAST();
  ast->decltypeLoc = SourceLocation(node->decltype_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->autoLoc = SourceLocation(node->auto_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeDecltypeSpecifier(const io::DecltypeSpecifier* node)
    -> DecltypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DecltypeSpecifierAST();
  ast->decltypeLoc = SourceLocation(node->decltype_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodePlaceholderTypeSpecifier(
    const io::PlaceholderTypeSpecifier* node) -> PlaceholderTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) PlaceholderTypeSpecifierAST();
  ast->typeConstraint = decodeTypeConstraint(node->type_constraint());
  ast->specifier = decodeSpecifier(node->specifier(), node->specifier_type());
  return ast;
}

auto ASTDecoder::decodeConstQualifier(const io::ConstQualifier* node)
    -> ConstQualifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConstQualifierAST();
  ast->constLoc = SourceLocation(node->const_loc());
  return ast;
}

auto ASTDecoder::decodeVolatileQualifier(const io::VolatileQualifier* node)
    -> VolatileQualifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) VolatileQualifierAST();
  ast->volatileLoc = SourceLocation(node->volatile_loc());
  return ast;
}

auto ASTDecoder::decodeRestrictQualifier(const io::RestrictQualifier* node)
    -> RestrictQualifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RestrictQualifierAST();
  ast->restrictLoc = SourceLocation(node->restrict_loc());
  return ast;
}

auto ASTDecoder::decodeEnumSpecifier(const io::EnumSpecifier* node)
    -> EnumSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EnumSpecifierAST();
  ast->enumLoc = SourceLocation(node->enum_loc());
  ast->classLoc = SourceLocation(node->class_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->unqualifiedId = decodeNameId(node->unqualified_id());
  ast->colonLoc = SourceLocation(node->colon_loc());
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::uint32_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->lbraceLoc = SourceLocation(node->lbrace_loc());
  if (node->enumerator_list()) {
    auto* inserter = &ast->enumeratorList;
    for (std::uint32_t i = 0; i < node->enumerator_list()->size(); ++i) {
      *inserter =
          new (pool_) List(decodeEnumerator(node->enumerator_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->commaLoc = SourceLocation(node->comma_loc());
  ast->rbraceLoc = SourceLocation(node->rbrace_loc());
  return ast;
}

auto ASTDecoder::decodeClassSpecifier(const io::ClassSpecifier* node)
    -> ClassSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ClassSpecifierAST();
  ast->classLoc = SourceLocation(node->class_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  ast->finalLoc = SourceLocation(node->final_loc());
  ast->colonLoc = SourceLocation(node->colon_loc());
  if (node->base_specifier_list()) {
    auto* inserter = &ast->baseSpecifierList;
    for (std::uint32_t i = 0; i < node->base_specifier_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeBaseSpecifier(node->base_specifier_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->lbraceLoc = SourceLocation(node->lbrace_loc());
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::uint32_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rbraceLoc = SourceLocation(node->rbrace_loc());
  ast->classKey = static_cast<TokenKind>(node->class_key());
  return ast;
}

auto ASTDecoder::decodeTypenameSpecifier(const io::TypenameSpecifier* node)
    -> TypenameSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypenameSpecifierAST();
  ast->typenameLoc = SourceLocation(node->typename_loc());
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  return ast;
}

auto ASTDecoder::decodeSplicerTypeSpecifier(
    const io::SplicerTypeSpecifier* node) -> SplicerTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SplicerTypeSpecifierAST();
  ast->typenameLoc = SourceLocation(node->typename_loc());
  ast->splicer = decodeSplicer(node->splicer());
  return ast;
}

auto ASTDecoder::decodePointerOperator(const io::PointerOperator* node)
    -> PointerOperatorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) PointerOperatorAST();
  ast->starLoc = SourceLocation(node->star_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->cv_qualifier_list()) {
    auto* inserter = &ast->cvQualifierList;
    for (std::uint32_t i = 0; i < node->cv_qualifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->cv_qualifier_list()->Get(i),
          io::Specifier(node->cv_qualifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeReferenceOperator(const io::ReferenceOperator* node)
    -> ReferenceOperatorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ReferenceOperatorAST();
  ast->refLoc = SourceLocation(node->ref_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->refOp = static_cast<TokenKind>(node->ref_op());
  return ast;
}

auto ASTDecoder::decodePtrToMemberOperator(const io::PtrToMemberOperator* node)
    -> PtrToMemberOperatorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) PtrToMemberOperatorAST();
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->starLoc = SourceLocation(node->star_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->cv_qualifier_list()) {
    auto* inserter = &ast->cvQualifierList;
    for (std::uint32_t i = 0; i < node->cv_qualifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->cv_qualifier_list()->Get(i),
          io::Specifier(node->cv_qualifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeBitfieldDeclarator(const io::BitfieldDeclarator* node)
    -> BitfieldDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BitfieldDeclaratorAST();
  ast->unqualifiedId = decodeNameId(node->unqualified_id());
  ast->colonLoc = SourceLocation(node->colon_loc());
  ast->sizeExpression =
      decodeExpression(node->size_expression(), node->size_expression_type());
  return ast;
}

auto ASTDecoder::decodeParameterPack(const io::ParameterPack* node)
    -> ParameterPackAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ParameterPackAST();
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->coreDeclarator = decodeCoreDeclarator(node->core_declarator(),
                                             node->core_declarator_type());
  return ast;
}

auto ASTDecoder::decodeIdDeclarator(const io::IdDeclarator* node)
    -> IdDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IdDeclaratorAST();
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeNestedDeclarator(const io::NestedDeclarator* node)
    -> NestedDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NestedDeclaratorAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->declarator = decodeDeclarator(node->declarator());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeFunctionDeclaratorChunk(
    const io::FunctionDeclaratorChunk* node) -> FunctionDeclaratorChunkAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FunctionDeclaratorChunkAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->parameterDeclarationClause =
      decodeParameterDeclarationClause(node->parameter_declaration_clause());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  if (node->cv_qualifier_list()) {
    auto* inserter = &ast->cvQualifierList;
    for (std::uint32_t i = 0; i < node->cv_qualifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->cv_qualifier_list()->Get(i),
          io::Specifier(node->cv_qualifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->refLoc = SourceLocation(node->ref_loc());
  ast->exceptionSpecifier = decodeExceptionSpecifier(
      node->exception_specifier(), node->exception_specifier_type());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->trailingReturnType =
      decodeTrailingReturnType(node->trailing_return_type());
  return ast;
}

auto ASTDecoder::decodeArrayDeclaratorChunk(
    const io::ArrayDeclaratorChunk* node) -> ArrayDeclaratorChunkAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ArrayDeclaratorChunkAST();
  ast->lbracketLoc = SourceLocation(node->lbracket_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rbracketLoc = SourceLocation(node->rbracket_loc());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeNameId(const io::NameId* node) -> NameIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NameIdAST();
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeDestructorId(const io::DestructorId* node)
    -> DestructorIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DestructorIdAST();
  ast->tildeLoc = SourceLocation(node->tilde_loc());
  ast->id = decodeUnqualifiedId(node->id(), node->id_type());
  return ast;
}

auto ASTDecoder::decodeDecltypeId(const io::DecltypeId* node)
    -> DecltypeIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DecltypeIdAST();
  ast->decltypeSpecifier = decodeDecltypeSpecifier(node->decltype_specifier());
  return ast;
}

auto ASTDecoder::decodeOperatorFunctionId(const io::OperatorFunctionId* node)
    -> OperatorFunctionIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) OperatorFunctionIdAST();
  ast->operatorLoc = SourceLocation(node->operator_loc());
  ast->opLoc = SourceLocation(node->op_loc());
  ast->openLoc = SourceLocation(node->open_loc());
  ast->closeLoc = SourceLocation(node->close_loc());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeLiteralOperatorId(const io::LiteralOperatorId* node)
    -> LiteralOperatorIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LiteralOperatorIdAST();
  ast->operatorLoc = SourceLocation(node->operator_loc());
  ast->literalLoc = SourceLocation(node->literal_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeConversionFunctionId(
    const io::ConversionFunctionId* node) -> ConversionFunctionIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConversionFunctionIdAST();
  ast->operatorLoc = SourceLocation(node->operator_loc());
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeSimpleTemplateId(const io::SimpleTemplateId* node)
    -> SimpleTemplateIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleTemplateIdAST();
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->lessLoc = SourceLocation(node->less_loc());
  if (node->template_argument_list()) {
    auto* inserter = &ast->templateArgumentList;
    for (std::uint32_t i = 0; i < node->template_argument_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeTemplateArgument(
          node->template_argument_list()->Get(i),
          io::TemplateArgument(node->template_argument_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->greaterLoc = SourceLocation(node->greater_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeLiteralOperatorTemplateId(
    const io::LiteralOperatorTemplateId* node)
    -> LiteralOperatorTemplateIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LiteralOperatorTemplateIdAST();
  ast->literalOperatorId = decodeLiteralOperatorId(node->literal_operator_id());
  ast->lessLoc = SourceLocation(node->less_loc());
  if (node->template_argument_list()) {
    auto* inserter = &ast->templateArgumentList;
    for (std::uint32_t i = 0; i < node->template_argument_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeTemplateArgument(
          node->template_argument_list()->Get(i),
          io::TemplateArgument(node->template_argument_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->greaterLoc = SourceLocation(node->greater_loc());
  return ast;
}

auto ASTDecoder::decodeOperatorFunctionTemplateId(
    const io::OperatorFunctionTemplateId* node)
    -> OperatorFunctionTemplateIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) OperatorFunctionTemplateIdAST();
  ast->operatorFunctionId =
      decodeOperatorFunctionId(node->operator_function_id());
  ast->lessLoc = SourceLocation(node->less_loc());
  if (node->template_argument_list()) {
    auto* inserter = &ast->templateArgumentList;
    for (std::uint32_t i = 0; i < node->template_argument_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeTemplateArgument(
          node->template_argument_list()->Get(i),
          io::TemplateArgument(node->template_argument_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->greaterLoc = SourceLocation(node->greater_loc());
  return ast;
}

auto ASTDecoder::decodeGlobalNestedNameSpecifier(
    const io::GlobalNestedNameSpecifier* node)
    -> GlobalNestedNameSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GlobalNestedNameSpecifierAST();
  ast->scopeLoc = SourceLocation(node->scope_loc());
  return ast;
}

auto ASTDecoder::decodeSimpleNestedNameSpecifier(
    const io::SimpleNestedNameSpecifier* node)
    -> SimpleNestedNameSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleNestedNameSpecifierAST();
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  ast->scopeLoc = SourceLocation(node->scope_loc());
  return ast;
}

auto ASTDecoder::decodeDecltypeNestedNameSpecifier(
    const io::DecltypeNestedNameSpecifier* node)
    -> DecltypeNestedNameSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DecltypeNestedNameSpecifierAST();
  ast->decltypeSpecifier = decodeDecltypeSpecifier(node->decltype_specifier());
  ast->scopeLoc = SourceLocation(node->scope_loc());
  return ast;
}

auto ASTDecoder::decodeTemplateNestedNameSpecifier(
    const io::TemplateNestedNameSpecifier* node)
    -> TemplateNestedNameSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TemplateNestedNameSpecifierAST();
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->templateId = decodeSimpleTemplateId(node->template_id());
  ast->scopeLoc = SourceLocation(node->scope_loc());
  return ast;
}

auto ASTDecoder::decodeDefaultFunctionBody(const io::DefaultFunctionBody* node)
    -> DefaultFunctionBodyAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DefaultFunctionBodyAST();
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->defaultLoc = SourceLocation(node->default_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeCompoundStatementFunctionBody(
    const io::CompoundStatementFunctionBody* node)
    -> CompoundStatementFunctionBodyAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CompoundStatementFunctionBodyAST();
  ast->colonLoc = SourceLocation(node->colon_loc());
  if (node->mem_initializer_list()) {
    auto* inserter = &ast->memInitializerList;
    for (std::uint32_t i = 0; i < node->mem_initializer_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeMemInitializer(
          node->mem_initializer_list()->Get(i),
          io::MemInitializer(node->mem_initializer_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->statement = decodeCompoundStatement(node->statement());
  return ast;
}

auto ASTDecoder::decodeTryStatementFunctionBody(
    const io::TryStatementFunctionBody* node) -> TryStatementFunctionBodyAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TryStatementFunctionBodyAST();
  ast->tryLoc = SourceLocation(node->try_loc());
  ast->colonLoc = SourceLocation(node->colon_loc());
  if (node->mem_initializer_list()) {
    auto* inserter = &ast->memInitializerList;
    for (std::uint32_t i = 0; i < node->mem_initializer_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeMemInitializer(
          node->mem_initializer_list()->Get(i),
          io::MemInitializer(node->mem_initializer_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->statement = decodeCompoundStatement(node->statement());
  if (node->handler_list()) {
    auto* inserter = &ast->handlerList;
    for (std::uint32_t i = 0; i < node->handler_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeHandler(node->handler_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeDeleteFunctionBody(const io::DeleteFunctionBody* node)
    -> DeleteFunctionBodyAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DeleteFunctionBodyAST();
  ast->equalLoc = SourceLocation(node->equal_loc());
  ast->deleteLoc = SourceLocation(node->delete_loc());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeTypeTemplateArgument(
    const io::TypeTemplateArgument* node) -> TypeTemplateArgumentAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeTemplateArgumentAST();
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeExpressionTemplateArgument(
    const io::ExpressionTemplateArgument* node)
    -> ExpressionTemplateArgumentAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExpressionTemplateArgumentAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeThrowExceptionSpecifier(
    const io::ThrowExceptionSpecifier* node) -> ThrowExceptionSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThrowExceptionSpecifierAST();
  ast->throwLoc = SourceLocation(node->throw_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeNoexceptSpecifier(const io::NoexceptSpecifier* node)
    -> NoexceptSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NoexceptSpecifierAST();
  ast->noexceptLoc = SourceLocation(node->noexcept_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeSimpleRequirement(const io::SimpleRequirement* node)
    -> SimpleRequirementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleRequirementAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeCompoundRequirement(const io::CompoundRequirement* node)
    -> CompoundRequirementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CompoundRequirementAST();
  ast->lbraceLoc = SourceLocation(node->lbrace_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->rbraceLoc = SourceLocation(node->rbrace_loc());
  ast->noexceptLoc = SourceLocation(node->noexcept_loc());
  ast->minusGreaterLoc = SourceLocation(node->minus_greater_loc());
  ast->typeConstraint = decodeTypeConstraint(node->type_constraint());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeTypeRequirement(const io::TypeRequirement* node)
    -> TypeRequirementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeRequirementAST();
  ast->typenameLoc = SourceLocation(node->typename_loc());
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->templateLoc = SourceLocation(node->template_loc());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeNestedRequirement(const io::NestedRequirement* node)
    -> NestedRequirementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NestedRequirementAST();
  ast->requiresLoc = SourceLocation(node->requires_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->semicolonLoc = SourceLocation(node->semicolon_loc());
  return ast;
}

auto ASTDecoder::decodeNewParenInitializer(const io::NewParenInitializer* node)
    -> NewParenInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NewParenInitializerAST();
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::uint32_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeNewBracedInitializer(
    const io::NewBracedInitializer* node) -> NewBracedInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NewBracedInitializerAST();
  ast->bracedInitList = decodeBracedInitList(node->braced_init_list());
  return ast;
}

auto ASTDecoder::decodeParenMemInitializer(const io::ParenMemInitializer* node)
    -> ParenMemInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ParenMemInitializerAST();
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::uint32_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  return ast;
}

auto ASTDecoder::decodeBracedMemInitializer(
    const io::BracedMemInitializer* node) -> BracedMemInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BracedMemInitializerAST();
  ast->nestedNameSpecifier = decodeNestedNameSpecifier(
      node->nested_name_specifier(), node->nested_name_specifier_type());
  ast->unqualifiedId =
      decodeUnqualifiedId(node->unqualified_id(), node->unqualified_id_type());
  ast->bracedInitList = decodeBracedInitList(node->braced_init_list());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  return ast;
}

auto ASTDecoder::decodeThisLambdaCapture(const io::ThisLambdaCapture* node)
    -> ThisLambdaCaptureAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThisLambdaCaptureAST();
  ast->thisLoc = SourceLocation(node->this_loc());
  return ast;
}

auto ASTDecoder::decodeDerefThisLambdaCapture(
    const io::DerefThisLambdaCapture* node) -> DerefThisLambdaCaptureAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DerefThisLambdaCaptureAST();
  ast->starLoc = SourceLocation(node->star_loc());
  ast->thisLoc = SourceLocation(node->this_loc());
  return ast;
}

auto ASTDecoder::decodeSimpleLambdaCapture(const io::SimpleLambdaCapture* node)
    -> SimpleLambdaCaptureAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleLambdaCaptureAST();
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeRefLambdaCapture(const io::RefLambdaCapture* node)
    -> RefLambdaCaptureAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RefLambdaCaptureAST();
  ast->ampLoc = SourceLocation(node->amp_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeRefInitLambdaCapture(
    const io::RefInitLambdaCapture* node) -> RefInitLambdaCaptureAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RefInitLambdaCaptureAST();
  ast->ampLoc = SourceLocation(node->amp_loc());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->initializer =
      decodeExpression(node->initializer(), node->initializer_type());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeInitLambdaCapture(const io::InitLambdaCapture* node)
    -> InitLambdaCaptureAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) InitLambdaCaptureAST();
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  ast->initializer =
      decodeExpression(node->initializer(), node->initializer_type());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeEllipsisExceptionDeclaration(
    const io::EllipsisExceptionDeclaration* node)
    -> EllipsisExceptionDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EllipsisExceptionDeclarationAST();
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  return ast;
}

auto ASTDecoder::decodeTypeExceptionDeclaration(
    const io::TypeExceptionDeclaration* node) -> TypeExceptionDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeExceptionDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::uint32_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->declarator = decodeDeclarator(node->declarator());
  return ast;
}

auto ASTDecoder::decodeCxxAttribute(const io::CxxAttribute* node)
    -> CxxAttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CxxAttributeAST();
  ast->lbracketLoc = SourceLocation(node->lbracket_loc());
  ast->lbracket2Loc = SourceLocation(node->lbracket2_loc());
  ast->attributeUsingPrefix =
      decodeAttributeUsingPrefix(node->attribute_using_prefix());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::uint32_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter =
          new (pool_) List(decodeAttribute(node->attribute_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->rbracketLoc = SourceLocation(node->rbracket_loc());
  ast->rbracket2Loc = SourceLocation(node->rbracket2_loc());
  return ast;
}

auto ASTDecoder::decodeGccAttribute(const io::GccAttribute* node)
    -> GccAttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GccAttributeAST();
  ast->attributeLoc = SourceLocation(node->attribute_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->lparen2Loc = SourceLocation(node->lparen2_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  ast->rparen2Loc = SourceLocation(node->rparen2_loc());
  return ast;
}

auto ASTDecoder::decodeAlignasAttribute(const io::AlignasAttribute* node)
    -> AlignasAttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AlignasAttributeAST();
  ast->alignasLoc = SourceLocation(node->alignas_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeAlignasTypeAttribute(
    const io::AlignasTypeAttribute* node) -> AlignasTypeAttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AlignasTypeAttributeAST();
  ast->alignasLoc = SourceLocation(node->alignas_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->typeId = decodeTypeId(node->type_id());
  ast->ellipsisLoc = SourceLocation(node->ellipsis_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeAsmAttribute(const io::AsmAttribute* node)
    -> AsmAttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AsmAttributeAST();
  ast->asmLoc = SourceLocation(node->asm_loc());
  ast->lparenLoc = SourceLocation(node->lparen_loc());
  ast->literalLoc = SourceLocation(node->literal_loc());
  ast->rparenLoc = SourceLocation(node->rparen_loc());
  return ast;
}

auto ASTDecoder::decodeScopedAttributeToken(
    const io::ScopedAttributeToken* node) -> ScopedAttributeTokenAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ScopedAttributeTokenAST();
  ast->attributeNamespaceLoc = SourceLocation(node->attribute_namespace_loc());
  ast->scopeLoc = SourceLocation(node->scope_loc());
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->attribute_namespace()) {
    ast->attributeNamespace =
        unit_->control()->getIdentifier(node->attribute_namespace()->str());
  }
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeSimpleAttributeToken(
    const io::SimpleAttributeToken* node) -> SimpleAttributeTokenAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleAttributeTokenAST();
  ast->identifierLoc = SourceLocation(node->identifier_loc());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

}  // namespace cxx
