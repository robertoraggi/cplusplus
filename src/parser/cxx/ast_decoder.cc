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

auto ASTDecoder::decodeInitializer(const void* ptr, io::Initializer type)
    -> InitializerAST* {
  switch (type) {
    case io::Initializer_EqualInitializer:
      return decodeEqualInitializer(
          reinterpret_cast<const io::EqualInitializer*>(ptr));
    case io::Initializer_BracedInitList:
      return decodeBracedInitList(
          reinterpret_cast<const io::BracedInitList*>(ptr));
    case io::Initializer_ParenInitializer:
      return decodeParenInitializer(
          reinterpret_cast<const io::ParenInitializer*>(ptr));
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

auto ASTDecoder::decodeExpression(const void* ptr, io::Expression type)
    -> ExpressionAST* {
  switch (type) {
    case io::Expression_ThisExpression:
      return decodeThisExpression(
          reinterpret_cast<const io::ThisExpression*>(ptr));
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
    case io::Expression_IdExpression:
      return decodeIdExpression(reinterpret_cast<const io::IdExpression*>(ptr));
    case io::Expression_RequiresExpression:
      return decodeRequiresExpression(
          reinterpret_cast<const io::RequiresExpression*>(ptr));
    case io::Expression_NestedExpression:
      return decodeNestedExpression(
          reinterpret_cast<const io::NestedExpression*>(ptr));
    case io::Expression_RightFoldExpression:
      return decodeRightFoldExpression(
          reinterpret_cast<const io::RightFoldExpression*>(ptr));
    case io::Expression_LeftFoldExpression:
      return decodeLeftFoldExpression(
          reinterpret_cast<const io::LeftFoldExpression*>(ptr));
    case io::Expression_FoldExpression:
      return decodeFoldExpression(
          reinterpret_cast<const io::FoldExpression*>(ptr));
    case io::Expression_LambdaExpression:
      return decodeLambdaExpression(
          reinterpret_cast<const io::LambdaExpression*>(ptr));
    case io::Expression_SizeofExpression:
      return decodeSizeofExpression(
          reinterpret_cast<const io::SizeofExpression*>(ptr));
    case io::Expression_SizeofTypeExpression:
      return decodeSizeofTypeExpression(
          reinterpret_cast<const io::SizeofTypeExpression*>(ptr));
    case io::Expression_SizeofPackExpression:
      return decodeSizeofPackExpression(
          reinterpret_cast<const io::SizeofPackExpression*>(ptr));
    case io::Expression_TypeidExpression:
      return decodeTypeidExpression(
          reinterpret_cast<const io::TypeidExpression*>(ptr));
    case io::Expression_TypeidOfTypeExpression:
      return decodeTypeidOfTypeExpression(
          reinterpret_cast<const io::TypeidOfTypeExpression*>(ptr));
    case io::Expression_AlignofExpression:
      return decodeAlignofExpression(
          reinterpret_cast<const io::AlignofExpression*>(ptr));
    case io::Expression_TypeTraitsExpression:
      return decodeTypeTraitsExpression(
          reinterpret_cast<const io::TypeTraitsExpression*>(ptr));
    case io::Expression_UnaryExpression:
      return decodeUnaryExpression(
          reinterpret_cast<const io::UnaryExpression*>(ptr));
    case io::Expression_BinaryExpression:
      return decodeBinaryExpression(
          reinterpret_cast<const io::BinaryExpression*>(ptr));
    case io::Expression_AssignmentExpression:
      return decodeAssignmentExpression(
          reinterpret_cast<const io::AssignmentExpression*>(ptr));
    case io::Expression_BracedTypeConstruction:
      return decodeBracedTypeConstruction(
          reinterpret_cast<const io::BracedTypeConstruction*>(ptr));
    case io::Expression_TypeConstruction:
      return decodeTypeConstruction(
          reinterpret_cast<const io::TypeConstruction*>(ptr));
    case io::Expression_CallExpression:
      return decodeCallExpression(
          reinterpret_cast<const io::CallExpression*>(ptr));
    case io::Expression_SubscriptExpression:
      return decodeSubscriptExpression(
          reinterpret_cast<const io::SubscriptExpression*>(ptr));
    case io::Expression_MemberExpression:
      return decodeMemberExpression(
          reinterpret_cast<const io::MemberExpression*>(ptr));
    case io::Expression_PostIncrExpression:
      return decodePostIncrExpression(
          reinterpret_cast<const io::PostIncrExpression*>(ptr));
    case io::Expression_ConditionalExpression:
      return decodeConditionalExpression(
          reinterpret_cast<const io::ConditionalExpression*>(ptr));
    case io::Expression_ImplicitCastExpression:
      return decodeImplicitCastExpression(
          reinterpret_cast<const io::ImplicitCastExpression*>(ptr));
    case io::Expression_CastExpression:
      return decodeCastExpression(
          reinterpret_cast<const io::CastExpression*>(ptr));
    case io::Expression_CppCastExpression:
      return decodeCppCastExpression(
          reinterpret_cast<const io::CppCastExpression*>(ptr));
    case io::Expression_NewExpression:
      return decodeNewExpression(
          reinterpret_cast<const io::NewExpression*>(ptr));
    case io::Expression_DeleteExpression:
      return decodeDeleteExpression(
          reinterpret_cast<const io::DeleteExpression*>(ptr));
    case io::Expression_ThrowExpression:
      return decodeThrowExpression(
          reinterpret_cast<const io::ThrowExpression*>(ptr));
    case io::Expression_NoexceptExpression:
      return decodeNoexceptExpression(
          reinterpret_cast<const io::NoexceptExpression*>(ptr));
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
    case io::Statement_GotoStatement:
      return decodeGotoStatement(
          reinterpret_cast<const io::GotoStatement*>(ptr));
    case io::Statement_CoroutineReturnStatement:
      return decodeCoroutineReturnStatement(
          reinterpret_cast<const io::CoroutineReturnStatement*>(ptr));
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

auto ASTDecoder::decodeDeclaration(const void* ptr, io::Declaration type)
    -> DeclarationAST* {
  switch (type) {
    case io::Declaration_AccessDeclaration:
      return decodeAccessDeclaration(
          reinterpret_cast<const io::AccessDeclaration*>(ptr));
    case io::Declaration_FunctionDefinition:
      return decodeFunctionDefinition(
          reinterpret_cast<const io::FunctionDefinition*>(ptr));
    case io::Declaration_ConceptDefinition:
      return decodeConceptDefinition(
          reinterpret_cast<const io::ConceptDefinition*>(ptr));
    case io::Declaration_ForRangeDeclaration:
      return decodeForRangeDeclaration(
          reinterpret_cast<const io::ForRangeDeclaration*>(ptr));
    case io::Declaration_AliasDeclaration:
      return decodeAliasDeclaration(
          reinterpret_cast<const io::AliasDeclaration*>(ptr));
    case io::Declaration_SimpleDeclaration:
      return decodeSimpleDeclaration(
          reinterpret_cast<const io::SimpleDeclaration*>(ptr));
    case io::Declaration_StaticAssertDeclaration:
      return decodeStaticAssertDeclaration(
          reinterpret_cast<const io::StaticAssertDeclaration*>(ptr));
    case io::Declaration_EmptyDeclaration:
      return decodeEmptyDeclaration(
          reinterpret_cast<const io::EmptyDeclaration*>(ptr));
    case io::Declaration_AttributeDeclaration:
      return decodeAttributeDeclaration(
          reinterpret_cast<const io::AttributeDeclaration*>(ptr));
    case io::Declaration_OpaqueEnumDeclaration:
      return decodeOpaqueEnumDeclaration(
          reinterpret_cast<const io::OpaqueEnumDeclaration*>(ptr));
    case io::Declaration_UsingEnumDeclaration:
      return decodeUsingEnumDeclaration(
          reinterpret_cast<const io::UsingEnumDeclaration*>(ptr));
    case io::Declaration_NamespaceDefinition:
      return decodeNamespaceDefinition(
          reinterpret_cast<const io::NamespaceDefinition*>(ptr));
    case io::Declaration_NamespaceAliasDefinition:
      return decodeNamespaceAliasDefinition(
          reinterpret_cast<const io::NamespaceAliasDefinition*>(ptr));
    case io::Declaration_UsingDirective:
      return decodeUsingDirective(
          reinterpret_cast<const io::UsingDirective*>(ptr));
    case io::Declaration_UsingDeclaration:
      return decodeUsingDeclaration(
          reinterpret_cast<const io::UsingDeclaration*>(ptr));
    case io::Declaration_AsmDeclaration:
      return decodeAsmDeclaration(
          reinterpret_cast<const io::AsmDeclaration*>(ptr));
    case io::Declaration_ExportDeclaration:
      return decodeExportDeclaration(
          reinterpret_cast<const io::ExportDeclaration*>(ptr));
    case io::Declaration_ExportCompoundDeclaration:
      return decodeExportCompoundDeclaration(
          reinterpret_cast<const io::ExportCompoundDeclaration*>(ptr));
    case io::Declaration_ModuleImportDeclaration:
      return decodeModuleImportDeclaration(
          reinterpret_cast<const io::ModuleImportDeclaration*>(ptr));
    case io::Declaration_TemplateDeclaration:
      return decodeTemplateDeclaration(
          reinterpret_cast<const io::TemplateDeclaration*>(ptr));
    case io::Declaration_TypenameTypeParameter:
      return decodeTypenameTypeParameter(
          reinterpret_cast<const io::TypenameTypeParameter*>(ptr));
    case io::Declaration_TemplateTypeParameter:
      return decodeTemplateTypeParameter(
          reinterpret_cast<const io::TemplateTypeParameter*>(ptr));
    case io::Declaration_TemplatePackTypeParameter:
      return decodeTemplatePackTypeParameter(
          reinterpret_cast<const io::TemplatePackTypeParameter*>(ptr));
    case io::Declaration_DeductionGuide:
      return decodeDeductionGuide(
          reinterpret_cast<const io::DeductionGuide*>(ptr));
    case io::Declaration_ExplicitInstantiation:
      return decodeExplicitInstantiation(
          reinterpret_cast<const io::ExplicitInstantiation*>(ptr));
    case io::Declaration_ParameterDeclaration:
      return decodeParameterDeclaration(
          reinterpret_cast<const io::ParameterDeclaration*>(ptr));
    case io::Declaration_LinkageSpecification:
      return decodeLinkageSpecification(
          reinterpret_cast<const io::LinkageSpecification*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeName(const void* ptr, io::Name type) -> NameAST* {
  switch (type) {
    case io::Name_SimpleName:
      return decodeSimpleName(reinterpret_cast<const io::SimpleName*>(ptr));
    case io::Name_DestructorName:
      return decodeDestructorName(
          reinterpret_cast<const io::DestructorName*>(ptr));
    case io::Name_DecltypeName:
      return decodeDecltypeName(reinterpret_cast<const io::DecltypeName*>(ptr));
    case io::Name_OperatorName:
      return decodeOperatorName(reinterpret_cast<const io::OperatorName*>(ptr));
    case io::Name_ConversionName:
      return decodeConversionName(
          reinterpret_cast<const io::ConversionName*>(ptr));
    case io::Name_TemplateName:
      return decodeTemplateName(reinterpret_cast<const io::TemplateName*>(ptr));
    case io::Name_QualifiedName:
      return decodeQualifiedName(
          reinterpret_cast<const io::QualifiedName*>(ptr));
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeSpecifier(const void* ptr, io::Specifier type)
    -> SpecifierAST* {
  switch (type) {
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
    case io::Specifier_StaticSpecifier:
      return decodeStaticSpecifier(
          reinterpret_cast<const io::StaticSpecifier*>(ptr));
    case io::Specifier_ExternSpecifier:
      return decodeExternSpecifier(
          reinterpret_cast<const io::ExternSpecifier*>(ptr));
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
    default:
      return nullptr;
  }  // switch
}

auto ASTDecoder::decodeCoreDeclarator(const void* ptr, io::CoreDeclarator type)
    -> CoreDeclaratorAST* {
  switch (type) {
    case io::CoreDeclarator_IdDeclarator:
      return decodeIdDeclarator(reinterpret_cast<const io::IdDeclarator*>(ptr));
    case io::CoreDeclarator_NestedDeclarator:
      return decodeNestedDeclarator(
          reinterpret_cast<const io::NestedDeclarator*>(ptr));
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

auto ASTDecoder::decodeDeclaratorModifier(const void* ptr,
                                          io::DeclaratorModifier type)
    -> DeclaratorModifierAST* {
  switch (type) {
    case io::DeclaratorModifier_FunctionDeclarator:
      return decodeFunctionDeclarator(
          reinterpret_cast<const io::FunctionDeclarator*>(ptr));
    case io::DeclaratorModifier_ArrayDeclarator:
      return decodeArrayDeclarator(
          reinterpret_cast<const io::ArrayDeclarator*>(ptr));
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
    case io::AttributeSpecifier_GCCAttribute:
      return decodeGCCAttribute(reinterpret_cast<const io::GCCAttribute*>(ptr));
    case io::AttributeSpecifier_AlignasAttribute:
      return decodeAlignasAttribute(
          reinterpret_cast<const io::AlignasAttribute*>(ptr));
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

auto ASTDecoder::decodeTypeId(const io::TypeId* node) -> TypeIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeIdAST();
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::size_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->declarator = decodeDeclarator(node->declarator());
  return ast;
}

auto ASTDecoder::decodeNestedNameSpecifier(const io::NestedNameSpecifier* node)
    -> NestedNameSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NestedNameSpecifierAST();
  if (node->name_list()) {
    auto* inserter = &ast->nameList;
    for (std::size_t i = 0; i < node->name_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeName(
          node->name_list()->Get(i), io::Name(node->name_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeUsingDeclarator(const io::UsingDeclarator* node)
    -> UsingDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UsingDeclaratorAST();
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  return ast;
}

auto ASTDecoder::decodeHandler(const io::Handler* node) -> HandlerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) HandlerAST();
  ast->exceptionDeclaration = decodeExceptionDeclaration(
      node->exception_declaration(), node->exception_declaration_type());
  ast->statement = decodeCompoundStatement(node->statement());
  return ast;
}

auto ASTDecoder::decodeEnumBase(const io::EnumBase* node) -> EnumBaseAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EnumBaseAST();
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::size_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeEnumerator(const io::Enumerator* node)
    -> EnumeratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EnumeratorAST();
  ast->name = decodeName(node->name(), node->name_type());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeDeclarator(const io::Declarator* node)
    -> DeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DeclaratorAST();
  if (node->ptr_op_list()) {
    auto* inserter = &ast->ptrOpList;
    for (std::size_t i = 0; i < node->ptr_op_list()->size(); ++i) {
      *inserter = new (pool_) List(
          decodePtrOperator(node->ptr_op_list()->Get(i),
                            io::PtrOperator(node->ptr_op_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->coreDeclarator = decodeCoreDeclarator(node->core_declarator(),
                                             node->core_declarator_type());
  if (node->modifiers()) {
    auto* inserter = &ast->modifiers;
    for (std::size_t i = 0; i < node->modifiers()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaratorModifier(
          node->modifiers()->Get(i),
          io::DeclaratorModifier(node->modifiers_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeInitDeclarator(const io::InitDeclarator* node)
    -> InitDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) InitDeclaratorAST();
  ast->declarator = decodeDeclarator(node->declarator());
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->initializer =
      decodeInitializer(node->initializer(), node->initializer_type());
  return ast;
}

auto ASTDecoder::decodeBaseSpecifier(const io::BaseSpecifier* node)
    -> BaseSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BaseSpecifierAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->name = decodeName(node->name(), node->name_type());
  return ast;
}

auto ASTDecoder::decodeBaseClause(const io::BaseClause* node)
    -> BaseClauseAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BaseClauseAST();
  if (node->base_specifier_list()) {
    auto* inserter = &ast->baseSpecifierList;
    for (std::size_t i = 0; i < node->base_specifier_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeBaseSpecifier(node->base_specifier_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeNewTypeId(const io::NewTypeId* node) -> NewTypeIdAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NewTypeIdAST();
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::size_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeRequiresClause(const io::RequiresClause* node)
    -> RequiresClauseAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RequiresClauseAST();
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
    for (std::size_t i = 0; i < node->parameter_declaration_list()->size();
         ++i) {
      *inserter = new (pool_) List(decodeParameterDeclaration(
          node->parameter_declaration_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeParametersAndQualifiers(
    const io::ParametersAndQualifiers* node) -> ParametersAndQualifiersAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ParametersAndQualifiersAST();
  ast->parameterDeclarationClause =
      decodeParameterDeclarationClause(node->parameter_declaration_clause());
  if (node->cv_qualifier_list()) {
    auto* inserter = &ast->cvQualifierList;
    for (std::size_t i = 0; i < node->cv_qualifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->cv_qualifier_list()->Get(i),
          io::Specifier(node->cv_qualifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeLambdaIntroducer(const io::LambdaIntroducer* node)
    -> LambdaIntroducerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LambdaIntroducerAST();
  if (node->capture_list()) {
    auto* inserter = &ast->captureList;
    for (std::size_t i = 0; i < node->capture_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeLambdaCapture(
          node->capture_list()->Get(i),
          io::LambdaCapture(node->capture_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeLambdaDeclarator(const io::LambdaDeclarator* node)
    -> LambdaDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LambdaDeclaratorAST();
  ast->parameterDeclarationClause =
      decodeParameterDeclarationClause(node->parameter_declaration_clause());
  if (node->decl_specifier_list()) {
    auto* inserter = &ast->declSpecifierList;
    for (std::size_t i = 0; i < node->decl_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->decl_specifier_list()->Get(i),
          io::Specifier(node->decl_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->trailingReturnType =
      decodeTrailingReturnType(node->trailing_return_type());
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  return ast;
}

auto ASTDecoder::decodeTrailingReturnType(const io::TrailingReturnType* node)
    -> TrailingReturnTypeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TrailingReturnTypeAST();
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeCtorInitializer(const io::CtorInitializer* node)
    -> CtorInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CtorInitializerAST();
  if (node->mem_initializer_list()) {
    auto* inserter = &ast->memInitializerList;
    for (std::size_t i = 0; i < node->mem_initializer_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeMemInitializer(
          node->mem_initializer_list()->Get(i),
          io::MemInitializer(node->mem_initializer_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeRequirementBody(const io::RequirementBody* node)
    -> RequirementBodyAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RequirementBodyAST();
  if (node->requirement_list()) {
    auto* inserter = &ast->requirementList;
    for (std::size_t i = 0; i < node->requirement_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeRequirement(
          node->requirement_list()->Get(i),
          io::Requirement(node->requirement_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeTypeConstraint(const io::TypeConstraint* node)
    -> TypeConstraintAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeConstraintAST();
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  return ast;
}

auto ASTDecoder::decodeGlobalModuleFragment(
    const io::GlobalModuleFragment* node) -> GlobalModuleFragmentAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GlobalModuleFragmentAST();
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::size_t i = 0; i < node->declaration_list()->size(); ++i) {
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
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::size_t i = 0; i < node->declaration_list()->size(); ++i) {
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
  ast->moduleName = decodeModuleName(node->module_name());
  ast->modulePartition = decodeModulePartition(node->module_partition());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeModuleName(const io::ModuleName* node)
    -> ModuleNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ModuleNameAST();
  return ast;
}

auto ASTDecoder::decodeImportName(const io::ImportName* node)
    -> ImportNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ImportNameAST();
  ast->modulePartition = decodeModulePartition(node->module_partition());
  ast->moduleName = decodeModuleName(node->module_name());
  return ast;
}

auto ASTDecoder::decodeModulePartition(const io::ModulePartition* node)
    -> ModulePartitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ModulePartitionAST();
  ast->moduleName = decodeModuleName(node->module_name());
  return ast;
}

auto ASTDecoder::decodeAttributeArgumentClause(
    const io::AttributeArgumentClause* node) -> AttributeArgumentClauseAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AttributeArgumentClauseAST();
  return ast;
}

auto ASTDecoder::decodeAttribute(const io::Attribute* node) -> AttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AttributeAST();
  ast->attributeToken = decodeAttributeToken(node->attribute_token(),
                                             node->attribute_token_type());
  ast->attributeArgumentClause =
      decodeAttributeArgumentClause(node->attribute_argument_clause());
  return ast;
}

auto ASTDecoder::decodeAttributeUsingPrefix(
    const io::AttributeUsingPrefix* node) -> AttributeUsingPrefixAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AttributeUsingPrefixAST();
  return ast;
}

auto ASTDecoder::decodeSimpleRequirement(const io::SimpleRequirement* node)
    -> SimpleRequirementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleRequirementAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeCompoundRequirement(const io::CompoundRequirement* node)
    -> CompoundRequirementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CompoundRequirementAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->typeConstraint = decodeTypeConstraint(node->type_constraint());
  return ast;
}

auto ASTDecoder::decodeTypeRequirement(const io::TypeRequirement* node)
    -> TypeRequirementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeRequirementAST();
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  return ast;
}

auto ASTDecoder::decodeNestedRequirement(const io::NestedRequirement* node)
    -> NestedRequirementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NestedRequirementAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
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

auto ASTDecoder::decodeParenMemInitializer(const io::ParenMemInitializer* node)
    -> ParenMemInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ParenMemInitializerAST();
  ast->name = decodeName(node->name(), node->name_type());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::size_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeBracedMemInitializer(
    const io::BracedMemInitializer* node) -> BracedMemInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BracedMemInitializerAST();
  ast->name = decodeName(node->name(), node->name_type());
  ast->bracedInitList = decodeBracedInitList(node->braced_init_list());
  return ast;
}

auto ASTDecoder::decodeThisLambdaCapture(const io::ThisLambdaCapture* node)
    -> ThisLambdaCaptureAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThisLambdaCaptureAST();
  return ast;
}

auto ASTDecoder::decodeDerefThisLambdaCapture(
    const io::DerefThisLambdaCapture* node) -> DerefThisLambdaCaptureAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DerefThisLambdaCaptureAST();
  return ast;
}

auto ASTDecoder::decodeSimpleLambdaCapture(const io::SimpleLambdaCapture* node)
    -> SimpleLambdaCaptureAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleLambdaCaptureAST();
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
  ast->initializer =
      decodeInitializer(node->initializer(), node->initializer_type());
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
  ast->initializer =
      decodeInitializer(node->initializer(), node->initializer_type());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeEqualInitializer(const io::EqualInitializer* node)
    -> EqualInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EqualInitializerAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeBracedInitList(const io::BracedInitList* node)
    -> BracedInitListAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BracedInitListAST();
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::size_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeParenInitializer(const io::ParenInitializer* node)
    -> ParenInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ParenInitializerAST();
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::size_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeNewParenInitializer(const io::NewParenInitializer* node)
    -> NewParenInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NewParenInitializerAST();
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::size_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeNewBracedInitializer(
    const io::NewBracedInitializer* node) -> NewBracedInitializerAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NewBracedInitializerAST();
  ast->bracedInit = decodeBracedInitList(node->braced_init());
  return ast;
}

auto ASTDecoder::decodeEllipsisExceptionDeclaration(
    const io::EllipsisExceptionDeclaration* node)
    -> EllipsisExceptionDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EllipsisExceptionDeclarationAST();
  return ast;
}

auto ASTDecoder::decodeTypeExceptionDeclaration(
    const io::TypeExceptionDeclaration* node) -> TypeExceptionDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeExceptionDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::size_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->declarator = decodeDeclarator(node->declarator());
  return ast;
}

auto ASTDecoder::decodeDefaultFunctionBody(const io::DefaultFunctionBody* node)
    -> DefaultFunctionBodyAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DefaultFunctionBodyAST();
  return ast;
}

auto ASTDecoder::decodeCompoundStatementFunctionBody(
    const io::CompoundStatementFunctionBody* node)
    -> CompoundStatementFunctionBodyAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CompoundStatementFunctionBodyAST();
  ast->ctorInitializer = decodeCtorInitializer(node->ctor_initializer());
  ast->statement = decodeCompoundStatement(node->statement());
  return ast;
}

auto ASTDecoder::decodeTryStatementFunctionBody(
    const io::TryStatementFunctionBody* node) -> TryStatementFunctionBodyAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TryStatementFunctionBodyAST();
  ast->ctorInitializer = decodeCtorInitializer(node->ctor_initializer());
  ast->statement = decodeCompoundStatement(node->statement());
  if (node->handler_list()) {
    auto* inserter = &ast->handlerList;
    for (std::size_t i = 0; i < node->handler_list()->size(); ++i) {
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
  return ast;
}

auto ASTDecoder::decodeTranslationUnit(const io::TranslationUnit* node)
    -> TranslationUnitAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TranslationUnitAST();
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::size_t i = 0; i < node->declaration_list()->size(); ++i) {
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
    for (std::size_t i = 0; i < node->declaration_list()->size(); ++i) {
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

auto ASTDecoder::decodeThisExpression(const io::ThisExpression* node)
    -> ThisExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThisExpressionAST();
  return ast;
}

auto ASTDecoder::decodeCharLiteralExpression(
    const io::CharLiteralExpression* node) -> CharLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CharLiteralExpressionAST();
  if (node->literal()) {
    ast->literal = unit_->control()->charLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeBoolLiteralExpression(
    const io::BoolLiteralExpression* node) -> BoolLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BoolLiteralExpressionAST();
  ast->literal = static_cast<TokenKind>(node->literal());
  return ast;
}

auto ASTDecoder::decodeIntLiteralExpression(
    const io::IntLiteralExpression* node) -> IntLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IntLiteralExpressionAST();
  if (node->literal()) {
    ast->literal = unit_->control()->integerLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeFloatLiteralExpression(
    const io::FloatLiteralExpression* node) -> FloatLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FloatLiteralExpressionAST();
  if (node->literal()) {
    ast->literal = unit_->control()->floatLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeNullptrLiteralExpression(
    const io::NullptrLiteralExpression* node) -> NullptrLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NullptrLiteralExpressionAST();
  ast->literal = static_cast<TokenKind>(node->literal());
  return ast;
}

auto ASTDecoder::decodeStringLiteralExpression(
    const io::StringLiteralExpression* node) -> StringLiteralExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) StringLiteralExpressionAST();
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
  if (node->literal()) {
    ast->literal = unit_->control()->stringLiteral(node->literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeIdExpression(const io::IdExpression* node)
    -> IdExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IdExpressionAST();
  ast->name = decodeName(node->name(), node->name_type());
  return ast;
}

auto ASTDecoder::decodeRequiresExpression(const io::RequiresExpression* node)
    -> RequiresExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RequiresExpressionAST();
  ast->parameterDeclarationClause =
      decodeParameterDeclarationClause(node->parameter_declaration_clause());
  ast->requirementBody = decodeRequirementBody(node->requirement_body());
  return ast;
}

auto ASTDecoder::decodeNestedExpression(const io::NestedExpression* node)
    -> NestedExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NestedExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeRightFoldExpression(const io::RightFoldExpression* node)
    -> RightFoldExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RightFoldExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeLeftFoldExpression(const io::LeftFoldExpression* node)
    -> LeftFoldExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LeftFoldExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeFoldExpression(const io::FoldExpression* node)
    -> FoldExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FoldExpressionAST();
  ast->leftExpression =
      decodeExpression(node->left_expression(), node->left_expression_type());
  ast->rightExpression =
      decodeExpression(node->right_expression(), node->right_expression_type());
  ast->op = static_cast<TokenKind>(node->op());
  ast->foldOp = static_cast<TokenKind>(node->fold_op());
  return ast;
}

auto ASTDecoder::decodeLambdaExpression(const io::LambdaExpression* node)
    -> LambdaExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LambdaExpressionAST();
  ast->lambdaIntroducer = decodeLambdaIntroducer(node->lambda_introducer());
  if (node->template_parameter_list()) {
    auto* inserter = &ast->templateParameterList;
    for (std::size_t i = 0; i < node->template_parameter_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->template_parameter_list()->Get(i),
          io::Declaration(node->template_parameter_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->lambdaDeclarator = decodeLambdaDeclarator(node->lambda_declarator());
  ast->statement = decodeCompoundStatement(node->statement());
  return ast;
}

auto ASTDecoder::decodeSizeofExpression(const io::SizeofExpression* node)
    -> SizeofExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SizeofExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeSizeofTypeExpression(
    const io::SizeofTypeExpression* node) -> SizeofTypeExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SizeofTypeExpressionAST();
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeSizeofPackExpression(
    const io::SizeofPackExpression* node) -> SizeofPackExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SizeofPackExpressionAST();
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeTypeidExpression(const io::TypeidExpression* node)
    -> TypeidExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeidExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeTypeidOfTypeExpression(
    const io::TypeidOfTypeExpression* node) -> TypeidOfTypeExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeidOfTypeExpressionAST();
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeAlignofExpression(const io::AlignofExpression* node)
    -> AlignofExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AlignofExpressionAST();
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeTypeTraitsExpression(
    const io::TypeTraitsExpression* node) -> TypeTraitsExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeTraitsExpressionAST();
  if (node->type_id_list()) {
    auto* inserter = &ast->typeIdList;
    for (std::size_t i = 0; i < node->type_id_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeTypeId(node->type_id_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->typeTraits = static_cast<TokenKind>(node->type_traits());
  return ast;
}

auto ASTDecoder::decodeUnaryExpression(const io::UnaryExpression* node)
    -> UnaryExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UnaryExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeBinaryExpression(const io::BinaryExpression* node)
    -> BinaryExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BinaryExpressionAST();
  ast->leftExpression =
      decodeExpression(node->left_expression(), node->left_expression_type());
  ast->rightExpression =
      decodeExpression(node->right_expression(), node->right_expression_type());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeAssignmentExpression(
    const io::AssignmentExpression* node) -> AssignmentExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AssignmentExpressionAST();
  ast->leftExpression =
      decodeExpression(node->left_expression(), node->left_expression_type());
  ast->rightExpression =
      decodeExpression(node->right_expression(), node->right_expression_type());
  ast->op = static_cast<TokenKind>(node->op());
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

auto ASTDecoder::decodeTypeConstruction(const io::TypeConstruction* node)
    -> TypeConstructionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypeConstructionAST();
  ast->typeSpecifier =
      decodeSpecifier(node->type_specifier(), node->type_specifier_type());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::size_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeCallExpression(const io::CallExpression* node)
    -> CallExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CallExpressionAST();
  ast->baseExpression =
      decodeExpression(node->base_expression(), node->base_expression_type());
  if (node->expression_list()) {
    auto* inserter = &ast->expressionList;
    for (std::size_t i = 0; i < node->expression_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeExpression(
          node->expression_list()->Get(i),
          io::Expression(node->expression_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeSubscriptExpression(const io::SubscriptExpression* node)
    -> SubscriptExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SubscriptExpressionAST();
  ast->baseExpression =
      decodeExpression(node->base_expression(), node->base_expression_type());
  ast->indexExpression =
      decodeExpression(node->index_expression(), node->index_expression_type());
  return ast;
}

auto ASTDecoder::decodeMemberExpression(const io::MemberExpression* node)
    -> MemberExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) MemberExpressionAST();
  ast->baseExpression =
      decodeExpression(node->base_expression(), node->base_expression_type());
  ast->name = decodeName(node->name(), node->name_type());
  ast->accessOp = static_cast<TokenKind>(node->access_op());
  return ast;
}

auto ASTDecoder::decodePostIncrExpression(const io::PostIncrExpression* node)
    -> PostIncrExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) PostIncrExpressionAST();
  ast->baseExpression =
      decodeExpression(node->base_expression(), node->base_expression_type());
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeConditionalExpression(
    const io::ConditionalExpression* node) -> ConditionalExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConditionalExpressionAST();
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->iftrueExpression = decodeExpression(node->iftrue_expression(),
                                           node->iftrue_expression_type());
  ast->iffalseExpression = decodeExpression(node->iffalse_expression(),
                                            node->iffalse_expression_type());
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

auto ASTDecoder::decodeCastExpression(const io::CastExpression* node)
    -> CastExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CastExpressionAST();
  ast->typeId = decodeTypeId(node->type_id());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeCppCastExpression(const io::CppCastExpression* node)
    -> CppCastExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CppCastExpressionAST();
  ast->typeId = decodeTypeId(node->type_id());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeNewExpression(const io::NewExpression* node)
    -> NewExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NewExpressionAST();
  ast->typeId = decodeNewTypeId(node->type_id());
  ast->newInitalizer =
      decodeNewInitializer(node->new_initalizer(), node->new_initalizer_type());
  return ast;
}

auto ASTDecoder::decodeDeleteExpression(const io::DeleteExpression* node)
    -> DeleteExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DeleteExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeThrowExpression(const io::ThrowExpression* node)
    -> ThrowExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThrowExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeNoexceptExpression(const io::NoexceptExpression* node)
    -> NoexceptExpressionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NoexceptExpressionAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeLabeledStatement(const io::LabeledStatement* node)
    -> LabeledStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LabeledStatementAST();
  ast->statement = decodeStatement(node->statement(), node->statement_type());
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
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeDefaultStatement(const io::DefaultStatement* node)
    -> DefaultStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DefaultStatementAST();
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeExpressionStatement(const io::ExpressionStatement* node)
    -> ExpressionStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExpressionStatementAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeCompoundStatement(const io::CompoundStatement* node)
    -> CompoundStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CompoundStatementAST();
  if (node->statement_list()) {
    auto* inserter = &ast->statementList;
    for (std::size_t i = 0; i < node->statement_list()->size(); ++i) {
      *inserter = new (pool_) List(
          decodeStatement(node->statement_list()->Get(i),
                          io::Statement(node->statement_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeIfStatement(const io::IfStatement* node)
    -> IfStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IfStatementAST();
  ast->initializer =
      decodeStatement(node->initializer(), node->initializer_type());
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  ast->elseStatement =
      decodeStatement(node->else_statement(), node->else_statement_type());
  return ast;
}

auto ASTDecoder::decodeSwitchStatement(const io::SwitchStatement* node)
    -> SwitchStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SwitchStatementAST();
  ast->initializer =
      decodeStatement(node->initializer(), node->initializer_type());
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeWhileStatement(const io::WhileStatement* node)
    -> WhileStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) WhileStatementAST();
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeDoStatement(const io::DoStatement* node)
    -> DoStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DoStatementAST();
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeForRangeStatement(const io::ForRangeStatement* node)
    -> ForRangeStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ForRangeStatementAST();
  ast->initializer =
      decodeStatement(node->initializer(), node->initializer_type());
  ast->rangeDeclaration = decodeDeclaration(node->range_declaration(),
                                            node->range_declaration_type());
  ast->rangeInitializer = decodeExpression(node->range_initializer(),
                                           node->range_initializer_type());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeForStatement(const io::ForStatement* node)
    -> ForStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ForStatementAST();
  ast->initializer =
      decodeStatement(node->initializer(), node->initializer_type());
  ast->condition = decodeExpression(node->condition(), node->condition_type());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  ast->statement = decodeStatement(node->statement(), node->statement_type());
  return ast;
}

auto ASTDecoder::decodeBreakStatement(const io::BreakStatement* node)
    -> BreakStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) BreakStatementAST();
  return ast;
}

auto ASTDecoder::decodeContinueStatement(const io::ContinueStatement* node)
    -> ContinueStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ContinueStatementAST();
  return ast;
}

auto ASTDecoder::decodeReturnStatement(const io::ReturnStatement* node)
    -> ReturnStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ReturnStatementAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeGotoStatement(const io::GotoStatement* node)
    -> GotoStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GotoStatementAST();
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeCoroutineReturnStatement(
    const io::CoroutineReturnStatement* node) -> CoroutineReturnStatementAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CoroutineReturnStatementAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
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
  ast->statement = decodeCompoundStatement(node->statement());
  if (node->handler_list()) {
    auto* inserter = &ast->handlerList;
    for (std::size_t i = 0; i < node->handler_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeHandler(node->handler_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeAccessDeclaration(const io::AccessDeclaration* node)
    -> AccessDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AccessDeclarationAST();
  return ast;
}

auto ASTDecoder::decodeFunctionDefinition(const io::FunctionDefinition* node)
    -> FunctionDefinitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FunctionDefinitionAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->decl_specifier_list()) {
    auto* inserter = &ast->declSpecifierList;
    for (std::size_t i = 0; i < node->decl_specifier_list()->size(); ++i) {
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

auto ASTDecoder::decodeConceptDefinition(const io::ConceptDefinition* node)
    -> ConceptDefinitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConceptDefinitionAST();
  ast->name = decodeName(node->name(), node->name_type());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeForRangeDeclaration(const io::ForRangeDeclaration* node)
    -> ForRangeDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ForRangeDeclarationAST();
  return ast;
}

auto ASTDecoder::decodeAliasDeclaration(const io::AliasDeclaration* node)
    -> AliasDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AliasDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->typeId = decodeTypeId(node->type_id());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeSimpleDeclaration(const io::SimpleDeclaration* node)
    -> SimpleDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->decl_specifier_list()) {
    auto* inserter = &ast->declSpecifierList;
    for (std::size_t i = 0; i < node->decl_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->decl_specifier_list()->Get(i),
          io::Specifier(node->decl_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->init_declarator_list()) {
    auto* inserter = &ast->initDeclaratorList;
    for (std::size_t i = 0; i < node->init_declarator_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeInitDeclarator(node->init_declarator_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  return ast;
}

auto ASTDecoder::decodeStaticAssertDeclaration(
    const io::StaticAssertDeclaration* node) -> StaticAssertDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) StaticAssertDeclarationAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeEmptyDeclaration(const io::EmptyDeclaration* node)
    -> EmptyDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EmptyDeclarationAST();
  return ast;
}

auto ASTDecoder::decodeAttributeDeclaration(
    const io::AttributeDeclaration* node) -> AttributeDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AttributeDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeOpaqueEnumDeclaration(
    const io::OpaqueEnumDeclaration* node) -> OpaqueEnumDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) OpaqueEnumDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  ast->enumBase = decodeEnumBase(node->enum_base());
  return ast;
}

auto ASTDecoder::decodeUsingEnumDeclaration(
    const io::UsingEnumDeclaration* node) -> UsingEnumDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UsingEnumDeclarationAST();
  return ast;
}

auto ASTDecoder::decodeNamespaceDefinition(const io::NamespaceDefinition* node)
    -> NamespaceDefinitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NamespaceDefinitionAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  if (node->extra_attribute_list()) {
    auto* inserter = &ast->extraAttributeList;
    for (std::size_t i = 0; i < node->extra_attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->extra_attribute_list()->Get(i),
          io::AttributeSpecifier(node->extra_attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::size_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeNamespaceAliasDefinition(
    const io::NamespaceAliasDefinition* node) -> NamespaceAliasDefinitionAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NamespaceAliasDefinitionAST();
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeUsingDirective(const io::UsingDirective* node)
    -> UsingDirectiveAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UsingDirectiveAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  return ast;
}

auto ASTDecoder::decodeUsingDeclaration(const io::UsingDeclaration* node)
    -> UsingDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UsingDeclarationAST();
  if (node->using_declarator_list()) {
    auto* inserter = &ast->usingDeclaratorList;
    for (std::size_t i = 0; i < node->using_declarator_list()->size(); ++i) {
      *inserter = new (pool_)
          List(decodeUsingDeclarator(node->using_declarator_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeAsmDeclaration(const io::AsmDeclaration* node)
    -> AsmDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AsmDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeExportDeclaration(const io::ExportDeclaration* node)
    -> ExportDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExportDeclarationAST();
  ast->declaration =
      decodeDeclaration(node->declaration(), node->declaration_type());
  return ast;
}

auto ASTDecoder::decodeExportCompoundDeclaration(
    const io::ExportCompoundDeclaration* node)
    -> ExportCompoundDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExportCompoundDeclarationAST();
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::size_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeModuleImportDeclaration(
    const io::ModuleImportDeclaration* node) -> ModuleImportDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ModuleImportDeclarationAST();
  ast->importName = decodeImportName(node->import_name());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeTemplateDeclaration(const io::TemplateDeclaration* node)
    -> TemplateDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TemplateDeclarationAST();
  if (node->template_parameter_list()) {
    auto* inserter = &ast->templateParameterList;
    for (std::size_t i = 0; i < node->template_parameter_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->template_parameter_list()->Get(i),
          io::Declaration(node->template_parameter_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->declaration =
      decodeDeclaration(node->declaration(), node->declaration_type());
  return ast;
}

auto ASTDecoder::decodeTypenameTypeParameter(
    const io::TypenameTypeParameter* node) -> TypenameTypeParameterAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypenameTypeParameterAST();
  ast->typeId = decodeTypeId(node->type_id());
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
  if (node->template_parameter_list()) {
    auto* inserter = &ast->templateParameterList;
    for (std::size_t i = 0; i < node->template_parameter_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->template_parameter_list()->Get(i),
          io::Declaration(node->template_parameter_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->requiresClause = decodeRequiresClause(node->requires_clause());
  ast->name = decodeName(node->name(), node->name_type());
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeTemplatePackTypeParameter(
    const io::TemplatePackTypeParameter* node)
    -> TemplatePackTypeParameterAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TemplatePackTypeParameterAST();
  if (node->template_parameter_list()) {
    auto* inserter = &ast->templateParameterList;
    for (std::size_t i = 0; i < node->template_parameter_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->template_parameter_list()->Get(i),
          io::Declaration(node->template_parameter_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
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
  return ast;
}

auto ASTDecoder::decodeExplicitInstantiation(
    const io::ExplicitInstantiation* node) -> ExplicitInstantiationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExplicitInstantiationAST();
  ast->declaration =
      decodeDeclaration(node->declaration(), node->declaration_type());
  return ast;
}

auto ASTDecoder::decodeParameterDeclaration(
    const io::ParameterDeclaration* node) -> ParameterDeclarationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ParameterDeclarationAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->type_specifier_list()) {
    auto* inserter = &ast->typeSpecifierList;
    for (std::size_t i = 0; i < node->type_specifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->type_specifier_list()->Get(i),
          io::Specifier(node->type_specifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->declarator = decodeDeclarator(node->declarator());
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeLinkageSpecification(
    const io::LinkageSpecification* node) -> LinkageSpecificationAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) LinkageSpecificationAST();
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::size_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->string_literal()) {
    ast->stringLiteral =
        unit_->control()->stringLiteral(node->string_literal()->str());
  }
  return ast;
}

auto ASTDecoder::decodeSimpleName(const io::SimpleName* node)
    -> SimpleNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleNameAST();
  if (node->identifier()) {
    ast->identifier =
        unit_->control()->getIdentifier(node->identifier()->str());
  }
  return ast;
}

auto ASTDecoder::decodeDestructorName(const io::DestructorName* node)
    -> DestructorNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DestructorNameAST();
  ast->id = decodeName(node->id(), node->id_type());
  return ast;
}

auto ASTDecoder::decodeDecltypeName(const io::DecltypeName* node)
    -> DecltypeNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DecltypeNameAST();
  ast->decltypeSpecifier = decodeSpecifier(node->decltype_specifier(),
                                           node->decltype_specifier_type());
  return ast;
}

auto ASTDecoder::decodeOperatorName(const io::OperatorName* node)
    -> OperatorNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) OperatorNameAST();
  ast->op = static_cast<TokenKind>(node->op());
  return ast;
}

auto ASTDecoder::decodeConversionName(const io::ConversionName* node)
    -> ConversionNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConversionNameAST();
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeTemplateName(const io::TemplateName* node)
    -> TemplateNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TemplateNameAST();
  ast->id = decodeName(node->id(), node->id_type());
  if (node->template_argument_list()) {
    auto* inserter = &ast->templateArgumentList;
    for (std::size_t i = 0; i < node->template_argument_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeTemplateArgument(
          node->template_argument_list()->Get(i),
          io::TemplateArgument(node->template_argument_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeQualifiedName(const io::QualifiedName* node)
    -> QualifiedNameAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) QualifiedNameAST();
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->id = decodeName(node->id(), node->id_type());
  return ast;
}

auto ASTDecoder::decodeTypedefSpecifier(const io::TypedefSpecifier* node)
    -> TypedefSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypedefSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeFriendSpecifier(const io::FriendSpecifier* node)
    -> FriendSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FriendSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeConstevalSpecifier(const io::ConstevalSpecifier* node)
    -> ConstevalSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConstevalSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeConstinitSpecifier(const io::ConstinitSpecifier* node)
    -> ConstinitSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConstinitSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeConstexprSpecifier(const io::ConstexprSpecifier* node)
    -> ConstexprSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ConstexprSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeInlineSpecifier(const io::InlineSpecifier* node)
    -> InlineSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) InlineSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeStaticSpecifier(const io::StaticSpecifier* node)
    -> StaticSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) StaticSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeExternSpecifier(const io::ExternSpecifier* node)
    -> ExternSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExternSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeThreadLocalSpecifier(
    const io::ThreadLocalSpecifier* node) -> ThreadLocalSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThreadLocalSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeThreadSpecifier(const io::ThreadSpecifier* node)
    -> ThreadSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ThreadSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeMutableSpecifier(const io::MutableSpecifier* node)
    -> MutableSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) MutableSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeVirtualSpecifier(const io::VirtualSpecifier* node)
    -> VirtualSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) VirtualSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeExplicitSpecifier(const io::ExplicitSpecifier* node)
    -> ExplicitSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ExplicitSpecifierAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeAutoTypeSpecifier(const io::AutoTypeSpecifier* node)
    -> AutoTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AutoTypeSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeVoidTypeSpecifier(const io::VoidTypeSpecifier* node)
    -> VoidTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) VoidTypeSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeVaListTypeSpecifier(const io::VaListTypeSpecifier* node)
    -> VaListTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) VaListTypeSpecifierAST();
  ast->specifier = static_cast<TokenKind>(node->specifier());
  return ast;
}

auto ASTDecoder::decodeIntegralTypeSpecifier(
    const io::IntegralTypeSpecifier* node) -> IntegralTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IntegralTypeSpecifierAST();
  ast->specifier = static_cast<TokenKind>(node->specifier());
  return ast;
}

auto ASTDecoder::decodeFloatingPointTypeSpecifier(
    const io::FloatingPointTypeSpecifier* node)
    -> FloatingPointTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FloatingPointTypeSpecifierAST();
  ast->specifier = static_cast<TokenKind>(node->specifier());
  return ast;
}

auto ASTDecoder::decodeComplexTypeSpecifier(
    const io::ComplexTypeSpecifier* node) -> ComplexTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ComplexTypeSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeNamedTypeSpecifier(const io::NamedTypeSpecifier* node)
    -> NamedTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) NamedTypeSpecifierAST();
  ast->name = decodeName(node->name(), node->name_type());
  return ast;
}

auto ASTDecoder::decodeAtomicTypeSpecifier(const io::AtomicTypeSpecifier* node)
    -> AtomicTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AtomicTypeSpecifierAST();
  ast->typeId = decodeTypeId(node->type_id());
  return ast;
}

auto ASTDecoder::decodeUnderlyingTypeSpecifier(
    const io::UnderlyingTypeSpecifier* node) -> UnderlyingTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) UnderlyingTypeSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeElaboratedTypeSpecifier(
    const io::ElaboratedTypeSpecifier* node) -> ElaboratedTypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ElaboratedTypeSpecifierAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  return ast;
}

auto ASTDecoder::decodeDecltypeAutoSpecifier(
    const io::DecltypeAutoSpecifier* node) -> DecltypeAutoSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DecltypeAutoSpecifierAST();
  return ast;
}

auto ASTDecoder::decodeDecltypeSpecifier(const io::DecltypeSpecifier* node)
    -> DecltypeSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) DecltypeSpecifierAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
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
  return ast;
}

auto ASTDecoder::decodeVolatileQualifier(const io::VolatileQualifier* node)
    -> VolatileQualifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) VolatileQualifierAST();
  return ast;
}

auto ASTDecoder::decodeRestrictQualifier(const io::RestrictQualifier* node)
    -> RestrictQualifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) RestrictQualifierAST();
  return ast;
}

auto ASTDecoder::decodeEnumSpecifier(const io::EnumSpecifier* node)
    -> EnumSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) EnumSpecifierAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  ast->enumBase = decodeEnumBase(node->enum_base());
  if (node->enumerator_list()) {
    auto* inserter = &ast->enumeratorList;
    for (std::size_t i = 0; i < node->enumerator_list()->size(); ++i) {
      *inserter =
          new (pool_) List(decodeEnumerator(node->enumerator_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeClassSpecifier(const io::ClassSpecifier* node)
    -> ClassSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ClassSpecifierAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  ast->name = decodeName(node->name(), node->name_type());
  ast->baseClause = decodeBaseClause(node->base_clause());
  if (node->declaration_list()) {
    auto* inserter = &ast->declarationList;
    for (std::size_t i = 0; i < node->declaration_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeDeclaration(
          node->declaration_list()->Get(i),
          io::Declaration(node->declaration_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeTypenameSpecifier(const io::TypenameSpecifier* node)
    -> TypenameSpecifierAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) TypenameSpecifierAST();
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  ast->name = decodeName(node->name(), node->name_type());
  return ast;
}

auto ASTDecoder::decodeIdDeclarator(const io::IdDeclarator* node)
    -> IdDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) IdDeclaratorAST();
  ast->name = decodeName(node->name(), node->name_type());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
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
  ast->declarator = decodeDeclarator(node->declarator());
  return ast;
}

auto ASTDecoder::decodePointerOperator(const io::PointerOperator* node)
    -> PointerOperatorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) PointerOperatorAST();
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->cv_qualifier_list()) {
    auto* inserter = &ast->cvQualifierList;
    for (std::size_t i = 0; i < node->cv_qualifier_list()->size(); ++i) {
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
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
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
  ast->nestedNameSpecifier =
      decodeNestedNameSpecifier(node->nested_name_specifier());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  if (node->cv_qualifier_list()) {
    auto* inserter = &ast->cvQualifierList;
    for (std::size_t i = 0; i < node->cv_qualifier_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeSpecifier(
          node->cv_qualifier_list()->Get(i),
          io::Specifier(node->cv_qualifier_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeFunctionDeclarator(const io::FunctionDeclarator* node)
    -> FunctionDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) FunctionDeclaratorAST();
  ast->parametersAndQualifiers =
      decodeParametersAndQualifiers(node->parameters_and_qualifiers());
  ast->trailingReturnType =
      decodeTrailingReturnType(node->trailing_return_type());
  return ast;
}

auto ASTDecoder::decodeArrayDeclarator(const io::ArrayDeclarator* node)
    -> ArrayDeclaratorAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ArrayDeclaratorAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter = new (pool_) List(decodeAttributeSpecifier(
          node->attribute_list()->Get(i),
          io::AttributeSpecifier(node->attribute_list_type()->Get(i))));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeCxxAttribute(const io::CxxAttribute* node)
    -> CxxAttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) CxxAttributeAST();
  ast->attributeUsingPrefix =
      decodeAttributeUsingPrefix(node->attribute_using_prefix());
  if (node->attribute_list()) {
    auto* inserter = &ast->attributeList;
    for (std::size_t i = 0; i < node->attribute_list()->size(); ++i) {
      *inserter =
          new (pool_) List(decodeAttribute(node->attribute_list()->Get(i)));
      inserter = &(*inserter)->next;
    }
  }
  return ast;
}

auto ASTDecoder::decodeGCCAttribute(const io::GCCAttribute* node)
    -> GCCAttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) GCCAttributeAST();
  return ast;
}

auto ASTDecoder::decodeAlignasAttribute(const io::AlignasAttribute* node)
    -> AlignasAttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AlignasAttributeAST();
  ast->expression =
      decodeExpression(node->expression(), node->expression_type());
  return ast;
}

auto ASTDecoder::decodeAsmAttribute(const io::AsmAttribute* node)
    -> AsmAttributeAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) AsmAttributeAST();
  return ast;
}

auto ASTDecoder::decodeScopedAttributeToken(
    const io::ScopedAttributeToken* node) -> ScopedAttributeTokenAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) ScopedAttributeTokenAST();
  return ast;
}

auto ASTDecoder::decodeSimpleAttributeToken(
    const io::SimpleAttributeToken* node) -> SimpleAttributeTokenAST* {
  if (!node) return nullptr;

  auto ast = new (pool_) SimpleAttributeTokenAST();
  return ast;
}

}  // namespace cxx
