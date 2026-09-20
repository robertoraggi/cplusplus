// Generated file by: gen_semantic_codec.ts
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

#pragma once

#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/const_value.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/semantic_archive.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <format>
#include <memory>
#include <ranges>
#include <string>
#include <type_traits>
#include <vector>

namespace cxx {

class SemanticEncoder final : public SemanticEncoderBase {
 public:
  explicit SemanticEncoder(TranslationUnit* unit) : SemanticEncoderBase(unit) {}

  [[nodiscard]] auto operator()(const SemanticArchiveRoots& roots,
                                ArchiveWriter& archive) -> bool;

 private:
  [[nodiscard]] auto nameRef(const cxx::Name* name) -> NameRef {
    return NameRef{names_.reference(name)};
  }

  [[nodiscard]] auto typeRef(const cxx::Type* type) -> TypeRef {
    return TypeRef{types_.reference(type)};
  }

  [[nodiscard]] auto symbolRef(const cxx::Symbol* symbol) -> SymbolRef {
    return SymbolRef{symbols_.reference(const_cast<cxx::Symbol*>(symbol))};
  }

  [[nodiscard]] auto astRef(const cxx::AST* ast) -> AstRef {
    return AstRef{nodes_.reference(const_cast<cxx::AST*>(ast))};
  }

  [[nodiscard]] auto identifierRef(const cxx::Identifier* identifier)
      -> StringRef {
    if (!identifier) return StringRef{0};
    return stringRef(identifier->name());
  }

  [[nodiscard]] auto constRef(const std::shared_ptr<cxx::Meta>& value)
      -> ConstRef {
    return ConstRef{constNodeRef(value, 0)};
  }

  [[nodiscard]] auto constRef(
      const std::shared_ptr<cxx::InitializerList>& value) -> ConstRef {
    return ConstRef{constNodeRef(value, 1)};
  }

  [[nodiscard]] auto constRef(const std::shared_ptr<cxx::ConstObject>& value)
      -> ConstRef {
    return ConstRef{constNodeRef(value, 2)};
  }

  [[nodiscard]] auto constRef(const std::shared_ptr<cxx::ConstAddress>& value)
      -> ConstRef {
    return ConstRef{constNodeRef(value, 3)};
  }

  [[nodiscard]] auto constRef(
      const std::shared_ptr<cxx::ConstLabelAddress>& value) -> ConstRef {
    return ConstRef{constNodeRef(value, 4)};
  }

  [[nodiscard]] auto constRef(const std::shared_ptr<cxx::ConstComplex>& value)
      -> ConstRef {
    return ConstRef{constNodeRef(value, 5)};
  }

  template <typename T>
  void writeAstList(ByteWriter& out, cxx::List<T*>* list) {
    std::uint32_t count = 0;
    for (auto node : cxx::ListView{list}) {
      (void)node;
      ++count;
    }
    out.varU32(count);
    for (auto node : cxx::ListView{list})
      out.varU32(static_cast<std::uint32_t>(astRef(node)));
  }

  void writeLiteral(ByteWriter& out, const cxx::Literal* literal);
  void writeAbiTags(ByteWriter& out,
                    const std::vector<const cxx::Identifier*>* tags);
  void writeAttributes(ByteWriter& out, const cxx::AttributeMap* attributes);
  void writeConstValue(ByteWriter& out, const cxx::ConstValue& value);
  void writeTemplateArgument(ByteWriter& out,
                             const cxx::TemplateArgument& argument);

  void writeName(ByteWriter& out, const cxx::Name* name);
  void writeType(ByteWriter& out, const cxx::Type* type);
  void writeSymbol(ByteWriter& out, cxx::Symbol* symbol);
  void writeAst(ByteWriter& out, cxx::AST* ast);
  void writeConstNode(ByteWriter& out, const ConstNode& node);

  void drain();

  void writeNameIdentifier(ByteWriter& out, const cxx::Identifier* self);
  void writeNameOperatorId(ByteWriter& out, const cxx::OperatorId* self);
  void writeNameDestructorId(ByteWriter& out, const cxx::DestructorId* self);
  void writeNameLiteralOperatorId(ByteWriter& out,
                                  const cxx::LiteralOperatorId* self);
  void writeNameConversionFunctionId(ByteWriter& out,
                                     const cxx::ConversionFunctionId* self);
  void writeNameTemplateId(ByteWriter& out, const cxx::TemplateId* self);
  void writeTypeVoidType(ByteWriter& out, const cxx::VoidType* self);
  void writeTypeNullptrType(ByteWriter& out, const cxx::NullptrType* self);
  void writeTypeDecltypeAutoType(ByteWriter& out,
                                 const cxx::DecltypeAutoType* self);
  void writeTypeAutoType(ByteWriter& out, const cxx::AutoType* self);
  void writeTypeBoolType(ByteWriter& out, const cxx::BoolType* self);
  void writeTypeSignedCharType(ByteWriter& out,
                               const cxx::SignedCharType* self);
  void writeTypeShortIntType(ByteWriter& out, const cxx::ShortIntType* self);
  void writeTypeIntType(ByteWriter& out, const cxx::IntType* self);
  void writeTypeLongIntType(ByteWriter& out, const cxx::LongIntType* self);
  void writeTypeLongLongIntType(ByteWriter& out,
                                const cxx::LongLongIntType* self);
  void writeTypeInt128Type(ByteWriter& out, const cxx::Int128Type* self);
  void writeTypeUnsignedCharType(ByteWriter& out,
                                 const cxx::UnsignedCharType* self);
  void writeTypeUnsignedShortIntType(ByteWriter& out,
                                     const cxx::UnsignedShortIntType* self);
  void writeTypeUnsignedIntType(ByteWriter& out,
                                const cxx::UnsignedIntType* self);
  void writeTypeUnsignedLongIntType(ByteWriter& out,
                                    const cxx::UnsignedLongIntType* self);
  void writeTypeUnsignedLongLongIntType(
      ByteWriter& out, const cxx::UnsignedLongLongIntType* self);
  void writeTypeUnsignedInt128Type(ByteWriter& out,
                                   const cxx::UnsignedInt128Type* self);
  void writeTypeCharType(ByteWriter& out, const cxx::CharType* self);
  void writeTypeChar8Type(ByteWriter& out, const cxx::Char8Type* self);
  void writeTypeChar16Type(ByteWriter& out, const cxx::Char16Type* self);
  void writeTypeChar32Type(ByteWriter& out, const cxx::Char32Type* self);
  void writeTypeWideCharType(ByteWriter& out, const cxx::WideCharType* self);
  void writeTypeFloatType(ByteWriter& out, const cxx::FloatType* self);
  void writeTypeDoubleType(ByteWriter& out, const cxx::DoubleType* self);
  void writeTypeLongDoubleType(ByteWriter& out,
                               const cxx::LongDoubleType* self);
  void writeTypeFloat16Type(ByteWriter& out, const cxx::Float16Type* self);
  void writeTypeQualType(ByteWriter& out, const cxx::QualType* self);
  void writeTypeBoundedArrayType(ByteWriter& out,
                                 const cxx::BoundedArrayType* self);
  void writeTypeUnboundedArrayType(ByteWriter& out,
                                   const cxx::UnboundedArrayType* self);
  void writeTypePointerType(ByteWriter& out, const cxx::PointerType* self);
  void writeTypeLvalueReferenceType(ByteWriter& out,
                                    const cxx::LvalueReferenceType* self);
  void writeTypeRvalueReferenceType(ByteWriter& out,
                                    const cxx::RvalueReferenceType* self);
  void writeTypeFunctionType(ByteWriter& out, const cxx::FunctionType* self);
  void writeTypeClassType(ByteWriter& out, const cxx::ClassType* self);
  void writeTypeEnumType(ByteWriter& out, const cxx::EnumType* self);
  void writeTypeScopedEnumType(ByteWriter& out,
                               const cxx::ScopedEnumType* self);
  void writeTypeMemberObjectPointerType(
      ByteWriter& out, const cxx::MemberObjectPointerType* self);
  void writeTypeMemberFunctionPointerType(
      ByteWriter& out, const cxx::MemberFunctionPointerType* self);
  void writeTypeNamespaceType(ByteWriter& out, const cxx::NamespaceType* self);
  void writeTypeTypeParameterType(ByteWriter& out,
                                  const cxx::TypeParameterType* self);
  void writeTypeTemplateTypeParameterType(
      ByteWriter& out, const cxx::TemplateTypeParameterType* self);
  void writeTypeUnresolvedNameType(ByteWriter& out,
                                   const cxx::UnresolvedNameType* self);
  void writeTypeUnresolvedBoundedArrayType(
      ByteWriter& out, const cxx::UnresolvedBoundedArrayType* self);
  void writeTypeUnresolvedUnderlyingType(
      ByteWriter& out, const cxx::UnresolvedUnderlyingType* self);
  void writeTypeUnresolvedBuiltinType(ByteWriter& out,
                                      const cxx::UnresolvedBuiltinType* self);
  void writeTypeOverloadSetType(ByteWriter& out,
                                const cxx::OverloadSetType* self);
  void writeTypeBuiltinVaListType(ByteWriter& out,
                                  const cxx::BuiltinVaListType* self);
  void writeTypeBuiltinMetaInfoType(ByteWriter& out,
                                    const cxx::BuiltinMetaInfoType* self);
  void writeTypeBitIntType(ByteWriter& out, const cxx::BitIntType* self);
  void writeTypeUnsignedBitIntType(ByteWriter& out,
                                   const cxx::UnsignedBitIntType* self);
  void writeTypeUnresolvedBitIntType(ByteWriter& out,
                                     const cxx::UnresolvedBitIntType* self);
  void writeTypeVectorType(ByteWriter& out, const cxx::VectorType* self);
  void writeTypeUnresolvedVectorType(ByteWriter& out,
                                     const cxx::UnresolvedVectorType* self);
  void writeTypeComplexType(ByteWriter& out, const cxx::ComplexType* self);
  void writeTypeAtomicType(ByteWriter& out, const cxx::AtomicType* self);
  void writeSymbolSymbol(ByteWriter& out, cxx::Symbol* self);
  void writeSymbolScopeSymbol(ByteWriter& out, cxx::ScopeSymbol* self);
  void writeSymbolNamespaceSymbol(ByteWriter& out, cxx::NamespaceSymbol* self);
  void writeSymbolNamespaceAliasSymbol(ByteWriter& out,
                                       cxx::NamespaceAliasSymbol* self);
  void writeSymbolConceptSymbol(ByteWriter& out, cxx::ConceptSymbol* self);
  void writeSymbolDeductionGuideSymbol(ByteWriter& out,
                                       cxx::DeductionGuideSymbol* self);
  void writeSymbolClassSymbol(ByteWriter& out, cxx::ClassSymbol* self);
  void writeSymbolEnumSymbol(ByteWriter& out, cxx::EnumSymbol* self);
  void writeSymbolScopedEnumSymbol(ByteWriter& out,
                                   cxx::ScopedEnumSymbol* self);
  void writeSymbolFunctionSymbol(ByteWriter& out, cxx::FunctionSymbol* self);
  void writeSymbolTypeAliasSymbol(ByteWriter& out, cxx::TypeAliasSymbol* self);
  void writeSymbolVariableSymbol(ByteWriter& out, cxx::VariableSymbol* self);
  void writeSymbolFieldSymbol(ByteWriter& out, cxx::FieldSymbol* self);
  void writeSymbolParameterSymbol(ByteWriter& out, cxx::ParameterSymbol* self);
  void writeSymbolParameterPackSymbol(ByteWriter& out,
                                      cxx::ParameterPackSymbol* self);
  void writeSymbolEnumeratorSymbol(ByteWriter& out,
                                   cxx::EnumeratorSymbol* self);
  void writeSymbolFunctionParametersSymbol(ByteWriter& out,
                                           cxx::FunctionParametersSymbol* self);
  void writeSymbolTemplateParametersSymbol(ByteWriter& out,
                                           cxx::TemplateParametersSymbol* self);
  void writeSymbolBlockSymbol(ByteWriter& out, cxx::BlockSymbol* self);
  void writeSymbolLambdaSymbol(ByteWriter& out, cxx::LambdaSymbol* self);
  void writeSymbolTypeParameterSymbol(ByteWriter& out,
                                      cxx::TypeParameterSymbol* self);
  void writeSymbolNonTypeParameterSymbol(ByteWriter& out,
                                         cxx::NonTypeParameterSymbol* self);
  void writeSymbolTemplateTypeParameterSymbol(
      ByteWriter& out, cxx::TemplateTypeParameterSymbol* self);
  void writeSymbolConstraintTypeParameterSymbol(
      ByteWriter& out, cxx::ConstraintTypeParameterSymbol* self);
  void writeSymbolOverloadSetSymbol(ByteWriter& out,
                                    cxx::OverloadSetSymbol* self);
  void writeSymbolBaseClassSymbol(ByteWriter& out, cxx::BaseClassSymbol* self);
  void writeSymbolInjectedClassNameSymbol(ByteWriter& out,
                                          cxx::InjectedClassNameSymbol* self);
  void writeSymbolUnresolvedSymbol(ByteWriter& out,
                                   cxx::UnresolvedSymbol* self);
  void writeSymbolUsingDeclarationSymbol(ByteWriter& out,
                                         cxx::UsingDeclarationSymbol* self);
  void writeAstManaged(ByteWriter& out, cxx::Managed* self);
  void writeAstAST(ByteWriter& out, cxx::AST* self);
  void writeAstUnitAST(ByteWriter& out, cxx::UnitAST* self);
  void writeAstDeclarationAST(ByteWriter& out, cxx::DeclarationAST* self);
  void writeAstStatementAST(ByteWriter& out, cxx::StatementAST* self);
  void writeAstExpressionAST(ByteWriter& out, cxx::ExpressionAST* self);
  void writeAstGenericAssociationAST(ByteWriter& out,
                                     cxx::GenericAssociationAST* self);
  void writeAstDesignatorAST(ByteWriter& out, cxx::DesignatorAST* self);
  void writeAstTemplateParameterAST(ByteWriter& out,
                                    cxx::TemplateParameterAST* self);
  void writeAstSpecifierAST(ByteWriter& out, cxx::SpecifierAST* self);
  void writeAstPtrOperatorAST(ByteWriter& out, cxx::PtrOperatorAST* self);
  void writeAstCoreDeclaratorAST(ByteWriter& out, cxx::CoreDeclaratorAST* self);
  void writeAstDeclaratorChunkAST(ByteWriter& out,
                                  cxx::DeclaratorChunkAST* self);
  void writeAstUnqualifiedIdAST(ByteWriter& out, cxx::UnqualifiedIdAST* self);
  void writeAstNestedNameSpecifierAST(ByteWriter& out,
                                      cxx::NestedNameSpecifierAST* self);
  void writeAstFunctionBodyAST(ByteWriter& out, cxx::FunctionBodyAST* self);
  void writeAstTemplateArgumentAST(ByteWriter& out,
                                   cxx::TemplateArgumentAST* self);
  void writeAstExceptionSpecifierAST(ByteWriter& out,
                                     cxx::ExceptionSpecifierAST* self);
  void writeAstRequirementAST(ByteWriter& out, cxx::RequirementAST* self);
  void writeAstNewInitializerAST(ByteWriter& out, cxx::NewInitializerAST* self);
  void writeAstMemInitializerAST(ByteWriter& out, cxx::MemInitializerAST* self);
  void writeAstLambdaCaptureAST(ByteWriter& out, cxx::LambdaCaptureAST* self);
  void writeAstExceptionDeclarationAST(ByteWriter& out,
                                       cxx::ExceptionDeclarationAST* self);
  void writeAstAttributeSpecifierAST(ByteWriter& out,
                                     cxx::AttributeSpecifierAST* self);
  void writeAstAttributeTokenAST(ByteWriter& out, cxx::AttributeTokenAST* self);
  void writeAstTranslationUnitAST(ByteWriter& out,
                                  cxx::TranslationUnitAST* self);
  void writeAstModuleUnitAST(ByteWriter& out, cxx::ModuleUnitAST* self);
  void writeAstSimpleDeclarationAST(ByteWriter& out,
                                    cxx::SimpleDeclarationAST* self);
  void writeAstAsmDeclarationAST(ByteWriter& out, cxx::AsmDeclarationAST* self);
  void writeAstNamespaceAliasDefinitionAST(
      ByteWriter& out, cxx::NamespaceAliasDefinitionAST* self);
  void writeAstUsingDeclarationAST(ByteWriter& out,
                                   cxx::UsingDeclarationAST* self);
  void writeAstUsingEnumDeclarationAST(ByteWriter& out,
                                       cxx::UsingEnumDeclarationAST* self);
  void writeAstUsingDirectiveAST(ByteWriter& out, cxx::UsingDirectiveAST* self);
  void writeAstStaticAssertDeclarationAST(
      ByteWriter& out, cxx::StaticAssertDeclarationAST* self);
  void writeAstAliasDeclarationAST(ByteWriter& out,
                                   cxx::AliasDeclarationAST* self);
  void writeAstOpaqueEnumDeclarationAST(ByteWriter& out,
                                        cxx::OpaqueEnumDeclarationAST* self);
  void writeAstFunctionDefinitionAST(ByteWriter& out,
                                     cxx::FunctionDefinitionAST* self);
  void writeAstTemplateDeclarationAST(ByteWriter& out,
                                      cxx::TemplateDeclarationAST* self);
  void writeAstConceptDefinitionAST(ByteWriter& out,
                                    cxx::ConceptDefinitionAST* self);
  void writeAstDeductionGuideAST(ByteWriter& out, cxx::DeductionGuideAST* self);
  void writeAstExplicitInstantiationAST(ByteWriter& out,
                                        cxx::ExplicitInstantiationAST* self);
  void writeAstExportDeclarationAST(ByteWriter& out,
                                    cxx::ExportDeclarationAST* self);
  void writeAstExportCompoundDeclarationAST(
      ByteWriter& out, cxx::ExportCompoundDeclarationAST* self);
  void writeAstLinkageSpecificationAST(ByteWriter& out,
                                       cxx::LinkageSpecificationAST* self);
  void writeAstNamespaceDefinitionAST(ByteWriter& out,
                                      cxx::NamespaceDefinitionAST* self);
  void writeAstEmptyDeclarationAST(ByteWriter& out,
                                   cxx::EmptyDeclarationAST* self);
  void writeAstAttributeDeclarationAST(ByteWriter& out,
                                       cxx::AttributeDeclarationAST* self);
  void writeAstModuleImportDeclarationAST(
      ByteWriter& out, cxx::ModuleImportDeclarationAST* self);
  void writeAstParameterDeclarationAST(ByteWriter& out,
                                       cxx::ParameterDeclarationAST* self);
  void writeAstAccessDeclarationAST(ByteWriter& out,
                                    cxx::AccessDeclarationAST* self);
  void writeAstForRangeDeclarationAST(ByteWriter& out,
                                      cxx::ForRangeDeclarationAST* self);
  void writeAstStructuredBindingDeclarationAST(
      ByteWriter& out, cxx::StructuredBindingDeclarationAST* self);
  void writeAstAsmOperandAST(ByteWriter& out, cxx::AsmOperandAST* self);
  void writeAstAsmQualifierAST(ByteWriter& out, cxx::AsmQualifierAST* self);
  void writeAstAsmClobberAST(ByteWriter& out, cxx::AsmClobberAST* self);
  void writeAstAsmGotoLabelAST(ByteWriter& out, cxx::AsmGotoLabelAST* self);
  void writeAstSplicerAST(ByteWriter& out, cxx::SplicerAST* self);
  void writeAstGlobalModuleFragmentAST(ByteWriter& out,
                                       cxx::GlobalModuleFragmentAST* self);
  void writeAstPrivateModuleFragmentAST(ByteWriter& out,
                                        cxx::PrivateModuleFragmentAST* self);
  void writeAstModuleDeclarationAST(ByteWriter& out,
                                    cxx::ModuleDeclarationAST* self);
  void writeAstModuleNameAST(ByteWriter& out, cxx::ModuleNameAST* self);
  void writeAstModuleQualifierAST(ByteWriter& out,
                                  cxx::ModuleQualifierAST* self);
  void writeAstModulePartitionAST(ByteWriter& out,
                                  cxx::ModulePartitionAST* self);
  void writeAstImportNameAST(ByteWriter& out, cxx::ImportNameAST* self);
  void writeAstInitDeclaratorAST(ByteWriter& out, cxx::InitDeclaratorAST* self);
  void writeAstDeclaratorAST(ByteWriter& out, cxx::DeclaratorAST* self);
  void writeAstUsingDeclaratorAST(ByteWriter& out,
                                  cxx::UsingDeclaratorAST* self);
  void writeAstEnumeratorAST(ByteWriter& out, cxx::EnumeratorAST* self);
  void writeAstTypeIdAST(ByteWriter& out, cxx::TypeIdAST* self);
  void writeAstHandlerAST(ByteWriter& out, cxx::HandlerAST* self);
  void writeAstBaseSpecifierAST(ByteWriter& out, cxx::BaseSpecifierAST* self);
  void writeAstRequiresClauseAST(ByteWriter& out, cxx::RequiresClauseAST* self);
  void writeAstParameterDeclarationClauseAST(
      ByteWriter& out, cxx::ParameterDeclarationClauseAST* self);
  void writeAstTrailingReturnTypeAST(ByteWriter& out,
                                     cxx::TrailingReturnTypeAST* self);
  void writeAstLambdaSpecifierAST(ByteWriter& out,
                                  cxx::LambdaSpecifierAST* self);
  void writeAstTypeConstraintAST(ByteWriter& out, cxx::TypeConstraintAST* self);
  void writeAstAttributeArgumentClauseAST(
      ByteWriter& out, cxx::AttributeArgumentClauseAST* self);
  void writeAstAttributeAST(ByteWriter& out, cxx::AttributeAST* self);
  void writeAstAttributeUsingPrefixAST(ByteWriter& out,
                                       cxx::AttributeUsingPrefixAST* self);
  void writeAstNewPlacementAST(ByteWriter& out, cxx::NewPlacementAST* self);
  void writeAstNestedNamespaceSpecifierAST(
      ByteWriter& out, cxx::NestedNamespaceSpecifierAST* self);
  void writeAstLabeledStatementAST(ByteWriter& out,
                                   cxx::LabeledStatementAST* self);
  void writeAstCaseStatementAST(ByteWriter& out, cxx::CaseStatementAST* self);
  void writeAstDefaultStatementAST(ByteWriter& out,
                                   cxx::DefaultStatementAST* self);
  void writeAstExpressionStatementAST(ByteWriter& out,
                                      cxx::ExpressionStatementAST* self);
  void writeAstCompoundStatementAST(ByteWriter& out,
                                    cxx::CompoundStatementAST* self);
  void writeAstIfStatementAST(ByteWriter& out, cxx::IfStatementAST* self);
  void writeAstConstevalIfStatementAST(ByteWriter& out,
                                       cxx::ConstevalIfStatementAST* self);
  void writeAstSwitchStatementAST(ByteWriter& out,
                                  cxx::SwitchStatementAST* self);
  void writeAstWhileStatementAST(ByteWriter& out, cxx::WhileStatementAST* self);
  void writeAstDoStatementAST(ByteWriter& out, cxx::DoStatementAST* self);
  void writeAstForRangeStatementAST(ByteWriter& out,
                                    cxx::ForRangeStatementAST* self);
  void writeAstForStatementAST(ByteWriter& out, cxx::ForStatementAST* self);
  void writeAstBreakStatementAST(ByteWriter& out, cxx::BreakStatementAST* self);
  void writeAstContinueStatementAST(ByteWriter& out,
                                    cxx::ContinueStatementAST* self);
  void writeAstReturnStatementAST(ByteWriter& out,
                                  cxx::ReturnStatementAST* self);
  void writeAstCoroutineReturnStatementAST(
      ByteWriter& out, cxx::CoroutineReturnStatementAST* self);
  void writeAstGotoStatementAST(ByteWriter& out, cxx::GotoStatementAST* self);
  void writeAstDeclarationStatementAST(ByteWriter& out,
                                       cxx::DeclarationStatementAST* self);
  void writeAstTryBlockStatementAST(ByteWriter& out,
                                    cxx::TryBlockStatementAST* self);
  void writeAstCharLiteralExpressionAST(ByteWriter& out,
                                        cxx::CharLiteralExpressionAST* self);
  void writeAstBoolLiteralExpressionAST(ByteWriter& out,
                                        cxx::BoolLiteralExpressionAST* self);
  void writeAstIntLiteralExpressionAST(ByteWriter& out,
                                       cxx::IntLiteralExpressionAST* self);
  void writeAstFloatLiteralExpressionAST(ByteWriter& out,
                                         cxx::FloatLiteralExpressionAST* self);
  void writeAstNullptrLiteralExpressionAST(
      ByteWriter& out, cxx::NullptrLiteralExpressionAST* self);
  void writeAstStringLiteralExpressionAST(
      ByteWriter& out, cxx::StringLiteralExpressionAST* self);
  void writeAstUserDefinedStringLiteralExpressionAST(
      ByteWriter& out, cxx::UserDefinedStringLiteralExpressionAST* self);
  void writeAstObjectLiteralExpressionAST(
      ByteWriter& out, cxx::ObjectLiteralExpressionAST* self);
  void writeAstThisExpressionAST(ByteWriter& out, cxx::ThisExpressionAST* self);
  void writeAstPackIndexExpressionAST(ByteWriter& out,
                                      cxx::PackIndexExpressionAST* self);
  void writeAstGenericSelectionExpressionAST(
      ByteWriter& out, cxx::GenericSelectionExpressionAST* self);
  void writeAstNestedStatementExpressionAST(
      ByteWriter& out, cxx::NestedStatementExpressionAST* self);
  void writeAstDefaultInitializerExpressionAST(
      ByteWriter& out, cxx::DefaultInitializerExpressionAST* self);
  void writeAstNestedExpressionAST(ByteWriter& out,
                                   cxx::NestedExpressionAST* self);
  void writeAstIdExpressionAST(ByteWriter& out, cxx::IdExpressionAST* self);
  void writeAstLambdaExpressionAST(ByteWriter& out,
                                   cxx::LambdaExpressionAST* self);
  void writeAstFoldExpressionAST(ByteWriter& out, cxx::FoldExpressionAST* self);
  void writeAstRightFoldExpressionAST(ByteWriter& out,
                                      cxx::RightFoldExpressionAST* self);
  void writeAstLeftFoldExpressionAST(ByteWriter& out,
                                     cxx::LeftFoldExpressionAST* self);
  void writeAstRequiresExpressionAST(ByteWriter& out,
                                     cxx::RequiresExpressionAST* self);
  void writeAstVaArgExpressionAST(ByteWriter& out,
                                  cxx::VaArgExpressionAST* self);
  void writeAstSubscriptExpressionAST(ByteWriter& out,
                                      cxx::SubscriptExpressionAST* self);
  void writeAstCallExpressionAST(ByteWriter& out, cxx::CallExpressionAST* self);
  void writeAstTypeConstructionAST(ByteWriter& out,
                                   cxx::TypeConstructionAST* self);
  void writeAstBracedTypeConstructionAST(ByteWriter& out,
                                         cxx::BracedTypeConstructionAST* self);
  void writeAstSpliceMemberExpressionAST(ByteWriter& out,
                                         cxx::SpliceMemberExpressionAST* self);
  void writeAstMemberExpressionAST(ByteWriter& out,
                                   cxx::MemberExpressionAST* self);
  void writeAstPostIncrExpressionAST(ByteWriter& out,
                                     cxx::PostIncrExpressionAST* self);
  void writeAstCppCastExpressionAST(ByteWriter& out,
                                    cxx::CppCastExpressionAST* self);
  void writeAstBuiltinBitCastExpressionAST(
      ByteWriter& out, cxx::BuiltinBitCastExpressionAST* self);
  void writeAstBuiltinOffsetofExpressionAST(
      ByteWriter& out, cxx::BuiltinOffsetofExpressionAST* self);
  void writeAstTypeidExpressionAST(ByteWriter& out,
                                   cxx::TypeidExpressionAST* self);
  void writeAstTypeidOfTypeExpressionAST(ByteWriter& out,
                                         cxx::TypeidOfTypeExpressionAST* self);
  void writeAstSpliceExpressionAST(ByteWriter& out,
                                   cxx::SpliceExpressionAST* self);
  void writeAstGlobalScopeReflectExpressionAST(
      ByteWriter& out, cxx::GlobalScopeReflectExpressionAST* self);
  void writeAstNamespaceReflectExpressionAST(
      ByteWriter& out, cxx::NamespaceReflectExpressionAST* self);
  void writeAstTypeIdReflectExpressionAST(
      ByteWriter& out, cxx::TypeIdReflectExpressionAST* self);
  void writeAstReflectExpressionAST(ByteWriter& out,
                                    cxx::ReflectExpressionAST* self);
  void writeAstLabelAddressExpressionAST(ByteWriter& out,
                                         cxx::LabelAddressExpressionAST* self);
  void writeAstUnaryExpressionAST(ByteWriter& out,
                                  cxx::UnaryExpressionAST* self);
  void writeAstAwaitExpressionAST(ByteWriter& out,
                                  cxx::AwaitExpressionAST* self);
  void writeAstSizeofExpressionAST(ByteWriter& out,
                                   cxx::SizeofExpressionAST* self);
  void writeAstSizeofTypeExpressionAST(ByteWriter& out,
                                       cxx::SizeofTypeExpressionAST* self);
  void writeAstSizeofPackExpressionAST(ByteWriter& out,
                                       cxx::SizeofPackExpressionAST* self);
  void writeAstAlignofTypeExpressionAST(ByteWriter& out,
                                        cxx::AlignofTypeExpressionAST* self);
  void writeAstAlignofExpressionAST(ByteWriter& out,
                                    cxx::AlignofExpressionAST* self);
  void writeAstNoexceptExpressionAST(ByteWriter& out,
                                     cxx::NoexceptExpressionAST* self);
  void writeAstNewExpressionAST(ByteWriter& out, cxx::NewExpressionAST* self);
  void writeAstDeleteExpressionAST(ByteWriter& out,
                                   cxx::DeleteExpressionAST* self);
  void writeAstCastExpressionAST(ByteWriter& out, cxx::CastExpressionAST* self);
  void writeAstImplicitCastExpressionAST(ByteWriter& out,
                                         cxx::ImplicitCastExpressionAST* self);
  void writeAstConstExpressionAST(ByteWriter& out,
                                  cxx::ConstExpressionAST* self);
  void writeAstBinaryExpressionAST(ByteWriter& out,
                                   cxx::BinaryExpressionAST* self);
  void writeAstConditionalExpressionAST(ByteWriter& out,
                                        cxx::ConditionalExpressionAST* self);
  void writeAstYieldExpressionAST(ByteWriter& out,
                                  cxx::YieldExpressionAST* self);
  void writeAstThrowExpressionAST(ByteWriter& out,
                                  cxx::ThrowExpressionAST* self);
  void writeAstAssignmentExpressionAST(ByteWriter& out,
                                       cxx::AssignmentExpressionAST* self);
  void writeAstTargetExpressionAST(ByteWriter& out,
                                   cxx::TargetExpressionAST* self);
  void writeAstRightExpressionAST(ByteWriter& out,
                                  cxx::RightExpressionAST* self);
  void writeAstCompoundAssignmentExpressionAST(
      ByteWriter& out, cxx::CompoundAssignmentExpressionAST* self);
  void writeAstPackExpansionExpressionAST(
      ByteWriter& out, cxx::PackExpansionExpressionAST* self);
  void writeAstDesignatedInitializerClauseAST(
      ByteWriter& out, cxx::DesignatedInitializerClauseAST* self);
  void writeAstTypeTraitExpressionAST(ByteWriter& out,
                                      cxx::TypeTraitExpressionAST* self);
  void writeAstConditionExpressionAST(ByteWriter& out,
                                      cxx::ConditionExpressionAST* self);
  void writeAstEqualInitializerAST(ByteWriter& out,
                                   cxx::EqualInitializerAST* self);
  void writeAstBracedInitListAST(ByteWriter& out, cxx::BracedInitListAST* self);
  void writeAstParenInitializerAST(ByteWriter& out,
                                   cxx::ParenInitializerAST* self);
  void writeAstThreeWayComparisonExpressionAST(
      ByteWriter& out, cxx::ThreeWayComparisonExpressionAST* self);
  void writeAstDefaultGenericAssociationAST(
      ByteWriter& out, cxx::DefaultGenericAssociationAST* self);
  void writeAstTypeGenericAssociationAST(ByteWriter& out,
                                         cxx::TypeGenericAssociationAST* self);
  void writeAstDotDesignatorAST(ByteWriter& out, cxx::DotDesignatorAST* self);
  void writeAstSubscriptDesignatorAST(ByteWriter& out,
                                      cxx::SubscriptDesignatorAST* self);
  void writeAstTemplateTypeParameterAST(ByteWriter& out,
                                        cxx::TemplateTypeParameterAST* self);
  void writeAstNonTypeTemplateParameterAST(
      ByteWriter& out, cxx::NonTypeTemplateParameterAST* self);
  void writeAstTypenameTypeParameterAST(ByteWriter& out,
                                        cxx::TypenameTypeParameterAST* self);
  void writeAstConstraintTypeParameterAST(
      ByteWriter& out, cxx::ConstraintTypeParameterAST* self);
  void writeAstTypedefSpecifierAST(ByteWriter& out,
                                   cxx::TypedefSpecifierAST* self);
  void writeAstFriendSpecifierAST(ByteWriter& out,
                                  cxx::FriendSpecifierAST* self);
  void writeAstConstevalSpecifierAST(ByteWriter& out,
                                     cxx::ConstevalSpecifierAST* self);
  void writeAstConstinitSpecifierAST(ByteWriter& out,
                                     cxx::ConstinitSpecifierAST* self);
  void writeAstConstexprSpecifierAST(ByteWriter& out,
                                     cxx::ConstexprSpecifierAST* self);
  void writeAstInlineSpecifierAST(ByteWriter& out,
                                  cxx::InlineSpecifierAST* self);
  void writeAstNoreturnSpecifierAST(ByteWriter& out,
                                    cxx::NoreturnSpecifierAST* self);
  void writeAstStaticSpecifierAST(ByteWriter& out,
                                  cxx::StaticSpecifierAST* self);
  void writeAstExternSpecifierAST(ByteWriter& out,
                                  cxx::ExternSpecifierAST* self);
  void writeAstRegisterSpecifierAST(ByteWriter& out,
                                    cxx::RegisterSpecifierAST* self);
  void writeAstThreadLocalSpecifierAST(ByteWriter& out,
                                       cxx::ThreadLocalSpecifierAST* self);
  void writeAstThreadSpecifierAST(ByteWriter& out,
                                  cxx::ThreadSpecifierAST* self);
  void writeAstMutableSpecifierAST(ByteWriter& out,
                                   cxx::MutableSpecifierAST* self);
  void writeAstVirtualSpecifierAST(ByteWriter& out,
                                   cxx::VirtualSpecifierAST* self);
  void writeAstExplicitSpecifierAST(ByteWriter& out,
                                    cxx::ExplicitSpecifierAST* self);
  void writeAstAutoTypeSpecifierAST(ByteWriter& out,
                                    cxx::AutoTypeSpecifierAST* self);
  void writeAstVoidTypeSpecifierAST(ByteWriter& out,
                                    cxx::VoidTypeSpecifierAST* self);
  void writeAstSizeTypeSpecifierAST(ByteWriter& out,
                                    cxx::SizeTypeSpecifierAST* self);
  void writeAstSignTypeSpecifierAST(ByteWriter& out,
                                    cxx::SignTypeSpecifierAST* self);
  void writeAstBuiltinTypeSpecifierAST(ByteWriter& out,
                                       cxx::BuiltinTypeSpecifierAST* self);
  void writeAstUnaryBuiltinTypeSpecifierAST(
      ByteWriter& out, cxx::UnaryBuiltinTypeSpecifierAST* self);
  void writeAstBinaryBuiltinTypeSpecifierAST(
      ByteWriter& out, cxx::BinaryBuiltinTypeSpecifierAST* self);
  void writeAstIntegralTypeSpecifierAST(ByteWriter& out,
                                        cxx::IntegralTypeSpecifierAST* self);
  void writeAstFloatingPointTypeSpecifierAST(
      ByteWriter& out, cxx::FloatingPointTypeSpecifierAST* self);
  void writeAstComplexTypeSpecifierAST(ByteWriter& out,
                                       cxx::ComplexTypeSpecifierAST* self);
  void writeAstNamedTypeSpecifierAST(ByteWriter& out,
                                     cxx::NamedTypeSpecifierAST* self);
  void writeAstAtomicTypeSpecifierAST(ByteWriter& out,
                                      cxx::AtomicTypeSpecifierAST* self);
  void writeAstBitIntTypeSpecifierAST(ByteWriter& out,
                                      cxx::BitIntTypeSpecifierAST* self);
  void writeAstUnderlyingTypeSpecifierAST(
      ByteWriter& out, cxx::UnderlyingTypeSpecifierAST* self);
  void writeAstElaboratedTypeSpecifierAST(
      ByteWriter& out, cxx::ElaboratedTypeSpecifierAST* self);
  void writeAstDecltypeAutoSpecifierAST(ByteWriter& out,
                                        cxx::DecltypeAutoSpecifierAST* self);
  void writeAstDecltypeSpecifierAST(ByteWriter& out,
                                    cxx::DecltypeSpecifierAST* self);
  void writeAstPlaceholderTypeSpecifierAST(
      ByteWriter& out, cxx::PlaceholderTypeSpecifierAST* self);
  void writeAstConstQualifierAST(ByteWriter& out, cxx::ConstQualifierAST* self);
  void writeAstVolatileQualifierAST(ByteWriter& out,
                                    cxx::VolatileQualifierAST* self);
  void writeAstAtomicQualifierAST(ByteWriter& out,
                                  cxx::AtomicQualifierAST* self);
  void writeAstRestrictQualifierAST(ByteWriter& out,
                                    cxx::RestrictQualifierAST* self);
  void writeAstEnumSpecifierAST(ByteWriter& out, cxx::EnumSpecifierAST* self);
  void writeAstClassSpecifierAST(ByteWriter& out, cxx::ClassSpecifierAST* self);
  void writeAstTypenameSpecifierAST(ByteWriter& out,
                                    cxx::TypenameSpecifierAST* self);
  void writeAstSplicerTypeSpecifierAST(ByteWriter& out,
                                       cxx::SplicerTypeSpecifierAST* self);
  void writeAstPointerOperatorAST(ByteWriter& out,
                                  cxx::PointerOperatorAST* self);
  void writeAstReferenceOperatorAST(ByteWriter& out,
                                    cxx::ReferenceOperatorAST* self);
  void writeAstPtrToMemberOperatorAST(ByteWriter& out,
                                      cxx::PtrToMemberOperatorAST* self);
  void writeAstBitfieldDeclaratorAST(ByteWriter& out,
                                     cxx::BitfieldDeclaratorAST* self);
  void writeAstParameterPackAST(ByteWriter& out, cxx::ParameterPackAST* self);
  void writeAstIdDeclaratorAST(ByteWriter& out, cxx::IdDeclaratorAST* self);
  void writeAstNestedDeclaratorAST(ByteWriter& out,
                                   cxx::NestedDeclaratorAST* self);
  void writeAstFunctionDeclaratorChunkAST(
      ByteWriter& out, cxx::FunctionDeclaratorChunkAST* self);
  void writeAstArrayDeclaratorChunkAST(ByteWriter& out,
                                       cxx::ArrayDeclaratorChunkAST* self);
  void writeAstNameIdAST(ByteWriter& out, cxx::NameIdAST* self);
  void writeAstDestructorIdAST(ByteWriter& out, cxx::DestructorIdAST* self);
  void writeAstDecltypeIdAST(ByteWriter& out, cxx::DecltypeIdAST* self);
  void writeAstOperatorFunctionIdAST(ByteWriter& out,
                                     cxx::OperatorFunctionIdAST* self);
  void writeAstLiteralOperatorIdAST(ByteWriter& out,
                                    cxx::LiteralOperatorIdAST* self);
  void writeAstConversionFunctionIdAST(ByteWriter& out,
                                       cxx::ConversionFunctionIdAST* self);
  void writeAstSimpleTemplateIdAST(ByteWriter& out,
                                   cxx::SimpleTemplateIdAST* self);
  void writeAstLiteralOperatorTemplateIdAST(
      ByteWriter& out, cxx::LiteralOperatorTemplateIdAST* self);
  void writeAstOperatorFunctionTemplateIdAST(
      ByteWriter& out, cxx::OperatorFunctionTemplateIdAST* self);
  void writeAstGlobalNestedNameSpecifierAST(
      ByteWriter& out, cxx::GlobalNestedNameSpecifierAST* self);
  void writeAstSimpleNestedNameSpecifierAST(
      ByteWriter& out, cxx::SimpleNestedNameSpecifierAST* self);
  void writeAstDecltypeNestedNameSpecifierAST(
      ByteWriter& out, cxx::DecltypeNestedNameSpecifierAST* self);
  void writeAstTemplateNestedNameSpecifierAST(
      ByteWriter& out, cxx::TemplateNestedNameSpecifierAST* self);
  void writeAstDefaultFunctionBodyAST(ByteWriter& out,
                                      cxx::DefaultFunctionBodyAST* self);
  void writeAstCompoundStatementFunctionBodyAST(
      ByteWriter& out, cxx::CompoundStatementFunctionBodyAST* self);
  void writeAstTryStatementFunctionBodyAST(
      ByteWriter& out, cxx::TryStatementFunctionBodyAST* self);
  void writeAstDeleteFunctionBodyAST(ByteWriter& out,
                                     cxx::DeleteFunctionBodyAST* self);
  void writeAstTypeTemplateArgumentAST(ByteWriter& out,
                                       cxx::TypeTemplateArgumentAST* self);
  void writeAstExpressionTemplateArgumentAST(
      ByteWriter& out, cxx::ExpressionTemplateArgumentAST* self);
  void writeAstThrowExceptionSpecifierAST(
      ByteWriter& out, cxx::ThrowExceptionSpecifierAST* self);
  void writeAstNoexceptSpecifierAST(ByteWriter& out,
                                    cxx::NoexceptSpecifierAST* self);
  void writeAstSimpleRequirementAST(ByteWriter& out,
                                    cxx::SimpleRequirementAST* self);
  void writeAstCompoundRequirementAST(ByteWriter& out,
                                      cxx::CompoundRequirementAST* self);
  void writeAstTypeRequirementAST(ByteWriter& out,
                                  cxx::TypeRequirementAST* self);
  void writeAstNestedRequirementAST(ByteWriter& out,
                                    cxx::NestedRequirementAST* self);
  void writeAstNewParenInitializerAST(ByteWriter& out,
                                      cxx::NewParenInitializerAST* self);
  void writeAstNewBracedInitializerAST(ByteWriter& out,
                                       cxx::NewBracedInitializerAST* self);
  void writeAstParenMemInitializerAST(ByteWriter& out,
                                      cxx::ParenMemInitializerAST* self);
  void writeAstBracedMemInitializerAST(ByteWriter& out,
                                       cxx::BracedMemInitializerAST* self);
  void writeAstThisLambdaCaptureAST(ByteWriter& out,
                                    cxx::ThisLambdaCaptureAST* self);
  void writeAstDerefThisLambdaCaptureAST(ByteWriter& out,
                                         cxx::DerefThisLambdaCaptureAST* self);
  void writeAstSimpleLambdaCaptureAST(ByteWriter& out,
                                      cxx::SimpleLambdaCaptureAST* self);
  void writeAstRefLambdaCaptureAST(ByteWriter& out,
                                   cxx::RefLambdaCaptureAST* self);
  void writeAstRefInitLambdaCaptureAST(ByteWriter& out,
                                       cxx::RefInitLambdaCaptureAST* self);
  void writeAstInitLambdaCaptureAST(ByteWriter& out,
                                    cxx::InitLambdaCaptureAST* self);
  void writeAstEllipsisExceptionDeclarationAST(
      ByteWriter& out, cxx::EllipsisExceptionDeclarationAST* self);
  void writeAstTypeExceptionDeclarationAST(
      ByteWriter& out, cxx::TypeExceptionDeclarationAST* self);
  void writeAstCxxAttributeAST(ByteWriter& out, cxx::CxxAttributeAST* self);
  void writeAstGccAttributeAST(ByteWriter& out, cxx::GccAttributeAST* self);
  void writeAstAlignasAttributeAST(ByteWriter& out,
                                   cxx::AlignasAttributeAST* self);
  void writeAstAlignasTypeAttributeAST(ByteWriter& out,
                                       cxx::AlignasTypeAttributeAST* self);
  void writeAstAsmAttributeAST(ByteWriter& out, cxx::AsmAttributeAST* self);
  void writeAstScopedAttributeTokenAST(ByteWriter& out,
                                       cxx::ScopedAttributeTokenAST* self);
  void writeAstSimpleAttributeTokenAST(ByteWriter& out,
                                       cxx::SimpleAttributeTokenAST* self);
  void writecxxAttribute(ByteWriter& out, const cxx::Attribute* self);
  void writecxxMeta(ByteWriter& out, const cxx::Meta* self);
  void writecxxMetaConstExpr(ByteWriter& out, const cxx::Meta::ConstExpr* self);
  void writecxxConstInt(ByteWriter& out, const cxx::ConstInt* self);
  void writecxxInitializerList(ByteWriter& out,
                               const cxx::InitializerList* self);
  void writecxxConstObject(ByteWriter& out, const cxx::ConstObject* self);
  void writecxxConstObjectMember(ByteWriter& out,
                                 const cxx::ConstObject::Member* self);
  void writecxxConstAddress(ByteWriter& out, const cxx::ConstAddress* self);
  void writecxxConstLabelAddress(ByteWriter& out,
                                 const cxx::ConstLabelAddress* self);
  void writecxxConstComplex(ByteWriter& out, const cxx::ConstComplex* self);
  void writecxxTemplateSpecialization(ByteWriter& out,
                                      const cxx::TemplateSpecialization* self);
  void writecxxInstantiationError(ByteWriter& out,
                                  const cxx::InstantiationError* self);
  void writecxxTemplateFriendship(ByteWriter& out,
                                  const cxx::TemplateFriendship* self);
  void writecxxClassLayout(ByteWriter& out, const cxx::ClassLayout* self);
  void writecxxClassLayoutMemberInfo(ByteWriter& out,
                                     const cxx::ClassLayout::MemberInfo* self);
  void writecxxClassLayoutPaddingInfo(
      ByteWriter& out, const cxx::ClassLayout::PaddingInfo* self);
  void writecxxVTableLayout(ByteWriter& out, const cxx::VTableLayout* self);
  void writecxxVTableLayoutGroup(ByteWriter& out,
                                 const cxx::VTableLayout::Group* self);
  void writecxxVTableLayoutSlot(ByteWriter& out,
                                const cxx::VTableLayout::Slot* self);
  void writecxxPendingBodyInstantiation(
      ByteWriter& out, const cxx::PendingBodyInstantiation* self);
  void writecxxPendingExceptionSpecification(
      ByteWriter& out, const cxx::PendingExceptionSpecification* self);
  void writecxxPendingFieldInitializerInstantiation(
      ByteWriter& out, const cxx::PendingFieldInitializerInstantiation* self);
  void writecxxDefaultInitializerContext(
      ByteWriter& out, const cxx::DefaultInitializerContext* self);
};

class SemanticDecoder final : public SemanticDecoderBase {
 public:
  explicit SemanticDecoder(TranslationUnit* unit) : SemanticDecoderBase(unit) {}

  [[nodiscard]] auto operator()(const ArchiveReader& archive,
                                SemanticArchiveRoots& roots) -> bool;

 private:
  [[nodiscard]] auto nameAt(NameRef ref) -> const cxx::Name*;
  [[nodiscard]] auto typeAt(TypeRef ref) -> const cxx::Type*;
  [[nodiscard]] auto symbolAt(SymbolRef ref) -> cxx::Symbol*;
  [[nodiscard]] auto astAt(AstRef ref) -> cxx::AST*;
  [[nodiscard]] auto constantAt(ConstRef ref) -> std::shared_ptr<void>;

  [[nodiscard]] auto readEnum(ByteReader& in, std::uint32_t count)
      -> std::uint32_t;
  [[nodiscard]] auto readAbiTags(ByteReader& in)
      -> const std::vector<const cxx::Identifier*>*;
  [[nodiscard]] auto readAttributes(ByteReader& in) -> const cxx::AttributeMap*;
  [[nodiscard]] auto readConstValue(ByteReader& in) -> cxx::ConstValue;
  [[nodiscard]] auto readTemplateArgument(ByteReader& in)
      -> cxx::TemplateArgument;

  template <typename T>
  [[nodiscard]] auto readAstList(ByteReader& in) -> cxx::List<T*>* {
    cxx::List<T*>* result = nullptr;
    auto tail = &result;
    const auto count = in.varCount(1);
    if (count > nodeRecords_.size()) {
      fail(std::format(
          "AST list of {} elements exceeds the {} nodes in the archive", count,
          nodeRecords_.size()));
      return nullptr;
    }
    for (std::uint32_t i = 0; ok() && i < count; ++i) {
      auto element = ast_cast<T>(astAt(AstRef{in.varU32()}));
      *tail = new (arena()) cxx::List<T*>(element);
      tail = &(*tail)->next;
    }
    return result;
  }

  [[nodiscard]] auto allocateSymbol(cxx::SymbolKind kind) -> cxx::Symbol*;
  [[nodiscard]] auto allocateAst(cxx::ASTKind kind) -> cxx::AST*;

  void decodeSymbolFields(ByteReader& in, cxx::Symbol* symbol);
  void decodeAstFields(ByteReader& in, cxx::AST* ast);

  [[nodiscard]] auto readNameIdentifier(ByteReader& in) -> const cxx::Name*;
  [[nodiscard]] auto readNameOperatorId(ByteReader& in) -> const cxx::Name*;
  [[nodiscard]] auto readNameDestructorId(ByteReader& in) -> const cxx::Name*;
  [[nodiscard]] auto readNameLiteralOperatorId(ByteReader& in)
      -> const cxx::Name*;
  [[nodiscard]] auto readNameConversionFunctionId(ByteReader& in)
      -> const cxx::Name*;
  [[nodiscard]] auto readNameTemplateId(ByteReader& in) -> const cxx::Name*;
  [[nodiscard]] auto readTypeVoidType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeNullptrType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeDecltypeAutoType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeAutoType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeBoolType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeSignedCharType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeShortIntType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeIntType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeLongIntType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeLongLongIntType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeInt128Type(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnsignedCharType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnsignedShortIntType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnsignedIntType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnsignedLongIntType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnsignedLongLongIntType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnsignedInt128Type(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeCharType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeChar8Type(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeChar16Type(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeChar32Type(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeWideCharType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeFloatType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeDoubleType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeLongDoubleType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeFloat16Type(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeQualType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeBoundedArrayType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnboundedArrayType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypePointerType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeLvalueReferenceType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeRvalueReferenceType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeFunctionType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeClassType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeEnumType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeScopedEnumType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeMemberObjectPointerType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeMemberFunctionPointerType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeNamespaceType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeTypeParameterType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeTemplateTypeParameterType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnresolvedNameType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnresolvedBoundedArrayType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnresolvedUnderlyingType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnresolvedBuiltinType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeOverloadSetType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeBuiltinVaListType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeBuiltinMetaInfoType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeBitIntType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnsignedBitIntType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnresolvedBitIntType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeVectorType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeUnresolvedVectorType(ByteReader& in)
      -> const cxx::Type*;
  [[nodiscard]] auto readTypeComplexType(ByteReader& in) -> const cxx::Type*;
  [[nodiscard]] auto readTypeAtomicType(ByteReader& in) -> const cxx::Type*;
  void readSymbolSymbol(ByteReader& in, cxx::Symbol* self);
  void readSymbolScopeSymbol(ByteReader& in, cxx::ScopeSymbol* self);
  void readSymbolNamespaceSymbol(ByteReader& in, cxx::NamespaceSymbol* self);
  void readSymbolNamespaceAliasSymbol(ByteReader& in,
                                      cxx::NamespaceAliasSymbol* self);
  void readSymbolConceptSymbol(ByteReader& in, cxx::ConceptSymbol* self);
  void readSymbolDeductionGuideSymbol(ByteReader& in,
                                      cxx::DeductionGuideSymbol* self);
  void readSymbolClassSymbol(ByteReader& in, cxx::ClassSymbol* self);
  void readSymbolEnumSymbol(ByteReader& in, cxx::EnumSymbol* self);
  void readSymbolScopedEnumSymbol(ByteReader& in, cxx::ScopedEnumSymbol* self);
  void readSymbolFunctionSymbol(ByteReader& in, cxx::FunctionSymbol* self);
  void readSymbolTypeAliasSymbol(ByteReader& in, cxx::TypeAliasSymbol* self);
  void readSymbolVariableSymbol(ByteReader& in, cxx::VariableSymbol* self);
  void readSymbolFieldSymbol(ByteReader& in, cxx::FieldSymbol* self);
  void readSymbolParameterSymbol(ByteReader& in, cxx::ParameterSymbol* self);
  void readSymbolParameterPackSymbol(ByteReader& in,
                                     cxx::ParameterPackSymbol* self);
  void readSymbolEnumeratorSymbol(ByteReader& in, cxx::EnumeratorSymbol* self);
  void readSymbolFunctionParametersSymbol(ByteReader& in,
                                          cxx::FunctionParametersSymbol* self);
  void readSymbolTemplateParametersSymbol(ByteReader& in,
                                          cxx::TemplateParametersSymbol* self);
  void readSymbolBlockSymbol(ByteReader& in, cxx::BlockSymbol* self);
  void readSymbolLambdaSymbol(ByteReader& in, cxx::LambdaSymbol* self);
  void readSymbolTypeParameterSymbol(ByteReader& in,
                                     cxx::TypeParameterSymbol* self);
  void readSymbolNonTypeParameterSymbol(ByteReader& in,
                                        cxx::NonTypeParameterSymbol* self);
  void readSymbolTemplateTypeParameterSymbol(
      ByteReader& in, cxx::TemplateTypeParameterSymbol* self);
  void readSymbolConstraintTypeParameterSymbol(
      ByteReader& in, cxx::ConstraintTypeParameterSymbol* self);
  void readSymbolOverloadSetSymbol(ByteReader& in,
                                   cxx::OverloadSetSymbol* self);
  void readSymbolBaseClassSymbol(ByteReader& in, cxx::BaseClassSymbol* self);
  void readSymbolInjectedClassNameSymbol(ByteReader& in,
                                         cxx::InjectedClassNameSymbol* self);
  void readSymbolUnresolvedSymbol(ByteReader& in, cxx::UnresolvedSymbol* self);
  void readSymbolUsingDeclarationSymbol(ByteReader& in,
                                        cxx::UsingDeclarationSymbol* self);
  void readAstManaged(ByteReader& in, cxx::Managed* self);
  void readAstAST(ByteReader& in, cxx::AST* self);
  void readAstUnitAST(ByteReader& in, cxx::UnitAST* self);
  void readAstDeclarationAST(ByteReader& in, cxx::DeclarationAST* self);
  void readAstStatementAST(ByteReader& in, cxx::StatementAST* self);
  void readAstExpressionAST(ByteReader& in, cxx::ExpressionAST* self);
  void readAstGenericAssociationAST(ByteReader& in,
                                    cxx::GenericAssociationAST* self);
  void readAstDesignatorAST(ByteReader& in, cxx::DesignatorAST* self);
  void readAstTemplateParameterAST(ByteReader& in,
                                   cxx::TemplateParameterAST* self);
  void readAstSpecifierAST(ByteReader& in, cxx::SpecifierAST* self);
  void readAstPtrOperatorAST(ByteReader& in, cxx::PtrOperatorAST* self);
  void readAstCoreDeclaratorAST(ByteReader& in, cxx::CoreDeclaratorAST* self);
  void readAstDeclaratorChunkAST(ByteReader& in, cxx::DeclaratorChunkAST* self);
  void readAstUnqualifiedIdAST(ByteReader& in, cxx::UnqualifiedIdAST* self);
  void readAstNestedNameSpecifierAST(ByteReader& in,
                                     cxx::NestedNameSpecifierAST* self);
  void readAstFunctionBodyAST(ByteReader& in, cxx::FunctionBodyAST* self);
  void readAstTemplateArgumentAST(ByteReader& in,
                                  cxx::TemplateArgumentAST* self);
  void readAstExceptionSpecifierAST(ByteReader& in,
                                    cxx::ExceptionSpecifierAST* self);
  void readAstRequirementAST(ByteReader& in, cxx::RequirementAST* self);
  void readAstNewInitializerAST(ByteReader& in, cxx::NewInitializerAST* self);
  void readAstMemInitializerAST(ByteReader& in, cxx::MemInitializerAST* self);
  void readAstLambdaCaptureAST(ByteReader& in, cxx::LambdaCaptureAST* self);
  void readAstExceptionDeclarationAST(ByteReader& in,
                                      cxx::ExceptionDeclarationAST* self);
  void readAstAttributeSpecifierAST(ByteReader& in,
                                    cxx::AttributeSpecifierAST* self);
  void readAstAttributeTokenAST(ByteReader& in, cxx::AttributeTokenAST* self);
  void readAstTranslationUnitAST(ByteReader& in, cxx::TranslationUnitAST* self);
  void readAstModuleUnitAST(ByteReader& in, cxx::ModuleUnitAST* self);
  void readAstSimpleDeclarationAST(ByteReader& in,
                                   cxx::SimpleDeclarationAST* self);
  void readAstAsmDeclarationAST(ByteReader& in, cxx::AsmDeclarationAST* self);
  void readAstNamespaceAliasDefinitionAST(
      ByteReader& in, cxx::NamespaceAliasDefinitionAST* self);
  void readAstUsingDeclarationAST(ByteReader& in,
                                  cxx::UsingDeclarationAST* self);
  void readAstUsingEnumDeclarationAST(ByteReader& in,
                                      cxx::UsingEnumDeclarationAST* self);
  void readAstUsingDirectiveAST(ByteReader& in, cxx::UsingDirectiveAST* self);
  void readAstStaticAssertDeclarationAST(ByteReader& in,
                                         cxx::StaticAssertDeclarationAST* self);
  void readAstAliasDeclarationAST(ByteReader& in,
                                  cxx::AliasDeclarationAST* self);
  void readAstOpaqueEnumDeclarationAST(ByteReader& in,
                                       cxx::OpaqueEnumDeclarationAST* self);
  void readAstFunctionDefinitionAST(ByteReader& in,
                                    cxx::FunctionDefinitionAST* self);
  void readAstTemplateDeclarationAST(ByteReader& in,
                                     cxx::TemplateDeclarationAST* self);
  void readAstConceptDefinitionAST(ByteReader& in,
                                   cxx::ConceptDefinitionAST* self);
  void readAstDeductionGuideAST(ByteReader& in, cxx::DeductionGuideAST* self);
  void readAstExplicitInstantiationAST(ByteReader& in,
                                       cxx::ExplicitInstantiationAST* self);
  void readAstExportDeclarationAST(ByteReader& in,
                                   cxx::ExportDeclarationAST* self);
  void readAstExportCompoundDeclarationAST(
      ByteReader& in, cxx::ExportCompoundDeclarationAST* self);
  void readAstLinkageSpecificationAST(ByteReader& in,
                                      cxx::LinkageSpecificationAST* self);
  void readAstNamespaceDefinitionAST(ByteReader& in,
                                     cxx::NamespaceDefinitionAST* self);
  void readAstEmptyDeclarationAST(ByteReader& in,
                                  cxx::EmptyDeclarationAST* self);
  void readAstAttributeDeclarationAST(ByteReader& in,
                                      cxx::AttributeDeclarationAST* self);
  void readAstModuleImportDeclarationAST(ByteReader& in,
                                         cxx::ModuleImportDeclarationAST* self);
  void readAstParameterDeclarationAST(ByteReader& in,
                                      cxx::ParameterDeclarationAST* self);
  void readAstAccessDeclarationAST(ByteReader& in,
                                   cxx::AccessDeclarationAST* self);
  void readAstForRangeDeclarationAST(ByteReader& in,
                                     cxx::ForRangeDeclarationAST* self);
  void readAstStructuredBindingDeclarationAST(
      ByteReader& in, cxx::StructuredBindingDeclarationAST* self);
  void readAstAsmOperandAST(ByteReader& in, cxx::AsmOperandAST* self);
  void readAstAsmQualifierAST(ByteReader& in, cxx::AsmQualifierAST* self);
  void readAstAsmClobberAST(ByteReader& in, cxx::AsmClobberAST* self);
  void readAstAsmGotoLabelAST(ByteReader& in, cxx::AsmGotoLabelAST* self);
  void readAstSplicerAST(ByteReader& in, cxx::SplicerAST* self);
  void readAstGlobalModuleFragmentAST(ByteReader& in,
                                      cxx::GlobalModuleFragmentAST* self);
  void readAstPrivateModuleFragmentAST(ByteReader& in,
                                       cxx::PrivateModuleFragmentAST* self);
  void readAstModuleDeclarationAST(ByteReader& in,
                                   cxx::ModuleDeclarationAST* self);
  void readAstModuleNameAST(ByteReader& in, cxx::ModuleNameAST* self);
  void readAstModuleQualifierAST(ByteReader& in, cxx::ModuleQualifierAST* self);
  void readAstModulePartitionAST(ByteReader& in, cxx::ModulePartitionAST* self);
  void readAstImportNameAST(ByteReader& in, cxx::ImportNameAST* self);
  void readAstInitDeclaratorAST(ByteReader& in, cxx::InitDeclaratorAST* self);
  void readAstDeclaratorAST(ByteReader& in, cxx::DeclaratorAST* self);
  void readAstUsingDeclaratorAST(ByteReader& in, cxx::UsingDeclaratorAST* self);
  void readAstEnumeratorAST(ByteReader& in, cxx::EnumeratorAST* self);
  void readAstTypeIdAST(ByteReader& in, cxx::TypeIdAST* self);
  void readAstHandlerAST(ByteReader& in, cxx::HandlerAST* self);
  void readAstBaseSpecifierAST(ByteReader& in, cxx::BaseSpecifierAST* self);
  void readAstRequiresClauseAST(ByteReader& in, cxx::RequiresClauseAST* self);
  void readAstParameterDeclarationClauseAST(
      ByteReader& in, cxx::ParameterDeclarationClauseAST* self);
  void readAstTrailingReturnTypeAST(ByteReader& in,
                                    cxx::TrailingReturnTypeAST* self);
  void readAstLambdaSpecifierAST(ByteReader& in, cxx::LambdaSpecifierAST* self);
  void readAstTypeConstraintAST(ByteReader& in, cxx::TypeConstraintAST* self);
  void readAstAttributeArgumentClauseAST(ByteReader& in,
                                         cxx::AttributeArgumentClauseAST* self);
  void readAstAttributeAST(ByteReader& in, cxx::AttributeAST* self);
  void readAstAttributeUsingPrefixAST(ByteReader& in,
                                      cxx::AttributeUsingPrefixAST* self);
  void readAstNewPlacementAST(ByteReader& in, cxx::NewPlacementAST* self);
  void readAstNestedNamespaceSpecifierAST(
      ByteReader& in, cxx::NestedNamespaceSpecifierAST* self);
  void readAstLabeledStatementAST(ByteReader& in,
                                  cxx::LabeledStatementAST* self);
  void readAstCaseStatementAST(ByteReader& in, cxx::CaseStatementAST* self);
  void readAstDefaultStatementAST(ByteReader& in,
                                  cxx::DefaultStatementAST* self);
  void readAstExpressionStatementAST(ByteReader& in,
                                     cxx::ExpressionStatementAST* self);
  void readAstCompoundStatementAST(ByteReader& in,
                                   cxx::CompoundStatementAST* self);
  void readAstIfStatementAST(ByteReader& in, cxx::IfStatementAST* self);
  void readAstConstevalIfStatementAST(ByteReader& in,
                                      cxx::ConstevalIfStatementAST* self);
  void readAstSwitchStatementAST(ByteReader& in, cxx::SwitchStatementAST* self);
  void readAstWhileStatementAST(ByteReader& in, cxx::WhileStatementAST* self);
  void readAstDoStatementAST(ByteReader& in, cxx::DoStatementAST* self);
  void readAstForRangeStatementAST(ByteReader& in,
                                   cxx::ForRangeStatementAST* self);
  void readAstForStatementAST(ByteReader& in, cxx::ForStatementAST* self);
  void readAstBreakStatementAST(ByteReader& in, cxx::BreakStatementAST* self);
  void readAstContinueStatementAST(ByteReader& in,
                                   cxx::ContinueStatementAST* self);
  void readAstReturnStatementAST(ByteReader& in, cxx::ReturnStatementAST* self);
  void readAstCoroutineReturnStatementAST(
      ByteReader& in, cxx::CoroutineReturnStatementAST* self);
  void readAstGotoStatementAST(ByteReader& in, cxx::GotoStatementAST* self);
  void readAstDeclarationStatementAST(ByteReader& in,
                                      cxx::DeclarationStatementAST* self);
  void readAstTryBlockStatementAST(ByteReader& in,
                                   cxx::TryBlockStatementAST* self);
  void readAstCharLiteralExpressionAST(ByteReader& in,
                                       cxx::CharLiteralExpressionAST* self);
  void readAstBoolLiteralExpressionAST(ByteReader& in,
                                       cxx::BoolLiteralExpressionAST* self);
  void readAstIntLiteralExpressionAST(ByteReader& in,
                                      cxx::IntLiteralExpressionAST* self);
  void readAstFloatLiteralExpressionAST(ByteReader& in,
                                        cxx::FloatLiteralExpressionAST* self);
  void readAstNullptrLiteralExpressionAST(
      ByteReader& in, cxx::NullptrLiteralExpressionAST* self);
  void readAstStringLiteralExpressionAST(ByteReader& in,
                                         cxx::StringLiteralExpressionAST* self);
  void readAstUserDefinedStringLiteralExpressionAST(
      ByteReader& in, cxx::UserDefinedStringLiteralExpressionAST* self);
  void readAstObjectLiteralExpressionAST(ByteReader& in,
                                         cxx::ObjectLiteralExpressionAST* self);
  void readAstThisExpressionAST(ByteReader& in, cxx::ThisExpressionAST* self);
  void readAstPackIndexExpressionAST(ByteReader& in,
                                     cxx::PackIndexExpressionAST* self);
  void readAstGenericSelectionExpressionAST(
      ByteReader& in, cxx::GenericSelectionExpressionAST* self);
  void readAstNestedStatementExpressionAST(
      ByteReader& in, cxx::NestedStatementExpressionAST* self);
  void readAstDefaultInitializerExpressionAST(
      ByteReader& in, cxx::DefaultInitializerExpressionAST* self);
  void readAstNestedExpressionAST(ByteReader& in,
                                  cxx::NestedExpressionAST* self);
  void readAstIdExpressionAST(ByteReader& in, cxx::IdExpressionAST* self);
  void readAstLambdaExpressionAST(ByteReader& in,
                                  cxx::LambdaExpressionAST* self);
  void readAstFoldExpressionAST(ByteReader& in, cxx::FoldExpressionAST* self);
  void readAstRightFoldExpressionAST(ByteReader& in,
                                     cxx::RightFoldExpressionAST* self);
  void readAstLeftFoldExpressionAST(ByteReader& in,
                                    cxx::LeftFoldExpressionAST* self);
  void readAstRequiresExpressionAST(ByteReader& in,
                                    cxx::RequiresExpressionAST* self);
  void readAstVaArgExpressionAST(ByteReader& in, cxx::VaArgExpressionAST* self);
  void readAstSubscriptExpressionAST(ByteReader& in,
                                     cxx::SubscriptExpressionAST* self);
  void readAstCallExpressionAST(ByteReader& in, cxx::CallExpressionAST* self);
  void readAstTypeConstructionAST(ByteReader& in,
                                  cxx::TypeConstructionAST* self);
  void readAstBracedTypeConstructionAST(ByteReader& in,
                                        cxx::BracedTypeConstructionAST* self);
  void readAstSpliceMemberExpressionAST(ByteReader& in,
                                        cxx::SpliceMemberExpressionAST* self);
  void readAstMemberExpressionAST(ByteReader& in,
                                  cxx::MemberExpressionAST* self);
  void readAstPostIncrExpressionAST(ByteReader& in,
                                    cxx::PostIncrExpressionAST* self);
  void readAstCppCastExpressionAST(ByteReader& in,
                                   cxx::CppCastExpressionAST* self);
  void readAstBuiltinBitCastExpressionAST(
      ByteReader& in, cxx::BuiltinBitCastExpressionAST* self);
  void readAstBuiltinOffsetofExpressionAST(
      ByteReader& in, cxx::BuiltinOffsetofExpressionAST* self);
  void readAstTypeidExpressionAST(ByteReader& in,
                                  cxx::TypeidExpressionAST* self);
  void readAstTypeidOfTypeExpressionAST(ByteReader& in,
                                        cxx::TypeidOfTypeExpressionAST* self);
  void readAstSpliceExpressionAST(ByteReader& in,
                                  cxx::SpliceExpressionAST* self);
  void readAstGlobalScopeReflectExpressionAST(
      ByteReader& in, cxx::GlobalScopeReflectExpressionAST* self);
  void readAstNamespaceReflectExpressionAST(
      ByteReader& in, cxx::NamespaceReflectExpressionAST* self);
  void readAstTypeIdReflectExpressionAST(ByteReader& in,
                                         cxx::TypeIdReflectExpressionAST* self);
  void readAstReflectExpressionAST(ByteReader& in,
                                   cxx::ReflectExpressionAST* self);
  void readAstLabelAddressExpressionAST(ByteReader& in,
                                        cxx::LabelAddressExpressionAST* self);
  void readAstUnaryExpressionAST(ByteReader& in, cxx::UnaryExpressionAST* self);
  void readAstAwaitExpressionAST(ByteReader& in, cxx::AwaitExpressionAST* self);
  void readAstSizeofExpressionAST(ByteReader& in,
                                  cxx::SizeofExpressionAST* self);
  void readAstSizeofTypeExpressionAST(ByteReader& in,
                                      cxx::SizeofTypeExpressionAST* self);
  void readAstSizeofPackExpressionAST(ByteReader& in,
                                      cxx::SizeofPackExpressionAST* self);
  void readAstAlignofTypeExpressionAST(ByteReader& in,
                                       cxx::AlignofTypeExpressionAST* self);
  void readAstAlignofExpressionAST(ByteReader& in,
                                   cxx::AlignofExpressionAST* self);
  void readAstNoexceptExpressionAST(ByteReader& in,
                                    cxx::NoexceptExpressionAST* self);
  void readAstNewExpressionAST(ByteReader& in, cxx::NewExpressionAST* self);
  void readAstDeleteExpressionAST(ByteReader& in,
                                  cxx::DeleteExpressionAST* self);
  void readAstCastExpressionAST(ByteReader& in, cxx::CastExpressionAST* self);
  void readAstImplicitCastExpressionAST(ByteReader& in,
                                        cxx::ImplicitCastExpressionAST* self);
  void readAstConstExpressionAST(ByteReader& in, cxx::ConstExpressionAST* self);
  void readAstBinaryExpressionAST(ByteReader& in,
                                  cxx::BinaryExpressionAST* self);
  void readAstConditionalExpressionAST(ByteReader& in,
                                       cxx::ConditionalExpressionAST* self);
  void readAstYieldExpressionAST(ByteReader& in, cxx::YieldExpressionAST* self);
  void readAstThrowExpressionAST(ByteReader& in, cxx::ThrowExpressionAST* self);
  void readAstAssignmentExpressionAST(ByteReader& in,
                                      cxx::AssignmentExpressionAST* self);
  void readAstTargetExpressionAST(ByteReader& in,
                                  cxx::TargetExpressionAST* self);
  void readAstRightExpressionAST(ByteReader& in, cxx::RightExpressionAST* self);
  void readAstCompoundAssignmentExpressionAST(
      ByteReader& in, cxx::CompoundAssignmentExpressionAST* self);
  void readAstPackExpansionExpressionAST(ByteReader& in,
                                         cxx::PackExpansionExpressionAST* self);
  void readAstDesignatedInitializerClauseAST(
      ByteReader& in, cxx::DesignatedInitializerClauseAST* self);
  void readAstTypeTraitExpressionAST(ByteReader& in,
                                     cxx::TypeTraitExpressionAST* self);
  void readAstConditionExpressionAST(ByteReader& in,
                                     cxx::ConditionExpressionAST* self);
  void readAstEqualInitializerAST(ByteReader& in,
                                  cxx::EqualInitializerAST* self);
  void readAstBracedInitListAST(ByteReader& in, cxx::BracedInitListAST* self);
  void readAstParenInitializerAST(ByteReader& in,
                                  cxx::ParenInitializerAST* self);
  void readAstThreeWayComparisonExpressionAST(
      ByteReader& in, cxx::ThreeWayComparisonExpressionAST* self);
  void readAstDefaultGenericAssociationAST(
      ByteReader& in, cxx::DefaultGenericAssociationAST* self);
  void readAstTypeGenericAssociationAST(ByteReader& in,
                                        cxx::TypeGenericAssociationAST* self);
  void readAstDotDesignatorAST(ByteReader& in, cxx::DotDesignatorAST* self);
  void readAstSubscriptDesignatorAST(ByteReader& in,
                                     cxx::SubscriptDesignatorAST* self);
  void readAstTemplateTypeParameterAST(ByteReader& in,
                                       cxx::TemplateTypeParameterAST* self);
  void readAstNonTypeTemplateParameterAST(
      ByteReader& in, cxx::NonTypeTemplateParameterAST* self);
  void readAstTypenameTypeParameterAST(ByteReader& in,
                                       cxx::TypenameTypeParameterAST* self);
  void readAstConstraintTypeParameterAST(ByteReader& in,
                                         cxx::ConstraintTypeParameterAST* self);
  void readAstTypedefSpecifierAST(ByteReader& in,
                                  cxx::TypedefSpecifierAST* self);
  void readAstFriendSpecifierAST(ByteReader& in, cxx::FriendSpecifierAST* self);
  void readAstConstevalSpecifierAST(ByteReader& in,
                                    cxx::ConstevalSpecifierAST* self);
  void readAstConstinitSpecifierAST(ByteReader& in,
                                    cxx::ConstinitSpecifierAST* self);
  void readAstConstexprSpecifierAST(ByteReader& in,
                                    cxx::ConstexprSpecifierAST* self);
  void readAstInlineSpecifierAST(ByteReader& in, cxx::InlineSpecifierAST* self);
  void readAstNoreturnSpecifierAST(ByteReader& in,
                                   cxx::NoreturnSpecifierAST* self);
  void readAstStaticSpecifierAST(ByteReader& in, cxx::StaticSpecifierAST* self);
  void readAstExternSpecifierAST(ByteReader& in, cxx::ExternSpecifierAST* self);
  void readAstRegisterSpecifierAST(ByteReader& in,
                                   cxx::RegisterSpecifierAST* self);
  void readAstThreadLocalSpecifierAST(ByteReader& in,
                                      cxx::ThreadLocalSpecifierAST* self);
  void readAstThreadSpecifierAST(ByteReader& in, cxx::ThreadSpecifierAST* self);
  void readAstMutableSpecifierAST(ByteReader& in,
                                  cxx::MutableSpecifierAST* self);
  void readAstVirtualSpecifierAST(ByteReader& in,
                                  cxx::VirtualSpecifierAST* self);
  void readAstExplicitSpecifierAST(ByteReader& in,
                                   cxx::ExplicitSpecifierAST* self);
  void readAstAutoTypeSpecifierAST(ByteReader& in,
                                   cxx::AutoTypeSpecifierAST* self);
  void readAstVoidTypeSpecifierAST(ByteReader& in,
                                   cxx::VoidTypeSpecifierAST* self);
  void readAstSizeTypeSpecifierAST(ByteReader& in,
                                   cxx::SizeTypeSpecifierAST* self);
  void readAstSignTypeSpecifierAST(ByteReader& in,
                                   cxx::SignTypeSpecifierAST* self);
  void readAstBuiltinTypeSpecifierAST(ByteReader& in,
                                      cxx::BuiltinTypeSpecifierAST* self);
  void readAstUnaryBuiltinTypeSpecifierAST(
      ByteReader& in, cxx::UnaryBuiltinTypeSpecifierAST* self);
  void readAstBinaryBuiltinTypeSpecifierAST(
      ByteReader& in, cxx::BinaryBuiltinTypeSpecifierAST* self);
  void readAstIntegralTypeSpecifierAST(ByteReader& in,
                                       cxx::IntegralTypeSpecifierAST* self);
  void readAstFloatingPointTypeSpecifierAST(
      ByteReader& in, cxx::FloatingPointTypeSpecifierAST* self);
  void readAstComplexTypeSpecifierAST(ByteReader& in,
                                      cxx::ComplexTypeSpecifierAST* self);
  void readAstNamedTypeSpecifierAST(ByteReader& in,
                                    cxx::NamedTypeSpecifierAST* self);
  void readAstAtomicTypeSpecifierAST(ByteReader& in,
                                     cxx::AtomicTypeSpecifierAST* self);
  void readAstBitIntTypeSpecifierAST(ByteReader& in,
                                     cxx::BitIntTypeSpecifierAST* self);
  void readAstUnderlyingTypeSpecifierAST(ByteReader& in,
                                         cxx::UnderlyingTypeSpecifierAST* self);
  void readAstElaboratedTypeSpecifierAST(ByteReader& in,
                                         cxx::ElaboratedTypeSpecifierAST* self);
  void readAstDecltypeAutoSpecifierAST(ByteReader& in,
                                       cxx::DecltypeAutoSpecifierAST* self);
  void readAstDecltypeSpecifierAST(ByteReader& in,
                                   cxx::DecltypeSpecifierAST* self);
  void readAstPlaceholderTypeSpecifierAST(
      ByteReader& in, cxx::PlaceholderTypeSpecifierAST* self);
  void readAstConstQualifierAST(ByteReader& in, cxx::ConstQualifierAST* self);
  void readAstVolatileQualifierAST(ByteReader& in,
                                   cxx::VolatileQualifierAST* self);
  void readAstAtomicQualifierAST(ByteReader& in, cxx::AtomicQualifierAST* self);
  void readAstRestrictQualifierAST(ByteReader& in,
                                   cxx::RestrictQualifierAST* self);
  void readAstEnumSpecifierAST(ByteReader& in, cxx::EnumSpecifierAST* self);
  void readAstClassSpecifierAST(ByteReader& in, cxx::ClassSpecifierAST* self);
  void readAstTypenameSpecifierAST(ByteReader& in,
                                   cxx::TypenameSpecifierAST* self);
  void readAstSplicerTypeSpecifierAST(ByteReader& in,
                                      cxx::SplicerTypeSpecifierAST* self);
  void readAstPointerOperatorAST(ByteReader& in, cxx::PointerOperatorAST* self);
  void readAstReferenceOperatorAST(ByteReader& in,
                                   cxx::ReferenceOperatorAST* self);
  void readAstPtrToMemberOperatorAST(ByteReader& in,
                                     cxx::PtrToMemberOperatorAST* self);
  void readAstBitfieldDeclaratorAST(ByteReader& in,
                                    cxx::BitfieldDeclaratorAST* self);
  void readAstParameterPackAST(ByteReader& in, cxx::ParameterPackAST* self);
  void readAstIdDeclaratorAST(ByteReader& in, cxx::IdDeclaratorAST* self);
  void readAstNestedDeclaratorAST(ByteReader& in,
                                  cxx::NestedDeclaratorAST* self);
  void readAstFunctionDeclaratorChunkAST(ByteReader& in,
                                         cxx::FunctionDeclaratorChunkAST* self);
  void readAstArrayDeclaratorChunkAST(ByteReader& in,
                                      cxx::ArrayDeclaratorChunkAST* self);
  void readAstNameIdAST(ByteReader& in, cxx::NameIdAST* self);
  void readAstDestructorIdAST(ByteReader& in, cxx::DestructorIdAST* self);
  void readAstDecltypeIdAST(ByteReader& in, cxx::DecltypeIdAST* self);
  void readAstOperatorFunctionIdAST(ByteReader& in,
                                    cxx::OperatorFunctionIdAST* self);
  void readAstLiteralOperatorIdAST(ByteReader& in,
                                   cxx::LiteralOperatorIdAST* self);
  void readAstConversionFunctionIdAST(ByteReader& in,
                                      cxx::ConversionFunctionIdAST* self);
  void readAstSimpleTemplateIdAST(ByteReader& in,
                                  cxx::SimpleTemplateIdAST* self);
  void readAstLiteralOperatorTemplateIdAST(
      ByteReader& in, cxx::LiteralOperatorTemplateIdAST* self);
  void readAstOperatorFunctionTemplateIdAST(
      ByteReader& in, cxx::OperatorFunctionTemplateIdAST* self);
  void readAstGlobalNestedNameSpecifierAST(
      ByteReader& in, cxx::GlobalNestedNameSpecifierAST* self);
  void readAstSimpleNestedNameSpecifierAST(
      ByteReader& in, cxx::SimpleNestedNameSpecifierAST* self);
  void readAstDecltypeNestedNameSpecifierAST(
      ByteReader& in, cxx::DecltypeNestedNameSpecifierAST* self);
  void readAstTemplateNestedNameSpecifierAST(
      ByteReader& in, cxx::TemplateNestedNameSpecifierAST* self);
  void readAstDefaultFunctionBodyAST(ByteReader& in,
                                     cxx::DefaultFunctionBodyAST* self);
  void readAstCompoundStatementFunctionBodyAST(
      ByteReader& in, cxx::CompoundStatementFunctionBodyAST* self);
  void readAstTryStatementFunctionBodyAST(
      ByteReader& in, cxx::TryStatementFunctionBodyAST* self);
  void readAstDeleteFunctionBodyAST(ByteReader& in,
                                    cxx::DeleteFunctionBodyAST* self);
  void readAstTypeTemplateArgumentAST(ByteReader& in,
                                      cxx::TypeTemplateArgumentAST* self);
  void readAstExpressionTemplateArgumentAST(
      ByteReader& in, cxx::ExpressionTemplateArgumentAST* self);
  void readAstThrowExceptionSpecifierAST(ByteReader& in,
                                         cxx::ThrowExceptionSpecifierAST* self);
  void readAstNoexceptSpecifierAST(ByteReader& in,
                                   cxx::NoexceptSpecifierAST* self);
  void readAstSimpleRequirementAST(ByteReader& in,
                                   cxx::SimpleRequirementAST* self);
  void readAstCompoundRequirementAST(ByteReader& in,
                                     cxx::CompoundRequirementAST* self);
  void readAstTypeRequirementAST(ByteReader& in, cxx::TypeRequirementAST* self);
  void readAstNestedRequirementAST(ByteReader& in,
                                   cxx::NestedRequirementAST* self);
  void readAstNewParenInitializerAST(ByteReader& in,
                                     cxx::NewParenInitializerAST* self);
  void readAstNewBracedInitializerAST(ByteReader& in,
                                      cxx::NewBracedInitializerAST* self);
  void readAstParenMemInitializerAST(ByteReader& in,
                                     cxx::ParenMemInitializerAST* self);
  void readAstBracedMemInitializerAST(ByteReader& in,
                                      cxx::BracedMemInitializerAST* self);
  void readAstThisLambdaCaptureAST(ByteReader& in,
                                   cxx::ThisLambdaCaptureAST* self);
  void readAstDerefThisLambdaCaptureAST(ByteReader& in,
                                        cxx::DerefThisLambdaCaptureAST* self);
  void readAstSimpleLambdaCaptureAST(ByteReader& in,
                                     cxx::SimpleLambdaCaptureAST* self);
  void readAstRefLambdaCaptureAST(ByteReader& in,
                                  cxx::RefLambdaCaptureAST* self);
  void readAstRefInitLambdaCaptureAST(ByteReader& in,
                                      cxx::RefInitLambdaCaptureAST* self);
  void readAstInitLambdaCaptureAST(ByteReader& in,
                                   cxx::InitLambdaCaptureAST* self);
  void readAstEllipsisExceptionDeclarationAST(
      ByteReader& in, cxx::EllipsisExceptionDeclarationAST* self);
  void readAstTypeExceptionDeclarationAST(
      ByteReader& in, cxx::TypeExceptionDeclarationAST* self);
  void readAstCxxAttributeAST(ByteReader& in, cxx::CxxAttributeAST* self);
  void readAstGccAttributeAST(ByteReader& in, cxx::GccAttributeAST* self);
  void readAstAlignasAttributeAST(ByteReader& in,
                                  cxx::AlignasAttributeAST* self);
  void readAstAlignasTypeAttributeAST(ByteReader& in,
                                      cxx::AlignasTypeAttributeAST* self);
  void readAstAsmAttributeAST(ByteReader& in, cxx::AsmAttributeAST* self);
  void readAstScopedAttributeTokenAST(ByteReader& in,
                                      cxx::ScopedAttributeTokenAST* self);
  void readAstSimpleAttributeTokenAST(ByteReader& in,
                                      cxx::SimpleAttributeTokenAST* self);
  void readcxxAttribute(ByteReader& in, cxx::Attribute* self);
  void readcxxMeta(ByteReader& in, cxx::Meta* self);
  void readcxxMetaConstExpr(ByteReader& in, cxx::Meta::ConstExpr* self);
  void readcxxConstInt(ByteReader& in, cxx::ConstInt* self);
  void readcxxInitializerList(ByteReader& in, cxx::InitializerList* self);
  void readcxxConstObject(ByteReader& in, cxx::ConstObject* self);
  void readcxxConstObjectMember(ByteReader& in, cxx::ConstObject::Member* self);
  void readcxxConstAddress(ByteReader& in, cxx::ConstAddress* self);
  void readcxxConstLabelAddress(ByteReader& in, cxx::ConstLabelAddress* self);
  void readcxxConstComplex(ByteReader& in, cxx::ConstComplex* self);
  void readcxxTemplateSpecialization(ByteReader& in,
                                     cxx::TemplateSpecialization* self);
  void readcxxInstantiationError(ByteReader& in, cxx::InstantiationError* self);
  void readcxxTemplateFriendship(ByteReader& in, cxx::TemplateFriendship* self);
  void readcxxClassLayout(ByteReader& in, cxx::ClassLayout* self);
  void readcxxClassLayoutMemberInfo(ByteReader& in,
                                    cxx::ClassLayout::MemberInfo* self);
  void readcxxClassLayoutPaddingInfo(ByteReader& in,
                                     cxx::ClassLayout::PaddingInfo* self);
  void readcxxVTableLayout(ByteReader& in, cxx::VTableLayout* self);
  void readcxxVTableLayoutGroup(ByteReader& in, cxx::VTableLayout::Group* self);
  void readcxxVTableLayoutSlot(ByteReader& in, cxx::VTableLayout::Slot* self);
  void readcxxPendingBodyInstantiation(ByteReader& in,
                                       cxx::PendingBodyInstantiation* self);
  void readcxxPendingExceptionSpecification(
      ByteReader& in, cxx::PendingExceptionSpecification* self);
  void readcxxPendingFieldInitializerInstantiation(
      ByteReader& in, cxx::PendingFieldInitializerInstantiation* self);
  void readcxxDefaultInitializerContext(ByteReader& in,
                                        cxx::DefaultInitializerContext* self);
};

}  // namespace cxx
