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

#include <cxx/arena.h>
#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/private/semantic_codec.h>
#include <cxx/symbols.h>
#include <cxx/time_trace.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>
#include <format>
#include <ranges>

namespace cxx {

namespace {

auto sectionDetail(const ByteReader& section) -> std::string {
  return std::format("{} bytes", section.remaining());
}

auto countDetail(std::size_t count) -> std::string {
  return std::format("{} records", count);
}

}  // namespace

auto SemanticEncoder::operator()(const SemanticArchiveRoots& roots,
                                 ArchiveWriter& archive) -> bool {
  ByteWriter session;

  session.varU32(static_cast<std::uint32_t>(symbolRef(roots.globalScope)));
  session.varU32(static_cast<std::uint32_t>(astRef(roots.ast)));
  session.varI32(roots.anonymousIdCount);
  session.varI32(roots.closureNameCount);
  session.varU32(roots.prefixTokenCount);

  session.varU32(
      static_cast<std::uint32_t>(roots.pendingBodyCompletions.size()));
  for (auto entry : roots.pendingBodyCompletions)
    session.varU32(static_cast<std::uint32_t>(symbolRef(entry)));

  session.varU32(
      static_cast<std::uint32_t>(roots.pendingMemberInstantiations.size()));
  for (auto entry : roots.pendingMemberInstantiations)
    session.varU32(static_cast<std::uint32_t>(symbolRef(entry)));

  session.varU32(
      static_cast<std::uint32_t>(roots.instantiatedMemberClasses.size()));
  for (auto entry : roots.instantiatedMemberClasses)
    session.varU32(static_cast<std::uint32_t>(symbolRef(entry)));

  session.varU32(static_cast<std::uint32_t>(roots.snippets.size()));
  for (const auto& [key, text] : roots.snippets) {
    session.varU64(key);
    session.varU32(static_cast<std::uint32_t>(stringRef(text)));
  }

  drain();
  resolveLocations();

  ByteWriter strings;
  flushStrings(strings);
  ByteWriter sourceMap;
  flushSourceMap(sourceMap);
  ByteWriter names;
  names_.flush(names);
  ByteWriter types;
  types_.flush(types);
  ByteWriter symbols;
  symbols_.flush(symbols);
  ByteWriter nodes;
  nodes_.flush(nodes);
  ByteWriter constants;
  flushConstants(constants);

  archive.addSection(ArchiveSection::kStrings, strings.take());
  archive.addSection(ArchiveSection::kSourceMap, sourceMap.take());
  archive.addSection(ArchiveSection::kNames, names.take());
  archive.addSection(ArchiveSection::kTypes, types.take());
  archive.addSection(ArchiveSection::kSymbols, symbols.take());
  archive.addSection(ArchiveSection::kAst, nodes.take());
  archive.addSection(ArchiveSection::kConstants, constants.take());
  archive.addSection(ArchiveSection::kSession, session.take());

  return errors().empty();
}

void SemanticEncoder::writeName(ByteWriter& out, const cxx::Name* name) {
  out.varU32(static_cast<std::uint32_t>(name->kind()));
  switch (name->kind()) {
    case cxx::NameKind::kIdentifier:
      writeNameIdentifier(out, static_cast<const cxx::Identifier*>(name));
      break;
    case cxx::NameKind::kOperatorId:
      writeNameOperatorId(out, static_cast<const cxx::OperatorId*>(name));
      break;
    case cxx::NameKind::kDestructorId:
      writeNameDestructorId(out, static_cast<const cxx::DestructorId*>(name));
      break;
    case cxx::NameKind::kLiteralOperatorId:
      writeNameLiteralOperatorId(
          out, static_cast<const cxx::LiteralOperatorId*>(name));
      break;
    case cxx::NameKind::kConversionFunctionId:
      writeNameConversionFunctionId(
          out, static_cast<const cxx::ConversionFunctionId*>(name));
      break;
    case cxx::NameKind::kTemplateId:
      writeNameTemplateId(out, static_cast<const cxx::TemplateId*>(name));
      break;
  }
}

void SemanticEncoder::writeType(ByteWriter& out, const cxx::Type* type) {
  out.varU32(static_cast<std::uint32_t>(type->kind()));
  switch (type->kind()) {
    case cxx::TypeKind::kVoid:
      writeTypeVoidType(out, static_cast<const cxx::VoidType*>(type));
      break;
    case cxx::TypeKind::kNullptr:
      writeTypeNullptrType(out, static_cast<const cxx::NullptrType*>(type));
      break;
    case cxx::TypeKind::kDecltypeAuto:
      writeTypeDecltypeAutoType(
          out, static_cast<const cxx::DecltypeAutoType*>(type));
      break;
    case cxx::TypeKind::kAuto:
      writeTypeAutoType(out, static_cast<const cxx::AutoType*>(type));
      break;
    case cxx::TypeKind::kBool:
      writeTypeBoolType(out, static_cast<const cxx::BoolType*>(type));
      break;
    case cxx::TypeKind::kSignedChar:
      writeTypeSignedCharType(out,
                              static_cast<const cxx::SignedCharType*>(type));
      break;
    case cxx::TypeKind::kShortInt:
      writeTypeShortIntType(out, static_cast<const cxx::ShortIntType*>(type));
      break;
    case cxx::TypeKind::kInt:
      writeTypeIntType(out, static_cast<const cxx::IntType*>(type));
      break;
    case cxx::TypeKind::kLongInt:
      writeTypeLongIntType(out, static_cast<const cxx::LongIntType*>(type));
      break;
    case cxx::TypeKind::kLongLongInt:
      writeTypeLongLongIntType(out,
                               static_cast<const cxx::LongLongIntType*>(type));
      break;
    case cxx::TypeKind::kInt128:
      writeTypeInt128Type(out, static_cast<const cxx::Int128Type*>(type));
      break;
    case cxx::TypeKind::kUnsignedChar:
      writeTypeUnsignedCharType(
          out, static_cast<const cxx::UnsignedCharType*>(type));
      break;
    case cxx::TypeKind::kUnsignedShortInt:
      writeTypeUnsignedShortIntType(
          out, static_cast<const cxx::UnsignedShortIntType*>(type));
      break;
    case cxx::TypeKind::kUnsignedInt:
      writeTypeUnsignedIntType(out,
                               static_cast<const cxx::UnsignedIntType*>(type));
      break;
    case cxx::TypeKind::kUnsignedLongInt:
      writeTypeUnsignedLongIntType(
          out, static_cast<const cxx::UnsignedLongIntType*>(type));
      break;
    case cxx::TypeKind::kUnsignedLongLongInt:
      writeTypeUnsignedLongLongIntType(
          out, static_cast<const cxx::UnsignedLongLongIntType*>(type));
      break;
    case cxx::TypeKind::kUnsignedInt128:
      writeTypeUnsignedInt128Type(
          out, static_cast<const cxx::UnsignedInt128Type*>(type));
      break;
    case cxx::TypeKind::kChar:
      writeTypeCharType(out, static_cast<const cxx::CharType*>(type));
      break;
    case cxx::TypeKind::kChar8:
      writeTypeChar8Type(out, static_cast<const cxx::Char8Type*>(type));
      break;
    case cxx::TypeKind::kChar16:
      writeTypeChar16Type(out, static_cast<const cxx::Char16Type*>(type));
      break;
    case cxx::TypeKind::kChar32:
      writeTypeChar32Type(out, static_cast<const cxx::Char32Type*>(type));
      break;
    case cxx::TypeKind::kWideChar:
      writeTypeWideCharType(out, static_cast<const cxx::WideCharType*>(type));
      break;
    case cxx::TypeKind::kFloat:
      writeTypeFloatType(out, static_cast<const cxx::FloatType*>(type));
      break;
    case cxx::TypeKind::kDouble:
      writeTypeDoubleType(out, static_cast<const cxx::DoubleType*>(type));
      break;
    case cxx::TypeKind::kLongDouble:
      writeTypeLongDoubleType(out,
                              static_cast<const cxx::LongDoubleType*>(type));
      break;
    case cxx::TypeKind::kFloat16:
      writeTypeFloat16Type(out, static_cast<const cxx::Float16Type*>(type));
      break;
    case cxx::TypeKind::kQual:
      writeTypeQualType(out, static_cast<const cxx::QualType*>(type));
      break;
    case cxx::TypeKind::kBoundedArray:
      writeTypeBoundedArrayType(
          out, static_cast<const cxx::BoundedArrayType*>(type));
      break;
    case cxx::TypeKind::kUnboundedArray:
      writeTypeUnboundedArrayType(
          out, static_cast<const cxx::UnboundedArrayType*>(type));
      break;
    case cxx::TypeKind::kPointer:
      writeTypePointerType(out, static_cast<const cxx::PointerType*>(type));
      break;
    case cxx::TypeKind::kLvalueReference:
      writeTypeLvalueReferenceType(
          out, static_cast<const cxx::LvalueReferenceType*>(type));
      break;
    case cxx::TypeKind::kRvalueReference:
      writeTypeRvalueReferenceType(
          out, static_cast<const cxx::RvalueReferenceType*>(type));
      break;
    case cxx::TypeKind::kFunction:
      writeTypeFunctionType(out, static_cast<const cxx::FunctionType*>(type));
      break;
    case cxx::TypeKind::kClass:
      writeTypeClassType(out, static_cast<const cxx::ClassType*>(type));
      break;
    case cxx::TypeKind::kEnum:
      writeTypeEnumType(out, static_cast<const cxx::EnumType*>(type));
      break;
    case cxx::TypeKind::kScopedEnum:
      writeTypeScopedEnumType(out,
                              static_cast<const cxx::ScopedEnumType*>(type));
      break;
    case cxx::TypeKind::kMemberObjectPointer:
      writeTypeMemberObjectPointerType(
          out, static_cast<const cxx::MemberObjectPointerType*>(type));
      break;
    case cxx::TypeKind::kMemberFunctionPointer:
      writeTypeMemberFunctionPointerType(
          out, static_cast<const cxx::MemberFunctionPointerType*>(type));
      break;
    case cxx::TypeKind::kNamespace:
      writeTypeNamespaceType(out, static_cast<const cxx::NamespaceType*>(type));
      break;
    case cxx::TypeKind::kTypeParameter:
      writeTypeTypeParameterType(
          out, static_cast<const cxx::TypeParameterType*>(type));
      break;
    case cxx::TypeKind::kTemplateTypeParameter:
      writeTypeTemplateTypeParameterType(
          out, static_cast<const cxx::TemplateTypeParameterType*>(type));
      break;
    case cxx::TypeKind::kUnresolvedName:
      writeTypeUnresolvedNameType(
          out, static_cast<const cxx::UnresolvedNameType*>(type));
      break;
    case cxx::TypeKind::kUnresolvedBoundedArray:
      writeTypeUnresolvedBoundedArrayType(
          out, static_cast<const cxx::UnresolvedBoundedArrayType*>(type));
      break;
    case cxx::TypeKind::kUnresolvedUnderlying:
      writeTypeUnresolvedUnderlyingType(
          out, static_cast<const cxx::UnresolvedUnderlyingType*>(type));
      break;
    case cxx::TypeKind::kUnresolvedBuiltin:
      writeTypeUnresolvedBuiltinType(
          out, static_cast<const cxx::UnresolvedBuiltinType*>(type));
      break;
    case cxx::TypeKind::kOverloadSet:
      writeTypeOverloadSetType(out,
                               static_cast<const cxx::OverloadSetType*>(type));
      break;
    case cxx::TypeKind::kBuiltinVaList:
      writeTypeBuiltinVaListType(
          out, static_cast<const cxx::BuiltinVaListType*>(type));
      break;
    case cxx::TypeKind::kBuiltinMetaInfo:
      writeTypeBuiltinMetaInfoType(
          out, static_cast<const cxx::BuiltinMetaInfoType*>(type));
      break;
    case cxx::TypeKind::kBitInt:
      writeTypeBitIntType(out, static_cast<const cxx::BitIntType*>(type));
      break;
    case cxx::TypeKind::kUnsignedBitInt:
      writeTypeUnsignedBitIntType(
          out, static_cast<const cxx::UnsignedBitIntType*>(type));
      break;
    case cxx::TypeKind::kUnresolvedBitInt:
      writeTypeUnresolvedBitIntType(
          out, static_cast<const cxx::UnresolvedBitIntType*>(type));
      break;
    case cxx::TypeKind::kVector:
      writeTypeVectorType(out, static_cast<const cxx::VectorType*>(type));
      break;
    case cxx::TypeKind::kUnresolvedVector:
      writeTypeUnresolvedVectorType(
          out, static_cast<const cxx::UnresolvedVectorType*>(type));
      break;
    case cxx::TypeKind::kComplex:
      writeTypeComplexType(out, static_cast<const cxx::ComplexType*>(type));
      break;
    case cxx::TypeKind::kAtomic:
      writeTypeAtomicType(out, static_cast<const cxx::AtomicType*>(type));
      break;
  }
}

void SemanticEncoder::writeSymbol(ByteWriter& out, cxx::Symbol* symbol) {
  out.varU32(static_cast<std::uint32_t>(symbol->kind()));
  switch (symbol->kind()) {
    case cxx::SymbolKind::kNamespace:
      writeSymbolNamespaceSymbol(out,
                                 static_cast<cxx::NamespaceSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kNamespaceAlias:
      writeSymbolNamespaceAliasSymbol(
          out, static_cast<cxx::NamespaceAliasSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kConcept:
      writeSymbolConceptSymbol(out, static_cast<cxx::ConceptSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kDeductionGuide:
      writeSymbolDeductionGuideSymbol(
          out, static_cast<cxx::DeductionGuideSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kClass:
      writeSymbolClassSymbol(out, static_cast<cxx::ClassSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kEnum:
      writeSymbolEnumSymbol(out, static_cast<cxx::EnumSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kScopedEnum:
      writeSymbolScopedEnumSymbol(out,
                                  static_cast<cxx::ScopedEnumSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kFunction:
      writeSymbolFunctionSymbol(out, static_cast<cxx::FunctionSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kTypeAlias:
      writeSymbolTypeAliasSymbol(out,
                                 static_cast<cxx::TypeAliasSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kVariable:
      writeSymbolVariableSymbol(out, static_cast<cxx::VariableSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kField:
      writeSymbolFieldSymbol(out, static_cast<cxx::FieldSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kParameter:
      writeSymbolParameterSymbol(out,
                                 static_cast<cxx::ParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kParameterPack:
      writeSymbolParameterPackSymbol(
          out, static_cast<cxx::ParameterPackSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kEnumerator:
      writeSymbolEnumeratorSymbol(out,
                                  static_cast<cxx::EnumeratorSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kFunctionParameters:
      writeSymbolFunctionParametersSymbol(
          out, static_cast<cxx::FunctionParametersSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kTemplateParameters:
      writeSymbolTemplateParametersSymbol(
          out, static_cast<cxx::TemplateParametersSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kBlock:
      writeSymbolBlockSymbol(out, static_cast<cxx::BlockSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kLambda:
      writeSymbolLambdaSymbol(out, static_cast<cxx::LambdaSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kTypeParameter:
      writeSymbolTypeParameterSymbol(
          out, static_cast<cxx::TypeParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kNonTypeParameter:
      writeSymbolNonTypeParameterSymbol(
          out, static_cast<cxx::NonTypeParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kTemplateTypeParameter:
      writeSymbolTemplateTypeParameterSymbol(
          out, static_cast<cxx::TemplateTypeParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kConstraintTypeParameter:
      writeSymbolConstraintTypeParameterSymbol(
          out, static_cast<cxx::ConstraintTypeParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kOverloadSet:
      writeSymbolOverloadSetSymbol(
          out, static_cast<cxx::OverloadSetSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kBaseClass:
      writeSymbolBaseClassSymbol(out,
                                 static_cast<cxx::BaseClassSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kInjectedClassName:
      writeSymbolInjectedClassNameSymbol(
          out, static_cast<cxx::InjectedClassNameSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kUnresolved:
      writeSymbolUnresolvedSymbol(out,
                                  static_cast<cxx::UnresolvedSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kUsingDeclaration:
      writeSymbolUsingDeclarationSymbol(
          out, static_cast<cxx::UsingDeclarationSymbol*>(symbol));
      break;
  }
}

void SemanticEncoder::writeAst(ByteWriter& out, cxx::AST* ast) {
  out.varU32(static_cast<std::uint32_t>(ast->kind()));
  switch (ast->kind()) {
    case cxx::ASTKind::TranslationUnit:
      writeAstTranslationUnitAST(out,
                                 static_cast<cxx::TranslationUnitAST*>(ast));
      break;
    case cxx::ASTKind::ModuleUnit:
      writeAstModuleUnitAST(out, static_cast<cxx::ModuleUnitAST*>(ast));
      break;
    case cxx::ASTKind::SimpleDeclaration:
      writeAstSimpleDeclarationAST(
          out, static_cast<cxx::SimpleDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AsmDeclaration:
      writeAstAsmDeclarationAST(out, static_cast<cxx::AsmDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::NamespaceAliasDefinition:
      writeAstNamespaceAliasDefinitionAST(
          out, static_cast<cxx::NamespaceAliasDefinitionAST*>(ast));
      break;
    case cxx::ASTKind::UsingDeclaration:
      writeAstUsingDeclarationAST(out,
                                  static_cast<cxx::UsingDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::UsingEnumDeclaration:
      writeAstUsingEnumDeclarationAST(
          out, static_cast<cxx::UsingEnumDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::UsingDirective:
      writeAstUsingDirectiveAST(out, static_cast<cxx::UsingDirectiveAST*>(ast));
      break;
    case cxx::ASTKind::StaticAssertDeclaration:
      writeAstStaticAssertDeclarationAST(
          out, static_cast<cxx::StaticAssertDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AliasDeclaration:
      writeAstAliasDeclarationAST(out,
                                  static_cast<cxx::AliasDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::OpaqueEnumDeclaration:
      writeAstOpaqueEnumDeclarationAST(
          out, static_cast<cxx::OpaqueEnumDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::FunctionDefinition:
      writeAstFunctionDefinitionAST(
          out, static_cast<cxx::FunctionDefinitionAST*>(ast));
      break;
    case cxx::ASTKind::TemplateDeclaration:
      writeAstTemplateDeclarationAST(
          out, static_cast<cxx::TemplateDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ConceptDefinition:
      writeAstConceptDefinitionAST(
          out, static_cast<cxx::ConceptDefinitionAST*>(ast));
      break;
    case cxx::ASTKind::DeductionGuide:
      writeAstDeductionGuideAST(out, static_cast<cxx::DeductionGuideAST*>(ast));
      break;
    case cxx::ASTKind::ExplicitInstantiation:
      writeAstExplicitInstantiationAST(
          out, static_cast<cxx::ExplicitInstantiationAST*>(ast));
      break;
    case cxx::ASTKind::ExportDeclaration:
      writeAstExportDeclarationAST(
          out, static_cast<cxx::ExportDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ExportCompoundDeclaration:
      writeAstExportCompoundDeclarationAST(
          out, static_cast<cxx::ExportCompoundDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::LinkageSpecification:
      writeAstLinkageSpecificationAST(
          out, static_cast<cxx::LinkageSpecificationAST*>(ast));
      break;
    case cxx::ASTKind::NamespaceDefinition:
      writeAstNamespaceDefinitionAST(
          out, static_cast<cxx::NamespaceDefinitionAST*>(ast));
      break;
    case cxx::ASTKind::EmptyDeclaration:
      writeAstEmptyDeclarationAST(out,
                                  static_cast<cxx::EmptyDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AttributeDeclaration:
      writeAstAttributeDeclarationAST(
          out, static_cast<cxx::AttributeDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ModuleImportDeclaration:
      writeAstModuleImportDeclarationAST(
          out, static_cast<cxx::ModuleImportDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ParameterDeclaration:
      writeAstParameterDeclarationAST(
          out, static_cast<cxx::ParameterDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AccessDeclaration:
      writeAstAccessDeclarationAST(
          out, static_cast<cxx::AccessDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ForRangeDeclaration:
      writeAstForRangeDeclarationAST(
          out, static_cast<cxx::ForRangeDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::StructuredBindingDeclaration:
      writeAstStructuredBindingDeclarationAST(
          out, static_cast<cxx::StructuredBindingDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AsmOperand:
      writeAstAsmOperandAST(out, static_cast<cxx::AsmOperandAST*>(ast));
      break;
    case cxx::ASTKind::AsmQualifier:
      writeAstAsmQualifierAST(out, static_cast<cxx::AsmQualifierAST*>(ast));
      break;
    case cxx::ASTKind::AsmClobber:
      writeAstAsmClobberAST(out, static_cast<cxx::AsmClobberAST*>(ast));
      break;
    case cxx::ASTKind::AsmGotoLabel:
      writeAstAsmGotoLabelAST(out, static_cast<cxx::AsmGotoLabelAST*>(ast));
      break;
    case cxx::ASTKind::Splicer:
      writeAstSplicerAST(out, static_cast<cxx::SplicerAST*>(ast));
      break;
    case cxx::ASTKind::GlobalModuleFragment:
      writeAstGlobalModuleFragmentAST(
          out, static_cast<cxx::GlobalModuleFragmentAST*>(ast));
      break;
    case cxx::ASTKind::PrivateModuleFragment:
      writeAstPrivateModuleFragmentAST(
          out, static_cast<cxx::PrivateModuleFragmentAST*>(ast));
      break;
    case cxx::ASTKind::ModuleDeclaration:
      writeAstModuleDeclarationAST(
          out, static_cast<cxx::ModuleDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ModuleName:
      writeAstModuleNameAST(out, static_cast<cxx::ModuleNameAST*>(ast));
      break;
    case cxx::ASTKind::ModuleQualifier:
      writeAstModuleQualifierAST(out,
                                 static_cast<cxx::ModuleQualifierAST*>(ast));
      break;
    case cxx::ASTKind::ModulePartition:
      writeAstModulePartitionAST(out,
                                 static_cast<cxx::ModulePartitionAST*>(ast));
      break;
    case cxx::ASTKind::ImportName:
      writeAstImportNameAST(out, static_cast<cxx::ImportNameAST*>(ast));
      break;
    case cxx::ASTKind::InitDeclarator:
      writeAstInitDeclaratorAST(out, static_cast<cxx::InitDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::Declarator:
      writeAstDeclaratorAST(out, static_cast<cxx::DeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::UsingDeclarator:
      writeAstUsingDeclaratorAST(out,
                                 static_cast<cxx::UsingDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::Enumerator:
      writeAstEnumeratorAST(out, static_cast<cxx::EnumeratorAST*>(ast));
      break;
    case cxx::ASTKind::TypeId:
      writeAstTypeIdAST(out, static_cast<cxx::TypeIdAST*>(ast));
      break;
    case cxx::ASTKind::Handler:
      writeAstHandlerAST(out, static_cast<cxx::HandlerAST*>(ast));
      break;
    case cxx::ASTKind::BaseSpecifier:
      writeAstBaseSpecifierAST(out, static_cast<cxx::BaseSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::RequiresClause:
      writeAstRequiresClauseAST(out, static_cast<cxx::RequiresClauseAST*>(ast));
      break;
    case cxx::ASTKind::ParameterDeclarationClause:
      writeAstParameterDeclarationClauseAST(
          out, static_cast<cxx::ParameterDeclarationClauseAST*>(ast));
      break;
    case cxx::ASTKind::TrailingReturnType:
      writeAstTrailingReturnTypeAST(
          out, static_cast<cxx::TrailingReturnTypeAST*>(ast));
      break;
    case cxx::ASTKind::LambdaSpecifier:
      writeAstLambdaSpecifierAST(out,
                                 static_cast<cxx::LambdaSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::TypeConstraint:
      writeAstTypeConstraintAST(out, static_cast<cxx::TypeConstraintAST*>(ast));
      break;
    case cxx::ASTKind::AttributeArgumentClause:
      writeAstAttributeArgumentClauseAST(
          out, static_cast<cxx::AttributeArgumentClauseAST*>(ast));
      break;
    case cxx::ASTKind::Attribute:
      writeAstAttributeAST(out, static_cast<cxx::AttributeAST*>(ast));
      break;
    case cxx::ASTKind::AttributeUsingPrefix:
      writeAstAttributeUsingPrefixAST(
          out, static_cast<cxx::AttributeUsingPrefixAST*>(ast));
      break;
    case cxx::ASTKind::NewPlacement:
      writeAstNewPlacementAST(out, static_cast<cxx::NewPlacementAST*>(ast));
      break;
    case cxx::ASTKind::NestedNamespaceSpecifier:
      writeAstNestedNamespaceSpecifierAST(
          out, static_cast<cxx::NestedNamespaceSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::LabeledStatement:
      writeAstLabeledStatementAST(out,
                                  static_cast<cxx::LabeledStatementAST*>(ast));
      break;
    case cxx::ASTKind::CaseStatement:
      writeAstCaseStatementAST(out, static_cast<cxx::CaseStatementAST*>(ast));
      break;
    case cxx::ASTKind::DefaultStatement:
      writeAstDefaultStatementAST(out,
                                  static_cast<cxx::DefaultStatementAST*>(ast));
      break;
    case cxx::ASTKind::ExpressionStatement:
      writeAstExpressionStatementAST(
          out, static_cast<cxx::ExpressionStatementAST*>(ast));
      break;
    case cxx::ASTKind::CompoundStatement:
      writeAstCompoundStatementAST(
          out, static_cast<cxx::CompoundStatementAST*>(ast));
      break;
    case cxx::ASTKind::IfStatement:
      writeAstIfStatementAST(out, static_cast<cxx::IfStatementAST*>(ast));
      break;
    case cxx::ASTKind::ConstevalIfStatement:
      writeAstConstevalIfStatementAST(
          out, static_cast<cxx::ConstevalIfStatementAST*>(ast));
      break;
    case cxx::ASTKind::SwitchStatement:
      writeAstSwitchStatementAST(out,
                                 static_cast<cxx::SwitchStatementAST*>(ast));
      break;
    case cxx::ASTKind::WhileStatement:
      writeAstWhileStatementAST(out, static_cast<cxx::WhileStatementAST*>(ast));
      break;
    case cxx::ASTKind::DoStatement:
      writeAstDoStatementAST(out, static_cast<cxx::DoStatementAST*>(ast));
      break;
    case cxx::ASTKind::ForRangeStatement:
      writeAstForRangeStatementAST(
          out, static_cast<cxx::ForRangeStatementAST*>(ast));
      break;
    case cxx::ASTKind::ForStatement:
      writeAstForStatementAST(out, static_cast<cxx::ForStatementAST*>(ast));
      break;
    case cxx::ASTKind::BreakStatement:
      writeAstBreakStatementAST(out, static_cast<cxx::BreakStatementAST*>(ast));
      break;
    case cxx::ASTKind::ContinueStatement:
      writeAstContinueStatementAST(
          out, static_cast<cxx::ContinueStatementAST*>(ast));
      break;
    case cxx::ASTKind::ReturnStatement:
      writeAstReturnStatementAST(out,
                                 static_cast<cxx::ReturnStatementAST*>(ast));
      break;
    case cxx::ASTKind::CoroutineReturnStatement:
      writeAstCoroutineReturnStatementAST(
          out, static_cast<cxx::CoroutineReturnStatementAST*>(ast));
      break;
    case cxx::ASTKind::GotoStatement:
      writeAstGotoStatementAST(out, static_cast<cxx::GotoStatementAST*>(ast));
      break;
    case cxx::ASTKind::DeclarationStatement:
      writeAstDeclarationStatementAST(
          out, static_cast<cxx::DeclarationStatementAST*>(ast));
      break;
    case cxx::ASTKind::TryBlockStatement:
      writeAstTryBlockStatementAST(
          out, static_cast<cxx::TryBlockStatementAST*>(ast));
      break;
    case cxx::ASTKind::CharLiteralExpression:
      writeAstCharLiteralExpressionAST(
          out, static_cast<cxx::CharLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::BoolLiteralExpression:
      writeAstBoolLiteralExpressionAST(
          out, static_cast<cxx::BoolLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::IntLiteralExpression:
      writeAstIntLiteralExpressionAST(
          out, static_cast<cxx::IntLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::FloatLiteralExpression:
      writeAstFloatLiteralExpressionAST(
          out, static_cast<cxx::FloatLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NullptrLiteralExpression:
      writeAstNullptrLiteralExpressionAST(
          out, static_cast<cxx::NullptrLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::StringLiteralExpression:
      writeAstStringLiteralExpressionAST(
          out, static_cast<cxx::StringLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::UserDefinedStringLiteralExpression:
      writeAstUserDefinedStringLiteralExpressionAST(
          out, static_cast<cxx::UserDefinedStringLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ObjectLiteralExpression:
      writeAstObjectLiteralExpressionAST(
          out, static_cast<cxx::ObjectLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ThisExpression:
      writeAstThisExpressionAST(out, static_cast<cxx::ThisExpressionAST*>(ast));
      break;
    case cxx::ASTKind::PackIndexExpression:
      writeAstPackIndexExpressionAST(
          out, static_cast<cxx::PackIndexExpressionAST*>(ast));
      break;
    case cxx::ASTKind::GenericSelectionExpression:
      writeAstGenericSelectionExpressionAST(
          out, static_cast<cxx::GenericSelectionExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NestedStatementExpression:
      writeAstNestedStatementExpressionAST(
          out, static_cast<cxx::NestedStatementExpressionAST*>(ast));
      break;
    case cxx::ASTKind::DefaultInitializerExpression:
      writeAstDefaultInitializerExpressionAST(
          out, static_cast<cxx::DefaultInitializerExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NestedExpression:
      writeAstNestedExpressionAST(out,
                                  static_cast<cxx::NestedExpressionAST*>(ast));
      break;
    case cxx::ASTKind::IdExpression:
      writeAstIdExpressionAST(out, static_cast<cxx::IdExpressionAST*>(ast));
      break;
    case cxx::ASTKind::LambdaExpression:
      writeAstLambdaExpressionAST(out,
                                  static_cast<cxx::LambdaExpressionAST*>(ast));
      break;
    case cxx::ASTKind::FoldExpression:
      writeAstFoldExpressionAST(out, static_cast<cxx::FoldExpressionAST*>(ast));
      break;
    case cxx::ASTKind::RightFoldExpression:
      writeAstRightFoldExpressionAST(
          out, static_cast<cxx::RightFoldExpressionAST*>(ast));
      break;
    case cxx::ASTKind::LeftFoldExpression:
      writeAstLeftFoldExpressionAST(
          out, static_cast<cxx::LeftFoldExpressionAST*>(ast));
      break;
    case cxx::ASTKind::RequiresExpression:
      writeAstRequiresExpressionAST(
          out, static_cast<cxx::RequiresExpressionAST*>(ast));
      break;
    case cxx::ASTKind::VaArgExpression:
      writeAstVaArgExpressionAST(out,
                                 static_cast<cxx::VaArgExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SubscriptExpression:
      writeAstSubscriptExpressionAST(
          out, static_cast<cxx::SubscriptExpressionAST*>(ast));
      break;
    case cxx::ASTKind::CallExpression:
      writeAstCallExpressionAST(out, static_cast<cxx::CallExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TypeConstruction:
      writeAstTypeConstructionAST(out,
                                  static_cast<cxx::TypeConstructionAST*>(ast));
      break;
    case cxx::ASTKind::BracedTypeConstruction:
      writeAstBracedTypeConstructionAST(
          out, static_cast<cxx::BracedTypeConstructionAST*>(ast));
      break;
    case cxx::ASTKind::SpliceMemberExpression:
      writeAstSpliceMemberExpressionAST(
          out, static_cast<cxx::SpliceMemberExpressionAST*>(ast));
      break;
    case cxx::ASTKind::MemberExpression:
      writeAstMemberExpressionAST(out,
                                  static_cast<cxx::MemberExpressionAST*>(ast));
      break;
    case cxx::ASTKind::PostIncrExpression:
      writeAstPostIncrExpressionAST(
          out, static_cast<cxx::PostIncrExpressionAST*>(ast));
      break;
    case cxx::ASTKind::CppCastExpression:
      writeAstCppCastExpressionAST(
          out, static_cast<cxx::CppCastExpressionAST*>(ast));
      break;
    case cxx::ASTKind::BuiltinBitCastExpression:
      writeAstBuiltinBitCastExpressionAST(
          out, static_cast<cxx::BuiltinBitCastExpressionAST*>(ast));
      break;
    case cxx::ASTKind::BuiltinOffsetofExpression:
      writeAstBuiltinOffsetofExpressionAST(
          out, static_cast<cxx::BuiltinOffsetofExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TypeidExpression:
      writeAstTypeidExpressionAST(out,
                                  static_cast<cxx::TypeidExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TypeidOfTypeExpression:
      writeAstTypeidOfTypeExpressionAST(
          out, static_cast<cxx::TypeidOfTypeExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SpliceExpression:
      writeAstSpliceExpressionAST(out,
                                  static_cast<cxx::SpliceExpressionAST*>(ast));
      break;
    case cxx::ASTKind::GlobalScopeReflectExpression:
      writeAstGlobalScopeReflectExpressionAST(
          out, static_cast<cxx::GlobalScopeReflectExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NamespaceReflectExpression:
      writeAstNamespaceReflectExpressionAST(
          out, static_cast<cxx::NamespaceReflectExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TypeIdReflectExpression:
      writeAstTypeIdReflectExpressionAST(
          out, static_cast<cxx::TypeIdReflectExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ReflectExpression:
      writeAstReflectExpressionAST(
          out, static_cast<cxx::ReflectExpressionAST*>(ast));
      break;
    case cxx::ASTKind::LabelAddressExpression:
      writeAstLabelAddressExpressionAST(
          out, static_cast<cxx::LabelAddressExpressionAST*>(ast));
      break;
    case cxx::ASTKind::UnaryExpression:
      writeAstUnaryExpressionAST(out,
                                 static_cast<cxx::UnaryExpressionAST*>(ast));
      break;
    case cxx::ASTKind::AwaitExpression:
      writeAstAwaitExpressionAST(out,
                                 static_cast<cxx::AwaitExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SizeofExpression:
      writeAstSizeofExpressionAST(out,
                                  static_cast<cxx::SizeofExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SizeofTypeExpression:
      writeAstSizeofTypeExpressionAST(
          out, static_cast<cxx::SizeofTypeExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SizeofPackExpression:
      writeAstSizeofPackExpressionAST(
          out, static_cast<cxx::SizeofPackExpressionAST*>(ast));
      break;
    case cxx::ASTKind::AlignofTypeExpression:
      writeAstAlignofTypeExpressionAST(
          out, static_cast<cxx::AlignofTypeExpressionAST*>(ast));
      break;
    case cxx::ASTKind::AlignofExpression:
      writeAstAlignofExpressionAST(
          out, static_cast<cxx::AlignofExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NoexceptExpression:
      writeAstNoexceptExpressionAST(
          out, static_cast<cxx::NoexceptExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NewExpression:
      writeAstNewExpressionAST(out, static_cast<cxx::NewExpressionAST*>(ast));
      break;
    case cxx::ASTKind::DeleteExpression:
      writeAstDeleteExpressionAST(out,
                                  static_cast<cxx::DeleteExpressionAST*>(ast));
      break;
    case cxx::ASTKind::CastExpression:
      writeAstCastExpressionAST(out, static_cast<cxx::CastExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ImplicitCastExpression:
      writeAstImplicitCastExpressionAST(
          out, static_cast<cxx::ImplicitCastExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ConstExpression:
      writeAstConstExpressionAST(out,
                                 static_cast<cxx::ConstExpressionAST*>(ast));
      break;
    case cxx::ASTKind::BinaryExpression:
      writeAstBinaryExpressionAST(out,
                                  static_cast<cxx::BinaryExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ConditionalExpression:
      writeAstConditionalExpressionAST(
          out, static_cast<cxx::ConditionalExpressionAST*>(ast));
      break;
    case cxx::ASTKind::YieldExpression:
      writeAstYieldExpressionAST(out,
                                 static_cast<cxx::YieldExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ThrowExpression:
      writeAstThrowExpressionAST(out,
                                 static_cast<cxx::ThrowExpressionAST*>(ast));
      break;
    case cxx::ASTKind::AssignmentExpression:
      writeAstAssignmentExpressionAST(
          out, static_cast<cxx::AssignmentExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TargetExpression:
      writeAstTargetExpressionAST(out,
                                  static_cast<cxx::TargetExpressionAST*>(ast));
      break;
    case cxx::ASTKind::RightExpression:
      writeAstRightExpressionAST(out,
                                 static_cast<cxx::RightExpressionAST*>(ast));
      break;
    case cxx::ASTKind::CompoundAssignmentExpression:
      writeAstCompoundAssignmentExpressionAST(
          out, static_cast<cxx::CompoundAssignmentExpressionAST*>(ast));
      break;
    case cxx::ASTKind::PackExpansionExpression:
      writeAstPackExpansionExpressionAST(
          out, static_cast<cxx::PackExpansionExpressionAST*>(ast));
      break;
    case cxx::ASTKind::DesignatedInitializerClause:
      writeAstDesignatedInitializerClauseAST(
          out, static_cast<cxx::DesignatedInitializerClauseAST*>(ast));
      break;
    case cxx::ASTKind::TypeTraitExpression:
      writeAstTypeTraitExpressionAST(
          out, static_cast<cxx::TypeTraitExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ConditionExpression:
      writeAstConditionExpressionAST(
          out, static_cast<cxx::ConditionExpressionAST*>(ast));
      break;
    case cxx::ASTKind::EqualInitializer:
      writeAstEqualInitializerAST(out,
                                  static_cast<cxx::EqualInitializerAST*>(ast));
      break;
    case cxx::ASTKind::BracedInitList:
      writeAstBracedInitListAST(out, static_cast<cxx::BracedInitListAST*>(ast));
      break;
    case cxx::ASTKind::ParenInitializer:
      writeAstParenInitializerAST(out,
                                  static_cast<cxx::ParenInitializerAST*>(ast));
      break;
    case cxx::ASTKind::ThreeWayComparisonExpression:
      writeAstThreeWayComparisonExpressionAST(
          out, static_cast<cxx::ThreeWayComparisonExpressionAST*>(ast));
      break;
    case cxx::ASTKind::DefaultGenericAssociation:
      writeAstDefaultGenericAssociationAST(
          out, static_cast<cxx::DefaultGenericAssociationAST*>(ast));
      break;
    case cxx::ASTKind::TypeGenericAssociation:
      writeAstTypeGenericAssociationAST(
          out, static_cast<cxx::TypeGenericAssociationAST*>(ast));
      break;
    case cxx::ASTKind::DotDesignator:
      writeAstDotDesignatorAST(out, static_cast<cxx::DotDesignatorAST*>(ast));
      break;
    case cxx::ASTKind::SubscriptDesignator:
      writeAstSubscriptDesignatorAST(
          out, static_cast<cxx::SubscriptDesignatorAST*>(ast));
      break;
    case cxx::ASTKind::TemplateTypeParameter:
      writeAstTemplateTypeParameterAST(
          out, static_cast<cxx::TemplateTypeParameterAST*>(ast));
      break;
    case cxx::ASTKind::NonTypeTemplateParameter:
      writeAstNonTypeTemplateParameterAST(
          out, static_cast<cxx::NonTypeTemplateParameterAST*>(ast));
      break;
    case cxx::ASTKind::TypenameTypeParameter:
      writeAstTypenameTypeParameterAST(
          out, static_cast<cxx::TypenameTypeParameterAST*>(ast));
      break;
    case cxx::ASTKind::ConstraintTypeParameter:
      writeAstConstraintTypeParameterAST(
          out, static_cast<cxx::ConstraintTypeParameterAST*>(ast));
      break;
    case cxx::ASTKind::TypedefSpecifier:
      writeAstTypedefSpecifierAST(out,
                                  static_cast<cxx::TypedefSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::FriendSpecifier:
      writeAstFriendSpecifierAST(out,
                                 static_cast<cxx::FriendSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ConstevalSpecifier:
      writeAstConstevalSpecifierAST(
          out, static_cast<cxx::ConstevalSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ConstinitSpecifier:
      writeAstConstinitSpecifierAST(
          out, static_cast<cxx::ConstinitSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ConstexprSpecifier:
      writeAstConstexprSpecifierAST(
          out, static_cast<cxx::ConstexprSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::InlineSpecifier:
      writeAstInlineSpecifierAST(out,
                                 static_cast<cxx::InlineSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::NoreturnSpecifier:
      writeAstNoreturnSpecifierAST(
          out, static_cast<cxx::NoreturnSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::StaticSpecifier:
      writeAstStaticSpecifierAST(out,
                                 static_cast<cxx::StaticSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ExternSpecifier:
      writeAstExternSpecifierAST(out,
                                 static_cast<cxx::ExternSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::RegisterSpecifier:
      writeAstRegisterSpecifierAST(
          out, static_cast<cxx::RegisterSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ThreadLocalSpecifier:
      writeAstThreadLocalSpecifierAST(
          out, static_cast<cxx::ThreadLocalSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ThreadSpecifier:
      writeAstThreadSpecifierAST(out,
                                 static_cast<cxx::ThreadSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::MutableSpecifier:
      writeAstMutableSpecifierAST(out,
                                  static_cast<cxx::MutableSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::VirtualSpecifier:
      writeAstVirtualSpecifierAST(out,
                                  static_cast<cxx::VirtualSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ExplicitSpecifier:
      writeAstExplicitSpecifierAST(
          out, static_cast<cxx::ExplicitSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::AutoTypeSpecifier:
      writeAstAutoTypeSpecifierAST(
          out, static_cast<cxx::AutoTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::VoidTypeSpecifier:
      writeAstVoidTypeSpecifierAST(
          out, static_cast<cxx::VoidTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SizeTypeSpecifier:
      writeAstSizeTypeSpecifierAST(
          out, static_cast<cxx::SizeTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SignTypeSpecifier:
      writeAstSignTypeSpecifierAST(
          out, static_cast<cxx::SignTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::BuiltinTypeSpecifier:
      writeAstBuiltinTypeSpecifierAST(
          out, static_cast<cxx::BuiltinTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::UnaryBuiltinTypeSpecifier:
      writeAstUnaryBuiltinTypeSpecifierAST(
          out, static_cast<cxx::UnaryBuiltinTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::BinaryBuiltinTypeSpecifier:
      writeAstBinaryBuiltinTypeSpecifierAST(
          out, static_cast<cxx::BinaryBuiltinTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::IntegralTypeSpecifier:
      writeAstIntegralTypeSpecifierAST(
          out, static_cast<cxx::IntegralTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::FloatingPointTypeSpecifier:
      writeAstFloatingPointTypeSpecifierAST(
          out, static_cast<cxx::FloatingPointTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ComplexTypeSpecifier:
      writeAstComplexTypeSpecifierAST(
          out, static_cast<cxx::ComplexTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::NamedTypeSpecifier:
      writeAstNamedTypeSpecifierAST(
          out, static_cast<cxx::NamedTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::AtomicTypeSpecifier:
      writeAstAtomicTypeSpecifierAST(
          out, static_cast<cxx::AtomicTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::BitIntTypeSpecifier:
      writeAstBitIntTypeSpecifierAST(
          out, static_cast<cxx::BitIntTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::UnderlyingTypeSpecifier:
      writeAstUnderlyingTypeSpecifierAST(
          out, static_cast<cxx::UnderlyingTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ElaboratedTypeSpecifier:
      writeAstElaboratedTypeSpecifierAST(
          out, static_cast<cxx::ElaboratedTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::DecltypeAutoSpecifier:
      writeAstDecltypeAutoSpecifierAST(
          out, static_cast<cxx::DecltypeAutoSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::DecltypeSpecifier:
      writeAstDecltypeSpecifierAST(
          out, static_cast<cxx::DecltypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::PlaceholderTypeSpecifier:
      writeAstPlaceholderTypeSpecifierAST(
          out, static_cast<cxx::PlaceholderTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ConstQualifier:
      writeAstConstQualifierAST(out, static_cast<cxx::ConstQualifierAST*>(ast));
      break;
    case cxx::ASTKind::VolatileQualifier:
      writeAstVolatileQualifierAST(
          out, static_cast<cxx::VolatileQualifierAST*>(ast));
      break;
    case cxx::ASTKind::AtomicQualifier:
      writeAstAtomicQualifierAST(out,
                                 static_cast<cxx::AtomicQualifierAST*>(ast));
      break;
    case cxx::ASTKind::RestrictQualifier:
      writeAstRestrictQualifierAST(
          out, static_cast<cxx::RestrictQualifierAST*>(ast));
      break;
    case cxx::ASTKind::EnumSpecifier:
      writeAstEnumSpecifierAST(out, static_cast<cxx::EnumSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ClassSpecifier:
      writeAstClassSpecifierAST(out, static_cast<cxx::ClassSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::TypenameSpecifier:
      writeAstTypenameSpecifierAST(
          out, static_cast<cxx::TypenameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SplicerTypeSpecifier:
      writeAstSplicerTypeSpecifierAST(
          out, static_cast<cxx::SplicerTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::PointerOperator:
      writeAstPointerOperatorAST(out,
                                 static_cast<cxx::PointerOperatorAST*>(ast));
      break;
    case cxx::ASTKind::ReferenceOperator:
      writeAstReferenceOperatorAST(
          out, static_cast<cxx::ReferenceOperatorAST*>(ast));
      break;
    case cxx::ASTKind::PtrToMemberOperator:
      writeAstPtrToMemberOperatorAST(
          out, static_cast<cxx::PtrToMemberOperatorAST*>(ast));
      break;
    case cxx::ASTKind::BitfieldDeclarator:
      writeAstBitfieldDeclaratorAST(
          out, static_cast<cxx::BitfieldDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::ParameterPack:
      writeAstParameterPackAST(out, static_cast<cxx::ParameterPackAST*>(ast));
      break;
    case cxx::ASTKind::IdDeclarator:
      writeAstIdDeclaratorAST(out, static_cast<cxx::IdDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::NestedDeclarator:
      writeAstNestedDeclaratorAST(out,
                                  static_cast<cxx::NestedDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::FunctionDeclaratorChunk:
      writeAstFunctionDeclaratorChunkAST(
          out, static_cast<cxx::FunctionDeclaratorChunkAST*>(ast));
      break;
    case cxx::ASTKind::ArrayDeclaratorChunk:
      writeAstArrayDeclaratorChunkAST(
          out, static_cast<cxx::ArrayDeclaratorChunkAST*>(ast));
      break;
    case cxx::ASTKind::NameId:
      writeAstNameIdAST(out, static_cast<cxx::NameIdAST*>(ast));
      break;
    case cxx::ASTKind::DestructorId:
      writeAstDestructorIdAST(out, static_cast<cxx::DestructorIdAST*>(ast));
      break;
    case cxx::ASTKind::DecltypeId:
      writeAstDecltypeIdAST(out, static_cast<cxx::DecltypeIdAST*>(ast));
      break;
    case cxx::ASTKind::OperatorFunctionId:
      writeAstOperatorFunctionIdAST(
          out, static_cast<cxx::OperatorFunctionIdAST*>(ast));
      break;
    case cxx::ASTKind::LiteralOperatorId:
      writeAstLiteralOperatorIdAST(
          out, static_cast<cxx::LiteralOperatorIdAST*>(ast));
      break;
    case cxx::ASTKind::ConversionFunctionId:
      writeAstConversionFunctionIdAST(
          out, static_cast<cxx::ConversionFunctionIdAST*>(ast));
      break;
    case cxx::ASTKind::SimpleTemplateId:
      writeAstSimpleTemplateIdAST(out,
                                  static_cast<cxx::SimpleTemplateIdAST*>(ast));
      break;
    case cxx::ASTKind::LiteralOperatorTemplateId:
      writeAstLiteralOperatorTemplateIdAST(
          out, static_cast<cxx::LiteralOperatorTemplateIdAST*>(ast));
      break;
    case cxx::ASTKind::OperatorFunctionTemplateId:
      writeAstOperatorFunctionTemplateIdAST(
          out, static_cast<cxx::OperatorFunctionTemplateIdAST*>(ast));
      break;
    case cxx::ASTKind::GlobalNestedNameSpecifier:
      writeAstGlobalNestedNameSpecifierAST(
          out, static_cast<cxx::GlobalNestedNameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SimpleNestedNameSpecifier:
      writeAstSimpleNestedNameSpecifierAST(
          out, static_cast<cxx::SimpleNestedNameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::DecltypeNestedNameSpecifier:
      writeAstDecltypeNestedNameSpecifierAST(
          out, static_cast<cxx::DecltypeNestedNameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::TemplateNestedNameSpecifier:
      writeAstTemplateNestedNameSpecifierAST(
          out, static_cast<cxx::TemplateNestedNameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::DefaultFunctionBody:
      writeAstDefaultFunctionBodyAST(
          out, static_cast<cxx::DefaultFunctionBodyAST*>(ast));
      break;
    case cxx::ASTKind::CompoundStatementFunctionBody:
      writeAstCompoundStatementFunctionBodyAST(
          out, static_cast<cxx::CompoundStatementFunctionBodyAST*>(ast));
      break;
    case cxx::ASTKind::TryStatementFunctionBody:
      writeAstTryStatementFunctionBodyAST(
          out, static_cast<cxx::TryStatementFunctionBodyAST*>(ast));
      break;
    case cxx::ASTKind::DeleteFunctionBody:
      writeAstDeleteFunctionBodyAST(
          out, static_cast<cxx::DeleteFunctionBodyAST*>(ast));
      break;
    case cxx::ASTKind::TypeTemplateArgument:
      writeAstTypeTemplateArgumentAST(
          out, static_cast<cxx::TypeTemplateArgumentAST*>(ast));
      break;
    case cxx::ASTKind::ExpressionTemplateArgument:
      writeAstExpressionTemplateArgumentAST(
          out, static_cast<cxx::ExpressionTemplateArgumentAST*>(ast));
      break;
    case cxx::ASTKind::ThrowExceptionSpecifier:
      writeAstThrowExceptionSpecifierAST(
          out, static_cast<cxx::ThrowExceptionSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::NoexceptSpecifier:
      writeAstNoexceptSpecifierAST(
          out, static_cast<cxx::NoexceptSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SimpleRequirement:
      writeAstSimpleRequirementAST(
          out, static_cast<cxx::SimpleRequirementAST*>(ast));
      break;
    case cxx::ASTKind::CompoundRequirement:
      writeAstCompoundRequirementAST(
          out, static_cast<cxx::CompoundRequirementAST*>(ast));
      break;
    case cxx::ASTKind::TypeRequirement:
      writeAstTypeRequirementAST(out,
                                 static_cast<cxx::TypeRequirementAST*>(ast));
      break;
    case cxx::ASTKind::NestedRequirement:
      writeAstNestedRequirementAST(
          out, static_cast<cxx::NestedRequirementAST*>(ast));
      break;
    case cxx::ASTKind::NewParenInitializer:
      writeAstNewParenInitializerAST(
          out, static_cast<cxx::NewParenInitializerAST*>(ast));
      break;
    case cxx::ASTKind::NewBracedInitializer:
      writeAstNewBracedInitializerAST(
          out, static_cast<cxx::NewBracedInitializerAST*>(ast));
      break;
    case cxx::ASTKind::ParenMemInitializer:
      writeAstParenMemInitializerAST(
          out, static_cast<cxx::ParenMemInitializerAST*>(ast));
      break;
    case cxx::ASTKind::BracedMemInitializer:
      writeAstBracedMemInitializerAST(
          out, static_cast<cxx::BracedMemInitializerAST*>(ast));
      break;
    case cxx::ASTKind::ThisLambdaCapture:
      writeAstThisLambdaCaptureAST(
          out, static_cast<cxx::ThisLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::DerefThisLambdaCapture:
      writeAstDerefThisLambdaCaptureAST(
          out, static_cast<cxx::DerefThisLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::SimpleLambdaCapture:
      writeAstSimpleLambdaCaptureAST(
          out, static_cast<cxx::SimpleLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::RefLambdaCapture:
      writeAstRefLambdaCaptureAST(out,
                                  static_cast<cxx::RefLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::RefInitLambdaCapture:
      writeAstRefInitLambdaCaptureAST(
          out, static_cast<cxx::RefInitLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::InitLambdaCapture:
      writeAstInitLambdaCaptureAST(
          out, static_cast<cxx::InitLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::EllipsisExceptionDeclaration:
      writeAstEllipsisExceptionDeclarationAST(
          out, static_cast<cxx::EllipsisExceptionDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::TypeExceptionDeclaration:
      writeAstTypeExceptionDeclarationAST(
          out, static_cast<cxx::TypeExceptionDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::CxxAttribute:
      writeAstCxxAttributeAST(out, static_cast<cxx::CxxAttributeAST*>(ast));
      break;
    case cxx::ASTKind::GccAttribute:
      writeAstGccAttributeAST(out, static_cast<cxx::GccAttributeAST*>(ast));
      break;
    case cxx::ASTKind::AlignasAttribute:
      writeAstAlignasAttributeAST(out,
                                  static_cast<cxx::AlignasAttributeAST*>(ast));
      break;
    case cxx::ASTKind::AlignasTypeAttribute:
      writeAstAlignasTypeAttributeAST(
          out, static_cast<cxx::AlignasTypeAttributeAST*>(ast));
      break;
    case cxx::ASTKind::AsmAttribute:
      writeAstAsmAttributeAST(out, static_cast<cxx::AsmAttributeAST*>(ast));
      break;
    case cxx::ASTKind::ScopedAttributeToken:
      writeAstScopedAttributeTokenAST(
          out, static_cast<cxx::ScopedAttributeTokenAST*>(ast));
      break;
    case cxx::ASTKind::SimpleAttributeToken:
      writeAstSimpleAttributeTokenAST(
          out, static_cast<cxx::SimpleAttributeTokenAST*>(ast));
      break;
  }
}

void SemanticEncoder::writeConstNode(ByteWriter& out, const ConstNode& node) {
  out.u8(static_cast<std::uint8_t>(node.kind));
  switch (node.kind) {
    case 0:
      writecxxMeta(out, static_cast<const cxx::Meta*>(node.owner.get()));
      break;
    case 1:
      writecxxInitializerList(
          out, static_cast<const cxx::InitializerList*>(node.owner.get()));
      break;
    case 2:
      writecxxConstObject(
          out, static_cast<const cxx::ConstObject*>(node.owner.get()));
      break;
    case 3:
      writecxxConstAddress(
          out, static_cast<const cxx::ConstAddress*>(node.owner.get()));
      break;
    case 4:
      writecxxConstLabelAddress(
          out, static_cast<const cxx::ConstLabelAddress*>(node.owner.get()));
      break;
    default:
      reportError("unknown constant node kind");
      break;
  }
}

void SemanticEncoder::drain() {
  for (;;) {
    bool progress = false;

    while (names_.hasPending()) {
      auto entity = names_.takePending();
      scratch_.clear();
      writeName(scratch_, entity);
      names_.store(entity, scratch_);
      progress = true;
    }

    while (types_.hasPending()) {
      auto entity = types_.takePending();
      scratch_.clear();
      writeType(scratch_, entity);
      types_.store(entity, scratch_);
      progress = true;
    }

    while (symbols_.hasPending()) {
      auto entity = symbols_.takePending();
      scratch_.clear();
      writeSymbol(scratch_, entity);
      symbols_.store(entity, scratch_);
      progress = true;
    }

    while (nodes_.hasPending()) {
      auto entity = nodes_.takePending();
      scratch_.clear();
      writeAst(scratch_, entity);
      nodes_.store(entity, scratch_);
      progress = true;
    }

    while (constCursor_ < constPending_.size()) {
      const auto index = constCursor_++;
      ByteWriter record;
      writeConstNode(record, constPending_[index]);
      constRecords_[index] = record.take();
      progress = true;
    }

    if (!progress) break;
  }
}
void SemanticEncoder::writeLiteral(ByteWriter& out,
                                   const cxx::Literal* literal) {
  out.boolean(literal != nullptr);
  if (literal)
    out.varU32(static_cast<std::uint32_t>(stringRef(literal->value())));
}

void SemanticEncoder::writeAbiTags(
    ByteWriter& out, const std::vector<const cxx::Identifier*>* tags) {
  out.boolean(tags != nullptr);
  if (!tags) return;
  out.varU32(static_cast<std::uint32_t>(tags->size()));
  for (auto tag : *tags)
    out.varU32(static_cast<std::uint32_t>(identifierRef(tag)));
}

void SemanticEncoder::writeAttributes(ByteWriter& out,
                                      const cxx::AttributeMap* attributes) {
  out.boolean(attributes != nullptr);
  if (!attributes) return;
  out.varU32(static_cast<std::uint32_t>(attributes->size()));
  for (const auto& attribute : *attributes) writecxxAttribute(out, &attribute);
}

void SemanticEncoder::writeConstValue(ByteWriter& out,
                                      const cxx::ConstValue& value) {
  out.u8(static_cast<std::uint8_t>(value.index()));
  switch (value.index()) {
    case 0: {
      out.i64(std::get<0>(value));
      break;
    }
    case 1: {
      writeLiteral(out, std::get<1>(value));
      break;
    }
    case 2: {
      out.f32(std::get<2>(value));
      break;
    }
    case 3: {
      out.f64(std::get<3>(value));
      break;
    }
    case 4: {
      out.f80(std::get<4>(value));
      break;
    }
    case 5: {
      out.varU32(static_cast<std::uint32_t>(constRef(std::get<5>(value))));
      break;
    }
    case 6: {
      out.varU32(static_cast<std::uint32_t>(constRef(std::get<6>(value))));
      break;
    }
    case 7: {
      out.varU32(static_cast<std::uint32_t>(constRef(std::get<7>(value))));
      break;
    }
    case 8: {
      out.varU32(static_cast<std::uint32_t>(constRef(std::get<8>(value))));
      break;
    }
    case 9: {
      out.varU32(static_cast<std::uint32_t>(constRef(std::get<9>(value))));
      break;
    }
    case 10: {
      break;
    }
  }
}

void SemanticEncoder::writeTemplateArgument(
    ByteWriter& out, const cxx::TemplateArgument& argument) {
  out.u8(static_cast<std::uint8_t>(argument.index()));
  switch (argument.index()) {
    case 0:
      out.varU32(static_cast<std::uint32_t>(typeRef(std::get<0>(argument))));
      break;
    case 1:
      out.varU32(static_cast<std::uint32_t>(symbolRef(std::get<1>(argument))));
      break;
    case 2:
      writeConstValue(out, std::get<2>(argument));
      break;
    case 3:
      out.varU32(static_cast<std::uint32_t>(astRef(std::get<3>(argument))));
      break;
  }
}

void SemanticEncoder::writeNameIdentifier(
    ByteWriter& out, [[maybe_unused]] const cxx::Identifier* self) {
  // name
  out.varU32(static_cast<std::uint32_t>(stringRef(self->name())));
}

void SemanticEncoder::writeNameOperatorId(
    ByteWriter& out, [[maybe_unused]] const cxx::OperatorId* self) {
  // op
  out.varU32(static_cast<std::uint32_t>(self->op()));
}

void SemanticEncoder::writeNameDestructorId(
    ByteWriter& out, [[maybe_unused]] const cxx::DestructorId* self) {
  //
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
}

void SemanticEncoder::writeNameLiteralOperatorId(
    ByteWriter& out, [[maybe_unused]] const cxx::LiteralOperatorId* self) {
  // name
  out.varU32(static_cast<std::uint32_t>(stringRef(self->name())));
}

void SemanticEncoder::writeNameConversionFunctionId(
    ByteWriter& out, [[maybe_unused]] const cxx::ConversionFunctionId* self) {
  // type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
}

void SemanticEncoder::writeNameTemplateId(
    ByteWriter& out, [[maybe_unused]] const cxx::TemplateId* self) {
  // name
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // arguments
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->arguments())));
  for (const auto& element1 : self->arguments()) {
    writeTemplateArgument(out, element1);
  }
}

void SemanticEncoder::writeTypeVoidType(
    ByteWriter& out, [[maybe_unused]] const cxx::VoidType* self) {}

void SemanticEncoder::writeTypeNullptrType(
    ByteWriter& out, [[maybe_unused]] const cxx::NullptrType* self) {}

void SemanticEncoder::writeTypeDecltypeAutoType(
    ByteWriter& out, [[maybe_unused]] const cxx::DecltypeAutoType* self) {}

void SemanticEncoder::writeTypeAutoType(
    ByteWriter& out, [[maybe_unused]] const cxx::AutoType* self) {}

void SemanticEncoder::writeTypeBoolType(
    ByteWriter& out, [[maybe_unused]] const cxx::BoolType* self) {}

void SemanticEncoder::writeTypeSignedCharType(
    ByteWriter& out, [[maybe_unused]] const cxx::SignedCharType* self) {}

void SemanticEncoder::writeTypeShortIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::ShortIntType* self) {}

void SemanticEncoder::writeTypeIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::IntType* self) {}

void SemanticEncoder::writeTypeLongIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::LongIntType* self) {}

void SemanticEncoder::writeTypeLongLongIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::LongLongIntType* self) {}

void SemanticEncoder::writeTypeInt128Type(
    ByteWriter& out, [[maybe_unused]] const cxx::Int128Type* self) {}

void SemanticEncoder::writeTypeUnsignedCharType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnsignedCharType* self) {}

void SemanticEncoder::writeTypeUnsignedShortIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnsignedShortIntType* self) {}

void SemanticEncoder::writeTypeUnsignedIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnsignedIntType* self) {}

void SemanticEncoder::writeTypeUnsignedLongIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnsignedLongIntType* self) {}

void SemanticEncoder::writeTypeUnsignedLongLongIntType(
    ByteWriter& out,
    [[maybe_unused]] const cxx::UnsignedLongLongIntType* self) {}

void SemanticEncoder::writeTypeUnsignedInt128Type(
    ByteWriter& out, [[maybe_unused]] const cxx::UnsignedInt128Type* self) {}

void SemanticEncoder::writeTypeCharType(
    ByteWriter& out, [[maybe_unused]] const cxx::CharType* self) {}

void SemanticEncoder::writeTypeChar8Type(
    ByteWriter& out, [[maybe_unused]] const cxx::Char8Type* self) {}

void SemanticEncoder::writeTypeChar16Type(
    ByteWriter& out, [[maybe_unused]] const cxx::Char16Type* self) {}

void SemanticEncoder::writeTypeChar32Type(
    ByteWriter& out, [[maybe_unused]] const cxx::Char32Type* self) {}

void SemanticEncoder::writeTypeWideCharType(
    ByteWriter& out, [[maybe_unused]] const cxx::WideCharType* self) {}

void SemanticEncoder::writeTypeFloatType(
    ByteWriter& out, [[maybe_unused]] const cxx::FloatType* self) {}

void SemanticEncoder::writeTypeDoubleType(
    ByteWriter& out, [[maybe_unused]] const cxx::DoubleType* self) {}

void SemanticEncoder::writeTypeLongDoubleType(
    ByteWriter& out, [[maybe_unused]] const cxx::LongDoubleType* self) {}

void SemanticEncoder::writeTypeFloat16Type(
    ByteWriter& out, [[maybe_unused]] const cxx::Float16Type* self) {}

void SemanticEncoder::writeTypeQualType(
    ByteWriter& out, [[maybe_unused]] const cxx::QualType* self) {
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
  // cvQualifiers
  out.varU32(static_cast<std::uint32_t>(self->cvQualifiers()));
}

void SemanticEncoder::writeTypeBoundedArrayType(
    ByteWriter& out, [[maybe_unused]] const cxx::BoundedArrayType* self) {
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
  // size
  out.varU64(static_cast<std::uint64_t>(self->size()));
}

void SemanticEncoder::writeTypeUnboundedArrayType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnboundedArrayType* self) {
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
}

void SemanticEncoder::writeTypePointerType(
    ByteWriter& out, [[maybe_unused]] const cxx::PointerType* self) {
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
}

void SemanticEncoder::writeTypeLvalueReferenceType(
    ByteWriter& out, [[maybe_unused]] const cxx::LvalueReferenceType* self) {
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
}

void SemanticEncoder::writeTypeRvalueReferenceType(
    ByteWriter& out, [[maybe_unused]] const cxx::RvalueReferenceType* self) {
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
}

void SemanticEncoder::writeTypeFunctionType(
    ByteWriter& out, [[maybe_unused]] const cxx::FunctionType* self) {
  // returnType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->returnType())));
  // parameterTypes
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->parameterTypes())));
  for (const auto& element1 : self->parameterTypes()) {
    out.varU32(static_cast<std::uint32_t>(typeRef(element1)));
  }
  // isVariadic
  out.boolean(self->isVariadic());
  // cvQualifiers
  out.varU32(static_cast<std::uint32_t>(self->cvQualifiers()));
  // refQualifier
  out.varU32(static_cast<std::uint32_t>(self->refQualifier()));
  // isNoexcept
  out.boolean(self->isNoexcept());
}

void SemanticEncoder::writeTypeClassType(
    ByteWriter& out, [[maybe_unused]] const cxx::ClassType* self) {
  // symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol())));
}

void SemanticEncoder::writeTypeEnumType(
    ByteWriter& out, [[maybe_unused]] const cxx::EnumType* self) {
  // symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol())));
}

void SemanticEncoder::writeTypeScopedEnumType(
    ByteWriter& out, [[maybe_unused]] const cxx::ScopedEnumType* self) {
  // symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol())));
}

void SemanticEncoder::writeTypeMemberObjectPointerType(
    ByteWriter& out,
    [[maybe_unused]] const cxx::MemberObjectPointerType* self) {
  // classType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->classType())));
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
}

void SemanticEncoder::writeTypeMemberFunctionPointerType(
    ByteWriter& out,
    [[maybe_unused]] const cxx::MemberFunctionPointerType* self) {
  // classType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->classType())));
  // functionType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->functionType())));
}

void SemanticEncoder::writeTypeNamespaceType(
    ByteWriter& out, [[maybe_unused]] const cxx::NamespaceType* self) {
  // symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol())));
}

void SemanticEncoder::writeTypeTypeParameterType(
    ByteWriter& out, [[maybe_unused]] const cxx::TypeParameterType* self) {
  // index
  out.varI32(static_cast<std::int32_t>(self->index()));
  // depth
  out.varI32(static_cast<std::int32_t>(self->depth()));
  // isPack
  out.boolean(self->isParameterPack());
}

void SemanticEncoder::writeTypeTemplateTypeParameterType(
    ByteWriter& out,
    [[maybe_unused]] const cxx::TemplateTypeParameterType* self) {
  // index
  out.varI32(static_cast<std::int32_t>(self->index()));
  // depth
  out.varI32(static_cast<std::int32_t>(self->depth()));
  // isPack
  out.boolean(self->isParameterPack());
  // templateParameters
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->templateParameters())));
  for (const auto& element1 : self->templateParameters()) {
    out.varU32(static_cast<std::uint32_t>(typeRef(element1)));
  }
}

void SemanticEncoder::writeTypeUnresolvedNameType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnresolvedNameType* self) {
  // unit
  // nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier())));
  // unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId())));
}

void SemanticEncoder::writeTypeUnresolvedBoundedArrayType(
    ByteWriter& out,
    [[maybe_unused]] const cxx::UnresolvedBoundedArrayType* self) {
  // unit
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
  // sizeExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->size())));
}

void SemanticEncoder::writeTypeUnresolvedUnderlyingType(
    ByteWriter& out,
    [[maybe_unused]] const cxx::UnresolvedUnderlyingType* self) {
  // unit
  // typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId())));
}

void SemanticEncoder::writeTypeUnresolvedBuiltinType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnresolvedBuiltinType* self) {
  // unit
  // builtinKind
  out.varU32(static_cast<std::uint32_t>(self->builtinKind()));
  // typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId())));
}

void SemanticEncoder::writeTypeOverloadSetType(
    ByteWriter& out, [[maybe_unused]] const cxx::OverloadSetType* self) {
  // symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol())));
}

void SemanticEncoder::writeTypeBuiltinVaListType(
    ByteWriter& out, [[maybe_unused]] const cxx::BuiltinVaListType* self) {}

void SemanticEncoder::writeTypeBuiltinMetaInfoType(
    ByteWriter& out, [[maybe_unused]] const cxx::BuiltinMetaInfoType* self) {}

void SemanticEncoder::writeTypeBitIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::BitIntType* self) {
  // numBits
  out.varI32(static_cast<std::int32_t>(self->numBits()));
}

void SemanticEncoder::writeTypeUnsignedBitIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnsignedBitIntType* self) {
  // numBits
  out.varI32(static_cast<std::int32_t>(self->numBits()));
}

void SemanticEncoder::writeTypeUnresolvedBitIntType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnresolvedBitIntType* self) {
  // unit
  // sizeExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->sizeExpression())));
  // isUnsigned
  out.boolean(self->isUnsigned());
}

void SemanticEncoder::writeTypeVectorType(
    ByteWriter& out, [[maybe_unused]] const cxx::VectorType* self) {
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
  // elementCount
  out.varU64(static_cast<std::uint64_t>(self->elementCount()));
  // vectorKind
  out.varU32(static_cast<std::uint32_t>(self->vectorKind()));
}

void SemanticEncoder::writeTypeUnresolvedVectorType(
    ByteWriter& out, [[maybe_unused]] const cxx::UnresolvedVectorType* self) {
  // unit
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
  // sizeExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->sizeExpression())));
  // vectorKind
  out.varU32(static_cast<std::uint32_t>(self->vectorKind()));
  // sizeKind
  out.varU32(static_cast<std::uint32_t>(self->sizeKind()));
}

void SemanticEncoder::writeTypeComplexType(
    ByteWriter& out, [[maybe_unused]] const cxx::ComplexType* self) {
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
}

void SemanticEncoder::writeTypeAtomicType(
    ByteWriter& out, [[maybe_unused]] const cxx::AtomicType* self) {
  // elementType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->elementType())));
}

void SemanticEncoder::writeSymbolNamespaceSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::NamespaceSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ScopeSymbol::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDirectives())));
  for (const auto& element2 : self->usingDirectives()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
  // ::cxx::NamespaceSymbol::unnamedNamespace_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->unnamedNamespace())));
  // ::cxx::NamespaceSymbol::anonNamespaceIndex_
  out.varI32(
      static_cast<std::int32_t>(self->anonNamespaceIndex().value_or(-1)));
  // ::cxx::NamespaceSymbol::isInline_
  out.boolean(self->isInline());
  // ::cxx::NamespaceSymbol::hasInlineNamespaces_
  out.boolean(self->hasInlineNamespaces());
}

void SemanticEncoder::writeSymbolNamespaceAliasSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::NamespaceAliasSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::NamespaceAliasSymbol::namespaceSymbol_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->namespaceSymbol())));
}

void SemanticEncoder::writeSymbolConceptSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::ConceptSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::MaybeTemplate::declaration_
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration())));
  // ::cxx::MaybeTemplate::templateDeclaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->templateDeclaration())));
  // ::cxx::MaybeTemplate::templateParameters
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->templateParameters())));
  // ::cxx::MaybeTemplate::specializations
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->specializations())));
  for (const auto& element1 : self->specializations()) {
    writecxxTemplateSpecialization(out, &element1);
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->primaryTemplateSymbol())));
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  out.varI32(static_cast<std::int32_t>(self->templateSpecializationIndex()));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->externInstantiationDeclarations())));
  for (const auto& element2 : self->externInstantiationDeclarations()) {
    out.varU32(static_cast<std::uint32_t>(std::ranges::size(element2)));
    for (const auto& element3 : element2) {
      writeTemplateArgument(out, element3);
    }
  }
}

void SemanticEncoder::writeSymbolDeductionGuideSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::DeductionGuideSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::MaybeTemplate::declaration_
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration())));
  // ::cxx::MaybeTemplate::templateDeclaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->templateDeclaration())));
  // ::cxx::MaybeTemplate::templateParameters
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->templateParameters())));
  // ::cxx::MaybeTemplate::specializations
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->specializations())));
  for (const auto& element1 : self->specializations()) {
    writecxxTemplateSpecialization(out, &element1);
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->primaryTemplateSymbol())));
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  out.varI32(static_cast<std::int32_t>(self->templateSpecializationIndex()));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->externInstantiationDeclarations())));
  for (const auto& element2 : self->externInstantiationDeclarations()) {
    out.varU32(static_cast<std::uint32_t>(std::ranges::size(element2)));
    for (const auto& element3 : element2) {
      writeTemplateArgument(out, element3);
    }
  }
  // ::cxx::DeductionGuideSymbol::isExplicit_
  out.boolean(self->isExplicit());
}

void SemanticEncoder::writeSymbolClassSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::ClassSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ScopeSymbol::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDirectives())));
  for (const auto& element2 : self->usingDirectives()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
  // ::cxx::MaybeTemplate::declaration_
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration())));
  // ::cxx::MaybeTemplate::templateDeclaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->templateDeclaration())));
  // ::cxx::MaybeTemplate::templateParameters
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->templateParameters())));
  // ::cxx::MaybeTemplate::specializations
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->specializations())));
  for (const auto& element3 : self->specializations()) {
    writecxxTemplateSpecialization(out, &element3);
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->primaryTemplateSymbol())));
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  out.varI32(static_cast<std::int32_t>(self->templateSpecializationIndex()));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->externInstantiationDeclarations())));
  for (const auto& element4 : self->externInstantiationDeclarations()) {
    out.varU32(static_cast<std::uint32_t>(std::ranges::size(element4)));
    for (const auto& element5 : element4) {
      writeTemplateArgument(out, element5);
    }
  }
  // ::cxx::MaybeRedecl::canonical_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->canonicalOrNull())));
  // ::cxx::MaybeRedecl::definition_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->definition())));
  // ::cxx::MaybeRedecl::redeclarations_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->redeclarations())));
  for (const auto& element6 : self->redeclarations()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element6)));
  }
  // ::cxx::ClassSymbol::baseClasses_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->baseClasses())));
  for (const auto& element7 : self->baseClasses()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element7)));
  }
  // ::cxx::ClassSymbol::befriendingClasses_
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->befriendingClasses())));
  for (const auto& element8 : self->befriendingClasses()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element8)));
  }
  // ::cxx::ClassSymbol::templateFriendships_
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->templateFriendships())));
  for (const auto& element9 : self->templateFriendships()) {
    writecxxTemplateFriendship(out, &element9);
  }
  // ::cxx::ClassSymbol::instantiationPattern_
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->instantiationPattern())));
  // ::cxx::ClassSymbol::instantiationSubstitutionArguments_
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->instantiationSubstitutionArguments())));
  for (const auto& element10 : self->instantiationSubstitutionArguments()) {
    writeTemplateArgument(out, element10);
  }
  // ::cxx::ClassSymbol::instantiationSubstitutionDepth_
  out.varI32(static_cast<std::int32_t>(self->instantiationSubstitutionDepth()));
  // ::cxx::ClassSymbol::constructorOverloadSet_
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->constructorOverloadSet())));
  // ::cxx::ClassSymbol::deductionGuides_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->deductionGuides())));
  for (const auto& element11 : self->deductionGuides()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element11)));
  }
  // ::cxx::ClassSymbol::layout_
  out.boolean(self->layout() != nullptr);
  if (self->layout()) {
    writecxxClassLayout(out, &(*self->layout()));
  }
  // ::cxx::ClassSymbol::vtableLayout_
  out.boolean(self->vtableLayout() != nullptr);
  if (self->vtableLayout()) {
    writecxxVTableLayout(out, &(*self->vtableLayout()));
  }
  // ::cxx::ClassSymbol::capturedThisField_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->capturedThisField())));
  // ::cxx::ClassSymbol::closureDiscriminator_
  out.varI32(static_cast<std::int32_t>(self->closureDiscriminator()));
  // ::cxx::ClassSymbol::sizeInBytes_
  out.varI32(static_cast<std::int32_t>(self->sizeInBytes()));
  // ::cxx::ClassSymbol::alignment_
  out.varI32(static_cast<std::int32_t>(self->alignment()));
  // ::cxx::ClassSymbol::explicitAlignment_
  out.varI32(static_cast<std::int32_t>(self->explicitAlignment()));
  // ::cxx::ClassSymbol::packAlignment_
  out.varI32(static_cast<std::int32_t>(self->packAlignment()));
  // ::cxx::ClassSymbol::isUnion_
  out.varU32(static_cast<std::uint32_t>(self->isUnion()));
  // ::cxx::ClassSymbol::isFinal_
  out.varU32(static_cast<std::uint32_t>(self->isFinal()));
  // ::cxx::ClassSymbol::isComplete_
  out.varU32(static_cast<std::uint32_t>(self->isComplete()));
  // ::cxx::ClassSymbol::isFriend_
  out.varU32(static_cast<std::uint32_t>(self->isFriend()));
  // ::cxx::ClassSymbol::isAccessControlDisabled_
  out.varU32(static_cast<std::uint32_t>(self->isAccessControlDisabled()));
  // ::cxx::ClassSymbol::isPolymorphic_
  out.varU32(static_cast<std::uint32_t>(self->isPolymorphic()));
  // ::cxx::ClassSymbol::isAbstract_
  out.varU32(static_cast<std::uint32_t>(self->isAbstract()));
  // ::cxx::ClassSymbol::hasVirtualDestructor_
  out.varU32(static_cast<std::uint32_t>(self->hasVirtualDestructor()));
  // ::cxx::ClassSymbol::isClosureType_
  out.varU32(static_cast<std::uint32_t>(self->isClosureType()));
  // ::cxx::ClassSymbol::hasLambdaCapture_
  out.varU32(static_cast<std::uint32_t>(self->hasLambdaCapture()));
  // ::cxx::ClassSymbol::hasUserDeclaredConstructors_
  out.varU32(static_cast<std::uint32_t>(self->hasUserDeclaredConstructors()));
}

void SemanticEncoder::writeSymbolEnumSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::EnumSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ScopeSymbol::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDirectives())));
  for (const auto& element2 : self->usingDirectives()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
  // ::cxx::EnumSymbol::underlyingType_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->underlyingType())));
  // ::cxx::EnumSymbol::hasFixedUnderlyingType_
  out.boolean(self->hasFixedUnderlyingType());
  // ::cxx::EnumSymbol::isDefined_
  out.boolean(self->isDefined());
}

void SemanticEncoder::writeSymbolScopedEnumSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::ScopedEnumSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ScopeSymbol::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDirectives())));
  for (const auto& element2 : self->usingDirectives()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
  // ::cxx::ScopedEnumSymbol::underlyingType_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->underlyingType())));
  // ::cxx::ScopedEnumSymbol::isDefined_
  out.boolean(self->isDefined());
}

void SemanticEncoder::writeSymbolFunctionSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::FunctionSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ScopeSymbol::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDirectives())));
  for (const auto& element2 : self->usingDirectives()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
  // ::cxx::MaybeTemplate::declaration_
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration())));
  // ::cxx::MaybeTemplate::templateDeclaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->templateDeclaration())));
  // ::cxx::MaybeTemplate::templateParameters
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->templateParameters())));
  // ::cxx::MaybeTemplate::specializations
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->specializations())));
  for (const auto& element3 : self->specializations()) {
    writecxxTemplateSpecialization(out, &element3);
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->primaryTemplateSymbol())));
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  out.varI32(static_cast<std::int32_t>(self->templateSpecializationIndex()));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->externInstantiationDeclarations())));
  for (const auto& element4 : self->externInstantiationDeclarations()) {
    out.varU32(static_cast<std::uint32_t>(std::ranges::size(element4)));
    for (const auto& element5 : element4) {
      writeTemplateArgument(out, element5);
    }
  }
  // ::cxx::MaybeRedecl::canonical_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->canonicalOrNull())));
  // ::cxx::MaybeRedecl::definition_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->definition())));
  // ::cxx::MaybeRedecl::redeclarations_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->redeclarations())));
  for (const auto& element6 : self->redeclarations()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element6)));
  }
  // ::cxx::FunctionSymbol::pendingBody_
  out.boolean(self->pendingBody() != nullptr);
  if (self->pendingBody()) {
    writecxxPendingBodyInstantiation(out, &(*self->pendingBody()));
  }
  // ::cxx::FunctionSymbol::pendingExceptionSpecification_
  out.boolean(self->pendingExceptionSpecification() != nullptr);
  if (self->pendingExceptionSpecification()) {
    writecxxPendingExceptionSpecification(
        out, &(*self->pendingExceptionSpecification()));
  }
  // ::cxx::FunctionSymbol::completeObjectVariant_
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->completeObjectVariant())));
  // ::cxx::FunctionSymbol::delegatingConstructor_
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->delegatingConstructor())));
  // ::cxx::FunctionSymbol::deletingDtorVariant_
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->deletingDtorVariant())));
  // ::cxx::FunctionSymbol::structorPrincipal_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->structorPrincipal())));
  // ::cxx::FunctionSymbol::inheritedConstructor_
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->inheritedConstructor())));
  // ::cxx::FunctionSymbol::overriddenFunctions_
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->overriddenFunctions())));
  for (const auto& element7 : self->overriddenFunctions()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element7)));
  }
  // ::cxx::FunctionSymbol::befriendingClasses_
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->befriendingClasses())));
  for (const auto& element8 : self->befriendingClasses()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element8)));
  }
  // ::cxx::FunctionSymbol::templateFriendships_
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->templateFriendships())));
  for (const auto& element9 : self->templateFriendships()) {
    writecxxTemplateFriendship(out, &element9);
  }
  // ::cxx::FunctionSymbol::vtableSlotIndex_
  out.varI32(static_cast<std::int32_t>(self->vtableSlotIndex()));
  // ::cxx::FunctionSymbol::externalName_
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->externalName())));
  // ::cxx::FunctionSymbol::aliasName_
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->aliasName())));
  // ::cxx::FunctionSymbol::importModule_
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->importModule())));
  // ::cxx::FunctionSymbol::importName_
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->importName())));
  // ::cxx::FunctionSymbol::exportName_
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->exportName())));
  // ::cxx::FunctionSymbol::trailingRequiresClause_
  out.varU32(
      static_cast<std::uint32_t>(astRef(self->trailingRequiresClause())));
  // ::cxx::FunctionSymbol::builtinKind_
  out.varU32(static_cast<std::uint32_t>(self->builtinKind()));
  // ::cxx::FunctionSymbol::isDefined_
  out.varU32(static_cast<std::uint32_t>(self->isDefined()));
  // ::cxx::FunctionSymbol::isStatic_
  out.varU32(static_cast<std::uint32_t>(self->isStatic()));
  // ::cxx::FunctionSymbol::isExtern_
  out.varU32(static_cast<std::uint32_t>(self->isExtern()));
  // ::cxx::FunctionSymbol::isFriend_
  out.varU32(static_cast<std::uint32_t>(self->isFriend()));
  // ::cxx::FunctionSymbol::isConstexpr_
  out.varU32(static_cast<std::uint32_t>(self->isConstexpr()));
  // ::cxx::FunctionSymbol::isConsteval_
  out.varU32(static_cast<std::uint32_t>(self->isConsteval()));
  // ::cxx::FunctionSymbol::isInline_
  out.varU32(static_cast<std::uint32_t>(self->isInline()));
  // ::cxx::FunctionSymbol::isVirtual_
  out.varU32(static_cast<std::uint32_t>(self->isVirtual()));
  // ::cxx::FunctionSymbol::isExplicit_
  out.varU32(static_cast<std::uint32_t>(self->isExplicit()));
  // ::cxx::FunctionSymbol::isDeleted_
  out.varU32(static_cast<std::uint32_t>(self->isDeleted()));
  // ::cxx::FunctionSymbol::isDefaulted_
  out.varU32(static_cast<std::uint32_t>(self->isDefaulted()));
  // ::cxx::FunctionSymbol::isPure_
  out.varU32(static_cast<std::uint32_t>(self->isPure()));
  // ::cxx::FunctionSymbol::hasCLinkage_
  out.varU32(static_cast<std::uint32_t>(self->hasCLinkage()));
  // ::cxx::FunctionSymbol::isOverride_
  out.varU32(static_cast<std::uint32_t>(self->isOverride()));
  // ::cxx::FunctionSymbol::isFinal_
  out.varU32(static_cast<std::uint32_t>(self->isFinal()));
  // ::cxx::FunctionSymbol::hasNoPrototype_
  out.varU32(static_cast<std::uint32_t>(self->hasNoPrototype()));
  // ::cxx::FunctionSymbol::hasHiddenVisibility_
  out.varU32(static_cast<std::uint32_t>(self->hasHiddenVisibility()));
  // ::cxx::FunctionSymbol::hasExceptionSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->hasExceptionSpecifier()));
  // ::cxx::FunctionSymbol::isDefinitionRequired_
  out.varU32(static_cast<std::uint32_t>(self->isDefinitionRequired()));
  // ::cxx::FunctionSymbol::hasExplicitObjectParameter_
  out.varU32(static_cast<std::uint32_t>(self->hasExplicitObjectParameter()));
  // ::cxx::FunctionSymbol::isNoReturn_
  out.varU32(static_cast<std::uint32_t>(self->isNoReturn()));
}

void SemanticEncoder::writeSymbolTypeAliasSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::TypeAliasSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::MaybeTemplate::declaration_
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration())));
  // ::cxx::MaybeTemplate::templateDeclaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->templateDeclaration())));
  // ::cxx::MaybeTemplate::templateParameters
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->templateParameters())));
  // ::cxx::MaybeTemplate::specializations
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->specializations())));
  for (const auto& element1 : self->specializations()) {
    writecxxTemplateSpecialization(out, &element1);
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->primaryTemplateSymbol())));
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  out.varI32(static_cast<std::int32_t>(self->templateSpecializationIndex()));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->externInstantiationDeclarations())));
  for (const auto& element2 : self->externInstantiationDeclarations()) {
    out.varU32(static_cast<std::uint32_t>(std::ranges::size(element2)));
    for (const auto& element3 : element2) {
      writeTemplateArgument(out, element3);
    }
  }
  // ::cxx::MaybeRedecl::canonical_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->canonicalOrNull())));
  // ::cxx::MaybeRedecl::definition_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->definition())));
  // ::cxx::MaybeRedecl::redeclarations_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->redeclarations())));
  for (const auto& element4 : self->redeclarations()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element4)));
  }
  // ::cxx::TypeAliasSymbol::expansionTypeId_
  out.varU32(static_cast<std::uint32_t>(astRef(self->expansionTypeId())));
}

void SemanticEncoder::writeSymbolVariableSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::VariableSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::MaybeTemplate::declaration_
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration())));
  // ::cxx::MaybeTemplate::templateDeclaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->templateDeclaration())));
  // ::cxx::MaybeTemplate::templateParameters
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->templateParameters())));
  // ::cxx::MaybeTemplate::specializations
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->specializations())));
  for (const auto& element1 : self->specializations()) {
    writecxxTemplateSpecialization(out, &element1);
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->primaryTemplateSymbol())));
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  out.varI32(static_cast<std::int32_t>(self->templateSpecializationIndex()));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  out.varU32(static_cast<std::uint32_t>(
      std::ranges::size(self->externInstantiationDeclarations())));
  for (const auto& element2 : self->externInstantiationDeclarations()) {
    out.varU32(static_cast<std::uint32_t>(std::ranges::size(element2)));
    for (const auto& element3 : element2) {
      writeTemplateArgument(out, element3);
    }
  }
  // ::cxx::MaybeRedecl::canonical_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->canonicalOrNull())));
  // ::cxx::MaybeRedecl::definition_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->definition())));
  // ::cxx::MaybeRedecl::redeclarations_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->redeclarations())));
  for (const auto& element4 : self->redeclarations()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element4)));
  }
  // ::cxx::VariableSymbol::initializer_
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer())));
  // ::cxx::VariableSymbol::constructor_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructor())));
  // ::cxx::VariableSymbol::constValue_
  out.boolean(self->constValue().has_value());
  if (self->constValue().has_value()) {
    writeConstValue(out, (*self->constValue()));
  }
  // ::cxx::VariableSymbol::explicitAlignment_
  out.varI32(static_cast<std::int32_t>(self->explicitAlignment()));
  // ::cxx::VariableSymbol::isStatic_
  out.varU32(static_cast<std::uint32_t>(self->isStatic()));
  // ::cxx::VariableSymbol::isThreadLocal_
  out.varU32(static_cast<std::uint32_t>(self->isThreadLocal()));
  // ::cxx::VariableSymbol::isExtern_
  out.varU32(static_cast<std::uint32_t>(self->isExtern()));
  // ::cxx::VariableSymbol::isConstexpr_
  out.varU32(static_cast<std::uint32_t>(self->isConstexpr()));
  // ::cxx::VariableSymbol::isConstinit_
  out.varU32(static_cast<std::uint32_t>(self->isConstinit()));
  // ::cxx::VariableSymbol::isInline_
  out.varU32(static_cast<std::uint32_t>(self->isInline()));
}

void SemanticEncoder::writeSymbolFieldSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::FieldSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::FieldSymbol::definition_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->definition())));
  // ::cxx::FieldSymbol::pendingInitializer_
  out.boolean(self->pendingInitializer() != nullptr);
  if (self->pendingInitializer()) {
    writecxxPendingFieldInitializerInstantiation(
        out, &(*self->pendingInitializer()));
  }
  // ::cxx::FieldSymbol::constValue_
  out.boolean(self->constValue().has_value());
  if (self->constValue().has_value()) {
    writeConstValue(out, (*self->constValue()));
  }
  // ::cxx::FieldSymbol::isDefinitionRequired_
  out.varU32(static_cast<std::uint32_t>(self->isDefinitionRequired()));
  // ::cxx::FieldSymbol::isBitField_
  out.varU32(static_cast<std::uint32_t>(self->isBitField()));
  // ::cxx::FieldSymbol::isStatic_
  out.varU32(static_cast<std::uint32_t>(self->isStatic()));
  // ::cxx::FieldSymbol::isThreadLocal_
  out.varU32(static_cast<std::uint32_t>(self->isThreadLocal()));
  // ::cxx::FieldSymbol::isConstexpr_
  out.varU32(static_cast<std::uint32_t>(self->isConstexpr()));
  // ::cxx::FieldSymbol::isConstinit_
  out.varU32(static_cast<std::uint32_t>(self->isConstinit()));
  // ::cxx::FieldSymbol::isInline_
  out.varU32(static_cast<std::uint32_t>(self->isInline()));
  // ::cxx::FieldSymbol::isMutable_
  out.varU32(static_cast<std::uint32_t>(self->isMutable()));
  // ::cxx::FieldSymbol::isNoUniqueAddress_
  out.varU32(static_cast<std::uint32_t>(self->isNoUniqueAddress()));
  // ::cxx::FieldSymbol::localOffset_
  out.varI32(static_cast<std::int32_t>(self->localOffset()));
  // ::cxx::FieldSymbol::alignment_
  out.varI32(static_cast<std::int32_t>(self->alignment()));
  // ::cxx::FieldSymbol::bitFieldOffset_
  out.varI32(static_cast<std::int32_t>(self->bitFieldOffset()));
  // ::cxx::FieldSymbol::bitFieldWidth_
  out.boolean(self->bitFieldWidth().has_value());
  if (self->bitFieldWidth().has_value()) {
    writeConstValue(out, (*self->bitFieldWidth()));
  }
  // ::cxx::FieldSymbol::initializer_
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer())));
  // ::cxx::FieldSymbol::constructor_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructor())));
}

void SemanticEncoder::writeSymbolParameterSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::ParameterSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ParameterSymbol::defaultArgument_
  out.varU32(static_cast<std::uint32_t>(astRef(self->defaultArgument())));
  // ::cxx::ParameterSymbol::isExplicitObject_
  out.boolean(self->isExplicitObject());
}

void SemanticEncoder::writeSymbolParameterPackSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::ParameterPackSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ParameterPackSymbol::elements_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->elements())));
  for (const auto& element1 : self->elements()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
}

void SemanticEncoder::writeSymbolEnumeratorSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::EnumeratorSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::EnumeratorSymbol::value_
  out.boolean(self->value().has_value());
  if (self->value().has_value()) {
    writeConstValue(out, (*self->value()));
  }
}

void SemanticEncoder::writeSymbolFunctionParametersSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::FunctionParametersSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ScopeSymbol::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDirectives())));
  for (const auto& element2 : self->usingDirectives()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
}

void SemanticEncoder::writeSymbolTemplateParametersSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::TemplateParametersSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ScopeSymbol::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDirectives())));
  for (const auto& element2 : self->usingDirectives()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
  // ::cxx::TemplateParametersSymbol::isExplicitTemplateSpecialization_
  out.boolean(self->isExplicitTemplateSpecialization());
}

void SemanticEncoder::writeSymbolBlockSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::BlockSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ScopeSymbol::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDirectives())));
  for (const auto& element2 : self->usingDirectives()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
  // ::cxx::BlockSymbol::isOutermostBlockScope_
  out.boolean(self->isOutermostBlockScope());
}

void SemanticEncoder::writeSymbolLambdaSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::LambdaSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ScopeSymbol::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDirectives())));
  for (const auto& element2 : self->usingDirectives()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
  // ::cxx::LambdaSymbol::closureType_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->closureType())));
  // ::cxx::LambdaSymbol::isConstexpr_
  out.varU32(static_cast<std::uint32_t>(self->isConstexpr()));
  // ::cxx::LambdaSymbol::isConsteval_
  out.varU32(static_cast<std::uint32_t>(self->isConsteval()));
  // ::cxx::LambdaSymbol::isMutable_
  out.varU32(static_cast<std::uint32_t>(self->isMutable()));
  // ::cxx::LambdaSymbol::isStatic_
  out.varU32(static_cast<std::uint32_t>(self->isStatic()));
  // ::cxx::LambdaSymbol::isTemplate_
  out.varU32(static_cast<std::uint32_t>(self->isTemplate()));
  // ::cxx::LambdaSymbol::isInTemplate_
  out.varU32(static_cast<std::uint32_t>(self->isInTemplate()));
}

void SemanticEncoder::writeSymbolTypeParameterSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::TypeParameterSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
}

void SemanticEncoder::writeSymbolNonTypeParameterSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::NonTypeParameterSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::NonTypeParameterSymbol::objectType_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->objectType())));
  // ::cxx::NonTypeParameterSymbol::index_
  out.varI32(static_cast<std::int32_t>(self->index()));
  // ::cxx::NonTypeParameterSymbol::depth_
  out.varI32(static_cast<std::int32_t>(self->depth()));
  // ::cxx::NonTypeParameterSymbol::isParameterPack_
  out.boolean(self->isParameterPack());
}

void SemanticEncoder::writeSymbolTemplateTypeParameterSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::TemplateTypeParameterSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
}

void SemanticEncoder::writeSymbolConstraintTypeParameterSymbol(
    ByteWriter& out,
    [[maybe_unused]] cxx::ConstraintTypeParameterSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::ConstraintTypeParameterSymbol::index_
  out.varI32(static_cast<std::int32_t>(self->index()));
  // ::cxx::ConstraintTypeParameterSymbol::depth_
  out.varI32(static_cast<std::int32_t>(self->depth()));
  // ::cxx::ConstraintTypeParameterSymbol::isParameterPack_
  out.boolean(self->isParameterPack());
  // ::cxx::ConstraintTypeParameterSymbol::typeConstraint_
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeConstraint())));
  // ::cxx::ConstraintTypeParameterSymbol::constraintExpression_
  out.varU32(static_cast<std::uint32_t>(astRef(self->constraintExpression())));
}

void SemanticEncoder::writeSymbolOverloadSetSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::OverloadSetSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::OverloadSetSymbol::declaredFunctions_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->declaredFunctions())));
  for (const auto& element1 : self->declaredFunctions()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1)));
  }
  // ::cxx::OverloadSetSymbol::usingDeclarations_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->usingDeclarations())));
  for (const auto& element2 : self->usingDeclarations()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2)));
  }
}

void SemanticEncoder::writeSymbolBaseClassSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::BaseClassSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::BaseClassSymbol::symbol_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol())));
  // ::cxx::BaseClassSymbol::isVirtual_
  out.boolean(self->isVirtual());
}

void SemanticEncoder::writeSymbolInjectedClassNameSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::InjectedClassNameSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::InjectedClassNameSymbol::classSymbol_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->classSymbol())));
}

void SemanticEncoder::writeSymbolUnresolvedSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::UnresolvedSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
}

void SemanticEncoder::writeSymbolUsingDeclarationSymbol(
    ByteWriter& out, [[maybe_unused]] cxx::UsingDeclarationSymbol* self) {
  // ::cxx::Symbol::name_
  out.varU32(static_cast<std::uint32_t>(nameRef(self->name())));
  // ::cxx::Symbol::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::Symbol::parent_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parent())));
  // ::cxx::Symbol::abiTags_
  writeAbiTags(out, self->abiTagList());
  // ::cxx::Symbol::attributes_
  writeAttributes(out, self->attributes());
  // ::cxx::Symbol::location_
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location())));
  // ::cxx::Symbol::isHidden_
  out.boolean(self->isHidden());
  // ::cxx::Symbol::isNodiscard_
  out.boolean(self->isNodiscard());
  // ::cxx::Symbol::isUsed_
  out.boolean(self->isUsed());
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  out.boolean(self->isExcludedFromExplicitInstantiation());
  // ::cxx::Symbol::isTrivialAbi_
  out.boolean(self->isTrivialAbi());
  // ::cxx::Symbol::hasDeducedReturnType_
  out.boolean(self->hasDeducedReturnType());
  // ::cxx::Symbol::accessSpecifier_
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier()));
  // ::cxx::UsingDeclarationSymbol::target_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->target())));
  // ::cxx::UsingDeclarationSymbol::declarator_
  out.varU32(static_cast<std::uint32_t>(astRef(self->declarator())));
}

void SemanticEncoder::writeAstTranslationUnitAST(
    ByteWriter& out, [[maybe_unused]] cxx::TranslationUnitAST* self) {
  // ::cxx::UnitAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::TranslationUnitAST::declarationList
  writeAstList(out, self->declarationList);
}

void SemanticEncoder::writeAstModuleUnitAST(
    ByteWriter& out, [[maybe_unused]] cxx::ModuleUnitAST* self) {
  // ::cxx::UnitAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::ModuleUnitAST::globalModuleFragment
  out.varU32(static_cast<std::uint32_t>(astRef(self->globalModuleFragment)));
  // ::cxx::ModuleUnitAST::moduleDeclaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->moduleDeclaration)));
  // ::cxx::ModuleUnitAST::declarationList
  writeAstList(out, self->declarationList);
  // ::cxx::ModuleUnitAST::privateModuleFragment
  out.varU32(static_cast<std::uint32_t>(astRef(self->privateModuleFragment)));
}

void SemanticEncoder::writeAstSimpleDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::SimpleDeclarationAST* self) {
  // ::cxx::SimpleDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::SimpleDeclarationAST::declSpecifierList
  writeAstList(out, self->declSpecifierList);
  // ::cxx::SimpleDeclarationAST::initDeclaratorList
  writeAstList(out, self->initDeclaratorList);
  // ::cxx::SimpleDeclarationAST::requiresClause
  out.varU32(static_cast<std::uint32_t>(astRef(self->requiresClause)));
  // ::cxx::SimpleDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstAsmDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::AsmDeclarationAST* self) {
  // ::cxx::AsmDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::AsmDeclarationAST::asmQualifierList
  writeAstList(out, self->asmQualifierList);
  // ::cxx::AsmDeclarationAST::asmLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->asmLoc)));
  // ::cxx::AsmDeclarationAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::AsmDeclarationAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::AsmDeclarationAST::outputOperandList
  writeAstList(out, self->outputOperandList);
  // ::cxx::AsmDeclarationAST::inputOperandList
  writeAstList(out, self->inputOperandList);
  // ::cxx::AsmDeclarationAST::clobberList
  writeAstList(out, self->clobberList);
  // ::cxx::AsmDeclarationAST::gotoLabelList
  writeAstList(out, self->gotoLabelList);
  // ::cxx::AsmDeclarationAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::AsmDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::AsmDeclarationAST::literal
  writeLiteral(out, self->literal);
}

void SemanticEncoder::writeAstNamespaceAliasDefinitionAST(
    ByteWriter& out, [[maybe_unused]] cxx::NamespaceAliasDefinitionAST* self) {
  // ::cxx::NamespaceAliasDefinitionAST::namespaceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->namespaceLoc)));
  // ::cxx::NamespaceAliasDefinitionAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::NamespaceAliasDefinitionAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::NamespaceAliasDefinitionAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::NamespaceAliasDefinitionAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::NamespaceAliasDefinitionAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::NamespaceAliasDefinitionAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::NamespaceAliasDefinitionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstUsingDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::UsingDeclarationAST* self) {
  // ::cxx::UsingDeclarationAST::usingLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->usingLoc)));
  // ::cxx::UsingDeclarationAST::usingDeclaratorList
  writeAstList(out, self->usingDeclaratorList);
  // ::cxx::UsingDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstUsingEnumDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::UsingEnumDeclarationAST* self) {
  // ::cxx::UsingEnumDeclarationAST::usingLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->usingLoc)));
  // ::cxx::UsingEnumDeclarationAST::enumTypeSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->enumTypeSpecifier)));
  // ::cxx::UsingEnumDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstUsingDirectiveAST(
    ByteWriter& out, [[maybe_unused]] cxx::UsingDirectiveAST* self) {
  // ::cxx::UsingDirectiveAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::UsingDirectiveAST::usingLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->usingLoc)));
  // ::cxx::UsingDirectiveAST::namespaceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->namespaceLoc)));
  // ::cxx::UsingDirectiveAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::UsingDirectiveAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::UsingDirectiveAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstStaticAssertDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::StaticAssertDeclarationAST* self) {
  // ::cxx::StaticAssertDeclarationAST::staticAssertLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->staticAssertLoc)));
  // ::cxx::StaticAssertDeclarationAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::StaticAssertDeclarationAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::StaticAssertDeclarationAST::commaLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->commaLoc)));
  // ::cxx::StaticAssertDeclarationAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::StaticAssertDeclarationAST::literal
  writeLiteral(out, self->literal);
  // ::cxx::StaticAssertDeclarationAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::StaticAssertDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::StaticAssertDeclarationAST::value
  out.boolean(self->value.has_value());
  if (self->value.has_value()) {
    out.boolean((*self->value));
  }
}

void SemanticEncoder::writeAstAliasDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::AliasDeclarationAST* self) {
  // ::cxx::AliasDeclarationAST::usingLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->usingLoc)));
  // ::cxx::AliasDeclarationAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::AliasDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::AliasDeclarationAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::AliasDeclarationAST::gnuAttributeList
  writeAstList(out, self->gnuAttributeList);
  // ::cxx::AliasDeclarationAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::AliasDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::AliasDeclarationAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::AliasDeclarationAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstOpaqueEnumDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::OpaqueEnumDeclarationAST* self) {
  // ::cxx::OpaqueEnumDeclarationAST::enumLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->enumLoc)));
  // ::cxx::OpaqueEnumDeclarationAST::classLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->classLoc)));
  // ::cxx::OpaqueEnumDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::OpaqueEnumDeclarationAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::OpaqueEnumDeclarationAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::OpaqueEnumDeclarationAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::OpaqueEnumDeclarationAST::typeSpecifierList
  writeAstList(out, self->typeSpecifierList);
  // ::cxx::OpaqueEnumDeclarationAST::emicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->emicolonLoc)));
  // ::cxx::OpaqueEnumDeclarationAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstFunctionDefinitionAST(
    ByteWriter& out, [[maybe_unused]] cxx::FunctionDefinitionAST* self) {
  // ::cxx::FunctionDefinitionAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::FunctionDefinitionAST::declSpecifierList
  writeAstList(out, self->declSpecifierList);
  // ::cxx::FunctionDefinitionAST::declarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->declarator)));
  // ::cxx::FunctionDefinitionAST::requiresClause
  out.varU32(static_cast<std::uint32_t>(astRef(self->requiresClause)));
  // ::cxx::FunctionDefinitionAST::functionBody
  out.varU32(static_cast<std::uint32_t>(astRef(self->functionBody)));
  // ::cxx::FunctionDefinitionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstTemplateDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::TemplateDeclarationAST* self) {
  // ::cxx::TemplateDeclarationAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::TemplateDeclarationAST::lessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lessLoc)));
  // ::cxx::TemplateDeclarationAST::templateParameterList
  writeAstList(out, self->templateParameterList);
  // ::cxx::TemplateDeclarationAST::greaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->greaterLoc)));
  // ::cxx::TemplateDeclarationAST::requiresClause
  out.varU32(static_cast<std::uint32_t>(astRef(self->requiresClause)));
  // ::cxx::TemplateDeclarationAST::declaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration)));
  // ::cxx::TemplateDeclarationAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::TemplateDeclarationAST::depth
  out.varI32(static_cast<std::int32_t>(self->depth));
}

void SemanticEncoder::writeAstConceptDefinitionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConceptDefinitionAST* self) {
  // ::cxx::ConceptDefinitionAST::conceptLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->conceptLoc)));
  // ::cxx::ConceptDefinitionAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::ConceptDefinitionAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::ConceptDefinitionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::ConceptDefinitionAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::ConceptDefinitionAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::ConceptDefinitionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstDeductionGuideAST(
    ByteWriter& out, [[maybe_unused]] cxx::DeductionGuideAST* self) {
  // ::cxx::DeductionGuideAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::DeductionGuideAST::explicitSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->explicitSpecifier)));
  // ::cxx::DeductionGuideAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::DeductionGuideAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::DeductionGuideAST::parameterDeclarationClause
  out.varU32(
      static_cast<std::uint32_t>(astRef(self->parameterDeclarationClause)));
  // ::cxx::DeductionGuideAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::DeductionGuideAST::arrowLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->arrowLoc)));
  // ::cxx::DeductionGuideAST::templateId
  out.varU32(static_cast<std::uint32_t>(astRef(self->templateId)));
  // ::cxx::DeductionGuideAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::DeductionGuideAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::DeductionGuideAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstExplicitInstantiationAST(
    ByteWriter& out, [[maybe_unused]] cxx::ExplicitInstantiationAST* self) {
  // ::cxx::ExplicitInstantiationAST::externLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->externLoc)));
  // ::cxx::ExplicitInstantiationAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::ExplicitInstantiationAST::declaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration)));
}

void SemanticEncoder::writeAstExportDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::ExportDeclarationAST* self) {
  // ::cxx::ExportDeclarationAST::exportLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->exportLoc)));
  // ::cxx::ExportDeclarationAST::declaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration)));
}

void SemanticEncoder::writeAstExportCompoundDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::ExportCompoundDeclarationAST* self) {
  // ::cxx::ExportCompoundDeclarationAST::exportLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->exportLoc)));
  // ::cxx::ExportCompoundDeclarationAST::lbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbraceLoc)));
  // ::cxx::ExportCompoundDeclarationAST::declarationList
  writeAstList(out, self->declarationList);
  // ::cxx::ExportCompoundDeclarationAST::rbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbraceLoc)));
}

void SemanticEncoder::writeAstLinkageSpecificationAST(
    ByteWriter& out, [[maybe_unused]] cxx::LinkageSpecificationAST* self) {
  // ::cxx::LinkageSpecificationAST::externLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->externLoc)));
  // ::cxx::LinkageSpecificationAST::stringliteralLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->stringliteralLoc)));
  // ::cxx::LinkageSpecificationAST::lbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbraceLoc)));
  // ::cxx::LinkageSpecificationAST::declarationList
  writeAstList(out, self->declarationList);
  // ::cxx::LinkageSpecificationAST::rbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbraceLoc)));
  // ::cxx::LinkageSpecificationAST::stringLiteral
  writeLiteral(out, self->stringLiteral);
}

void SemanticEncoder::writeAstNamespaceDefinitionAST(
    ByteWriter& out, [[maybe_unused]] cxx::NamespaceDefinitionAST* self) {
  // ::cxx::NamespaceDefinitionAST::inlineLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->inlineLoc)));
  // ::cxx::NamespaceDefinitionAST::namespaceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->namespaceLoc)));
  // ::cxx::NamespaceDefinitionAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::NamespaceDefinitionAST::nestedNamespaceSpecifierList
  writeAstList(out, self->nestedNamespaceSpecifierList);
  // ::cxx::NamespaceDefinitionAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::NamespaceDefinitionAST::extraAttributeList
  writeAstList(out, self->extraAttributeList);
  // ::cxx::NamespaceDefinitionAST::lbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbraceLoc)));
  // ::cxx::NamespaceDefinitionAST::declarationList
  writeAstList(out, self->declarationList);
  // ::cxx::NamespaceDefinitionAST::rbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbraceLoc)));
  // ::cxx::NamespaceDefinitionAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::NamespaceDefinitionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::NamespaceDefinitionAST::isInline
  out.boolean(self->isInline);
}

void SemanticEncoder::writeAstEmptyDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::EmptyDeclarationAST* self) {
  // ::cxx::EmptyDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstAttributeDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::AttributeDeclarationAST* self) {
  // ::cxx::AttributeDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::AttributeDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstModuleImportDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::ModuleImportDeclarationAST* self) {
  // ::cxx::ModuleImportDeclarationAST::importLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->importLoc)));
  // ::cxx::ModuleImportDeclarationAST::importName
  out.varU32(static_cast<std::uint32_t>(astRef(self->importName)));
  // ::cxx::ModuleImportDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ModuleImportDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstParameterDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::ParameterDeclarationAST* self) {
  // ::cxx::ParameterDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ParameterDeclarationAST::thisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->thisLoc)));
  // ::cxx::ParameterDeclarationAST::typeSpecifierList
  writeAstList(out, self->typeSpecifierList);
  // ::cxx::ParameterDeclarationAST::declarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->declarator)));
  // ::cxx::ParameterDeclarationAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::ParameterDeclarationAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::ParameterDeclarationAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ParameterDeclarationAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::ParameterDeclarationAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::ParameterDeclarationAST::isThisIntroduced
  out.boolean(self->isThisIntroduced);
  // ::cxx::ParameterDeclarationAST::isPack
  out.boolean(self->isPack);
}

void SemanticEncoder::writeAstAccessDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::AccessDeclarationAST* self) {
  // ::cxx::AccessDeclarationAST::accessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->accessLoc)));
  // ::cxx::AccessDeclarationAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::AccessDeclarationAST::accessSpecifier
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier));
}

void SemanticEncoder::writeAstForRangeDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::ForRangeDeclarationAST* self) {}

void SemanticEncoder::writeAstStructuredBindingDeclarationAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::StructuredBindingDeclarationAST* self) {
  // ::cxx::StructuredBindingDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::StructuredBindingDeclarationAST::declSpecifierList
  writeAstList(out, self->declSpecifierList);
  // ::cxx::StructuredBindingDeclarationAST::refQualifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->refQualifierLoc)));
  // ::cxx::StructuredBindingDeclarationAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::StructuredBindingDeclarationAST::bindingList
  writeAstList(out, self->bindingList);
  // ::cxx::StructuredBindingDeclarationAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
  // ::cxx::StructuredBindingDeclarationAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::StructuredBindingDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::StructuredBindingDeclarationAST::hiddenVariable
  out.varU32(static_cast<std::uint32_t>(astRef(self->hiddenVariable)));
  // ::cxx::StructuredBindingDeclarationAST::bindingDeclaratorList
  writeAstList(out, self->bindingDeclaratorList);
}

void SemanticEncoder::writeAstAsmOperandAST(
    ByteWriter& out, [[maybe_unused]] cxx::AsmOperandAST* self) {
  // ::cxx::AsmOperandAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::AsmOperandAST::symbolicNameLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->symbolicNameLoc)));
  // ::cxx::AsmOperandAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
  // ::cxx::AsmOperandAST::constraintLiteralLoc
  out.varU32(
      static_cast<std::uint32_t>(locationRef(self->constraintLiteralLoc)));
  // ::cxx::AsmOperandAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::AsmOperandAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::AsmOperandAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::AsmOperandAST::symbolicName
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->symbolicName)));
  // ::cxx::AsmOperandAST::constraintLiteral
  writeLiteral(out, self->constraintLiteral);
}

void SemanticEncoder::writeAstAsmQualifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::AsmQualifierAST* self) {
  // ::cxx::AsmQualifierAST::qualifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->qualifierLoc)));
  // ::cxx::AsmQualifierAST::qualifier
  out.varU32(static_cast<std::uint32_t>(self->qualifier));
}

void SemanticEncoder::writeAstAsmClobberAST(
    ByteWriter& out, [[maybe_unused]] cxx::AsmClobberAST* self) {
  // ::cxx::AsmClobberAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::AsmClobberAST::literal
  writeLiteral(out, self->literal);
}

void SemanticEncoder::writeAstAsmGotoLabelAST(
    ByteWriter& out, [[maybe_unused]] cxx::AsmGotoLabelAST* self) {
  // ::cxx::AsmGotoLabelAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::AsmGotoLabelAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writeAstSplicerAST(
    ByteWriter& out, [[maybe_unused]] cxx::SplicerAST* self) {
  // ::cxx::SplicerAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::SplicerAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::SplicerAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::SplicerAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::SplicerAST::secondColonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->secondColonLoc)));
  // ::cxx::SplicerAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
}

void SemanticEncoder::writeAstGlobalModuleFragmentAST(
    ByteWriter& out, [[maybe_unused]] cxx::GlobalModuleFragmentAST* self) {
  // ::cxx::GlobalModuleFragmentAST::moduleLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->moduleLoc)));
  // ::cxx::GlobalModuleFragmentAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::GlobalModuleFragmentAST::declarationList
  writeAstList(out, self->declarationList);
}

void SemanticEncoder::writeAstPrivateModuleFragmentAST(
    ByteWriter& out, [[maybe_unused]] cxx::PrivateModuleFragmentAST* self) {
  // ::cxx::PrivateModuleFragmentAST::moduleLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->moduleLoc)));
  // ::cxx::PrivateModuleFragmentAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::PrivateModuleFragmentAST::privateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->privateLoc)));
  // ::cxx::PrivateModuleFragmentAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::PrivateModuleFragmentAST::declarationList
  writeAstList(out, self->declarationList);
}

void SemanticEncoder::writeAstModuleDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::ModuleDeclarationAST* self) {
  // ::cxx::ModuleDeclarationAST::exportLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->exportLoc)));
  // ::cxx::ModuleDeclarationAST::moduleLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->moduleLoc)));
  // ::cxx::ModuleDeclarationAST::moduleName
  out.varU32(static_cast<std::uint32_t>(astRef(self->moduleName)));
  // ::cxx::ModuleDeclarationAST::modulePartition
  out.varU32(static_cast<std::uint32_t>(astRef(self->modulePartition)));
  // ::cxx::ModuleDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ModuleDeclarationAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstModuleNameAST(
    ByteWriter& out, [[maybe_unused]] cxx::ModuleNameAST* self) {
  // ::cxx::ModuleNameAST::moduleQualifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->moduleQualifier)));
  // ::cxx::ModuleNameAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::ModuleNameAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writeAstModuleQualifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ModuleQualifierAST* self) {
  // ::cxx::ModuleQualifierAST::moduleQualifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->moduleQualifier)));
  // ::cxx::ModuleQualifierAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::ModuleQualifierAST::dotLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->dotLoc)));
  // ::cxx::ModuleQualifierAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writeAstModulePartitionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ModulePartitionAST* self) {
  // ::cxx::ModulePartitionAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::ModulePartitionAST::moduleName
  out.varU32(static_cast<std::uint32_t>(astRef(self->moduleName)));
}

void SemanticEncoder::writeAstImportNameAST(
    ByteWriter& out, [[maybe_unused]] cxx::ImportNameAST* self) {
  // ::cxx::ImportNameAST::headerLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->headerLoc)));
  // ::cxx::ImportNameAST::modulePartition
  out.varU32(static_cast<std::uint32_t>(astRef(self->modulePartition)));
  // ::cxx::ImportNameAST::moduleName
  out.varU32(static_cast<std::uint32_t>(astRef(self->moduleName)));
}

void SemanticEncoder::writeAstInitDeclaratorAST(
    ByteWriter& out, [[maybe_unused]] cxx::InitDeclaratorAST* self) {
  // ::cxx::InitDeclaratorAST::declarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->declarator)));
  // ::cxx::InitDeclaratorAST::requiresClause
  out.varU32(static_cast<std::uint32_t>(astRef(self->requiresClause)));
  // ::cxx::InitDeclaratorAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::InitDeclaratorAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstDeclaratorAST(
    ByteWriter& out, [[maybe_unused]] cxx::DeclaratorAST* self) {
  // ::cxx::DeclaratorAST::ptrOpList
  writeAstList(out, self->ptrOpList);
  // ::cxx::DeclaratorAST::coreDeclarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->coreDeclarator)));
  // ::cxx::DeclaratorAST::declaratorChunkList
  writeAstList(out, self->declaratorChunkList);
}

void SemanticEncoder::writeAstUsingDeclaratorAST(
    ByteWriter& out, [[maybe_unused]] cxx::UsingDeclaratorAST* self) {
  // ::cxx::UsingDeclaratorAST::typenameLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->typenameLoc)));
  // ::cxx::UsingDeclaratorAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::UsingDeclaratorAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::UsingDeclaratorAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::UsingDeclaratorAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::UsingDeclaratorAST::isPack
  out.boolean(self->isPack);
}

void SemanticEncoder::writeAstEnumeratorAST(
    ByteWriter& out, [[maybe_unused]] cxx::EnumeratorAST* self) {
  // ::cxx::EnumeratorAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::EnumeratorAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::EnumeratorAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::EnumeratorAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::EnumeratorAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::EnumeratorAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstTypeIdAST(ByteWriter& out,
                                        [[maybe_unused]] cxx::TypeIdAST* self) {
  // ::cxx::TypeIdAST::typeSpecifierList
  writeAstList(out, self->typeSpecifierList);
  // ::cxx::TypeIdAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::TypeIdAST::declarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->declarator)));
  // ::cxx::TypeIdAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
}

void SemanticEncoder::writeAstHandlerAST(
    ByteWriter& out, [[maybe_unused]] cxx::HandlerAST* self) {
  // ::cxx::HandlerAST::catchLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->catchLoc)));
  // ::cxx::HandlerAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::HandlerAST::exceptionDeclaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->exceptionDeclaration)));
  // ::cxx::HandlerAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::HandlerAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::HandlerAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstBaseSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::BaseSpecifierAST* self) {
  // ::cxx::BaseSpecifierAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::BaseSpecifierAST::virtualOrAccessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->virtualOrAccessLoc)));
  // ::cxx::BaseSpecifierAST::otherVirtualOrAccessLoc
  out.varU32(
      static_cast<std::uint32_t>(locationRef(self->otherVirtualOrAccessLoc)));
  // ::cxx::BaseSpecifierAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::BaseSpecifierAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::BaseSpecifierAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::BaseSpecifierAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::BaseSpecifierAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
  // ::cxx::BaseSpecifierAST::isVirtual
  out.boolean(self->isVirtual);
  // ::cxx::BaseSpecifierAST::isVariadic
  out.boolean(self->isVariadic);
  // ::cxx::BaseSpecifierAST::accessSpecifier
  out.varU32(static_cast<std::uint32_t>(self->accessSpecifier));
  // ::cxx::BaseSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstRequiresClauseAST(
    ByteWriter& out, [[maybe_unused]] cxx::RequiresClauseAST* self) {
  // ::cxx::RequiresClauseAST::requiresLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->requiresLoc)));
  // ::cxx::RequiresClauseAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstParameterDeclarationClauseAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::ParameterDeclarationClauseAST* self) {
  // ::cxx::ParameterDeclarationClauseAST::parameterDeclarationList
  writeAstList(out, self->parameterDeclarationList);
  // ::cxx::ParameterDeclarationClauseAST::commaLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->commaLoc)));
  // ::cxx::ParameterDeclarationClauseAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::ParameterDeclarationClauseAST::functionParametersSymbol
  out.varU32(
      static_cast<std::uint32_t>(symbolRef(self->functionParametersSymbol)));
  // ::cxx::ParameterDeclarationClauseAST::isVariadic
  out.boolean(self->isVariadic);
}

void SemanticEncoder::writeAstTrailingReturnTypeAST(
    ByteWriter& out, [[maybe_unused]] cxx::TrailingReturnTypeAST* self) {
  // ::cxx::TrailingReturnTypeAST::minusGreaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->minusGreaterLoc)));
  // ::cxx::TrailingReturnTypeAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
}

void SemanticEncoder::writeAstLambdaSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::LambdaSpecifierAST* self) {
  // ::cxx::LambdaSpecifierAST::specifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->specifierLoc)));
  // ::cxx::LambdaSpecifierAST::specifier
  out.varU32(static_cast<std::uint32_t>(self->specifier));
}

void SemanticEncoder::writeAstTypeConstraintAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeConstraintAST* self) {
  // ::cxx::TypeConstraintAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::TypeConstraintAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::TypeConstraintAST::lessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lessLoc)));
  // ::cxx::TypeConstraintAST::templateArgumentList
  writeAstList(out, self->templateArgumentList);
  // ::cxx::TypeConstraintAST::greaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->greaterLoc)));
  // ::cxx::TypeConstraintAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::TypeConstraintAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstAttributeArgumentClauseAST(
    ByteWriter& out, [[maybe_unused]] cxx::AttributeArgumentClauseAST* self) {
  // ::cxx::AttributeArgumentClauseAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::AttributeArgumentClauseAST::expressionList
  writeAstList(out, self->expressionList);
  // ::cxx::AttributeArgumentClauseAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstAttributeAST(
    ByteWriter& out, [[maybe_unused]] cxx::AttributeAST* self) {
  // ::cxx::AttributeAST::attributeToken
  out.varU32(static_cast<std::uint32_t>(astRef(self->attributeToken)));
  // ::cxx::AttributeAST::attributeArgumentClause
  out.varU32(static_cast<std::uint32_t>(astRef(self->attributeArgumentClause)));
  // ::cxx::AttributeAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
}

void SemanticEncoder::writeAstAttributeUsingPrefixAST(
    ByteWriter& out, [[maybe_unused]] cxx::AttributeUsingPrefixAST* self) {
  // ::cxx::AttributeUsingPrefixAST::usingLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->usingLoc)));
  // ::cxx::AttributeUsingPrefixAST::attributeNamespaceLoc
  out.varU32(
      static_cast<std::uint32_t>(locationRef(self->attributeNamespaceLoc)));
  // ::cxx::AttributeUsingPrefixAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
}

void SemanticEncoder::writeAstNewPlacementAST(
    ByteWriter& out, [[maybe_unused]] cxx::NewPlacementAST* self) {
  // ::cxx::NewPlacementAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::NewPlacementAST::expressionList
  writeAstList(out, self->expressionList);
  // ::cxx::NewPlacementAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstNestedNamespaceSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::NestedNamespaceSpecifierAST* self) {
  // ::cxx::NestedNamespaceSpecifierAST::inlineLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->inlineLoc)));
  // ::cxx::NestedNamespaceSpecifierAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::NestedNamespaceSpecifierAST::scopeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->scopeLoc)));
  // ::cxx::NestedNamespaceSpecifierAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::NestedNamespaceSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::NestedNamespaceSpecifierAST::isInline
  out.boolean(self->isInline);
}

void SemanticEncoder::writeAstLabeledStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::LabeledStatementAST* self) {
  // ::cxx::LabeledStatementAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::LabeledStatementAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::LabeledStatementAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::LabeledStatementAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writeAstCaseStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::CaseStatementAST* self) {
  // ::cxx::CaseStatementAST::caseLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->caseLoc)));
  // ::cxx::CaseStatementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::CaseStatementAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::CaseStatementAST::caseValue
  out.varI64(static_cast<std::int64_t>(self->caseValue));
}

void SemanticEncoder::writeAstDefaultStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::DefaultStatementAST* self) {
  // ::cxx::DefaultStatementAST::defaultLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->defaultLoc)));
  // ::cxx::DefaultStatementAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
}

void SemanticEncoder::writeAstExpressionStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::ExpressionStatementAST* self) {
  // ::cxx::ExpressionStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ExpressionStatementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::ExpressionStatementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstCompoundStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::CompoundStatementAST* self) {
  // ::cxx::CompoundStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::CompoundStatementAST::lbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbraceLoc)));
  // ::cxx::CompoundStatementAST::statementList
  writeAstList(out, self->statementList);
  // ::cxx::CompoundStatementAST::rbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbraceLoc)));
  // ::cxx::CompoundStatementAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstIfStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::IfStatementAST* self) {
  // ::cxx::IfStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::IfStatementAST::ifLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ifLoc)));
  // ::cxx::IfStatementAST::constexprLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->constexprLoc)));
  // ::cxx::IfStatementAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::IfStatementAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::IfStatementAST::condition
  out.varU32(static_cast<std::uint32_t>(astRef(self->condition)));
  // ::cxx::IfStatementAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::IfStatementAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::IfStatementAST::elseLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->elseLoc)));
  // ::cxx::IfStatementAST::elseStatement
  out.varU32(static_cast<std::uint32_t>(astRef(self->elseStatement)));
  // ::cxx::IfStatementAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstConstevalIfStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConstevalIfStatementAST* self) {
  // ::cxx::ConstevalIfStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ConstevalIfStatementAST::ifLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ifLoc)));
  // ::cxx::ConstevalIfStatementAST::exclaimLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->exclaimLoc)));
  // ::cxx::ConstevalIfStatementAST::constvalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->constvalLoc)));
  // ::cxx::ConstevalIfStatementAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::ConstevalIfStatementAST::elseLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->elseLoc)));
  // ::cxx::ConstevalIfStatementAST::elseStatement
  out.varU32(static_cast<std::uint32_t>(astRef(self->elseStatement)));
  // ::cxx::ConstevalIfStatementAST::isNot
  out.boolean(self->isNot);
}

void SemanticEncoder::writeAstSwitchStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::SwitchStatementAST* self) {
  // ::cxx::SwitchStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::SwitchStatementAST::switchLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->switchLoc)));
  // ::cxx::SwitchStatementAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::SwitchStatementAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::SwitchStatementAST::condition
  out.varU32(static_cast<std::uint32_t>(astRef(self->condition)));
  // ::cxx::SwitchStatementAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::SwitchStatementAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::SwitchStatementAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstWhileStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::WhileStatementAST* self) {
  // ::cxx::WhileStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::WhileStatementAST::whileLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->whileLoc)));
  // ::cxx::WhileStatementAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::WhileStatementAST::condition
  out.varU32(static_cast<std::uint32_t>(astRef(self->condition)));
  // ::cxx::WhileStatementAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::WhileStatementAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::WhileStatementAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstDoStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::DoStatementAST* self) {
  // ::cxx::DoStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::DoStatementAST::doLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->doLoc)));
  // ::cxx::DoStatementAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::DoStatementAST::whileLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->whileLoc)));
  // ::cxx::DoStatementAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::DoStatementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::DoStatementAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::DoStatementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstForRangeStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::ForRangeStatementAST* self) {
  // ::cxx::ForRangeStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ForRangeStatementAST::forLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->forLoc)));
  // ::cxx::ForRangeStatementAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::ForRangeStatementAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::ForRangeStatementAST::rangeDeclaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->rangeDeclaration)));
  // ::cxx::ForRangeStatementAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::ForRangeStatementAST::rangeInitializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->rangeInitializer)));
  // ::cxx::ForRangeStatementAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::ForRangeStatementAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::ForRangeStatementAST::beginInitializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->beginInitializer)));
  // ::cxx::ForRangeStatementAST::endInitializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->endInitializer)));
  // ::cxx::ForRangeStatementAST::condition
  out.varU32(static_cast<std::uint32_t>(astRef(self->condition)));
  // ::cxx::ForRangeStatementAST::increment
  out.varU32(static_cast<std::uint32_t>(astRef(self->increment)));
  // ::cxx::ForRangeStatementAST::element
  out.varU32(static_cast<std::uint32_t>(astRef(self->element)));
  // ::cxx::ForRangeStatementAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::ForRangeStatementAST::rangeVariable
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->rangeVariable)));
  // ::cxx::ForRangeStatementAST::beginVariable
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->beginVariable)));
  // ::cxx::ForRangeStatementAST::endVariable
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->endVariable)));
  // ::cxx::ForRangeStatementAST::beginFunction
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->beginFunction)));
  // ::cxx::ForRangeStatementAST::endFunction
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->endFunction)));
  // ::cxx::ForRangeStatementAST::derefFunction
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->derefFunction)));
  // ::cxx::ForRangeStatementAST::incrementFunction
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->incrementFunction)));
  // ::cxx::ForRangeStatementAST::notEqualFunction
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->notEqualFunction)));
  // ::cxx::ForRangeStatementAST::usesMemberBeginEnd
  out.boolean(self->usesMemberBeginEnd);
  // ::cxx::ForRangeStatementAST::isPointerIterator
  out.boolean(self->isPointerIterator);
  // ::cxx::ForRangeStatementAST::notEqualRewritten
  out.boolean(self->notEqualRewritten);
  // ::cxx::ForRangeStatementAST::notEqualReversed
  out.boolean(self->notEqualReversed);
}

void SemanticEncoder::writeAstForStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::ForStatementAST* self) {
  // ::cxx::ForStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ForStatementAST::forLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->forLoc)));
  // ::cxx::ForStatementAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::ForStatementAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::ForStatementAST::condition
  out.varU32(static_cast<std::uint32_t>(astRef(self->condition)));
  // ::cxx::ForStatementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::ForStatementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::ForStatementAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::ForStatementAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::ForStatementAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstBreakStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::BreakStatementAST* self) {
  // ::cxx::BreakStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::BreakStatementAST::breakLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->breakLoc)));
  // ::cxx::BreakStatementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstContinueStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::ContinueStatementAST* self) {
  // ::cxx::ContinueStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ContinueStatementAST::continueLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->continueLoc)));
  // ::cxx::ContinueStatementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstReturnStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::ReturnStatementAST* self) {
  // ::cxx::ReturnStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ReturnStatementAST::returnLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->returnLoc)));
  // ::cxx::ReturnStatementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::ReturnStatementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstCoroutineReturnStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::CoroutineReturnStatementAST* self) {
  // ::cxx::CoroutineReturnStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::CoroutineReturnStatementAST::coreturnLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->coreturnLoc)));
  // ::cxx::CoroutineReturnStatementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::CoroutineReturnStatementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstGotoStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::GotoStatementAST* self) {
  // ::cxx::GotoStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::GotoStatementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::GotoStatementAST::gotoLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->gotoLoc)));
  // ::cxx::GotoStatementAST::starLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->starLoc)));
  // ::cxx::GotoStatementAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::GotoStatementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::GotoStatementAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::GotoStatementAST::isIndirect
  out.boolean(self->isIndirect);
}

void SemanticEncoder::writeAstDeclarationStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::DeclarationStatementAST* self) {
  // ::cxx::DeclarationStatementAST::declaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration)));
}

void SemanticEncoder::writeAstTryBlockStatementAST(
    ByteWriter& out, [[maybe_unused]] cxx::TryBlockStatementAST* self) {
  // ::cxx::TryBlockStatementAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::TryBlockStatementAST::tryLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->tryLoc)));
  // ::cxx::TryBlockStatementAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::TryBlockStatementAST::handlerList
  writeAstList(out, self->handlerList);
}

void SemanticEncoder::writeAstCharLiteralExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::CharLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::CharLiteralExpressionAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::CharLiteralExpressionAST::literal
  writeLiteral(out, self->literal);
  // ::cxx::CharLiteralExpressionAST::literalOperatorCall
  out.varU32(static_cast<std::uint32_t>(astRef(self->literalOperatorCall)));
}

void SemanticEncoder::writeAstBoolLiteralExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::BoolLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::BoolLiteralExpressionAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::BoolLiteralExpressionAST::isTrue
  out.boolean(self->isTrue);
}

void SemanticEncoder::writeAstIntLiteralExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::IntLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::IntLiteralExpressionAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::IntLiteralExpressionAST::literal
  writeLiteral(out, self->literal);
  // ::cxx::IntLiteralExpressionAST::literalOperatorCall
  out.varU32(static_cast<std::uint32_t>(astRef(self->literalOperatorCall)));
}

void SemanticEncoder::writeAstFloatLiteralExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::FloatLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::FloatLiteralExpressionAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::FloatLiteralExpressionAST::literal
  writeLiteral(out, self->literal);
  // ::cxx::FloatLiteralExpressionAST::literalOperatorCall
  out.varU32(static_cast<std::uint32_t>(astRef(self->literalOperatorCall)));
}

void SemanticEncoder::writeAstNullptrLiteralExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::NullptrLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::NullptrLiteralExpressionAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::NullptrLiteralExpressionAST::literal
  out.varU32(static_cast<std::uint32_t>(self->literal));
}

void SemanticEncoder::writeAstStringLiteralExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::StringLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::StringLiteralExpressionAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::StringLiteralExpressionAST::literal
  writeLiteral(out, self->literal);
  // ::cxx::StringLiteralExpressionAST::encoding
  out.varU32(static_cast<std::uint32_t>(self->encoding));
}

void SemanticEncoder::writeAstUserDefinedStringLiteralExpressionAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::UserDefinedStringLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::UserDefinedStringLiteralExpressionAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::UserDefinedStringLiteralExpressionAST::literal
  writeLiteral(out, self->literal);
  // ::cxx::UserDefinedStringLiteralExpressionAST::literalOperatorCall
  out.varU32(static_cast<std::uint32_t>(astRef(self->literalOperatorCall)));
  // ::cxx::UserDefinedStringLiteralExpressionAST::encoding
  out.varU32(static_cast<std::uint32_t>(self->encoding));
}

void SemanticEncoder::writeAstObjectLiteralExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ObjectLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ObjectLiteralExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::ObjectLiteralExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::ObjectLiteralExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::ObjectLiteralExpressionAST::bracedInitList
  out.varU32(static_cast<std::uint32_t>(astRef(self->bracedInitList)));
  // ::cxx::ObjectLiteralExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstThisExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ThisExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ThisExpressionAST::thisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->thisLoc)));
}

void SemanticEncoder::writeAstPackIndexExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::PackIndexExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::PackIndexExpressionAST::packExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->packExpression)));
  // ::cxx::PackIndexExpressionAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::PackIndexExpressionAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::PackIndexExpressionAST::indexExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->indexExpression)));
  // ::cxx::PackIndexExpressionAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
}

void SemanticEncoder::writeAstGenericSelectionExpressionAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::GenericSelectionExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::GenericSelectionExpressionAST::genericLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->genericLoc)));
  // ::cxx::GenericSelectionExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::GenericSelectionExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::GenericSelectionExpressionAST::commaLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->commaLoc)));
  // ::cxx::GenericSelectionExpressionAST::genericAssociationList
  writeAstList(out, self->genericAssociationList);
  // ::cxx::GenericSelectionExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::GenericSelectionExpressionAST::matchedAssocIndex
  out.varI32(static_cast<std::int32_t>(self->matchedAssocIndex));
}

void SemanticEncoder::writeAstNestedStatementExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::NestedStatementExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::NestedStatementExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::NestedStatementExpressionAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::NestedStatementExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstDefaultInitializerExpressionAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::DefaultInitializerExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::DefaultInitializerExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::DefaultInitializerExpressionAST::context
  writecxxDefaultInitializerContext(out, &self->context);
}

void SemanticEncoder::writeAstNestedExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::NestedExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::NestedExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::NestedExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::NestedExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstIdExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::IdExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::IdExpressionAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::IdExpressionAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::IdExpressionAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::IdExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::IdExpressionAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
}

void SemanticEncoder::writeAstLambdaExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::LambdaExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::LambdaExpressionAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::LambdaExpressionAST::captureDefaultLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->captureDefaultLoc)));
  // ::cxx::LambdaExpressionAST::captureList
  writeAstList(out, self->captureList);
  // ::cxx::LambdaExpressionAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
  // ::cxx::LambdaExpressionAST::lessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lessLoc)));
  // ::cxx::LambdaExpressionAST::templateParameterList
  writeAstList(out, self->templateParameterList);
  // ::cxx::LambdaExpressionAST::greaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->greaterLoc)));
  // ::cxx::LambdaExpressionAST::templateRequiresClause
  out.varU32(static_cast<std::uint32_t>(astRef(self->templateRequiresClause)));
  // ::cxx::LambdaExpressionAST::expressionAttributeList
  writeAstList(out, self->expressionAttributeList);
  // ::cxx::LambdaExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::LambdaExpressionAST::parameterDeclarationClause
  out.varU32(
      static_cast<std::uint32_t>(astRef(self->parameterDeclarationClause)));
  // ::cxx::LambdaExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::LambdaExpressionAST::gnuAtributeList
  writeAstList(out, self->gnuAtributeList);
  // ::cxx::LambdaExpressionAST::lambdaSpecifierList
  writeAstList(out, self->lambdaSpecifierList);
  // ::cxx::LambdaExpressionAST::exceptionSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->exceptionSpecifier)));
  // ::cxx::LambdaExpressionAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::LambdaExpressionAST::trailingReturnType
  out.varU32(static_cast<std::uint32_t>(astRef(self->trailingReturnType)));
  // ::cxx::LambdaExpressionAST::requiresClause
  out.varU32(static_cast<std::uint32_t>(astRef(self->requiresClause)));
  // ::cxx::LambdaExpressionAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::LambdaExpressionAST::captureDefault
  out.varU32(static_cast<std::uint32_t>(self->captureDefault));
  // ::cxx::LambdaExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::LambdaExpressionAST::constructorSymbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructorSymbol)));
}

void SemanticEncoder::writeAstFoldExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::FoldExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::FoldExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::FoldExpressionAST::leftExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->leftExpression)));
  // ::cxx::FoldExpressionAST::opLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->opLoc)));
  // ::cxx::FoldExpressionAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::FoldExpressionAST::foldOpLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->foldOpLoc)));
  // ::cxx::FoldExpressionAST::rightExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->rightExpression)));
  // ::cxx::FoldExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::FoldExpressionAST::op
  out.varU32(static_cast<std::uint32_t>(self->op));
  // ::cxx::FoldExpressionAST::foldOp
  out.varU32(static_cast<std::uint32_t>(self->foldOp));
}

void SemanticEncoder::writeAstRightFoldExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::RightFoldExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::RightFoldExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::RightFoldExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::RightFoldExpressionAST::opLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->opLoc)));
  // ::cxx::RightFoldExpressionAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::RightFoldExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::RightFoldExpressionAST::op
  out.varU32(static_cast<std::uint32_t>(self->op));
}

void SemanticEncoder::writeAstLeftFoldExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::LeftFoldExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::LeftFoldExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::LeftFoldExpressionAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::LeftFoldExpressionAST::opLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->opLoc)));
  // ::cxx::LeftFoldExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::LeftFoldExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::LeftFoldExpressionAST::op
  out.varU32(static_cast<std::uint32_t>(self->op));
}

void SemanticEncoder::writeAstRequiresExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::RequiresExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::RequiresExpressionAST::requiresLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->requiresLoc)));
  // ::cxx::RequiresExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::RequiresExpressionAST::parameterDeclarationClause
  out.varU32(
      static_cast<std::uint32_t>(astRef(self->parameterDeclarationClause)));
  // ::cxx::RequiresExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::RequiresExpressionAST::lbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbraceLoc)));
  // ::cxx::RequiresExpressionAST::requirementList
  writeAstList(out, self->requirementList);
  // ::cxx::RequiresExpressionAST::rbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbraceLoc)));
}

void SemanticEncoder::writeAstVaArgExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::VaArgExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::VaArgExpressionAST::vaArgLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->vaArgLoc)));
  // ::cxx::VaArgExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::VaArgExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::VaArgExpressionAST::commaLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->commaLoc)));
  // ::cxx::VaArgExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::VaArgExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstSubscriptExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::SubscriptExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::SubscriptExpressionAST::baseExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->baseExpression)));
  // ::cxx::SubscriptExpressionAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::SubscriptExpressionAST::indexExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->indexExpression)));
  // ::cxx::SubscriptExpressionAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
  // ::cxx::SubscriptExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::SubscriptExpressionAST::isVirtualDispatch
  out.boolean(self->isVirtualDispatch);
}

void SemanticEncoder::writeAstCallExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::CallExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::CallExpressionAST::baseExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->baseExpression)));
  // ::cxx::CallExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::CallExpressionAST::expressionList
  writeAstList(out, self->expressionList);
  // ::cxx::CallExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::CallExpressionAST::isVirtualDispatch
  out.boolean(self->isVirtualDispatch);
  // ::cxx::CallExpressionAST::constructorSymbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructorSymbol)));
}

void SemanticEncoder::writeAstTypeConstructionAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeConstructionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::TypeConstructionAST::typeSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeSpecifier)));
  // ::cxx::TypeConstructionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::TypeConstructionAST::expressionList
  writeAstList(out, self->expressionList);
  // ::cxx::TypeConstructionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::TypeConstructionAST::constructorSymbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructorSymbol)));
}

void SemanticEncoder::writeAstBracedTypeConstructionAST(
    ByteWriter& out, [[maybe_unused]] cxx::BracedTypeConstructionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::BracedTypeConstructionAST::typeSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeSpecifier)));
  // ::cxx::BracedTypeConstructionAST::bracedInitList
  out.varU32(static_cast<std::uint32_t>(astRef(self->bracedInitList)));
  // ::cxx::BracedTypeConstructionAST::constructorSymbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructorSymbol)));
}

void SemanticEncoder::writeAstSpliceMemberExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::SpliceMemberExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::SpliceMemberExpressionAST::baseExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->baseExpression)));
  // ::cxx::SpliceMemberExpressionAST::accessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->accessLoc)));
  // ::cxx::SpliceMemberExpressionAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::SpliceMemberExpressionAST::splicer
  out.varU32(static_cast<std::uint32_t>(astRef(self->splicer)));
  // ::cxx::SpliceMemberExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::SpliceMemberExpressionAST::accessOp
  out.varU32(static_cast<std::uint32_t>(self->accessOp));
  // ::cxx::SpliceMemberExpressionAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
}

void SemanticEncoder::writeAstMemberExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::MemberExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::MemberExpressionAST::baseExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->baseExpression)));
  // ::cxx::MemberExpressionAST::accessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->accessLoc)));
  // ::cxx::MemberExpressionAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::MemberExpressionAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::MemberExpressionAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::MemberExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::MemberExpressionAST::accessOp
  out.varU32(static_cast<std::uint32_t>(self->accessOp));
  // ::cxx::MemberExpressionAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
}

void SemanticEncoder::writeAstPostIncrExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::PostIncrExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::PostIncrExpressionAST::baseExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->baseExpression)));
  // ::cxx::PostIncrExpressionAST::opLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->opLoc)));
  // ::cxx::PostIncrExpressionAST::op
  out.varU32(static_cast<std::uint32_t>(self->op));
  // ::cxx::PostIncrExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::PostIncrExpressionAST::isVirtualDispatch
  out.boolean(self->isVirtualDispatch);
}

void SemanticEncoder::writeAstCppCastExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::CppCastExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::CppCastExpressionAST::castLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->castLoc)));
  // ::cxx::CppCastExpressionAST::lessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lessLoc)));
  // ::cxx::CppCastExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::CppCastExpressionAST::greaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->greaterLoc)));
  // ::cxx::CppCastExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::CppCastExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::CppCastExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::CppCastExpressionAST::castOp
  out.varU32(static_cast<std::uint32_t>(self->castOp));
}

void SemanticEncoder::writeAstBuiltinBitCastExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::BuiltinBitCastExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::BuiltinBitCastExpressionAST::castLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->castLoc)));
  // ::cxx::BuiltinBitCastExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::BuiltinBitCastExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::BuiltinBitCastExpressionAST::commaLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->commaLoc)));
  // ::cxx::BuiltinBitCastExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::BuiltinBitCastExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstBuiltinOffsetofExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::BuiltinOffsetofExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::BuiltinOffsetofExpressionAST::offsetofLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->offsetofLoc)));
  // ::cxx::BuiltinOffsetofExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::BuiltinOffsetofExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::BuiltinOffsetofExpressionAST::commaLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->commaLoc)));
  // ::cxx::BuiltinOffsetofExpressionAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::BuiltinOffsetofExpressionAST::designatorList
  writeAstList(out, self->designatorList);
  // ::cxx::BuiltinOffsetofExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::BuiltinOffsetofExpressionAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::BuiltinOffsetofExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstTypeidExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeidExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::TypeidExpressionAST::typeidLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->typeidLoc)));
  // ::cxx::TypeidExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::TypeidExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::TypeidExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstTypeidOfTypeExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeidOfTypeExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::TypeidOfTypeExpressionAST::typeidLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->typeidLoc)));
  // ::cxx::TypeidOfTypeExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::TypeidOfTypeExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::TypeidOfTypeExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstSpliceExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::SpliceExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::SpliceExpressionAST::splicer
  out.varU32(static_cast<std::uint32_t>(astRef(self->splicer)));
}

void SemanticEncoder::writeAstGlobalScopeReflectExpressionAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::GlobalScopeReflectExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::GlobalScopeReflectExpressionAST::caretCaretLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->caretCaretLoc)));
  // ::cxx::GlobalScopeReflectExpressionAST::scopeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->scopeLoc)));
}

void SemanticEncoder::writeAstNamespaceReflectExpressionAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::NamespaceReflectExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::NamespaceReflectExpressionAST::caretCaretLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->caretCaretLoc)));
  // ::cxx::NamespaceReflectExpressionAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::NamespaceReflectExpressionAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::NamespaceReflectExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstTypeIdReflectExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeIdReflectExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::TypeIdReflectExpressionAST::caretCaretLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->caretCaretLoc)));
  // ::cxx::TypeIdReflectExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
}

void SemanticEncoder::writeAstReflectExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ReflectExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ReflectExpressionAST::caretCaretLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->caretCaretLoc)));
  // ::cxx::ReflectExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstLabelAddressExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::LabelAddressExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::LabelAddressExpressionAST::ampAmpLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ampAmpLoc)));
  // ::cxx::LabelAddressExpressionAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::LabelAddressExpressionAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writeAstUnaryExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::UnaryExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::UnaryExpressionAST::opLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->opLoc)));
  // ::cxx::UnaryExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::UnaryExpressionAST::op
  out.varU32(static_cast<std::uint32_t>(self->op));
  // ::cxx::UnaryExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::UnaryExpressionAST::isVirtualDispatch
  out.boolean(self->isVirtualDispatch);
}

void SemanticEncoder::writeAstAwaitExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::AwaitExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::AwaitExpressionAST::awaitLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->awaitLoc)));
  // ::cxx::AwaitExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstSizeofExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::SizeofExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::SizeofExpressionAST::sizeofLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->sizeofLoc)));
  // ::cxx::SizeofExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::SizeofExpressionAST::value
  out.boolean(self->value.has_value());
  if (self->value.has_value()) {
    out.varI64(static_cast<std::int64_t>((*self->value)));
  }
}

void SemanticEncoder::writeAstSizeofTypeExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::SizeofTypeExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::SizeofTypeExpressionAST::sizeofLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->sizeofLoc)));
  // ::cxx::SizeofTypeExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::SizeofTypeExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::SizeofTypeExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::SizeofTypeExpressionAST::value
  out.boolean(self->value.has_value());
  if (self->value.has_value()) {
    out.varI64(static_cast<std::int64_t>((*self->value)));
  }
}

void SemanticEncoder::writeAstSizeofPackExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::SizeofPackExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::SizeofPackExpressionAST::sizeofLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->sizeofLoc)));
  // ::cxx::SizeofPackExpressionAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::SizeofPackExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::SizeofPackExpressionAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::SizeofPackExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::SizeofPackExpressionAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::SizeofPackExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstAlignofTypeExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::AlignofTypeExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::AlignofTypeExpressionAST::alignofLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->alignofLoc)));
  // ::cxx::AlignofTypeExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::AlignofTypeExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::AlignofTypeExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstAlignofExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::AlignofExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::AlignofExpressionAST::alignofLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->alignofLoc)));
  // ::cxx::AlignofExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstNoexceptExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::NoexceptExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::NoexceptExpressionAST::noexceptLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->noexceptLoc)));
  // ::cxx::NoexceptExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::NoexceptExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::NoexceptExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::NoexceptExpressionAST::value
  out.boolean(self->value.has_value());
  if (self->value.has_value()) {
    out.boolean((*self->value));
  }
}

void SemanticEncoder::writeAstNewExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::NewExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::NewExpressionAST::scopeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->scopeLoc)));
  // ::cxx::NewExpressionAST::newLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->newLoc)));
  // ::cxx::NewExpressionAST::newPlacement
  out.varU32(static_cast<std::uint32_t>(astRef(self->newPlacement)));
  // ::cxx::NewExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::NewExpressionAST::typeSpecifierList
  writeAstList(out, self->typeSpecifierList);
  // ::cxx::NewExpressionAST::declarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->declarator)));
  // ::cxx::NewExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::NewExpressionAST::newInitalizer
  out.varU32(static_cast<std::uint32_t>(astRef(self->newInitalizer)));
  // ::cxx::NewExpressionAST::objectType
  out.varU32(static_cast<std::uint32_t>(typeRef(self->objectType)));
  // ::cxx::NewExpressionAST::constructorSymbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructorSymbol)));
  // ::cxx::NewExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstDeleteExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::DeleteExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::DeleteExpressionAST::scopeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->scopeLoc)));
  // ::cxx::DeleteExpressionAST::deleteLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->deleteLoc)));
  // ::cxx::DeleteExpressionAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::DeleteExpressionAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
  // ::cxx::DeleteExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::DeleteExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstCastExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::CastExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::CastExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::CastExpressionAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::CastExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::CastExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstImplicitCastExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ImplicitCastExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ImplicitCastExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::ImplicitCastExpressionAST::castKind
  out.varU32(static_cast<std::uint32_t>(self->castKind));
  // ::cxx::ImplicitCastExpressionAST::conversionFunction
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->conversionFunction)));
  // ::cxx::ImplicitCastExpressionAST::isVirtualDispatch
  out.boolean(self->isVirtualDispatch);
}

void SemanticEncoder::writeAstConstExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConstExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ConstExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::ConstExpressionAST::constValue
  out.boolean(self->constValue != nullptr);
  if (self->constValue) writeConstValue(out, *self->constValue);
}

void SemanticEncoder::writeAstBinaryExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::BinaryExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::BinaryExpressionAST::leftExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->leftExpression)));
  // ::cxx::BinaryExpressionAST::opLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->opLoc)));
  // ::cxx::BinaryExpressionAST::rightExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->rightExpression)));
  // ::cxx::BinaryExpressionAST::op
  out.varU32(static_cast<std::uint32_t>(self->op));
  // ::cxx::BinaryExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::BinaryExpressionAST::isVirtualDispatch
  out.boolean(self->isVirtualDispatch);
}

void SemanticEncoder::writeAstConditionalExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConditionalExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ConditionalExpressionAST::condition
  out.varU32(static_cast<std::uint32_t>(astRef(self->condition)));
  // ::cxx::ConditionalExpressionAST::questionLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->questionLoc)));
  // ::cxx::ConditionalExpressionAST::iftrueExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->iftrueExpression)));
  // ::cxx::ConditionalExpressionAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::ConditionalExpressionAST::iffalseExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->iffalseExpression)));
}

void SemanticEncoder::writeAstYieldExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::YieldExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::YieldExpressionAST::yieldLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->yieldLoc)));
  // ::cxx::YieldExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstThrowExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ThrowExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ThrowExpressionAST::throwLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->throwLoc)));
  // ::cxx::ThrowExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstAssignmentExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::AssignmentExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::AssignmentExpressionAST::leftExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->leftExpression)));
  // ::cxx::AssignmentExpressionAST::opLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->opLoc)));
  // ::cxx::AssignmentExpressionAST::rightExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->rightExpression)));
  // ::cxx::AssignmentExpressionAST::op
  out.varU32(static_cast<std::uint32_t>(self->op));
  // ::cxx::AssignmentExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::AssignmentExpressionAST::isVirtualDispatch
  out.boolean(self->isVirtualDispatch);
}

void SemanticEncoder::writeAstTargetExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::TargetExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
}

void SemanticEncoder::writeAstRightExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::RightExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
}

void SemanticEncoder::writeAstCompoundAssignmentExpressionAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::CompoundAssignmentExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::CompoundAssignmentExpressionAST::targetExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->targetExpression)));
  // ::cxx::CompoundAssignmentExpressionAST::opLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->opLoc)));
  // ::cxx::CompoundAssignmentExpressionAST::leftExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->leftExpression)));
  // ::cxx::CompoundAssignmentExpressionAST::rightExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->rightExpression)));
  // ::cxx::CompoundAssignmentExpressionAST::adjustExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->adjustExpression)));
  // ::cxx::CompoundAssignmentExpressionAST::op
  out.varU32(static_cast<std::uint32_t>(self->op));
  // ::cxx::CompoundAssignmentExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::CompoundAssignmentExpressionAST::isVirtualDispatch
  out.boolean(self->isVirtualDispatch);
}

void SemanticEncoder::writeAstPackExpansionExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::PackExpansionExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::PackExpansionExpressionAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::PackExpansionExpressionAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
}

void SemanticEncoder::writeAstDesignatedInitializerClauseAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::DesignatedInitializerClauseAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::DesignatedInitializerClauseAST::designatorList
  writeAstList(out, self->designatorList);
  // ::cxx::DesignatedInitializerClauseAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::DesignatedInitializerClauseAST::constructorSymbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructorSymbol)));
}

void SemanticEncoder::writeAstTypeTraitExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeTraitExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::TypeTraitExpressionAST::typeTraitLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->typeTraitLoc)));
  // ::cxx::TypeTraitExpressionAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::TypeTraitExpressionAST::typeIdList
  writeAstList(out, self->typeIdList);
  // ::cxx::TypeTraitExpressionAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::TypeTraitExpressionAST::typeTrait
  out.varU32(static_cast<std::uint32_t>(self->typeTrait));
  // ::cxx::TypeTraitExpressionAST::value
  out.boolean(self->value.has_value());
  if (self->value.has_value()) {
    out.boolean((*self->value));
  }
}

void SemanticEncoder::writeAstConditionExpressionAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConditionExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ConditionExpressionAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ConditionExpressionAST::declSpecifierList
  writeAstList(out, self->declSpecifierList);
  // ::cxx::ConditionExpressionAST::declarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->declarator)));
  // ::cxx::ConditionExpressionAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::ConditionExpressionAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstEqualInitializerAST(
    ByteWriter& out, [[maybe_unused]] cxx::EqualInitializerAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::EqualInitializerAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::EqualInitializerAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstBracedInitListAST(
    ByteWriter& out, [[maybe_unused]] cxx::BracedInitListAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::BracedInitListAST::lbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbraceLoc)));
  // ::cxx::BracedInitListAST::expressionList
  writeAstList(out, self->expressionList);
  // ::cxx::BracedInitListAST::commaLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->commaLoc)));
  // ::cxx::BracedInitListAST::rbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbraceLoc)));
}

void SemanticEncoder::writeAstParenInitializerAST(
    ByteWriter& out, [[maybe_unused]] cxx::ParenInitializerAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ParenInitializerAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::ParenInitializerAST::expressionList
  writeAstList(out, self->expressionList);
  // ::cxx::ParenInitializerAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstThreeWayComparisonExpressionAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::ThreeWayComparisonExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  out.varU32(static_cast<std::uint32_t>(self->valueCategory));
  // ::cxx::ExpressionAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
  // ::cxx::ThreeWayComparisonExpressionAST::comparison
  out.varU32(static_cast<std::uint32_t>(astRef(self->comparison)));
  // ::cxx::ThreeWayComparisonExpressionAST::lessResult
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->lessResult)));
  // ::cxx::ThreeWayComparisonExpressionAST::equalResult
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->equalResult)));
  // ::cxx::ThreeWayComparisonExpressionAST::greaterResult
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->greaterResult)));
  // ::cxx::ThreeWayComparisonExpressionAST::unorderedResult
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->unorderedResult)));
}

void SemanticEncoder::writeAstDefaultGenericAssociationAST(
    ByteWriter& out, [[maybe_unused]] cxx::DefaultGenericAssociationAST* self) {
  // ::cxx::DefaultGenericAssociationAST::defaultLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->defaultLoc)));
  // ::cxx::DefaultGenericAssociationAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::DefaultGenericAssociationAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstTypeGenericAssociationAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeGenericAssociationAST* self) {
  // ::cxx::TypeGenericAssociationAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::TypeGenericAssociationAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::TypeGenericAssociationAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstDotDesignatorAST(
    ByteWriter& out, [[maybe_unused]] cxx::DotDesignatorAST* self) {
  // ::cxx::DotDesignatorAST::dotLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->dotLoc)));
  // ::cxx::DotDesignatorAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::DotDesignatorAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::DotDesignatorAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstSubscriptDesignatorAST(
    ByteWriter& out, [[maybe_unused]] cxx::SubscriptDesignatorAST* self) {
  // ::cxx::SubscriptDesignatorAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::SubscriptDesignatorAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::SubscriptDesignatorAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
}

void SemanticEncoder::writeAstTemplateTypeParameterAST(
    ByteWriter& out, [[maybe_unused]] cxx::TemplateTypeParameterAST* self) {
  // ::cxx::TemplateParameterAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::TemplateParameterAST::depth
  out.varI32(static_cast<std::int32_t>(self->depth));
  // ::cxx::TemplateParameterAST::index
  out.varI32(static_cast<std::int32_t>(self->index));
  // ::cxx::TemplateTypeParameterAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::TemplateTypeParameterAST::lessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lessLoc)));
  // ::cxx::TemplateTypeParameterAST::templateParameterList
  writeAstList(out, self->templateParameterList);
  // ::cxx::TemplateTypeParameterAST::greaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->greaterLoc)));
  // ::cxx::TemplateTypeParameterAST::requiresClause
  out.varU32(static_cast<std::uint32_t>(astRef(self->requiresClause)));
  // ::cxx::TemplateTypeParameterAST::classKeyLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->classKeyLoc)));
  // ::cxx::TemplateTypeParameterAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::TemplateTypeParameterAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::TemplateTypeParameterAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::TemplateTypeParameterAST::idExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->idExpression)));
  // ::cxx::TemplateTypeParameterAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::TemplateTypeParameterAST::isPack
  out.boolean(self->isPack);
}

void SemanticEncoder::writeAstNonTypeTemplateParameterAST(
    ByteWriter& out, [[maybe_unused]] cxx::NonTypeTemplateParameterAST* self) {
  // ::cxx::TemplateParameterAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::TemplateParameterAST::depth
  out.varI32(static_cast<std::int32_t>(self->depth));
  // ::cxx::TemplateParameterAST::index
  out.varI32(static_cast<std::int32_t>(self->index));
  // ::cxx::NonTypeTemplateParameterAST::declaration
  out.varU32(static_cast<std::uint32_t>(astRef(self->declaration)));
}

void SemanticEncoder::writeAstTypenameTypeParameterAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypenameTypeParameterAST* self) {
  // ::cxx::TemplateParameterAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::TemplateParameterAST::depth
  out.varI32(static_cast<std::int32_t>(self->depth));
  // ::cxx::TemplateParameterAST::index
  out.varI32(static_cast<std::int32_t>(self->index));
  // ::cxx::TypenameTypeParameterAST::classKeyLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->classKeyLoc)));
  // ::cxx::TypenameTypeParameterAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::TypenameTypeParameterAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::TypenameTypeParameterAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::TypenameTypeParameterAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::TypenameTypeParameterAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::TypenameTypeParameterAST::isPack
  out.boolean(self->isPack);
}

void SemanticEncoder::writeAstConstraintTypeParameterAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConstraintTypeParameterAST* self) {
  // ::cxx::TemplateParameterAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::TemplateParameterAST::depth
  out.varI32(static_cast<std::int32_t>(self->depth));
  // ::cxx::TemplateParameterAST::index
  out.varI32(static_cast<std::int32_t>(self->index));
  // ::cxx::ConstraintTypeParameterAST::typeConstraint
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeConstraint)));
  // ::cxx::ConstraintTypeParameterAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::ConstraintTypeParameterAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::ConstraintTypeParameterAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::ConstraintTypeParameterAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::ConstraintTypeParameterAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writeAstTypedefSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypedefSpecifierAST* self) {
  // ::cxx::TypedefSpecifierAST::typedefLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->typedefLoc)));
}

void SemanticEncoder::writeAstFriendSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::FriendSpecifierAST* self) {
  // ::cxx::FriendSpecifierAST::friendLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->friendLoc)));
}

void SemanticEncoder::writeAstConstevalSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConstevalSpecifierAST* self) {
  // ::cxx::ConstevalSpecifierAST::constevalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->constevalLoc)));
}

void SemanticEncoder::writeAstConstinitSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConstinitSpecifierAST* self) {
  // ::cxx::ConstinitSpecifierAST::constinitLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->constinitLoc)));
}

void SemanticEncoder::writeAstConstexprSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConstexprSpecifierAST* self) {
  // ::cxx::ConstexprSpecifierAST::constexprLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->constexprLoc)));
}

void SemanticEncoder::writeAstInlineSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::InlineSpecifierAST* self) {
  // ::cxx::InlineSpecifierAST::inlineLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->inlineLoc)));
}

void SemanticEncoder::writeAstNoreturnSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::NoreturnSpecifierAST* self) {
  // ::cxx::NoreturnSpecifierAST::noreturnLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->noreturnLoc)));
}

void SemanticEncoder::writeAstStaticSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::StaticSpecifierAST* self) {
  // ::cxx::StaticSpecifierAST::staticLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->staticLoc)));
}

void SemanticEncoder::writeAstExternSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ExternSpecifierAST* self) {
  // ::cxx::ExternSpecifierAST::externLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->externLoc)));
}

void SemanticEncoder::writeAstRegisterSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::RegisterSpecifierAST* self) {
  // ::cxx::RegisterSpecifierAST::registerLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->registerLoc)));
}

void SemanticEncoder::writeAstThreadLocalSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ThreadLocalSpecifierAST* self) {
  // ::cxx::ThreadLocalSpecifierAST::threadLocalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->threadLocalLoc)));
}

void SemanticEncoder::writeAstThreadSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ThreadSpecifierAST* self) {
  // ::cxx::ThreadSpecifierAST::threadLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->threadLoc)));
}

void SemanticEncoder::writeAstMutableSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::MutableSpecifierAST* self) {
  // ::cxx::MutableSpecifierAST::mutableLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->mutableLoc)));
}

void SemanticEncoder::writeAstVirtualSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::VirtualSpecifierAST* self) {
  // ::cxx::VirtualSpecifierAST::virtualLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->virtualLoc)));
}

void SemanticEncoder::writeAstExplicitSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ExplicitSpecifierAST* self) {
  // ::cxx::ExplicitSpecifierAST::explicitLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->explicitLoc)));
  // ::cxx::ExplicitSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::ExplicitSpecifierAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::ExplicitSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstAutoTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::AutoTypeSpecifierAST* self) {
  // ::cxx::AutoTypeSpecifierAST::autoLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->autoLoc)));
}

void SemanticEncoder::writeAstVoidTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::VoidTypeSpecifierAST* self) {
  // ::cxx::VoidTypeSpecifierAST::voidLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->voidLoc)));
}

void SemanticEncoder::writeAstSizeTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::SizeTypeSpecifierAST* self) {
  // ::cxx::SizeTypeSpecifierAST::specifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->specifierLoc)));
  // ::cxx::SizeTypeSpecifierAST::specifier
  out.varU32(static_cast<std::uint32_t>(self->specifier));
}

void SemanticEncoder::writeAstSignTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::SignTypeSpecifierAST* self) {
  // ::cxx::SignTypeSpecifierAST::specifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->specifierLoc)));
  // ::cxx::SignTypeSpecifierAST::specifier
  out.varU32(static_cast<std::uint32_t>(self->specifier));
}

void SemanticEncoder::writeAstBuiltinTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::BuiltinTypeSpecifierAST* self) {
  // ::cxx::BuiltinTypeSpecifierAST::specifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->specifierLoc)));
  // ::cxx::BuiltinTypeSpecifierAST::specifier
  out.varU32(static_cast<std::uint32_t>(self->specifier));
}

void SemanticEncoder::writeAstUnaryBuiltinTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::UnaryBuiltinTypeSpecifierAST* self) {
  // ::cxx::UnaryBuiltinTypeSpecifierAST::builtinLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->builtinLoc)));
  // ::cxx::UnaryBuiltinTypeSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::UnaryBuiltinTypeSpecifierAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::UnaryBuiltinTypeSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::UnaryBuiltinTypeSpecifierAST::builtinKind
  out.varU32(static_cast<std::uint32_t>(self->builtinKind));
}

void SemanticEncoder::writeAstBinaryBuiltinTypeSpecifierAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::BinaryBuiltinTypeSpecifierAST* self) {
  // ::cxx::BinaryBuiltinTypeSpecifierAST::builtinLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->builtinLoc)));
  // ::cxx::BinaryBuiltinTypeSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::BinaryBuiltinTypeSpecifierAST::leftTypeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->leftTypeId)));
  // ::cxx::BinaryBuiltinTypeSpecifierAST::commaLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->commaLoc)));
  // ::cxx::BinaryBuiltinTypeSpecifierAST::rightTypeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->rightTypeId)));
  // ::cxx::BinaryBuiltinTypeSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::BinaryBuiltinTypeSpecifierAST::builtinKind
  out.varU32(static_cast<std::uint32_t>(self->builtinKind));
}

void SemanticEncoder::writeAstIntegralTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::IntegralTypeSpecifierAST* self) {
  // ::cxx::IntegralTypeSpecifierAST::specifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->specifierLoc)));
  // ::cxx::IntegralTypeSpecifierAST::specifier
  out.varU32(static_cast<std::uint32_t>(self->specifier));
}

void SemanticEncoder::writeAstFloatingPointTypeSpecifierAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::FloatingPointTypeSpecifierAST* self) {
  // ::cxx::FloatingPointTypeSpecifierAST::specifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->specifierLoc)));
  // ::cxx::FloatingPointTypeSpecifierAST::specifier
  out.varU32(static_cast<std::uint32_t>(self->specifier));
}

void SemanticEncoder::writeAstComplexTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ComplexTypeSpecifierAST* self) {
  // ::cxx::ComplexTypeSpecifierAST::complexLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->complexLoc)));
}

void SemanticEncoder::writeAstNamedTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::NamedTypeSpecifierAST* self) {
  // ::cxx::NamedTypeSpecifierAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::NamedTypeSpecifierAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::NamedTypeSpecifierAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::NamedTypeSpecifierAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
  // ::cxx::NamedTypeSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstAtomicTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::AtomicTypeSpecifierAST* self) {
  // ::cxx::AtomicTypeSpecifierAST::atomicLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->atomicLoc)));
  // ::cxx::AtomicTypeSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::AtomicTypeSpecifierAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::AtomicTypeSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstBitIntTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::BitIntTypeSpecifierAST* self) {
  // ::cxx::BitIntTypeSpecifierAST::bitintLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->bitintLoc)));
  // ::cxx::BitIntTypeSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::BitIntTypeSpecifierAST::sizeExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->sizeExpression)));
  // ::cxx::BitIntTypeSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::BitIntTypeSpecifierAST::bitCount
  out.varI32(static_cast<std::int32_t>(self->bitCount));
}

void SemanticEncoder::writeAstUnderlyingTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::UnderlyingTypeSpecifierAST* self) {
  // ::cxx::UnderlyingTypeSpecifierAST::underlyingTypeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->underlyingTypeLoc)));
  // ::cxx::UnderlyingTypeSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::UnderlyingTypeSpecifierAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::UnderlyingTypeSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstElaboratedTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ElaboratedTypeSpecifierAST* self) {
  // ::cxx::ElaboratedTypeSpecifierAST::classLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->classLoc)));
  // ::cxx::ElaboratedTypeSpecifierAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ElaboratedTypeSpecifierAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::ElaboratedTypeSpecifierAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::ElaboratedTypeSpecifierAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::ElaboratedTypeSpecifierAST::classKey
  out.varU32(static_cast<std::uint32_t>(self->classKey));
  // ::cxx::ElaboratedTypeSpecifierAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
  // ::cxx::ElaboratedTypeSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstDecltypeAutoSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::DecltypeAutoSpecifierAST* self) {
  // ::cxx::DecltypeAutoSpecifierAST::decltypeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->decltypeLoc)));
  // ::cxx::DecltypeAutoSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::DecltypeAutoSpecifierAST::autoLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->autoLoc)));
  // ::cxx::DecltypeAutoSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstDecltypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::DecltypeSpecifierAST* self) {
  // ::cxx::DecltypeSpecifierAST::decltypeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->decltypeLoc)));
  // ::cxx::DecltypeSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::DecltypeSpecifierAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::DecltypeSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::DecltypeSpecifierAST::type
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type)));
}

void SemanticEncoder::writeAstPlaceholderTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::PlaceholderTypeSpecifierAST* self) {
  // ::cxx::PlaceholderTypeSpecifierAST::typeConstraint
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeConstraint)));
  // ::cxx::PlaceholderTypeSpecifierAST::specifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->specifier)));
}

void SemanticEncoder::writeAstConstQualifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConstQualifierAST* self) {
  // ::cxx::ConstQualifierAST::constLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->constLoc)));
}

void SemanticEncoder::writeAstVolatileQualifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::VolatileQualifierAST* self) {
  // ::cxx::VolatileQualifierAST::volatileLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->volatileLoc)));
}

void SemanticEncoder::writeAstAtomicQualifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::AtomicQualifierAST* self) {
  // ::cxx::AtomicQualifierAST::atomicLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->atomicLoc)));
}

void SemanticEncoder::writeAstRestrictQualifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::RestrictQualifierAST* self) {
  // ::cxx::RestrictQualifierAST::restrictLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->restrictLoc)));
}

void SemanticEncoder::writeAstEnumSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::EnumSpecifierAST* self) {
  // ::cxx::EnumSpecifierAST::enumLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->enumLoc)));
  // ::cxx::EnumSpecifierAST::classLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->classLoc)));
  // ::cxx::EnumSpecifierAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::EnumSpecifierAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::EnumSpecifierAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::EnumSpecifierAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::EnumSpecifierAST::typeSpecifierList
  writeAstList(out, self->typeSpecifierList);
  // ::cxx::EnumSpecifierAST::lbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbraceLoc)));
  // ::cxx::EnumSpecifierAST::enumeratorList
  writeAstList(out, self->enumeratorList);
  // ::cxx::EnumSpecifierAST::commaLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->commaLoc)));
  // ::cxx::EnumSpecifierAST::rbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbraceLoc)));
  // ::cxx::EnumSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstClassSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ClassSpecifierAST* self) {
  // ::cxx::ClassSpecifierAST::classLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->classLoc)));
  // ::cxx::ClassSpecifierAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ClassSpecifierAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::ClassSpecifierAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::ClassSpecifierAST::finalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->finalLoc)));
  // ::cxx::ClassSpecifierAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::ClassSpecifierAST::baseSpecifierList
  writeAstList(out, self->baseSpecifierList);
  // ::cxx::ClassSpecifierAST::lbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbraceLoc)));
  // ::cxx::ClassSpecifierAST::declarationList
  writeAstList(out, self->declarationList);
  // ::cxx::ClassSpecifierAST::rbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbraceLoc)));
  // ::cxx::ClassSpecifierAST::classKey
  out.varU32(static_cast<std::uint32_t>(self->classKey));
  // ::cxx::ClassSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::ClassSpecifierAST::isFinal
  out.boolean(self->isFinal);
}

void SemanticEncoder::writeAstTypenameSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypenameSpecifierAST* self) {
  // ::cxx::TypenameSpecifierAST::typenameLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->typenameLoc)));
  // ::cxx::TypenameSpecifierAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::TypenameSpecifierAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::TypenameSpecifierAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::TypenameSpecifierAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
  // ::cxx::TypenameSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstSplicerTypeSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::SplicerTypeSpecifierAST* self) {
  // ::cxx::SplicerTypeSpecifierAST::typenameLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->typenameLoc)));
  // ::cxx::SplicerTypeSpecifierAST::splicer
  out.varU32(static_cast<std::uint32_t>(astRef(self->splicer)));
}

void SemanticEncoder::writeAstPointerOperatorAST(
    ByteWriter& out, [[maybe_unused]] cxx::PointerOperatorAST* self) {
  // ::cxx::PointerOperatorAST::starLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->starLoc)));
  // ::cxx::PointerOperatorAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::PointerOperatorAST::cvQualifierList
  writeAstList(out, self->cvQualifierList);
}

void SemanticEncoder::writeAstReferenceOperatorAST(
    ByteWriter& out, [[maybe_unused]] cxx::ReferenceOperatorAST* self) {
  // ::cxx::ReferenceOperatorAST::refLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->refLoc)));
  // ::cxx::ReferenceOperatorAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::ReferenceOperatorAST::refOp
  out.varU32(static_cast<std::uint32_t>(self->refOp));
}

void SemanticEncoder::writeAstPtrToMemberOperatorAST(
    ByteWriter& out, [[maybe_unused]] cxx::PtrToMemberOperatorAST* self) {
  // ::cxx::PtrToMemberOperatorAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::PtrToMemberOperatorAST::starLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->starLoc)));
  // ::cxx::PtrToMemberOperatorAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::PtrToMemberOperatorAST::cvQualifierList
  writeAstList(out, self->cvQualifierList);
}

void SemanticEncoder::writeAstBitfieldDeclaratorAST(
    ByteWriter& out, [[maybe_unused]] cxx::BitfieldDeclaratorAST* self) {
  // ::cxx::BitfieldDeclaratorAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::BitfieldDeclaratorAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::BitfieldDeclaratorAST::sizeExpression
  out.varU32(static_cast<std::uint32_t>(astRef(self->sizeExpression)));
}

void SemanticEncoder::writeAstParameterPackAST(
    ByteWriter& out, [[maybe_unused]] cxx::ParameterPackAST* self) {
  // ::cxx::ParameterPackAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::ParameterPackAST::coreDeclarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->coreDeclarator)));
}

void SemanticEncoder::writeAstIdDeclaratorAST(
    ByteWriter& out, [[maybe_unused]] cxx::IdDeclaratorAST* self) {
  // ::cxx::IdDeclaratorAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::IdDeclaratorAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::IdDeclaratorAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::IdDeclaratorAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::IdDeclaratorAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
}

void SemanticEncoder::writeAstNestedDeclaratorAST(
    ByteWriter& out, [[maybe_unused]] cxx::NestedDeclaratorAST* self) {
  // ::cxx::NestedDeclaratorAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::NestedDeclaratorAST::declarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->declarator)));
  // ::cxx::NestedDeclaratorAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstFunctionDeclaratorChunkAST(
    ByteWriter& out, [[maybe_unused]] cxx::FunctionDeclaratorChunkAST* self) {
  // ::cxx::FunctionDeclaratorChunkAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::FunctionDeclaratorChunkAST::parameterDeclarationClause
  out.varU32(
      static_cast<std::uint32_t>(astRef(self->parameterDeclarationClause)));
  // ::cxx::FunctionDeclaratorChunkAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::FunctionDeclaratorChunkAST::cvQualifierList
  writeAstList(out, self->cvQualifierList);
  // ::cxx::FunctionDeclaratorChunkAST::refLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->refLoc)));
  // ::cxx::FunctionDeclaratorChunkAST::exceptionSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->exceptionSpecifier)));
  // ::cxx::FunctionDeclaratorChunkAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::FunctionDeclaratorChunkAST::trailingReturnType
  out.varU32(static_cast<std::uint32_t>(astRef(self->trailingReturnType)));
  // ::cxx::FunctionDeclaratorChunkAST::refOp
  out.varU32(static_cast<std::uint32_t>(self->refOp));
  // ::cxx::FunctionDeclaratorChunkAST::isFinal
  out.boolean(self->isFinal);
  // ::cxx::FunctionDeclaratorChunkAST::isOverride
  out.boolean(self->isOverride);
  // ::cxx::FunctionDeclaratorChunkAST::isPure
  out.boolean(self->isPure);
}

void SemanticEncoder::writeAstArrayDeclaratorChunkAST(
    ByteWriter& out, [[maybe_unused]] cxx::ArrayDeclaratorChunkAST* self) {
  // ::cxx::ArrayDeclaratorChunkAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::ArrayDeclaratorChunkAST::typeQualifierList
  writeAstList(out, self->typeQualifierList);
  // ::cxx::ArrayDeclaratorChunkAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::ArrayDeclaratorChunkAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
  // ::cxx::ArrayDeclaratorChunkAST::attributeList
  writeAstList(out, self->attributeList);
}

void SemanticEncoder::writeAstNameIdAST(ByteWriter& out,
                                        [[maybe_unused]] cxx::NameIdAST* self) {
  // ::cxx::NameIdAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::NameIdAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writeAstDestructorIdAST(
    ByteWriter& out, [[maybe_unused]] cxx::DestructorIdAST* self) {
  // ::cxx::DestructorIdAST::tildeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->tildeLoc)));
  // ::cxx::DestructorIdAST::id
  out.varU32(static_cast<std::uint32_t>(astRef(self->id)));
}

void SemanticEncoder::writeAstDecltypeIdAST(
    ByteWriter& out, [[maybe_unused]] cxx::DecltypeIdAST* self) {
  // ::cxx::DecltypeIdAST::decltypeSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->decltypeSpecifier)));
}

void SemanticEncoder::writeAstOperatorFunctionIdAST(
    ByteWriter& out, [[maybe_unused]] cxx::OperatorFunctionIdAST* self) {
  // ::cxx::OperatorFunctionIdAST::operatorLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->operatorLoc)));
  // ::cxx::OperatorFunctionIdAST::opLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->opLoc)));
  // ::cxx::OperatorFunctionIdAST::openLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->openLoc)));
  // ::cxx::OperatorFunctionIdAST::closeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->closeLoc)));
  // ::cxx::OperatorFunctionIdAST::op
  out.varU32(static_cast<std::uint32_t>(self->op));
}

void SemanticEncoder::writeAstLiteralOperatorIdAST(
    ByteWriter& out, [[maybe_unused]] cxx::LiteralOperatorIdAST* self) {
  // ::cxx::LiteralOperatorIdAST::operatorLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->operatorLoc)));
  // ::cxx::LiteralOperatorIdAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::LiteralOperatorIdAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::LiteralOperatorIdAST::literal
  writeLiteral(out, self->literal);
  // ::cxx::LiteralOperatorIdAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writeAstConversionFunctionIdAST(
    ByteWriter& out, [[maybe_unused]] cxx::ConversionFunctionIdAST* self) {
  // ::cxx::ConversionFunctionIdAST::operatorLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->operatorLoc)));
  // ::cxx::ConversionFunctionIdAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
}

void SemanticEncoder::writeAstSimpleTemplateIdAST(
    ByteWriter& out, [[maybe_unused]] cxx::SimpleTemplateIdAST* self) {
  // ::cxx::SimpleTemplateIdAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::SimpleTemplateIdAST::lessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lessLoc)));
  // ::cxx::SimpleTemplateIdAST::templateArgumentList
  writeAstList(out, self->templateArgumentList);
  // ::cxx::SimpleTemplateIdAST::greaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->greaterLoc)));
  // ::cxx::SimpleTemplateIdAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::SimpleTemplateIdAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstLiteralOperatorTemplateIdAST(
    ByteWriter& out, [[maybe_unused]] cxx::LiteralOperatorTemplateIdAST* self) {
  // ::cxx::LiteralOperatorTemplateIdAST::literalOperatorId
  out.varU32(static_cast<std::uint32_t>(astRef(self->literalOperatorId)));
  // ::cxx::LiteralOperatorTemplateIdAST::lessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lessLoc)));
  // ::cxx::LiteralOperatorTemplateIdAST::templateArgumentList
  writeAstList(out, self->templateArgumentList);
  // ::cxx::LiteralOperatorTemplateIdAST::greaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->greaterLoc)));
}

void SemanticEncoder::writeAstOperatorFunctionTemplateIdAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::OperatorFunctionTemplateIdAST* self) {
  // ::cxx::OperatorFunctionTemplateIdAST::operatorFunctionId
  out.varU32(static_cast<std::uint32_t>(astRef(self->operatorFunctionId)));
  // ::cxx::OperatorFunctionTemplateIdAST::lessLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lessLoc)));
  // ::cxx::OperatorFunctionTemplateIdAST::templateArgumentList
  writeAstList(out, self->templateArgumentList);
  // ::cxx::OperatorFunctionTemplateIdAST::greaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->greaterLoc)));
}

void SemanticEncoder::writeAstGlobalNestedNameSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::GlobalNestedNameSpecifierAST* self) {
  // ::cxx::NestedNameSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::GlobalNestedNameSpecifierAST::scopeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->scopeLoc)));
}

void SemanticEncoder::writeAstSimpleNestedNameSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::SimpleNestedNameSpecifierAST* self) {
  // ::cxx::NestedNameSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::SimpleNestedNameSpecifierAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::SimpleNestedNameSpecifierAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::SimpleNestedNameSpecifierAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::SimpleNestedNameSpecifierAST::scopeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->scopeLoc)));
}

void SemanticEncoder::writeAstDecltypeNestedNameSpecifierAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::DecltypeNestedNameSpecifierAST* self) {
  // ::cxx::NestedNameSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::DecltypeNestedNameSpecifierAST::decltypeSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->decltypeSpecifier)));
  // ::cxx::DecltypeNestedNameSpecifierAST::scopeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->scopeLoc)));
}

void SemanticEncoder::writeAstTemplateNestedNameSpecifierAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::TemplateNestedNameSpecifierAST* self) {
  // ::cxx::NestedNameSpecifierAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::TemplateNestedNameSpecifierAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::TemplateNestedNameSpecifierAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::TemplateNestedNameSpecifierAST::templateId
  out.varU32(static_cast<std::uint32_t>(astRef(self->templateId)));
  // ::cxx::TemplateNestedNameSpecifierAST::scopeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->scopeLoc)));
  // ::cxx::TemplateNestedNameSpecifierAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
}

void SemanticEncoder::writeAstDefaultFunctionBodyAST(
    ByteWriter& out, [[maybe_unused]] cxx::DefaultFunctionBodyAST* self) {
  // ::cxx::DefaultFunctionBodyAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::DefaultFunctionBodyAST::defaultLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->defaultLoc)));
  // ::cxx::DefaultFunctionBodyAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstCompoundStatementFunctionBodyAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::CompoundStatementFunctionBodyAST* self) {
  // ::cxx::CompoundStatementFunctionBodyAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::CompoundStatementFunctionBodyAST::memInitializerList
  writeAstList(out, self->memInitializerList);
  // ::cxx::CompoundStatementFunctionBodyAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
}

void SemanticEncoder::writeAstTryStatementFunctionBodyAST(
    ByteWriter& out, [[maybe_unused]] cxx::TryStatementFunctionBodyAST* self) {
  // ::cxx::TryStatementFunctionBodyAST::tryLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->tryLoc)));
  // ::cxx::TryStatementFunctionBodyAST::colonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->colonLoc)));
  // ::cxx::TryStatementFunctionBodyAST::memInitializerList
  writeAstList(out, self->memInitializerList);
  // ::cxx::TryStatementFunctionBodyAST::statement
  out.varU32(static_cast<std::uint32_t>(astRef(self->statement)));
  // ::cxx::TryStatementFunctionBodyAST::handlerList
  writeAstList(out, self->handlerList);
}

void SemanticEncoder::writeAstDeleteFunctionBodyAST(
    ByteWriter& out, [[maybe_unused]] cxx::DeleteFunctionBodyAST* self) {
  // ::cxx::DeleteFunctionBodyAST::equalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->equalLoc)));
  // ::cxx::DeleteFunctionBodyAST::deleteLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->deleteLoc)));
  // ::cxx::DeleteFunctionBodyAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstTypeTemplateArgumentAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeTemplateArgumentAST* self) {
  // ::cxx::TypeTemplateArgumentAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
}

void SemanticEncoder::writeAstExpressionTemplateArgumentAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::ExpressionTemplateArgumentAST* self) {
  // ::cxx::ExpressionTemplateArgumentAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
}

void SemanticEncoder::writeAstThrowExceptionSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::ThrowExceptionSpecifierAST* self) {
  // ::cxx::ThrowExceptionSpecifierAST::throwLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->throwLoc)));
  // ::cxx::ThrowExceptionSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::ThrowExceptionSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstNoexceptSpecifierAST(
    ByteWriter& out, [[maybe_unused]] cxx::NoexceptSpecifierAST* self) {
  // ::cxx::NoexceptSpecifierAST::noexceptLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->noexceptLoc)));
  // ::cxx::NoexceptSpecifierAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::NoexceptSpecifierAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::NoexceptSpecifierAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstSimpleRequirementAST(
    ByteWriter& out, [[maybe_unused]] cxx::SimpleRequirementAST* self) {
  // ::cxx::SimpleRequirementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::SimpleRequirementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstCompoundRequirementAST(
    ByteWriter& out, [[maybe_unused]] cxx::CompoundRequirementAST* self) {
  // ::cxx::CompoundRequirementAST::lbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbraceLoc)));
  // ::cxx::CompoundRequirementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::CompoundRequirementAST::rbraceLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbraceLoc)));
  // ::cxx::CompoundRequirementAST::noexceptLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->noexceptLoc)));
  // ::cxx::CompoundRequirementAST::minusGreaterLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->minusGreaterLoc)));
  // ::cxx::CompoundRequirementAST::typeConstraint
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeConstraint)));
  // ::cxx::CompoundRequirementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstTypeRequirementAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeRequirementAST* self) {
  // ::cxx::TypeRequirementAST::typenameLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->typenameLoc)));
  // ::cxx::TypeRequirementAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::TypeRequirementAST::templateLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->templateLoc)));
  // ::cxx::TypeRequirementAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::TypeRequirementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
  // ::cxx::TypeRequirementAST::isTemplateIntroduced
  out.boolean(self->isTemplateIntroduced);
}

void SemanticEncoder::writeAstNestedRequirementAST(
    ByteWriter& out, [[maybe_unused]] cxx::NestedRequirementAST* self) {
  // ::cxx::NestedRequirementAST::requiresLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->requiresLoc)));
  // ::cxx::NestedRequirementAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::NestedRequirementAST::semicolonLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->semicolonLoc)));
}

void SemanticEncoder::writeAstNewParenInitializerAST(
    ByteWriter& out, [[maybe_unused]] cxx::NewParenInitializerAST* self) {
  // ::cxx::NewParenInitializerAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::NewParenInitializerAST::expressionList
  writeAstList(out, self->expressionList);
  // ::cxx::NewParenInitializerAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
}

void SemanticEncoder::writeAstNewBracedInitializerAST(
    ByteWriter& out, [[maybe_unused]] cxx::NewBracedInitializerAST* self) {
  // ::cxx::NewBracedInitializerAST::bracedInitList
  out.varU32(static_cast<std::uint32_t>(astRef(self->bracedInitList)));
}

void SemanticEncoder::writeAstParenMemInitializerAST(
    ByteWriter& out, [[maybe_unused]] cxx::ParenMemInitializerAST* self) {
  // ::cxx::MemInitializerAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::MemInitializerAST::constructor
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructor)));
  // ::cxx::ParenMemInitializerAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::ParenMemInitializerAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::ParenMemInitializerAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::ParenMemInitializerAST::expressionList
  writeAstList(out, self->expressionList);
  // ::cxx::ParenMemInitializerAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::ParenMemInitializerAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
}

void SemanticEncoder::writeAstBracedMemInitializerAST(
    ByteWriter& out, [[maybe_unused]] cxx::BracedMemInitializerAST* self) {
  // ::cxx::MemInitializerAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::MemInitializerAST::constructor
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->constructor)));
  // ::cxx::BracedMemInitializerAST::nestedNameSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->nestedNameSpecifier)));
  // ::cxx::BracedMemInitializerAST::unqualifiedId
  out.varU32(static_cast<std::uint32_t>(astRef(self->unqualifiedId)));
  // ::cxx::BracedMemInitializerAST::bracedInitList
  out.varU32(static_cast<std::uint32_t>(astRef(self->bracedInitList)));
  // ::cxx::BracedMemInitializerAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
}

void SemanticEncoder::writeAstThisLambdaCaptureAST(
    ByteWriter& out, [[maybe_unused]] cxx::ThisLambdaCaptureAST* self) {
  // ::cxx::ThisLambdaCaptureAST::thisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->thisLoc)));
  // ::cxx::ThisLambdaCaptureAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::ThisLambdaCaptureAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstDerefThisLambdaCaptureAST(
    ByteWriter& out, [[maybe_unused]] cxx::DerefThisLambdaCaptureAST* self) {
  // ::cxx::DerefThisLambdaCaptureAST::starLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->starLoc)));
  // ::cxx::DerefThisLambdaCaptureAST::thisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->thisLoc)));
  // ::cxx::DerefThisLambdaCaptureAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstSimpleLambdaCaptureAST(
    ByteWriter& out, [[maybe_unused]] cxx::SimpleLambdaCaptureAST* self) {
  // ::cxx::SimpleLambdaCaptureAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::SimpleLambdaCaptureAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::SimpleLambdaCaptureAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::SimpleLambdaCaptureAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::SimpleLambdaCaptureAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstRefLambdaCaptureAST(
    ByteWriter& out, [[maybe_unused]] cxx::RefLambdaCaptureAST* self) {
  // ::cxx::RefLambdaCaptureAST::ampLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ampLoc)));
  // ::cxx::RefLambdaCaptureAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::RefLambdaCaptureAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::RefLambdaCaptureAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::RefLambdaCaptureAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::RefLambdaCaptureAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstRefInitLambdaCaptureAST(
    ByteWriter& out, [[maybe_unused]] cxx::RefInitLambdaCaptureAST* self) {
  // ::cxx::RefInitLambdaCaptureAST::ampLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ampLoc)));
  // ::cxx::RefInitLambdaCaptureAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::RefInitLambdaCaptureAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::RefInitLambdaCaptureAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::RefInitLambdaCaptureAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::RefInitLambdaCaptureAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstInitLambdaCaptureAST(
    ByteWriter& out, [[maybe_unused]] cxx::InitLambdaCaptureAST* self) {
  // ::cxx::InitLambdaCaptureAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::InitLambdaCaptureAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::InitLambdaCaptureAST::initializer
  out.varU32(static_cast<std::uint32_t>(astRef(self->initializer)));
  // ::cxx::InitLambdaCaptureAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
  // ::cxx::InitLambdaCaptureAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstEllipsisExceptionDeclarationAST(
    ByteWriter& out,
    [[maybe_unused]] cxx::EllipsisExceptionDeclarationAST* self) {
  // ::cxx::EllipsisExceptionDeclarationAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
}

void SemanticEncoder::writeAstTypeExceptionDeclarationAST(
    ByteWriter& out, [[maybe_unused]] cxx::TypeExceptionDeclarationAST* self) {
  // ::cxx::TypeExceptionDeclarationAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::TypeExceptionDeclarationAST::typeSpecifierList
  writeAstList(out, self->typeSpecifierList);
  // ::cxx::TypeExceptionDeclarationAST::declarator
  out.varU32(static_cast<std::uint32_t>(astRef(self->declarator)));
  // ::cxx::TypeExceptionDeclarationAST::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
}

void SemanticEncoder::writeAstCxxAttributeAST(
    ByteWriter& out, [[maybe_unused]] cxx::CxxAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  writeAttributes(out, self->attributes);
  // ::cxx::CxxAttributeAST::lbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracketLoc)));
  // ::cxx::CxxAttributeAST::lbracket2Loc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lbracket2Loc)));
  // ::cxx::CxxAttributeAST::attributeUsingPrefix
  out.varU32(static_cast<std::uint32_t>(astRef(self->attributeUsingPrefix)));
  // ::cxx::CxxAttributeAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::CxxAttributeAST::rbracketLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracketLoc)));
  // ::cxx::CxxAttributeAST::rbracket2Loc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rbracket2Loc)));
}

void SemanticEncoder::writeAstGccAttributeAST(
    ByteWriter& out, [[maybe_unused]] cxx::GccAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  writeAttributes(out, self->attributes);
  // ::cxx::GccAttributeAST::attributeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->attributeLoc)));
  // ::cxx::GccAttributeAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::GccAttributeAST::lparen2Loc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparen2Loc)));
  // ::cxx::GccAttributeAST::attributeList
  writeAstList(out, self->attributeList);
  // ::cxx::GccAttributeAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::GccAttributeAST::rparen2Loc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparen2Loc)));
}

void SemanticEncoder::writeAstAlignasAttributeAST(
    ByteWriter& out, [[maybe_unused]] cxx::AlignasAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  writeAttributes(out, self->attributes);
  // ::cxx::AlignasAttributeAST::alignasLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->alignasLoc)));
  // ::cxx::AlignasAttributeAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::AlignasAttributeAST::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::AlignasAttributeAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::AlignasAttributeAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::AlignasAttributeAST::isPack
  out.boolean(self->isPack);
}

void SemanticEncoder::writeAstAlignasTypeAttributeAST(
    ByteWriter& out, [[maybe_unused]] cxx::AlignasTypeAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  writeAttributes(out, self->attributes);
  // ::cxx::AlignasTypeAttributeAST::alignasLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->alignasLoc)));
  // ::cxx::AlignasTypeAttributeAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::AlignasTypeAttributeAST::typeId
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeId)));
  // ::cxx::AlignasTypeAttributeAST::ellipsisLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->ellipsisLoc)));
  // ::cxx::AlignasTypeAttributeAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::AlignasTypeAttributeAST::isPack
  out.boolean(self->isPack);
}

void SemanticEncoder::writeAstAsmAttributeAST(
    ByteWriter& out, [[maybe_unused]] cxx::AsmAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  writeAttributes(out, self->attributes);
  // ::cxx::AsmAttributeAST::asmLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->asmLoc)));
  // ::cxx::AsmAttributeAST::lparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->lparenLoc)));
  // ::cxx::AsmAttributeAST::literalLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->literalLoc)));
  // ::cxx::AsmAttributeAST::rparenLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->rparenLoc)));
  // ::cxx::AsmAttributeAST::literal
  writeLiteral(out, self->literal);
}

void SemanticEncoder::writeAstScopedAttributeTokenAST(
    ByteWriter& out, [[maybe_unused]] cxx::ScopedAttributeTokenAST* self) {
  // ::cxx::ScopedAttributeTokenAST::attributeNamespaceLoc
  out.varU32(
      static_cast<std::uint32_t>(locationRef(self->attributeNamespaceLoc)));
  // ::cxx::ScopedAttributeTokenAST::scopeLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->scopeLoc)));
  // ::cxx::ScopedAttributeTokenAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::ScopedAttributeTokenAST::attributeNamespace
  out.varU32(
      static_cast<std::uint32_t>(identifierRef(self->attributeNamespace)));
  // ::cxx::ScopedAttributeTokenAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writeAstSimpleAttributeTokenAST(
    ByteWriter& out, [[maybe_unused]] cxx::SimpleAttributeTokenAST* self) {
  // ::cxx::SimpleAttributeTokenAST::identifierLoc
  out.varU32(static_cast<std::uint32_t>(locationRef(self->identifierLoc)));
  // ::cxx::SimpleAttributeTokenAST::identifier
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->identifier)));
}

void SemanticEncoder::writecxxAttribute(
    ByteWriter& out, [[maybe_unused]] const cxx::Attribute* self) {
  // ::cxx::Attribute::attributeNamespace
  out.varU32(
      static_cast<std::uint32_t>(identifierRef(self->attributeNamespace)));
  // ::cxx::Attribute::name
  out.varU32(static_cast<std::uint32_t>(identifierRef(self->name)));
  // ::cxx::Attribute::arguments
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->arguments)));
  for (const auto& element1 : self->arguments) {
    out.varU32(static_cast<std::uint32_t>(identifierRef(element1)));
  }
}

void SemanticEncoder::writecxxMeta(ByteWriter& out,
                                   [[maybe_unused]] const cxx::Meta* self) {
  // ::cxx::Meta::value
  out.u8(static_cast<std::uint8_t>(self->value.index()));
  switch (self->value.index()) {
    case 0: {
      out.varU32(static_cast<std::uint32_t>(typeRef(std::get<0>(self->value))));
      break;
    }
    case 1: {
      out.varU32(
          static_cast<std::uint32_t>(symbolRef(std::get<1>(self->value))));
      break;
    }
    case 2: {
      writecxxMetaConstExpr(out, &std::get<2>(self->value));
      break;
    }
  }
}

void SemanticEncoder::writecxxMetaConstExpr(
    ByteWriter& out, [[maybe_unused]] const cxx::Meta::ConstExpr* self) {
  // ::cxx::Meta::ConstExpr::expression
  out.varU32(static_cast<std::uint32_t>(astRef(self->expression)));
  // ::cxx::Meta::ConstExpr::value
  writeConstValue(out, self->value);
}

void SemanticEncoder::writecxxInitializerList(
    ByteWriter& out, [[maybe_unused]] const cxx::InitializerList* self) {
  // ::cxx::InitializerList::elements
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->elements)));
  for (const auto& element1 : self->elements) {
    writeConstValue(out, std::get<0>(element1));
    out.varU32(static_cast<std::uint32_t>(typeRef(std::get<1>(element1))));
  }
}

void SemanticEncoder::writecxxConstComplex(
    ByteWriter& out, [[maybe_unused]] const cxx::ConstComplex* self) {
  // ::cxx::ConstComplex::real_
  writeConstValue(out, self->real());
  // ::cxx::ConstComplex::imag_
  writeConstValue(out, self->imag());
}

void SemanticEncoder::writecxxConstObject(
    ByteWriter& out, [[maybe_unused]] const cxx::ConstObject* self) {
  // ::cxx::ConstObject::type_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->type())));
  // ::cxx::ConstObject::members_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->members())));
  for (const auto& element1 : self->members()) {
    writecxxConstObjectMember(out, &element1);
  }
}

void SemanticEncoder::writecxxConstObjectMember(
    ByteWriter& out, [[maybe_unused]] const cxx::ConstObject::Member* self) {
  // ::cxx::ConstObject::Member::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::ConstObject::Member::value
  writeConstValue(out, self->value);
}

void SemanticEncoder::writecxxConstAddress(
    ByteWriter& out, [[maybe_unused]] const cxx::ConstAddress* self) {
  // ::cxx::ConstAddress::symbol_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol())));
  // ::cxx::ConstAddress::owner_
  out.varU32(static_cast<std::uint32_t>(constRef(self->owner())));
  // ::cxx::ConstAddress::string_
  writeLiteral(out, self->stringLiteral());
  // ::cxx::ConstAddress::typeInfoFor_
  out.varU32(static_cast<std::uint32_t>(typeRef(self->typeInfoFor())));
  // ::cxx::ConstAddress::offset_
  out.varI64(static_cast<std::int64_t>(self->offset()));
}

void SemanticEncoder::writecxxConstLabelAddress(
    ByteWriter& out, [[maybe_unused]] const cxx::ConstLabelAddress* self) {
  // ::cxx::ConstLabelAddress::name_
  out.varU32(static_cast<std::uint32_t>(stringRef(self->name())));
}

void SemanticEncoder::writecxxTemplateSpecialization(
    ByteWriter& out, [[maybe_unused]] const cxx::TemplateSpecialization* self) {
  // ::cxx::TemplateSpecialization::templateSymbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->templateSymbol)));
  // ::cxx::TemplateSpecialization::arguments
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->arguments)));
  for (const auto& element1 : self->arguments) {
    writeTemplateArgument(out, element1);
  }
  // ::cxx::TemplateSpecialization::symbol
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->symbol)));
  // ::cxx::TemplateSpecialization::instantiationErrors
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->instantiationErrors)));
  for (const auto& element2 : self->instantiationErrors) {
    writecxxInstantiationError(out, &element2);
  }
  // ::cxx::TemplateSpecialization::pendingArgumentList
  writeAstList(out, self->pendingArgumentList);
  // ::cxx::TemplateSpecialization::pendingInstantiationLoc
  out.varU32(
      static_cast<std::uint32_t>(locationRef(self->pendingInstantiationLoc)));
  // ::cxx::TemplateSpecialization::isPendingInstantiation
  out.boolean(self->isPendingInstantiation);
}

void SemanticEncoder::writecxxInstantiationError(
    ByteWriter& out, [[maybe_unused]] const cxx::InstantiationError* self) {
  // ::cxx::InstantiationError::location
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location)));
  // ::cxx::InstantiationError::message
  out.varU32(static_cast<std::uint32_t>(stringRef(self->message)));
  // ::cxx::InstantiationError::severity
  out.varU32(static_cast<std::uint32_t>(self->severity));
}

void SemanticEncoder::writecxxTemplateFriendship(
    ByteWriter& out, [[maybe_unused]] const cxx::TemplateFriendship* self) {
  // ::cxx::TemplateFriendship::arguments
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->arguments)));
  for (const auto& element1 : self->arguments) {
    writeTemplateArgument(out, element1);
  }
  // ::cxx::TemplateFriendship::befriendingClass
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->befriendingClass)));
}

void SemanticEncoder::writecxxClassLayout(
    ByteWriter& out, [[maybe_unused]] const cxx::ClassLayout* self) {
  // ::cxx::ClassLayout::fields_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->sortedFieldInfos())));
  for (const auto& element1 : self->sortedFieldInfos()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1.first)));
    writecxxClassLayoutMemberInfo(out, &element1.second);
  }
  // ::cxx::ClassLayout::bases_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->sortedBaseInfos())));
  for (const auto& element2 : self->sortedBaseInfos()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2.first)));
    writecxxClassLayoutMemberInfo(out, &element2.second);
  }
  // ::cxx::ClassLayout::virtualBases_
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->virtualBases())));
  for (const auto& element3 : self->virtualBases()) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element3)));
  }
  // ::cxx::ClassLayout::padding_
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->padding())));
  for (const auto& element4 : self->padding()) {
    writecxxClassLayoutPaddingInfo(out, &element4);
  }
  // ::cxx::ClassLayout::size_
  out.varU64(static_cast<std::uint64_t>(self->size()));
  // ::cxx::ClassLayout::dataSize_
  out.varU64(static_cast<std::uint64_t>(self->dataSize()));
  // ::cxx::ClassLayout::alignment_
  out.varU64(static_cast<std::uint64_t>(self->alignment()));
  // ::cxx::ClassLayout::nonVirtualSize_
  out.varU64(static_cast<std::uint64_t>(self->nonVirtualSize()));
  // ::cxx::ClassLayout::nonVirtualAlignment_
  out.varU64(static_cast<std::uint64_t>(self->nonVirtualAlignment()));
  // ::cxx::ClassLayout::vtableIndex_
  out.varU32(static_cast<std::uint32_t>(self->vtableIndex()));
  // ::cxx::ClassLayout::primaryBase_
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->primaryBase())));
  // ::cxx::ClassLayout::hasVtable_
  out.boolean(self->hasVtable());
  // ::cxx::ClassLayout::hasDirectVtable_
  out.boolean(self->hasDirectVtable());
  // ::cxx::ClassLayout::primaryBaseIsVirtual_
  out.boolean(self->primaryBaseIsVirtual());
  // ::cxx::ClassLayout::abiEmpty_
  out.boolean(self->isAbiEmpty());
}

void SemanticEncoder::writecxxClassLayoutMemberInfo(
    ByteWriter& out,
    [[maybe_unused]] const cxx::ClassLayout::MemberInfo* self) {
  // ::cxx::ClassLayout::MemberInfo::offset
  out.varU64(static_cast<std::uint64_t>(self->offset));
  // ::cxx::ClassLayout::MemberInfo::index
  out.varU32(static_cast<std::uint32_t>(self->index));
  // ::cxx::ClassLayout::MemberInfo::bitOffset
  out.varU32(static_cast<std::uint32_t>(self->bitOffset));
  // ::cxx::ClassLayout::MemberInfo::bitWidth
  out.varU32(static_cast<std::uint32_t>(self->bitWidth));
  // ::cxx::ClassLayout::MemberInfo::allocUnitSizeBytes
  out.varU32(static_cast<std::uint32_t>(self->allocUnitSizeBytes));
}

void SemanticEncoder::writecxxClassLayoutPaddingInfo(
    ByteWriter& out,
    [[maybe_unused]] const cxx::ClassLayout::PaddingInfo* self) {
  // ::cxx::ClassLayout::PaddingInfo::index
  out.varU32(static_cast<std::uint32_t>(self->index));
  // ::cxx::ClassLayout::PaddingInfo::offset
  out.varU64(static_cast<std::uint64_t>(self->offset));
  // ::cxx::ClassLayout::PaddingInfo::sizeInBytes
  out.varU64(static_cast<std::uint64_t>(self->sizeInBytes));
}

void SemanticEncoder::writecxxVTableLayout(
    ByteWriter& out, [[maybe_unused]] const cxx::VTableLayout* self) {
  // ::cxx::VTableLayout::primary
  writecxxVTableLayoutGroup(out, &self->primary);
  // ::cxx::VTableLayout::secondary
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->secondary)));
  for (const auto& element1 : self->secondary) {
    writecxxVTableLayoutGroup(out, &element1);
  }
  // ::cxx::VTableLayout::keyFunction
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->keyFunction)));
}

void SemanticEncoder::writecxxVTableLayoutGroup(
    ByteWriter& out, [[maybe_unused]] const cxx::VTableLayout::Group* self) {
  // ::cxx::VTableLayout::Group::base
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->base)));
  // ::cxx::VTableLayout::Group::offset
  out.varU64(static_cast<std::uint64_t>(self->offset));
  // ::cxx::VTableLayout::Group::vbaseOffsets
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->vbaseOffsets)));
  for (const auto& element1 : self->vbaseOffsets) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element1.first)));
    out.varI64(static_cast<std::int64_t>(element1.second));
  }
  // ::cxx::VTableLayout::Group::vcallOffsets
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->vcallOffsets)));
  for (const auto& element2 : self->vcallOffsets) {
    out.varU32(static_cast<std::uint32_t>(symbolRef(element2.first)));
    out.varI64(static_cast<std::int64_t>(element2.second));
  }
  // ::cxx::VTableLayout::Group::slots
  out.varU32(static_cast<std::uint32_t>(std::ranges::size(self->slots)));
  for (const auto& element3 : self->slots) {
    writecxxVTableLayoutSlot(out, &element3);
  }
}

void SemanticEncoder::writecxxVTableLayoutSlot(
    ByteWriter& out, [[maybe_unused]] const cxx::VTableLayout::Slot* self) {
  // ::cxx::VTableLayout::Slot::function
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->function)));
  // ::cxx::VTableLayout::Slot::kind
  out.varU32(static_cast<std::uint32_t>(self->kind));
  // ::cxx::VTableLayout::Slot::introducingFunction
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->introducingFunction)));
  // ::cxx::VTableLayout::Slot::vcallBase
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->vcallBase)));
  // ::cxx::VTableLayout::Slot::thisAdjustment
  out.varI64(static_cast<std::int64_t>(self->thisAdjustment));
  // ::cxx::VTableLayout::Slot::vcallOffsetIndex
  out.varI32(static_cast<std::int32_t>(self->vcallOffsetIndex));
  // ::cxx::VTableLayout::Slot::usesVcallOffset
  out.boolean(self->usesVcallOffset);
}

void SemanticEncoder::writecxxPendingBodyInstantiation(
    ByteWriter& out,
    [[maybe_unused]] const cxx::PendingBodyInstantiation* self) {
  // ::cxx::PendingBodyInstantiation::originalDefinition
  out.varU32(static_cast<std::uint32_t>(astRef(self->originalDefinition)));
  // ::cxx::PendingBodyInstantiation::templateArguments
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->templateArguments)));
  for (const auto& element1 : self->templateArguments) {
    writeTemplateArgument(out, element1);
  }
  // ::cxx::PendingBodyInstantiation::parentScope
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parentScope)));
  // ::cxx::PendingBodyInstantiation::depth
  out.varI32(static_cast<std::int32_t>(self->depth));
}

void SemanticEncoder::writecxxPendingExceptionSpecification(
    ByteWriter& out,
    [[maybe_unused]] const cxx::PendingExceptionSpecification* self) {
  // ::cxx::PendingExceptionSpecification::original
  out.varU32(static_cast<std::uint32_t>(astRef(self->original)));
  // ::cxx::PendingExceptionSpecification::instance
  out.varU32(static_cast<std::uint32_t>(astRef(self->instance)));
  // ::cxx::PendingExceptionSpecification::originalFunction
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->originalFunction)));
  // ::cxx::PendingExceptionSpecification::templateArguments
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->templateArguments)));
  for (const auto& element1 : self->templateArguments) {
    writeTemplateArgument(out, element1);
  }
  // ::cxx::PendingExceptionSpecification::parentScope
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parentScope)));
  // ::cxx::PendingExceptionSpecification::depth
  out.varI32(static_cast<std::int32_t>(self->depth));
  // ::cxx::PendingExceptionSpecification::state
  out.varU32(static_cast<std::uint32_t>(self->state));
  // ::cxx::PendingExceptionSpecification::recursionDiagnosed
  out.boolean(self->recursionDiagnosed);
}

void SemanticEncoder::writecxxPendingFieldInitializerInstantiation(
    ByteWriter& out,
    [[maybe_unused]] const cxx::PendingFieldInitializerInstantiation* self) {
  // ::cxx::PendingFieldInitializerInstantiation::unit
  // ::cxx::PendingFieldInitializerInstantiation::pattern
  out.varU32(static_cast<std::uint32_t>(astRef(self->pattern)));
  // ::cxx::PendingFieldInitializerInstantiation::instance
  out.varU32(static_cast<std::uint32_t>(astRef(self->instance)));
  // ::cxx::PendingFieldInitializerInstantiation::typeSpecifier
  out.varU32(static_cast<std::uint32_t>(astRef(self->typeSpecifier)));
  // ::cxx::PendingFieldInitializerInstantiation::templateArguments
  out.varU32(
      static_cast<std::uint32_t>(std::ranges::size(self->templateArguments)));
  for (const auto& element1 : self->templateArguments) {
    writeTemplateArgument(out, element1);
  }
  // ::cxx::PendingFieldInitializerInstantiation::parentScope
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->parentScope)));
  // ::cxx::PendingFieldInitializerInstantiation::depth
  out.varI32(static_cast<std::int32_t>(self->depth));
}

void SemanticEncoder::writecxxDefaultInitializerContext(
    ByteWriter& out,
    [[maybe_unused]] const cxx::DefaultInitializerContext* self) {
  // ::cxx::DefaultInitializerContext::location
  out.varU32(static_cast<std::uint32_t>(locationRef(self->location)));
  // ::cxx::DefaultInitializerContext::scope
  out.varU32(static_cast<std::uint32_t>(symbolRef(self->scope)));
}

auto SemanticDecoder::operator()(const ArchiveReader& archive,
                                 SemanticArchiveRoots& roots) -> bool {
  auto* timeTrace = unit()->timeTrace();

  {
    auto strings = archive.section(ArchiveSection::kStrings);
    TimeTrace::Scope trace{timeTrace, "Decode strings", sectionDetail(strings)};
    if (!readStrings(strings)) return false;
  }

  {
    auto sourceMap = archive.section(ArchiveSection::kSourceMap);
    TimeTrace::Scope trace{timeTrace, "Decode source map",
                           sectionDetail(sourceMap)};
    if (!readSourceMap(sourceMap)) return false;
  }

  {
    auto section = archive.section(ArchiveSection::kNames);
    TimeTrace::Scope trace{timeTrace, "Split kNames records",
                           sectionDetail(section)};
    if (!readRecords(section, nameRecords_)) {
      fail("kNames section is truncated");
      return false;
    }
  }
  {
    auto section = archive.section(ArchiveSection::kTypes);
    TimeTrace::Scope trace{timeTrace, "Split kTypes records",
                           sectionDetail(section)};
    if (!readRecords(section, typeRecords_)) {
      fail("kTypes section is truncated");
      return false;
    }
  }
  {
    auto section = archive.section(ArchiveSection::kSymbols);
    TimeTrace::Scope trace{timeTrace, "Split kSymbols records",
                           sectionDetail(section)};
    if (!readRecords(section, symbolRecords_)) {
      fail("kSymbols section is truncated");
      return false;
    }
  }
  {
    auto section = archive.section(ArchiveSection::kAst);
    TimeTrace::Scope trace{timeTrace, "Split kAst records",
                           sectionDetail(section)};
    if (!readRecords(section, nodeRecords_)) {
      fail("kAst section is truncated");
      return false;
    }
  }
  {
    auto section = archive.section(ArchiveSection::kConstants);
    TimeTrace::Scope trace{timeTrace, "Split kConstants records",
                           sectionDetail(section)};
    if (!readRecords(section, constRecords_)) {
      fail("kConstants section is truncated");
      return false;
    }
  }

  names_.assign(nameRecords_.size(), nullptr);
  nameDecoded_.assign(nameRecords_.size(), false);
  types_.assign(typeRecords_.size(), nullptr);
  typeDecoded_.assign(typeRecords_.size(), false);
  constants_.assign(constRecords_.size(), nullptr);

  {
    TimeTrace::Scope trace{timeTrace, "Allocate symbols",
                           countDetail(symbolRecords_.size())};
    symbols_.reserve(symbolRecords_.size());
    for (const auto& record : symbolRecords_) {
      ByteReader in{record.bytes};
      const auto kind = static_cast<cxx::SymbolKind>(readEnum(in, 27));
      if (!ok()) return false;
      auto symbol = allocateSymbol(kind);
      if (!symbol) {
        fail("archive names a symbol kind this compiler cannot allocate");
        return false;
      }
      symbols_.push_back(symbol);
    }
  }

  {
    TimeTrace::Scope trace{timeTrace, "Allocate AST nodes",
                           countDetail(nodeRecords_.size())};
    nodes_.reserve(nodeRecords_.size());
    for (const auto& record : nodeRecords_) {
      ByteReader in{record.bytes};
      const auto kind = static_cast<cxx::ASTKind>(readEnum(in, 245));
      if (!ok()) return false;
      auto node = allocateAst(kind);
      if (!node) {
        fail("archive names an AST kind this compiler cannot allocate");
        return false;
      }
      nodes_.push_back(node);
    }
  }

  {
    TimeTrace::Scope trace{timeTrace, "Decode symbols",
                           countDetail(symbolRecords_.size())};
    for (std::size_t i = 0; ok() && i < symbolRecords_.size(); ++i) {
      ByteReader in{symbolRecords_[i].bytes};
      const auto kind = readEnum(in, 27);
      decodeSymbolFields(in, symbols_[i]);
      if (ok() && !in.atEnd()) {
        fail(std::format("symbol record {} of kind {} has {} trailing bytes", i,
                         kind, in.remaining()));
      }
    }
  }

  {
    TimeTrace::Scope trace{timeTrace, "Decode AST nodes",
                           countDetail(nodeRecords_.size())};
    for (std::size_t i = 0; ok() && i < nodeRecords_.size(); ++i) {
      ByteReader in{nodeRecords_[i].bytes};
      const auto kind = readEnum(in, 245);
      decodeAstFields(in, nodes_[i]);
      if (ok() && !in.atEnd()) {
        fail(std::format("AST record {} of kind {} has {} trailing bytes", i,
                         kind, in.remaining()));
      }
    }
  }

  {
    TimeTrace::Scope trace{timeTrace, "Rebuild lookup tables"};
    for (auto symbol : symbols_) {
      if (auto scope = symbol_cast<ScopeSymbol>(symbol))
        scope->rebuildLookupTable();
    }
  }

  TimeTrace::Scope sessionTrace{timeTrace, "Decode session"};
  auto session = archive.section(ArchiveSection::kSession);

  roots.globalScope =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{session.varU32()}));
  roots.ast = ast_cast<UnitAST>(astAt(AstRef{session.varU32()}));
  roots.anonymousIdCount = session.varI32();
  roots.closureNameCount = session.varI32();
  roots.prefixTokenCount = session.varU32();

  {
    const auto count = session.varCount(1);
    for (std::uint32_t i = 0; ok() && i < count; ++i) {
      auto entry =
          symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{session.varU32()}));
      if (entry) roots.pendingBodyCompletions.push_back(entry);
    }
  }
  {
    const auto count = session.varCount(1);
    for (std::uint32_t i = 0; ok() && i < count; ++i) {
      auto entry =
          symbol_cast<ClassSymbol>(symbolAt(SymbolRef{session.varU32()}));
      if (entry) roots.pendingMemberInstantiations.push_back(entry);
    }
  }
  {
    const auto count = session.varCount(1);
    for (std::uint32_t i = 0; ok() && i < count; ++i) {
      auto entry =
          symbol_cast<ClassSymbol>(symbolAt(SymbolRef{session.varU32()}));
      if (entry) roots.instantiatedMemberClasses.push_back(entry);
    }
  }
  {
    const auto count = session.varCount(2);
    for (std::uint32_t i = 0; ok() && i < count; ++i) {
      const auto key = session.varU64();
      roots.snippets.emplace_back(
          key, std::string{stringAt(StringRef{session.varU32()})});
    }
  }

  if (!session.ok()) fail("session section is truncated");

  return ok();
}

auto SemanticDecoder::allocateSymbol(cxx::SymbolKind kind) -> cxx::Symbol* {
  switch (kind) {
    case cxx::SymbolKind::kNamespace:
      return control()->newNamespaceSymbol(nullptr, {});
    case cxx::SymbolKind::kNamespaceAlias:
      return control()->newNamespaceAliasSymbol(nullptr, {});
    case cxx::SymbolKind::kConcept:
      return control()->newConceptSymbol(nullptr, {});
    case cxx::SymbolKind::kDeductionGuide:
      return control()->newDeductionGuideSymbol(nullptr, {});
    case cxx::SymbolKind::kClass:
      return control()->newClassSymbol(nullptr, {});
    case cxx::SymbolKind::kEnum:
      return control()->newEnumSymbol(nullptr, {});
    case cxx::SymbolKind::kScopedEnum:
      return control()->newScopedEnumSymbol(nullptr, {});
    case cxx::SymbolKind::kFunction:
      return control()->newFunctionSymbol(nullptr, {});
    case cxx::SymbolKind::kTypeAlias:
      return control()->newTypeAliasSymbol(nullptr, {});
    case cxx::SymbolKind::kVariable:
      return control()->newVariableSymbol(nullptr, {});
    case cxx::SymbolKind::kField:
      return control()->newFieldSymbol(nullptr, {});
    case cxx::SymbolKind::kParameter:
      return control()->newParameterSymbol(nullptr, {});
    case cxx::SymbolKind::kParameterPack:
      return control()->newParameterPackSymbol(nullptr, {});
    case cxx::SymbolKind::kEnumerator:
      return control()->newEnumeratorSymbol(nullptr, {});
    case cxx::SymbolKind::kFunctionParameters:
      return control()->newFunctionParametersSymbol(nullptr, {});
    case cxx::SymbolKind::kTemplateParameters:
      return control()->newTemplateParametersSymbol(nullptr, {});
    case cxx::SymbolKind::kBlock:
      return control()->newBlockSymbol(nullptr, {});
    case cxx::SymbolKind::kLambda:
      return control()->newLambdaSymbol(nullptr, {});
    case cxx::SymbolKind::kTypeParameter:
      return control()->newTypeParameterSymbol(nullptr, {}, {}, {}, {});
    case cxx::SymbolKind::kNonTypeParameter:
      return control()->newNonTypeParameterSymbol(nullptr, {});
    case cxx::SymbolKind::kTemplateTypeParameter:
      return control()->newTemplateTypeParameterSymbol(nullptr, {}, {}, {}, {},
                                                       {});
    case cxx::SymbolKind::kConstraintTypeParameter:
      return control()->newConstraintTypeParameterSymbol(nullptr, {}, {}, {},
                                                         {});
    case cxx::SymbolKind::kOverloadSet:
      return control()->newOverloadSetSymbol(nullptr, {});
    case cxx::SymbolKind::kBaseClass:
      return control()->newBaseClassSymbol(nullptr, {});
    case cxx::SymbolKind::kInjectedClassName:
      return control()->newInjectedClassNameSymbol(nullptr, {});
    case cxx::SymbolKind::kUnresolved:
      return control()->newUnresolvedSymbol(nullptr, {});
    case cxx::SymbolKind::kUsingDeclaration:
      return control()->newUsingDeclarationSymbol(nullptr, {});
  }
  return nullptr;
}

auto SemanticDecoder::allocateAst(cxx::ASTKind kind) -> cxx::AST* {
  switch (kind) {
    case cxx::ASTKind::TranslationUnit:
      return cxx::TranslationUnitAST::create(arena());
    case cxx::ASTKind::ModuleUnit:
      return cxx::ModuleUnitAST::create(arena());
    case cxx::ASTKind::SimpleDeclaration:
      return cxx::SimpleDeclarationAST::create(arena());
    case cxx::ASTKind::AsmDeclaration:
      return cxx::AsmDeclarationAST::create(arena());
    case cxx::ASTKind::NamespaceAliasDefinition:
      return cxx::NamespaceAliasDefinitionAST::create(arena());
    case cxx::ASTKind::UsingDeclaration:
      return cxx::UsingDeclarationAST::create(arena());
    case cxx::ASTKind::UsingEnumDeclaration:
      return cxx::UsingEnumDeclarationAST::create(arena());
    case cxx::ASTKind::UsingDirective:
      return cxx::UsingDirectiveAST::create(arena());
    case cxx::ASTKind::StaticAssertDeclaration:
      return cxx::StaticAssertDeclarationAST::create(arena());
    case cxx::ASTKind::AliasDeclaration:
      return cxx::AliasDeclarationAST::create(arena());
    case cxx::ASTKind::OpaqueEnumDeclaration:
      return cxx::OpaqueEnumDeclarationAST::create(arena());
    case cxx::ASTKind::FunctionDefinition:
      return cxx::FunctionDefinitionAST::create(arena());
    case cxx::ASTKind::TemplateDeclaration:
      return cxx::TemplateDeclarationAST::create(arena());
    case cxx::ASTKind::ConceptDefinition:
      return cxx::ConceptDefinitionAST::create(arena());
    case cxx::ASTKind::DeductionGuide:
      return cxx::DeductionGuideAST::create(arena());
    case cxx::ASTKind::ExplicitInstantiation:
      return cxx::ExplicitInstantiationAST::create(arena());
    case cxx::ASTKind::ExportDeclaration:
      return cxx::ExportDeclarationAST::create(arena());
    case cxx::ASTKind::ExportCompoundDeclaration:
      return cxx::ExportCompoundDeclarationAST::create(arena());
    case cxx::ASTKind::LinkageSpecification:
      return cxx::LinkageSpecificationAST::create(arena());
    case cxx::ASTKind::NamespaceDefinition:
      return cxx::NamespaceDefinitionAST::create(arena());
    case cxx::ASTKind::EmptyDeclaration:
      return cxx::EmptyDeclarationAST::create(arena());
    case cxx::ASTKind::AttributeDeclaration:
      return cxx::AttributeDeclarationAST::create(arena());
    case cxx::ASTKind::ModuleImportDeclaration:
      return cxx::ModuleImportDeclarationAST::create(arena());
    case cxx::ASTKind::ParameterDeclaration:
      return cxx::ParameterDeclarationAST::create(arena());
    case cxx::ASTKind::AccessDeclaration:
      return cxx::AccessDeclarationAST::create(arena());
    case cxx::ASTKind::ForRangeDeclaration:
      return cxx::ForRangeDeclarationAST::create(arena());
    case cxx::ASTKind::StructuredBindingDeclaration:
      return cxx::StructuredBindingDeclarationAST::create(arena());
    case cxx::ASTKind::AsmOperand:
      return cxx::AsmOperandAST::create(arena());
    case cxx::ASTKind::AsmQualifier:
      return cxx::AsmQualifierAST::create(arena());
    case cxx::ASTKind::AsmClobber:
      return cxx::AsmClobberAST::create(arena());
    case cxx::ASTKind::AsmGotoLabel:
      return cxx::AsmGotoLabelAST::create(arena());
    case cxx::ASTKind::Splicer:
      return cxx::SplicerAST::create(arena());
    case cxx::ASTKind::GlobalModuleFragment:
      return cxx::GlobalModuleFragmentAST::create(arena());
    case cxx::ASTKind::PrivateModuleFragment:
      return cxx::PrivateModuleFragmentAST::create(arena());
    case cxx::ASTKind::ModuleDeclaration:
      return cxx::ModuleDeclarationAST::create(arena());
    case cxx::ASTKind::ModuleName:
      return cxx::ModuleNameAST::create(arena());
    case cxx::ASTKind::ModuleQualifier:
      return cxx::ModuleQualifierAST::create(arena());
    case cxx::ASTKind::ModulePartition:
      return cxx::ModulePartitionAST::create(arena());
    case cxx::ASTKind::ImportName:
      return cxx::ImportNameAST::create(arena());
    case cxx::ASTKind::InitDeclarator:
      return cxx::InitDeclaratorAST::create(arena());
    case cxx::ASTKind::Declarator:
      return cxx::DeclaratorAST::create(arena());
    case cxx::ASTKind::UsingDeclarator:
      return cxx::UsingDeclaratorAST::create(arena());
    case cxx::ASTKind::Enumerator:
      return cxx::EnumeratorAST::create(arena());
    case cxx::ASTKind::TypeId:
      return cxx::TypeIdAST::create(arena());
    case cxx::ASTKind::Handler:
      return cxx::HandlerAST::create(arena());
    case cxx::ASTKind::BaseSpecifier:
      return cxx::BaseSpecifierAST::create(arena());
    case cxx::ASTKind::RequiresClause:
      return cxx::RequiresClauseAST::create(arena());
    case cxx::ASTKind::ParameterDeclarationClause:
      return cxx::ParameterDeclarationClauseAST::create(arena());
    case cxx::ASTKind::TrailingReturnType:
      return cxx::TrailingReturnTypeAST::create(arena());
    case cxx::ASTKind::LambdaSpecifier:
      return cxx::LambdaSpecifierAST::create(arena());
    case cxx::ASTKind::TypeConstraint:
      return cxx::TypeConstraintAST::create(arena());
    case cxx::ASTKind::AttributeArgumentClause:
      return cxx::AttributeArgumentClauseAST::create(arena());
    case cxx::ASTKind::Attribute:
      return cxx::AttributeAST::create(arena());
    case cxx::ASTKind::AttributeUsingPrefix:
      return cxx::AttributeUsingPrefixAST::create(arena());
    case cxx::ASTKind::NewPlacement:
      return cxx::NewPlacementAST::create(arena());
    case cxx::ASTKind::NestedNamespaceSpecifier:
      return cxx::NestedNamespaceSpecifierAST::create(arena());
    case cxx::ASTKind::LabeledStatement:
      return cxx::LabeledStatementAST::create(arena());
    case cxx::ASTKind::CaseStatement:
      return cxx::CaseStatementAST::create(arena());
    case cxx::ASTKind::DefaultStatement:
      return cxx::DefaultStatementAST::create(arena());
    case cxx::ASTKind::ExpressionStatement:
      return cxx::ExpressionStatementAST::create(arena());
    case cxx::ASTKind::CompoundStatement:
      return cxx::CompoundStatementAST::create(arena());
    case cxx::ASTKind::IfStatement:
      return cxx::IfStatementAST::create(arena());
    case cxx::ASTKind::ConstevalIfStatement:
      return cxx::ConstevalIfStatementAST::create(arena());
    case cxx::ASTKind::SwitchStatement:
      return cxx::SwitchStatementAST::create(arena());
    case cxx::ASTKind::WhileStatement:
      return cxx::WhileStatementAST::create(arena());
    case cxx::ASTKind::DoStatement:
      return cxx::DoStatementAST::create(arena());
    case cxx::ASTKind::ForRangeStatement:
      return cxx::ForRangeStatementAST::create(arena());
    case cxx::ASTKind::ForStatement:
      return cxx::ForStatementAST::create(arena());
    case cxx::ASTKind::BreakStatement:
      return cxx::BreakStatementAST::create(arena());
    case cxx::ASTKind::ContinueStatement:
      return cxx::ContinueStatementAST::create(arena());
    case cxx::ASTKind::ReturnStatement:
      return cxx::ReturnStatementAST::create(arena());
    case cxx::ASTKind::CoroutineReturnStatement:
      return cxx::CoroutineReturnStatementAST::create(arena());
    case cxx::ASTKind::GotoStatement:
      return cxx::GotoStatementAST::create(arena());
    case cxx::ASTKind::DeclarationStatement:
      return cxx::DeclarationStatementAST::create(arena());
    case cxx::ASTKind::TryBlockStatement:
      return cxx::TryBlockStatementAST::create(arena());
    case cxx::ASTKind::CharLiteralExpression:
      return cxx::CharLiteralExpressionAST::create(arena());
    case cxx::ASTKind::BoolLiteralExpression:
      return cxx::BoolLiteralExpressionAST::create(arena());
    case cxx::ASTKind::IntLiteralExpression:
      return cxx::IntLiteralExpressionAST::create(arena());
    case cxx::ASTKind::FloatLiteralExpression:
      return cxx::FloatLiteralExpressionAST::create(arena());
    case cxx::ASTKind::NullptrLiteralExpression:
      return cxx::NullptrLiteralExpressionAST::create(arena());
    case cxx::ASTKind::StringLiteralExpression:
      return cxx::StringLiteralExpressionAST::create(arena());
    case cxx::ASTKind::UserDefinedStringLiteralExpression:
      return cxx::UserDefinedStringLiteralExpressionAST::create(arena());
    case cxx::ASTKind::ObjectLiteralExpression:
      return cxx::ObjectLiteralExpressionAST::create(arena());
    case cxx::ASTKind::ThisExpression:
      return cxx::ThisExpressionAST::create(arena());
    case cxx::ASTKind::PackIndexExpression:
      return cxx::PackIndexExpressionAST::create(arena());
    case cxx::ASTKind::GenericSelectionExpression:
      return cxx::GenericSelectionExpressionAST::create(arena());
    case cxx::ASTKind::NestedStatementExpression:
      return cxx::NestedStatementExpressionAST::create(arena());
    case cxx::ASTKind::DefaultInitializerExpression:
      return cxx::DefaultInitializerExpressionAST::create(arena());
    case cxx::ASTKind::NestedExpression:
      return cxx::NestedExpressionAST::create(arena());
    case cxx::ASTKind::IdExpression:
      return cxx::IdExpressionAST::create(arena());
    case cxx::ASTKind::LambdaExpression:
      return cxx::LambdaExpressionAST::create(arena());
    case cxx::ASTKind::FoldExpression:
      return cxx::FoldExpressionAST::create(arena());
    case cxx::ASTKind::RightFoldExpression:
      return cxx::RightFoldExpressionAST::create(arena());
    case cxx::ASTKind::LeftFoldExpression:
      return cxx::LeftFoldExpressionAST::create(arena());
    case cxx::ASTKind::RequiresExpression:
      return cxx::RequiresExpressionAST::create(arena());
    case cxx::ASTKind::VaArgExpression:
      return cxx::VaArgExpressionAST::create(arena());
    case cxx::ASTKind::SubscriptExpression:
      return cxx::SubscriptExpressionAST::create(arena());
    case cxx::ASTKind::CallExpression:
      return cxx::CallExpressionAST::create(arena());
    case cxx::ASTKind::TypeConstruction:
      return cxx::TypeConstructionAST::create(arena());
    case cxx::ASTKind::BracedTypeConstruction:
      return cxx::BracedTypeConstructionAST::create(arena());
    case cxx::ASTKind::SpliceMemberExpression:
      return cxx::SpliceMemberExpressionAST::create(arena());
    case cxx::ASTKind::MemberExpression:
      return cxx::MemberExpressionAST::create(arena());
    case cxx::ASTKind::PostIncrExpression:
      return cxx::PostIncrExpressionAST::create(arena());
    case cxx::ASTKind::CppCastExpression:
      return cxx::CppCastExpressionAST::create(arena());
    case cxx::ASTKind::BuiltinBitCastExpression:
      return cxx::BuiltinBitCastExpressionAST::create(arena());
    case cxx::ASTKind::BuiltinOffsetofExpression:
      return cxx::BuiltinOffsetofExpressionAST::create(arena());
    case cxx::ASTKind::TypeidExpression:
      return cxx::TypeidExpressionAST::create(arena());
    case cxx::ASTKind::TypeidOfTypeExpression:
      return cxx::TypeidOfTypeExpressionAST::create(arena());
    case cxx::ASTKind::SpliceExpression:
      return cxx::SpliceExpressionAST::create(arena());
    case cxx::ASTKind::GlobalScopeReflectExpression:
      return cxx::GlobalScopeReflectExpressionAST::create(arena());
    case cxx::ASTKind::NamespaceReflectExpression:
      return cxx::NamespaceReflectExpressionAST::create(arena());
    case cxx::ASTKind::TypeIdReflectExpression:
      return cxx::TypeIdReflectExpressionAST::create(arena());
    case cxx::ASTKind::ReflectExpression:
      return cxx::ReflectExpressionAST::create(arena());
    case cxx::ASTKind::LabelAddressExpression:
      return cxx::LabelAddressExpressionAST::create(arena());
    case cxx::ASTKind::UnaryExpression:
      return cxx::UnaryExpressionAST::create(arena());
    case cxx::ASTKind::AwaitExpression:
      return cxx::AwaitExpressionAST::create(arena());
    case cxx::ASTKind::SizeofExpression:
      return cxx::SizeofExpressionAST::create(arena());
    case cxx::ASTKind::SizeofTypeExpression:
      return cxx::SizeofTypeExpressionAST::create(arena());
    case cxx::ASTKind::SizeofPackExpression:
      return cxx::SizeofPackExpressionAST::create(arena());
    case cxx::ASTKind::AlignofTypeExpression:
      return cxx::AlignofTypeExpressionAST::create(arena());
    case cxx::ASTKind::AlignofExpression:
      return cxx::AlignofExpressionAST::create(arena());
    case cxx::ASTKind::NoexceptExpression:
      return cxx::NoexceptExpressionAST::create(arena());
    case cxx::ASTKind::NewExpression:
      return cxx::NewExpressionAST::create(arena());
    case cxx::ASTKind::DeleteExpression:
      return cxx::DeleteExpressionAST::create(arena());
    case cxx::ASTKind::CastExpression:
      return cxx::CastExpressionAST::create(arena());
    case cxx::ASTKind::ImplicitCastExpression:
      return cxx::ImplicitCastExpressionAST::create(arena());
    case cxx::ASTKind::ConstExpression:
      return cxx::ConstExpressionAST::create(arena());
    case cxx::ASTKind::BinaryExpression:
      return cxx::BinaryExpressionAST::create(arena());
    case cxx::ASTKind::ConditionalExpression:
      return cxx::ConditionalExpressionAST::create(arena());
    case cxx::ASTKind::YieldExpression:
      return cxx::YieldExpressionAST::create(arena());
    case cxx::ASTKind::ThrowExpression:
      return cxx::ThrowExpressionAST::create(arena());
    case cxx::ASTKind::AssignmentExpression:
      return cxx::AssignmentExpressionAST::create(arena());
    case cxx::ASTKind::TargetExpression:
      return cxx::TargetExpressionAST::create(arena());
    case cxx::ASTKind::RightExpression:
      return cxx::RightExpressionAST::create(arena());
    case cxx::ASTKind::CompoundAssignmentExpression:
      return cxx::CompoundAssignmentExpressionAST::create(arena());
    case cxx::ASTKind::PackExpansionExpression:
      return cxx::PackExpansionExpressionAST::create(arena());
    case cxx::ASTKind::DesignatedInitializerClause:
      return cxx::DesignatedInitializerClauseAST::create(arena());
    case cxx::ASTKind::TypeTraitExpression:
      return cxx::TypeTraitExpressionAST::create(arena());
    case cxx::ASTKind::ConditionExpression:
      return cxx::ConditionExpressionAST::create(arena());
    case cxx::ASTKind::EqualInitializer:
      return cxx::EqualInitializerAST::create(arena());
    case cxx::ASTKind::BracedInitList:
      return cxx::BracedInitListAST::create(arena());
    case cxx::ASTKind::ParenInitializer:
      return cxx::ParenInitializerAST::create(arena());
    case cxx::ASTKind::ThreeWayComparisonExpression:
      return cxx::ThreeWayComparisonExpressionAST::create(arena());
    case cxx::ASTKind::DefaultGenericAssociation:
      return cxx::DefaultGenericAssociationAST::create(arena());
    case cxx::ASTKind::TypeGenericAssociation:
      return cxx::TypeGenericAssociationAST::create(arena());
    case cxx::ASTKind::DotDesignator:
      return cxx::DotDesignatorAST::create(arena());
    case cxx::ASTKind::SubscriptDesignator:
      return cxx::SubscriptDesignatorAST::create(arena());
    case cxx::ASTKind::TemplateTypeParameter:
      return cxx::TemplateTypeParameterAST::create(arena());
    case cxx::ASTKind::NonTypeTemplateParameter:
      return cxx::NonTypeTemplateParameterAST::create(arena());
    case cxx::ASTKind::TypenameTypeParameter:
      return cxx::TypenameTypeParameterAST::create(arena());
    case cxx::ASTKind::ConstraintTypeParameter:
      return cxx::ConstraintTypeParameterAST::create(arena());
    case cxx::ASTKind::TypedefSpecifier:
      return cxx::TypedefSpecifierAST::create(arena());
    case cxx::ASTKind::FriendSpecifier:
      return cxx::FriendSpecifierAST::create(arena());
    case cxx::ASTKind::ConstevalSpecifier:
      return cxx::ConstevalSpecifierAST::create(arena());
    case cxx::ASTKind::ConstinitSpecifier:
      return cxx::ConstinitSpecifierAST::create(arena());
    case cxx::ASTKind::ConstexprSpecifier:
      return cxx::ConstexprSpecifierAST::create(arena());
    case cxx::ASTKind::InlineSpecifier:
      return cxx::InlineSpecifierAST::create(arena());
    case cxx::ASTKind::NoreturnSpecifier:
      return cxx::NoreturnSpecifierAST::create(arena());
    case cxx::ASTKind::StaticSpecifier:
      return cxx::StaticSpecifierAST::create(arena());
    case cxx::ASTKind::ExternSpecifier:
      return cxx::ExternSpecifierAST::create(arena());
    case cxx::ASTKind::RegisterSpecifier:
      return cxx::RegisterSpecifierAST::create(arena());
    case cxx::ASTKind::ThreadLocalSpecifier:
      return cxx::ThreadLocalSpecifierAST::create(arena());
    case cxx::ASTKind::ThreadSpecifier:
      return cxx::ThreadSpecifierAST::create(arena());
    case cxx::ASTKind::MutableSpecifier:
      return cxx::MutableSpecifierAST::create(arena());
    case cxx::ASTKind::VirtualSpecifier:
      return cxx::VirtualSpecifierAST::create(arena());
    case cxx::ASTKind::ExplicitSpecifier:
      return cxx::ExplicitSpecifierAST::create(arena());
    case cxx::ASTKind::AutoTypeSpecifier:
      return cxx::AutoTypeSpecifierAST::create(arena());
    case cxx::ASTKind::VoidTypeSpecifier:
      return cxx::VoidTypeSpecifierAST::create(arena());
    case cxx::ASTKind::SizeTypeSpecifier:
      return cxx::SizeTypeSpecifierAST::create(arena());
    case cxx::ASTKind::SignTypeSpecifier:
      return cxx::SignTypeSpecifierAST::create(arena());
    case cxx::ASTKind::BuiltinTypeSpecifier:
      return cxx::BuiltinTypeSpecifierAST::create(arena());
    case cxx::ASTKind::UnaryBuiltinTypeSpecifier:
      return cxx::UnaryBuiltinTypeSpecifierAST::create(arena());
    case cxx::ASTKind::BinaryBuiltinTypeSpecifier:
      return cxx::BinaryBuiltinTypeSpecifierAST::create(arena());
    case cxx::ASTKind::IntegralTypeSpecifier:
      return cxx::IntegralTypeSpecifierAST::create(arena());
    case cxx::ASTKind::FloatingPointTypeSpecifier:
      return cxx::FloatingPointTypeSpecifierAST::create(arena());
    case cxx::ASTKind::ComplexTypeSpecifier:
      return cxx::ComplexTypeSpecifierAST::create(arena());
    case cxx::ASTKind::NamedTypeSpecifier:
      return cxx::NamedTypeSpecifierAST::create(arena());
    case cxx::ASTKind::AtomicTypeSpecifier:
      return cxx::AtomicTypeSpecifierAST::create(arena());
    case cxx::ASTKind::BitIntTypeSpecifier:
      return cxx::BitIntTypeSpecifierAST::create(arena());
    case cxx::ASTKind::UnderlyingTypeSpecifier:
      return cxx::UnderlyingTypeSpecifierAST::create(arena());
    case cxx::ASTKind::ElaboratedTypeSpecifier:
      return cxx::ElaboratedTypeSpecifierAST::create(arena());
    case cxx::ASTKind::DecltypeAutoSpecifier:
      return cxx::DecltypeAutoSpecifierAST::create(arena());
    case cxx::ASTKind::DecltypeSpecifier:
      return cxx::DecltypeSpecifierAST::create(arena());
    case cxx::ASTKind::PlaceholderTypeSpecifier:
      return cxx::PlaceholderTypeSpecifierAST::create(arena());
    case cxx::ASTKind::ConstQualifier:
      return cxx::ConstQualifierAST::create(arena());
    case cxx::ASTKind::VolatileQualifier:
      return cxx::VolatileQualifierAST::create(arena());
    case cxx::ASTKind::AtomicQualifier:
      return cxx::AtomicQualifierAST::create(arena());
    case cxx::ASTKind::RestrictQualifier:
      return cxx::RestrictQualifierAST::create(arena());
    case cxx::ASTKind::EnumSpecifier:
      return cxx::EnumSpecifierAST::create(arena());
    case cxx::ASTKind::ClassSpecifier:
      return cxx::ClassSpecifierAST::create(arena());
    case cxx::ASTKind::TypenameSpecifier:
      return cxx::TypenameSpecifierAST::create(arena());
    case cxx::ASTKind::SplicerTypeSpecifier:
      return cxx::SplicerTypeSpecifierAST::create(arena());
    case cxx::ASTKind::PointerOperator:
      return cxx::PointerOperatorAST::create(arena());
    case cxx::ASTKind::ReferenceOperator:
      return cxx::ReferenceOperatorAST::create(arena());
    case cxx::ASTKind::PtrToMemberOperator:
      return cxx::PtrToMemberOperatorAST::create(arena());
    case cxx::ASTKind::BitfieldDeclarator:
      return cxx::BitfieldDeclaratorAST::create(arena());
    case cxx::ASTKind::ParameterPack:
      return cxx::ParameterPackAST::create(arena());
    case cxx::ASTKind::IdDeclarator:
      return cxx::IdDeclaratorAST::create(arena());
    case cxx::ASTKind::NestedDeclarator:
      return cxx::NestedDeclaratorAST::create(arena());
    case cxx::ASTKind::FunctionDeclaratorChunk:
      return cxx::FunctionDeclaratorChunkAST::create(arena());
    case cxx::ASTKind::ArrayDeclaratorChunk:
      return cxx::ArrayDeclaratorChunkAST::create(arena());
    case cxx::ASTKind::NameId:
      return cxx::NameIdAST::create(arena());
    case cxx::ASTKind::DestructorId:
      return cxx::DestructorIdAST::create(arena());
    case cxx::ASTKind::DecltypeId:
      return cxx::DecltypeIdAST::create(arena());
    case cxx::ASTKind::OperatorFunctionId:
      return cxx::OperatorFunctionIdAST::create(arena());
    case cxx::ASTKind::LiteralOperatorId:
      return cxx::LiteralOperatorIdAST::create(arena());
    case cxx::ASTKind::ConversionFunctionId:
      return cxx::ConversionFunctionIdAST::create(arena());
    case cxx::ASTKind::SimpleTemplateId:
      return cxx::SimpleTemplateIdAST::create(arena());
    case cxx::ASTKind::LiteralOperatorTemplateId:
      return cxx::LiteralOperatorTemplateIdAST::create(arena());
    case cxx::ASTKind::OperatorFunctionTemplateId:
      return cxx::OperatorFunctionTemplateIdAST::create(arena());
    case cxx::ASTKind::GlobalNestedNameSpecifier:
      return cxx::GlobalNestedNameSpecifierAST::create(arena());
    case cxx::ASTKind::SimpleNestedNameSpecifier:
      return cxx::SimpleNestedNameSpecifierAST::create(arena());
    case cxx::ASTKind::DecltypeNestedNameSpecifier:
      return cxx::DecltypeNestedNameSpecifierAST::create(arena());
    case cxx::ASTKind::TemplateNestedNameSpecifier:
      return cxx::TemplateNestedNameSpecifierAST::create(arena());
    case cxx::ASTKind::DefaultFunctionBody:
      return cxx::DefaultFunctionBodyAST::create(arena());
    case cxx::ASTKind::CompoundStatementFunctionBody:
      return cxx::CompoundStatementFunctionBodyAST::create(arena());
    case cxx::ASTKind::TryStatementFunctionBody:
      return cxx::TryStatementFunctionBodyAST::create(arena());
    case cxx::ASTKind::DeleteFunctionBody:
      return cxx::DeleteFunctionBodyAST::create(arena());
    case cxx::ASTKind::TypeTemplateArgument:
      return cxx::TypeTemplateArgumentAST::create(arena());
    case cxx::ASTKind::ExpressionTemplateArgument:
      return cxx::ExpressionTemplateArgumentAST::create(arena());
    case cxx::ASTKind::ThrowExceptionSpecifier:
      return cxx::ThrowExceptionSpecifierAST::create(arena());
    case cxx::ASTKind::NoexceptSpecifier:
      return cxx::NoexceptSpecifierAST::create(arena());
    case cxx::ASTKind::SimpleRequirement:
      return cxx::SimpleRequirementAST::create(arena());
    case cxx::ASTKind::CompoundRequirement:
      return cxx::CompoundRequirementAST::create(arena());
    case cxx::ASTKind::TypeRequirement:
      return cxx::TypeRequirementAST::create(arena());
    case cxx::ASTKind::NestedRequirement:
      return cxx::NestedRequirementAST::create(arena());
    case cxx::ASTKind::NewParenInitializer:
      return cxx::NewParenInitializerAST::create(arena());
    case cxx::ASTKind::NewBracedInitializer:
      return cxx::NewBracedInitializerAST::create(arena());
    case cxx::ASTKind::ParenMemInitializer:
      return cxx::ParenMemInitializerAST::create(arena());
    case cxx::ASTKind::BracedMemInitializer:
      return cxx::BracedMemInitializerAST::create(arena());
    case cxx::ASTKind::ThisLambdaCapture:
      return cxx::ThisLambdaCaptureAST::create(arena());
    case cxx::ASTKind::DerefThisLambdaCapture:
      return cxx::DerefThisLambdaCaptureAST::create(arena());
    case cxx::ASTKind::SimpleLambdaCapture:
      return cxx::SimpleLambdaCaptureAST::create(arena());
    case cxx::ASTKind::RefLambdaCapture:
      return cxx::RefLambdaCaptureAST::create(arena());
    case cxx::ASTKind::RefInitLambdaCapture:
      return cxx::RefInitLambdaCaptureAST::create(arena());
    case cxx::ASTKind::InitLambdaCapture:
      return cxx::InitLambdaCaptureAST::create(arena());
    case cxx::ASTKind::EllipsisExceptionDeclaration:
      return cxx::EllipsisExceptionDeclarationAST::create(arena());
    case cxx::ASTKind::TypeExceptionDeclaration:
      return cxx::TypeExceptionDeclarationAST::create(arena());
    case cxx::ASTKind::CxxAttribute:
      return cxx::CxxAttributeAST::create(arena());
    case cxx::ASTKind::GccAttribute:
      return cxx::GccAttributeAST::create(arena());
    case cxx::ASTKind::AlignasAttribute:
      return cxx::AlignasAttributeAST::create(arena());
    case cxx::ASTKind::AlignasTypeAttribute:
      return cxx::AlignasTypeAttributeAST::create(arena());
    case cxx::ASTKind::AsmAttribute:
      return cxx::AsmAttributeAST::create(arena());
    case cxx::ASTKind::ScopedAttributeToken:
      return cxx::ScopedAttributeTokenAST::create(arena());
    case cxx::ASTKind::SimpleAttributeToken:
      return cxx::SimpleAttributeTokenAST::create(arena());
  }
  return nullptr;
}

auto SemanticDecoder::nameAt(NameRef ref) -> const cxx::Name* {
  const auto index = static_cast<std::uint32_t>(ref);
  if (index == 0) return nullptr;
  if (index > nameRecords_.size()) {
    fail("name reference is out of range");
    return nullptr;
  }
  if (nameDecoded_[index - 1]) return names_[index - 1];
  nameDecoded_[index - 1] = true;
  ByteReader in{nameRecords_[index - 1].bytes};
  const auto kind = static_cast<cxx::NameKind>(readEnum(in, 6));
  const cxx::Name* name = nullptr;
  switch (kind) {
    case cxx::NameKind::kIdentifier:
      name = readNameIdentifier(in);
      break;
    case cxx::NameKind::kOperatorId:
      name = readNameOperatorId(in);
      break;
    case cxx::NameKind::kDestructorId:
      name = readNameDestructorId(in);
      break;
    case cxx::NameKind::kLiteralOperatorId:
      name = readNameLiteralOperatorId(in);
      break;
    case cxx::NameKind::kConversionFunctionId:
      name = readNameConversionFunctionId(in);
      break;
    case cxx::NameKind::kTemplateId:
      name = readNameTemplateId(in);
      break;
  }
  names_[index - 1] = name;
  return name;
}

auto SemanticDecoder::typeAt(TypeRef ref) -> const cxx::Type* {
  const auto index = static_cast<std::uint32_t>(ref);
  if (index == 0) return nullptr;
  if (index > typeRecords_.size()) {
    fail("type reference is out of range");
    return nullptr;
  }
  if (typeDecoded_[index - 1]) return types_[index - 1];
  typeDecoded_[index - 1] = true;
  ByteReader in{typeRecords_[index - 1].bytes};
  const auto kind = static_cast<cxx::TypeKind>(readEnum(in, 55));
  const cxx::Type* type = nullptr;
  switch (kind) {
    case cxx::TypeKind::kVoid:
      type = readTypeVoidType(in);
      break;
    case cxx::TypeKind::kNullptr:
      type = readTypeNullptrType(in);
      break;
    case cxx::TypeKind::kDecltypeAuto:
      type = readTypeDecltypeAutoType(in);
      break;
    case cxx::TypeKind::kAuto:
      type = readTypeAutoType(in);
      break;
    case cxx::TypeKind::kBool:
      type = readTypeBoolType(in);
      break;
    case cxx::TypeKind::kSignedChar:
      type = readTypeSignedCharType(in);
      break;
    case cxx::TypeKind::kShortInt:
      type = readTypeShortIntType(in);
      break;
    case cxx::TypeKind::kInt:
      type = readTypeIntType(in);
      break;
    case cxx::TypeKind::kLongInt:
      type = readTypeLongIntType(in);
      break;
    case cxx::TypeKind::kLongLongInt:
      type = readTypeLongLongIntType(in);
      break;
    case cxx::TypeKind::kInt128:
      type = readTypeInt128Type(in);
      break;
    case cxx::TypeKind::kUnsignedChar:
      type = readTypeUnsignedCharType(in);
      break;
    case cxx::TypeKind::kUnsignedShortInt:
      type = readTypeUnsignedShortIntType(in);
      break;
    case cxx::TypeKind::kUnsignedInt:
      type = readTypeUnsignedIntType(in);
      break;
    case cxx::TypeKind::kUnsignedLongInt:
      type = readTypeUnsignedLongIntType(in);
      break;
    case cxx::TypeKind::kUnsignedLongLongInt:
      type = readTypeUnsignedLongLongIntType(in);
      break;
    case cxx::TypeKind::kUnsignedInt128:
      type = readTypeUnsignedInt128Type(in);
      break;
    case cxx::TypeKind::kChar:
      type = readTypeCharType(in);
      break;
    case cxx::TypeKind::kChar8:
      type = readTypeChar8Type(in);
      break;
    case cxx::TypeKind::kChar16:
      type = readTypeChar16Type(in);
      break;
    case cxx::TypeKind::kChar32:
      type = readTypeChar32Type(in);
      break;
    case cxx::TypeKind::kWideChar:
      type = readTypeWideCharType(in);
      break;
    case cxx::TypeKind::kFloat:
      type = readTypeFloatType(in);
      break;
    case cxx::TypeKind::kDouble:
      type = readTypeDoubleType(in);
      break;
    case cxx::TypeKind::kLongDouble:
      type = readTypeLongDoubleType(in);
      break;
    case cxx::TypeKind::kFloat16:
      type = readTypeFloat16Type(in);
      break;
    case cxx::TypeKind::kQual:
      type = readTypeQualType(in);
      break;
    case cxx::TypeKind::kBoundedArray:
      type = readTypeBoundedArrayType(in);
      break;
    case cxx::TypeKind::kUnboundedArray:
      type = readTypeUnboundedArrayType(in);
      break;
    case cxx::TypeKind::kPointer:
      type = readTypePointerType(in);
      break;
    case cxx::TypeKind::kLvalueReference:
      type = readTypeLvalueReferenceType(in);
      break;
    case cxx::TypeKind::kRvalueReference:
      type = readTypeRvalueReferenceType(in);
      break;
    case cxx::TypeKind::kFunction:
      type = readTypeFunctionType(in);
      break;
    case cxx::TypeKind::kClass:
      type = readTypeClassType(in);
      break;
    case cxx::TypeKind::kEnum:
      type = readTypeEnumType(in);
      break;
    case cxx::TypeKind::kScopedEnum:
      type = readTypeScopedEnumType(in);
      break;
    case cxx::TypeKind::kMemberObjectPointer:
      type = readTypeMemberObjectPointerType(in);
      break;
    case cxx::TypeKind::kMemberFunctionPointer:
      type = readTypeMemberFunctionPointerType(in);
      break;
    case cxx::TypeKind::kNamespace:
      type = readTypeNamespaceType(in);
      break;
    case cxx::TypeKind::kTypeParameter:
      type = readTypeTypeParameterType(in);
      break;
    case cxx::TypeKind::kTemplateTypeParameter:
      type = readTypeTemplateTypeParameterType(in);
      break;
    case cxx::TypeKind::kUnresolvedName:
      type = readTypeUnresolvedNameType(in);
      break;
    case cxx::TypeKind::kUnresolvedBoundedArray:
      type = readTypeUnresolvedBoundedArrayType(in);
      break;
    case cxx::TypeKind::kUnresolvedUnderlying:
      type = readTypeUnresolvedUnderlyingType(in);
      break;
    case cxx::TypeKind::kUnresolvedBuiltin:
      type = readTypeUnresolvedBuiltinType(in);
      break;
    case cxx::TypeKind::kOverloadSet:
      type = readTypeOverloadSetType(in);
      break;
    case cxx::TypeKind::kBuiltinVaList:
      type = readTypeBuiltinVaListType(in);
      break;
    case cxx::TypeKind::kBuiltinMetaInfo:
      type = readTypeBuiltinMetaInfoType(in);
      break;
    case cxx::TypeKind::kBitInt:
      type = readTypeBitIntType(in);
      break;
    case cxx::TypeKind::kUnsignedBitInt:
      type = readTypeUnsignedBitIntType(in);
      break;
    case cxx::TypeKind::kUnresolvedBitInt:
      type = readTypeUnresolvedBitIntType(in);
      break;
    case cxx::TypeKind::kVector:
      type = readTypeVectorType(in);
      break;
    case cxx::TypeKind::kUnresolvedVector:
      type = readTypeUnresolvedVectorType(in);
      break;
    case cxx::TypeKind::kComplex:
      type = readTypeComplexType(in);
      break;
    case cxx::TypeKind::kAtomic:
      type = readTypeAtomicType(in);
      break;
  }
  types_[index - 1] = type;
  return type;
}

auto SemanticDecoder::symbolAt(SymbolRef ref) -> cxx::Symbol* {
  const auto index = static_cast<std::uint32_t>(ref);
  if (index == 0) return nullptr;
  if (index > symbols_.size()) {
    fail("symbol reference is out of range");
    return nullptr;
  }
  return symbols_[index - 1];
}

auto SemanticDecoder::astAt(AstRef ref) -> cxx::AST* {
  const auto index = static_cast<std::uint32_t>(ref);
  if (index == 0) return nullptr;
  if (index > nodes_.size()) {
    fail("AST reference is out of range");
    return nullptr;
  }
  return nodes_[index - 1];
}

void SemanticDecoder::decodeSymbolFields(ByteReader& in, cxx::Symbol* symbol) {
  switch (symbol->kind()) {
    case cxx::SymbolKind::kNamespace:
      readSymbolNamespaceSymbol(in, static_cast<cxx::NamespaceSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kNamespaceAlias:
      readSymbolNamespaceAliasSymbol(
          in, static_cast<cxx::NamespaceAliasSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kConcept:
      readSymbolConceptSymbol(in, static_cast<cxx::ConceptSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kDeductionGuide:
      readSymbolDeductionGuideSymbol(
          in, static_cast<cxx::DeductionGuideSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kClass:
      readSymbolClassSymbol(in, static_cast<cxx::ClassSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kEnum:
      readSymbolEnumSymbol(in, static_cast<cxx::EnumSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kScopedEnum:
      readSymbolScopedEnumSymbol(in,
                                 static_cast<cxx::ScopedEnumSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kFunction:
      readSymbolFunctionSymbol(in, static_cast<cxx::FunctionSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kTypeAlias:
      readSymbolTypeAliasSymbol(in, static_cast<cxx::TypeAliasSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kVariable:
      readSymbolVariableSymbol(in, static_cast<cxx::VariableSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kField:
      readSymbolFieldSymbol(in, static_cast<cxx::FieldSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kParameter:
      readSymbolParameterSymbol(in, static_cast<cxx::ParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kParameterPack:
      readSymbolParameterPackSymbol(
          in, static_cast<cxx::ParameterPackSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kEnumerator:
      readSymbolEnumeratorSymbol(in,
                                 static_cast<cxx::EnumeratorSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kFunctionParameters:
      readSymbolFunctionParametersSymbol(
          in, static_cast<cxx::FunctionParametersSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kTemplateParameters:
      readSymbolTemplateParametersSymbol(
          in, static_cast<cxx::TemplateParametersSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kBlock:
      readSymbolBlockSymbol(in, static_cast<cxx::BlockSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kLambda:
      readSymbolLambdaSymbol(in, static_cast<cxx::LambdaSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kTypeParameter:
      readSymbolTypeParameterSymbol(
          in, static_cast<cxx::TypeParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kNonTypeParameter:
      readSymbolNonTypeParameterSymbol(
          in, static_cast<cxx::NonTypeParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kTemplateTypeParameter:
      readSymbolTemplateTypeParameterSymbol(
          in, static_cast<cxx::TemplateTypeParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kConstraintTypeParameter:
      readSymbolConstraintTypeParameterSymbol(
          in, static_cast<cxx::ConstraintTypeParameterSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kOverloadSet:
      readSymbolOverloadSetSymbol(in,
                                  static_cast<cxx::OverloadSetSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kBaseClass:
      readSymbolBaseClassSymbol(in, static_cast<cxx::BaseClassSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kInjectedClassName:
      readSymbolInjectedClassNameSymbol(
          in, static_cast<cxx::InjectedClassNameSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kUnresolved:
      readSymbolUnresolvedSymbol(in,
                                 static_cast<cxx::UnresolvedSymbol*>(symbol));
      break;
    case cxx::SymbolKind::kUsingDeclaration:
      readSymbolUsingDeclarationSymbol(
          in, static_cast<cxx::UsingDeclarationSymbol*>(symbol));
      break;
  }
}

void SemanticDecoder::decodeAstFields(ByteReader& in, cxx::AST* ast) {
  switch (ast->kind()) {
    case cxx::ASTKind::TranslationUnit:
      readAstTranslationUnitAST(in, static_cast<cxx::TranslationUnitAST*>(ast));
      break;
    case cxx::ASTKind::ModuleUnit:
      readAstModuleUnitAST(in, static_cast<cxx::ModuleUnitAST*>(ast));
      break;
    case cxx::ASTKind::SimpleDeclaration:
      readAstSimpleDeclarationAST(in,
                                  static_cast<cxx::SimpleDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AsmDeclaration:
      readAstAsmDeclarationAST(in, static_cast<cxx::AsmDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::NamespaceAliasDefinition:
      readAstNamespaceAliasDefinitionAST(
          in, static_cast<cxx::NamespaceAliasDefinitionAST*>(ast));
      break;
    case cxx::ASTKind::UsingDeclaration:
      readAstUsingDeclarationAST(in,
                                 static_cast<cxx::UsingDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::UsingEnumDeclaration:
      readAstUsingEnumDeclarationAST(
          in, static_cast<cxx::UsingEnumDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::UsingDirective:
      readAstUsingDirectiveAST(in, static_cast<cxx::UsingDirectiveAST*>(ast));
      break;
    case cxx::ASTKind::StaticAssertDeclaration:
      readAstStaticAssertDeclarationAST(
          in, static_cast<cxx::StaticAssertDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AliasDeclaration:
      readAstAliasDeclarationAST(in,
                                 static_cast<cxx::AliasDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::OpaqueEnumDeclaration:
      readAstOpaqueEnumDeclarationAST(
          in, static_cast<cxx::OpaqueEnumDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::FunctionDefinition:
      readAstFunctionDefinitionAST(
          in, static_cast<cxx::FunctionDefinitionAST*>(ast));
      break;
    case cxx::ASTKind::TemplateDeclaration:
      readAstTemplateDeclarationAST(
          in, static_cast<cxx::TemplateDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ConceptDefinition:
      readAstConceptDefinitionAST(in,
                                  static_cast<cxx::ConceptDefinitionAST*>(ast));
      break;
    case cxx::ASTKind::DeductionGuide:
      readAstDeductionGuideAST(in, static_cast<cxx::DeductionGuideAST*>(ast));
      break;
    case cxx::ASTKind::ExplicitInstantiation:
      readAstExplicitInstantiationAST(
          in, static_cast<cxx::ExplicitInstantiationAST*>(ast));
      break;
    case cxx::ASTKind::ExportDeclaration:
      readAstExportDeclarationAST(in,
                                  static_cast<cxx::ExportDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ExportCompoundDeclaration:
      readAstExportCompoundDeclarationAST(
          in, static_cast<cxx::ExportCompoundDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::LinkageSpecification:
      readAstLinkageSpecificationAST(
          in, static_cast<cxx::LinkageSpecificationAST*>(ast));
      break;
    case cxx::ASTKind::NamespaceDefinition:
      readAstNamespaceDefinitionAST(
          in, static_cast<cxx::NamespaceDefinitionAST*>(ast));
      break;
    case cxx::ASTKind::EmptyDeclaration:
      readAstEmptyDeclarationAST(in,
                                 static_cast<cxx::EmptyDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AttributeDeclaration:
      readAstAttributeDeclarationAST(
          in, static_cast<cxx::AttributeDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ModuleImportDeclaration:
      readAstModuleImportDeclarationAST(
          in, static_cast<cxx::ModuleImportDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ParameterDeclaration:
      readAstParameterDeclarationAST(
          in, static_cast<cxx::ParameterDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AccessDeclaration:
      readAstAccessDeclarationAST(in,
                                  static_cast<cxx::AccessDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ForRangeDeclaration:
      readAstForRangeDeclarationAST(
          in, static_cast<cxx::ForRangeDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::StructuredBindingDeclaration:
      readAstStructuredBindingDeclarationAST(
          in, static_cast<cxx::StructuredBindingDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::AsmOperand:
      readAstAsmOperandAST(in, static_cast<cxx::AsmOperandAST*>(ast));
      break;
    case cxx::ASTKind::AsmQualifier:
      readAstAsmQualifierAST(in, static_cast<cxx::AsmQualifierAST*>(ast));
      break;
    case cxx::ASTKind::AsmClobber:
      readAstAsmClobberAST(in, static_cast<cxx::AsmClobberAST*>(ast));
      break;
    case cxx::ASTKind::AsmGotoLabel:
      readAstAsmGotoLabelAST(in, static_cast<cxx::AsmGotoLabelAST*>(ast));
      break;
    case cxx::ASTKind::Splicer:
      readAstSplicerAST(in, static_cast<cxx::SplicerAST*>(ast));
      break;
    case cxx::ASTKind::GlobalModuleFragment:
      readAstGlobalModuleFragmentAST(
          in, static_cast<cxx::GlobalModuleFragmentAST*>(ast));
      break;
    case cxx::ASTKind::PrivateModuleFragment:
      readAstPrivateModuleFragmentAST(
          in, static_cast<cxx::PrivateModuleFragmentAST*>(ast));
      break;
    case cxx::ASTKind::ModuleDeclaration:
      readAstModuleDeclarationAST(in,
                                  static_cast<cxx::ModuleDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::ModuleName:
      readAstModuleNameAST(in, static_cast<cxx::ModuleNameAST*>(ast));
      break;
    case cxx::ASTKind::ModuleQualifier:
      readAstModuleQualifierAST(in, static_cast<cxx::ModuleQualifierAST*>(ast));
      break;
    case cxx::ASTKind::ModulePartition:
      readAstModulePartitionAST(in, static_cast<cxx::ModulePartitionAST*>(ast));
      break;
    case cxx::ASTKind::ImportName:
      readAstImportNameAST(in, static_cast<cxx::ImportNameAST*>(ast));
      break;
    case cxx::ASTKind::InitDeclarator:
      readAstInitDeclaratorAST(in, static_cast<cxx::InitDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::Declarator:
      readAstDeclaratorAST(in, static_cast<cxx::DeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::UsingDeclarator:
      readAstUsingDeclaratorAST(in, static_cast<cxx::UsingDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::Enumerator:
      readAstEnumeratorAST(in, static_cast<cxx::EnumeratorAST*>(ast));
      break;
    case cxx::ASTKind::TypeId:
      readAstTypeIdAST(in, static_cast<cxx::TypeIdAST*>(ast));
      break;
    case cxx::ASTKind::Handler:
      readAstHandlerAST(in, static_cast<cxx::HandlerAST*>(ast));
      break;
    case cxx::ASTKind::BaseSpecifier:
      readAstBaseSpecifierAST(in, static_cast<cxx::BaseSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::RequiresClause:
      readAstRequiresClauseAST(in, static_cast<cxx::RequiresClauseAST*>(ast));
      break;
    case cxx::ASTKind::ParameterDeclarationClause:
      readAstParameterDeclarationClauseAST(
          in, static_cast<cxx::ParameterDeclarationClauseAST*>(ast));
      break;
    case cxx::ASTKind::TrailingReturnType:
      readAstTrailingReturnTypeAST(
          in, static_cast<cxx::TrailingReturnTypeAST*>(ast));
      break;
    case cxx::ASTKind::LambdaSpecifier:
      readAstLambdaSpecifierAST(in, static_cast<cxx::LambdaSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::TypeConstraint:
      readAstTypeConstraintAST(in, static_cast<cxx::TypeConstraintAST*>(ast));
      break;
    case cxx::ASTKind::AttributeArgumentClause:
      readAstAttributeArgumentClauseAST(
          in, static_cast<cxx::AttributeArgumentClauseAST*>(ast));
      break;
    case cxx::ASTKind::Attribute:
      readAstAttributeAST(in, static_cast<cxx::AttributeAST*>(ast));
      break;
    case cxx::ASTKind::AttributeUsingPrefix:
      readAstAttributeUsingPrefixAST(
          in, static_cast<cxx::AttributeUsingPrefixAST*>(ast));
      break;
    case cxx::ASTKind::NewPlacement:
      readAstNewPlacementAST(in, static_cast<cxx::NewPlacementAST*>(ast));
      break;
    case cxx::ASTKind::NestedNamespaceSpecifier:
      readAstNestedNamespaceSpecifierAST(
          in, static_cast<cxx::NestedNamespaceSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::LabeledStatement:
      readAstLabeledStatementAST(in,
                                 static_cast<cxx::LabeledStatementAST*>(ast));
      break;
    case cxx::ASTKind::CaseStatement:
      readAstCaseStatementAST(in, static_cast<cxx::CaseStatementAST*>(ast));
      break;
    case cxx::ASTKind::DefaultStatement:
      readAstDefaultStatementAST(in,
                                 static_cast<cxx::DefaultStatementAST*>(ast));
      break;
    case cxx::ASTKind::ExpressionStatement:
      readAstExpressionStatementAST(
          in, static_cast<cxx::ExpressionStatementAST*>(ast));
      break;
    case cxx::ASTKind::CompoundStatement:
      readAstCompoundStatementAST(in,
                                  static_cast<cxx::CompoundStatementAST*>(ast));
      break;
    case cxx::ASTKind::IfStatement:
      readAstIfStatementAST(in, static_cast<cxx::IfStatementAST*>(ast));
      break;
    case cxx::ASTKind::ConstevalIfStatement:
      readAstConstevalIfStatementAST(
          in, static_cast<cxx::ConstevalIfStatementAST*>(ast));
      break;
    case cxx::ASTKind::SwitchStatement:
      readAstSwitchStatementAST(in, static_cast<cxx::SwitchStatementAST*>(ast));
      break;
    case cxx::ASTKind::WhileStatement:
      readAstWhileStatementAST(in, static_cast<cxx::WhileStatementAST*>(ast));
      break;
    case cxx::ASTKind::DoStatement:
      readAstDoStatementAST(in, static_cast<cxx::DoStatementAST*>(ast));
      break;
    case cxx::ASTKind::ForRangeStatement:
      readAstForRangeStatementAST(in,
                                  static_cast<cxx::ForRangeStatementAST*>(ast));
      break;
    case cxx::ASTKind::ForStatement:
      readAstForStatementAST(in, static_cast<cxx::ForStatementAST*>(ast));
      break;
    case cxx::ASTKind::BreakStatement:
      readAstBreakStatementAST(in, static_cast<cxx::BreakStatementAST*>(ast));
      break;
    case cxx::ASTKind::ContinueStatement:
      readAstContinueStatementAST(in,
                                  static_cast<cxx::ContinueStatementAST*>(ast));
      break;
    case cxx::ASTKind::ReturnStatement:
      readAstReturnStatementAST(in, static_cast<cxx::ReturnStatementAST*>(ast));
      break;
    case cxx::ASTKind::CoroutineReturnStatement:
      readAstCoroutineReturnStatementAST(
          in, static_cast<cxx::CoroutineReturnStatementAST*>(ast));
      break;
    case cxx::ASTKind::GotoStatement:
      readAstGotoStatementAST(in, static_cast<cxx::GotoStatementAST*>(ast));
      break;
    case cxx::ASTKind::DeclarationStatement:
      readAstDeclarationStatementAST(
          in, static_cast<cxx::DeclarationStatementAST*>(ast));
      break;
    case cxx::ASTKind::TryBlockStatement:
      readAstTryBlockStatementAST(in,
                                  static_cast<cxx::TryBlockStatementAST*>(ast));
      break;
    case cxx::ASTKind::CharLiteralExpression:
      readAstCharLiteralExpressionAST(
          in, static_cast<cxx::CharLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::BoolLiteralExpression:
      readAstBoolLiteralExpressionAST(
          in, static_cast<cxx::BoolLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::IntLiteralExpression:
      readAstIntLiteralExpressionAST(
          in, static_cast<cxx::IntLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::FloatLiteralExpression:
      readAstFloatLiteralExpressionAST(
          in, static_cast<cxx::FloatLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NullptrLiteralExpression:
      readAstNullptrLiteralExpressionAST(
          in, static_cast<cxx::NullptrLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::StringLiteralExpression:
      readAstStringLiteralExpressionAST(
          in, static_cast<cxx::StringLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::UserDefinedStringLiteralExpression:
      readAstUserDefinedStringLiteralExpressionAST(
          in, static_cast<cxx::UserDefinedStringLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ObjectLiteralExpression:
      readAstObjectLiteralExpressionAST(
          in, static_cast<cxx::ObjectLiteralExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ThisExpression:
      readAstThisExpressionAST(in, static_cast<cxx::ThisExpressionAST*>(ast));
      break;
    case cxx::ASTKind::PackIndexExpression:
      readAstPackIndexExpressionAST(
          in, static_cast<cxx::PackIndexExpressionAST*>(ast));
      break;
    case cxx::ASTKind::GenericSelectionExpression:
      readAstGenericSelectionExpressionAST(
          in, static_cast<cxx::GenericSelectionExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NestedStatementExpression:
      readAstNestedStatementExpressionAST(
          in, static_cast<cxx::NestedStatementExpressionAST*>(ast));
      break;
    case cxx::ASTKind::DefaultInitializerExpression:
      readAstDefaultInitializerExpressionAST(
          in, static_cast<cxx::DefaultInitializerExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NestedExpression:
      readAstNestedExpressionAST(in,
                                 static_cast<cxx::NestedExpressionAST*>(ast));
      break;
    case cxx::ASTKind::IdExpression:
      readAstIdExpressionAST(in, static_cast<cxx::IdExpressionAST*>(ast));
      break;
    case cxx::ASTKind::LambdaExpression:
      readAstLambdaExpressionAST(in,
                                 static_cast<cxx::LambdaExpressionAST*>(ast));
      break;
    case cxx::ASTKind::FoldExpression:
      readAstFoldExpressionAST(in, static_cast<cxx::FoldExpressionAST*>(ast));
      break;
    case cxx::ASTKind::RightFoldExpression:
      readAstRightFoldExpressionAST(
          in, static_cast<cxx::RightFoldExpressionAST*>(ast));
      break;
    case cxx::ASTKind::LeftFoldExpression:
      readAstLeftFoldExpressionAST(
          in, static_cast<cxx::LeftFoldExpressionAST*>(ast));
      break;
    case cxx::ASTKind::RequiresExpression:
      readAstRequiresExpressionAST(
          in, static_cast<cxx::RequiresExpressionAST*>(ast));
      break;
    case cxx::ASTKind::VaArgExpression:
      readAstVaArgExpressionAST(in, static_cast<cxx::VaArgExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SubscriptExpression:
      readAstSubscriptExpressionAST(
          in, static_cast<cxx::SubscriptExpressionAST*>(ast));
      break;
    case cxx::ASTKind::CallExpression:
      readAstCallExpressionAST(in, static_cast<cxx::CallExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TypeConstruction:
      readAstTypeConstructionAST(in,
                                 static_cast<cxx::TypeConstructionAST*>(ast));
      break;
    case cxx::ASTKind::BracedTypeConstruction:
      readAstBracedTypeConstructionAST(
          in, static_cast<cxx::BracedTypeConstructionAST*>(ast));
      break;
    case cxx::ASTKind::SpliceMemberExpression:
      readAstSpliceMemberExpressionAST(
          in, static_cast<cxx::SpliceMemberExpressionAST*>(ast));
      break;
    case cxx::ASTKind::MemberExpression:
      readAstMemberExpressionAST(in,
                                 static_cast<cxx::MemberExpressionAST*>(ast));
      break;
    case cxx::ASTKind::PostIncrExpression:
      readAstPostIncrExpressionAST(
          in, static_cast<cxx::PostIncrExpressionAST*>(ast));
      break;
    case cxx::ASTKind::CppCastExpression:
      readAstCppCastExpressionAST(in,
                                  static_cast<cxx::CppCastExpressionAST*>(ast));
      break;
    case cxx::ASTKind::BuiltinBitCastExpression:
      readAstBuiltinBitCastExpressionAST(
          in, static_cast<cxx::BuiltinBitCastExpressionAST*>(ast));
      break;
    case cxx::ASTKind::BuiltinOffsetofExpression:
      readAstBuiltinOffsetofExpressionAST(
          in, static_cast<cxx::BuiltinOffsetofExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TypeidExpression:
      readAstTypeidExpressionAST(in,
                                 static_cast<cxx::TypeidExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TypeidOfTypeExpression:
      readAstTypeidOfTypeExpressionAST(
          in, static_cast<cxx::TypeidOfTypeExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SpliceExpression:
      readAstSpliceExpressionAST(in,
                                 static_cast<cxx::SpliceExpressionAST*>(ast));
      break;
    case cxx::ASTKind::GlobalScopeReflectExpression:
      readAstGlobalScopeReflectExpressionAST(
          in, static_cast<cxx::GlobalScopeReflectExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NamespaceReflectExpression:
      readAstNamespaceReflectExpressionAST(
          in, static_cast<cxx::NamespaceReflectExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TypeIdReflectExpression:
      readAstTypeIdReflectExpressionAST(
          in, static_cast<cxx::TypeIdReflectExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ReflectExpression:
      readAstReflectExpressionAST(in,
                                  static_cast<cxx::ReflectExpressionAST*>(ast));
      break;
    case cxx::ASTKind::LabelAddressExpression:
      readAstLabelAddressExpressionAST(
          in, static_cast<cxx::LabelAddressExpressionAST*>(ast));
      break;
    case cxx::ASTKind::UnaryExpression:
      readAstUnaryExpressionAST(in, static_cast<cxx::UnaryExpressionAST*>(ast));
      break;
    case cxx::ASTKind::AwaitExpression:
      readAstAwaitExpressionAST(in, static_cast<cxx::AwaitExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SizeofExpression:
      readAstSizeofExpressionAST(in,
                                 static_cast<cxx::SizeofExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SizeofTypeExpression:
      readAstSizeofTypeExpressionAST(
          in, static_cast<cxx::SizeofTypeExpressionAST*>(ast));
      break;
    case cxx::ASTKind::SizeofPackExpression:
      readAstSizeofPackExpressionAST(
          in, static_cast<cxx::SizeofPackExpressionAST*>(ast));
      break;
    case cxx::ASTKind::AlignofTypeExpression:
      readAstAlignofTypeExpressionAST(
          in, static_cast<cxx::AlignofTypeExpressionAST*>(ast));
      break;
    case cxx::ASTKind::AlignofExpression:
      readAstAlignofExpressionAST(in,
                                  static_cast<cxx::AlignofExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NoexceptExpression:
      readAstNoexceptExpressionAST(
          in, static_cast<cxx::NoexceptExpressionAST*>(ast));
      break;
    case cxx::ASTKind::NewExpression:
      readAstNewExpressionAST(in, static_cast<cxx::NewExpressionAST*>(ast));
      break;
    case cxx::ASTKind::DeleteExpression:
      readAstDeleteExpressionAST(in,
                                 static_cast<cxx::DeleteExpressionAST*>(ast));
      break;
    case cxx::ASTKind::CastExpression:
      readAstCastExpressionAST(in, static_cast<cxx::CastExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ImplicitCastExpression:
      readAstImplicitCastExpressionAST(
          in, static_cast<cxx::ImplicitCastExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ConstExpression:
      readAstConstExpressionAST(in, static_cast<cxx::ConstExpressionAST*>(ast));
      break;
    case cxx::ASTKind::BinaryExpression:
      readAstBinaryExpressionAST(in,
                                 static_cast<cxx::BinaryExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ConditionalExpression:
      readAstConditionalExpressionAST(
          in, static_cast<cxx::ConditionalExpressionAST*>(ast));
      break;
    case cxx::ASTKind::YieldExpression:
      readAstYieldExpressionAST(in, static_cast<cxx::YieldExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ThrowExpression:
      readAstThrowExpressionAST(in, static_cast<cxx::ThrowExpressionAST*>(ast));
      break;
    case cxx::ASTKind::AssignmentExpression:
      readAstAssignmentExpressionAST(
          in, static_cast<cxx::AssignmentExpressionAST*>(ast));
      break;
    case cxx::ASTKind::TargetExpression:
      readAstTargetExpressionAST(in,
                                 static_cast<cxx::TargetExpressionAST*>(ast));
      break;
    case cxx::ASTKind::RightExpression:
      readAstRightExpressionAST(in, static_cast<cxx::RightExpressionAST*>(ast));
      break;
    case cxx::ASTKind::CompoundAssignmentExpression:
      readAstCompoundAssignmentExpressionAST(
          in, static_cast<cxx::CompoundAssignmentExpressionAST*>(ast));
      break;
    case cxx::ASTKind::PackExpansionExpression:
      readAstPackExpansionExpressionAST(
          in, static_cast<cxx::PackExpansionExpressionAST*>(ast));
      break;
    case cxx::ASTKind::DesignatedInitializerClause:
      readAstDesignatedInitializerClauseAST(
          in, static_cast<cxx::DesignatedInitializerClauseAST*>(ast));
      break;
    case cxx::ASTKind::TypeTraitExpression:
      readAstTypeTraitExpressionAST(
          in, static_cast<cxx::TypeTraitExpressionAST*>(ast));
      break;
    case cxx::ASTKind::ConditionExpression:
      readAstConditionExpressionAST(
          in, static_cast<cxx::ConditionExpressionAST*>(ast));
      break;
    case cxx::ASTKind::EqualInitializer:
      readAstEqualInitializerAST(in,
                                 static_cast<cxx::EqualInitializerAST*>(ast));
      break;
    case cxx::ASTKind::BracedInitList:
      readAstBracedInitListAST(in, static_cast<cxx::BracedInitListAST*>(ast));
      break;
    case cxx::ASTKind::ParenInitializer:
      readAstParenInitializerAST(in,
                                 static_cast<cxx::ParenInitializerAST*>(ast));
      break;
    case cxx::ASTKind::ThreeWayComparisonExpression:
      readAstThreeWayComparisonExpressionAST(
          in, static_cast<cxx::ThreeWayComparisonExpressionAST*>(ast));
      break;
    case cxx::ASTKind::DefaultGenericAssociation:
      readAstDefaultGenericAssociationAST(
          in, static_cast<cxx::DefaultGenericAssociationAST*>(ast));
      break;
    case cxx::ASTKind::TypeGenericAssociation:
      readAstTypeGenericAssociationAST(
          in, static_cast<cxx::TypeGenericAssociationAST*>(ast));
      break;
    case cxx::ASTKind::DotDesignator:
      readAstDotDesignatorAST(in, static_cast<cxx::DotDesignatorAST*>(ast));
      break;
    case cxx::ASTKind::SubscriptDesignator:
      readAstSubscriptDesignatorAST(
          in, static_cast<cxx::SubscriptDesignatorAST*>(ast));
      break;
    case cxx::ASTKind::TemplateTypeParameter:
      readAstTemplateTypeParameterAST(
          in, static_cast<cxx::TemplateTypeParameterAST*>(ast));
      break;
    case cxx::ASTKind::NonTypeTemplateParameter:
      readAstNonTypeTemplateParameterAST(
          in, static_cast<cxx::NonTypeTemplateParameterAST*>(ast));
      break;
    case cxx::ASTKind::TypenameTypeParameter:
      readAstTypenameTypeParameterAST(
          in, static_cast<cxx::TypenameTypeParameterAST*>(ast));
      break;
    case cxx::ASTKind::ConstraintTypeParameter:
      readAstConstraintTypeParameterAST(
          in, static_cast<cxx::ConstraintTypeParameterAST*>(ast));
      break;
    case cxx::ASTKind::TypedefSpecifier:
      readAstTypedefSpecifierAST(in,
                                 static_cast<cxx::TypedefSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::FriendSpecifier:
      readAstFriendSpecifierAST(in, static_cast<cxx::FriendSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ConstevalSpecifier:
      readAstConstevalSpecifierAST(
          in, static_cast<cxx::ConstevalSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ConstinitSpecifier:
      readAstConstinitSpecifierAST(
          in, static_cast<cxx::ConstinitSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ConstexprSpecifier:
      readAstConstexprSpecifierAST(
          in, static_cast<cxx::ConstexprSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::InlineSpecifier:
      readAstInlineSpecifierAST(in, static_cast<cxx::InlineSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::NoreturnSpecifier:
      readAstNoreturnSpecifierAST(in,
                                  static_cast<cxx::NoreturnSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::StaticSpecifier:
      readAstStaticSpecifierAST(in, static_cast<cxx::StaticSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ExternSpecifier:
      readAstExternSpecifierAST(in, static_cast<cxx::ExternSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::RegisterSpecifier:
      readAstRegisterSpecifierAST(in,
                                  static_cast<cxx::RegisterSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ThreadLocalSpecifier:
      readAstThreadLocalSpecifierAST(
          in, static_cast<cxx::ThreadLocalSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ThreadSpecifier:
      readAstThreadSpecifierAST(in, static_cast<cxx::ThreadSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::MutableSpecifier:
      readAstMutableSpecifierAST(in,
                                 static_cast<cxx::MutableSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::VirtualSpecifier:
      readAstVirtualSpecifierAST(in,
                                 static_cast<cxx::VirtualSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ExplicitSpecifier:
      readAstExplicitSpecifierAST(in,
                                  static_cast<cxx::ExplicitSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::AutoTypeSpecifier:
      readAstAutoTypeSpecifierAST(in,
                                  static_cast<cxx::AutoTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::VoidTypeSpecifier:
      readAstVoidTypeSpecifierAST(in,
                                  static_cast<cxx::VoidTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SizeTypeSpecifier:
      readAstSizeTypeSpecifierAST(in,
                                  static_cast<cxx::SizeTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SignTypeSpecifier:
      readAstSignTypeSpecifierAST(in,
                                  static_cast<cxx::SignTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::BuiltinTypeSpecifier:
      readAstBuiltinTypeSpecifierAST(
          in, static_cast<cxx::BuiltinTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::UnaryBuiltinTypeSpecifier:
      readAstUnaryBuiltinTypeSpecifierAST(
          in, static_cast<cxx::UnaryBuiltinTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::BinaryBuiltinTypeSpecifier:
      readAstBinaryBuiltinTypeSpecifierAST(
          in, static_cast<cxx::BinaryBuiltinTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::IntegralTypeSpecifier:
      readAstIntegralTypeSpecifierAST(
          in, static_cast<cxx::IntegralTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::FloatingPointTypeSpecifier:
      readAstFloatingPointTypeSpecifierAST(
          in, static_cast<cxx::FloatingPointTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ComplexTypeSpecifier:
      readAstComplexTypeSpecifierAST(
          in, static_cast<cxx::ComplexTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::NamedTypeSpecifier:
      readAstNamedTypeSpecifierAST(
          in, static_cast<cxx::NamedTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::AtomicTypeSpecifier:
      readAstAtomicTypeSpecifierAST(
          in, static_cast<cxx::AtomicTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::BitIntTypeSpecifier:
      readAstBitIntTypeSpecifierAST(
          in, static_cast<cxx::BitIntTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::UnderlyingTypeSpecifier:
      readAstUnderlyingTypeSpecifierAST(
          in, static_cast<cxx::UnderlyingTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ElaboratedTypeSpecifier:
      readAstElaboratedTypeSpecifierAST(
          in, static_cast<cxx::ElaboratedTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::DecltypeAutoSpecifier:
      readAstDecltypeAutoSpecifierAST(
          in, static_cast<cxx::DecltypeAutoSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::DecltypeSpecifier:
      readAstDecltypeSpecifierAST(in,
                                  static_cast<cxx::DecltypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::PlaceholderTypeSpecifier:
      readAstPlaceholderTypeSpecifierAST(
          in, static_cast<cxx::PlaceholderTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ConstQualifier:
      readAstConstQualifierAST(in, static_cast<cxx::ConstQualifierAST*>(ast));
      break;
    case cxx::ASTKind::VolatileQualifier:
      readAstVolatileQualifierAST(in,
                                  static_cast<cxx::VolatileQualifierAST*>(ast));
      break;
    case cxx::ASTKind::AtomicQualifier:
      readAstAtomicQualifierAST(in, static_cast<cxx::AtomicQualifierAST*>(ast));
      break;
    case cxx::ASTKind::RestrictQualifier:
      readAstRestrictQualifierAST(in,
                                  static_cast<cxx::RestrictQualifierAST*>(ast));
      break;
    case cxx::ASTKind::EnumSpecifier:
      readAstEnumSpecifierAST(in, static_cast<cxx::EnumSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::ClassSpecifier:
      readAstClassSpecifierAST(in, static_cast<cxx::ClassSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::TypenameSpecifier:
      readAstTypenameSpecifierAST(in,
                                  static_cast<cxx::TypenameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SplicerTypeSpecifier:
      readAstSplicerTypeSpecifierAST(
          in, static_cast<cxx::SplicerTypeSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::PointerOperator:
      readAstPointerOperatorAST(in, static_cast<cxx::PointerOperatorAST*>(ast));
      break;
    case cxx::ASTKind::ReferenceOperator:
      readAstReferenceOperatorAST(in,
                                  static_cast<cxx::ReferenceOperatorAST*>(ast));
      break;
    case cxx::ASTKind::PtrToMemberOperator:
      readAstPtrToMemberOperatorAST(
          in, static_cast<cxx::PtrToMemberOperatorAST*>(ast));
      break;
    case cxx::ASTKind::BitfieldDeclarator:
      readAstBitfieldDeclaratorAST(
          in, static_cast<cxx::BitfieldDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::ParameterPack:
      readAstParameterPackAST(in, static_cast<cxx::ParameterPackAST*>(ast));
      break;
    case cxx::ASTKind::IdDeclarator:
      readAstIdDeclaratorAST(in, static_cast<cxx::IdDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::NestedDeclarator:
      readAstNestedDeclaratorAST(in,
                                 static_cast<cxx::NestedDeclaratorAST*>(ast));
      break;
    case cxx::ASTKind::FunctionDeclaratorChunk:
      readAstFunctionDeclaratorChunkAST(
          in, static_cast<cxx::FunctionDeclaratorChunkAST*>(ast));
      break;
    case cxx::ASTKind::ArrayDeclaratorChunk:
      readAstArrayDeclaratorChunkAST(
          in, static_cast<cxx::ArrayDeclaratorChunkAST*>(ast));
      break;
    case cxx::ASTKind::NameId:
      readAstNameIdAST(in, static_cast<cxx::NameIdAST*>(ast));
      break;
    case cxx::ASTKind::DestructorId:
      readAstDestructorIdAST(in, static_cast<cxx::DestructorIdAST*>(ast));
      break;
    case cxx::ASTKind::DecltypeId:
      readAstDecltypeIdAST(in, static_cast<cxx::DecltypeIdAST*>(ast));
      break;
    case cxx::ASTKind::OperatorFunctionId:
      readAstOperatorFunctionIdAST(
          in, static_cast<cxx::OperatorFunctionIdAST*>(ast));
      break;
    case cxx::ASTKind::LiteralOperatorId:
      readAstLiteralOperatorIdAST(in,
                                  static_cast<cxx::LiteralOperatorIdAST*>(ast));
      break;
    case cxx::ASTKind::ConversionFunctionId:
      readAstConversionFunctionIdAST(
          in, static_cast<cxx::ConversionFunctionIdAST*>(ast));
      break;
    case cxx::ASTKind::SimpleTemplateId:
      readAstSimpleTemplateIdAST(in,
                                 static_cast<cxx::SimpleTemplateIdAST*>(ast));
      break;
    case cxx::ASTKind::LiteralOperatorTemplateId:
      readAstLiteralOperatorTemplateIdAST(
          in, static_cast<cxx::LiteralOperatorTemplateIdAST*>(ast));
      break;
    case cxx::ASTKind::OperatorFunctionTemplateId:
      readAstOperatorFunctionTemplateIdAST(
          in, static_cast<cxx::OperatorFunctionTemplateIdAST*>(ast));
      break;
    case cxx::ASTKind::GlobalNestedNameSpecifier:
      readAstGlobalNestedNameSpecifierAST(
          in, static_cast<cxx::GlobalNestedNameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SimpleNestedNameSpecifier:
      readAstSimpleNestedNameSpecifierAST(
          in, static_cast<cxx::SimpleNestedNameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::DecltypeNestedNameSpecifier:
      readAstDecltypeNestedNameSpecifierAST(
          in, static_cast<cxx::DecltypeNestedNameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::TemplateNestedNameSpecifier:
      readAstTemplateNestedNameSpecifierAST(
          in, static_cast<cxx::TemplateNestedNameSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::DefaultFunctionBody:
      readAstDefaultFunctionBodyAST(
          in, static_cast<cxx::DefaultFunctionBodyAST*>(ast));
      break;
    case cxx::ASTKind::CompoundStatementFunctionBody:
      readAstCompoundStatementFunctionBodyAST(
          in, static_cast<cxx::CompoundStatementFunctionBodyAST*>(ast));
      break;
    case cxx::ASTKind::TryStatementFunctionBody:
      readAstTryStatementFunctionBodyAST(
          in, static_cast<cxx::TryStatementFunctionBodyAST*>(ast));
      break;
    case cxx::ASTKind::DeleteFunctionBody:
      readAstDeleteFunctionBodyAST(
          in, static_cast<cxx::DeleteFunctionBodyAST*>(ast));
      break;
    case cxx::ASTKind::TypeTemplateArgument:
      readAstTypeTemplateArgumentAST(
          in, static_cast<cxx::TypeTemplateArgumentAST*>(ast));
      break;
    case cxx::ASTKind::ExpressionTemplateArgument:
      readAstExpressionTemplateArgumentAST(
          in, static_cast<cxx::ExpressionTemplateArgumentAST*>(ast));
      break;
    case cxx::ASTKind::ThrowExceptionSpecifier:
      readAstThrowExceptionSpecifierAST(
          in, static_cast<cxx::ThrowExceptionSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::NoexceptSpecifier:
      readAstNoexceptSpecifierAST(in,
                                  static_cast<cxx::NoexceptSpecifierAST*>(ast));
      break;
    case cxx::ASTKind::SimpleRequirement:
      readAstSimpleRequirementAST(in,
                                  static_cast<cxx::SimpleRequirementAST*>(ast));
      break;
    case cxx::ASTKind::CompoundRequirement:
      readAstCompoundRequirementAST(
          in, static_cast<cxx::CompoundRequirementAST*>(ast));
      break;
    case cxx::ASTKind::TypeRequirement:
      readAstTypeRequirementAST(in, static_cast<cxx::TypeRequirementAST*>(ast));
      break;
    case cxx::ASTKind::NestedRequirement:
      readAstNestedRequirementAST(in,
                                  static_cast<cxx::NestedRequirementAST*>(ast));
      break;
    case cxx::ASTKind::NewParenInitializer:
      readAstNewParenInitializerAST(
          in, static_cast<cxx::NewParenInitializerAST*>(ast));
      break;
    case cxx::ASTKind::NewBracedInitializer:
      readAstNewBracedInitializerAST(
          in, static_cast<cxx::NewBracedInitializerAST*>(ast));
      break;
    case cxx::ASTKind::ParenMemInitializer:
      readAstParenMemInitializerAST(
          in, static_cast<cxx::ParenMemInitializerAST*>(ast));
      break;
    case cxx::ASTKind::BracedMemInitializer:
      readAstBracedMemInitializerAST(
          in, static_cast<cxx::BracedMemInitializerAST*>(ast));
      break;
    case cxx::ASTKind::ThisLambdaCapture:
      readAstThisLambdaCaptureAST(in,
                                  static_cast<cxx::ThisLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::DerefThisLambdaCapture:
      readAstDerefThisLambdaCaptureAST(
          in, static_cast<cxx::DerefThisLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::SimpleLambdaCapture:
      readAstSimpleLambdaCaptureAST(
          in, static_cast<cxx::SimpleLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::RefLambdaCapture:
      readAstRefLambdaCaptureAST(in,
                                 static_cast<cxx::RefLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::RefInitLambdaCapture:
      readAstRefInitLambdaCaptureAST(
          in, static_cast<cxx::RefInitLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::InitLambdaCapture:
      readAstInitLambdaCaptureAST(in,
                                  static_cast<cxx::InitLambdaCaptureAST*>(ast));
      break;
    case cxx::ASTKind::EllipsisExceptionDeclaration:
      readAstEllipsisExceptionDeclarationAST(
          in, static_cast<cxx::EllipsisExceptionDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::TypeExceptionDeclaration:
      readAstTypeExceptionDeclarationAST(
          in, static_cast<cxx::TypeExceptionDeclarationAST*>(ast));
      break;
    case cxx::ASTKind::CxxAttribute:
      readAstCxxAttributeAST(in, static_cast<cxx::CxxAttributeAST*>(ast));
      break;
    case cxx::ASTKind::GccAttribute:
      readAstGccAttributeAST(in, static_cast<cxx::GccAttributeAST*>(ast));
      break;
    case cxx::ASTKind::AlignasAttribute:
      readAstAlignasAttributeAST(in,
                                 static_cast<cxx::AlignasAttributeAST*>(ast));
      break;
    case cxx::ASTKind::AlignasTypeAttribute:
      readAstAlignasTypeAttributeAST(
          in, static_cast<cxx::AlignasTypeAttributeAST*>(ast));
      break;
    case cxx::ASTKind::AsmAttribute:
      readAstAsmAttributeAST(in, static_cast<cxx::AsmAttributeAST*>(ast));
      break;
    case cxx::ASTKind::ScopedAttributeToken:
      readAstScopedAttributeTokenAST(
          in, static_cast<cxx::ScopedAttributeTokenAST*>(ast));
      break;
    case cxx::ASTKind::SimpleAttributeToken:
      readAstSimpleAttributeTokenAST(
          in, static_cast<cxx::SimpleAttributeTokenAST*>(ast));
      break;
  }
}
auto SemanticDecoder::readEnum(ByteReader& in, std::uint32_t count)
    -> std::uint32_t {
  const auto value = in.varU32();
  if (value >= count) {
    fail("enumerator is out of range");
    return 0;
  }
  return value;
}

auto SemanticDecoder::readAbiTags(ByteReader& in)
    -> const std::vector<const cxx::Identifier*>* {
  if (!in.boolean()) return nullptr;
  const auto count = in.varCount(1);
  std::vector<const cxx::Identifier*> tags;
  tags.reserve(count);
  for (std::uint32_t i = 0; ok() && i < count; ++i)
    tags.push_back(identifierAt(StringRef{in.varU32()}));
  return control()->getAbiTags(std::move(tags));
}

auto SemanticDecoder::readAttributes(ByteReader& in)
    -> const cxx::AttributeMap* {
  if (!in.boolean()) return nullptr;
  const auto count = in.varCount(1);
  cxx::AttributeMap attributes;
  attributes.reserve(count);
  for (std::uint32_t i = 0; ok() && i < count; ++i) {
    cxx::Attribute attribute;
    readcxxAttribute(in, &attribute);
    attributes.push_back(std::move(attribute));
  }
  return control()->getAttributes(std::move(attributes));
}

auto SemanticDecoder::readConstValue(ByteReader& in) -> cxx::ConstValue {
  const auto tag = in.u8();
  switch (tag) {
    case 0:
      return cxx::ConstValue{static_cast<std::intmax_t>(in.i64())};
    case 1:
      return cxx::ConstValue{readStringLiteral(in)};
    case 2:
      return cxx::ConstValue{static_cast<float>(in.f32())};
    case 3:
      return cxx::ConstValue{static_cast<double>(in.f64())};
    case 4:
      return cxx::ConstValue{static_cast<long double>(in.f80())};
    case 5:
      return cxx::ConstValue{std::static_pointer_cast<cxx::Meta>(
          constantAt(ConstRef{in.varU32()}))};
    case 6:
      return cxx::ConstValue{std::static_pointer_cast<cxx::InitializerList>(
          constantAt(ConstRef{in.varU32()}))};
    case 7:
      return cxx::ConstValue{std::static_pointer_cast<cxx::ConstObject>(
          constantAt(ConstRef{in.varU32()}))};
    case 8:
      return cxx::ConstValue{std::static_pointer_cast<cxx::ConstAddress>(
          constantAt(ConstRef{in.varU32()}))};
    case 9:
      return cxx::ConstValue{std::static_pointer_cast<cxx::ConstLabelAddress>(
          constantAt(ConstRef{in.varU32()}))};
    case 10:
      return cxx::ConstValue{cxx::IndeterminateValue{}};
    default:
      fail("unknown constant value alternative");
      return cxx::ConstValue{};
  }
}

auto SemanticDecoder::readTemplateArgument(ByteReader& in)
    -> cxx::TemplateArgument {
  const auto tag = in.u8();
  switch (tag) {
    case 0:
      return cxx::TemplateArgument{typeAt(TypeRef{in.varU32()})};
    case 1:
      return cxx::TemplateArgument{symbolAt(SymbolRef{in.varU32()})};
    case 2:
      return cxx::TemplateArgument{readConstValue(in)};
    case 3:
      return cxx::TemplateArgument{
          ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}))};
    default:
      fail("unknown template argument alternative");
      return cxx::TemplateArgument{};
  }
}

auto SemanticDecoder::constantAt(ConstRef ref) -> std::shared_ptr<void> {
  const auto index = static_cast<std::uint32_t>(ref);
  if (index == 0) return {};
  if (index > constRecords_.size()) {
    fail("constant reference is out of range");
    return {};
  }
  if (constants_[index - 1]) return constants_[index - 1];
  ByteReader in{constRecords_[index - 1].bytes};
  const auto kind = in.u8();
  switch (kind) {
    case 0: {
      auto value = std::make_shared<cxx::Meta>();
      constants_[index - 1] = value;
      readcxxMeta(in, value.get());
      return value;
    }
    case 1: {
      auto value = std::make_shared<cxx::InitializerList>();
      constants_[index - 1] = value;
      readcxxInitializerList(in, value.get());
      return value;
    }
    case 2: {
      auto value = std::make_shared<cxx::ConstObject>();
      constants_[index - 1] = value;
      readcxxConstObject(in, value.get());
      return value;
    }
    case 3: {
      auto value = std::make_shared<cxx::ConstAddress>();
      constants_[index - 1] = value;
      readcxxConstAddress(in, value.get());
      return value;
    }
    case 4: {
      auto value = std::make_shared<cxx::ConstLabelAddress>();
      constants_[index - 1] = value;
      readcxxConstLabelAddress(in, value.get());
      return value;
    }
    default:
      fail("unknown constant node kind");
      return {};
  }
}

auto SemanticDecoder::readNameIdentifier(ByteReader& in) -> const cxx::Name* {
  std::string argument1{stringAt(StringRef{in.varU32()})};
  return control()->getIdentifier(std::move(argument1));
}

auto SemanticDecoder::readNameOperatorId(ByteReader& in) -> const cxx::Name* {
  ::cxx::TokenKind argument1 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  return control()->getOperatorId(std::move(argument1));
}

auto SemanticDecoder::readNameDestructorId(ByteReader& in) -> const cxx::Name* {
  const cxx::Name* argument1 = nameAt(NameRef{in.varU32()});
  return control()->getDestructorId(std::move(argument1));
}

auto SemanticDecoder::readNameLiteralOperatorId(ByteReader& in)
    -> const cxx::Name* {
  std::string argument1{stringAt(StringRef{in.varU32()})};
  return control()->getLiteralOperatorId(std::move(argument1));
}

auto SemanticDecoder::readNameConversionFunctionId(ByteReader& in)
    -> const cxx::Name* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  return control()->getConversionFunctionId(std::move(argument1));
}

auto SemanticDecoder::readNameTemplateId(ByteReader& in) -> const cxx::Name* {
  const cxx::Name* argument1 = nameAt(NameRef{in.varU32()});
  std::vector<cxx::TemplateArgument> argument2;
  {
    const auto count3 = in.varCount(1);
    for (std::uint32_t i4 = 0; ok() && i4 < count3; ++i4) {
      cxx::TemplateArgument element5 = readTemplateArgument(in);
      argument2.push_back(std::move(element5));
    }
  }
  return control()->getTemplateId(std::move(argument1), std::move(argument2));
}

auto SemanticDecoder::readTypeVoidType(ByteReader& in) -> const cxx::Type* {
  return control()->getVoidType();
}

auto SemanticDecoder::readTypeNullptrType(ByteReader& in) -> const cxx::Type* {
  return control()->getNullptrType();
}

auto SemanticDecoder::readTypeDecltypeAutoType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getDecltypeAutoType();
}

auto SemanticDecoder::readTypeAutoType(ByteReader& in) -> const cxx::Type* {
  return control()->getAutoType();
}

auto SemanticDecoder::readTypeBoolType(ByteReader& in) -> const cxx::Type* {
  return control()->getBoolType();
}

auto SemanticDecoder::readTypeSignedCharType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getSignedCharType();
}

auto SemanticDecoder::readTypeShortIntType(ByteReader& in) -> const cxx::Type* {
  return control()->getShortIntType();
}

auto SemanticDecoder::readTypeIntType(ByteReader& in) -> const cxx::Type* {
  return control()->getIntType();
}

auto SemanticDecoder::readTypeLongIntType(ByteReader& in) -> const cxx::Type* {
  return control()->getLongIntType();
}

auto SemanticDecoder::readTypeLongLongIntType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getLongLongIntType();
}

auto SemanticDecoder::readTypeInt128Type(ByteReader& in) -> const cxx::Type* {
  return control()->getInt128Type();
}

auto SemanticDecoder::readTypeUnsignedCharType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getUnsignedCharType();
}

auto SemanticDecoder::readTypeUnsignedShortIntType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getUnsignedShortIntType();
}

auto SemanticDecoder::readTypeUnsignedIntType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getUnsignedIntType();
}

auto SemanticDecoder::readTypeUnsignedLongIntType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getUnsignedLongIntType();
}

auto SemanticDecoder::readTypeUnsignedLongLongIntType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getUnsignedLongLongIntType();
}

auto SemanticDecoder::readTypeUnsignedInt128Type(ByteReader& in)
    -> const cxx::Type* {
  return control()->getUnsignedInt128Type();
}

auto SemanticDecoder::readTypeCharType(ByteReader& in) -> const cxx::Type* {
  return control()->getCharType();
}

auto SemanticDecoder::readTypeChar8Type(ByteReader& in) -> const cxx::Type* {
  return control()->getChar8Type();
}

auto SemanticDecoder::readTypeChar16Type(ByteReader& in) -> const cxx::Type* {
  return control()->getChar16Type();
}

auto SemanticDecoder::readTypeChar32Type(ByteReader& in) -> const cxx::Type* {
  return control()->getChar32Type();
}

auto SemanticDecoder::readTypeWideCharType(ByteReader& in) -> const cxx::Type* {
  return control()->getWideCharType();
}

auto SemanticDecoder::readTypeFloatType(ByteReader& in) -> const cxx::Type* {
  return control()->getFloatType();
}

auto SemanticDecoder::readTypeDoubleType(ByteReader& in) -> const cxx::Type* {
  return control()->getDoubleType();
}

auto SemanticDecoder::readTypeLongDoubleType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getLongDoubleType();
}

auto SemanticDecoder::readTypeFloat16Type(ByteReader& in) -> const cxx::Type* {
  return control()->getFloat16Type();
}

auto SemanticDecoder::readTypeQualType(ByteReader& in) -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  static_assert(
      static_cast<std::uint32_t>(::cxx::CvQualifiers::kConstVolatile) + 1 == 4);
  ::cxx::CvQualifiers argument2 =
      static_cast<::cxx::CvQualifiers>(readEnum(in, 4));
  return control()->getQualType(std::move(argument1), std::move(argument2));
}

auto SemanticDecoder::readTypeBoundedArrayType(ByteReader& in)
    -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  unsigned long argument2 = static_cast<unsigned long>(in.varU64());
  return control()->getBoundedArrayType(std::move(argument1),
                                        std::move(argument2));
}

auto SemanticDecoder::readTypeUnboundedArrayType(ByteReader& in)
    -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  return control()->getUnboundedArrayType(std::move(argument1));
}

auto SemanticDecoder::readTypePointerType(ByteReader& in) -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  return control()->getPointerType(std::move(argument1));
}

auto SemanticDecoder::readTypeLvalueReferenceType(ByteReader& in)
    -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  return control()->getLvalueReferenceType(std::move(argument1));
}

auto SemanticDecoder::readTypeRvalueReferenceType(ByteReader& in)
    -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  return control()->getRvalueReferenceType(std::move(argument1));
}

auto SemanticDecoder::readTypeFunctionType(ByteReader& in) -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  std::vector<const cxx::Type*> argument2;
  {
    const auto count3 = in.varCount(1);
    for (std::uint32_t i4 = 0; ok() && i4 < count3; ++i4) {
      const cxx::Type* element5 = typeAt(TypeRef{in.varU32()});
      argument2.push_back(std::move(element5));
    }
  }
  bool argument6 = in.boolean();
  static_assert(
      static_cast<std::uint32_t>(::cxx::CvQualifiers::kConstVolatile) + 1 == 4);
  ::cxx::CvQualifiers argument7 =
      static_cast<::cxx::CvQualifiers>(readEnum(in, 4));
  static_assert(static_cast<std::uint32_t>(::cxx::RefQualifier::kRvalue) + 1 ==
                3);
  ::cxx::RefQualifier argument8 =
      static_cast<::cxx::RefQualifier>(readEnum(in, 3));
  bool argument9 = in.boolean();
  return control()->getFunctionType(std::move(argument1), std::move(argument2),
                                    std::move(argument6), std::move(argument7),
                                    std::move(argument8), std::move(argument9));
}

auto SemanticDecoder::readTypeClassType(ByteReader& in) -> const cxx::Type* {
  cxx::ClassSymbol* argument1 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  return control()->getClassType(std::move(argument1));
}

auto SemanticDecoder::readTypeEnumType(ByteReader& in) -> const cxx::Type* {
  cxx::EnumSymbol* argument1 =
      symbol_cast<EnumSymbol>(symbolAt(SymbolRef{in.varU32()}));
  return control()->getEnumType(std::move(argument1));
}

auto SemanticDecoder::readTypeScopedEnumType(ByteReader& in)
    -> const cxx::Type* {
  cxx::ScopedEnumSymbol* argument1 =
      symbol_cast<ScopedEnumSymbol>(symbolAt(SymbolRef{in.varU32()}));
  return control()->getScopedEnumType(std::move(argument1));
}

auto SemanticDecoder::readTypeMemberObjectPointerType(ByteReader& in)
    -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  const cxx::Type* argument2 = typeAt(TypeRef{in.varU32()});
  return control()->getMemberObjectPointerType(std::move(argument1),
                                               std::move(argument2));
}

auto SemanticDecoder::readTypeMemberFunctionPointerType(ByteReader& in)
    -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  const cxx::FunctionType* argument2 =
      type_cast<FunctionType>(typeAt(TypeRef{in.varU32()}));
  return control()->getMemberFunctionPointerType(std::move(argument1),
                                                 std::move(argument2));
}

auto SemanticDecoder::readTypeNamespaceType(ByteReader& in)
    -> const cxx::Type* {
  cxx::NamespaceSymbol* argument1 =
      symbol_cast<NamespaceSymbol>(symbolAt(SymbolRef{in.varU32()}));
  return control()->getNamespaceType(std::move(argument1));
}

auto SemanticDecoder::readTypeTypeParameterType(ByteReader& in)
    -> const cxx::Type* {
  int argument1 = static_cast<int>(in.varI32());
  int argument2 = static_cast<int>(in.varI32());
  bool argument3 = in.boolean();
  return control()->getTypeParameterType(
      std::move(argument1), std::move(argument2), std::move(argument3));
}

auto SemanticDecoder::readTypeTemplateTypeParameterType(ByteReader& in)
    -> const cxx::Type* {
  int argument1 = static_cast<int>(in.varI32());
  int argument2 = static_cast<int>(in.varI32());
  bool argument3 = in.boolean();
  std::vector<const cxx::Type*> argument4;
  {
    const auto count5 = in.varCount(1);
    for (std::uint32_t i6 = 0; ok() && i6 < count5; ++i6) {
      const cxx::Type* element7 = typeAt(TypeRef{in.varU32()});
      argument4.push_back(std::move(element7));
    }
  }
  return control()->getTemplateTypeParameterType(
      std::move(argument1), std::move(argument2), std::move(argument3),
      std::move(argument4));
}

auto SemanticDecoder::readTypeUnresolvedNameType(ByteReader& in)
    -> const cxx::Type* {
  cxx::TranslationUnit* argument1 = unit();
  cxx::NestedNameSpecifierAST* argument2 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  cxx::UnqualifiedIdAST* argument3 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  return control()->getUnresolvedNameType(
      std::move(argument1), std::move(argument2), std::move(argument3));
}

auto SemanticDecoder::readTypeUnresolvedBoundedArrayType(ByteReader& in)
    -> const cxx::Type* {
  cxx::TranslationUnit* argument1 = unit();
  const cxx::Type* argument2 = typeAt(TypeRef{in.varU32()});
  cxx::ExpressionAST* argument3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  return control()->getUnresolvedBoundedArrayType(
      std::move(argument1), std::move(argument2), std::move(argument3));
}

auto SemanticDecoder::readTypeUnresolvedUnderlyingType(ByteReader& in)
    -> const cxx::Type* {
  cxx::TranslationUnit* argument1 = unit();
  cxx::TypeIdAST* argument2 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  return control()->getUnresolvedUnderlyingType(std::move(argument1),
                                                std::move(argument2));
}

auto SemanticDecoder::readTypeUnresolvedBuiltinType(ByteReader& in)
    -> const cxx::Type* {
  cxx::TranslationUnit* argument1 = unit();
  static_assert(static_cast<std::uint32_t>(
                    ::cxx::UnaryBuiltinTypeKind::T___REMOVE_VOLATILE) +
                    1 ==
                16);
  ::cxx::UnaryBuiltinTypeKind argument2 =
      static_cast<::cxx::UnaryBuiltinTypeKind>(readEnum(in, 16));
  cxx::TypeIdAST* argument3 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  return control()->getUnresolvedBuiltinType(
      std::move(argument1), std::move(argument2), std::move(argument3));
}

auto SemanticDecoder::readTypeOverloadSetType(ByteReader& in)
    -> const cxx::Type* {
  cxx::OverloadSetSymbol* argument1 =
      symbol_cast<OverloadSetSymbol>(symbolAt(SymbolRef{in.varU32()}));
  return control()->getOverloadSetType(std::move(argument1));
}

auto SemanticDecoder::readTypeBuiltinVaListType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getBuiltinVaListType();
}

auto SemanticDecoder::readTypeBuiltinMetaInfoType(ByteReader& in)
    -> const cxx::Type* {
  return control()->getBuiltinMetaInfoType();
}

auto SemanticDecoder::readTypeBitIntType(ByteReader& in) -> const cxx::Type* {
  int argument1 = static_cast<int>(in.varI32());
  return control()->getBitIntType(std::move(argument1));
}

auto SemanticDecoder::readTypeUnsignedBitIntType(ByteReader& in)
    -> const cxx::Type* {
  int argument1 = static_cast<int>(in.varI32());
  return control()->getUnsignedBitIntType(std::move(argument1));
}

auto SemanticDecoder::readTypeUnresolvedBitIntType(ByteReader& in)
    -> const cxx::Type* {
  cxx::TranslationUnit* argument1 = unit();
  cxx::ExpressionAST* argument2 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  bool argument3 = in.boolean();
  return control()->getUnresolvedBitIntType(
      std::move(argument1), std::move(argument2), std::move(argument3));
}

auto SemanticDecoder::readTypeVectorType(ByteReader& in) -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  unsigned long argument2 = static_cast<unsigned long>(in.varU64());
  static_assert(static_cast<std::uint32_t>(::cxx::VectorKind::kExt) + 1 == 2);
  ::cxx::VectorKind argument3 = static_cast<::cxx::VectorKind>(readEnum(in, 2));
  return control()->getVectorType(std::move(argument1), std::move(argument2),
                                  std::move(argument3));
}

auto SemanticDecoder::readTypeUnresolvedVectorType(ByteReader& in)
    -> const cxx::Type* {
  cxx::TranslationUnit* argument1 = unit();
  const cxx::Type* argument2 = typeAt(TypeRef{in.varU32()});
  cxx::ExpressionAST* argument3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  static_assert(static_cast<std::uint32_t>(::cxx::VectorKind::kExt) + 1 == 2);
  ::cxx::VectorKind argument4 = static_cast<::cxx::VectorKind>(readEnum(in, 2));
  static_assert(
      static_cast<std::uint32_t>(::cxx::VectorSizeKind::kElements) + 1 == 2);
  ::cxx::VectorSizeKind argument5 =
      static_cast<::cxx::VectorSizeKind>(readEnum(in, 2));
  return control()->getUnresolvedVectorType(
      std::move(argument1), std::move(argument2), std::move(argument3),
      std::move(argument4), std::move(argument5));
}

auto SemanticDecoder::readTypeComplexType(ByteReader& in) -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  return control()->getComplexType(std::move(argument1));
}

auto SemanticDecoder::readTypeAtomicType(ByteReader& in) -> const cxx::Type* {
  const cxx::Type* argument1 = typeAt(TypeRef{in.varU32()});
  return control()->getAtomicType(std::move(argument1));
}

void SemanticDecoder::readSymbolNamespaceSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NamespaceSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ScopeSymbol::members_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addMember(element18);
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  std::vector<cxx::ScopeSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::ScopeSymbol* element22 =
          symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDirective(element23);
  }
  // ::cxx::NamespaceSymbol::unnamedNamespace_
  cxx::NamespaceSymbol* value24 =
      symbol_cast<NamespaceSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setUnnamedNamespace(std::move(value24));
  // ::cxx::NamespaceSymbol::anonNamespaceIndex_
  int value25 = static_cast<int>(in.varI32());
  if (std::move(value25) >= 0) self->setAnonNamespaceIndex(std::move(value25));
  // ::cxx::NamespaceSymbol::isInline_
  bool value26 = in.boolean();
  self->setInline(std::move(value26));
  // ::cxx::NamespaceSymbol::hasInlineNamespaces_
  bool value27 = in.boolean();
  self->setHasInlineNamespaces(std::move(value27));
}

void SemanticDecoder::readSymbolNamespaceAliasSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NamespaceAliasSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::NamespaceAliasSymbol::namespaceSymbol_
  cxx::NamespaceSymbol* value14 =
      symbol_cast<NamespaceSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setNamespaceSymbol(std::move(value14));
}

void SemanticDecoder::readSymbolConceptSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConceptSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::MaybeTemplate::declaration_
  cxx::ConceptDefinitionAST* value14 =
      ast_cast<ConceptDefinitionAST>(astAt(AstRef{in.varU32()}));
  self->setDeclaration(std::move(value14));
  // ::cxx::MaybeTemplate::templateDeclaration
  cxx::TemplateDeclarationAST* value15 =
      ast_cast<TemplateDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->setTemplateDeclaration(std::move(value15));
  // ::cxx::MaybeTemplate::templateParameters
  cxx::TemplateParametersSymbol* value16 =
      symbol_cast<TemplateParametersSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setTemplateParameters(std::move(value16));
  // ::cxx::MaybeTemplate::specializations
  std::vector<cxx::TemplateSpecialization> value17;
  {
    const auto count18 = in.varCount(1);
    for (std::uint32_t i19 = 0; ok() && i19 < count18; ++i19) {
      cxx::TemplateSpecialization element20{};
      readcxxTemplateSpecialization(in, &element20);
      value17.push_back(std::move(element20));
    }
  }
  for (auto&& element21 : value17) {
    self->restoreSpecialization(std::move(element21));
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  cxx::ConceptSymbol* value22 =
      symbol_cast<ConceptSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->restoreSpecializationInfo(std::move(value22),
                                  self->templateSpecializationIndex());
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  int value23 = static_cast<int>(in.varI32());
  self->restoreSpecializationInfo(self->primaryTemplateSymbol(),
                                  std::move(value23));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  std::vector<std::vector<cxx::TemplateArgument>> value24;
  {
    const auto count25 = in.varCount(1);
    for (std::uint32_t i26 = 0; ok() && i26 < count25; ++i26) {
      std::vector<cxx::TemplateArgument> element27;
      {
        const auto count28 = in.varCount(1);
        for (std::uint32_t i29 = 0; ok() && i29 < count28; ++i29) {
          cxx::TemplateArgument element30 = readTemplateArgument(in);
          element27.push_back(std::move(element30));
        }
      }
      value24.push_back(std::move(element27));
    }
  }
  for (auto&& element31 : value24) {
    self->addExternInstantiationDeclaration(element31);
  }
}

void SemanticDecoder::readSymbolDeductionGuideSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DeductionGuideSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::MaybeTemplate::declaration_
  cxx::DeductionGuideAST* value14 =
      ast_cast<DeductionGuideAST>(astAt(AstRef{in.varU32()}));
  self->setDeclaration(std::move(value14));
  // ::cxx::MaybeTemplate::templateDeclaration
  cxx::TemplateDeclarationAST* value15 =
      ast_cast<TemplateDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->setTemplateDeclaration(std::move(value15));
  // ::cxx::MaybeTemplate::templateParameters
  cxx::TemplateParametersSymbol* value16 =
      symbol_cast<TemplateParametersSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setTemplateParameters(std::move(value16));
  // ::cxx::MaybeTemplate::specializations
  std::vector<cxx::TemplateSpecialization> value17;
  {
    const auto count18 = in.varCount(1);
    for (std::uint32_t i19 = 0; ok() && i19 < count18; ++i19) {
      cxx::TemplateSpecialization element20{};
      readcxxTemplateSpecialization(in, &element20);
      value17.push_back(std::move(element20));
    }
  }
  for (auto&& element21 : value17) {
    self->restoreSpecialization(std::move(element21));
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  cxx::DeductionGuideSymbol* value22 =
      symbol_cast<DeductionGuideSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->restoreSpecializationInfo(std::move(value22),
                                  self->templateSpecializationIndex());
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  int value23 = static_cast<int>(in.varI32());
  self->restoreSpecializationInfo(self->primaryTemplateSymbol(),
                                  std::move(value23));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  std::vector<std::vector<cxx::TemplateArgument>> value24;
  {
    const auto count25 = in.varCount(1);
    for (std::uint32_t i26 = 0; ok() && i26 < count25; ++i26) {
      std::vector<cxx::TemplateArgument> element27;
      {
        const auto count28 = in.varCount(1);
        for (std::uint32_t i29 = 0; ok() && i29 < count28; ++i29) {
          cxx::TemplateArgument element30 = readTemplateArgument(in);
          element27.push_back(std::move(element30));
        }
      }
      value24.push_back(std::move(element27));
    }
  }
  for (auto&& element31 : value24) {
    self->addExternInstantiationDeclaration(element31);
  }
  // ::cxx::DeductionGuideSymbol::isExplicit_
  bool value32 = in.boolean();
  self->setExplicit(std::move(value32));
}

void SemanticDecoder::readSymbolClassSymbol(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::ClassSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ScopeSymbol::members_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addMember(element18);
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  std::vector<cxx::ScopeSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::ScopeSymbol* element22 =
          symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDirective(element23);
  }
  // ::cxx::MaybeTemplate::declaration_
  cxx::SpecifierAST* value24 =
      ast_cast<SpecifierAST>(astAt(AstRef{in.varU32()}));
  self->setDeclaration(std::move(value24));
  // ::cxx::MaybeTemplate::templateDeclaration
  cxx::TemplateDeclarationAST* value25 =
      ast_cast<TemplateDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->setTemplateDeclaration(std::move(value25));
  // ::cxx::MaybeTemplate::templateParameters
  cxx::TemplateParametersSymbol* value26 =
      symbol_cast<TemplateParametersSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setTemplateParameters(std::move(value26));
  // ::cxx::MaybeTemplate::specializations
  std::vector<cxx::TemplateSpecialization> value27;
  {
    const auto count28 = in.varCount(1);
    for (std::uint32_t i29 = 0; ok() && i29 < count28; ++i29) {
      cxx::TemplateSpecialization element30{};
      readcxxTemplateSpecialization(in, &element30);
      value27.push_back(std::move(element30));
    }
  }
  for (auto&& element31 : value27) {
    self->restoreSpecialization(std::move(element31));
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  cxx::ClassSymbol* value32 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->restoreSpecializationInfo(std::move(value32),
                                  self->templateSpecializationIndex());
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  int value33 = static_cast<int>(in.varI32());
  self->restoreSpecializationInfo(self->primaryTemplateSymbol(),
                                  std::move(value33));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  std::vector<std::vector<cxx::TemplateArgument>> value34;
  {
    const auto count35 = in.varCount(1);
    for (std::uint32_t i36 = 0; ok() && i36 < count35; ++i36) {
      std::vector<cxx::TemplateArgument> element37;
      {
        const auto count38 = in.varCount(1);
        for (std::uint32_t i39 = 0; ok() && i39 < count38; ++i39) {
          cxx::TemplateArgument element40 = readTemplateArgument(in);
          element37.push_back(std::move(element40));
        }
      }
      value34.push_back(std::move(element37));
    }
  }
  for (auto&& element41 : value34) {
    self->addExternInstantiationDeclaration(element41);
  }
  // ::cxx::MaybeRedecl::canonical_
  cxx::ClassSymbol* value42 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setCanonical(std::move(value42));
  // ::cxx::MaybeRedecl::definition_
  cxx::ClassSymbol* value43 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setDefinition(std::move(value43));
  // ::cxx::MaybeRedecl::redeclarations_
  std::vector<cxx::ClassSymbol*> value44;
  {
    const auto count45 = in.varCount(1);
    for (std::uint32_t i46 = 0; ok() && i46 < count45; ++i46) {
      cxx::ClassSymbol* element47 =
          symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value44.push_back(std::move(element47));
    }
  }
  for (auto&& element48 : value44) {
    self->addRedeclaration(element48);
  }
  // ::cxx::ClassSymbol::baseClasses_
  std::vector<cxx::BaseClassSymbol*> value49;
  {
    const auto count50 = in.varCount(1);
    for (std::uint32_t i51 = 0; ok() && i51 < count50; ++i51) {
      cxx::BaseClassSymbol* element52 =
          symbol_cast<BaseClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value49.push_back(std::move(element52));
    }
  }
  for (auto&& element53 : value49) {
    self->addBaseClass(element53);
  }
  // ::cxx::ClassSymbol::befriendingClasses_
  std::vector<cxx::ClassSymbol*> value54;
  {
    const auto count55 = in.varCount(1);
    for (std::uint32_t i56 = 0; ok() && i56 < count55; ++i56) {
      cxx::ClassSymbol* element57 =
          symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value54.push_back(std::move(element57));
    }
  }
  for (auto&& element58 : value54) {
    self->addBefriendingClass(element58);
  }
  // ::cxx::ClassSymbol::templateFriendships_
  std::vector<cxx::TemplateFriendship> value59;
  {
    const auto count60 = in.varCount(1);
    for (std::uint32_t i61 = 0; ok() && i61 < count60; ++i61) {
      cxx::TemplateFriendship element62{};
      readcxxTemplateFriendship(in, &element62);
      value59.push_back(std::move(element62));
    }
  }
  for (auto&& element63 : value59) {
    self->addBefriendingClass(element63.befriendingClass, element63.arguments);
  }
  // ::cxx::ClassSymbol::instantiationPattern_
  cxx::ClassSymbol* value64 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setInstantiationPattern(std::move(value64));
  // ::cxx::ClassSymbol::instantiationSubstitutionArguments_
  std::vector<cxx::TemplateArgument> value65;
  {
    const auto count66 = in.varCount(1);
    for (std::uint32_t i67 = 0; ok() && i67 < count66; ++i67) {
      cxx::TemplateArgument element68 = readTemplateArgument(in);
      value65.push_back(std::move(element68));
    }
  }
  self->setInstantiationSubstitution(self->instantiationSubstitutionDepth(),
                                     std::move(std::move(value65)));
  // ::cxx::ClassSymbol::instantiationSubstitutionDepth_
  int value69 = static_cast<int>(in.varI32());
  self->setInstantiationSubstitution(
      std::move(value69), self->instantiationSubstitutionArguments());
  // ::cxx::ClassSymbol::constructorOverloadSet_
  cxx::OverloadSetSymbol* value70 =
      symbol_cast<OverloadSetSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setConstructorOverloadSet(std::move(value70));
  // ::cxx::ClassSymbol::deductionGuides_
  std::vector<cxx::DeductionGuideSymbol*> value71;
  {
    const auto count72 = in.varCount(1);
    for (std::uint32_t i73 = 0; ok() && i73 < count72; ++i73) {
      cxx::DeductionGuideSymbol* element74 =
          symbol_cast<DeductionGuideSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value71.push_back(std::move(element74));
    }
  }
  for (auto&& element75 : value71) {
    self->addDeductionGuide(element75);
  }
  // ::cxx::ClassSymbol::layout_
  std::unique_ptr<cxx::ClassLayout> value76;
  if (in.boolean()) {
    value76 = std::make_unique<cxx::ClassLayout>();
    readcxxClassLayout(in, value76.get());
  }
  self->setLayout(std::move(std::move(value76)));
  // ::cxx::ClassSymbol::vtableLayout_
  std::unique_ptr<cxx::VTableLayout> value77;
  if (in.boolean()) {
    value77 = std::make_unique<cxx::VTableLayout>();
    readcxxVTableLayout(in, value77.get());
  }
  self->setVTableLayout(std::move(std::move(value77)));
  // ::cxx::ClassSymbol::capturedThisField_
  cxx::FieldSymbol* value78 =
      symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setCapturedThisField(std::move(value78));
  // ::cxx::ClassSymbol::closureDiscriminator_
  int value79 = static_cast<int>(in.varI32());
  self->setClosureDiscriminator(std::move(value79));
  // ::cxx::ClassSymbol::sizeInBytes_
  int value80 = static_cast<int>(in.varI32());
  self->setSizeInBytes(std::move(value80));
  // ::cxx::ClassSymbol::alignment_
  int value81 = static_cast<int>(in.varI32());
  self->setAlignment(std::move(value81));
  // ::cxx::ClassSymbol::explicitAlignment_
  int value82 = static_cast<int>(in.varI32());
  self->setExplicitAlignment(std::move(value82));
  // ::cxx::ClassSymbol::packAlignment_
  int value83 = static_cast<int>(in.varI32());
  self->setPackAlignment(std::move(value83));
  // ::cxx::ClassSymbol::isUnion_
  unsigned int value84 = static_cast<unsigned int>(in.varU32());
  self->setIsUnion(std::move(value84));
  // ::cxx::ClassSymbol::isFinal_
  unsigned int value85 = static_cast<unsigned int>(in.varU32());
  self->setFinal(std::move(value85));
  // ::cxx::ClassSymbol::isComplete_
  unsigned int value86 = static_cast<unsigned int>(in.varU32());
  self->setComplete(std::move(value86));
  // ::cxx::ClassSymbol::isFriend_
  unsigned int value87 = static_cast<unsigned int>(in.varU32());
  self->setFriend(std::move(value87));
  // ::cxx::ClassSymbol::isAccessControlDisabled_
  unsigned int value88 = static_cast<unsigned int>(in.varU32());
  self->setAccessControlDisabled(std::move(value88));
  // ::cxx::ClassSymbol::isPolymorphic_
  unsigned int value89 = static_cast<unsigned int>(in.varU32());
  self->setPolymorphic(std::move(value89));
  // ::cxx::ClassSymbol::isAbstract_
  unsigned int value90 = static_cast<unsigned int>(in.varU32());
  self->setAbstract(std::move(value90));
  // ::cxx::ClassSymbol::hasVirtualDestructor_
  unsigned int value91 = static_cast<unsigned int>(in.varU32());
  self->setHasVirtualDestructor(std::move(value91));
  // ::cxx::ClassSymbol::isClosureType_
  unsigned int value92 = static_cast<unsigned int>(in.varU32());
  self->setIsClosureType(std::move(value92));
  // ::cxx::ClassSymbol::hasLambdaCapture_
  unsigned int value93 = static_cast<unsigned int>(in.varU32());
  self->setHasLambdaCapture(std::move(value93));
  // ::cxx::ClassSymbol::hasUserDeclaredConstructors_
  unsigned int value94 = static_cast<unsigned int>(in.varU32());
  self->setHasUserDeclaredConstructors(std::move(value94));
}

void SemanticDecoder::readSymbolEnumSymbol(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::EnumSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ScopeSymbol::members_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addMember(element18);
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  std::vector<cxx::ScopeSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::ScopeSymbol* element22 =
          symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDirective(element23);
  }
  // ::cxx::EnumSymbol::underlyingType_
  const cxx::Type* value24 = typeAt(TypeRef{in.varU32()});
  self->setUnderlyingType(std::move(value24));
  // ::cxx::EnumSymbol::hasFixedUnderlyingType_
  bool value25 = in.boolean();
  self->setHasFixedUnderlyingType(std::move(value25));
  // ::cxx::EnumSymbol::isDefined_
  bool value26 = in.boolean();
  self->setDefined(std::move(value26));
}

void SemanticDecoder::readSymbolScopedEnumSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ScopedEnumSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ScopeSymbol::members_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addMember(element18);
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  std::vector<cxx::ScopeSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::ScopeSymbol* element22 =
          symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDirective(element23);
  }
  // ::cxx::ScopedEnumSymbol::underlyingType_
  const cxx::Type* value24 = typeAt(TypeRef{in.varU32()});
  self->setUnderlyingType(std::move(value24));
  // ::cxx::ScopedEnumSymbol::isDefined_
  bool value25 = in.boolean();
  self->setDefined(std::move(value25));
}

void SemanticDecoder::readSymbolFunctionSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::FunctionSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ScopeSymbol::members_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addMember(element18);
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  std::vector<cxx::ScopeSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::ScopeSymbol* element22 =
          symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDirective(element23);
  }
  // ::cxx::MaybeTemplate::declaration_
  cxx::FunctionDefinitionAST* value24 =
      ast_cast<FunctionDefinitionAST>(astAt(AstRef{in.varU32()}));
  self->setDeclaration(std::move(value24));
  // ::cxx::MaybeTemplate::templateDeclaration
  cxx::TemplateDeclarationAST* value25 =
      ast_cast<TemplateDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->setTemplateDeclaration(std::move(value25));
  // ::cxx::MaybeTemplate::templateParameters
  cxx::TemplateParametersSymbol* value26 =
      symbol_cast<TemplateParametersSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setTemplateParameters(std::move(value26));
  // ::cxx::MaybeTemplate::specializations
  std::vector<cxx::TemplateSpecialization> value27;
  {
    const auto count28 = in.varCount(1);
    for (std::uint32_t i29 = 0; ok() && i29 < count28; ++i29) {
      cxx::TemplateSpecialization element30{};
      readcxxTemplateSpecialization(in, &element30);
      value27.push_back(std::move(element30));
    }
  }
  for (auto&& element31 : value27) {
    self->restoreSpecialization(std::move(element31));
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  cxx::FunctionSymbol* value32 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->restoreSpecializationInfo(std::move(value32),
                                  self->templateSpecializationIndex());
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  int value33 = static_cast<int>(in.varI32());
  self->restoreSpecializationInfo(self->primaryTemplateSymbol(),
                                  std::move(value33));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  std::vector<std::vector<cxx::TemplateArgument>> value34;
  {
    const auto count35 = in.varCount(1);
    for (std::uint32_t i36 = 0; ok() && i36 < count35; ++i36) {
      std::vector<cxx::TemplateArgument> element37;
      {
        const auto count38 = in.varCount(1);
        for (std::uint32_t i39 = 0; ok() && i39 < count38; ++i39) {
          cxx::TemplateArgument element40 = readTemplateArgument(in);
          element37.push_back(std::move(element40));
        }
      }
      value34.push_back(std::move(element37));
    }
  }
  for (auto&& element41 : value34) {
    self->addExternInstantiationDeclaration(element41);
  }
  // ::cxx::MaybeRedecl::canonical_
  cxx::FunctionSymbol* value42 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setCanonical(std::move(value42));
  // ::cxx::MaybeRedecl::definition_
  cxx::FunctionSymbol* value43 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setDefinition(std::move(value43));
  // ::cxx::MaybeRedecl::redeclarations_
  std::vector<cxx::FunctionSymbol*> value44;
  {
    const auto count45 = in.varCount(1);
    for (std::uint32_t i46 = 0; ok() && i46 < count45; ++i46) {
      cxx::FunctionSymbol* element47 =
          symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value44.push_back(std::move(element47));
    }
  }
  for (auto&& element48 : value44) {
    self->addRedeclaration(element48);
  }
  // ::cxx::FunctionSymbol::pendingBody_
  std::unique_ptr<cxx::PendingBodyInstantiation> value49;
  if (in.boolean()) {
    value49 = std::make_unique<cxx::PendingBodyInstantiation>();
    readcxxPendingBodyInstantiation(in, value49.get());
  }
  self->setPendingBody(std::move(std::move(value49)));
  // ::cxx::FunctionSymbol::pendingExceptionSpecification_
  std::unique_ptr<cxx::PendingExceptionSpecification> value50;
  if (in.boolean()) {
    value50 = std::make_unique<cxx::PendingExceptionSpecification>();
    readcxxPendingExceptionSpecification(in, value50.get());
  }
  self->setPendingExceptionSpecification(std::move(std::move(value50)));
  // ::cxx::FunctionSymbol::completeObjectVariant_
  cxx::FunctionSymbol* value51 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setCompleteObjectVariant(std::move(value51));
  // ::cxx::FunctionSymbol::delegatingConstructor_
  cxx::FunctionSymbol* value52 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setDelegatingConstructor(std::move(value52));
  // ::cxx::FunctionSymbol::deletingDtorVariant_
  cxx::FunctionSymbol* value53 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setDeletingDtorVariant(std::move(value53));
  // ::cxx::FunctionSymbol::structorPrincipal_
  cxx::FunctionSymbol* value54 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setStructorPrincipal(std::move(value54));
  // ::cxx::FunctionSymbol::inheritedConstructor_
  cxx::FunctionSymbol* value55 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setInheritedConstructor(std::move(value55));
  // ::cxx::FunctionSymbol::overriddenFunctions_
  std::vector<cxx::FunctionSymbol*> value56;
  {
    const auto count57 = in.varCount(1);
    for (std::uint32_t i58 = 0; ok() && i58 < count57; ++i58) {
      cxx::FunctionSymbol* element59 =
          symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value56.push_back(std::move(element59));
    }
  }
  for (auto&& element60 : value56) {
    self->addOverriddenFunction(element60);
  }
  // ::cxx::FunctionSymbol::befriendingClasses_
  std::vector<cxx::ClassSymbol*> value61;
  {
    const auto count62 = in.varCount(1);
    for (std::uint32_t i63 = 0; ok() && i63 < count62; ++i63) {
      cxx::ClassSymbol* element64 =
          symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value61.push_back(std::move(element64));
    }
  }
  for (auto&& element65 : value61) {
    self->addBefriendingClass(element65);
  }
  // ::cxx::FunctionSymbol::templateFriendships_
  std::vector<cxx::TemplateFriendship> value66;
  {
    const auto count67 = in.varCount(1);
    for (std::uint32_t i68 = 0; ok() && i68 < count67; ++i68) {
      cxx::TemplateFriendship element69{};
      readcxxTemplateFriendship(in, &element69);
      value66.push_back(std::move(element69));
    }
  }
  for (auto&& element70 : value66) {
    self->addBefriendingClass(element70.befriendingClass, element70.arguments);
  }
  // ::cxx::FunctionSymbol::vtableSlotIndex_
  int value71 = static_cast<int>(in.varI32());
  self->setVtableSlotIndex(std::move(value71));
  // ::cxx::FunctionSymbol::externalName_
  const cxx::Identifier* value72 = identifierAt(StringRef{in.varU32()});
  self->setExternalName(std::move(value72));
  // ::cxx::FunctionSymbol::aliasName_
  const cxx::Identifier* value73 = identifierAt(StringRef{in.varU32()});
  self->setAliasName(std::move(value73));
  // ::cxx::FunctionSymbol::importModule_
  const cxx::Identifier* value74 = identifierAt(StringRef{in.varU32()});
  self->setImportModule(std::move(value74));
  // ::cxx::FunctionSymbol::importName_
  const cxx::Identifier* value75 = identifierAt(StringRef{in.varU32()});
  self->setImportName(std::move(value75));
  // ::cxx::FunctionSymbol::exportName_
  const cxx::Identifier* value76 = identifierAt(StringRef{in.varU32()});
  self->setExportName(std::move(value76));
  // ::cxx::FunctionSymbol::trailingRequiresClause_
  cxx::RequiresClauseAST* value77 =
      ast_cast<RequiresClauseAST>(astAt(AstRef{in.varU32()}));
  self->setTrailingRequiresClause(std::move(value77));
  // ::cxx::FunctionSymbol::builtinKind_
  static_assert(static_cast<std::uint32_t>(
                    ::cxx::BuiltinFunctionKind::T___C11_ATOMIC_THREAD_FENCE) +
                    1 ==
                450);
  ::cxx::BuiltinFunctionKind value78 =
      static_cast<::cxx::BuiltinFunctionKind>(readEnum(in, 450));
  self->setBuiltinKind(std::move(value78));
  // ::cxx::FunctionSymbol::isDefined_
  unsigned int value79 = static_cast<unsigned int>(in.varU32());
  self->setDefined(std::move(value79));
  // ::cxx::FunctionSymbol::isStatic_
  unsigned int value80 = static_cast<unsigned int>(in.varU32());
  self->setStatic(std::move(value80));
  // ::cxx::FunctionSymbol::isExtern_
  unsigned int value81 = static_cast<unsigned int>(in.varU32());
  self->setExtern(std::move(value81));
  // ::cxx::FunctionSymbol::isFriend_
  unsigned int value82 = static_cast<unsigned int>(in.varU32());
  self->setFriend(std::move(value82));
  // ::cxx::FunctionSymbol::isConstexpr_
  unsigned int value83 = static_cast<unsigned int>(in.varU32());
  self->setConstexpr(std::move(value83));
  // ::cxx::FunctionSymbol::isConsteval_
  unsigned int value84 = static_cast<unsigned int>(in.varU32());
  self->setConsteval(std::move(value84));
  // ::cxx::FunctionSymbol::isInline_
  unsigned int value85 = static_cast<unsigned int>(in.varU32());
  self->setInline(std::move(value85));
  // ::cxx::FunctionSymbol::isVirtual_
  unsigned int value86 = static_cast<unsigned int>(in.varU32());
  self->setVirtual(std::move(value86));
  // ::cxx::FunctionSymbol::isExplicit_
  unsigned int value87 = static_cast<unsigned int>(in.varU32());
  self->setExplicit(std::move(value87));
  // ::cxx::FunctionSymbol::isDeleted_
  unsigned int value88 = static_cast<unsigned int>(in.varU32());
  self->setDeleted(std::move(value88));
  // ::cxx::FunctionSymbol::isDefaulted_
  unsigned int value89 = static_cast<unsigned int>(in.varU32());
  self->setDefaulted(std::move(value89));
  // ::cxx::FunctionSymbol::isPure_
  unsigned int value90 = static_cast<unsigned int>(in.varU32());
  self->setPure(std::move(value90));
  // ::cxx::FunctionSymbol::hasCLinkage_
  unsigned int value91 = static_cast<unsigned int>(in.varU32());
  self->setLanguageLinkage(std::move(value91) ? LanguageKind::kC
                                              : LanguageKind::kCXX);
  // ::cxx::FunctionSymbol::isOverride_
  unsigned int value92 = static_cast<unsigned int>(in.varU32());
  self->setOverride(std::move(value92));
  // ::cxx::FunctionSymbol::isFinal_
  unsigned int value93 = static_cast<unsigned int>(in.varU32());
  self->setFinal(std::move(value93));
  // ::cxx::FunctionSymbol::hasNoPrototype_
  unsigned int value94 = static_cast<unsigned int>(in.varU32());
  self->setNoPrototype(std::move(value94));
  // ::cxx::FunctionSymbol::hasHiddenVisibility_
  unsigned int value95 = static_cast<unsigned int>(in.varU32());
  self->setHiddenVisibility(std::move(value95));
  // ::cxx::FunctionSymbol::hasExceptionSpecifier_
  unsigned int value96 = static_cast<unsigned int>(in.varU32());
  self->setExceptionSpecifier(std::move(value96));
  // ::cxx::FunctionSymbol::isDefinitionRequired_
  unsigned int value97 = static_cast<unsigned int>(in.varU32());
  self->setDefinitionRequired(std::move(value97));
  // ::cxx::FunctionSymbol::hasExplicitObjectParameter_
  unsigned int value98 = static_cast<unsigned int>(in.varU32());
  self->setExplicitObjectParameter(std::move(value98));
  // ::cxx::FunctionSymbol::isNoReturn_
  unsigned int value99 = static_cast<unsigned int>(in.varU32());
  self->setNoReturn(std::move(value99));
}

void SemanticDecoder::readSymbolTypeAliasSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeAliasSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::MaybeTemplate::declaration_
  cxx::AliasDeclarationAST* value14 =
      ast_cast<AliasDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->setDeclaration(std::move(value14));
  // ::cxx::MaybeTemplate::templateDeclaration
  cxx::TemplateDeclarationAST* value15 =
      ast_cast<TemplateDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->setTemplateDeclaration(std::move(value15));
  // ::cxx::MaybeTemplate::templateParameters
  cxx::TemplateParametersSymbol* value16 =
      symbol_cast<TemplateParametersSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setTemplateParameters(std::move(value16));
  // ::cxx::MaybeTemplate::specializations
  std::vector<cxx::TemplateSpecialization> value17;
  {
    const auto count18 = in.varCount(1);
    for (std::uint32_t i19 = 0; ok() && i19 < count18; ++i19) {
      cxx::TemplateSpecialization element20{};
      readcxxTemplateSpecialization(in, &element20);
      value17.push_back(std::move(element20));
    }
  }
  for (auto&& element21 : value17) {
    self->restoreSpecialization(std::move(element21));
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  cxx::TypeAliasSymbol* value22 =
      symbol_cast<TypeAliasSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->restoreSpecializationInfo(std::move(value22),
                                  self->templateSpecializationIndex());
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  int value23 = static_cast<int>(in.varI32());
  self->restoreSpecializationInfo(self->primaryTemplateSymbol(),
                                  std::move(value23));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  std::vector<std::vector<cxx::TemplateArgument>> value24;
  {
    const auto count25 = in.varCount(1);
    for (std::uint32_t i26 = 0; ok() && i26 < count25; ++i26) {
      std::vector<cxx::TemplateArgument> element27;
      {
        const auto count28 = in.varCount(1);
        for (std::uint32_t i29 = 0; ok() && i29 < count28; ++i29) {
          cxx::TemplateArgument element30 = readTemplateArgument(in);
          element27.push_back(std::move(element30));
        }
      }
      value24.push_back(std::move(element27));
    }
  }
  for (auto&& element31 : value24) {
    self->addExternInstantiationDeclaration(element31);
  }
  // ::cxx::MaybeRedecl::canonical_
  cxx::TypeAliasSymbol* value32 =
      symbol_cast<TypeAliasSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setCanonical(std::move(value32));
  // ::cxx::MaybeRedecl::definition_
  cxx::TypeAliasSymbol* value33 =
      symbol_cast<TypeAliasSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setDefinition(std::move(value33));
  // ::cxx::MaybeRedecl::redeclarations_
  std::vector<cxx::TypeAliasSymbol*> value34;
  {
    const auto count35 = in.varCount(1);
    for (std::uint32_t i36 = 0; ok() && i36 < count35; ++i36) {
      cxx::TypeAliasSymbol* element37 =
          symbol_cast<TypeAliasSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value34.push_back(std::move(element37));
    }
  }
  for (auto&& element38 : value34) {
    self->addRedeclaration(element38);
  }
  // ::cxx::TypeAliasSymbol::expansionTypeId_
  cxx::TypeIdAST* value39 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->setExpansionTypeId(std::move(value39));
}

void SemanticDecoder::readSymbolVariableSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::VariableSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::MaybeTemplate::declaration_
  cxx::SimpleDeclarationAST* value14 =
      ast_cast<SimpleDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->setDeclaration(std::move(value14));
  // ::cxx::MaybeTemplate::templateDeclaration
  cxx::TemplateDeclarationAST* value15 =
      ast_cast<TemplateDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->setTemplateDeclaration(std::move(value15));
  // ::cxx::MaybeTemplate::templateParameters
  cxx::TemplateParametersSymbol* value16 =
      symbol_cast<TemplateParametersSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setTemplateParameters(std::move(value16));
  // ::cxx::MaybeTemplate::specializations
  std::vector<cxx::TemplateSpecialization> value17;
  {
    const auto count18 = in.varCount(1);
    for (std::uint32_t i19 = 0; ok() && i19 < count18; ++i19) {
      cxx::TemplateSpecialization element20{};
      readcxxTemplateSpecialization(in, &element20);
      value17.push_back(std::move(element20));
    }
  }
  for (auto&& element21 : value17) {
    self->restoreSpecialization(std::move(element21));
  }
  // ::cxx::MaybeTemplate::primaryTemplateSymbol
  cxx::VariableSymbol* value22 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->restoreSpecializationInfo(std::move(value22),
                                  self->templateSpecializationIndex());
  // ::cxx::MaybeTemplate::templateSpecializationIndex
  int value23 = static_cast<int>(in.varI32());
  self->restoreSpecializationInfo(self->primaryTemplateSymbol(),
                                  std::move(value23));
  // ::cxx::MaybeTemplate::externInstantiationDeclarations
  std::vector<std::vector<cxx::TemplateArgument>> value24;
  {
    const auto count25 = in.varCount(1);
    for (std::uint32_t i26 = 0; ok() && i26 < count25; ++i26) {
      std::vector<cxx::TemplateArgument> element27;
      {
        const auto count28 = in.varCount(1);
        for (std::uint32_t i29 = 0; ok() && i29 < count28; ++i29) {
          cxx::TemplateArgument element30 = readTemplateArgument(in);
          element27.push_back(std::move(element30));
        }
      }
      value24.push_back(std::move(element27));
    }
  }
  for (auto&& element31 : value24) {
    self->addExternInstantiationDeclaration(element31);
  }
  // ::cxx::MaybeRedecl::canonical_
  cxx::VariableSymbol* value32 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setCanonical(std::move(value32));
  // ::cxx::MaybeRedecl::definition_
  cxx::VariableSymbol* value33 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setDefinition(std::move(value33));
  // ::cxx::MaybeRedecl::redeclarations_
  std::vector<cxx::VariableSymbol*> value34;
  {
    const auto count35 = in.varCount(1);
    for (std::uint32_t i36 = 0; ok() && i36 < count35; ++i36) {
      cxx::VariableSymbol* element37 =
          symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value34.push_back(std::move(element37));
    }
  }
  for (auto&& element38 : value34) {
    self->addRedeclaration(element38);
  }
  // ::cxx::VariableSymbol::initializer_
  cxx::ExpressionAST* value39 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->setInitializer(std::move(value39));
  // ::cxx::VariableSymbol::constructor_
  cxx::FunctionSymbol* value40 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setConstructor(std::move(value40));
  // ::cxx::VariableSymbol::constValue_
  std::optional<cxx::ConstValue> value41;
  if (in.boolean()) {
    cxx::ConstValue value42 = readConstValue(in);
    value41 = std::move(value42);
  }
  self->setConstValue(std::move(value41));
  // ::cxx::VariableSymbol::explicitAlignment_
  int value43 = static_cast<int>(in.varI32());
  self->setExplicitAlignment(std::move(value43));
  // ::cxx::VariableSymbol::isStatic_
  unsigned int value44 = static_cast<unsigned int>(in.varU32());
  self->setStatic(std::move(value44));
  // ::cxx::VariableSymbol::isThreadLocal_
  unsigned int value45 = static_cast<unsigned int>(in.varU32());
  self->setThreadLocal(std::move(value45));
  // ::cxx::VariableSymbol::isExtern_
  unsigned int value46 = static_cast<unsigned int>(in.varU32());
  self->setExtern(std::move(value46));
  // ::cxx::VariableSymbol::isConstexpr_
  unsigned int value47 = static_cast<unsigned int>(in.varU32());
  self->setConstexpr(std::move(value47));
  // ::cxx::VariableSymbol::isConstinit_
  unsigned int value48 = static_cast<unsigned int>(in.varU32());
  self->setConstinit(std::move(value48));
  // ::cxx::VariableSymbol::isInline_
  unsigned int value49 = static_cast<unsigned int>(in.varU32());
  self->setInline(std::move(value49));
}

void SemanticDecoder::readSymbolFieldSymbol(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::FieldSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::FieldSymbol::definition_
  cxx::VariableSymbol* value14 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setDefinition(std::move(value14));
  // ::cxx::FieldSymbol::pendingInitializer_
  std::unique_ptr<cxx::PendingFieldInitializerInstantiation> value15;
  if (in.boolean()) {
    value15 = std::make_unique<cxx::PendingFieldInitializerInstantiation>();
    readcxxPendingFieldInitializerInstantiation(in, value15.get());
  }
  self->setPendingInitializer(std::move(std::move(value15)));
  // ::cxx::FieldSymbol::constValue_
  std::optional<cxx::ConstValue> value16;
  if (in.boolean()) {
    cxx::ConstValue value17 = readConstValue(in);
    value16 = std::move(value17);
  }
  self->setConstValue(std::move(value16));
  // ::cxx::FieldSymbol::isDefinitionRequired_
  unsigned int value18 = static_cast<unsigned int>(in.varU32());
  self->setDefinitionRequired(std::move(value18));
  // ::cxx::FieldSymbol::isBitField_
  unsigned int value19 = static_cast<unsigned int>(in.varU32());
  self->setBitField(std::move(value19));
  // ::cxx::FieldSymbol::isStatic_
  unsigned int value20 = static_cast<unsigned int>(in.varU32());
  self->setStatic(std::move(value20));
  // ::cxx::FieldSymbol::isThreadLocal_
  unsigned int value21 = static_cast<unsigned int>(in.varU32());
  self->setThreadLocal(std::move(value21));
  // ::cxx::FieldSymbol::isConstexpr_
  unsigned int value22 = static_cast<unsigned int>(in.varU32());
  self->setConstexpr(std::move(value22));
  // ::cxx::FieldSymbol::isConstinit_
  unsigned int value23 = static_cast<unsigned int>(in.varU32());
  self->setConstinit(std::move(value23));
  // ::cxx::FieldSymbol::isInline_
  unsigned int value24 = static_cast<unsigned int>(in.varU32());
  self->setInline(std::move(value24));
  // ::cxx::FieldSymbol::isMutable_
  unsigned int value25 = static_cast<unsigned int>(in.varU32());
  self->setMutable(std::move(value25));
  // ::cxx::FieldSymbol::isNoUniqueAddress_
  unsigned int value26 = static_cast<unsigned int>(in.varU32());
  self->setNoUniqueAddress(std::move(value26));
  // ::cxx::FieldSymbol::localOffset_
  int value27 = static_cast<int>(in.varI32());
  self->setLocalOffset(std::move(value27));
  // ::cxx::FieldSymbol::alignment_
  int value28 = static_cast<int>(in.varI32());
  self->setAlignment(std::move(value28));
  // ::cxx::FieldSymbol::bitFieldOffset_
  int value29 = static_cast<int>(in.varI32());
  self->setBitFieldOffset(std::move(value29));
  // ::cxx::FieldSymbol::bitFieldWidth_
  std::optional<cxx::ConstValue> value30;
  if (in.boolean()) {
    cxx::ConstValue value31 = readConstValue(in);
    value30 = std::move(value31);
  }
  self->setBitFieldWidth(std::move(value30));
  // ::cxx::FieldSymbol::initializer_
  cxx::ExpressionAST* value32 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->setInitializer(std::move(value32));
  // ::cxx::FieldSymbol::constructor_
  cxx::FunctionSymbol* value33 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setConstructor(std::move(value33));
}

void SemanticDecoder::readSymbolParameterSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ParameterSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ParameterSymbol::defaultArgument_
  cxx::ExpressionAST* value14 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->setDefaultArgument(std::move(value14));
  // ::cxx::ParameterSymbol::isExplicitObject_
  bool value15 = in.boolean();
  self->setExplicitObject(std::move(value15));
}

void SemanticDecoder::readSymbolParameterPackSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ParameterPackSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ParameterPackSymbol::elements_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addElement(element18);
  }
}

void SemanticDecoder::readSymbolEnumeratorSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::EnumeratorSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::EnumeratorSymbol::value_
  std::optional<cxx::ConstValue> value14;
  if (in.boolean()) {
    cxx::ConstValue value15 = readConstValue(in);
    value14 = std::move(value15);
  }
  self->setValue(std::move(value14));
}

void SemanticDecoder::readSymbolFunctionParametersSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::FunctionParametersSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ScopeSymbol::members_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addMember(element18);
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  std::vector<cxx::ScopeSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::ScopeSymbol* element22 =
          symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDirective(element23);
  }
}

void SemanticDecoder::readSymbolTemplateParametersSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TemplateParametersSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ScopeSymbol::members_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addMember(element18);
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  std::vector<cxx::ScopeSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::ScopeSymbol* element22 =
          symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDirective(element23);
  }
  // ::cxx::TemplateParametersSymbol::isExplicitTemplateSpecialization_
  bool value24 = in.boolean();
  self->setExplicitTemplateSpecialization(std::move(value24));
}

void SemanticDecoder::readSymbolBlockSymbol(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::BlockSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ScopeSymbol::members_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addMember(element18);
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  std::vector<cxx::ScopeSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::ScopeSymbol* element22 =
          symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDirective(element23);
  }
  // ::cxx::BlockSymbol::isOutermostBlockScope_
  bool value24 = in.boolean();
  self->setOutermostBlockScope(std::move(value24));
}

void SemanticDecoder::readSymbolLambdaSymbol(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::LambdaSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ScopeSymbol::members_
  std::vector<cxx::Symbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::Symbol* element17 = symbolAt(SymbolRef{in.varU32()});
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addMember(element18);
  }
  // ::cxx::ScopeSymbol::usingDirectives_
  std::vector<cxx::ScopeSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::ScopeSymbol* element22 =
          symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDirective(element23);
  }
  // ::cxx::LambdaSymbol::closureType_
  cxx::ClassSymbol* value24 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setClosureType(std::move(value24));
  // ::cxx::LambdaSymbol::isConstexpr_
  unsigned int value25 = static_cast<unsigned int>(in.varU32());
  self->setConstexpr(std::move(value25));
  // ::cxx::LambdaSymbol::isConsteval_
  unsigned int value26 = static_cast<unsigned int>(in.varU32());
  self->setConsteval(std::move(value26));
  // ::cxx::LambdaSymbol::isMutable_
  unsigned int value27 = static_cast<unsigned int>(in.varU32());
  self->setMutable(std::move(value27));
  // ::cxx::LambdaSymbol::isStatic_
  unsigned int value28 = static_cast<unsigned int>(in.varU32());
  self->setStatic(std::move(value28));
  // ::cxx::LambdaSymbol::isTemplate_
  unsigned int value29 = static_cast<unsigned int>(in.varU32());
  self->setTemplate(std::move(value29));
  // ::cxx::LambdaSymbol::isInTemplate_
  unsigned int value30 = static_cast<unsigned int>(in.varU32());
  self->setInTemplate(std::move(value30));
}

void SemanticDecoder::readSymbolTypeParameterSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeParameterSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
}

void SemanticDecoder::readSymbolNonTypeParameterSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NonTypeParameterSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::NonTypeParameterSymbol::objectType_
  const cxx::Type* value14 = typeAt(TypeRef{in.varU32()});
  self->setObjectType(std::move(value14));
  // ::cxx::NonTypeParameterSymbol::index_
  int value15 = static_cast<int>(in.varI32());
  self->setIndex(std::move(value15));
  // ::cxx::NonTypeParameterSymbol::depth_
  int value16 = static_cast<int>(in.varI32());
  self->setDepth(std::move(value16));
  // ::cxx::NonTypeParameterSymbol::isParameterPack_
  bool value17 = in.boolean();
  self->setParameterPack(std::move(value17));
}

void SemanticDecoder::readSymbolTemplateTypeParameterSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TemplateTypeParameterSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
}

void SemanticDecoder::readSymbolConstraintTypeParameterSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstraintTypeParameterSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::ConstraintTypeParameterSymbol::index_
  int value14 = static_cast<int>(in.varI32());
  self->setIndex(std::move(value14));
  // ::cxx::ConstraintTypeParameterSymbol::depth_
  int value15 = static_cast<int>(in.varI32());
  self->setDepth(std::move(value15));
  // ::cxx::ConstraintTypeParameterSymbol::isParameterPack_
  bool value16 = in.boolean();
  self->setParameterPack(std::move(value16));
  // ::cxx::ConstraintTypeParameterSymbol::typeConstraint_
  cxx::TypeConstraintAST* value17 =
      ast_cast<TypeConstraintAST>(astAt(AstRef{in.varU32()}));
  self->setTypeConstraint(std::move(value17));
  // ::cxx::ConstraintTypeParameterSymbol::constraintExpression_
  cxx::ExpressionAST* value18 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->setConstraintExpression(std::move(value18));
}

void SemanticDecoder::readSymbolOverloadSetSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::OverloadSetSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::OverloadSetSymbol::declaredFunctions_
  std::vector<cxx::FunctionSymbol*> value14;
  {
    const auto count15 = in.varCount(1);
    for (std::uint32_t i16 = 0; ok() && i16 < count15; ++i16) {
      cxx::FunctionSymbol* element17 =
          symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value14.push_back(std::move(element17));
    }
  }
  for (auto&& element18 : value14) {
    self->addFunction(element18);
  }
  // ::cxx::OverloadSetSymbol::usingDeclarations_
  std::vector<cxx::UsingDeclarationSymbol*> value19;
  {
    const auto count20 = in.varCount(1);
    for (std::uint32_t i21 = 0; ok() && i21 < count20; ++i21) {
      cxx::UsingDeclarationSymbol* element22 =
          symbol_cast<UsingDeclarationSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value19.push_back(std::move(element22));
    }
  }
  for (auto&& element23 : value19) {
    self->addUsingDeclaration(element23);
  }
}

void SemanticDecoder::readSymbolBaseClassSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BaseClassSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::BaseClassSymbol::symbol_
  cxx::Symbol* value14 = symbolAt(SymbolRef{in.varU32()});
  self->setSymbol(std::move(value14));
  // ::cxx::BaseClassSymbol::isVirtual_
  bool value15 = in.boolean();
  self->setVirtual(std::move(value15));
}

void SemanticDecoder::readSymbolInjectedClassNameSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::InjectedClassNameSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::InjectedClassNameSymbol::classSymbol_
  cxx::ClassSymbol* value14 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setClassSymbol(std::move(value14));
}

void SemanticDecoder::readSymbolUnresolvedSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UnresolvedSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
}

void SemanticDecoder::readSymbolUsingDeclarationSymbol(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UsingDeclarationSymbol* self) {
  // ::cxx::Symbol::name_
  const cxx::Name* value1 = nameAt(NameRef{in.varU32()});
  self->setName(std::move(value1));
  // ::cxx::Symbol::type_
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value2));
  // ::cxx::Symbol::parent_
  cxx::ScopeSymbol* value3 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setParent(std::move(value3));
  // ::cxx::Symbol::abiTags_
  const std::vector<const cxx::Identifier*>* value4 = readAbiTags(in);
  self->setAbiTags(std::move(value4));
  // ::cxx::Symbol::attributes_
  const std::vector<cxx::Attribute>* value5 = readAttributes(in);
  self->setAttributes(std::move(value5));
  // ::cxx::Symbol::location_
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->setLocation(std::move(value6));
  // ::cxx::Symbol::isHidden_
  bool value7 = in.boolean();
  self->setHidden(std::move(value7));
  // ::cxx::Symbol::isNodiscard_
  bool value8 = in.boolean();
  self->setNodiscard(std::move(value8));
  // ::cxx::Symbol::isUsed_
  bool value9 = in.boolean();
  self->setUsed(std::move(value9));
  // ::cxx::Symbol::isExcludedFromExplicitInstantiation_
  bool value10 = in.boolean();
  self->setExcludedFromExplicitInstantiation(std::move(value10));
  // ::cxx::Symbol::isTrivialAbi_
  bool value11 = in.boolean();
  self->setTrivialAbi(std::move(value11));
  // ::cxx::Symbol::hasDeducedReturnType_
  bool value12 = in.boolean();
  self->setDeducedReturnType(std::move(value12));
  // ::cxx::Symbol::accessSpecifier_
  static_assert(
      static_cast<std::uint32_t>(::cxx::AccessSpecifier::kPrivate) + 1 == 3);
  ::cxx::AccessSpecifier value13 =
      static_cast<::cxx::AccessSpecifier>(readEnum(in, 3));
  self->setAccessSpecifier(std::move(value13));
  // ::cxx::UsingDeclarationSymbol::target_
  cxx::Symbol* value14 = symbolAt(SymbolRef{in.varU32()});
  self->setTarget(std::move(value14));
  // ::cxx::UsingDeclarationSymbol::declarator_
  cxx::UsingDeclaratorAST* value15 =
      ast_cast<UsingDeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->setDeclarator(std::move(value15));
}

void SemanticDecoder::readAstTranslationUnitAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TranslationUnitAST* self) {
  // ::cxx::UnitAST::symbol
  cxx::NamespaceSymbol* value1 =
      symbol_cast<NamespaceSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value1);
  // ::cxx::TranslationUnitAST::declarationList
  cxx::List<cxx::DeclarationAST*>* value2 =
      readAstList<cxx::DeclarationAST>(in);
  self->declarationList = std::move(value2);
}

void SemanticDecoder::readAstModuleUnitAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ModuleUnitAST* self) {
  // ::cxx::UnitAST::symbol
  cxx::NamespaceSymbol* value1 =
      symbol_cast<NamespaceSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value1);
  // ::cxx::ModuleUnitAST::globalModuleFragment
  cxx::GlobalModuleFragmentAST* value2 =
      ast_cast<GlobalModuleFragmentAST>(astAt(AstRef{in.varU32()}));
  self->globalModuleFragment = std::move(value2);
  // ::cxx::ModuleUnitAST::moduleDeclaration
  cxx::ModuleDeclarationAST* value3 =
      ast_cast<ModuleDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->moduleDeclaration = std::move(value3);
  // ::cxx::ModuleUnitAST::declarationList
  cxx::List<cxx::DeclarationAST*>* value4 =
      readAstList<cxx::DeclarationAST>(in);
  self->declarationList = std::move(value4);
  // ::cxx::ModuleUnitAST::privateModuleFragment
  cxx::PrivateModuleFragmentAST* value5 =
      ast_cast<PrivateModuleFragmentAST>(astAt(AstRef{in.varU32()}));
  self->privateModuleFragment = std::move(value5);
}

void SemanticDecoder::readAstSimpleDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SimpleDeclarationAST* self) {
  // ::cxx::SimpleDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::SimpleDeclarationAST::declSpecifierList
  cxx::List<cxx::SpecifierAST*>* value2 = readAstList<cxx::SpecifierAST>(in);
  self->declSpecifierList = std::move(value2);
  // ::cxx::SimpleDeclarationAST::initDeclaratorList
  cxx::List<cxx::InitDeclaratorAST*>* value3 =
      readAstList<cxx::InitDeclaratorAST>(in);
  self->initDeclaratorList = std::move(value3);
  // ::cxx::SimpleDeclarationAST::requiresClause
  cxx::RequiresClauseAST* value4 =
      ast_cast<RequiresClauseAST>(astAt(AstRef{in.varU32()}));
  self->requiresClause = std::move(value4);
  // ::cxx::SimpleDeclarationAST::semicolonLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value5);
}

void SemanticDecoder::readAstAsmDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AsmDeclarationAST* self) {
  // ::cxx::AsmDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::AsmDeclarationAST::asmQualifierList
  cxx::List<cxx::AsmQualifierAST*>* value2 =
      readAstList<cxx::AsmQualifierAST>(in);
  self->asmQualifierList = std::move(value2);
  // ::cxx::AsmDeclarationAST::asmLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->asmLoc = std::move(value3);
  // ::cxx::AsmDeclarationAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::AsmDeclarationAST::literalLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value5);
  // ::cxx::AsmDeclarationAST::outputOperandList
  cxx::List<cxx::AsmOperandAST*>* value6 = readAstList<cxx::AsmOperandAST>(in);
  self->outputOperandList = std::move(value6);
  // ::cxx::AsmDeclarationAST::inputOperandList
  cxx::List<cxx::AsmOperandAST*>* value7 = readAstList<cxx::AsmOperandAST>(in);
  self->inputOperandList = std::move(value7);
  // ::cxx::AsmDeclarationAST::clobberList
  cxx::List<cxx::AsmClobberAST*>* value8 = readAstList<cxx::AsmClobberAST>(in);
  self->clobberList = std::move(value8);
  // ::cxx::AsmDeclarationAST::gotoLabelList
  cxx::List<cxx::AsmGotoLabelAST*>* value9 =
      readAstList<cxx::AsmGotoLabelAST>(in);
  self->gotoLabelList = std::move(value9);
  // ::cxx::AsmDeclarationAST::rparenLoc
  cxx::SourceLocation value10 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value10);
  // ::cxx::AsmDeclarationAST::semicolonLoc
  cxx::SourceLocation value11 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value11);
  // ::cxx::AsmDeclarationAST::literal
  const cxx::Literal* value12 = readStringLiteral(in);
  self->literal = std::move(value12);
}

void SemanticDecoder::readAstNamespaceAliasDefinitionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NamespaceAliasDefinitionAST* self) {
  // ::cxx::NamespaceAliasDefinitionAST::namespaceLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->namespaceLoc = std::move(value1);
  // ::cxx::NamespaceAliasDefinitionAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::NamespaceAliasDefinitionAST::equalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value3);
  // ::cxx::NamespaceAliasDefinitionAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value4 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value4);
  // ::cxx::NamespaceAliasDefinitionAST::unqualifiedId
  cxx::NameIdAST* value5 = ast_cast<NameIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value5);
  // ::cxx::NamespaceAliasDefinitionAST::semicolonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value6);
  // ::cxx::NamespaceAliasDefinitionAST::identifier
  const cxx::Identifier* value7 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value7);
  // ::cxx::NamespaceAliasDefinitionAST::symbol
  cxx::NamespaceAliasSymbol* value8 =
      symbol_cast<NamespaceAliasSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value8);
}

void SemanticDecoder::readAstUsingDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UsingDeclarationAST* self) {
  // ::cxx::UsingDeclarationAST::usingLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->usingLoc = std::move(value1);
  // ::cxx::UsingDeclarationAST::usingDeclaratorList
  cxx::List<cxx::UsingDeclaratorAST*>* value2 =
      readAstList<cxx::UsingDeclaratorAST>(in);
  self->usingDeclaratorList = std::move(value2);
  // ::cxx::UsingDeclarationAST::semicolonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value3);
}

void SemanticDecoder::readAstUsingEnumDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UsingEnumDeclarationAST* self) {
  // ::cxx::UsingEnumDeclarationAST::usingLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->usingLoc = std::move(value1);
  // ::cxx::UsingEnumDeclarationAST::enumTypeSpecifier
  cxx::ElaboratedTypeSpecifierAST* value2 =
      ast_cast<ElaboratedTypeSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->enumTypeSpecifier = std::move(value2);
  // ::cxx::UsingEnumDeclarationAST::semicolonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value3);
}

void SemanticDecoder::readAstUsingDirectiveAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UsingDirectiveAST* self) {
  // ::cxx::UsingDirectiveAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::UsingDirectiveAST::usingLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->usingLoc = std::move(value2);
  // ::cxx::UsingDirectiveAST::namespaceLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->namespaceLoc = std::move(value3);
  // ::cxx::UsingDirectiveAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value4 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value4);
  // ::cxx::UsingDirectiveAST::unqualifiedId
  cxx::NameIdAST* value5 = ast_cast<NameIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value5);
  // ::cxx::UsingDirectiveAST::semicolonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value6);
}

void SemanticDecoder::readAstStaticAssertDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::StaticAssertDeclarationAST* self) {
  // ::cxx::StaticAssertDeclarationAST::staticAssertLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->staticAssertLoc = std::move(value1);
  // ::cxx::StaticAssertDeclarationAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::StaticAssertDeclarationAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::StaticAssertDeclarationAST::commaLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->commaLoc = std::move(value4);
  // ::cxx::StaticAssertDeclarationAST::literalLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value5);
  // ::cxx::StaticAssertDeclarationAST::literal
  const cxx::Literal* value6 = readStringLiteral(in);
  self->literal = std::move(value6);
  // ::cxx::StaticAssertDeclarationAST::rparenLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value7);
  // ::cxx::StaticAssertDeclarationAST::semicolonLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value8);
  // ::cxx::StaticAssertDeclarationAST::value
  std::optional<bool> value9;
  if (in.boolean()) {
    bool value10 = in.boolean();
    value9 = std::move(value10);
  }
  self->value = std::move(value9);
}

void SemanticDecoder::readAstAliasDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AliasDeclarationAST* self) {
  // ::cxx::AliasDeclarationAST::usingLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->usingLoc = std::move(value1);
  // ::cxx::AliasDeclarationAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::AliasDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value3 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value3);
  // ::cxx::AliasDeclarationAST::equalLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value4);
  // ::cxx::AliasDeclarationAST::gnuAttributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value5 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->gnuAttributeList = std::move(value5);
  // ::cxx::AliasDeclarationAST::typeId
  cxx::TypeIdAST* value6 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value6);
  // ::cxx::AliasDeclarationAST::semicolonLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value7);
  // ::cxx::AliasDeclarationAST::identifier
  const cxx::Identifier* value8 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value8);
  // ::cxx::AliasDeclarationAST::symbol
  cxx::TypeAliasSymbol* value9 =
      symbol_cast<TypeAliasSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value9);
}

void SemanticDecoder::readAstOpaqueEnumDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::OpaqueEnumDeclarationAST* self) {
  // ::cxx::OpaqueEnumDeclarationAST::enumLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->enumLoc = std::move(value1);
  // ::cxx::OpaqueEnumDeclarationAST::classLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->classLoc = std::move(value2);
  // ::cxx::OpaqueEnumDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value3 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value3);
  // ::cxx::OpaqueEnumDeclarationAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value4 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value4);
  // ::cxx::OpaqueEnumDeclarationAST::unqualifiedId
  cxx::NameIdAST* value5 = ast_cast<NameIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value5);
  // ::cxx::OpaqueEnumDeclarationAST::colonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value6);
  // ::cxx::OpaqueEnumDeclarationAST::typeSpecifierList
  cxx::List<cxx::SpecifierAST*>* value7 = readAstList<cxx::SpecifierAST>(in);
  self->typeSpecifierList = std::move(value7);
  // ::cxx::OpaqueEnumDeclarationAST::emicolonLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->emicolonLoc = std::move(value8);
  // ::cxx::OpaqueEnumDeclarationAST::symbol
  cxx::Symbol* value9 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value9);
}

void SemanticDecoder::readAstFunctionDefinitionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::FunctionDefinitionAST* self) {
  // ::cxx::FunctionDefinitionAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::FunctionDefinitionAST::declSpecifierList
  cxx::List<cxx::SpecifierAST*>* value2 = readAstList<cxx::SpecifierAST>(in);
  self->declSpecifierList = std::move(value2);
  // ::cxx::FunctionDefinitionAST::declarator
  cxx::DeclaratorAST* value3 =
      ast_cast<DeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->declarator = std::move(value3);
  // ::cxx::FunctionDefinitionAST::requiresClause
  cxx::RequiresClauseAST* value4 =
      ast_cast<RequiresClauseAST>(astAt(AstRef{in.varU32()}));
  self->requiresClause = std::move(value4);
  // ::cxx::FunctionDefinitionAST::functionBody
  cxx::FunctionBodyAST* value5 =
      ast_cast<FunctionBodyAST>(astAt(AstRef{in.varU32()}));
  self->functionBody = std::move(value5);
  // ::cxx::FunctionDefinitionAST::symbol
  cxx::FunctionSymbol* value6 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value6);
}

void SemanticDecoder::readAstTemplateDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TemplateDeclarationAST* self) {
  // ::cxx::TemplateDeclarationAST::templateLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value1);
  // ::cxx::TemplateDeclarationAST::lessLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lessLoc = std::move(value2);
  // ::cxx::TemplateDeclarationAST::templateParameterList
  cxx::List<cxx::TemplateParameterAST*>* value3 =
      readAstList<cxx::TemplateParameterAST>(in);
  self->templateParameterList = std::move(value3);
  // ::cxx::TemplateDeclarationAST::greaterLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->greaterLoc = std::move(value4);
  // ::cxx::TemplateDeclarationAST::requiresClause
  cxx::RequiresClauseAST* value5 =
      ast_cast<RequiresClauseAST>(astAt(AstRef{in.varU32()}));
  self->requiresClause = std::move(value5);
  // ::cxx::TemplateDeclarationAST::declaration
  cxx::DeclarationAST* value6 =
      ast_cast<DeclarationAST>(astAt(AstRef{in.varU32()}));
  self->declaration = std::move(value6);
  // ::cxx::TemplateDeclarationAST::symbol
  cxx::TemplateParametersSymbol* value7 =
      symbol_cast<TemplateParametersSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value7);
  // ::cxx::TemplateDeclarationAST::depth
  int value8 = static_cast<int>(in.varI32());
  self->depth = std::move(value8);
}

void SemanticDecoder::readAstConceptDefinitionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConceptDefinitionAST* self) {
  // ::cxx::ConceptDefinitionAST::conceptLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->conceptLoc = std::move(value1);
  // ::cxx::ConceptDefinitionAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::ConceptDefinitionAST::equalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value3);
  // ::cxx::ConceptDefinitionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
  // ::cxx::ConceptDefinitionAST::semicolonLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value5);
  // ::cxx::ConceptDefinitionAST::identifier
  const cxx::Identifier* value6 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value6);
  // ::cxx::ConceptDefinitionAST::symbol
  cxx::ConceptSymbol* value7 =
      symbol_cast<ConceptSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value7);
}

void SemanticDecoder::readAstDeductionGuideAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DeductionGuideAST* self) {
  // ::cxx::DeductionGuideAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::DeductionGuideAST::explicitSpecifier
  cxx::SpecifierAST* value2 =
      ast_cast<SpecifierAST>(astAt(AstRef{in.varU32()}));
  self->explicitSpecifier = std::move(value2);
  // ::cxx::DeductionGuideAST::identifierLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value3);
  // ::cxx::DeductionGuideAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::DeductionGuideAST::parameterDeclarationClause
  cxx::ParameterDeclarationClauseAST* value5 =
      ast_cast<ParameterDeclarationClauseAST>(astAt(AstRef{in.varU32()}));
  self->parameterDeclarationClause = std::move(value5);
  // ::cxx::DeductionGuideAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::DeductionGuideAST::arrowLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->arrowLoc = std::move(value7);
  // ::cxx::DeductionGuideAST::templateId
  cxx::SimpleTemplateIdAST* value8 =
      ast_cast<SimpleTemplateIdAST>(astAt(AstRef{in.varU32()}));
  self->templateId = std::move(value8);
  // ::cxx::DeductionGuideAST::semicolonLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value9);
  // ::cxx::DeductionGuideAST::identifier
  const cxx::Identifier* value10 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value10);
  // ::cxx::DeductionGuideAST::symbol
  cxx::DeductionGuideSymbol* value11 =
      symbol_cast<DeductionGuideSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value11);
}

void SemanticDecoder::readAstExplicitInstantiationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ExplicitInstantiationAST* self) {
  // ::cxx::ExplicitInstantiationAST::externLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->externLoc = std::move(value1);
  // ::cxx::ExplicitInstantiationAST::templateLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value2);
  // ::cxx::ExplicitInstantiationAST::declaration
  cxx::DeclarationAST* value3 =
      ast_cast<DeclarationAST>(astAt(AstRef{in.varU32()}));
  self->declaration = std::move(value3);
}

void SemanticDecoder::readAstExportDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ExportDeclarationAST* self) {
  // ::cxx::ExportDeclarationAST::exportLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->exportLoc = std::move(value1);
  // ::cxx::ExportDeclarationAST::declaration
  cxx::DeclarationAST* value2 =
      ast_cast<DeclarationAST>(astAt(AstRef{in.varU32()}));
  self->declaration = std::move(value2);
}

void SemanticDecoder::readAstExportCompoundDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ExportCompoundDeclarationAST* self) {
  // ::cxx::ExportCompoundDeclarationAST::exportLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->exportLoc = std::move(value1);
  // ::cxx::ExportCompoundDeclarationAST::lbraceLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lbraceLoc = std::move(value2);
  // ::cxx::ExportCompoundDeclarationAST::declarationList
  cxx::List<cxx::DeclarationAST*>* value3 =
      readAstList<cxx::DeclarationAST>(in);
  self->declarationList = std::move(value3);
  // ::cxx::ExportCompoundDeclarationAST::rbraceLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rbraceLoc = std::move(value4);
}

void SemanticDecoder::readAstLinkageSpecificationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::LinkageSpecificationAST* self) {
  // ::cxx::LinkageSpecificationAST::externLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->externLoc = std::move(value1);
  // ::cxx::LinkageSpecificationAST::stringliteralLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->stringliteralLoc = std::move(value2);
  // ::cxx::LinkageSpecificationAST::lbraceLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lbraceLoc = std::move(value3);
  // ::cxx::LinkageSpecificationAST::declarationList
  cxx::List<cxx::DeclarationAST*>* value4 =
      readAstList<cxx::DeclarationAST>(in);
  self->declarationList = std::move(value4);
  // ::cxx::LinkageSpecificationAST::rbraceLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->rbraceLoc = std::move(value5);
  // ::cxx::LinkageSpecificationAST::stringLiteral
  const cxx::StringLiteral* value6 = readStringLiteral(in);
  self->stringLiteral = std::move(value6);
}

void SemanticDecoder::readAstNamespaceDefinitionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NamespaceDefinitionAST* self) {
  // ::cxx::NamespaceDefinitionAST::inlineLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->inlineLoc = std::move(value1);
  // ::cxx::NamespaceDefinitionAST::namespaceLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->namespaceLoc = std::move(value2);
  // ::cxx::NamespaceDefinitionAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value3 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value3);
  // ::cxx::NamespaceDefinitionAST::nestedNamespaceSpecifierList
  cxx::List<cxx::NestedNamespaceSpecifierAST*>* value4 =
      readAstList<cxx::NestedNamespaceSpecifierAST>(in);
  self->nestedNamespaceSpecifierList = std::move(value4);
  // ::cxx::NamespaceDefinitionAST::identifierLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value5);
  // ::cxx::NamespaceDefinitionAST::extraAttributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value6 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->extraAttributeList = std::move(value6);
  // ::cxx::NamespaceDefinitionAST::lbraceLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->lbraceLoc = std::move(value7);
  // ::cxx::NamespaceDefinitionAST::declarationList
  cxx::List<cxx::DeclarationAST*>* value8 =
      readAstList<cxx::DeclarationAST>(in);
  self->declarationList = std::move(value8);
  // ::cxx::NamespaceDefinitionAST::rbraceLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->rbraceLoc = std::move(value9);
  // ::cxx::NamespaceDefinitionAST::identifier
  const cxx::Identifier* value10 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value10);
  // ::cxx::NamespaceDefinitionAST::symbol
  cxx::NamespaceSymbol* value11 =
      symbol_cast<NamespaceSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value11);
  // ::cxx::NamespaceDefinitionAST::isInline
  bool value12 = in.boolean();
  self->isInline = std::move(value12);
}

void SemanticDecoder::readAstEmptyDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::EmptyDeclarationAST* self) {
  // ::cxx::EmptyDeclarationAST::semicolonLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value1);
}

void SemanticDecoder::readAstAttributeDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AttributeDeclarationAST* self) {
  // ::cxx::AttributeDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::AttributeDeclarationAST::semicolonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value2);
}

void SemanticDecoder::readAstModuleImportDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ModuleImportDeclarationAST* self) {
  // ::cxx::ModuleImportDeclarationAST::importLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->importLoc = std::move(value1);
  // ::cxx::ModuleImportDeclarationAST::importName
  cxx::ImportNameAST* value2 =
      ast_cast<ImportNameAST>(astAt(AstRef{in.varU32()}));
  self->importName = std::move(value2);
  // ::cxx::ModuleImportDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value3 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value3);
  // ::cxx::ModuleImportDeclarationAST::semicolonLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value4);
}

void SemanticDecoder::readAstParameterDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ParameterDeclarationAST* self) {
  // ::cxx::ParameterDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::ParameterDeclarationAST::thisLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->thisLoc = std::move(value2);
  // ::cxx::ParameterDeclarationAST::typeSpecifierList
  cxx::List<cxx::SpecifierAST*>* value3 = readAstList<cxx::SpecifierAST>(in);
  self->typeSpecifierList = std::move(value3);
  // ::cxx::ParameterDeclarationAST::declarator
  cxx::DeclaratorAST* value4 =
      ast_cast<DeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->declarator = std::move(value4);
  // ::cxx::ParameterDeclarationAST::equalLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value5);
  // ::cxx::ParameterDeclarationAST::expression
  cxx::ExpressionAST* value6 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value6);
  // ::cxx::ParameterDeclarationAST::type
  const cxx::Type* value7 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value7);
  // ::cxx::ParameterDeclarationAST::identifier
  const cxx::Identifier* value8 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value8);
  // ::cxx::ParameterDeclarationAST::symbol
  cxx::ParameterSymbol* value9 =
      symbol_cast<ParameterSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value9);
  // ::cxx::ParameterDeclarationAST::isThisIntroduced
  bool value10 = in.boolean();
  self->isThisIntroduced = std::move(value10);
  // ::cxx::ParameterDeclarationAST::isPack
  bool value11 = in.boolean();
  self->isPack = std::move(value11);
}

void SemanticDecoder::readAstAccessDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AccessDeclarationAST* self) {
  // ::cxx::AccessDeclarationAST::accessLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->accessLoc = std::move(value1);
  // ::cxx::AccessDeclarationAST::colonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value2);
  // ::cxx::AccessDeclarationAST::accessSpecifier
  ::cxx::TokenKind value3 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->accessSpecifier = std::move(value3);
}

void SemanticDecoder::readAstForRangeDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ForRangeDeclarationAST* self) {}

void SemanticDecoder::readAstStructuredBindingDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::StructuredBindingDeclarationAST* self) {
  // ::cxx::StructuredBindingDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::StructuredBindingDeclarationAST::declSpecifierList
  cxx::List<cxx::SpecifierAST*>* value2 = readAstList<cxx::SpecifierAST>(in);
  self->declSpecifierList = std::move(value2);
  // ::cxx::StructuredBindingDeclarationAST::refQualifierLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->refQualifierLoc = std::move(value3);
  // ::cxx::StructuredBindingDeclarationAST::lbracketLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value4);
  // ::cxx::StructuredBindingDeclarationAST::bindingList
  cxx::List<cxx::NameIdAST*>* value5 = readAstList<cxx::NameIdAST>(in);
  self->bindingList = std::move(value5);
  // ::cxx::StructuredBindingDeclarationAST::rbracketLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value6);
  // ::cxx::StructuredBindingDeclarationAST::initializer
  cxx::ExpressionAST* value7 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value7);
  // ::cxx::StructuredBindingDeclarationAST::semicolonLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value8);
  // ::cxx::StructuredBindingDeclarationAST::hiddenVariable
  cxx::InitDeclaratorAST* value9 =
      ast_cast<InitDeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->hiddenVariable = std::move(value9);
  // ::cxx::StructuredBindingDeclarationAST::bindingDeclaratorList
  cxx::List<cxx::InitDeclaratorAST*>* value10 =
      readAstList<cxx::InitDeclaratorAST>(in);
  self->bindingDeclaratorList = std::move(value10);
}

void SemanticDecoder::readAstAsmOperandAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AsmOperandAST* self) {
  // ::cxx::AsmOperandAST::lbracketLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value1);
  // ::cxx::AsmOperandAST::symbolicNameLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->symbolicNameLoc = std::move(value2);
  // ::cxx::AsmOperandAST::rbracketLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value3);
  // ::cxx::AsmOperandAST::constraintLiteralLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->constraintLiteralLoc = std::move(value4);
  // ::cxx::AsmOperandAST::lparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value5);
  // ::cxx::AsmOperandAST::expression
  cxx::ExpressionAST* value6 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value6);
  // ::cxx::AsmOperandAST::rparenLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value7);
  // ::cxx::AsmOperandAST::symbolicName
  const cxx::Identifier* value8 = identifierAt(StringRef{in.varU32()});
  self->symbolicName = std::move(value8);
  // ::cxx::AsmOperandAST::constraintLiteral
  const cxx::Literal* value9 = readStringLiteral(in);
  self->constraintLiteral = std::move(value9);
}

void SemanticDecoder::readAstAsmQualifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AsmQualifierAST* self) {
  // ::cxx::AsmQualifierAST::qualifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->qualifierLoc = std::move(value1);
  // ::cxx::AsmQualifierAST::qualifier
  ::cxx::TokenKind value2 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->qualifier = std::move(value2);
}

void SemanticDecoder::readAstAsmClobberAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AsmClobberAST* self) {
  // ::cxx::AsmClobberAST::literalLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value1);
  // ::cxx::AsmClobberAST::literal
  const cxx::StringLiteral* value2 = readStringLiteral(in);
  self->literal = std::move(value2);
}

void SemanticDecoder::readAstAsmGotoLabelAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AsmGotoLabelAST* self) {
  // ::cxx::AsmGotoLabelAST::identifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value1);
  // ::cxx::AsmGotoLabelAST::identifier
  const cxx::Identifier* value2 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value2);
}

void SemanticDecoder::readAstSplicerAST(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::SplicerAST* self) {
  // ::cxx::SplicerAST::lbracketLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value1);
  // ::cxx::SplicerAST::colonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value2);
  // ::cxx::SplicerAST::ellipsisLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value3);
  // ::cxx::SplicerAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
  // ::cxx::SplicerAST::secondColonLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->secondColonLoc = std::move(value5);
  // ::cxx::SplicerAST::rbracketLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value6);
}

void SemanticDecoder::readAstGlobalModuleFragmentAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::GlobalModuleFragmentAST* self) {
  // ::cxx::GlobalModuleFragmentAST::moduleLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->moduleLoc = std::move(value1);
  // ::cxx::GlobalModuleFragmentAST::semicolonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value2);
  // ::cxx::GlobalModuleFragmentAST::declarationList
  cxx::List<cxx::DeclarationAST*>* value3 =
      readAstList<cxx::DeclarationAST>(in);
  self->declarationList = std::move(value3);
}

void SemanticDecoder::readAstPrivateModuleFragmentAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PrivateModuleFragmentAST* self) {
  // ::cxx::PrivateModuleFragmentAST::moduleLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->moduleLoc = std::move(value1);
  // ::cxx::PrivateModuleFragmentAST::colonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value2);
  // ::cxx::PrivateModuleFragmentAST::privateLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->privateLoc = std::move(value3);
  // ::cxx::PrivateModuleFragmentAST::semicolonLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value4);
  // ::cxx::PrivateModuleFragmentAST::declarationList
  cxx::List<cxx::DeclarationAST*>* value5 =
      readAstList<cxx::DeclarationAST>(in);
  self->declarationList = std::move(value5);
}

void SemanticDecoder::readAstModuleDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ModuleDeclarationAST* self) {
  // ::cxx::ModuleDeclarationAST::exportLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->exportLoc = std::move(value1);
  // ::cxx::ModuleDeclarationAST::moduleLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->moduleLoc = std::move(value2);
  // ::cxx::ModuleDeclarationAST::moduleName
  cxx::ModuleNameAST* value3 =
      ast_cast<ModuleNameAST>(astAt(AstRef{in.varU32()}));
  self->moduleName = std::move(value3);
  // ::cxx::ModuleDeclarationAST::modulePartition
  cxx::ModulePartitionAST* value4 =
      ast_cast<ModulePartitionAST>(astAt(AstRef{in.varU32()}));
  self->modulePartition = std::move(value4);
  // ::cxx::ModuleDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value5 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value5);
  // ::cxx::ModuleDeclarationAST::semicolonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value6);
}

void SemanticDecoder::readAstModuleNameAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ModuleNameAST* self) {
  // ::cxx::ModuleNameAST::moduleQualifier
  cxx::ModuleQualifierAST* value1 =
      ast_cast<ModuleQualifierAST>(astAt(AstRef{in.varU32()}));
  self->moduleQualifier = std::move(value1);
  // ::cxx::ModuleNameAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::ModuleNameAST::identifier
  const cxx::Identifier* value3 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value3);
}

void SemanticDecoder::readAstModuleQualifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ModuleQualifierAST* self) {
  // ::cxx::ModuleQualifierAST::moduleQualifier
  cxx::ModuleQualifierAST* value1 =
      ast_cast<ModuleQualifierAST>(astAt(AstRef{in.varU32()}));
  self->moduleQualifier = std::move(value1);
  // ::cxx::ModuleQualifierAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::ModuleQualifierAST::dotLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->dotLoc = std::move(value3);
  // ::cxx::ModuleQualifierAST::identifier
  const cxx::Identifier* value4 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value4);
}

void SemanticDecoder::readAstModulePartitionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ModulePartitionAST* self) {
  // ::cxx::ModulePartitionAST::colonLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value1);
  // ::cxx::ModulePartitionAST::moduleName
  cxx::ModuleNameAST* value2 =
      ast_cast<ModuleNameAST>(astAt(AstRef{in.varU32()}));
  self->moduleName = std::move(value2);
}

void SemanticDecoder::readAstImportNameAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ImportNameAST* self) {
  // ::cxx::ImportNameAST::headerLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->headerLoc = std::move(value1);
  // ::cxx::ImportNameAST::modulePartition
  cxx::ModulePartitionAST* value2 =
      ast_cast<ModulePartitionAST>(astAt(AstRef{in.varU32()}));
  self->modulePartition = std::move(value2);
  // ::cxx::ImportNameAST::moduleName
  cxx::ModuleNameAST* value3 =
      ast_cast<ModuleNameAST>(astAt(AstRef{in.varU32()}));
  self->moduleName = std::move(value3);
}

void SemanticDecoder::readAstInitDeclaratorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::InitDeclaratorAST* self) {
  // ::cxx::InitDeclaratorAST::declarator
  cxx::DeclaratorAST* value1 =
      ast_cast<DeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->declarator = std::move(value1);
  // ::cxx::InitDeclaratorAST::requiresClause
  cxx::RequiresClauseAST* value2 =
      ast_cast<RequiresClauseAST>(astAt(AstRef{in.varU32()}));
  self->requiresClause = std::move(value2);
  // ::cxx::InitDeclaratorAST::initializer
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value3);
  // ::cxx::InitDeclaratorAST::symbol
  cxx::Symbol* value4 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value4);
}

void SemanticDecoder::readAstDeclaratorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DeclaratorAST* self) {
  // ::cxx::DeclaratorAST::ptrOpList
  cxx::List<cxx::PtrOperatorAST*>* value1 =
      readAstList<cxx::PtrOperatorAST>(in);
  self->ptrOpList = std::move(value1);
  // ::cxx::DeclaratorAST::coreDeclarator
  cxx::CoreDeclaratorAST* value2 =
      ast_cast<CoreDeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->coreDeclarator = std::move(value2);
  // ::cxx::DeclaratorAST::declaratorChunkList
  cxx::List<cxx::DeclaratorChunkAST*>* value3 =
      readAstList<cxx::DeclaratorChunkAST>(in);
  self->declaratorChunkList = std::move(value3);
}

void SemanticDecoder::readAstUsingDeclaratorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UsingDeclaratorAST* self) {
  // ::cxx::UsingDeclaratorAST::typenameLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->typenameLoc = std::move(value1);
  // ::cxx::UsingDeclaratorAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value2 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value2);
  // ::cxx::UsingDeclaratorAST::unqualifiedId
  cxx::UnqualifiedIdAST* value3 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value3);
  // ::cxx::UsingDeclaratorAST::ellipsisLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value4);
  // ::cxx::UsingDeclaratorAST::symbol
  cxx::UsingDeclarationSymbol* value5 =
      symbol_cast<UsingDeclarationSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value5);
  // ::cxx::UsingDeclaratorAST::isPack
  bool value6 = in.boolean();
  self->isPack = std::move(value6);
}

void SemanticDecoder::readAstEnumeratorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::EnumeratorAST* self) {
  // ::cxx::EnumeratorAST::identifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value1);
  // ::cxx::EnumeratorAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value2 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value2);
  // ::cxx::EnumeratorAST::equalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value3);
  // ::cxx::EnumeratorAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
  // ::cxx::EnumeratorAST::identifier
  const cxx::Identifier* value5 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value5);
  // ::cxx::EnumeratorAST::symbol
  cxx::EnumeratorSymbol* value6 =
      symbol_cast<EnumeratorSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value6);
}

void SemanticDecoder::readAstTypeIdAST([[maybe_unused]] ByteReader& in,
                                       [[maybe_unused]] cxx::TypeIdAST* self) {
  // ::cxx::TypeIdAST::typeSpecifierList
  cxx::List<cxx::SpecifierAST*>* value1 = readAstList<cxx::SpecifierAST>(in);
  self->typeSpecifierList = std::move(value1);
  // ::cxx::TypeIdAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value2 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value2);
  // ::cxx::TypeIdAST::declarator
  cxx::DeclaratorAST* value3 =
      ast_cast<DeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->declarator = std::move(value3);
  // ::cxx::TypeIdAST::type
  const cxx::Type* value4 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value4);
}

void SemanticDecoder::readAstHandlerAST(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::HandlerAST* self) {
  // ::cxx::HandlerAST::catchLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->catchLoc = std::move(value1);
  // ::cxx::HandlerAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::HandlerAST::exceptionDeclaration
  cxx::ExceptionDeclarationAST* value3 =
      ast_cast<ExceptionDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->exceptionDeclaration = std::move(value3);
  // ::cxx::HandlerAST::rparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value4);
  // ::cxx::HandlerAST::statement
  cxx::CompoundStatementAST* value5 =
      ast_cast<CompoundStatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value5);
  // ::cxx::HandlerAST::symbol
  cxx::BlockSymbol* value6 =
      symbol_cast<BlockSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value6);
}

void SemanticDecoder::readAstBaseSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BaseSpecifierAST* self) {
  // ::cxx::BaseSpecifierAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::BaseSpecifierAST::virtualOrAccessLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->virtualOrAccessLoc = std::move(value2);
  // ::cxx::BaseSpecifierAST::otherVirtualOrAccessLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->otherVirtualOrAccessLoc = std::move(value3);
  // ::cxx::BaseSpecifierAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value4 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value4);
  // ::cxx::BaseSpecifierAST::templateLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value5);
  // ::cxx::BaseSpecifierAST::unqualifiedId
  cxx::UnqualifiedIdAST* value6 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value6);
  // ::cxx::BaseSpecifierAST::ellipsisLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value7);
  // ::cxx::BaseSpecifierAST::isTemplateIntroduced
  bool value8 = in.boolean();
  self->isTemplateIntroduced = std::move(value8);
  // ::cxx::BaseSpecifierAST::isVirtual
  bool value9 = in.boolean();
  self->isVirtual = std::move(value9);
  // ::cxx::BaseSpecifierAST::isVariadic
  bool value10 = in.boolean();
  self->isVariadic = std::move(value10);
  // ::cxx::BaseSpecifierAST::accessSpecifier
  ::cxx::TokenKind value11 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->accessSpecifier = std::move(value11);
  // ::cxx::BaseSpecifierAST::symbol
  cxx::BaseClassSymbol* value12 =
      symbol_cast<BaseClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value12);
}

void SemanticDecoder::readAstRequiresClauseAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::RequiresClauseAST* self) {
  // ::cxx::RequiresClauseAST::requiresLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->requiresLoc = std::move(value1);
  // ::cxx::RequiresClauseAST::expression
  cxx::ExpressionAST* value2 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value2);
}

void SemanticDecoder::readAstParameterDeclarationClauseAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ParameterDeclarationClauseAST* self) {
  // ::cxx::ParameterDeclarationClauseAST::parameterDeclarationList
  cxx::List<cxx::ParameterDeclarationAST*>* value1 =
      readAstList<cxx::ParameterDeclarationAST>(in);
  self->parameterDeclarationList = std::move(value1);
  // ::cxx::ParameterDeclarationClauseAST::commaLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->commaLoc = std::move(value2);
  // ::cxx::ParameterDeclarationClauseAST::ellipsisLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value3);
  // ::cxx::ParameterDeclarationClauseAST::functionParametersSymbol
  cxx::FunctionParametersSymbol* value4 =
      symbol_cast<FunctionParametersSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->functionParametersSymbol = std::move(value4);
  // ::cxx::ParameterDeclarationClauseAST::isVariadic
  bool value5 = in.boolean();
  self->isVariadic = std::move(value5);
}

void SemanticDecoder::readAstTrailingReturnTypeAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TrailingReturnTypeAST* self) {
  // ::cxx::TrailingReturnTypeAST::minusGreaterLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->minusGreaterLoc = std::move(value1);
  // ::cxx::TrailingReturnTypeAST::typeId
  cxx::TypeIdAST* value2 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value2);
}

void SemanticDecoder::readAstLambdaSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::LambdaSpecifierAST* self) {
  // ::cxx::LambdaSpecifierAST::specifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->specifierLoc = std::move(value1);
  // ::cxx::LambdaSpecifierAST::specifier
  ::cxx::TokenKind value2 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->specifier = std::move(value2);
}

void SemanticDecoder::readAstTypeConstraintAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeConstraintAST* self) {
  // ::cxx::TypeConstraintAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value1 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value1);
  // ::cxx::TypeConstraintAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::TypeConstraintAST::lessLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lessLoc = std::move(value3);
  // ::cxx::TypeConstraintAST::templateArgumentList
  cxx::List<cxx::TemplateArgumentAST*>* value4 =
      readAstList<cxx::TemplateArgumentAST>(in);
  self->templateArgumentList = std::move(value4);
  // ::cxx::TypeConstraintAST::greaterLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->greaterLoc = std::move(value5);
  // ::cxx::TypeConstraintAST::identifier
  const cxx::Identifier* value6 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value6);
  // ::cxx::TypeConstraintAST::symbol
  cxx::ConceptSymbol* value7 =
      symbol_cast<ConceptSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value7);
}

void SemanticDecoder::readAstAttributeArgumentClauseAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AttributeArgumentClauseAST* self) {
  // ::cxx::AttributeArgumentClauseAST::lparenLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value1);
  // ::cxx::AttributeArgumentClauseAST::expressionList
  cxx::List<cxx::ExpressionAST*>* value2 = readAstList<cxx::ExpressionAST>(in);
  self->expressionList = std::move(value2);
  // ::cxx::AttributeArgumentClauseAST::rparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value3);
}

void SemanticDecoder::readAstAttributeAST(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::AttributeAST* self) {
  // ::cxx::AttributeAST::attributeToken
  cxx::AttributeTokenAST* value1 =
      ast_cast<AttributeTokenAST>(astAt(AstRef{in.varU32()}));
  self->attributeToken = std::move(value1);
  // ::cxx::AttributeAST::attributeArgumentClause
  cxx::AttributeArgumentClauseAST* value2 =
      ast_cast<AttributeArgumentClauseAST>(astAt(AstRef{in.varU32()}));
  self->attributeArgumentClause = std::move(value2);
  // ::cxx::AttributeAST::ellipsisLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value3);
}

void SemanticDecoder::readAstAttributeUsingPrefixAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AttributeUsingPrefixAST* self) {
  // ::cxx::AttributeUsingPrefixAST::usingLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->usingLoc = std::move(value1);
  // ::cxx::AttributeUsingPrefixAST::attributeNamespaceLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->attributeNamespaceLoc = std::move(value2);
  // ::cxx::AttributeUsingPrefixAST::colonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value3);
}

void SemanticDecoder::readAstNewPlacementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NewPlacementAST* self) {
  // ::cxx::NewPlacementAST::lparenLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value1);
  // ::cxx::NewPlacementAST::expressionList
  cxx::List<cxx::ExpressionAST*>* value2 = readAstList<cxx::ExpressionAST>(in);
  self->expressionList = std::move(value2);
  // ::cxx::NewPlacementAST::rparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value3);
}

void SemanticDecoder::readAstNestedNamespaceSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NestedNamespaceSpecifierAST* self) {
  // ::cxx::NestedNamespaceSpecifierAST::inlineLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->inlineLoc = std::move(value1);
  // ::cxx::NestedNamespaceSpecifierAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::NestedNamespaceSpecifierAST::scopeLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->scopeLoc = std::move(value3);
  // ::cxx::NestedNamespaceSpecifierAST::identifier
  const cxx::Identifier* value4 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value4);
  // ::cxx::NestedNamespaceSpecifierAST::symbol
  cxx::NamespaceSymbol* value5 =
      symbol_cast<NamespaceSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value5);
  // ::cxx::NestedNamespaceSpecifierAST::isInline
  bool value6 = in.boolean();
  self->isInline = std::move(value6);
}

void SemanticDecoder::readAstLabeledStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::LabeledStatementAST* self) {
  // ::cxx::LabeledStatementAST::identifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value1);
  // ::cxx::LabeledStatementAST::colonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value2);
  // ::cxx::LabeledStatementAST::statement
  cxx::StatementAST* value3 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value3);
  // ::cxx::LabeledStatementAST::identifier
  const cxx::Identifier* value4 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value4);
}

void SemanticDecoder::readAstCaseStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CaseStatementAST* self) {
  // ::cxx::CaseStatementAST::caseLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->caseLoc = std::move(value1);
  // ::cxx::CaseStatementAST::expression
  cxx::ExpressionAST* value2 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value2);
  // ::cxx::CaseStatementAST::colonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value3);
  // ::cxx::CaseStatementAST::caseValue
  long long value4 = static_cast<long long>(in.varI64());
  self->caseValue = std::move(value4);
}

void SemanticDecoder::readAstDefaultStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DefaultStatementAST* self) {
  // ::cxx::DefaultStatementAST::defaultLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->defaultLoc = std::move(value1);
  // ::cxx::DefaultStatementAST::colonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value2);
}

void SemanticDecoder::readAstExpressionStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ExpressionStatementAST* self) {
  // ::cxx::ExpressionStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::ExpressionStatementAST::expression
  cxx::ExpressionAST* value2 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value2);
  // ::cxx::ExpressionStatementAST::semicolonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value3);
}

void SemanticDecoder::readAstCompoundStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CompoundStatementAST* self) {
  // ::cxx::CompoundStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::CompoundStatementAST::lbraceLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lbraceLoc = std::move(value2);
  // ::cxx::CompoundStatementAST::statementList
  cxx::List<cxx::StatementAST*>* value3 = readAstList<cxx::StatementAST>(in);
  self->statementList = std::move(value3);
  // ::cxx::CompoundStatementAST::rbraceLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rbraceLoc = std::move(value4);
  // ::cxx::CompoundStatementAST::symbol
  cxx::BlockSymbol* value5 =
      symbol_cast<BlockSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value5);
}

void SemanticDecoder::readAstIfStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::IfStatementAST* self) {
  // ::cxx::IfStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::IfStatementAST::ifLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->ifLoc = std::move(value2);
  // ::cxx::IfStatementAST::constexprLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->constexprLoc = std::move(value3);
  // ::cxx::IfStatementAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::IfStatementAST::initializer
  cxx::StatementAST* value5 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value5);
  // ::cxx::IfStatementAST::condition
  cxx::ExpressionAST* value6 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->condition = std::move(value6);
  // ::cxx::IfStatementAST::rparenLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value7);
  // ::cxx::IfStatementAST::statement
  cxx::StatementAST* value8 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value8);
  // ::cxx::IfStatementAST::elseLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->elseLoc = std::move(value9);
  // ::cxx::IfStatementAST::elseStatement
  cxx::StatementAST* value10 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->elseStatement = std::move(value10);
  // ::cxx::IfStatementAST::symbol
  cxx::BlockSymbol* value11 =
      symbol_cast<BlockSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value11);
}

void SemanticDecoder::readAstConstevalIfStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstevalIfStatementAST* self) {
  // ::cxx::ConstevalIfStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::ConstevalIfStatementAST::ifLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->ifLoc = std::move(value2);
  // ::cxx::ConstevalIfStatementAST::exclaimLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->exclaimLoc = std::move(value3);
  // ::cxx::ConstevalIfStatementAST::constvalLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->constvalLoc = std::move(value4);
  // ::cxx::ConstevalIfStatementAST::statement
  cxx::StatementAST* value5 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value5);
  // ::cxx::ConstevalIfStatementAST::elseLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->elseLoc = std::move(value6);
  // ::cxx::ConstevalIfStatementAST::elseStatement
  cxx::StatementAST* value7 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->elseStatement = std::move(value7);
  // ::cxx::ConstevalIfStatementAST::isNot
  bool value8 = in.boolean();
  self->isNot = std::move(value8);
}

void SemanticDecoder::readAstSwitchStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SwitchStatementAST* self) {
  // ::cxx::SwitchStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::SwitchStatementAST::switchLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->switchLoc = std::move(value2);
  // ::cxx::SwitchStatementAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::SwitchStatementAST::initializer
  cxx::StatementAST* value4 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value4);
  // ::cxx::SwitchStatementAST::condition
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->condition = std::move(value5);
  // ::cxx::SwitchStatementAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::SwitchStatementAST::statement
  cxx::StatementAST* value7 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value7);
  // ::cxx::SwitchStatementAST::symbol
  cxx::BlockSymbol* value8 =
      symbol_cast<BlockSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value8);
}

void SemanticDecoder::readAstWhileStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::WhileStatementAST* self) {
  // ::cxx::WhileStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::WhileStatementAST::whileLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->whileLoc = std::move(value2);
  // ::cxx::WhileStatementAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::WhileStatementAST::condition
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->condition = std::move(value4);
  // ::cxx::WhileStatementAST::rparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value5);
  // ::cxx::WhileStatementAST::statement
  cxx::StatementAST* value6 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value6);
  // ::cxx::WhileStatementAST::symbol
  cxx::BlockSymbol* value7 =
      symbol_cast<BlockSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value7);
}

void SemanticDecoder::readAstDoStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DoStatementAST* self) {
  // ::cxx::DoStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::DoStatementAST::doLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->doLoc = std::move(value2);
  // ::cxx::DoStatementAST::statement
  cxx::StatementAST* value3 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value3);
  // ::cxx::DoStatementAST::whileLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->whileLoc = std::move(value4);
  // ::cxx::DoStatementAST::lparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value5);
  // ::cxx::DoStatementAST::expression
  cxx::ExpressionAST* value6 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value6);
  // ::cxx::DoStatementAST::rparenLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value7);
  // ::cxx::DoStatementAST::semicolonLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value8);
}

void SemanticDecoder::readAstForRangeStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ForRangeStatementAST* self) {
  // ::cxx::ForRangeStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::ForRangeStatementAST::forLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->forLoc = std::move(value2);
  // ::cxx::ForRangeStatementAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::ForRangeStatementAST::initializer
  cxx::StatementAST* value4 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value4);
  // ::cxx::ForRangeStatementAST::rangeDeclaration
  cxx::DeclarationAST* value5 =
      ast_cast<DeclarationAST>(astAt(AstRef{in.varU32()}));
  self->rangeDeclaration = std::move(value5);
  // ::cxx::ForRangeStatementAST::colonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value6);
  // ::cxx::ForRangeStatementAST::rangeInitializer
  cxx::ExpressionAST* value7 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->rangeInitializer = std::move(value7);
  // ::cxx::ForRangeStatementAST::rparenLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value8);
  // ::cxx::ForRangeStatementAST::statement
  cxx::StatementAST* value9 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value9);
  // ::cxx::ForRangeStatementAST::beginInitializer
  cxx::ExpressionAST* value10 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->beginInitializer = std::move(value10);
  // ::cxx::ForRangeStatementAST::endInitializer
  cxx::ExpressionAST* value11 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->endInitializer = std::move(value11);
  // ::cxx::ForRangeStatementAST::condition
  cxx::ExpressionAST* value12 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->condition = std::move(value12);
  // ::cxx::ForRangeStatementAST::increment
  cxx::ExpressionAST* value13 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->increment = std::move(value13);
  // ::cxx::ForRangeStatementAST::element
  cxx::ExpressionAST* value14 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->element = std::move(value14);
  // ::cxx::ForRangeStatementAST::symbol
  cxx::BlockSymbol* value15 =
      symbol_cast<BlockSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value15);
  // ::cxx::ForRangeStatementAST::rangeVariable
  cxx::VariableSymbol* value16 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->rangeVariable = std::move(value16);
  // ::cxx::ForRangeStatementAST::beginVariable
  cxx::VariableSymbol* value17 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->beginVariable = std::move(value17);
  // ::cxx::ForRangeStatementAST::endVariable
  cxx::VariableSymbol* value18 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->endVariable = std::move(value18);
  // ::cxx::ForRangeStatementAST::beginFunction
  cxx::FunctionSymbol* value19 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->beginFunction = std::move(value19);
  // ::cxx::ForRangeStatementAST::endFunction
  cxx::FunctionSymbol* value20 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->endFunction = std::move(value20);
  // ::cxx::ForRangeStatementAST::derefFunction
  cxx::FunctionSymbol* value21 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->derefFunction = std::move(value21);
  // ::cxx::ForRangeStatementAST::incrementFunction
  cxx::FunctionSymbol* value22 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->incrementFunction = std::move(value22);
  // ::cxx::ForRangeStatementAST::notEqualFunction
  cxx::FunctionSymbol* value23 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->notEqualFunction = std::move(value23);
  // ::cxx::ForRangeStatementAST::usesMemberBeginEnd
  bool value24 = in.boolean();
  self->usesMemberBeginEnd = std::move(value24);
  // ::cxx::ForRangeStatementAST::isPointerIterator
  bool value25 = in.boolean();
  self->isPointerIterator = std::move(value25);
  // ::cxx::ForRangeStatementAST::notEqualRewritten
  bool value26 = in.boolean();
  self->notEqualRewritten = std::move(value26);
  // ::cxx::ForRangeStatementAST::notEqualReversed
  bool value27 = in.boolean();
  self->notEqualReversed = std::move(value27);
}

void SemanticDecoder::readAstForStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ForStatementAST* self) {
  // ::cxx::ForStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::ForStatementAST::forLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->forLoc = std::move(value2);
  // ::cxx::ForStatementAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::ForStatementAST::initializer
  cxx::StatementAST* value4 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value4);
  // ::cxx::ForStatementAST::condition
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->condition = std::move(value5);
  // ::cxx::ForStatementAST::semicolonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value6);
  // ::cxx::ForStatementAST::expression
  cxx::ExpressionAST* value7 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value7);
  // ::cxx::ForStatementAST::rparenLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value8);
  // ::cxx::ForStatementAST::statement
  cxx::StatementAST* value9 =
      ast_cast<StatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value9);
  // ::cxx::ForStatementAST::symbol
  cxx::BlockSymbol* value10 =
      symbol_cast<BlockSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value10);
}

void SemanticDecoder::readAstBreakStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BreakStatementAST* self) {
  // ::cxx::BreakStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::BreakStatementAST::breakLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->breakLoc = std::move(value2);
  // ::cxx::BreakStatementAST::semicolonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value3);
}

void SemanticDecoder::readAstContinueStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ContinueStatementAST* self) {
  // ::cxx::ContinueStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::ContinueStatementAST::continueLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->continueLoc = std::move(value2);
  // ::cxx::ContinueStatementAST::semicolonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value3);
}

void SemanticDecoder::readAstReturnStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ReturnStatementAST* self) {
  // ::cxx::ReturnStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::ReturnStatementAST::returnLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->returnLoc = std::move(value2);
  // ::cxx::ReturnStatementAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::ReturnStatementAST::semicolonLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value4);
}

void SemanticDecoder::readAstCoroutineReturnStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CoroutineReturnStatementAST* self) {
  // ::cxx::CoroutineReturnStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::CoroutineReturnStatementAST::coreturnLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->coreturnLoc = std::move(value2);
  // ::cxx::CoroutineReturnStatementAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::CoroutineReturnStatementAST::semicolonLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value4);
}

void SemanticDecoder::readAstGotoStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::GotoStatementAST* self) {
  // ::cxx::GotoStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::GotoStatementAST::expression
  cxx::ExpressionAST* value2 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value2);
  // ::cxx::GotoStatementAST::gotoLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->gotoLoc = std::move(value3);
  // ::cxx::GotoStatementAST::starLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->starLoc = std::move(value4);
  // ::cxx::GotoStatementAST::identifierLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value5);
  // ::cxx::GotoStatementAST::semicolonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value6);
  // ::cxx::GotoStatementAST::identifier
  const cxx::Identifier* value7 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value7);
  // ::cxx::GotoStatementAST::isIndirect
  bool value8 = in.boolean();
  self->isIndirect = std::move(value8);
}

void SemanticDecoder::readAstDeclarationStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DeclarationStatementAST* self) {
  // ::cxx::DeclarationStatementAST::declaration
  cxx::DeclarationAST* value1 =
      ast_cast<DeclarationAST>(astAt(AstRef{in.varU32()}));
  self->declaration = std::move(value1);
}

void SemanticDecoder::readAstTryBlockStatementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TryBlockStatementAST* self) {
  // ::cxx::TryBlockStatementAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::TryBlockStatementAST::tryLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->tryLoc = std::move(value2);
  // ::cxx::TryBlockStatementAST::statement
  cxx::CompoundStatementAST* value3 =
      ast_cast<CompoundStatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value3);
  // ::cxx::TryBlockStatementAST::handlerList
  cxx::List<cxx::HandlerAST*>* value4 = readAstList<cxx::HandlerAST>(in);
  self->handlerList = std::move(value4);
}

void SemanticDecoder::readAstCharLiteralExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CharLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::CharLiteralExpressionAST::literalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value3);
  // ::cxx::CharLiteralExpressionAST::literal
  const cxx::CharLiteral* value4 = readCharLiteral(in);
  self->literal = std::move(value4);
  // ::cxx::CharLiteralExpressionAST::literalOperatorCall
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->literalOperatorCall = std::move(value5);
}

void SemanticDecoder::readAstBoolLiteralExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BoolLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::BoolLiteralExpressionAST::literalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value3);
  // ::cxx::BoolLiteralExpressionAST::isTrue
  bool value4 = in.boolean();
  self->isTrue = std::move(value4);
}

void SemanticDecoder::readAstIntLiteralExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::IntLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::IntLiteralExpressionAST::literalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value3);
  // ::cxx::IntLiteralExpressionAST::literal
  const cxx::IntegerLiteral* value4 = readIntegerLiteral(in);
  self->literal = std::move(value4);
  // ::cxx::IntLiteralExpressionAST::literalOperatorCall
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->literalOperatorCall = std::move(value5);
}

void SemanticDecoder::readAstFloatLiteralExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::FloatLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::FloatLiteralExpressionAST::literalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value3);
  // ::cxx::FloatLiteralExpressionAST::literal
  const cxx::FloatLiteral* value4 = readFloatLiteral(in);
  self->literal = std::move(value4);
  // ::cxx::FloatLiteralExpressionAST::literalOperatorCall
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->literalOperatorCall = std::move(value5);
}

void SemanticDecoder::readAstNullptrLiteralExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NullptrLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::NullptrLiteralExpressionAST::literalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value3);
  // ::cxx::NullptrLiteralExpressionAST::literal
  ::cxx::TokenKind value4 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->literal = std::move(value4);
}

void SemanticDecoder::readAstStringLiteralExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::StringLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::StringLiteralExpressionAST::literalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value3);
  // ::cxx::StringLiteralExpressionAST::literal
  const cxx::StringLiteral* value4 = readStringLiteral(in);
  self->literal = std::move(value4);
  // ::cxx::StringLiteralExpressionAST::encoding
  ::cxx::TokenKind value5 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->encoding = std::move(value5);
}

void SemanticDecoder::readAstUserDefinedStringLiteralExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UserDefinedStringLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::UserDefinedStringLiteralExpressionAST::literalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value3);
  // ::cxx::UserDefinedStringLiteralExpressionAST::literal
  const cxx::StringLiteral* value4 = readStringLiteral(in);
  self->literal = std::move(value4);
  // ::cxx::UserDefinedStringLiteralExpressionAST::literalOperatorCall
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->literalOperatorCall = std::move(value5);
  // ::cxx::UserDefinedStringLiteralExpressionAST::encoding
  ::cxx::TokenKind value6 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->encoding = std::move(value6);
}

void SemanticDecoder::readAstObjectLiteralExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ObjectLiteralExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ObjectLiteralExpressionAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::ObjectLiteralExpressionAST::typeId
  cxx::TypeIdAST* value4 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value4);
  // ::cxx::ObjectLiteralExpressionAST::rparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value5);
  // ::cxx::ObjectLiteralExpressionAST::bracedInitList
  cxx::BracedInitListAST* value6 =
      ast_cast<BracedInitListAST>(astAt(AstRef{in.varU32()}));
  self->bracedInitList = std::move(value6);
  // ::cxx::ObjectLiteralExpressionAST::symbol
  cxx::VariableSymbol* value7 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value7);
}

void SemanticDecoder::readAstThisExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ThisExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ThisExpressionAST::thisLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->thisLoc = std::move(value3);
}

void SemanticDecoder::readAstPackIndexExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PackIndexExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::PackIndexExpressionAST::packExpression
  cxx::IdExpressionAST* value3 =
      ast_cast<IdExpressionAST>(astAt(AstRef{in.varU32()}));
  self->packExpression = std::move(value3);
  // ::cxx::PackIndexExpressionAST::ellipsisLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value4);
  // ::cxx::PackIndexExpressionAST::lbracketLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value5);
  // ::cxx::PackIndexExpressionAST::indexExpression
  cxx::ExpressionAST* value6 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->indexExpression = std::move(value6);
  // ::cxx::PackIndexExpressionAST::rbracketLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value7);
}

void SemanticDecoder::readAstGenericSelectionExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::GenericSelectionExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::GenericSelectionExpressionAST::genericLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->genericLoc = std::move(value3);
  // ::cxx::GenericSelectionExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::GenericSelectionExpressionAST::expression
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value5);
  // ::cxx::GenericSelectionExpressionAST::commaLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->commaLoc = std::move(value6);
  // ::cxx::GenericSelectionExpressionAST::genericAssociationList
  cxx::List<cxx::GenericAssociationAST*>* value7 =
      readAstList<cxx::GenericAssociationAST>(in);
  self->genericAssociationList = std::move(value7);
  // ::cxx::GenericSelectionExpressionAST::rparenLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value8);
  // ::cxx::GenericSelectionExpressionAST::matchedAssocIndex
  int value9 = static_cast<int>(in.varI32());
  self->matchedAssocIndex = std::move(value9);
}

void SemanticDecoder::readAstNestedStatementExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NestedStatementExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::NestedStatementExpressionAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::NestedStatementExpressionAST::statement
  cxx::CompoundStatementAST* value4 =
      ast_cast<CompoundStatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value4);
  // ::cxx::NestedStatementExpressionAST::rparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value5);
}

void SemanticDecoder::readAstDefaultInitializerExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DefaultInitializerExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::DefaultInitializerExpressionAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::DefaultInitializerExpressionAST::context
  cxx::DefaultInitializerContext value4{};
  readcxxDefaultInitializerContext(in, &value4);
  self->context = std::move(value4);
}

void SemanticDecoder::readAstNestedExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NestedExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::NestedExpressionAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::NestedExpressionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
  // ::cxx::NestedExpressionAST::rparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value5);
}

void SemanticDecoder::readAstIdExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::IdExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::IdExpressionAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value3 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value3);
  // ::cxx::IdExpressionAST::templateLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value4);
  // ::cxx::IdExpressionAST::unqualifiedId
  cxx::UnqualifiedIdAST* value5 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value5);
  // ::cxx::IdExpressionAST::symbol
  cxx::Symbol* value6 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value6);
  // ::cxx::IdExpressionAST::isTemplateIntroduced
  bool value7 = in.boolean();
  self->isTemplateIntroduced = std::move(value7);
}

void SemanticDecoder::readAstLambdaExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::LambdaExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::LambdaExpressionAST::lbracketLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value3);
  // ::cxx::LambdaExpressionAST::captureDefaultLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->captureDefaultLoc = std::move(value4);
  // ::cxx::LambdaExpressionAST::captureList
  cxx::List<cxx::LambdaCaptureAST*>* value5 =
      readAstList<cxx::LambdaCaptureAST>(in);
  self->captureList = std::move(value5);
  // ::cxx::LambdaExpressionAST::rbracketLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value6);
  // ::cxx::LambdaExpressionAST::lessLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->lessLoc = std::move(value7);
  // ::cxx::LambdaExpressionAST::templateParameterList
  cxx::List<cxx::TemplateParameterAST*>* value8 =
      readAstList<cxx::TemplateParameterAST>(in);
  self->templateParameterList = std::move(value8);
  // ::cxx::LambdaExpressionAST::greaterLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->greaterLoc = std::move(value9);
  // ::cxx::LambdaExpressionAST::templateRequiresClause
  cxx::RequiresClauseAST* value10 =
      ast_cast<RequiresClauseAST>(astAt(AstRef{in.varU32()}));
  self->templateRequiresClause = std::move(value10);
  // ::cxx::LambdaExpressionAST::expressionAttributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value11 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->expressionAttributeList = std::move(value11);
  // ::cxx::LambdaExpressionAST::lparenLoc
  cxx::SourceLocation value12 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value12);
  // ::cxx::LambdaExpressionAST::parameterDeclarationClause
  cxx::ParameterDeclarationClauseAST* value13 =
      ast_cast<ParameterDeclarationClauseAST>(astAt(AstRef{in.varU32()}));
  self->parameterDeclarationClause = std::move(value13);
  // ::cxx::LambdaExpressionAST::rparenLoc
  cxx::SourceLocation value14 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value14);
  // ::cxx::LambdaExpressionAST::gnuAtributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value15 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->gnuAtributeList = std::move(value15);
  // ::cxx::LambdaExpressionAST::lambdaSpecifierList
  cxx::List<cxx::LambdaSpecifierAST*>* value16 =
      readAstList<cxx::LambdaSpecifierAST>(in);
  self->lambdaSpecifierList = std::move(value16);
  // ::cxx::LambdaExpressionAST::exceptionSpecifier
  cxx::ExceptionSpecifierAST* value17 =
      ast_cast<ExceptionSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->exceptionSpecifier = std::move(value17);
  // ::cxx::LambdaExpressionAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value18 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value18);
  // ::cxx::LambdaExpressionAST::trailingReturnType
  cxx::TrailingReturnTypeAST* value19 =
      ast_cast<TrailingReturnTypeAST>(astAt(AstRef{in.varU32()}));
  self->trailingReturnType = std::move(value19);
  // ::cxx::LambdaExpressionAST::requiresClause
  cxx::RequiresClauseAST* value20 =
      ast_cast<RequiresClauseAST>(astAt(AstRef{in.varU32()}));
  self->requiresClause = std::move(value20);
  // ::cxx::LambdaExpressionAST::statement
  cxx::CompoundStatementAST* value21 =
      ast_cast<CompoundStatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value21);
  // ::cxx::LambdaExpressionAST::captureDefault
  ::cxx::TokenKind value22 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->captureDefault = std::move(value22);
  // ::cxx::LambdaExpressionAST::symbol
  cxx::LambdaSymbol* value23 =
      symbol_cast<LambdaSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value23);
  // ::cxx::LambdaExpressionAST::constructorSymbol
  cxx::FunctionSymbol* value24 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->constructorSymbol = std::move(value24);
}

void SemanticDecoder::readAstFoldExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::FoldExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::FoldExpressionAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::FoldExpressionAST::leftExpression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->leftExpression = std::move(value4);
  // ::cxx::FoldExpressionAST::opLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->opLoc = std::move(value5);
  // ::cxx::FoldExpressionAST::ellipsisLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value6);
  // ::cxx::FoldExpressionAST::foldOpLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->foldOpLoc = std::move(value7);
  // ::cxx::FoldExpressionAST::rightExpression
  cxx::ExpressionAST* value8 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->rightExpression = std::move(value8);
  // ::cxx::FoldExpressionAST::rparenLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value9);
  // ::cxx::FoldExpressionAST::op
  ::cxx::TokenKind value10 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->op = std::move(value10);
  // ::cxx::FoldExpressionAST::foldOp
  ::cxx::TokenKind value11 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->foldOp = std::move(value11);
}

void SemanticDecoder::readAstRightFoldExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::RightFoldExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::RightFoldExpressionAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::RightFoldExpressionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
  // ::cxx::RightFoldExpressionAST::opLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->opLoc = std::move(value5);
  // ::cxx::RightFoldExpressionAST::ellipsisLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value6);
  // ::cxx::RightFoldExpressionAST::rparenLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value7);
  // ::cxx::RightFoldExpressionAST::op
  ::cxx::TokenKind value8 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->op = std::move(value8);
}

void SemanticDecoder::readAstLeftFoldExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::LeftFoldExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::LeftFoldExpressionAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::LeftFoldExpressionAST::ellipsisLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value4);
  // ::cxx::LeftFoldExpressionAST::opLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->opLoc = std::move(value5);
  // ::cxx::LeftFoldExpressionAST::expression
  cxx::ExpressionAST* value6 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value6);
  // ::cxx::LeftFoldExpressionAST::rparenLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value7);
  // ::cxx::LeftFoldExpressionAST::op
  ::cxx::TokenKind value8 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->op = std::move(value8);
}

void SemanticDecoder::readAstRequiresExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::RequiresExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::RequiresExpressionAST::requiresLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->requiresLoc = std::move(value3);
  // ::cxx::RequiresExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::RequiresExpressionAST::parameterDeclarationClause
  cxx::ParameterDeclarationClauseAST* value5 =
      ast_cast<ParameterDeclarationClauseAST>(astAt(AstRef{in.varU32()}));
  self->parameterDeclarationClause = std::move(value5);
  // ::cxx::RequiresExpressionAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::RequiresExpressionAST::lbraceLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->lbraceLoc = std::move(value7);
  // ::cxx::RequiresExpressionAST::requirementList
  cxx::List<cxx::RequirementAST*>* value8 =
      readAstList<cxx::RequirementAST>(in);
  self->requirementList = std::move(value8);
  // ::cxx::RequiresExpressionAST::rbraceLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->rbraceLoc = std::move(value9);
}

void SemanticDecoder::readAstVaArgExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::VaArgExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::VaArgExpressionAST::vaArgLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->vaArgLoc = std::move(value3);
  // ::cxx::VaArgExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::VaArgExpressionAST::expression
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value5);
  // ::cxx::VaArgExpressionAST::commaLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->commaLoc = std::move(value6);
  // ::cxx::VaArgExpressionAST::typeId
  cxx::TypeIdAST* value7 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value7);
  // ::cxx::VaArgExpressionAST::rparenLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value8);
}

void SemanticDecoder::readAstSubscriptExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SubscriptExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::SubscriptExpressionAST::baseExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->baseExpression = std::move(value3);
  // ::cxx::SubscriptExpressionAST::lbracketLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value4);
  // ::cxx::SubscriptExpressionAST::indexExpression
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->indexExpression = std::move(value5);
  // ::cxx::SubscriptExpressionAST::rbracketLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value6);
  // ::cxx::SubscriptExpressionAST::symbol
  cxx::FunctionSymbol* value7 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value7);
  // ::cxx::SubscriptExpressionAST::isVirtualDispatch
  bool value8 = in.boolean();
  self->isVirtualDispatch = std::move(value8);
}

void SemanticDecoder::readAstCallExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CallExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::CallExpressionAST::baseExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->baseExpression = std::move(value3);
  // ::cxx::CallExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::CallExpressionAST::expressionList
  cxx::List<cxx::ExpressionAST*>* value5 = readAstList<cxx::ExpressionAST>(in);
  self->expressionList = std::move(value5);
  // ::cxx::CallExpressionAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::CallExpressionAST::isVirtualDispatch
  bool value7 = in.boolean();
  self->isVirtualDispatch = std::move(value7);
  // ::cxx::CallExpressionAST::constructorSymbol
  cxx::FunctionSymbol* value8 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->constructorSymbol = std::move(value8);
}

void SemanticDecoder::readAstTypeConstructionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeConstructionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::TypeConstructionAST::typeSpecifier
  cxx::SpecifierAST* value3 =
      ast_cast<SpecifierAST>(astAt(AstRef{in.varU32()}));
  self->typeSpecifier = std::move(value3);
  // ::cxx::TypeConstructionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::TypeConstructionAST::expressionList
  cxx::List<cxx::ExpressionAST*>* value5 = readAstList<cxx::ExpressionAST>(in);
  self->expressionList = std::move(value5);
  // ::cxx::TypeConstructionAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::TypeConstructionAST::constructorSymbol
  cxx::FunctionSymbol* value7 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->constructorSymbol = std::move(value7);
}

void SemanticDecoder::readAstBracedTypeConstructionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BracedTypeConstructionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::BracedTypeConstructionAST::typeSpecifier
  cxx::SpecifierAST* value3 =
      ast_cast<SpecifierAST>(astAt(AstRef{in.varU32()}));
  self->typeSpecifier = std::move(value3);
  // ::cxx::BracedTypeConstructionAST::bracedInitList
  cxx::BracedInitListAST* value4 =
      ast_cast<BracedInitListAST>(astAt(AstRef{in.varU32()}));
  self->bracedInitList = std::move(value4);
  // ::cxx::BracedTypeConstructionAST::constructorSymbol
  cxx::FunctionSymbol* value5 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->constructorSymbol = std::move(value5);
}

void SemanticDecoder::readAstSpliceMemberExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SpliceMemberExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::SpliceMemberExpressionAST::baseExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->baseExpression = std::move(value3);
  // ::cxx::SpliceMemberExpressionAST::accessLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->accessLoc = std::move(value4);
  // ::cxx::SpliceMemberExpressionAST::templateLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value5);
  // ::cxx::SpliceMemberExpressionAST::splicer
  cxx::SplicerAST* value6 = ast_cast<SplicerAST>(astAt(AstRef{in.varU32()}));
  self->splicer = std::move(value6);
  // ::cxx::SpliceMemberExpressionAST::symbol
  cxx::Symbol* value7 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value7);
  // ::cxx::SpliceMemberExpressionAST::accessOp
  ::cxx::TokenKind value8 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->accessOp = std::move(value8);
  // ::cxx::SpliceMemberExpressionAST::isTemplateIntroduced
  bool value9 = in.boolean();
  self->isTemplateIntroduced = std::move(value9);
}

void SemanticDecoder::readAstMemberExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::MemberExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::MemberExpressionAST::baseExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->baseExpression = std::move(value3);
  // ::cxx::MemberExpressionAST::accessLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->accessLoc = std::move(value4);
  // ::cxx::MemberExpressionAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value5 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value5);
  // ::cxx::MemberExpressionAST::templateLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value6);
  // ::cxx::MemberExpressionAST::unqualifiedId
  cxx::UnqualifiedIdAST* value7 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value7);
  // ::cxx::MemberExpressionAST::symbol
  cxx::Symbol* value8 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value8);
  // ::cxx::MemberExpressionAST::accessOp
  ::cxx::TokenKind value9 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->accessOp = std::move(value9);
  // ::cxx::MemberExpressionAST::isTemplateIntroduced
  bool value10 = in.boolean();
  self->isTemplateIntroduced = std::move(value10);
}

void SemanticDecoder::readAstPostIncrExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PostIncrExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::PostIncrExpressionAST::baseExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->baseExpression = std::move(value3);
  // ::cxx::PostIncrExpressionAST::opLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->opLoc = std::move(value4);
  // ::cxx::PostIncrExpressionAST::op
  ::cxx::TokenKind value5 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->op = std::move(value5);
  // ::cxx::PostIncrExpressionAST::symbol
  cxx::FunctionSymbol* value6 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value6);
  // ::cxx::PostIncrExpressionAST::isVirtualDispatch
  bool value7 = in.boolean();
  self->isVirtualDispatch = std::move(value7);
}

void SemanticDecoder::readAstCppCastExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CppCastExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::CppCastExpressionAST::castLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->castLoc = std::move(value3);
  // ::cxx::CppCastExpressionAST::lessLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lessLoc = std::move(value4);
  // ::cxx::CppCastExpressionAST::typeId
  cxx::TypeIdAST* value5 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value5);
  // ::cxx::CppCastExpressionAST::greaterLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->greaterLoc = std::move(value6);
  // ::cxx::CppCastExpressionAST::lparenLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value7);
  // ::cxx::CppCastExpressionAST::expression
  cxx::ExpressionAST* value8 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value8);
  // ::cxx::CppCastExpressionAST::rparenLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value9);
  // ::cxx::CppCastExpressionAST::castOp
  ::cxx::TokenKind value10 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->castOp = std::move(value10);
}

void SemanticDecoder::readAstBuiltinBitCastExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BuiltinBitCastExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::BuiltinBitCastExpressionAST::castLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->castLoc = std::move(value3);
  // ::cxx::BuiltinBitCastExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::BuiltinBitCastExpressionAST::typeId
  cxx::TypeIdAST* value5 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value5);
  // ::cxx::BuiltinBitCastExpressionAST::commaLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->commaLoc = std::move(value6);
  // ::cxx::BuiltinBitCastExpressionAST::expression
  cxx::ExpressionAST* value7 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value7);
  // ::cxx::BuiltinBitCastExpressionAST::rparenLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value8);
}

void SemanticDecoder::readAstBuiltinOffsetofExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BuiltinOffsetofExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::BuiltinOffsetofExpressionAST::offsetofLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->offsetofLoc = std::move(value3);
  // ::cxx::BuiltinOffsetofExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::BuiltinOffsetofExpressionAST::typeId
  cxx::TypeIdAST* value5 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value5);
  // ::cxx::BuiltinOffsetofExpressionAST::commaLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->commaLoc = std::move(value6);
  // ::cxx::BuiltinOffsetofExpressionAST::identifierLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value7);
  // ::cxx::BuiltinOffsetofExpressionAST::designatorList
  cxx::List<cxx::DesignatorAST*>* value8 = readAstList<cxx::DesignatorAST>(in);
  self->designatorList = std::move(value8);
  // ::cxx::BuiltinOffsetofExpressionAST::rparenLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value9);
  // ::cxx::BuiltinOffsetofExpressionAST::identifier
  const cxx::Identifier* value10 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value10);
  // ::cxx::BuiltinOffsetofExpressionAST::symbol
  cxx::FieldSymbol* value11 =
      symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value11);
}

void SemanticDecoder::readAstTypeidExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeidExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::TypeidExpressionAST::typeidLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->typeidLoc = std::move(value3);
  // ::cxx::TypeidExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::TypeidExpressionAST::expression
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value5);
  // ::cxx::TypeidExpressionAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
}

void SemanticDecoder::readAstTypeidOfTypeExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeidOfTypeExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::TypeidOfTypeExpressionAST::typeidLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->typeidLoc = std::move(value3);
  // ::cxx::TypeidOfTypeExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::TypeidOfTypeExpressionAST::typeId
  cxx::TypeIdAST* value5 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value5);
  // ::cxx::TypeidOfTypeExpressionAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
}

void SemanticDecoder::readAstSpliceExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SpliceExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::SpliceExpressionAST::splicer
  cxx::SplicerAST* value3 = ast_cast<SplicerAST>(astAt(AstRef{in.varU32()}));
  self->splicer = std::move(value3);
}

void SemanticDecoder::readAstGlobalScopeReflectExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::GlobalScopeReflectExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::GlobalScopeReflectExpressionAST::caretCaretLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->caretCaretLoc = std::move(value3);
  // ::cxx::GlobalScopeReflectExpressionAST::scopeLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->scopeLoc = std::move(value4);
}

void SemanticDecoder::readAstNamespaceReflectExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NamespaceReflectExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::NamespaceReflectExpressionAST::caretCaretLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->caretCaretLoc = std::move(value3);
  // ::cxx::NamespaceReflectExpressionAST::identifierLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value4);
  // ::cxx::NamespaceReflectExpressionAST::identifier
  const cxx::Identifier* value5 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value5);
  // ::cxx::NamespaceReflectExpressionAST::symbol
  cxx::NamespaceSymbol* value6 =
      symbol_cast<NamespaceSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value6);
}

void SemanticDecoder::readAstTypeIdReflectExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeIdReflectExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::TypeIdReflectExpressionAST::caretCaretLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->caretCaretLoc = std::move(value3);
  // ::cxx::TypeIdReflectExpressionAST::typeId
  cxx::TypeIdAST* value4 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value4);
}

void SemanticDecoder::readAstReflectExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ReflectExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ReflectExpressionAST::caretCaretLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->caretCaretLoc = std::move(value3);
  // ::cxx::ReflectExpressionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
}

void SemanticDecoder::readAstLabelAddressExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::LabelAddressExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::LabelAddressExpressionAST::ampAmpLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->ampAmpLoc = std::move(value3);
  // ::cxx::LabelAddressExpressionAST::identifierLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value4);
  // ::cxx::LabelAddressExpressionAST::identifier
  const cxx::Identifier* value5 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value5);
}

void SemanticDecoder::readAstUnaryExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UnaryExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::UnaryExpressionAST::opLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->opLoc = std::move(value3);
  // ::cxx::UnaryExpressionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
  // ::cxx::UnaryExpressionAST::op
  ::cxx::TokenKind value5 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->op = std::move(value5);
  // ::cxx::UnaryExpressionAST::symbol
  cxx::FunctionSymbol* value6 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value6);
  // ::cxx::UnaryExpressionAST::isVirtualDispatch
  bool value7 = in.boolean();
  self->isVirtualDispatch = std::move(value7);
}

void SemanticDecoder::readAstAwaitExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AwaitExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::AwaitExpressionAST::awaitLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->awaitLoc = std::move(value3);
  // ::cxx::AwaitExpressionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
}

void SemanticDecoder::readAstSizeofExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SizeofExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::SizeofExpressionAST::sizeofLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->sizeofLoc = std::move(value3);
  // ::cxx::SizeofExpressionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
  // ::cxx::SizeofExpressionAST::value
  std::optional<long long> value5;
  if (in.boolean()) {
    long long value6 = static_cast<long long>(in.varI64());
    value5 = std::move(value6);
  }
  self->value = std::move(value5);
}

void SemanticDecoder::readAstSizeofTypeExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SizeofTypeExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::SizeofTypeExpressionAST::sizeofLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->sizeofLoc = std::move(value3);
  // ::cxx::SizeofTypeExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::SizeofTypeExpressionAST::typeId
  cxx::TypeIdAST* value5 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value5);
  // ::cxx::SizeofTypeExpressionAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::SizeofTypeExpressionAST::value
  std::optional<long long> value7;
  if (in.boolean()) {
    long long value8 = static_cast<long long>(in.varI64());
    value7 = std::move(value8);
  }
  self->value = std::move(value7);
}

void SemanticDecoder::readAstSizeofPackExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SizeofPackExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::SizeofPackExpressionAST::sizeofLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->sizeofLoc = std::move(value3);
  // ::cxx::SizeofPackExpressionAST::ellipsisLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value4);
  // ::cxx::SizeofPackExpressionAST::lparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value5);
  // ::cxx::SizeofPackExpressionAST::identifierLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value6);
  // ::cxx::SizeofPackExpressionAST::rparenLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value7);
  // ::cxx::SizeofPackExpressionAST::identifier
  const cxx::Identifier* value8 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value8);
  // ::cxx::SizeofPackExpressionAST::symbol
  cxx::Symbol* value9 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value9);
}

void SemanticDecoder::readAstAlignofTypeExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AlignofTypeExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::AlignofTypeExpressionAST::alignofLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->alignofLoc = std::move(value3);
  // ::cxx::AlignofTypeExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::AlignofTypeExpressionAST::typeId
  cxx::TypeIdAST* value5 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value5);
  // ::cxx::AlignofTypeExpressionAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
}

void SemanticDecoder::readAstAlignofExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AlignofExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::AlignofExpressionAST::alignofLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->alignofLoc = std::move(value3);
  // ::cxx::AlignofExpressionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
}

void SemanticDecoder::readAstNoexceptExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NoexceptExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::NoexceptExpressionAST::noexceptLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->noexceptLoc = std::move(value3);
  // ::cxx::NoexceptExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::NoexceptExpressionAST::expression
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value5);
  // ::cxx::NoexceptExpressionAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::NoexceptExpressionAST::value
  std::optional<bool> value7;
  if (in.boolean()) {
    bool value8 = in.boolean();
    value7 = std::move(value8);
  }
  self->value = std::move(value7);
}

void SemanticDecoder::readAstNewExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NewExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::NewExpressionAST::scopeLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->scopeLoc = std::move(value3);
  // ::cxx::NewExpressionAST::newLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->newLoc = std::move(value4);
  // ::cxx::NewExpressionAST::newPlacement
  cxx::NewPlacementAST* value5 =
      ast_cast<NewPlacementAST>(astAt(AstRef{in.varU32()}));
  self->newPlacement = std::move(value5);
  // ::cxx::NewExpressionAST::lparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value6);
  // ::cxx::NewExpressionAST::typeSpecifierList
  cxx::List<cxx::SpecifierAST*>* value7 = readAstList<cxx::SpecifierAST>(in);
  self->typeSpecifierList = std::move(value7);
  // ::cxx::NewExpressionAST::declarator
  cxx::DeclaratorAST* value8 =
      ast_cast<DeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->declarator = std::move(value8);
  // ::cxx::NewExpressionAST::rparenLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value9);
  // ::cxx::NewExpressionAST::newInitalizer
  cxx::NewInitializerAST* value10 =
      ast_cast<NewInitializerAST>(astAt(AstRef{in.varU32()}));
  self->newInitalizer = std::move(value10);
  // ::cxx::NewExpressionAST::objectType
  const cxx::Type* value11 = typeAt(TypeRef{in.varU32()});
  self->objectType = std::move(value11);
  // ::cxx::NewExpressionAST::constructorSymbol
  cxx::FunctionSymbol* value12 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->constructorSymbol = std::move(value12);
  // ::cxx::NewExpressionAST::symbol
  cxx::FunctionSymbol* value13 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value13);
}

void SemanticDecoder::readAstDeleteExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DeleteExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::DeleteExpressionAST::scopeLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->scopeLoc = std::move(value3);
  // ::cxx::DeleteExpressionAST::deleteLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->deleteLoc = std::move(value4);
  // ::cxx::DeleteExpressionAST::lbracketLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value5);
  // ::cxx::DeleteExpressionAST::rbracketLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value6);
  // ::cxx::DeleteExpressionAST::expression
  cxx::ExpressionAST* value7 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value7);
  // ::cxx::DeleteExpressionAST::symbol
  cxx::FunctionSymbol* value8 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value8);
}

void SemanticDecoder::readAstCastExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CastExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::CastExpressionAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::CastExpressionAST::typeId
  cxx::TypeIdAST* value4 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value4);
  // ::cxx::CastExpressionAST::rparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value5);
  // ::cxx::CastExpressionAST::expression
  cxx::ExpressionAST* value6 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value6);
}

void SemanticDecoder::readAstImplicitCastExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ImplicitCastExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ImplicitCastExpressionAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::ImplicitCastExpressionAST::castKind
  static_assert(static_cast<std::uint32_t>(
                    ::cxx::ImplicitCastKind::kUserDefinedConversion) +
                    1 ==
                25);
  ::cxx::ImplicitCastKind value4 =
      static_cast<::cxx::ImplicitCastKind>(readEnum(in, 25));
  self->castKind = std::move(value4);
  // ::cxx::ImplicitCastExpressionAST::conversionFunction
  cxx::FunctionSymbol* value5 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->conversionFunction = std::move(value5);
  // ::cxx::ImplicitCastExpressionAST::isVirtualDispatch
  bool value6 = in.boolean();
  self->isVirtualDispatch = std::move(value6);
}

void SemanticDecoder::readAstConstExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ConstExpressionAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::ConstExpressionAST::constValue
  const cxx::ConstValue* value4 = nullptr;
  if (in.boolean()) value4 = arena()->make<cxx::ConstValue>(readConstValue(in));
  self->constValue = std::move(value4);
}

void SemanticDecoder::readAstBinaryExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BinaryExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::BinaryExpressionAST::leftExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->leftExpression = std::move(value3);
  // ::cxx::BinaryExpressionAST::opLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->opLoc = std::move(value4);
  // ::cxx::BinaryExpressionAST::rightExpression
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->rightExpression = std::move(value5);
  // ::cxx::BinaryExpressionAST::op
  ::cxx::TokenKind value6 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->op = std::move(value6);
  // ::cxx::BinaryExpressionAST::symbol
  cxx::FunctionSymbol* value7 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value7);
  // ::cxx::BinaryExpressionAST::isVirtualDispatch
  bool value8 = in.boolean();
  self->isVirtualDispatch = std::move(value8);
}

void SemanticDecoder::readAstConditionalExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConditionalExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ConditionalExpressionAST::condition
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->condition = std::move(value3);
  // ::cxx::ConditionalExpressionAST::questionLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->questionLoc = std::move(value4);
  // ::cxx::ConditionalExpressionAST::iftrueExpression
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->iftrueExpression = std::move(value5);
  // ::cxx::ConditionalExpressionAST::colonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value6);
  // ::cxx::ConditionalExpressionAST::iffalseExpression
  cxx::ExpressionAST* value7 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->iffalseExpression = std::move(value7);
}

void SemanticDecoder::readAstYieldExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::YieldExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::YieldExpressionAST::yieldLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->yieldLoc = std::move(value3);
  // ::cxx::YieldExpressionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
}

void SemanticDecoder::readAstThrowExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ThrowExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ThrowExpressionAST::throwLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->throwLoc = std::move(value3);
  // ::cxx::ThrowExpressionAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
}

void SemanticDecoder::readAstAssignmentExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AssignmentExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::AssignmentExpressionAST::leftExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->leftExpression = std::move(value3);
  // ::cxx::AssignmentExpressionAST::opLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->opLoc = std::move(value4);
  // ::cxx::AssignmentExpressionAST::rightExpression
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->rightExpression = std::move(value5);
  // ::cxx::AssignmentExpressionAST::op
  ::cxx::TokenKind value6 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->op = std::move(value6);
  // ::cxx::AssignmentExpressionAST::symbol
  cxx::FunctionSymbol* value7 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value7);
  // ::cxx::AssignmentExpressionAST::isVirtualDispatch
  bool value8 = in.boolean();
  self->isVirtualDispatch = std::move(value8);
}

void SemanticDecoder::readAstTargetExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TargetExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
}

void SemanticDecoder::readAstRightExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::RightExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
}

void SemanticDecoder::readAstCompoundAssignmentExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CompoundAssignmentExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::CompoundAssignmentExpressionAST::targetExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->targetExpression = std::move(value3);
  // ::cxx::CompoundAssignmentExpressionAST::opLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->opLoc = std::move(value4);
  // ::cxx::CompoundAssignmentExpressionAST::leftExpression
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->leftExpression = std::move(value5);
  // ::cxx::CompoundAssignmentExpressionAST::rightExpression
  cxx::ExpressionAST* value6 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->rightExpression = std::move(value6);
  // ::cxx::CompoundAssignmentExpressionAST::adjustExpression
  cxx::ExpressionAST* value7 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->adjustExpression = std::move(value7);
  // ::cxx::CompoundAssignmentExpressionAST::op
  ::cxx::TokenKind value8 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->op = std::move(value8);
  // ::cxx::CompoundAssignmentExpressionAST::symbol
  cxx::FunctionSymbol* value9 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value9);
  // ::cxx::CompoundAssignmentExpressionAST::isVirtualDispatch
  bool value10 = in.boolean();
  self->isVirtualDispatch = std::move(value10);
}

void SemanticDecoder::readAstPackExpansionExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PackExpansionExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::PackExpansionExpressionAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::PackExpansionExpressionAST::ellipsisLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value4);
}

void SemanticDecoder::readAstDesignatedInitializerClauseAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DesignatedInitializerClauseAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::DesignatedInitializerClauseAST::designatorList
  cxx::List<cxx::DesignatorAST*>* value3 = readAstList<cxx::DesignatorAST>(in);
  self->designatorList = std::move(value3);
  // ::cxx::DesignatedInitializerClauseAST::initializer
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value4);
  // ::cxx::DesignatedInitializerClauseAST::constructorSymbol
  cxx::FunctionSymbol* value5 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->constructorSymbol = std::move(value5);
}

void SemanticDecoder::readAstTypeTraitExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeTraitExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::TypeTraitExpressionAST::typeTraitLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->typeTraitLoc = std::move(value3);
  // ::cxx::TypeTraitExpressionAST::lparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value4);
  // ::cxx::TypeTraitExpressionAST::typeIdList
  cxx::List<cxx::TypeIdAST*>* value5 = readAstList<cxx::TypeIdAST>(in);
  self->typeIdList = std::move(value5);
  // ::cxx::TypeTraitExpressionAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::TypeTraitExpressionAST::typeTrait
  static_assert(
      static_cast<std::uint32_t>(
          ::cxx::BuiltinTypeTraitKind::T___REFERENCE_CONVERTS_FROM_TEMPORARY) +
          1 ==
      60);
  ::cxx::BuiltinTypeTraitKind value7 =
      static_cast<::cxx::BuiltinTypeTraitKind>(readEnum(in, 60));
  self->typeTrait = std::move(value7);
  // ::cxx::TypeTraitExpressionAST::value
  std::optional<bool> value8;
  if (in.boolean()) {
    bool value9 = in.boolean();
    value8 = std::move(value9);
  }
  self->value = std::move(value8);
}

void SemanticDecoder::readAstConditionExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConditionExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ConditionExpressionAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value3 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value3);
  // ::cxx::ConditionExpressionAST::declSpecifierList
  cxx::List<cxx::SpecifierAST*>* value4 = readAstList<cxx::SpecifierAST>(in);
  self->declSpecifierList = std::move(value4);
  // ::cxx::ConditionExpressionAST::declarator
  cxx::DeclaratorAST* value5 =
      ast_cast<DeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->declarator = std::move(value5);
  // ::cxx::ConditionExpressionAST::initializer
  cxx::ExpressionAST* value6 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value6);
  // ::cxx::ConditionExpressionAST::symbol
  cxx::VariableSymbol* value7 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value7);
}

void SemanticDecoder::readAstEqualInitializerAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::EqualInitializerAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::EqualInitializerAST::equalLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value3);
  // ::cxx::EqualInitializerAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
}

void SemanticDecoder::readAstBracedInitListAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BracedInitListAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::BracedInitListAST::lbraceLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lbraceLoc = std::move(value3);
  // ::cxx::BracedInitListAST::expressionList
  cxx::List<cxx::ExpressionAST*>* value4 = readAstList<cxx::ExpressionAST>(in);
  self->expressionList = std::move(value4);
  // ::cxx::BracedInitListAST::commaLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->commaLoc = std::move(value5);
  // ::cxx::BracedInitListAST::rbraceLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rbraceLoc = std::move(value6);
}

void SemanticDecoder::readAstParenInitializerAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ParenInitializerAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ParenInitializerAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::ParenInitializerAST::expressionList
  cxx::List<cxx::ExpressionAST*>* value4 = readAstList<cxx::ExpressionAST>(in);
  self->expressionList = std::move(value4);
  // ::cxx::ParenInitializerAST::rparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value5);
}

void SemanticDecoder::readAstThreeWayComparisonExpressionAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ThreeWayComparisonExpressionAST* self) {
  // ::cxx::ExpressionAST::valueCategory
  static_assert(
      static_cast<std::uint32_t>(::cxx::ValueCategory::kPrValue) + 1 == 4);
  ::cxx::ValueCategory value1 =
      static_cast<::cxx::ValueCategory>(readEnum(in, 4));
  self->valueCategory = std::move(value1);
  // ::cxx::ExpressionAST::type
  const cxx::Type* value2 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value2);
  // ::cxx::ThreeWayComparisonExpressionAST::comparison
  cxx::BinaryExpressionAST* value3 =
      ast_cast<BinaryExpressionAST>(astAt(AstRef{in.varU32()}));
  self->comparison = std::move(value3);
  // ::cxx::ThreeWayComparisonExpressionAST::lessResult
  cxx::Symbol* value4 = symbolAt(SymbolRef{in.varU32()});
  self->lessResult = std::move(value4);
  // ::cxx::ThreeWayComparisonExpressionAST::equalResult
  cxx::Symbol* value5 = symbolAt(SymbolRef{in.varU32()});
  self->equalResult = std::move(value5);
  // ::cxx::ThreeWayComparisonExpressionAST::greaterResult
  cxx::Symbol* value6 = symbolAt(SymbolRef{in.varU32()});
  self->greaterResult = std::move(value6);
  // ::cxx::ThreeWayComparisonExpressionAST::unorderedResult
  cxx::Symbol* value7 = symbolAt(SymbolRef{in.varU32()});
  self->unorderedResult = std::move(value7);
}

void SemanticDecoder::readAstDefaultGenericAssociationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DefaultGenericAssociationAST* self) {
  // ::cxx::DefaultGenericAssociationAST::defaultLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->defaultLoc = std::move(value1);
  // ::cxx::DefaultGenericAssociationAST::colonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value2);
  // ::cxx::DefaultGenericAssociationAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
}

void SemanticDecoder::readAstTypeGenericAssociationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeGenericAssociationAST* self) {
  // ::cxx::TypeGenericAssociationAST::typeId
  cxx::TypeIdAST* value1 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value1);
  // ::cxx::TypeGenericAssociationAST::colonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value2);
  // ::cxx::TypeGenericAssociationAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
}

void SemanticDecoder::readAstDotDesignatorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DotDesignatorAST* self) {
  // ::cxx::DotDesignatorAST::dotLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->dotLoc = std::move(value1);
  // ::cxx::DotDesignatorAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::DotDesignatorAST::identifier
  const cxx::Identifier* value3 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value3);
  // ::cxx::DotDesignatorAST::symbol
  cxx::FieldSymbol* value4 =
      symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value4);
}

void SemanticDecoder::readAstSubscriptDesignatorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SubscriptDesignatorAST* self) {
  // ::cxx::SubscriptDesignatorAST::lbracketLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value1);
  // ::cxx::SubscriptDesignatorAST::expression
  cxx::ExpressionAST* value2 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value2);
  // ::cxx::SubscriptDesignatorAST::rbracketLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value3);
}

void SemanticDecoder::readAstTemplateTypeParameterAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TemplateTypeParameterAST* self) {
  // ::cxx::TemplateParameterAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::TemplateParameterAST::depth
  int value2 = static_cast<int>(in.varI32());
  self->depth = std::move(value2);
  // ::cxx::TemplateParameterAST::index
  int value3 = static_cast<int>(in.varI32());
  self->index = std::move(value3);
  // ::cxx::TemplateTypeParameterAST::templateLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value4);
  // ::cxx::TemplateTypeParameterAST::lessLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->lessLoc = std::move(value5);
  // ::cxx::TemplateTypeParameterAST::templateParameterList
  cxx::List<cxx::TemplateParameterAST*>* value6 =
      readAstList<cxx::TemplateParameterAST>(in);
  self->templateParameterList = std::move(value6);
  // ::cxx::TemplateTypeParameterAST::greaterLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->greaterLoc = std::move(value7);
  // ::cxx::TemplateTypeParameterAST::requiresClause
  cxx::RequiresClauseAST* value8 =
      ast_cast<RequiresClauseAST>(astAt(AstRef{in.varU32()}));
  self->requiresClause = std::move(value8);
  // ::cxx::TemplateTypeParameterAST::classKeyLoc
  cxx::SourceLocation value9 = locationAt(LocationRef{in.varU32()});
  self->classKeyLoc = std::move(value9);
  // ::cxx::TemplateTypeParameterAST::ellipsisLoc
  cxx::SourceLocation value10 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value10);
  // ::cxx::TemplateTypeParameterAST::identifierLoc
  cxx::SourceLocation value11 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value11);
  // ::cxx::TemplateTypeParameterAST::equalLoc
  cxx::SourceLocation value12 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value12);
  // ::cxx::TemplateTypeParameterAST::idExpression
  cxx::IdExpressionAST* value13 =
      ast_cast<IdExpressionAST>(astAt(AstRef{in.varU32()}));
  self->idExpression = std::move(value13);
  // ::cxx::TemplateTypeParameterAST::identifier
  const cxx::Identifier* value14 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value14);
  // ::cxx::TemplateTypeParameterAST::isPack
  bool value15 = in.boolean();
  self->isPack = std::move(value15);
}

void SemanticDecoder::readAstNonTypeTemplateParameterAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NonTypeTemplateParameterAST* self) {
  // ::cxx::TemplateParameterAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::TemplateParameterAST::depth
  int value2 = static_cast<int>(in.varI32());
  self->depth = std::move(value2);
  // ::cxx::TemplateParameterAST::index
  int value3 = static_cast<int>(in.varI32());
  self->index = std::move(value3);
  // ::cxx::NonTypeTemplateParameterAST::declaration
  cxx::ParameterDeclarationAST* value4 =
      ast_cast<ParameterDeclarationAST>(astAt(AstRef{in.varU32()}));
  self->declaration = std::move(value4);
}

void SemanticDecoder::readAstTypenameTypeParameterAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypenameTypeParameterAST* self) {
  // ::cxx::TemplateParameterAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::TemplateParameterAST::depth
  int value2 = static_cast<int>(in.varI32());
  self->depth = std::move(value2);
  // ::cxx::TemplateParameterAST::index
  int value3 = static_cast<int>(in.varI32());
  self->index = std::move(value3);
  // ::cxx::TypenameTypeParameterAST::classKeyLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->classKeyLoc = std::move(value4);
  // ::cxx::TypenameTypeParameterAST::ellipsisLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value5);
  // ::cxx::TypenameTypeParameterAST::identifierLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value6);
  // ::cxx::TypenameTypeParameterAST::equalLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value7);
  // ::cxx::TypenameTypeParameterAST::typeId
  cxx::TypeIdAST* value8 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value8);
  // ::cxx::TypenameTypeParameterAST::identifier
  const cxx::Identifier* value9 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value9);
  // ::cxx::TypenameTypeParameterAST::isPack
  bool value10 = in.boolean();
  self->isPack = std::move(value10);
}

void SemanticDecoder::readAstConstraintTypeParameterAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstraintTypeParameterAST* self) {
  // ::cxx::TemplateParameterAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::TemplateParameterAST::depth
  int value2 = static_cast<int>(in.varI32());
  self->depth = std::move(value2);
  // ::cxx::TemplateParameterAST::index
  int value3 = static_cast<int>(in.varI32());
  self->index = std::move(value3);
  // ::cxx::ConstraintTypeParameterAST::typeConstraint
  cxx::TypeConstraintAST* value4 =
      ast_cast<TypeConstraintAST>(astAt(AstRef{in.varU32()}));
  self->typeConstraint = std::move(value4);
  // ::cxx::ConstraintTypeParameterAST::ellipsisLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value5);
  // ::cxx::ConstraintTypeParameterAST::identifierLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value6);
  // ::cxx::ConstraintTypeParameterAST::equalLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value7);
  // ::cxx::ConstraintTypeParameterAST::typeId
  cxx::TypeIdAST* value8 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value8);
  // ::cxx::ConstraintTypeParameterAST::identifier
  const cxx::Identifier* value9 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value9);
}

void SemanticDecoder::readAstTypedefSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypedefSpecifierAST* self) {
  // ::cxx::TypedefSpecifierAST::typedefLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->typedefLoc = std::move(value1);
}

void SemanticDecoder::readAstFriendSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::FriendSpecifierAST* self) {
  // ::cxx::FriendSpecifierAST::friendLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->friendLoc = std::move(value1);
}

void SemanticDecoder::readAstConstevalSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstevalSpecifierAST* self) {
  // ::cxx::ConstevalSpecifierAST::constevalLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->constevalLoc = std::move(value1);
}

void SemanticDecoder::readAstConstinitSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstinitSpecifierAST* self) {
  // ::cxx::ConstinitSpecifierAST::constinitLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->constinitLoc = std::move(value1);
}

void SemanticDecoder::readAstConstexprSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstexprSpecifierAST* self) {
  // ::cxx::ConstexprSpecifierAST::constexprLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->constexprLoc = std::move(value1);
}

void SemanticDecoder::readAstInlineSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::InlineSpecifierAST* self) {
  // ::cxx::InlineSpecifierAST::inlineLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->inlineLoc = std::move(value1);
}

void SemanticDecoder::readAstNoreturnSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NoreturnSpecifierAST* self) {
  // ::cxx::NoreturnSpecifierAST::noreturnLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->noreturnLoc = std::move(value1);
}

void SemanticDecoder::readAstStaticSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::StaticSpecifierAST* self) {
  // ::cxx::StaticSpecifierAST::staticLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->staticLoc = std::move(value1);
}

void SemanticDecoder::readAstExternSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ExternSpecifierAST* self) {
  // ::cxx::ExternSpecifierAST::externLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->externLoc = std::move(value1);
}

void SemanticDecoder::readAstRegisterSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::RegisterSpecifierAST* self) {
  // ::cxx::RegisterSpecifierAST::registerLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->registerLoc = std::move(value1);
}

void SemanticDecoder::readAstThreadLocalSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ThreadLocalSpecifierAST* self) {
  // ::cxx::ThreadLocalSpecifierAST::threadLocalLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->threadLocalLoc = std::move(value1);
}

void SemanticDecoder::readAstThreadSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ThreadSpecifierAST* self) {
  // ::cxx::ThreadSpecifierAST::threadLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->threadLoc = std::move(value1);
}

void SemanticDecoder::readAstMutableSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::MutableSpecifierAST* self) {
  // ::cxx::MutableSpecifierAST::mutableLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->mutableLoc = std::move(value1);
}

void SemanticDecoder::readAstVirtualSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::VirtualSpecifierAST* self) {
  // ::cxx::VirtualSpecifierAST::virtualLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->virtualLoc = std::move(value1);
}

void SemanticDecoder::readAstExplicitSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ExplicitSpecifierAST* self) {
  // ::cxx::ExplicitSpecifierAST::explicitLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->explicitLoc = std::move(value1);
  // ::cxx::ExplicitSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::ExplicitSpecifierAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::ExplicitSpecifierAST::rparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value4);
}

void SemanticDecoder::readAstAutoTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AutoTypeSpecifierAST* self) {
  // ::cxx::AutoTypeSpecifierAST::autoLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->autoLoc = std::move(value1);
}

void SemanticDecoder::readAstVoidTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::VoidTypeSpecifierAST* self) {
  // ::cxx::VoidTypeSpecifierAST::voidLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->voidLoc = std::move(value1);
}

void SemanticDecoder::readAstSizeTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SizeTypeSpecifierAST* self) {
  // ::cxx::SizeTypeSpecifierAST::specifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->specifierLoc = std::move(value1);
  // ::cxx::SizeTypeSpecifierAST::specifier
  ::cxx::TokenKind value2 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->specifier = std::move(value2);
}

void SemanticDecoder::readAstSignTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SignTypeSpecifierAST* self) {
  // ::cxx::SignTypeSpecifierAST::specifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->specifierLoc = std::move(value1);
  // ::cxx::SignTypeSpecifierAST::specifier
  ::cxx::TokenKind value2 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->specifier = std::move(value2);
}

void SemanticDecoder::readAstBuiltinTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BuiltinTypeSpecifierAST* self) {
  // ::cxx::BuiltinTypeSpecifierAST::specifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->specifierLoc = std::move(value1);
  // ::cxx::BuiltinTypeSpecifierAST::specifier
  ::cxx::TokenKind value2 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->specifier = std::move(value2);
}

void SemanticDecoder::readAstUnaryBuiltinTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UnaryBuiltinTypeSpecifierAST* self) {
  // ::cxx::UnaryBuiltinTypeSpecifierAST::builtinLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->builtinLoc = std::move(value1);
  // ::cxx::UnaryBuiltinTypeSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::UnaryBuiltinTypeSpecifierAST::typeId
  cxx::TypeIdAST* value3 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value3);
  // ::cxx::UnaryBuiltinTypeSpecifierAST::rparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value4);
  // ::cxx::UnaryBuiltinTypeSpecifierAST::builtinKind
  static_assert(static_cast<std::uint32_t>(
                    ::cxx::UnaryBuiltinTypeKind::T___REMOVE_VOLATILE) +
                    1 ==
                16);
  ::cxx::UnaryBuiltinTypeKind value5 =
      static_cast<::cxx::UnaryBuiltinTypeKind>(readEnum(in, 16));
  self->builtinKind = std::move(value5);
}

void SemanticDecoder::readAstBinaryBuiltinTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BinaryBuiltinTypeSpecifierAST* self) {
  // ::cxx::BinaryBuiltinTypeSpecifierAST::builtinLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->builtinLoc = std::move(value1);
  // ::cxx::BinaryBuiltinTypeSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::BinaryBuiltinTypeSpecifierAST::leftTypeId
  cxx::TypeIdAST* value3 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->leftTypeId = std::move(value3);
  // ::cxx::BinaryBuiltinTypeSpecifierAST::commaLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->commaLoc = std::move(value4);
  // ::cxx::BinaryBuiltinTypeSpecifierAST::rightTypeId
  cxx::TypeIdAST* value5 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->rightTypeId = std::move(value5);
  // ::cxx::BinaryBuiltinTypeSpecifierAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::BinaryBuiltinTypeSpecifierAST::builtinKind
  static_assert(
      static_cast<std::uint32_t>(::cxx::BinaryBuiltinTypeKind::T_NONE) + 1 ==
      1);
  ::cxx::BinaryBuiltinTypeKind value7 =
      static_cast<::cxx::BinaryBuiltinTypeKind>(readEnum(in, 1));
  self->builtinKind = std::move(value7);
}

void SemanticDecoder::readAstIntegralTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::IntegralTypeSpecifierAST* self) {
  // ::cxx::IntegralTypeSpecifierAST::specifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->specifierLoc = std::move(value1);
  // ::cxx::IntegralTypeSpecifierAST::specifier
  ::cxx::TokenKind value2 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->specifier = std::move(value2);
}

void SemanticDecoder::readAstFloatingPointTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::FloatingPointTypeSpecifierAST* self) {
  // ::cxx::FloatingPointTypeSpecifierAST::specifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->specifierLoc = std::move(value1);
  // ::cxx::FloatingPointTypeSpecifierAST::specifier
  ::cxx::TokenKind value2 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->specifier = std::move(value2);
}

void SemanticDecoder::readAstComplexTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ComplexTypeSpecifierAST* self) {
  // ::cxx::ComplexTypeSpecifierAST::complexLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->complexLoc = std::move(value1);
}

void SemanticDecoder::readAstNamedTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NamedTypeSpecifierAST* self) {
  // ::cxx::NamedTypeSpecifierAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value1 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value1);
  // ::cxx::NamedTypeSpecifierAST::templateLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value2);
  // ::cxx::NamedTypeSpecifierAST::unqualifiedId
  cxx::UnqualifiedIdAST* value3 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value3);
  // ::cxx::NamedTypeSpecifierAST::isTemplateIntroduced
  bool value4 = in.boolean();
  self->isTemplateIntroduced = std::move(value4);
  // ::cxx::NamedTypeSpecifierAST::symbol
  cxx::Symbol* value5 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value5);
}

void SemanticDecoder::readAstAtomicTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AtomicTypeSpecifierAST* self) {
  // ::cxx::AtomicTypeSpecifierAST::atomicLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->atomicLoc = std::move(value1);
  // ::cxx::AtomicTypeSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::AtomicTypeSpecifierAST::typeId
  cxx::TypeIdAST* value3 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value3);
  // ::cxx::AtomicTypeSpecifierAST::rparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value4);
}

void SemanticDecoder::readAstBitIntTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BitIntTypeSpecifierAST* self) {
  // ::cxx::BitIntTypeSpecifierAST::bitintLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->bitintLoc = std::move(value1);
  // ::cxx::BitIntTypeSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::BitIntTypeSpecifierAST::sizeExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->sizeExpression = std::move(value3);
  // ::cxx::BitIntTypeSpecifierAST::rparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value4);
  // ::cxx::BitIntTypeSpecifierAST::bitCount
  int value5 = static_cast<int>(in.varI32());
  self->bitCount = std::move(value5);
}

void SemanticDecoder::readAstUnderlyingTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::UnderlyingTypeSpecifierAST* self) {
  // ::cxx::UnderlyingTypeSpecifierAST::underlyingTypeLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->underlyingTypeLoc = std::move(value1);
  // ::cxx::UnderlyingTypeSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::UnderlyingTypeSpecifierAST::typeId
  cxx::TypeIdAST* value3 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value3);
  // ::cxx::UnderlyingTypeSpecifierAST::rparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value4);
}

void SemanticDecoder::readAstElaboratedTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ElaboratedTypeSpecifierAST* self) {
  // ::cxx::ElaboratedTypeSpecifierAST::classLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->classLoc = std::move(value1);
  // ::cxx::ElaboratedTypeSpecifierAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value2 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value2);
  // ::cxx::ElaboratedTypeSpecifierAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value3 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value3);
  // ::cxx::ElaboratedTypeSpecifierAST::templateLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value4);
  // ::cxx::ElaboratedTypeSpecifierAST::unqualifiedId
  cxx::UnqualifiedIdAST* value5 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value5);
  // ::cxx::ElaboratedTypeSpecifierAST::classKey
  ::cxx::TokenKind value6 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->classKey = std::move(value6);
  // ::cxx::ElaboratedTypeSpecifierAST::isTemplateIntroduced
  bool value7 = in.boolean();
  self->isTemplateIntroduced = std::move(value7);
  // ::cxx::ElaboratedTypeSpecifierAST::symbol
  cxx::Symbol* value8 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value8);
}

void SemanticDecoder::readAstDecltypeAutoSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DecltypeAutoSpecifierAST* self) {
  // ::cxx::DecltypeAutoSpecifierAST::decltypeLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->decltypeLoc = std::move(value1);
  // ::cxx::DecltypeAutoSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::DecltypeAutoSpecifierAST::autoLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->autoLoc = std::move(value3);
  // ::cxx::DecltypeAutoSpecifierAST::rparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value4);
}

void SemanticDecoder::readAstDecltypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DecltypeSpecifierAST* self) {
  // ::cxx::DecltypeSpecifierAST::decltypeLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->decltypeLoc = std::move(value1);
  // ::cxx::DecltypeSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::DecltypeSpecifierAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::DecltypeSpecifierAST::rparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value4);
  // ::cxx::DecltypeSpecifierAST::type
  const cxx::Type* value5 = typeAt(TypeRef{in.varU32()});
  self->type = std::move(value5);
}

void SemanticDecoder::readAstPlaceholderTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PlaceholderTypeSpecifierAST* self) {
  // ::cxx::PlaceholderTypeSpecifierAST::typeConstraint
  cxx::TypeConstraintAST* value1 =
      ast_cast<TypeConstraintAST>(astAt(AstRef{in.varU32()}));
  self->typeConstraint = std::move(value1);
  // ::cxx::PlaceholderTypeSpecifierAST::specifier
  cxx::SpecifierAST* value2 =
      ast_cast<SpecifierAST>(astAt(AstRef{in.varU32()}));
  self->specifier = std::move(value2);
}

void SemanticDecoder::readAstConstQualifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstQualifierAST* self) {
  // ::cxx::ConstQualifierAST::constLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->constLoc = std::move(value1);
}

void SemanticDecoder::readAstVolatileQualifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::VolatileQualifierAST* self) {
  // ::cxx::VolatileQualifierAST::volatileLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->volatileLoc = std::move(value1);
}

void SemanticDecoder::readAstAtomicQualifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AtomicQualifierAST* self) {
  // ::cxx::AtomicQualifierAST::atomicLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->atomicLoc = std::move(value1);
}

void SemanticDecoder::readAstRestrictQualifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::RestrictQualifierAST* self) {
  // ::cxx::RestrictQualifierAST::restrictLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->restrictLoc = std::move(value1);
}

void SemanticDecoder::readAstEnumSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::EnumSpecifierAST* self) {
  // ::cxx::EnumSpecifierAST::enumLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->enumLoc = std::move(value1);
  // ::cxx::EnumSpecifierAST::classLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->classLoc = std::move(value2);
  // ::cxx::EnumSpecifierAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value3 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value3);
  // ::cxx::EnumSpecifierAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value4 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value4);
  // ::cxx::EnumSpecifierAST::unqualifiedId
  cxx::NameIdAST* value5 = ast_cast<NameIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value5);
  // ::cxx::EnumSpecifierAST::colonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value6);
  // ::cxx::EnumSpecifierAST::typeSpecifierList
  cxx::List<cxx::SpecifierAST*>* value7 = readAstList<cxx::SpecifierAST>(in);
  self->typeSpecifierList = std::move(value7);
  // ::cxx::EnumSpecifierAST::lbraceLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->lbraceLoc = std::move(value8);
  // ::cxx::EnumSpecifierAST::enumeratorList
  cxx::List<cxx::EnumeratorAST*>* value9 = readAstList<cxx::EnumeratorAST>(in);
  self->enumeratorList = std::move(value9);
  // ::cxx::EnumSpecifierAST::commaLoc
  cxx::SourceLocation value10 = locationAt(LocationRef{in.varU32()});
  self->commaLoc = std::move(value10);
  // ::cxx::EnumSpecifierAST::rbraceLoc
  cxx::SourceLocation value11 = locationAt(LocationRef{in.varU32()});
  self->rbraceLoc = std::move(value11);
  // ::cxx::EnumSpecifierAST::symbol
  cxx::Symbol* value12 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value12);
}

void SemanticDecoder::readAstClassSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ClassSpecifierAST* self) {
  // ::cxx::ClassSpecifierAST::classLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->classLoc = std::move(value1);
  // ::cxx::ClassSpecifierAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value2 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value2);
  // ::cxx::ClassSpecifierAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value3 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value3);
  // ::cxx::ClassSpecifierAST::unqualifiedId
  cxx::UnqualifiedIdAST* value4 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value4);
  // ::cxx::ClassSpecifierAST::finalLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->finalLoc = std::move(value5);
  // ::cxx::ClassSpecifierAST::colonLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value6);
  // ::cxx::ClassSpecifierAST::baseSpecifierList
  cxx::List<cxx::BaseSpecifierAST*>* value7 =
      readAstList<cxx::BaseSpecifierAST>(in);
  self->baseSpecifierList = std::move(value7);
  // ::cxx::ClassSpecifierAST::lbraceLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->lbraceLoc = std::move(value8);
  // ::cxx::ClassSpecifierAST::declarationList
  cxx::List<cxx::DeclarationAST*>* value9 =
      readAstList<cxx::DeclarationAST>(in);
  self->declarationList = std::move(value9);
  // ::cxx::ClassSpecifierAST::rbraceLoc
  cxx::SourceLocation value10 = locationAt(LocationRef{in.varU32()});
  self->rbraceLoc = std::move(value10);
  // ::cxx::ClassSpecifierAST::classKey
  ::cxx::TokenKind value11 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->classKey = std::move(value11);
  // ::cxx::ClassSpecifierAST::symbol
  cxx::ClassSymbol* value12 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value12);
  // ::cxx::ClassSpecifierAST::isFinal
  bool value13 = in.boolean();
  self->isFinal = std::move(value13);
}

void SemanticDecoder::readAstTypenameSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypenameSpecifierAST* self) {
  // ::cxx::TypenameSpecifierAST::typenameLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->typenameLoc = std::move(value1);
  // ::cxx::TypenameSpecifierAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value2 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value2);
  // ::cxx::TypenameSpecifierAST::templateLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value3);
  // ::cxx::TypenameSpecifierAST::unqualifiedId
  cxx::UnqualifiedIdAST* value4 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value4);
  // ::cxx::TypenameSpecifierAST::isTemplateIntroduced
  bool value5 = in.boolean();
  self->isTemplateIntroduced = std::move(value5);
  // ::cxx::TypenameSpecifierAST::symbol
  cxx::Symbol* value6 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value6);
}

void SemanticDecoder::readAstSplicerTypeSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SplicerTypeSpecifierAST* self) {
  // ::cxx::SplicerTypeSpecifierAST::typenameLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->typenameLoc = std::move(value1);
  // ::cxx::SplicerTypeSpecifierAST::splicer
  cxx::SplicerAST* value2 = ast_cast<SplicerAST>(astAt(AstRef{in.varU32()}));
  self->splicer = std::move(value2);
}

void SemanticDecoder::readAstPointerOperatorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PointerOperatorAST* self) {
  // ::cxx::PointerOperatorAST::starLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->starLoc = std::move(value1);
  // ::cxx::PointerOperatorAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value2 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value2);
  // ::cxx::PointerOperatorAST::cvQualifierList
  cxx::List<cxx::SpecifierAST*>* value3 = readAstList<cxx::SpecifierAST>(in);
  self->cvQualifierList = std::move(value3);
}

void SemanticDecoder::readAstReferenceOperatorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ReferenceOperatorAST* self) {
  // ::cxx::ReferenceOperatorAST::refLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->refLoc = std::move(value1);
  // ::cxx::ReferenceOperatorAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value2 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value2);
  // ::cxx::ReferenceOperatorAST::refOp
  ::cxx::TokenKind value3 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->refOp = std::move(value3);
}

void SemanticDecoder::readAstPtrToMemberOperatorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PtrToMemberOperatorAST* self) {
  // ::cxx::PtrToMemberOperatorAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value1 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value1);
  // ::cxx::PtrToMemberOperatorAST::starLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->starLoc = std::move(value2);
  // ::cxx::PtrToMemberOperatorAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value3 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value3);
  // ::cxx::PtrToMemberOperatorAST::cvQualifierList
  cxx::List<cxx::SpecifierAST*>* value4 = readAstList<cxx::SpecifierAST>(in);
  self->cvQualifierList = std::move(value4);
}

void SemanticDecoder::readAstBitfieldDeclaratorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BitfieldDeclaratorAST* self) {
  // ::cxx::BitfieldDeclaratorAST::unqualifiedId
  cxx::NameIdAST* value1 = ast_cast<NameIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value1);
  // ::cxx::BitfieldDeclaratorAST::colonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value2);
  // ::cxx::BitfieldDeclaratorAST::sizeExpression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->sizeExpression = std::move(value3);
}

void SemanticDecoder::readAstParameterPackAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ParameterPackAST* self) {
  // ::cxx::ParameterPackAST::ellipsisLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value1);
  // ::cxx::ParameterPackAST::coreDeclarator
  cxx::CoreDeclaratorAST* value2 =
      ast_cast<CoreDeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->coreDeclarator = std::move(value2);
}

void SemanticDecoder::readAstIdDeclaratorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::IdDeclaratorAST* self) {
  // ::cxx::IdDeclaratorAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value1 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value1);
  // ::cxx::IdDeclaratorAST::templateLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value2);
  // ::cxx::IdDeclaratorAST::unqualifiedId
  cxx::UnqualifiedIdAST* value3 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value3);
  // ::cxx::IdDeclaratorAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value4 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value4);
  // ::cxx::IdDeclaratorAST::isTemplateIntroduced
  bool value5 = in.boolean();
  self->isTemplateIntroduced = std::move(value5);
}

void SemanticDecoder::readAstNestedDeclaratorAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NestedDeclaratorAST* self) {
  // ::cxx::NestedDeclaratorAST::lparenLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value1);
  // ::cxx::NestedDeclaratorAST::declarator
  cxx::DeclaratorAST* value2 =
      ast_cast<DeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->declarator = std::move(value2);
  // ::cxx::NestedDeclaratorAST::rparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value3);
}

void SemanticDecoder::readAstFunctionDeclaratorChunkAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::FunctionDeclaratorChunkAST* self) {
  // ::cxx::FunctionDeclaratorChunkAST::lparenLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value1);
  // ::cxx::FunctionDeclaratorChunkAST::parameterDeclarationClause
  cxx::ParameterDeclarationClauseAST* value2 =
      ast_cast<ParameterDeclarationClauseAST>(astAt(AstRef{in.varU32()}));
  self->parameterDeclarationClause = std::move(value2);
  // ::cxx::FunctionDeclaratorChunkAST::rparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value3);
  // ::cxx::FunctionDeclaratorChunkAST::cvQualifierList
  cxx::List<cxx::SpecifierAST*>* value4 = readAstList<cxx::SpecifierAST>(in);
  self->cvQualifierList = std::move(value4);
  // ::cxx::FunctionDeclaratorChunkAST::refLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->refLoc = std::move(value5);
  // ::cxx::FunctionDeclaratorChunkAST::exceptionSpecifier
  cxx::ExceptionSpecifierAST* value6 =
      ast_cast<ExceptionSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->exceptionSpecifier = std::move(value6);
  // ::cxx::FunctionDeclaratorChunkAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value7 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value7);
  // ::cxx::FunctionDeclaratorChunkAST::trailingReturnType
  cxx::TrailingReturnTypeAST* value8 =
      ast_cast<TrailingReturnTypeAST>(astAt(AstRef{in.varU32()}));
  self->trailingReturnType = std::move(value8);
  // ::cxx::FunctionDeclaratorChunkAST::refOp
  ::cxx::TokenKind value9 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->refOp = std::move(value9);
  // ::cxx::FunctionDeclaratorChunkAST::isFinal
  bool value10 = in.boolean();
  self->isFinal = std::move(value10);
  // ::cxx::FunctionDeclaratorChunkAST::isOverride
  bool value11 = in.boolean();
  self->isOverride = std::move(value11);
  // ::cxx::FunctionDeclaratorChunkAST::isPure
  bool value12 = in.boolean();
  self->isPure = std::move(value12);
}

void SemanticDecoder::readAstArrayDeclaratorChunkAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ArrayDeclaratorChunkAST* self) {
  // ::cxx::ArrayDeclaratorChunkAST::lbracketLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value1);
  // ::cxx::ArrayDeclaratorChunkAST::typeQualifierList
  cxx::List<cxx::SpecifierAST*>* value2 = readAstList<cxx::SpecifierAST>(in);
  self->typeQualifierList = std::move(value2);
  // ::cxx::ArrayDeclaratorChunkAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::ArrayDeclaratorChunkAST::rbracketLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value4);
  // ::cxx::ArrayDeclaratorChunkAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value5 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value5);
}

void SemanticDecoder::readAstNameIdAST([[maybe_unused]] ByteReader& in,
                                       [[maybe_unused]] cxx::NameIdAST* self) {
  // ::cxx::NameIdAST::identifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value1);
  // ::cxx::NameIdAST::identifier
  const cxx::Identifier* value2 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value2);
}

void SemanticDecoder::readAstDestructorIdAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DestructorIdAST* self) {
  // ::cxx::DestructorIdAST::tildeLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->tildeLoc = std::move(value1);
  // ::cxx::DestructorIdAST::id
  cxx::UnqualifiedIdAST* value2 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->id = std::move(value2);
}

void SemanticDecoder::readAstDecltypeIdAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DecltypeIdAST* self) {
  // ::cxx::DecltypeIdAST::decltypeSpecifier
  cxx::DecltypeSpecifierAST* value1 =
      ast_cast<DecltypeSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->decltypeSpecifier = std::move(value1);
}

void SemanticDecoder::readAstOperatorFunctionIdAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::OperatorFunctionIdAST* self) {
  // ::cxx::OperatorFunctionIdAST::operatorLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->operatorLoc = std::move(value1);
  // ::cxx::OperatorFunctionIdAST::opLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->opLoc = std::move(value2);
  // ::cxx::OperatorFunctionIdAST::openLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->openLoc = std::move(value3);
  // ::cxx::OperatorFunctionIdAST::closeLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->closeLoc = std::move(value4);
  // ::cxx::OperatorFunctionIdAST::op
  ::cxx::TokenKind value5 = static_cast<::cxx::TokenKind>(readEnum(in, 217));
  self->op = std::move(value5);
}

void SemanticDecoder::readAstLiteralOperatorIdAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::LiteralOperatorIdAST* self) {
  // ::cxx::LiteralOperatorIdAST::operatorLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->operatorLoc = std::move(value1);
  // ::cxx::LiteralOperatorIdAST::literalLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value2);
  // ::cxx::LiteralOperatorIdAST::identifierLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value3);
  // ::cxx::LiteralOperatorIdAST::literal
  const cxx::Literal* value4 = readStringLiteral(in);
  self->literal = std::move(value4);
  // ::cxx::LiteralOperatorIdAST::identifier
  const cxx::Identifier* value5 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value5);
}

void SemanticDecoder::readAstConversionFunctionIdAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConversionFunctionIdAST* self) {
  // ::cxx::ConversionFunctionIdAST::operatorLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->operatorLoc = std::move(value1);
  // ::cxx::ConversionFunctionIdAST::typeId
  cxx::TypeIdAST* value2 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value2);
}

void SemanticDecoder::readAstSimpleTemplateIdAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SimpleTemplateIdAST* self) {
  // ::cxx::SimpleTemplateIdAST::identifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value1);
  // ::cxx::SimpleTemplateIdAST::lessLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lessLoc = std::move(value2);
  // ::cxx::SimpleTemplateIdAST::templateArgumentList
  cxx::List<cxx::TemplateArgumentAST*>* value3 =
      readAstList<cxx::TemplateArgumentAST>(in);
  self->templateArgumentList = std::move(value3);
  // ::cxx::SimpleTemplateIdAST::greaterLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->greaterLoc = std::move(value4);
  // ::cxx::SimpleTemplateIdAST::identifier
  const cxx::Identifier* value5 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value5);
  // ::cxx::SimpleTemplateIdAST::symbol
  cxx::Symbol* value6 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value6);
}

void SemanticDecoder::readAstLiteralOperatorTemplateIdAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::LiteralOperatorTemplateIdAST* self) {
  // ::cxx::LiteralOperatorTemplateIdAST::literalOperatorId
  cxx::LiteralOperatorIdAST* value1 =
      ast_cast<LiteralOperatorIdAST>(astAt(AstRef{in.varU32()}));
  self->literalOperatorId = std::move(value1);
  // ::cxx::LiteralOperatorTemplateIdAST::lessLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lessLoc = std::move(value2);
  // ::cxx::LiteralOperatorTemplateIdAST::templateArgumentList
  cxx::List<cxx::TemplateArgumentAST*>* value3 =
      readAstList<cxx::TemplateArgumentAST>(in);
  self->templateArgumentList = std::move(value3);
  // ::cxx::LiteralOperatorTemplateIdAST::greaterLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->greaterLoc = std::move(value4);
}

void SemanticDecoder::readAstOperatorFunctionTemplateIdAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::OperatorFunctionTemplateIdAST* self) {
  // ::cxx::OperatorFunctionTemplateIdAST::operatorFunctionId
  cxx::OperatorFunctionIdAST* value1 =
      ast_cast<OperatorFunctionIdAST>(astAt(AstRef{in.varU32()}));
  self->operatorFunctionId = std::move(value1);
  // ::cxx::OperatorFunctionTemplateIdAST::lessLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lessLoc = std::move(value2);
  // ::cxx::OperatorFunctionTemplateIdAST::templateArgumentList
  cxx::List<cxx::TemplateArgumentAST*>* value3 =
      readAstList<cxx::TemplateArgumentAST>(in);
  self->templateArgumentList = std::move(value3);
  // ::cxx::OperatorFunctionTemplateIdAST::greaterLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->greaterLoc = std::move(value4);
}

void SemanticDecoder::readAstGlobalNestedNameSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::GlobalNestedNameSpecifierAST* self) {
  // ::cxx::NestedNameSpecifierAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::GlobalNestedNameSpecifierAST::scopeLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->scopeLoc = std::move(value2);
}

void SemanticDecoder::readAstSimpleNestedNameSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SimpleNestedNameSpecifierAST* self) {
  // ::cxx::NestedNameSpecifierAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::SimpleNestedNameSpecifierAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value2 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value2);
  // ::cxx::SimpleNestedNameSpecifierAST::identifierLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value3);
  // ::cxx::SimpleNestedNameSpecifierAST::identifier
  const cxx::Identifier* value4 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value4);
  // ::cxx::SimpleNestedNameSpecifierAST::scopeLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->scopeLoc = std::move(value5);
}

void SemanticDecoder::readAstDecltypeNestedNameSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DecltypeNestedNameSpecifierAST* self) {
  // ::cxx::NestedNameSpecifierAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::DecltypeNestedNameSpecifierAST::decltypeSpecifier
  cxx::DecltypeSpecifierAST* value2 =
      ast_cast<DecltypeSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->decltypeSpecifier = std::move(value2);
  // ::cxx::DecltypeNestedNameSpecifierAST::scopeLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->scopeLoc = std::move(value3);
}

void SemanticDecoder::readAstTemplateNestedNameSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TemplateNestedNameSpecifierAST* self) {
  // ::cxx::NestedNameSpecifierAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::TemplateNestedNameSpecifierAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value2 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value2);
  // ::cxx::TemplateNestedNameSpecifierAST::templateLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value3);
  // ::cxx::TemplateNestedNameSpecifierAST::templateId
  cxx::SimpleTemplateIdAST* value4 =
      ast_cast<SimpleTemplateIdAST>(astAt(AstRef{in.varU32()}));
  self->templateId = std::move(value4);
  // ::cxx::TemplateNestedNameSpecifierAST::scopeLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->scopeLoc = std::move(value5);
  // ::cxx::TemplateNestedNameSpecifierAST::isTemplateIntroduced
  bool value6 = in.boolean();
  self->isTemplateIntroduced = std::move(value6);
}

void SemanticDecoder::readAstDefaultFunctionBodyAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DefaultFunctionBodyAST* self) {
  // ::cxx::DefaultFunctionBodyAST::equalLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value1);
  // ::cxx::DefaultFunctionBodyAST::defaultLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->defaultLoc = std::move(value2);
  // ::cxx::DefaultFunctionBodyAST::semicolonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value3);
}

void SemanticDecoder::readAstCompoundStatementFunctionBodyAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CompoundStatementFunctionBodyAST* self) {
  // ::cxx::CompoundStatementFunctionBodyAST::colonLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value1);
  // ::cxx::CompoundStatementFunctionBodyAST::memInitializerList
  cxx::List<cxx::MemInitializerAST*>* value2 =
      readAstList<cxx::MemInitializerAST>(in);
  self->memInitializerList = std::move(value2);
  // ::cxx::CompoundStatementFunctionBodyAST::statement
  cxx::CompoundStatementAST* value3 =
      ast_cast<CompoundStatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value3);
}

void SemanticDecoder::readAstTryStatementFunctionBodyAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TryStatementFunctionBodyAST* self) {
  // ::cxx::TryStatementFunctionBodyAST::tryLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->tryLoc = std::move(value1);
  // ::cxx::TryStatementFunctionBodyAST::colonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->colonLoc = std::move(value2);
  // ::cxx::TryStatementFunctionBodyAST::memInitializerList
  cxx::List<cxx::MemInitializerAST*>* value3 =
      readAstList<cxx::MemInitializerAST>(in);
  self->memInitializerList = std::move(value3);
  // ::cxx::TryStatementFunctionBodyAST::statement
  cxx::CompoundStatementAST* value4 =
      ast_cast<CompoundStatementAST>(astAt(AstRef{in.varU32()}));
  self->statement = std::move(value4);
  // ::cxx::TryStatementFunctionBodyAST::handlerList
  cxx::List<cxx::HandlerAST*>* value5 = readAstList<cxx::HandlerAST>(in);
  self->handlerList = std::move(value5);
}

void SemanticDecoder::readAstDeleteFunctionBodyAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DeleteFunctionBodyAST* self) {
  // ::cxx::DeleteFunctionBodyAST::equalLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->equalLoc = std::move(value1);
  // ::cxx::DeleteFunctionBodyAST::deleteLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->deleteLoc = std::move(value2);
  // ::cxx::DeleteFunctionBodyAST::semicolonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value3);
}

void SemanticDecoder::readAstTypeTemplateArgumentAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeTemplateArgumentAST* self) {
  // ::cxx::TypeTemplateArgumentAST::typeId
  cxx::TypeIdAST* value1 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value1);
}

void SemanticDecoder::readAstExpressionTemplateArgumentAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ExpressionTemplateArgumentAST* self) {
  // ::cxx::ExpressionTemplateArgumentAST::expression
  cxx::ExpressionAST* value1 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value1);
}

void SemanticDecoder::readAstThrowExceptionSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ThrowExceptionSpecifierAST* self) {
  // ::cxx::ThrowExceptionSpecifierAST::throwLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->throwLoc = std::move(value1);
  // ::cxx::ThrowExceptionSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::ThrowExceptionSpecifierAST::rparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value3);
}

void SemanticDecoder::readAstNoexceptSpecifierAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NoexceptSpecifierAST* self) {
  // ::cxx::NoexceptSpecifierAST::noexceptLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->noexceptLoc = std::move(value1);
  // ::cxx::NoexceptSpecifierAST::lparenLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value2);
  // ::cxx::NoexceptSpecifierAST::expression
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value3);
  // ::cxx::NoexceptSpecifierAST::rparenLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value4);
}

void SemanticDecoder::readAstSimpleRequirementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SimpleRequirementAST* self) {
  // ::cxx::SimpleRequirementAST::expression
  cxx::ExpressionAST* value1 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value1);
  // ::cxx::SimpleRequirementAST::semicolonLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value2);
}

void SemanticDecoder::readAstCompoundRequirementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CompoundRequirementAST* self) {
  // ::cxx::CompoundRequirementAST::lbraceLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lbraceLoc = std::move(value1);
  // ::cxx::CompoundRequirementAST::expression
  cxx::ExpressionAST* value2 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value2);
  // ::cxx::CompoundRequirementAST::rbraceLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->rbraceLoc = std::move(value3);
  // ::cxx::CompoundRequirementAST::noexceptLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->noexceptLoc = std::move(value4);
  // ::cxx::CompoundRequirementAST::minusGreaterLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->minusGreaterLoc = std::move(value5);
  // ::cxx::CompoundRequirementAST::typeConstraint
  cxx::TypeConstraintAST* value6 =
      ast_cast<TypeConstraintAST>(astAt(AstRef{in.varU32()}));
  self->typeConstraint = std::move(value6);
  // ::cxx::CompoundRequirementAST::semicolonLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value7);
}

void SemanticDecoder::readAstTypeRequirementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeRequirementAST* self) {
  // ::cxx::TypeRequirementAST::typenameLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->typenameLoc = std::move(value1);
  // ::cxx::TypeRequirementAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value2 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value2);
  // ::cxx::TypeRequirementAST::templateLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->templateLoc = std::move(value3);
  // ::cxx::TypeRequirementAST::unqualifiedId
  cxx::UnqualifiedIdAST* value4 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value4);
  // ::cxx::TypeRequirementAST::semicolonLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value5);
  // ::cxx::TypeRequirementAST::isTemplateIntroduced
  bool value6 = in.boolean();
  self->isTemplateIntroduced = std::move(value6);
}

void SemanticDecoder::readAstNestedRequirementAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NestedRequirementAST* self) {
  // ::cxx::NestedRequirementAST::requiresLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->requiresLoc = std::move(value1);
  // ::cxx::NestedRequirementAST::expression
  cxx::ExpressionAST* value2 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value2);
  // ::cxx::NestedRequirementAST::semicolonLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->semicolonLoc = std::move(value3);
}

void SemanticDecoder::readAstNewParenInitializerAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NewParenInitializerAST* self) {
  // ::cxx::NewParenInitializerAST::lparenLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value1);
  // ::cxx::NewParenInitializerAST::expressionList
  cxx::List<cxx::ExpressionAST*>* value2 = readAstList<cxx::ExpressionAST>(in);
  self->expressionList = std::move(value2);
  // ::cxx::NewParenInitializerAST::rparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value3);
}

void SemanticDecoder::readAstNewBracedInitializerAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::NewBracedInitializerAST* self) {
  // ::cxx::NewBracedInitializerAST::bracedInitList
  cxx::BracedInitListAST* value1 =
      ast_cast<BracedInitListAST>(astAt(AstRef{in.varU32()}));
  self->bracedInitList = std::move(value1);
}

void SemanticDecoder::readAstParenMemInitializerAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ParenMemInitializerAST* self) {
  // ::cxx::MemInitializerAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::MemInitializerAST::constructor
  cxx::FunctionSymbol* value2 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->constructor = std::move(value2);
  // ::cxx::ParenMemInitializerAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value3 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value3);
  // ::cxx::ParenMemInitializerAST::unqualifiedId
  cxx::UnqualifiedIdAST* value4 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value4);
  // ::cxx::ParenMemInitializerAST::lparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value5);
  // ::cxx::ParenMemInitializerAST::expressionList
  cxx::List<cxx::ExpressionAST*>* value6 = readAstList<cxx::ExpressionAST>(in);
  self->expressionList = std::move(value6);
  // ::cxx::ParenMemInitializerAST::rparenLoc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value7);
  // ::cxx::ParenMemInitializerAST::ellipsisLoc
  cxx::SourceLocation value8 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value8);
}

void SemanticDecoder::readAstBracedMemInitializerAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::BracedMemInitializerAST* self) {
  // ::cxx::MemInitializerAST::symbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::MemInitializerAST::constructor
  cxx::FunctionSymbol* value2 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->constructor = std::move(value2);
  // ::cxx::BracedMemInitializerAST::nestedNameSpecifier
  cxx::NestedNameSpecifierAST* value3 =
      ast_cast<NestedNameSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->nestedNameSpecifier = std::move(value3);
  // ::cxx::BracedMemInitializerAST::unqualifiedId
  cxx::UnqualifiedIdAST* value4 =
      ast_cast<UnqualifiedIdAST>(astAt(AstRef{in.varU32()}));
  self->unqualifiedId = std::move(value4);
  // ::cxx::BracedMemInitializerAST::bracedInitList
  cxx::BracedInitListAST* value5 =
      ast_cast<BracedInitListAST>(astAt(AstRef{in.varU32()}));
  self->bracedInitList = std::move(value5);
  // ::cxx::BracedMemInitializerAST::ellipsisLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value6);
}

void SemanticDecoder::readAstThisLambdaCaptureAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ThisLambdaCaptureAST* self) {
  // ::cxx::ThisLambdaCaptureAST::thisLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->thisLoc = std::move(value1);
  // ::cxx::ThisLambdaCaptureAST::initializer
  cxx::ExpressionAST* value2 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value2);
  // ::cxx::ThisLambdaCaptureAST::symbol
  cxx::FieldSymbol* value3 =
      symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value3);
}

void SemanticDecoder::readAstDerefThisLambdaCaptureAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DerefThisLambdaCaptureAST* self) {
  // ::cxx::DerefThisLambdaCaptureAST::starLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->starLoc = std::move(value1);
  // ::cxx::DerefThisLambdaCaptureAST::thisLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->thisLoc = std::move(value2);
  // ::cxx::DerefThisLambdaCaptureAST::symbol
  cxx::FieldSymbol* value3 =
      symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value3);
}

void SemanticDecoder::readAstSimpleLambdaCaptureAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SimpleLambdaCaptureAST* self) {
  // ::cxx::SimpleLambdaCaptureAST::identifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value1);
  // ::cxx::SimpleLambdaCaptureAST::ellipsisLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value2);
  // ::cxx::SimpleLambdaCaptureAST::identifier
  const cxx::Identifier* value3 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value3);
  // ::cxx::SimpleLambdaCaptureAST::initializer
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value4);
  // ::cxx::SimpleLambdaCaptureAST::symbol
  cxx::FieldSymbol* value5 =
      symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value5);
}

void SemanticDecoder::readAstRefLambdaCaptureAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::RefLambdaCaptureAST* self) {
  // ::cxx::RefLambdaCaptureAST::ampLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->ampLoc = std::move(value1);
  // ::cxx::RefLambdaCaptureAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::RefLambdaCaptureAST::ellipsisLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value3);
  // ::cxx::RefLambdaCaptureAST::identifier
  const cxx::Identifier* value4 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value4);
  // ::cxx::RefLambdaCaptureAST::initializer
  cxx::ExpressionAST* value5 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value5);
  // ::cxx::RefLambdaCaptureAST::symbol
  cxx::FieldSymbol* value6 =
      symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value6);
}

void SemanticDecoder::readAstRefInitLambdaCaptureAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::RefInitLambdaCaptureAST* self) {
  // ::cxx::RefInitLambdaCaptureAST::ampLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->ampLoc = std::move(value1);
  // ::cxx::RefInitLambdaCaptureAST::ellipsisLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value2);
  // ::cxx::RefInitLambdaCaptureAST::identifierLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value3);
  // ::cxx::RefInitLambdaCaptureAST::initializer
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value4);
  // ::cxx::RefInitLambdaCaptureAST::identifier
  const cxx::Identifier* value5 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value5);
  // ::cxx::RefInitLambdaCaptureAST::symbol
  cxx::FieldSymbol* value6 =
      symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value6);
}

void SemanticDecoder::readAstInitLambdaCaptureAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::InitLambdaCaptureAST* self) {
  // ::cxx::InitLambdaCaptureAST::ellipsisLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value1);
  // ::cxx::InitLambdaCaptureAST::identifierLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value2);
  // ::cxx::InitLambdaCaptureAST::initializer
  cxx::ExpressionAST* value3 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->initializer = std::move(value3);
  // ::cxx::InitLambdaCaptureAST::identifier
  const cxx::Identifier* value4 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value4);
  // ::cxx::InitLambdaCaptureAST::symbol
  cxx::FieldSymbol* value5 =
      symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value5);
}

void SemanticDecoder::readAstEllipsisExceptionDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::EllipsisExceptionDeclarationAST* self) {
  // ::cxx::EllipsisExceptionDeclarationAST::ellipsisLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value1);
}

void SemanticDecoder::readAstTypeExceptionDeclarationAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TypeExceptionDeclarationAST* self) {
  // ::cxx::TypeExceptionDeclarationAST::attributeList
  cxx::List<cxx::AttributeSpecifierAST*>* value1 =
      readAstList<cxx::AttributeSpecifierAST>(in);
  self->attributeList = std::move(value1);
  // ::cxx::TypeExceptionDeclarationAST::typeSpecifierList
  cxx::List<cxx::SpecifierAST*>* value2 = readAstList<cxx::SpecifierAST>(in);
  self->typeSpecifierList = std::move(value2);
  // ::cxx::TypeExceptionDeclarationAST::declarator
  cxx::DeclaratorAST* value3 =
      ast_cast<DeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->declarator = std::move(value3);
  // ::cxx::TypeExceptionDeclarationAST::symbol
  cxx::VariableSymbol* value4 =
      symbol_cast<VariableSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->symbol = std::move(value4);
}

void SemanticDecoder::readAstCxxAttributeAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::CxxAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  const std::vector<cxx::Attribute>* value1 = readAttributes(in);
  self->attributes = std::move(value1);
  // ::cxx::CxxAttributeAST::lbracketLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->lbracketLoc = std::move(value2);
  // ::cxx::CxxAttributeAST::lbracket2Loc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lbracket2Loc = std::move(value3);
  // ::cxx::CxxAttributeAST::attributeUsingPrefix
  cxx::AttributeUsingPrefixAST* value4 =
      ast_cast<AttributeUsingPrefixAST>(astAt(AstRef{in.varU32()}));
  self->attributeUsingPrefix = std::move(value4);
  // ::cxx::CxxAttributeAST::attributeList
  cxx::List<cxx::AttributeAST*>* value5 = readAstList<cxx::AttributeAST>(in);
  self->attributeList = std::move(value5);
  // ::cxx::CxxAttributeAST::rbracketLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rbracketLoc = std::move(value6);
  // ::cxx::CxxAttributeAST::rbracket2Loc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rbracket2Loc = std::move(value7);
}

void SemanticDecoder::readAstGccAttributeAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::GccAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  const std::vector<cxx::Attribute>* value1 = readAttributes(in);
  self->attributes = std::move(value1);
  // ::cxx::GccAttributeAST::attributeLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->attributeLoc = std::move(value2);
  // ::cxx::GccAttributeAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::GccAttributeAST::lparen2Loc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->lparen2Loc = std::move(value4);
  // ::cxx::GccAttributeAST::attributeList
  cxx::List<cxx::AttributeAST*>* value5 = readAstList<cxx::AttributeAST>(in);
  self->attributeList = std::move(value5);
  // ::cxx::GccAttributeAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::GccAttributeAST::rparen2Loc
  cxx::SourceLocation value7 = locationAt(LocationRef{in.varU32()});
  self->rparen2Loc = std::move(value7);
}

void SemanticDecoder::readAstAlignasAttributeAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AlignasAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  const std::vector<cxx::Attribute>* value1 = readAttributes(in);
  self->attributes = std::move(value1);
  // ::cxx::AlignasAttributeAST::alignasLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->alignasLoc = std::move(value2);
  // ::cxx::AlignasAttributeAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::AlignasAttributeAST::expression
  cxx::ExpressionAST* value4 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value4);
  // ::cxx::AlignasAttributeAST::ellipsisLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value5);
  // ::cxx::AlignasAttributeAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::AlignasAttributeAST::isPack
  bool value7 = in.boolean();
  self->isPack = std::move(value7);
}

void SemanticDecoder::readAstAlignasTypeAttributeAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AlignasTypeAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  const std::vector<cxx::Attribute>* value1 = readAttributes(in);
  self->attributes = std::move(value1);
  // ::cxx::AlignasTypeAttributeAST::alignasLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->alignasLoc = std::move(value2);
  // ::cxx::AlignasTypeAttributeAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::AlignasTypeAttributeAST::typeId
  cxx::TypeIdAST* value4 = ast_cast<TypeIdAST>(astAt(AstRef{in.varU32()}));
  self->typeId = std::move(value4);
  // ::cxx::AlignasTypeAttributeAST::ellipsisLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->ellipsisLoc = std::move(value5);
  // ::cxx::AlignasTypeAttributeAST::rparenLoc
  cxx::SourceLocation value6 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value6);
  // ::cxx::AlignasTypeAttributeAST::isPack
  bool value7 = in.boolean();
  self->isPack = std::move(value7);
}

void SemanticDecoder::readAstAsmAttributeAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::AsmAttributeAST* self) {
  // ::cxx::AttributeSpecifierAST::attributes
  const std::vector<cxx::Attribute>* value1 = readAttributes(in);
  self->attributes = std::move(value1);
  // ::cxx::AsmAttributeAST::asmLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->asmLoc = std::move(value2);
  // ::cxx::AsmAttributeAST::lparenLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->lparenLoc = std::move(value3);
  // ::cxx::AsmAttributeAST::literalLoc
  cxx::SourceLocation value4 = locationAt(LocationRef{in.varU32()});
  self->literalLoc = std::move(value4);
  // ::cxx::AsmAttributeAST::rparenLoc
  cxx::SourceLocation value5 = locationAt(LocationRef{in.varU32()});
  self->rparenLoc = std::move(value5);
  // ::cxx::AsmAttributeAST::literal
  const cxx::Literal* value6 = readStringLiteral(in);
  self->literal = std::move(value6);
}

void SemanticDecoder::readAstScopedAttributeTokenAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ScopedAttributeTokenAST* self) {
  // ::cxx::ScopedAttributeTokenAST::attributeNamespaceLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->attributeNamespaceLoc = std::move(value1);
  // ::cxx::ScopedAttributeTokenAST::scopeLoc
  cxx::SourceLocation value2 = locationAt(LocationRef{in.varU32()});
  self->scopeLoc = std::move(value2);
  // ::cxx::ScopedAttributeTokenAST::identifierLoc
  cxx::SourceLocation value3 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value3);
  // ::cxx::ScopedAttributeTokenAST::attributeNamespace
  const cxx::Identifier* value4 = identifierAt(StringRef{in.varU32()});
  self->attributeNamespace = std::move(value4);
  // ::cxx::ScopedAttributeTokenAST::identifier
  const cxx::Identifier* value5 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value5);
}

void SemanticDecoder::readAstSimpleAttributeTokenAST(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::SimpleAttributeTokenAST* self) {
  // ::cxx::SimpleAttributeTokenAST::identifierLoc
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->identifierLoc = std::move(value1);
  // ::cxx::SimpleAttributeTokenAST::identifier
  const cxx::Identifier* value2 = identifierAt(StringRef{in.varU32()});
  self->identifier = std::move(value2);
}

void SemanticDecoder::readcxxAttribute([[maybe_unused]] ByteReader& in,
                                       [[maybe_unused]] cxx::Attribute* self) {
  // ::cxx::Attribute::attributeNamespace
  const cxx::Identifier* value1 = identifierAt(StringRef{in.varU32()});
  self->attributeNamespace = std::move(value1);
  // ::cxx::Attribute::name
  const cxx::Identifier* value2 = identifierAt(StringRef{in.varU32()});
  self->name = std::move(value2);
  // ::cxx::Attribute::arguments
  decltype(self->arguments) value3;
  {
    const auto count4 = in.varCount(1);
    for (std::uint32_t i5 = 0; ok() && i5 < count4; ++i5) {
      decltype(value3)::value_type element6 =
          identifierAt(StringRef{in.varU32()});
      value3.push_back(std::move(element6));
    }
  }
  self->arguments = std::move(value3);
}

void SemanticDecoder::readcxxMeta([[maybe_unused]] ByteReader& in,
                                  [[maybe_unused]] cxx::Meta* self) {
  // ::cxx::Meta::value
  decltype(self->value) value1;
  {
    const auto tag2 = in.u8();
    switch (tag2) {
      case 0: {
        std::variant_alternative_t<0, decltype(value1)> alternative3 =
            typeAt(TypeRef{in.varU32()});
        value1 = std::move(alternative3);
        break;
      }
      case 1: {
        std::variant_alternative_t<1, decltype(value1)> alternative4 =
            symbolAt(SymbolRef{in.varU32()});
        value1 = std::move(alternative4);
        break;
      }
      case 2: {
        std::variant_alternative_t<2, decltype(value1)> alternative5{};
        readcxxMetaConstExpr(in, &alternative5);
        value1 = std::move(alternative5);
        break;
      }
      default:
        fail("unknown variant alternative");
        break;
    }
  }
  self->value = std::move(value1);
}

void SemanticDecoder::readcxxMetaConstExpr(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::Meta::ConstExpr* self) {
  // ::cxx::Meta::ConstExpr::expression
  cxx::ExpressionAST* value1 =
      ast_cast<ExpressionAST>(astAt(AstRef{in.varU32()}));
  self->expression = std::move(value1);
  // ::cxx::Meta::ConstExpr::value
  cxx::ConstValue value2 = readConstValue(in);
  self->value = std::move(value2);
}

void SemanticDecoder::readcxxInitializerList(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::InitializerList* self) {
  // ::cxx::InitializerList::elements
  decltype(self->elements) value1;
  {
    const auto count2 = in.varCount(1);
    for (std::uint32_t i3 = 0; ok() && i3 < count2; ++i3) {
      decltype(value1)::value_type element4;
      {
        std::tuple_element_t<0, decltype(element4)> item5 = readConstValue(in);
        std::tuple_element_t<1, decltype(element4)> item6 =
            typeAt(TypeRef{in.varU32()});
        element4 = decltype(element4){std::move(item5), std::move(item6)};
      }
      value1.push_back(std::move(element4));
    }
  }
  self->elements = std::move(value1);
}

void SemanticDecoder::readcxxConstComplex(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::ConstComplex* self) {
  // ::cxx::ConstComplex::real_
  cxx::ConstValue value1 = readConstValue(in);
  self->setReal(std::move(value1));
  // ::cxx::ConstComplex::imag_
  cxx::ConstValue value2 = readConstValue(in);
  self->setImag(std::move(value2));
}

void SemanticDecoder::readcxxConstObject(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::ConstObject* self) {
  // ::cxx::ConstObject::type_
  const cxx::Type* value1 = typeAt(TypeRef{in.varU32()});
  self->setType(std::move(value1));
  // ::cxx::ConstObject::members_
  std::deque<cxx::ConstObject::Member> value2;
  {
    const auto count3 = in.varCount(1);
    for (std::uint32_t i4 = 0; ok() && i4 < count3; ++i4) {
      cxx::ConstObject::Member element5{};
      readcxxConstObjectMember(in, &element5);
      value2.push_back(std::move(element5));
    }
  }
  for (auto&& element6 : value2) {
    self->addMember(element6.symbol, element6.value);
  }
}

void SemanticDecoder::readcxxConstObjectMember(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstObject::Member* self) {
  // ::cxx::ConstObject::Member::symbol
  const cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value1);
  // ::cxx::ConstObject::Member::value
  cxx::ConstValue value2 = readConstValue(in);
  self->value = std::move(value2);
}

void SemanticDecoder::readcxxConstAddress(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::ConstAddress* self) {
  // ::cxx::ConstAddress::symbol_
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->setSymbol(std::move(value1));
  // ::cxx::ConstAddress::owner_
  std::shared_ptr<cxx::ConstObject> value2 =
      std::static_pointer_cast<cxx::ConstObject>(
          constantAt(ConstRef{in.varU32()}));
  self->setOwner(std::move(value2));
  // ::cxx::ConstAddress::string_
  const cxx::StringLiteral* value3 = readStringLiteral(in);
  self->setStringLiteral(std::move(value3));
  // ::cxx::ConstAddress::typeInfoFor_
  const cxx::Type* value4 = typeAt(TypeRef{in.varU32()});
  self->setTypeInfoFor(std::move(value4));
  // ::cxx::ConstAddress::offset_
  long long value5 = static_cast<long long>(in.varI64());
  self->setOffset(std::move(value5));
}

void SemanticDecoder::readcxxConstLabelAddress(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ConstLabelAddress* self) {
  // ::cxx::ConstLabelAddress::name_
  std::string value1{stringAt(StringRef{in.varU32()})};
  self->setName(std::move(value1));
}

void SemanticDecoder::readcxxTemplateSpecialization(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TemplateSpecialization* self) {
  // ::cxx::TemplateSpecialization::templateSymbol
  cxx::Symbol* value1 = symbolAt(SymbolRef{in.varU32()});
  self->templateSymbol = std::move(value1);
  // ::cxx::TemplateSpecialization::arguments
  decltype(self->arguments) value2;
  {
    const auto count3 = in.varCount(1);
    for (std::uint32_t i4 = 0; ok() && i4 < count3; ++i4) {
      decltype(value2)::value_type element5 = readTemplateArgument(in);
      value2.push_back(std::move(element5));
    }
  }
  self->arguments = std::move(value2);
  // ::cxx::TemplateSpecialization::symbol
  cxx::Symbol* value6 = symbolAt(SymbolRef{in.varU32()});
  self->symbol = std::move(value6);
  // ::cxx::TemplateSpecialization::instantiationErrors
  decltype(self->instantiationErrors) value7;
  {
    const auto count8 = in.varCount(1);
    for (std::uint32_t i9 = 0; ok() && i9 < count8; ++i9) {
      decltype(value7)::value_type element10{};
      readcxxInstantiationError(in, &element10);
      value7.push_back(std::move(element10));
    }
  }
  self->instantiationErrors = std::move(value7);
  // ::cxx::TemplateSpecialization::pendingArgumentList
  cxx::List<cxx::TemplateArgumentAST*>* value11 =
      readAstList<cxx::TemplateArgumentAST>(in);
  self->pendingArgumentList = std::move(value11);
  // ::cxx::TemplateSpecialization::pendingInstantiationLoc
  cxx::SourceLocation value12 = locationAt(LocationRef{in.varU32()});
  self->pendingInstantiationLoc = std::move(value12);
  // ::cxx::TemplateSpecialization::isPendingInstantiation
  bool value13 = in.boolean();
  self->isPendingInstantiation = std::move(value13);
}

void SemanticDecoder::readcxxInstantiationError(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::InstantiationError* self) {
  // ::cxx::InstantiationError::location
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->location = std::move(value1);
  // ::cxx::InstantiationError::message
  std::string value2{stringAt(StringRef{in.varU32()})};
  self->message = std::move(value2);
  // ::cxx::InstantiationError::severity
  static_assert(static_cast<std::uint32_t>(::cxx::Severity::Fatal) + 1 == 5);
  ::cxx::Severity value3 = static_cast<::cxx::Severity>(readEnum(in, 5));
  self->severity = std::move(value3);
}

void SemanticDecoder::readcxxTemplateFriendship(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::TemplateFriendship* self) {
  // ::cxx::TemplateFriendship::arguments
  decltype(self->arguments) value1;
  {
    const auto count2 = in.varCount(1);
    for (std::uint32_t i3 = 0; ok() && i3 < count2; ++i3) {
      decltype(value1)::value_type element4 = readTemplateArgument(in);
      value1.push_back(std::move(element4));
    }
  }
  self->arguments = std::move(value1);
  // ::cxx::TemplateFriendship::befriendingClass
  cxx::ClassSymbol* value5 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->befriendingClass = std::move(value5);
}

void SemanticDecoder::readcxxClassLayout(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::ClassLayout* self) {
  // ::cxx::ClassLayout::fields_
  std::remove_cvref_t<decltype(self->sortedFieldInfos())> value1;
  {
    const auto count2 = in.varCount(1);
    for (std::uint32_t i3 = 0; ok() && i3 < count2; ++i3) {
      decltype(value1)::value_type element4;
      {
        decltype(element4.first) first5 =
            symbol_cast<FieldSymbol>(symbolAt(SymbolRef{in.varU32()}));
        decltype(element4.second) second6{};
        readcxxClassLayoutMemberInfo(in, &second6);
        element4 = {std::move(first5), std::move(second6)};
      }
      value1.push_back(std::move(element4));
    }
  }
  for (auto&& element7 : value1) {
    self->setFieldInfo(element7.first, element7.second);
  }
  // ::cxx::ClassLayout::bases_
  std::remove_cvref_t<decltype(self->sortedBaseInfos())> value8;
  {
    const auto count9 = in.varCount(1);
    for (std::uint32_t i10 = 0; ok() && i10 < count9; ++i10) {
      decltype(value8)::value_type element11;
      {
        decltype(element11.first) first12 =
            symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
        decltype(element11.second) second13{};
        readcxxClassLayoutMemberInfo(in, &second13);
        element11 = {std::move(first12), std::move(second13)};
      }
      value8.push_back(std::move(element11));
    }
  }
  for (auto&& element14 : value8) {
    self->setBaseInfo(element14.first, element14.second);
  }
  // ::cxx::ClassLayout::virtualBases_
  std::vector<cxx::ClassSymbol*> value15;
  {
    const auto count16 = in.varCount(1);
    for (std::uint32_t i17 = 0; ok() && i17 < count16; ++i17) {
      cxx::ClassSymbol* element18 =
          symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
      value15.push_back(std::move(element18));
    }
  }
  for (auto&& element19 : value15) {
    self->addVirtualBase(element19);
  }
  // ::cxx::ClassLayout::padding_
  std::vector<cxx::ClassLayout::PaddingInfo> value20;
  {
    const auto count21 = in.varCount(1);
    for (std::uint32_t i22 = 0; ok() && i22 < count21; ++i22) {
      cxx::ClassLayout::PaddingInfo element23{};
      readcxxClassLayoutPaddingInfo(in, &element23);
      value20.push_back(std::move(element23));
    }
  }
  for (auto&& element24 : value20) {
    self->addPadding(element24.index, element24.offset, element24.sizeInBytes);
  }
  // ::cxx::ClassLayout::size_
  unsigned long long value25 = static_cast<unsigned long long>(in.varU64());
  self->setSize(std::move(value25));
  // ::cxx::ClassLayout::dataSize_
  unsigned long long value26 = static_cast<unsigned long long>(in.varU64());
  self->setDataSize(std::move(value26));
  // ::cxx::ClassLayout::alignment_
  unsigned long long value27 = static_cast<unsigned long long>(in.varU64());
  self->setAlignment(std::move(value27));
  // ::cxx::ClassLayout::nonVirtualSize_
  unsigned long long value28 = static_cast<unsigned long long>(in.varU64());
  self->setNonVirtualSize(std::move(value28));
  // ::cxx::ClassLayout::nonVirtualAlignment_
  unsigned long long value29 = static_cast<unsigned long long>(in.varU64());
  self->setNonVirtualAlignment(std::move(value29));
  // ::cxx::ClassLayout::vtableIndex_
  unsigned int value30 = static_cast<unsigned int>(in.varU32());
  self->setVtableIndex(std::move(value30));
  // ::cxx::ClassLayout::primaryBase_
  cxx::ClassSymbol* value31 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->setPrimaryBase(std::move(value31), self->primaryBaseIsVirtual());
  // ::cxx::ClassLayout::hasVtable_
  bool value32 = in.boolean();
  self->setHasVtable(std::move(value32));
  // ::cxx::ClassLayout::hasDirectVtable_
  bool value33 = in.boolean();
  self->setHasDirectVtable(std::move(value33));
  // ::cxx::ClassLayout::primaryBaseIsVirtual_
  bool value34 = in.boolean();
  self->setPrimaryBase(self->primaryBase(), std::move(value34));
  // ::cxx::ClassLayout::abiEmpty_
  bool value35 = in.boolean();
  self->setAbiEmpty(std::move(value35));
}

void SemanticDecoder::readcxxClassLayoutMemberInfo(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ClassLayout::MemberInfo* self) {
  // ::cxx::ClassLayout::MemberInfo::offset
  unsigned long long value1 = static_cast<unsigned long long>(in.varU64());
  self->offset = std::move(value1);
  // ::cxx::ClassLayout::MemberInfo::index
  unsigned int value2 = static_cast<unsigned int>(in.varU32());
  self->index = std::move(value2);
  // ::cxx::ClassLayout::MemberInfo::bitOffset
  unsigned int value3 = static_cast<unsigned int>(in.varU32());
  self->bitOffset = std::move(value3);
  // ::cxx::ClassLayout::MemberInfo::bitWidth
  unsigned int value4 = static_cast<unsigned int>(in.varU32());
  self->bitWidth = std::move(value4);
  // ::cxx::ClassLayout::MemberInfo::allocUnitSizeBytes
  unsigned int value5 = static_cast<unsigned int>(in.varU32());
  self->allocUnitSizeBytes = std::move(value5);
}

void SemanticDecoder::readcxxClassLayoutPaddingInfo(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::ClassLayout::PaddingInfo* self) {
  // ::cxx::ClassLayout::PaddingInfo::index
  unsigned int value1 = static_cast<unsigned int>(in.varU32());
  self->index = std::move(value1);
  // ::cxx::ClassLayout::PaddingInfo::offset
  unsigned long long value2 = static_cast<unsigned long long>(in.varU64());
  self->offset = std::move(value2);
  // ::cxx::ClassLayout::PaddingInfo::sizeInBytes
  unsigned long long value3 = static_cast<unsigned long long>(in.varU64());
  self->sizeInBytes = std::move(value3);
}

void SemanticDecoder::readcxxVTableLayout(
    [[maybe_unused]] ByteReader& in, [[maybe_unused]] cxx::VTableLayout* self) {
  // ::cxx::VTableLayout::primary
  cxx::VTableLayout::Group value1{};
  readcxxVTableLayoutGroup(in, &value1);
  self->primary = std::move(value1);
  // ::cxx::VTableLayout::secondary
  decltype(self->secondary) value2;
  {
    const auto count3 = in.varCount(1);
    for (std::uint32_t i4 = 0; ok() && i4 < count3; ++i4) {
      decltype(value2)::value_type element5{};
      readcxxVTableLayoutGroup(in, &element5);
      value2.push_back(std::move(element5));
    }
  }
  self->secondary = std::move(value2);
  // ::cxx::VTableLayout::keyFunction
  cxx::FunctionSymbol* value6 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->keyFunction = std::move(value6);
}

void SemanticDecoder::readcxxVTableLayoutGroup(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::VTableLayout::Group* self) {
  // ::cxx::VTableLayout::Group::base
  cxx::ClassSymbol* value1 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->base = std::move(value1);
  // ::cxx::VTableLayout::Group::offset
  unsigned long long value2 = static_cast<unsigned long long>(in.varU64());
  self->offset = std::move(value2);
  // ::cxx::VTableLayout::Group::vbaseOffsets
  decltype(self->vbaseOffsets) value3;
  {
    const auto count4 = in.varCount(1);
    for (std::uint32_t i5 = 0; ok() && i5 < count4; ++i5) {
      decltype(value3)::value_type element6;
      {
        decltype(element6.first) first7 =
            symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
        decltype(element6.second) second8 =
            static_cast<decltype(element6.second)>(in.varI64());
        element6 = {std::move(first7), std::move(second8)};
      }
      value3.push_back(std::move(element6));
    }
  }
  self->vbaseOffsets = std::move(value3);
  // ::cxx::VTableLayout::Group::vcallOffsets
  decltype(self->vcallOffsets) value9;
  {
    const auto count10 = in.varCount(1);
    for (std::uint32_t i11 = 0; ok() && i11 < count10; ++i11) {
      decltype(value9)::value_type element12;
      {
        decltype(element12.first) first13 =
            symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
        decltype(element12.second) second14 =
            static_cast<decltype(element12.second)>(in.varI64());
        element12 = {std::move(first13), std::move(second14)};
      }
      value9.push_back(std::move(element12));
    }
  }
  self->vcallOffsets = std::move(value9);
  // ::cxx::VTableLayout::Group::slots
  decltype(self->slots) value15;
  {
    const auto count16 = in.varCount(1);
    for (std::uint32_t i17 = 0; ok() && i17 < count16; ++i17) {
      decltype(value15)::value_type element18{};
      readcxxVTableLayoutSlot(in, &element18);
      value15.push_back(std::move(element18));
    }
  }
  self->slots = std::move(value15);
}

void SemanticDecoder::readcxxVTableLayoutSlot(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::VTableLayout::Slot* self) {
  // ::cxx::VTableLayout::Slot::function
  cxx::FunctionSymbol* value1 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->function = std::move(value1);
  // ::cxx::VTableLayout::Slot::kind
  static_assert(
      static_cast<std::uint32_t>(::cxx::VTableLayout::SlotKind::kDeletingDtor) +
          1 ==
      3);
  ::cxx::VTableLayout::SlotKind value2 =
      static_cast<::cxx::VTableLayout::SlotKind>(readEnum(in, 3));
  self->kind = std::move(value2);
  // ::cxx::VTableLayout::Slot::introducingFunction
  cxx::FunctionSymbol* value3 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->introducingFunction = std::move(value3);
  // ::cxx::VTableLayout::Slot::vcallBase
  cxx::ClassSymbol* value4 =
      symbol_cast<ClassSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->vcallBase = std::move(value4);
  // ::cxx::VTableLayout::Slot::thisAdjustment
  long long value5 = static_cast<long long>(in.varI64());
  self->thisAdjustment = std::move(value5);
  // ::cxx::VTableLayout::Slot::vcallOffsetIndex
  int value6 = static_cast<int>(in.varI32());
  self->vcallOffsetIndex = std::move(value6);
  // ::cxx::VTableLayout::Slot::usesVcallOffset
  bool value7 = in.boolean();
  self->usesVcallOffset = std::move(value7);
}

void SemanticDecoder::readcxxPendingBodyInstantiation(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PendingBodyInstantiation* self) {
  // ::cxx::PendingBodyInstantiation::originalDefinition
  cxx::FunctionDefinitionAST* value1 =
      ast_cast<FunctionDefinitionAST>(astAt(AstRef{in.varU32()}));
  self->originalDefinition = std::move(value1);
  // ::cxx::PendingBodyInstantiation::templateArguments
  decltype(self->templateArguments) value2;
  {
    const auto count3 = in.varCount(1);
    for (std::uint32_t i4 = 0; ok() && i4 < count3; ++i4) {
      decltype(value2)::value_type element5 = readTemplateArgument(in);
      value2.push_back(std::move(element5));
    }
  }
  self->templateArguments = std::move(value2);
  // ::cxx::PendingBodyInstantiation::parentScope
  cxx::ScopeSymbol* value6 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->parentScope = std::move(value6);
  // ::cxx::PendingBodyInstantiation::depth
  int value7 = static_cast<int>(in.varI32());
  self->depth = std::move(value7);
}

void SemanticDecoder::readcxxPendingExceptionSpecification(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PendingExceptionSpecification* self) {
  // ::cxx::PendingExceptionSpecification::original
  cxx::NoexceptSpecifierAST* value1 =
      ast_cast<NoexceptSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->original = std::move(value1);
  // ::cxx::PendingExceptionSpecification::instance
  cxx::NoexceptSpecifierAST* value2 =
      ast_cast<NoexceptSpecifierAST>(astAt(AstRef{in.varU32()}));
  self->instance = std::move(value2);
  // ::cxx::PendingExceptionSpecification::originalFunction
  cxx::FunctionSymbol* value3 =
      symbol_cast<FunctionSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->originalFunction = std::move(value3);
  // ::cxx::PendingExceptionSpecification::templateArguments
  decltype(self->templateArguments) value4;
  {
    const auto count5 = in.varCount(1);
    for (std::uint32_t i6 = 0; ok() && i6 < count5; ++i6) {
      decltype(value4)::value_type element7 = readTemplateArgument(in);
      value4.push_back(std::move(element7));
    }
  }
  self->templateArguments = std::move(value4);
  // ::cxx::PendingExceptionSpecification::parentScope
  cxx::ScopeSymbol* value8 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->parentScope = std::move(value8);
  // ::cxx::PendingExceptionSpecification::depth
  int value9 = static_cast<int>(in.varI32());
  self->depth = std::move(value9);
  // ::cxx::PendingExceptionSpecification::state
  static_assert(static_cast<std::uint32_t>(
                    ::cxx::PendingExceptionSpecificationState::kResolved) +
                    1 ==
                3);
  ::cxx::PendingExceptionSpecificationState value10 =
      static_cast<::cxx::PendingExceptionSpecificationState>(readEnum(in, 3));
  self->state = std::move(value10);
  // ::cxx::PendingExceptionSpecification::recursionDiagnosed
  bool value11 = in.boolean();
  self->recursionDiagnosed = std::move(value11);
}

void SemanticDecoder::readcxxPendingFieldInitializerInstantiation(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::PendingFieldInitializerInstantiation* self) {
  // ::cxx::PendingFieldInitializerInstantiation::unit
  cxx::TranslationUnit* value1 = unit();
  self->unit = std::move(value1);
  // ::cxx::PendingFieldInitializerInstantiation::pattern
  cxx::InitDeclaratorAST* value2 =
      ast_cast<InitDeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->pattern = std::move(value2);
  // ::cxx::PendingFieldInitializerInstantiation::instance
  cxx::InitDeclaratorAST* value3 =
      ast_cast<InitDeclaratorAST>(astAt(AstRef{in.varU32()}));
  self->instance = std::move(value3);
  // ::cxx::PendingFieldInitializerInstantiation::typeSpecifier
  cxx::SpecifierAST* value4 =
      ast_cast<SpecifierAST>(astAt(AstRef{in.varU32()}));
  self->typeSpecifier = std::move(value4);
  // ::cxx::PendingFieldInitializerInstantiation::templateArguments
  decltype(self->templateArguments) value5;
  {
    const auto count6 = in.varCount(1);
    for (std::uint32_t i7 = 0; ok() && i7 < count6; ++i7) {
      decltype(value5)::value_type element8 = readTemplateArgument(in);
      value5.push_back(std::move(element8));
    }
  }
  self->templateArguments = std::move(value5);
  // ::cxx::PendingFieldInitializerInstantiation::parentScope
  cxx::ScopeSymbol* value9 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->parentScope = std::move(value9);
  // ::cxx::PendingFieldInitializerInstantiation::depth
  int value10 = static_cast<int>(in.varI32());
  self->depth = std::move(value10);
}

void SemanticDecoder::readcxxDefaultInitializerContext(
    [[maybe_unused]] ByteReader& in,
    [[maybe_unused]] cxx::DefaultInitializerContext* self) {
  // ::cxx::DefaultInitializerContext::location
  cxx::SourceLocation value1 = locationAt(LocationRef{in.varU32()});
  self->location = std::move(value1);
  // ::cxx::DefaultInitializerContext::scope
  cxx::ScopeSymbol* value2 =
      symbol_cast<ScopeSymbol>(symbolAt(SymbolRef{in.varU32()}));
  self->scope = std::move(value2);
}

}  // namespace cxx
