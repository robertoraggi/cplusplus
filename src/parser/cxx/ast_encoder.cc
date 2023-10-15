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

#include <cxx/private/ast_encoder.h>

// cxx
#include <cxx-ast-flatbuffers/ast_generated.h>
#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/private/format.h>
#include <cxx/translation_unit.h>

#include <algorithm>

namespace cxx {

auto ASTEncoder::operator()(TranslationUnit* unit)
    -> std::span<const std::uint8_t> {
  if (!unit) return {};
  Table<Identifier> identifiers;
  Table<CharLiteral> charLiterals;
  Table<StringLiteral> stringLiterals;
  Table<IntegerLiteral> integerLiterals;
  Table<FloatLiteral> floatLiterals;
  SourceFiles sourceFiles;
  SourceLines sourceLines;

  std::swap(unit_, unit);
  std::swap(identifiers_, identifiers);
  std::swap(charLiterals_, charLiterals);
  std::swap(stringLiterals_, stringLiterals);
  std::swap(integerLiterals_, integerLiterals);
  std::swap(floatLiterals_, floatLiterals);
  std::swap(sourceFiles_, sourceFiles);
  std::swap(sourceLines_, sourceLines);

  auto [unitOffset, unitType] = acceptUnit(unit_->ast());

  auto file_name = fbb_.CreateString(unit_->fileName());

  io::SerializedUnitBuilder builder{fbb_};
  builder.add_unit(unitOffset);
  builder.add_unit_type(static_cast<io::Unit>(unitType));
  builder.add_file_name(file_name);

  std::swap(unit_, unit);
  std::swap(identifiers_, identifiers);
  std::swap(charLiterals_, charLiterals);
  std::swap(stringLiterals_, stringLiterals);
  std::swap(integerLiterals_, integerLiterals);
  std::swap(floatLiterals_, floatLiterals);
  std::swap(sourceFiles_, sourceFiles);
  std::swap(sourceLines_, sourceLines);

  fbb_.Finish(builder.Finish(), io::SerializedUnitIdentifier());

  return std::span{fbb_.GetBufferPointer(), fbb_.GetSize()};
}

auto ASTEncoder::accept(AST* ast) -> flatbuffers::Offset<> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::swap(offset_, offset);
  ast->accept(this);
  std::swap(offset_, offset);
  return offset;
}

auto ASTEncoder::encodeSourceLocation(const SourceLocation& loc)
    -> flatbuffers::Offset<> {
  if (!loc) {
    return {};
  }

  std::string_view fileName;
  std::uint32_t line = 0, column = 0;
  unit_->getTokenStartPosition(loc, &line, &column, &fileName);

  flatbuffers::Offset<io::SourceLine> sourceLineOffset;

  auto key = std::tuple(fileName, line);

  if (sourceLines_.contains(key)) {
    sourceLineOffset = sourceLines_.at(key).o;
  } else {
    flatbuffers::Offset<flatbuffers::String> fileNameOffset;

    if (sourceFiles_.contains(fileName)) {
      fileNameOffset = sourceFiles_.at(fileName);
    } else {
      fileNameOffset = fbb_.CreateString(fileName);
      sourceFiles_.emplace(fileName, fileNameOffset.o);
    }

    io::SourceLineBuilder sourceLineBuilder{fbb_};
    sourceLineBuilder.add_file_name(fileNameOffset);
    sourceLineBuilder.add_line(line);
    sourceLineOffset = sourceLineBuilder.Finish();
    sourceLines_.emplace(std::move(key), sourceLineOffset.o);
  }

  io::SourceLocationBuilder sourceLocationBuilder{fbb_};
  sourceLocationBuilder.add_source_line(sourceLineOffset);
  sourceLocationBuilder.add_column(column);

  auto offset = sourceLocationBuilder.Finish();

  return offset.Union();
}

auto ASTEncoder::acceptUnit(UnitAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptDeclaration(DeclarationAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptStatement(StatementAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptExpression(ExpressionAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptTemplateParameter(TemplateParameterAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptSpecifier(SpecifierAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptPtrOperator(PtrOperatorAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptCoreDeclarator(CoreDeclaratorAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptDeclaratorChunk(DeclaratorChunkAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptUnqualifiedId(UnqualifiedIdAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptNestedNameSpecifier(NestedNameSpecifierAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptFunctionBody(FunctionBodyAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptTemplateArgument(TemplateArgumentAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptExceptionSpecifier(ExceptionSpecifierAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptRequirement(RequirementAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptNewInitializer(NewInitializerAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptMemInitializer(MemInitializerAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptLambdaCapture(LambdaCaptureAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptExceptionDeclaration(ExceptionDeclarationAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptAttributeSpecifier(AttributeSpecifierAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

auto ASTEncoder::acceptAttributeToken(AttributeTokenAST* ast)
    -> std::tuple<flatbuffers::Offset<>, std::uint32_t> {
  if (!ast) return {};
  flatbuffers::Offset<> offset;
  std::uint32_t type = 0;
  std::swap(offset, offset_);
  std::swap(type, type_);
  ast->accept(this);
  std::swap(offset, offset_);
  std::swap(type, type_);
  return {offset, type};
}

void ASTEncoder::visit(TranslationUnitAST* ast) {
  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto it = ast->declarationList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptDeclaration(it->value);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  io::TranslationUnit::Builder builder{fbb_};
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);

  offset_ = builder.Finish().Union();
  type_ = io::Unit_TranslationUnit;
}

void ASTEncoder::visit(ModuleUnitAST* ast) {
  const auto globalModuleFragment = accept(ast->globalModuleFragment);

  const auto moduleDeclaration = accept(ast->moduleDeclaration);

  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto it = ast->declarationList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptDeclaration(it->value);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  const auto privateModuleFragment = accept(ast->privateModuleFragment);

  io::ModuleUnit::Builder builder{fbb_};
  builder.add_global_module_fragment(globalModuleFragment.o);
  builder.add_module_declaration(moduleDeclaration.o);
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);
  builder.add_private_module_fragment(privateModuleFragment.o);

  offset_ = builder.Finish().Union();
  type_ = io::Unit_ModuleUnit;
}

void ASTEncoder::visit(SimpleDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    declSpecifierListOffsets.push_back(offset);
    declSpecifierListTypes.push_back(type);
  }

  auto declSpecifierListOffsetsVector =
      fbb_.CreateVector(declSpecifierListOffsets);
  auto declSpecifierListTypesVector = fbb_.CreateVector(declSpecifierListTypes);

  std::vector<flatbuffers::Offset<io::InitDeclarator>>
      initDeclaratorListOffsets;
  for (auto it = ast->initDeclaratorList; it; it = it->next) {
    if (!it->value) continue;
    initDeclaratorListOffsets.emplace_back(accept(it->value).o);
  }

  auto initDeclaratorListOffsetsVector =
      fbb_.CreateVector(initDeclaratorListOffsets);

  const auto requiresClause = accept(ast->requiresClause);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::SimpleDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_decl_specifier_list(declSpecifierListOffsetsVector);
  builder.add_decl_specifier_list_type(declSpecifierListTypesVector);
  builder.add_init_declarator_list(initDeclaratorListOffsetsVector);
  builder.add_requires_clause(requiresClause.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_SimpleDeclaration;
}

void ASTEncoder::visit(AsmDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<io::AsmQualifier>> asmQualifierListOffsets;
  for (auto it = ast->asmQualifierList; it; it = it->next) {
    if (!it->value) continue;
    asmQualifierListOffsets.emplace_back(accept(it->value).o);
  }

  auto asmQualifierListOffsetsVector =
      fbb_.CreateVector(asmQualifierListOffsets);

  auto asmLoc = encodeSourceLocation(ast->asmLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  std::vector<flatbuffers::Offset<io::AsmOperand>> outputOperandListOffsets;
  for (auto it = ast->outputOperandList; it; it = it->next) {
    if (!it->value) continue;
    outputOperandListOffsets.emplace_back(accept(it->value).o);
  }

  auto outputOperandListOffsetsVector =
      fbb_.CreateVector(outputOperandListOffsets);

  std::vector<flatbuffers::Offset<io::AsmOperand>> inputOperandListOffsets;
  for (auto it = ast->inputOperandList; it; it = it->next) {
    if (!it->value) continue;
    inputOperandListOffsets.emplace_back(accept(it->value).o);
  }

  auto inputOperandListOffsetsVector =
      fbb_.CreateVector(inputOperandListOffsets);

  std::vector<flatbuffers::Offset<io::AsmClobber>> clobberListOffsets;
  for (auto it = ast->clobberList; it; it = it->next) {
    if (!it->value) continue;
    clobberListOffsets.emplace_back(accept(it->value).o);
  }

  auto clobberListOffsetsVector = fbb_.CreateVector(clobberListOffsets);

  std::vector<flatbuffers::Offset<io::AsmGotoLabel>> gotoLabelListOffsets;
  for (auto it = ast->gotoLabelList; it; it = it->next) {
    if (!it->value) continue;
    gotoLabelListOffsets.emplace_back(accept(it->value).o);
  }

  auto gotoLabelListOffsetsVector = fbb_.CreateVector(gotoLabelListOffsets);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::AsmDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_asm_qualifier_list(asmQualifierListOffsetsVector);
  builder.add_asm_loc(asmLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_literal_loc(literalLoc.o);
  builder.add_output_operand_list(outputOperandListOffsetsVector);
  builder.add_input_operand_list(inputOperandListOffsetsVector);
  builder.add_clobber_list(clobberListOffsetsVector);
  builder.add_goto_label_list(gotoLabelListOffsetsVector);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmDeclaration;
}

void ASTEncoder::visit(NamespaceAliasDefinitionAST* ast) {
  auto namespaceLoc = encodeSourceLocation(ast->namespaceLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto unqualifiedId = accept(ast->unqualifiedId);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::NamespaceAliasDefinition::Builder builder{fbb_};
  builder.add_namespace_loc(namespaceLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_equal_loc(equalLoc.o);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_semicolon_loc(semicolonLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_NamespaceAliasDefinition;
}

void ASTEncoder::visit(UsingDeclarationAST* ast) {
  auto usingLoc = encodeSourceLocation(ast->usingLoc);

  std::vector<flatbuffers::Offset<io::UsingDeclarator>>
      usingDeclaratorListOffsets;
  for (auto it = ast->usingDeclaratorList; it; it = it->next) {
    if (!it->value) continue;
    usingDeclaratorListOffsets.emplace_back(accept(it->value).o);
  }

  auto usingDeclaratorListOffsetsVector =
      fbb_.CreateVector(usingDeclaratorListOffsets);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::UsingDeclaration::Builder builder{fbb_};
  builder.add_using_loc(usingLoc.o);
  builder.add_using_declarator_list(usingDeclaratorListOffsetsVector);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_UsingDeclaration;
}

void ASTEncoder::visit(UsingEnumDeclarationAST* ast) {
  auto usingLoc = encodeSourceLocation(ast->usingLoc);

  const auto enumTypeSpecifier = accept(ast->enumTypeSpecifier);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::UsingEnumDeclaration::Builder builder{fbb_};
  builder.add_using_loc(usingLoc.o);
  builder.add_enum_type_specifier(enumTypeSpecifier.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_UsingEnumDeclaration;
}

void ASTEncoder::visit(UsingDirectiveAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  auto usingLoc = encodeSourceLocation(ast->usingLoc);

  auto namespaceLoc = encodeSourceLocation(ast->namespaceLoc);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto unqualifiedId = accept(ast->unqualifiedId);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::UsingDirective::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_using_loc(usingLoc.o);
  builder.add_namespace_loc(namespaceLoc.o);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_UsingDirective;
}

void ASTEncoder::visit(StaticAssertDeclarationAST* ast) {
  auto staticAssertLoc = encodeSourceLocation(ast->staticAssertLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto commaLoc = encodeSourceLocation(ast->commaLoc);

  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::StaticAssertDeclaration::Builder builder{fbb_};
  builder.add_static_assert_loc(staticAssertLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_comma_loc(commaLoc.o);
  builder.add_literal_loc(literalLoc.o);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_StaticAssertDeclaration;
}

void ASTEncoder::visit(AliasDeclarationAST* ast) {
  auto usingLoc = encodeSourceLocation(ast->usingLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  const auto typeId = accept(ast->typeId);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::AliasDeclaration::Builder builder{fbb_};
  builder.add_using_loc(usingLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_equal_loc(equalLoc.o);
  builder.add_type_id(typeId.o);
  builder.add_semicolon_loc(semicolonLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AliasDeclaration;
}

void ASTEncoder::visit(OpaqueEnumDeclarationAST* ast) {
  auto enumLoc = encodeSourceLocation(ast->enumLoc);

  auto classLoc = encodeSourceLocation(ast->classLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto unqualifiedId = accept(ast->unqualifiedId);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  auto emicolonLoc = encodeSourceLocation(ast->emicolonLoc);

  io::OpaqueEnumDeclaration::Builder builder{fbb_};
  builder.add_enum_loc(enumLoc.o);
  builder.add_class_loc(classLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_colon_loc(colonLoc.o);
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_emicolon_loc(emicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_OpaqueEnumDeclaration;
}

void ASTEncoder::visit(FunctionDefinitionAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    declSpecifierListOffsets.push_back(offset);
    declSpecifierListTypes.push_back(type);
  }

  auto declSpecifierListOffsetsVector =
      fbb_.CreateVector(declSpecifierListOffsets);
  auto declSpecifierListTypesVector = fbb_.CreateVector(declSpecifierListTypes);

  const auto declarator = accept(ast->declarator);

  const auto requiresClause = accept(ast->requiresClause);

  const auto [functionBody, functionBodyType] =
      acceptFunctionBody(ast->functionBody);

  io::FunctionDefinition::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_decl_specifier_list(declSpecifierListOffsetsVector);
  builder.add_decl_specifier_list_type(declSpecifierListTypesVector);
  builder.add_declarator(declarator.o);
  builder.add_requires_clause(requiresClause.o);
  builder.add_function_body(functionBody);
  builder.add_function_body_type(
      static_cast<io::FunctionBody>(functionBodyType));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_FunctionDefinition;
}

void ASTEncoder::visit(TemplateDeclarationAST* ast) {
  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  auto lessLoc = encodeSourceLocation(ast->lessLoc);

  std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
  std::vector<std::underlying_type_t<io::TemplateParameter>>
      templateParameterListTypes;

  for (auto it = ast->templateParameterList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptTemplateParameter(it->value);
    templateParameterListOffsets.push_back(offset);
    templateParameterListTypes.push_back(type);
  }

  auto templateParameterListOffsetsVector =
      fbb_.CreateVector(templateParameterListOffsets);
  auto templateParameterListTypesVector =
      fbb_.CreateVector(templateParameterListTypes);

  auto greaterLoc = encodeSourceLocation(ast->greaterLoc);

  const auto requiresClause = accept(ast->requiresClause);

  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  io::TemplateDeclaration::Builder builder{fbb_};
  builder.add_template_loc(templateLoc.o);
  builder.add_less_loc(lessLoc.o);
  builder.add_template_parameter_list(templateParameterListOffsetsVector);
  builder.add_template_parameter_list_type(templateParameterListTypesVector);
  builder.add_greater_loc(greaterLoc.o);
  builder.add_requires_clause(requiresClause.o);
  builder.add_declaration(declaration);
  builder.add_declaration_type(static_cast<io::Declaration>(declarationType));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_TemplateDeclaration;
}

void ASTEncoder::visit(ConceptDefinitionAST* ast) {
  auto conceptLoc = encodeSourceLocation(ast->conceptLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::ConceptDefinition::Builder builder{fbb_};
  builder.add_concept_loc(conceptLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_equal_loc(equalLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(semicolonLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ConceptDefinition;
}

void ASTEncoder::visit(DeductionGuideAST* ast) {
  const auto [explicitSpecifier, explicitSpecifierType] =
      acceptSpecifier(ast->explicitSpecifier);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  auto arrowLoc = encodeSourceLocation(ast->arrowLoc);

  const auto templateId = accept(ast->templateId);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::DeductionGuide::Builder builder{fbb_};
  builder.add_explicit_specifier(explicitSpecifier);
  builder.add_explicit_specifier_type(
      static_cast<io::Specifier>(explicitSpecifierType));
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_arrow_loc(arrowLoc.o);
  builder.add_template_id(templateId.o);
  builder.add_semicolon_loc(semicolonLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_DeductionGuide;
}

void ASTEncoder::visit(ExplicitInstantiationAST* ast) {
  auto externLoc = encodeSourceLocation(ast->externLoc);

  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  io::ExplicitInstantiation::Builder builder{fbb_};
  builder.add_extern_loc(externLoc.o);
  builder.add_template_loc(templateLoc.o);
  builder.add_declaration(declaration);
  builder.add_declaration_type(static_cast<io::Declaration>(declarationType));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ExplicitInstantiation;
}

void ASTEncoder::visit(ExportDeclarationAST* ast) {
  auto exportLoc = encodeSourceLocation(ast->exportLoc);

  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  io::ExportDeclaration::Builder builder{fbb_};
  builder.add_export_loc(exportLoc.o);
  builder.add_declaration(declaration);
  builder.add_declaration_type(static_cast<io::Declaration>(declarationType));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ExportDeclaration;
}

void ASTEncoder::visit(ExportCompoundDeclarationAST* ast) {
  auto exportLoc = encodeSourceLocation(ast->exportLoc);

  auto lbraceLoc = encodeSourceLocation(ast->lbraceLoc);

  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto it = ast->declarationList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptDeclaration(it->value);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  auto rbraceLoc = encodeSourceLocation(ast->rbraceLoc);

  io::ExportCompoundDeclaration::Builder builder{fbb_};
  builder.add_export_loc(exportLoc.o);
  builder.add_lbrace_loc(lbraceLoc.o);
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);
  builder.add_rbrace_loc(rbraceLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ExportCompoundDeclaration;
}

void ASTEncoder::visit(LinkageSpecificationAST* ast) {
  auto externLoc = encodeSourceLocation(ast->externLoc);

  auto stringliteralLoc = encodeSourceLocation(ast->stringliteralLoc);

  auto lbraceLoc = encodeSourceLocation(ast->lbraceLoc);

  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto it = ast->declarationList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptDeclaration(it->value);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  auto rbraceLoc = encodeSourceLocation(ast->rbraceLoc);

  flatbuffers::Offset<flatbuffers::String> stringLiteral;
  if (ast->stringLiteral) {
    if (stringLiterals_.contains(ast->stringLiteral)) {
      stringLiteral = stringLiterals_.at(ast->stringLiteral);
    } else {
      stringLiteral = fbb_.CreateString(ast->stringLiteral->value());
      stringLiterals_.emplace(ast->stringLiteral, stringLiteral);
    }
  }

  io::LinkageSpecification::Builder builder{fbb_};
  builder.add_extern_loc(externLoc.o);
  builder.add_stringliteral_loc(stringliteralLoc.o);
  builder.add_lbrace_loc(lbraceLoc.o);
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);
  builder.add_rbrace_loc(rbraceLoc.o);
  if (ast->stringLiteral) {
    builder.add_string_literal(stringLiteral);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_LinkageSpecification;
}

void ASTEncoder::visit(NamespaceDefinitionAST* ast) {
  auto inlineLoc = encodeSourceLocation(ast->inlineLoc);

  auto namespaceLoc = encodeSourceLocation(ast->namespaceLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<io::NestedNamespaceSpecifier>>
      nestedNamespaceSpecifierListOffsets;
  for (auto it = ast->nestedNamespaceSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    nestedNamespaceSpecifierListOffsets.emplace_back(accept(it->value).o);
  }

  auto nestedNamespaceSpecifierListOffsetsVector =
      fbb_.CreateVector(nestedNamespaceSpecifierListOffsets);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  std::vector<flatbuffers::Offset<>> extraAttributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      extraAttributeListTypes;

  for (auto it = ast->extraAttributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    extraAttributeListOffsets.push_back(offset);
    extraAttributeListTypes.push_back(type);
  }

  auto extraAttributeListOffsetsVector =
      fbb_.CreateVector(extraAttributeListOffsets);
  auto extraAttributeListTypesVector =
      fbb_.CreateVector(extraAttributeListTypes);

  auto lbraceLoc = encodeSourceLocation(ast->lbraceLoc);

  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto it = ast->declarationList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptDeclaration(it->value);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  auto rbraceLoc = encodeSourceLocation(ast->rbraceLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::NamespaceDefinition::Builder builder{fbb_};
  builder.add_inline_loc(inlineLoc.o);
  builder.add_namespace_loc(namespaceLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_namespace_specifier_list(
      nestedNamespaceSpecifierListOffsetsVector);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_extra_attribute_list(extraAttributeListOffsetsVector);
  builder.add_extra_attribute_list_type(extraAttributeListTypesVector);
  builder.add_lbrace_loc(lbraceLoc.o);
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);
  builder.add_rbrace_loc(rbraceLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_NamespaceDefinition;
}

void ASTEncoder::visit(EmptyDeclarationAST* ast) {
  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::EmptyDeclaration::Builder builder{fbb_};
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_EmptyDeclaration;
}

void ASTEncoder::visit(AttributeDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::AttributeDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AttributeDeclaration;
}

void ASTEncoder::visit(ModuleImportDeclarationAST* ast) {
  auto importLoc = encodeSourceLocation(ast->importLoc);

  const auto importName = accept(ast->importName);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::ModuleImportDeclaration::Builder builder{fbb_};
  builder.add_import_loc(importLoc.o);
  builder.add_import_name(importName.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ModuleImportDeclaration;
}

void ASTEncoder::visit(ParameterDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  auto thisLoc = encodeSourceLocation(ast->thisLoc);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  const auto declarator = accept(ast->declarator);

  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::ParameterDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_this_loc(thisLoc.o);
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_declarator(declarator.o);
  builder.add_equal_loc(equalLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ParameterDeclaration;
}

void ASTEncoder::visit(AccessDeclarationAST* ast) {
  auto accessLoc = encodeSourceLocation(ast->accessLoc);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  io::AccessDeclaration::Builder builder{fbb_};
  builder.add_access_loc(accessLoc.o);
  builder.add_colon_loc(colonLoc.o);
  builder.add_access_specifier(
      static_cast<std::uint32_t>(ast->accessSpecifier));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AccessDeclaration;
}

void ASTEncoder::visit(ForRangeDeclarationAST* ast) {
  io::ForRangeDeclaration::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ForRangeDeclaration;
}

void ASTEncoder::visit(StructuredBindingDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    declSpecifierListOffsets.push_back(offset);
    declSpecifierListTypes.push_back(type);
  }

  auto declSpecifierListOffsetsVector =
      fbb_.CreateVector(declSpecifierListOffsets);
  auto declSpecifierListTypesVector = fbb_.CreateVector(declSpecifierListTypes);

  auto refQualifierLoc = encodeSourceLocation(ast->refQualifierLoc);

  auto lbracketLoc = encodeSourceLocation(ast->lbracketLoc);

  std::vector<flatbuffers::Offset<io::NameId>> bindingListOffsets;
  for (auto it = ast->bindingList; it; it = it->next) {
    if (!it->value) continue;
    bindingListOffsets.emplace_back(accept(it->value).o);
  }

  auto bindingListOffsetsVector = fbb_.CreateVector(bindingListOffsets);

  auto rbracketLoc = encodeSourceLocation(ast->rbracketLoc);

  const auto [initializer, initializerType] =
      acceptExpression(ast->initializer);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::StructuredBindingDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_decl_specifier_list(declSpecifierListOffsetsVector);
  builder.add_decl_specifier_list_type(declSpecifierListTypesVector);
  builder.add_ref_qualifier_loc(refQualifierLoc.o);
  builder.add_lbracket_loc(lbracketLoc.o);
  builder.add_binding_list(bindingListOffsetsVector);
  builder.add_rbracket_loc(rbracketLoc.o);
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_StructuredBindingDeclaration;
}

void ASTEncoder::visit(AsmOperandAST* ast) {
  auto lbracketLoc = encodeSourceLocation(ast->lbracketLoc);

  auto symbolicNameLoc = encodeSourceLocation(ast->symbolicNameLoc);

  auto rbracketLoc = encodeSourceLocation(ast->rbracketLoc);

  auto constraintLiteralLoc = encodeSourceLocation(ast->constraintLiteralLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  flatbuffers::Offset<flatbuffers::String> symbolicName;
  if (ast->symbolicName) {
    if (identifiers_.contains(ast->symbolicName)) {
      symbolicName = identifiers_.at(ast->symbolicName);
    } else {
      symbolicName = fbb_.CreateString(ast->symbolicName->value());
      identifiers_.emplace(ast->symbolicName, symbolicName);
    }
  }

  io::AsmOperand::Builder builder{fbb_};
  builder.add_lbracket_loc(lbracketLoc.o);
  builder.add_symbolic_name_loc(symbolicNameLoc.o);
  builder.add_rbracket_loc(rbracketLoc.o);
  builder.add_constraint_literal_loc(constraintLiteralLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);
  if (ast->symbolicName) {
    builder.add_symbolic_name(symbolicName);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmOperand;
}

void ASTEncoder::visit(AsmQualifierAST* ast) {
  auto qualifierLoc = encodeSourceLocation(ast->qualifierLoc);

  io::AsmQualifier::Builder builder{fbb_};
  builder.add_qualifier_loc(qualifierLoc.o);
  builder.add_qualifier(static_cast<std::uint32_t>(ast->qualifier));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmQualifier;
}

void ASTEncoder::visit(AsmClobberAST* ast) {
  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  flatbuffers::Offset<flatbuffers::String> literal;
  if (ast->literal) {
    if (stringLiterals_.contains(ast->literal)) {
      literal = stringLiterals_.at(ast->literal);
    } else {
      literal = fbb_.CreateString(ast->literal->value());
      stringLiterals_.emplace(ast->literal, literal);
    }
  }

  io::AsmClobber::Builder builder{fbb_};
  builder.add_literal_loc(literalLoc.o);
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmClobber;
}

void ASTEncoder::visit(AsmGotoLabelAST* ast) {
  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::AsmGotoLabel::Builder builder{fbb_};
  builder.add_identifier_loc(identifierLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmGotoLabel;
}

void ASTEncoder::visit(LabeledStatementAST* ast) {
  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::LabeledStatement::Builder builder{fbb_};
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_colon_loc(colonLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_LabeledStatement;
}

void ASTEncoder::visit(CaseStatementAST* ast) {
  auto caseLoc = encodeSourceLocation(ast->caseLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  io::CaseStatement::Builder builder{fbb_};
  builder.add_case_loc(caseLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_colon_loc(colonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_CaseStatement;
}

void ASTEncoder::visit(DefaultStatementAST* ast) {
  auto defaultLoc = encodeSourceLocation(ast->defaultLoc);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  io::DefaultStatement::Builder builder{fbb_};
  builder.add_default_loc(defaultLoc.o);
  builder.add_colon_loc(colonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_DefaultStatement;
}

void ASTEncoder::visit(ExpressionStatementAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::ExpressionStatement::Builder builder{fbb_};
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ExpressionStatement;
}

void ASTEncoder::visit(CompoundStatementAST* ast) {
  auto lbraceLoc = encodeSourceLocation(ast->lbraceLoc);

  std::vector<flatbuffers::Offset<>> statementListOffsets;
  std::vector<std::underlying_type_t<io::Statement>> statementListTypes;

  for (auto it = ast->statementList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptStatement(it->value);
    statementListOffsets.push_back(offset);
    statementListTypes.push_back(type);
  }

  auto statementListOffsetsVector = fbb_.CreateVector(statementListOffsets);
  auto statementListTypesVector = fbb_.CreateVector(statementListTypes);

  auto rbraceLoc = encodeSourceLocation(ast->rbraceLoc);

  io::CompoundStatement::Builder builder{fbb_};
  builder.add_lbrace_loc(lbraceLoc.o);
  builder.add_statement_list(statementListOffsetsVector);
  builder.add_statement_list_type(statementListTypesVector);
  builder.add_rbrace_loc(rbraceLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_CompoundStatement;
}

void ASTEncoder::visit(IfStatementAST* ast) {
  auto ifLoc = encodeSourceLocation(ast->ifLoc);

  auto constexprLoc = encodeSourceLocation(ast->constexprLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  auto elseLoc = encodeSourceLocation(ast->elseLoc);

  const auto [elseStatement, elseStatementType] =
      acceptStatement(ast->elseStatement);

  io::IfStatement::Builder builder{fbb_};
  builder.add_if_loc(ifLoc.o);
  builder.add_constexpr_loc(constexprLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));
  builder.add_else_loc(elseLoc.o);
  builder.add_else_statement(elseStatement);
  builder.add_else_statement_type(
      static_cast<io::Statement>(elseStatementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_IfStatement;
}

void ASTEncoder::visit(ConstevalIfStatementAST* ast) {
  auto ifLoc = encodeSourceLocation(ast->ifLoc);

  auto exclaimLoc = encodeSourceLocation(ast->exclaimLoc);

  auto constvalLoc = encodeSourceLocation(ast->constvalLoc);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  auto elseLoc = encodeSourceLocation(ast->elseLoc);

  const auto [elseStatement, elseStatementType] =
      acceptStatement(ast->elseStatement);

  io::ConstevalIfStatement::Builder builder{fbb_};
  builder.add_if_loc(ifLoc.o);
  builder.add_exclaim_loc(exclaimLoc.o);
  builder.add_constval_loc(constvalLoc.o);
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));
  builder.add_else_loc(elseLoc.o);
  builder.add_else_statement(elseStatement);
  builder.add_else_statement_type(
      static_cast<io::Statement>(elseStatementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ConstevalIfStatement;
}

void ASTEncoder::visit(SwitchStatementAST* ast) {
  auto switchLoc = encodeSourceLocation(ast->switchLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  io::SwitchStatement::Builder builder{fbb_};
  builder.add_switch_loc(switchLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_SwitchStatement;
}

void ASTEncoder::visit(WhileStatementAST* ast) {
  auto whileLoc = encodeSourceLocation(ast->whileLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  io::WhileStatement::Builder builder{fbb_};
  builder.add_while_loc(whileLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_WhileStatement;
}

void ASTEncoder::visit(DoStatementAST* ast) {
  auto doLoc = encodeSourceLocation(ast->doLoc);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  auto whileLoc = encodeSourceLocation(ast->whileLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::DoStatement::Builder builder{fbb_};
  builder.add_do_loc(doLoc.o);
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));
  builder.add_while_loc(whileLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_DoStatement;
}

void ASTEncoder::visit(ForRangeStatementAST* ast) {
  auto forLoc = encodeSourceLocation(ast->forLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  const auto [rangeDeclaration, rangeDeclarationType] =
      acceptDeclaration(ast->rangeDeclaration);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  const auto [rangeInitializer, rangeInitializerType] =
      acceptExpression(ast->rangeInitializer);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  io::ForRangeStatement::Builder builder{fbb_};
  builder.add_for_loc(forLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  builder.add_range_declaration(rangeDeclaration);
  builder.add_range_declaration_type(
      static_cast<io::Declaration>(rangeDeclarationType));
  builder.add_colon_loc(colonLoc.o);
  builder.add_range_initializer(rangeInitializer);
  builder.add_range_initializer_type(
      static_cast<io::Expression>(rangeInitializerType));
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ForRangeStatement;
}

void ASTEncoder::visit(ForStatementAST* ast) {
  auto forLoc = encodeSourceLocation(ast->forLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  io::ForStatement::Builder builder{fbb_};
  builder.add_for_loc(forLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_semicolon_loc(semicolonLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ForStatement;
}

void ASTEncoder::visit(BreakStatementAST* ast) {
  auto breakLoc = encodeSourceLocation(ast->breakLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::BreakStatement::Builder builder{fbb_};
  builder.add_break_loc(breakLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_BreakStatement;
}

void ASTEncoder::visit(ContinueStatementAST* ast) {
  auto continueLoc = encodeSourceLocation(ast->continueLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::ContinueStatement::Builder builder{fbb_};
  builder.add_continue_loc(continueLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ContinueStatement;
}

void ASTEncoder::visit(ReturnStatementAST* ast) {
  auto returnLoc = encodeSourceLocation(ast->returnLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::ReturnStatement::Builder builder{fbb_};
  builder.add_return_loc(returnLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ReturnStatement;
}

void ASTEncoder::visit(CoroutineReturnStatementAST* ast) {
  auto coreturnLoc = encodeSourceLocation(ast->coreturnLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::CoroutineReturnStatement::Builder builder{fbb_};
  builder.add_coreturn_loc(coreturnLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_CoroutineReturnStatement;
}

void ASTEncoder::visit(GotoStatementAST* ast) {
  auto gotoLoc = encodeSourceLocation(ast->gotoLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::GotoStatement::Builder builder{fbb_};
  builder.add_goto_loc(gotoLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_GotoStatement;
}

void ASTEncoder::visit(DeclarationStatementAST* ast) {
  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  io::DeclarationStatement::Builder builder{fbb_};
  builder.add_declaration(declaration);
  builder.add_declaration_type(static_cast<io::Declaration>(declarationType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_DeclarationStatement;
}

void ASTEncoder::visit(TryBlockStatementAST* ast) {
  auto tryLoc = encodeSourceLocation(ast->tryLoc);

  const auto statement = accept(ast->statement);

  std::vector<flatbuffers::Offset<io::Handler>> handlerListOffsets;
  for (auto it = ast->handlerList; it; it = it->next) {
    if (!it->value) continue;
    handlerListOffsets.emplace_back(accept(it->value).o);
  }

  auto handlerListOffsetsVector = fbb_.CreateVector(handlerListOffsets);

  io::TryBlockStatement::Builder builder{fbb_};
  builder.add_try_loc(tryLoc.o);
  builder.add_statement(statement.o);
  builder.add_handler_list(handlerListOffsetsVector);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_TryBlockStatement;
}

void ASTEncoder::visit(CharLiteralExpressionAST* ast) {
  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  flatbuffers::Offset<flatbuffers::String> literal;
  if (ast->literal) {
    if (charLiterals_.contains(ast->literal)) {
      literal = charLiterals_.at(ast->literal);
    } else {
      literal = fbb_.CreateString(ast->literal->value());
      charLiterals_.emplace(ast->literal, literal);
    }
  }

  io::CharLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(literalLoc.o);
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CharLiteralExpression;
}

void ASTEncoder::visit(BoolLiteralExpressionAST* ast) {
  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  io::BoolLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(literalLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BoolLiteralExpression;
}

void ASTEncoder::visit(IntLiteralExpressionAST* ast) {
  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  flatbuffers::Offset<flatbuffers::String> literal;
  if (ast->literal) {
    if (integerLiterals_.contains(ast->literal)) {
      literal = integerLiterals_.at(ast->literal);
    } else {
      literal = fbb_.CreateString(ast->literal->value());
      integerLiterals_.emplace(ast->literal, literal);
    }
  }

  io::IntLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(literalLoc.o);
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_IntLiteralExpression;
}

void ASTEncoder::visit(FloatLiteralExpressionAST* ast) {
  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  flatbuffers::Offset<flatbuffers::String> literal;
  if (ast->literal) {
    if (floatLiterals_.contains(ast->literal)) {
      literal = floatLiterals_.at(ast->literal);
    } else {
      literal = fbb_.CreateString(ast->literal->value());
      floatLiterals_.emplace(ast->literal, literal);
    }
  }

  io::FloatLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(literalLoc.o);
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_FloatLiteralExpression;
}

void ASTEncoder::visit(NullptrLiteralExpressionAST* ast) {
  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  io::NullptrLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(literalLoc.o);
  builder.add_literal(static_cast<std::uint32_t>(ast->literal));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NullptrLiteralExpression;
}

void ASTEncoder::visit(StringLiteralExpressionAST* ast) {
  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  flatbuffers::Offset<flatbuffers::String> literal;
  if (ast->literal) {
    if (stringLiterals_.contains(ast->literal)) {
      literal = stringLiterals_.at(ast->literal);
    } else {
      literal = fbb_.CreateString(ast->literal->value());
      stringLiterals_.emplace(ast->literal, literal);
    }
  }

  io::StringLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(literalLoc.o);
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_StringLiteralExpression;
}

void ASTEncoder::visit(UserDefinedStringLiteralExpressionAST* ast) {
  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  flatbuffers::Offset<flatbuffers::String> literal;
  if (ast->literal) {
    if (stringLiterals_.contains(ast->literal)) {
      literal = stringLiterals_.at(ast->literal);
    } else {
      literal = fbb_.CreateString(ast->literal->value());
      stringLiterals_.emplace(ast->literal, literal);
    }
  }

  io::UserDefinedStringLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(literalLoc.o);
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_UserDefinedStringLiteralExpression;
}

void ASTEncoder::visit(ThisExpressionAST* ast) {
  auto thisLoc = encodeSourceLocation(ast->thisLoc);

  io::ThisExpression::Builder builder{fbb_};
  builder.add_this_loc(thisLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ThisExpression;
}

void ASTEncoder::visit(NestedExpressionAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::NestedExpression::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NestedExpression;
}

void ASTEncoder::visit(IdExpressionAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::IdExpression::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(templateLoc.o);
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_IdExpression;
}

void ASTEncoder::visit(LambdaExpressionAST* ast) {
  auto lbracketLoc = encodeSourceLocation(ast->lbracketLoc);

  auto captureDefaultLoc = encodeSourceLocation(ast->captureDefaultLoc);

  std::vector<flatbuffers::Offset<>> captureListOffsets;
  std::vector<std::underlying_type_t<io::LambdaCapture>> captureListTypes;

  for (auto it = ast->captureList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptLambdaCapture(it->value);
    captureListOffsets.push_back(offset);
    captureListTypes.push_back(type);
  }

  auto captureListOffsetsVector = fbb_.CreateVector(captureListOffsets);
  auto captureListTypesVector = fbb_.CreateVector(captureListTypes);

  auto rbracketLoc = encodeSourceLocation(ast->rbracketLoc);

  auto lessLoc = encodeSourceLocation(ast->lessLoc);

  std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
  std::vector<std::underlying_type_t<io::TemplateParameter>>
      templateParameterListTypes;

  for (auto it = ast->templateParameterList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptTemplateParameter(it->value);
    templateParameterListOffsets.push_back(offset);
    templateParameterListTypes.push_back(type);
  }

  auto templateParameterListOffsetsVector =
      fbb_.CreateVector(templateParameterListOffsets);
  auto templateParameterListTypesVector =
      fbb_.CreateVector(templateParameterListTypes);

  auto greaterLoc = encodeSourceLocation(ast->greaterLoc);

  const auto templateRequiresClause = accept(ast->templateRequiresClause);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  std::vector<flatbuffers::Offset<io::LambdaSpecifier>>
      lambdaSpecifierListOffsets;
  for (auto it = ast->lambdaSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    lambdaSpecifierListOffsets.emplace_back(accept(it->value).o);
  }

  auto lambdaSpecifierListOffsetsVector =
      fbb_.CreateVector(lambdaSpecifierListOffsets);

  const auto [exceptionSpecifier, exceptionSpecifierType] =
      acceptExceptionSpecifier(ast->exceptionSpecifier);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto trailingReturnType = accept(ast->trailingReturnType);

  const auto requiresClause = accept(ast->requiresClause);

  const auto statement = accept(ast->statement);

  io::LambdaExpression::Builder builder{fbb_};
  builder.add_lbracket_loc(lbracketLoc.o);
  builder.add_capture_default_loc(captureDefaultLoc.o);
  builder.add_capture_list(captureListOffsetsVector);
  builder.add_capture_list_type(captureListTypesVector);
  builder.add_rbracket_loc(rbracketLoc.o);
  builder.add_less_loc(lessLoc.o);
  builder.add_template_parameter_list(templateParameterListOffsetsVector);
  builder.add_template_parameter_list_type(templateParameterListTypesVector);
  builder.add_greater_loc(greaterLoc.o);
  builder.add_template_requires_clause(templateRequiresClause.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_lambda_specifier_list(lambdaSpecifierListOffsetsVector);
  builder.add_exception_specifier(exceptionSpecifier);
  builder.add_exception_specifier_type(
      static_cast<io::ExceptionSpecifier>(exceptionSpecifierType));
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_trailing_return_type(trailingReturnType.o);
  builder.add_requires_clause(requiresClause.o);
  builder.add_statement(statement.o);
  builder.add_capture_default(static_cast<std::uint32_t>(ast->captureDefault));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_LambdaExpression;
}

void ASTEncoder::visit(FoldExpressionAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [leftExpression, leftExpressionType] =
      acceptExpression(ast->leftExpression);

  auto opLoc = encodeSourceLocation(ast->opLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto foldOpLoc = encodeSourceLocation(ast->foldOpLoc);

  const auto [rightExpression, rightExpressionType] =
      acceptExpression(ast->rightExpression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::FoldExpression::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_left_expression(leftExpression);
  builder.add_left_expression_type(
      static_cast<io::Expression>(leftExpressionType));
  builder.add_op_loc(opLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_fold_op_loc(foldOpLoc.o);
  builder.add_right_expression(rightExpression);
  builder.add_right_expression_type(
      static_cast<io::Expression>(rightExpressionType));
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_op(static_cast<std::uint32_t>(ast->op));
  builder.add_fold_op(static_cast<std::uint32_t>(ast->foldOp));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_FoldExpression;
}

void ASTEncoder::visit(RightFoldExpressionAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto opLoc = encodeSourceLocation(ast->opLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::RightFoldExpression::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_op_loc(opLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_RightFoldExpression;
}

void ASTEncoder::visit(LeftFoldExpressionAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto opLoc = encodeSourceLocation(ast->opLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::LeftFoldExpression::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_op_loc(opLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_LeftFoldExpression;
}

void ASTEncoder::visit(RequiresExpressionAST* ast) {
  auto requiresLoc = encodeSourceLocation(ast->requiresLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  auto lbraceLoc = encodeSourceLocation(ast->lbraceLoc);

  std::vector<flatbuffers::Offset<>> requirementListOffsets;
  std::vector<std::underlying_type_t<io::Requirement>> requirementListTypes;

  for (auto it = ast->requirementList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptRequirement(it->value);
    requirementListOffsets.push_back(offset);
    requirementListTypes.push_back(type);
  }

  auto requirementListOffsetsVector = fbb_.CreateVector(requirementListOffsets);
  auto requirementListTypesVector = fbb_.CreateVector(requirementListTypes);

  auto rbraceLoc = encodeSourceLocation(ast->rbraceLoc);

  io::RequiresExpression::Builder builder{fbb_};
  builder.add_requires_loc(requiresLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_lbrace_loc(lbraceLoc.o);
  builder.add_requirement_list(requirementListOffsetsVector);
  builder.add_requirement_list_type(requirementListTypesVector);
  builder.add_rbrace_loc(rbraceLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_RequiresExpression;
}

void ASTEncoder::visit(SubscriptExpressionAST* ast) {
  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  auto lbracketLoc = encodeSourceLocation(ast->lbracketLoc);

  const auto [indexExpression, indexExpressionType] =
      acceptExpression(ast->indexExpression);

  auto rbracketLoc = encodeSourceLocation(ast->rbracketLoc);

  io::SubscriptExpression::Builder builder{fbb_};
  builder.add_base_expression(baseExpression);
  builder.add_base_expression_type(
      static_cast<io::Expression>(baseExpressionType));
  builder.add_lbracket_loc(lbracketLoc.o);
  builder.add_index_expression(indexExpression);
  builder.add_index_expression_type(
      static_cast<io::Expression>(indexExpressionType));
  builder.add_rbracket_loc(rbracketLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SubscriptExpression;
}

void ASTEncoder::visit(CallExpressionAST* ast) {
  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptExpression(it->value);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::CallExpression::Builder builder{fbb_};
  builder.add_base_expression(baseExpression);
  builder.add_base_expression_type(
      static_cast<io::Expression>(baseExpressionType));
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CallExpression;
}

void ASTEncoder::visit(TypeConstructionAST* ast) {
  const auto [typeSpecifier, typeSpecifierType] =
      acceptSpecifier(ast->typeSpecifier);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptExpression(it->value);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::TypeConstruction::Builder builder{fbb_};
  builder.add_type_specifier(typeSpecifier);
  builder.add_type_specifier_type(
      static_cast<io::Specifier>(typeSpecifierType));
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeConstruction;
}

void ASTEncoder::visit(BracedTypeConstructionAST* ast) {
  const auto [typeSpecifier, typeSpecifierType] =
      acceptSpecifier(ast->typeSpecifier);

  const auto bracedInitList = accept(ast->bracedInitList);

  io::BracedTypeConstruction::Builder builder{fbb_};
  builder.add_type_specifier(typeSpecifier);
  builder.add_type_specifier_type(
      static_cast<io::Specifier>(typeSpecifierType));
  builder.add_braced_init_list(bracedInitList.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BracedTypeConstruction;
}

void ASTEncoder::visit(MemberExpressionAST* ast) {
  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  auto accessLoc = encodeSourceLocation(ast->accessLoc);

  const auto memberId = accept(ast->memberId);

  io::MemberExpression::Builder builder{fbb_};
  builder.add_base_expression(baseExpression);
  builder.add_base_expression_type(
      static_cast<io::Expression>(baseExpressionType));
  builder.add_access_loc(accessLoc.o);
  builder.add_member_id(memberId.o);
  builder.add_access_op(static_cast<std::uint32_t>(ast->accessOp));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_MemberExpression;
}

void ASTEncoder::visit(PostIncrExpressionAST* ast) {
  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  auto opLoc = encodeSourceLocation(ast->opLoc);

  io::PostIncrExpression::Builder builder{fbb_};
  builder.add_base_expression(baseExpression);
  builder.add_base_expression_type(
      static_cast<io::Expression>(baseExpressionType));
  builder.add_op_loc(opLoc.o);
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_PostIncrExpression;
}

void ASTEncoder::visit(CppCastExpressionAST* ast) {
  auto castLoc = encodeSourceLocation(ast->castLoc);

  auto lessLoc = encodeSourceLocation(ast->lessLoc);

  const auto typeId = accept(ast->typeId);

  auto greaterLoc = encodeSourceLocation(ast->greaterLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::CppCastExpression::Builder builder{fbb_};
  builder.add_cast_loc(castLoc.o);
  builder.add_less_loc(lessLoc.o);
  builder.add_type_id(typeId.o);
  builder.add_greater_loc(greaterLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CppCastExpression;
}

void ASTEncoder::visit(TypeidExpressionAST* ast) {
  auto typeidLoc = encodeSourceLocation(ast->typeidLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::TypeidExpression::Builder builder{fbb_};
  builder.add_typeid_loc(typeidLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeidExpression;
}

void ASTEncoder::visit(TypeidOfTypeExpressionAST* ast) {
  auto typeidLoc = encodeSourceLocation(ast->typeidLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto typeId = accept(ast->typeId);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::TypeidOfTypeExpression::Builder builder{fbb_};
  builder.add_typeid_loc(typeidLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeidOfTypeExpression;
}

void ASTEncoder::visit(UnaryExpressionAST* ast) {
  auto opLoc = encodeSourceLocation(ast->opLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::UnaryExpression::Builder builder{fbb_};
  builder.add_op_loc(opLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_UnaryExpression;
}

void ASTEncoder::visit(AwaitExpressionAST* ast) {
  auto awaitLoc = encodeSourceLocation(ast->awaitLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::AwaitExpression::Builder builder{fbb_};
  builder.add_await_loc(awaitLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AwaitExpression;
}

void ASTEncoder::visit(SizeofExpressionAST* ast) {
  auto sizeofLoc = encodeSourceLocation(ast->sizeofLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::SizeofExpression::Builder builder{fbb_};
  builder.add_sizeof_loc(sizeofLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SizeofExpression;
}

void ASTEncoder::visit(SizeofTypeExpressionAST* ast) {
  auto sizeofLoc = encodeSourceLocation(ast->sizeofLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto typeId = accept(ast->typeId);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::SizeofTypeExpression::Builder builder{fbb_};
  builder.add_sizeof_loc(sizeofLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SizeofTypeExpression;
}

void ASTEncoder::visit(SizeofPackExpressionAST* ast) {
  auto sizeofLoc = encodeSourceLocation(ast->sizeofLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::SizeofPackExpression::Builder builder{fbb_};
  builder.add_sizeof_loc(sizeofLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_rparen_loc(rparenLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SizeofPackExpression;
}

void ASTEncoder::visit(AlignofTypeExpressionAST* ast) {
  auto alignofLoc = encodeSourceLocation(ast->alignofLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto typeId = accept(ast->typeId);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::AlignofTypeExpression::Builder builder{fbb_};
  builder.add_alignof_loc(alignofLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AlignofTypeExpression;
}

void ASTEncoder::visit(AlignofExpressionAST* ast) {
  auto alignofLoc = encodeSourceLocation(ast->alignofLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::AlignofExpression::Builder builder{fbb_};
  builder.add_alignof_loc(alignofLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AlignofExpression;
}

void ASTEncoder::visit(NoexceptExpressionAST* ast) {
  auto noexceptLoc = encodeSourceLocation(ast->noexceptLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::NoexceptExpression::Builder builder{fbb_};
  builder.add_noexcept_loc(noexceptLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NoexceptExpression;
}

void ASTEncoder::visit(NewExpressionAST* ast) {
  auto scopeLoc = encodeSourceLocation(ast->scopeLoc);

  auto newLoc = encodeSourceLocation(ast->newLoc);

  const auto newPlacement = accept(ast->newPlacement);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  const auto declarator = accept(ast->declarator);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  const auto [newInitalizer, newInitalizerType] =
      acceptNewInitializer(ast->newInitalizer);

  io::NewExpression::Builder builder{fbb_};
  builder.add_scope_loc(scopeLoc.o);
  builder.add_new_loc(newLoc.o);
  builder.add_new_placement(newPlacement.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_declarator(declarator.o);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_new_initalizer(newInitalizer);
  builder.add_new_initalizer_type(
      static_cast<io::NewInitializer>(newInitalizerType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NewExpression;
}

void ASTEncoder::visit(DeleteExpressionAST* ast) {
  auto scopeLoc = encodeSourceLocation(ast->scopeLoc);

  auto deleteLoc = encodeSourceLocation(ast->deleteLoc);

  auto lbracketLoc = encodeSourceLocation(ast->lbracketLoc);

  auto rbracketLoc = encodeSourceLocation(ast->rbracketLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::DeleteExpression::Builder builder{fbb_};
  builder.add_scope_loc(scopeLoc.o);
  builder.add_delete_loc(deleteLoc.o);
  builder.add_lbracket_loc(lbracketLoc.o);
  builder.add_rbracket_loc(rbracketLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_DeleteExpression;
}

void ASTEncoder::visit(CastExpressionAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto typeId = accept(ast->typeId);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::CastExpression::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CastExpression;
}

void ASTEncoder::visit(ImplicitCastExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::ImplicitCastExpression::Builder builder{fbb_};
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ImplicitCastExpression;
}

void ASTEncoder::visit(BinaryExpressionAST* ast) {
  const auto [leftExpression, leftExpressionType] =
      acceptExpression(ast->leftExpression);

  auto opLoc = encodeSourceLocation(ast->opLoc);

  const auto [rightExpression, rightExpressionType] =
      acceptExpression(ast->rightExpression);

  io::BinaryExpression::Builder builder{fbb_};
  builder.add_left_expression(leftExpression);
  builder.add_left_expression_type(
      static_cast<io::Expression>(leftExpressionType));
  builder.add_op_loc(opLoc.o);
  builder.add_right_expression(rightExpression);
  builder.add_right_expression_type(
      static_cast<io::Expression>(rightExpressionType));
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BinaryExpression;
}

void ASTEncoder::visit(ConditionalExpressionAST* ast) {
  const auto [condition, conditionType] = acceptExpression(ast->condition);

  auto questionLoc = encodeSourceLocation(ast->questionLoc);

  const auto [iftrueExpression, iftrueExpressionType] =
      acceptExpression(ast->iftrueExpression);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  const auto [iffalseExpression, iffalseExpressionType] =
      acceptExpression(ast->iffalseExpression);

  io::ConditionalExpression::Builder builder{fbb_};
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_question_loc(questionLoc.o);
  builder.add_iftrue_expression(iftrueExpression);
  builder.add_iftrue_expression_type(
      static_cast<io::Expression>(iftrueExpressionType));
  builder.add_colon_loc(colonLoc.o);
  builder.add_iffalse_expression(iffalseExpression);
  builder.add_iffalse_expression_type(
      static_cast<io::Expression>(iffalseExpressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ConditionalExpression;
}

void ASTEncoder::visit(YieldExpressionAST* ast) {
  auto yieldLoc = encodeSourceLocation(ast->yieldLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::YieldExpression::Builder builder{fbb_};
  builder.add_yield_loc(yieldLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_YieldExpression;
}

void ASTEncoder::visit(ThrowExpressionAST* ast) {
  auto throwLoc = encodeSourceLocation(ast->throwLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::ThrowExpression::Builder builder{fbb_};
  builder.add_throw_loc(throwLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ThrowExpression;
}

void ASTEncoder::visit(AssignmentExpressionAST* ast) {
  const auto [leftExpression, leftExpressionType] =
      acceptExpression(ast->leftExpression);

  auto opLoc = encodeSourceLocation(ast->opLoc);

  const auto [rightExpression, rightExpressionType] =
      acceptExpression(ast->rightExpression);

  io::AssignmentExpression::Builder builder{fbb_};
  builder.add_left_expression(leftExpression);
  builder.add_left_expression_type(
      static_cast<io::Expression>(leftExpressionType));
  builder.add_op_loc(opLoc.o);
  builder.add_right_expression(rightExpression);
  builder.add_right_expression_type(
      static_cast<io::Expression>(rightExpressionType));
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AssignmentExpression;
}

void ASTEncoder::visit(PackExpansionExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  io::PackExpansionExpression::Builder builder{fbb_};
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_ellipsis_loc(ellipsisLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_PackExpansionExpression;
}

void ASTEncoder::visit(DesignatedInitializerClauseAST* ast) {
  auto dotLoc = encodeSourceLocation(ast->dotLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  const auto [initializer, initializerType] =
      acceptExpression(ast->initializer);

  io::DesignatedInitializerClause::Builder builder{fbb_};
  builder.add_dot_loc(dotLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_DesignatedInitializerClause;
}

void ASTEncoder::visit(TypeTraitsExpressionAST* ast) {
  auto typeTraitsLoc = encodeSourceLocation(ast->typeTraitsLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  std::vector<flatbuffers::Offset<io::TypeId>> typeIdListOffsets;
  for (auto it = ast->typeIdList; it; it = it->next) {
    if (!it->value) continue;
    typeIdListOffsets.emplace_back(accept(it->value).o);
  }

  auto typeIdListOffsetsVector = fbb_.CreateVector(typeIdListOffsets);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::TypeTraitsExpression::Builder builder{fbb_};
  builder.add_type_traits_loc(typeTraitsLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_type_id_list(typeIdListOffsetsVector);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_type_traits(static_cast<std::uint32_t>(ast->typeTraits));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeTraitsExpression;
}

void ASTEncoder::visit(ConditionExpressionAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    declSpecifierListOffsets.push_back(offset);
    declSpecifierListTypes.push_back(type);
  }

  auto declSpecifierListOffsetsVector =
      fbb_.CreateVector(declSpecifierListOffsets);
  auto declSpecifierListTypesVector = fbb_.CreateVector(declSpecifierListTypes);

  const auto declarator = accept(ast->declarator);

  const auto [initializer, initializerType] =
      acceptExpression(ast->initializer);

  io::ConditionExpression::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_decl_specifier_list(declSpecifierListOffsetsVector);
  builder.add_decl_specifier_list_type(declSpecifierListTypesVector);
  builder.add_declarator(declarator.o);
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ConditionExpression;
}

void ASTEncoder::visit(EqualInitializerAST* ast) {
  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::EqualInitializer::Builder builder{fbb_};
  builder.add_equal_loc(equalLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_EqualInitializer;
}

void ASTEncoder::visit(BracedInitListAST* ast) {
  auto lbraceLoc = encodeSourceLocation(ast->lbraceLoc);

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptExpression(it->value);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  auto commaLoc = encodeSourceLocation(ast->commaLoc);

  auto rbraceLoc = encodeSourceLocation(ast->rbraceLoc);

  io::BracedInitList::Builder builder{fbb_};
  builder.add_lbrace_loc(lbraceLoc.o);
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_comma_loc(commaLoc.o);
  builder.add_rbrace_loc(rbraceLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BracedInitList;
}

void ASTEncoder::visit(ParenInitializerAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptExpression(it->value);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::ParenInitializer::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ParenInitializer;
}

void ASTEncoder::visit(TemplateTypeParameterAST* ast) {
  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  auto lessLoc = encodeSourceLocation(ast->lessLoc);

  std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
  std::vector<std::underlying_type_t<io::TemplateParameter>>
      templateParameterListTypes;

  for (auto it = ast->templateParameterList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptTemplateParameter(it->value);
    templateParameterListOffsets.push_back(offset);
    templateParameterListTypes.push_back(type);
  }

  auto templateParameterListOffsetsVector =
      fbb_.CreateVector(templateParameterListOffsets);
  auto templateParameterListTypesVector =
      fbb_.CreateVector(templateParameterListTypes);

  auto greaterLoc = encodeSourceLocation(ast->greaterLoc);

  const auto requiresClause = accept(ast->requiresClause);

  auto classKeyLoc = encodeSourceLocation(ast->classKeyLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  const auto idExpression = accept(ast->idExpression);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::TemplateTypeParameter::Builder builder{fbb_};
  builder.add_template_loc(templateLoc.o);
  builder.add_less_loc(lessLoc.o);
  builder.add_template_parameter_list(templateParameterListOffsetsVector);
  builder.add_template_parameter_list_type(templateParameterListTypesVector);
  builder.add_greater_loc(greaterLoc.o);
  builder.add_requires_clause(requiresClause.o);
  builder.add_class_key_loc(classKeyLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_equal_loc(equalLoc.o);
  builder.add_id_expression(idExpression.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::TemplateParameter_TemplateTypeParameter;
}

void ASTEncoder::visit(TemplatePackTypeParameterAST* ast) {
  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  auto lessLoc = encodeSourceLocation(ast->lessLoc);

  std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
  std::vector<std::underlying_type_t<io::TemplateParameter>>
      templateParameterListTypes;

  for (auto it = ast->templateParameterList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptTemplateParameter(it->value);
    templateParameterListOffsets.push_back(offset);
    templateParameterListTypes.push_back(type);
  }

  auto templateParameterListOffsetsVector =
      fbb_.CreateVector(templateParameterListOffsets);
  auto templateParameterListTypesVector =
      fbb_.CreateVector(templateParameterListTypes);

  auto greaterLoc = encodeSourceLocation(ast->greaterLoc);

  auto classKeyLoc = encodeSourceLocation(ast->classKeyLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::TemplatePackTypeParameter::Builder builder{fbb_};
  builder.add_template_loc(templateLoc.o);
  builder.add_less_loc(lessLoc.o);
  builder.add_template_parameter_list(templateParameterListOffsetsVector);
  builder.add_template_parameter_list_type(templateParameterListTypesVector);
  builder.add_greater_loc(greaterLoc.o);
  builder.add_class_key_loc(classKeyLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::TemplateParameter_TemplatePackTypeParameter;
}

void ASTEncoder::visit(NonTypeTemplateParameterAST* ast) {
  const auto declaration = accept(ast->declaration);

  io::NonTypeTemplateParameter::Builder builder{fbb_};
  builder.add_declaration(declaration.o);

  offset_ = builder.Finish().Union();
  type_ = io::TemplateParameter_NonTypeTemplateParameter;
}

void ASTEncoder::visit(TypenameTypeParameterAST* ast) {
  auto classKeyLoc = encodeSourceLocation(ast->classKeyLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  const auto typeId = accept(ast->typeId);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::TypenameTypeParameter::Builder builder{fbb_};
  builder.add_class_key_loc(classKeyLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_equal_loc(equalLoc.o);
  builder.add_type_id(typeId.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::TemplateParameter_TypenameTypeParameter;
}

void ASTEncoder::visit(ConstraintTypeParameterAST* ast) {
  const auto typeConstraint = accept(ast->typeConstraint);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  const auto typeId = accept(ast->typeId);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::ConstraintTypeParameter::Builder builder{fbb_};
  builder.add_type_constraint(typeConstraint.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_equal_loc(equalLoc.o);
  builder.add_type_id(typeId.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::TemplateParameter_ConstraintTypeParameter;
}

void ASTEncoder::visit(TypedefSpecifierAST* ast) {
  auto typedefLoc = encodeSourceLocation(ast->typedefLoc);

  io::TypedefSpecifier::Builder builder{fbb_};
  builder.add_typedef_loc(typedefLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_TypedefSpecifier;
}

void ASTEncoder::visit(FriendSpecifierAST* ast) {
  auto friendLoc = encodeSourceLocation(ast->friendLoc);

  io::FriendSpecifier::Builder builder{fbb_};
  builder.add_friend_loc(friendLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_FriendSpecifier;
}

void ASTEncoder::visit(ConstevalSpecifierAST* ast) {
  auto constevalLoc = encodeSourceLocation(ast->constevalLoc);

  io::ConstevalSpecifier::Builder builder{fbb_};
  builder.add_consteval_loc(constevalLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstevalSpecifier;
}

void ASTEncoder::visit(ConstinitSpecifierAST* ast) {
  auto constinitLoc = encodeSourceLocation(ast->constinitLoc);

  io::ConstinitSpecifier::Builder builder{fbb_};
  builder.add_constinit_loc(constinitLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstinitSpecifier;
}

void ASTEncoder::visit(ConstexprSpecifierAST* ast) {
  auto constexprLoc = encodeSourceLocation(ast->constexprLoc);

  io::ConstexprSpecifier::Builder builder{fbb_};
  builder.add_constexpr_loc(constexprLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstexprSpecifier;
}

void ASTEncoder::visit(InlineSpecifierAST* ast) {
  auto inlineLoc = encodeSourceLocation(ast->inlineLoc);

  io::InlineSpecifier::Builder builder{fbb_};
  builder.add_inline_loc(inlineLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_InlineSpecifier;
}

void ASTEncoder::visit(StaticSpecifierAST* ast) {
  auto staticLoc = encodeSourceLocation(ast->staticLoc);

  io::StaticSpecifier::Builder builder{fbb_};
  builder.add_static_loc(staticLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_StaticSpecifier;
}

void ASTEncoder::visit(ExternSpecifierAST* ast) {
  auto externLoc = encodeSourceLocation(ast->externLoc);

  io::ExternSpecifier::Builder builder{fbb_};
  builder.add_extern_loc(externLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ExternSpecifier;
}

void ASTEncoder::visit(ThreadLocalSpecifierAST* ast) {
  auto threadLocalLoc = encodeSourceLocation(ast->threadLocalLoc);

  io::ThreadLocalSpecifier::Builder builder{fbb_};
  builder.add_thread_local_loc(threadLocalLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ThreadLocalSpecifier;
}

void ASTEncoder::visit(ThreadSpecifierAST* ast) {
  auto threadLoc = encodeSourceLocation(ast->threadLoc);

  io::ThreadSpecifier::Builder builder{fbb_};
  builder.add_thread_loc(threadLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ThreadSpecifier;
}

void ASTEncoder::visit(MutableSpecifierAST* ast) {
  auto mutableLoc = encodeSourceLocation(ast->mutableLoc);

  io::MutableSpecifier::Builder builder{fbb_};
  builder.add_mutable_loc(mutableLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_MutableSpecifier;
}

void ASTEncoder::visit(VirtualSpecifierAST* ast) {
  auto virtualLoc = encodeSourceLocation(ast->virtualLoc);

  io::VirtualSpecifier::Builder builder{fbb_};
  builder.add_virtual_loc(virtualLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VirtualSpecifier;
}

void ASTEncoder::visit(ExplicitSpecifierAST* ast) {
  auto explicitLoc = encodeSourceLocation(ast->explicitLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::ExplicitSpecifier::Builder builder{fbb_};
  builder.add_explicit_loc(explicitLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ExplicitSpecifier;
}

void ASTEncoder::visit(AutoTypeSpecifierAST* ast) {
  auto autoLoc = encodeSourceLocation(ast->autoLoc);

  io::AutoTypeSpecifier::Builder builder{fbb_};
  builder.add_auto_loc(autoLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_AutoTypeSpecifier;
}

void ASTEncoder::visit(VoidTypeSpecifierAST* ast) {
  auto voidLoc = encodeSourceLocation(ast->voidLoc);

  io::VoidTypeSpecifier::Builder builder{fbb_};
  builder.add_void_loc(voidLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VoidTypeSpecifier;
}

void ASTEncoder::visit(SizeTypeSpecifierAST* ast) {
  auto specifierLoc = encodeSourceLocation(ast->specifierLoc);

  io::SizeTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(specifierLoc.o);
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_SizeTypeSpecifier;
}

void ASTEncoder::visit(SignTypeSpecifierAST* ast) {
  auto specifierLoc = encodeSourceLocation(ast->specifierLoc);

  io::SignTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(specifierLoc.o);
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_SignTypeSpecifier;
}

void ASTEncoder::visit(VaListTypeSpecifierAST* ast) {
  auto specifierLoc = encodeSourceLocation(ast->specifierLoc);

  io::VaListTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(specifierLoc.o);
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VaListTypeSpecifier;
}

void ASTEncoder::visit(IntegralTypeSpecifierAST* ast) {
  auto specifierLoc = encodeSourceLocation(ast->specifierLoc);

  io::IntegralTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(specifierLoc.o);
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_IntegralTypeSpecifier;
}

void ASTEncoder::visit(FloatingPointTypeSpecifierAST* ast) {
  auto specifierLoc = encodeSourceLocation(ast->specifierLoc);

  io::FloatingPointTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(specifierLoc.o);
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_FloatingPointTypeSpecifier;
}

void ASTEncoder::visit(ComplexTypeSpecifierAST* ast) {
  auto complexLoc = encodeSourceLocation(ast->complexLoc);

  io::ComplexTypeSpecifier::Builder builder{fbb_};
  builder.add_complex_loc(complexLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ComplexTypeSpecifier;
}

void ASTEncoder::visit(NamedTypeSpecifierAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::NamedTypeSpecifier::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(templateLoc.o);
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_NamedTypeSpecifier;
}

void ASTEncoder::visit(AtomicTypeSpecifierAST* ast) {
  auto atomicLoc = encodeSourceLocation(ast->atomicLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto typeId = accept(ast->typeId);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::AtomicTypeSpecifier::Builder builder{fbb_};
  builder.add_atomic_loc(atomicLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_AtomicTypeSpecifier;
}

void ASTEncoder::visit(UnderlyingTypeSpecifierAST* ast) {
  auto underlyingTypeLoc = encodeSourceLocation(ast->underlyingTypeLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto typeId = accept(ast->typeId);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::UnderlyingTypeSpecifier::Builder builder{fbb_};
  builder.add_underlying_type_loc(underlyingTypeLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_UnderlyingTypeSpecifier;
}

void ASTEncoder::visit(ElaboratedTypeSpecifierAST* ast) {
  auto classLoc = encodeSourceLocation(ast->classLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::ElaboratedTypeSpecifier::Builder builder{fbb_};
  builder.add_class_loc(classLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(templateLoc.o);
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_class_key(static_cast<std::uint32_t>(ast->classKey));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ElaboratedTypeSpecifier;
}

void ASTEncoder::visit(DecltypeAutoSpecifierAST* ast) {
  auto decltypeLoc = encodeSourceLocation(ast->decltypeLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  auto autoLoc = encodeSourceLocation(ast->autoLoc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::DecltypeAutoSpecifier::Builder builder{fbb_};
  builder.add_decltype_loc(decltypeLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_auto_loc(autoLoc.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_DecltypeAutoSpecifier;
}

void ASTEncoder::visit(DecltypeSpecifierAST* ast) {
  auto decltypeLoc = encodeSourceLocation(ast->decltypeLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::DecltypeSpecifier::Builder builder{fbb_};
  builder.add_decltype_loc(decltypeLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_DecltypeSpecifier;
}

void ASTEncoder::visit(PlaceholderTypeSpecifierAST* ast) {
  const auto typeConstraint = accept(ast->typeConstraint);

  const auto [specifier, specifierType] = acceptSpecifier(ast->specifier);

  io::PlaceholderTypeSpecifier::Builder builder{fbb_};
  builder.add_type_constraint(typeConstraint.o);
  builder.add_specifier(specifier);
  builder.add_specifier_type(static_cast<io::Specifier>(specifierType));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_PlaceholderTypeSpecifier;
}

void ASTEncoder::visit(ConstQualifierAST* ast) {
  auto constLoc = encodeSourceLocation(ast->constLoc);

  io::ConstQualifier::Builder builder{fbb_};
  builder.add_const_loc(constLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstQualifier;
}

void ASTEncoder::visit(VolatileQualifierAST* ast) {
  auto volatileLoc = encodeSourceLocation(ast->volatileLoc);

  io::VolatileQualifier::Builder builder{fbb_};
  builder.add_volatile_loc(volatileLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VolatileQualifier;
}

void ASTEncoder::visit(RestrictQualifierAST* ast) {
  auto restrictLoc = encodeSourceLocation(ast->restrictLoc);

  io::RestrictQualifier::Builder builder{fbb_};
  builder.add_restrict_loc(restrictLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_RestrictQualifier;
}

void ASTEncoder::visit(EnumSpecifierAST* ast) {
  auto enumLoc = encodeSourceLocation(ast->enumLoc);

  auto classLoc = encodeSourceLocation(ast->classLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto unqualifiedId = accept(ast->unqualifiedId);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  auto lbraceLoc = encodeSourceLocation(ast->lbraceLoc);

  auto commaLoc = encodeSourceLocation(ast->commaLoc);

  std::vector<flatbuffers::Offset<io::Enumerator>> enumeratorListOffsets;
  for (auto it = ast->enumeratorList; it; it = it->next) {
    if (!it->value) continue;
    enumeratorListOffsets.emplace_back(accept(it->value).o);
  }

  auto enumeratorListOffsetsVector = fbb_.CreateVector(enumeratorListOffsets);

  auto rbraceLoc = encodeSourceLocation(ast->rbraceLoc);

  io::EnumSpecifier::Builder builder{fbb_};
  builder.add_enum_loc(enumLoc.o);
  builder.add_class_loc(classLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_colon_loc(colonLoc.o);
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_lbrace_loc(lbraceLoc.o);
  builder.add_comma_loc(commaLoc.o);
  builder.add_enumerator_list(enumeratorListOffsetsVector);
  builder.add_rbrace_loc(rbraceLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_EnumSpecifier;
}

void ASTEncoder::visit(ClassSpecifierAST* ast) {
  auto classLoc = encodeSourceLocation(ast->classLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  auto finalLoc = encodeSourceLocation(ast->finalLoc);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  std::vector<flatbuffers::Offset<io::BaseSpecifier>> baseSpecifierListOffsets;
  for (auto it = ast->baseSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    baseSpecifierListOffsets.emplace_back(accept(it->value).o);
  }

  auto baseSpecifierListOffsetsVector =
      fbb_.CreateVector(baseSpecifierListOffsets);

  auto lbraceLoc = encodeSourceLocation(ast->lbraceLoc);

  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto it = ast->declarationList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptDeclaration(it->value);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  auto rbraceLoc = encodeSourceLocation(ast->rbraceLoc);

  io::ClassSpecifier::Builder builder{fbb_};
  builder.add_class_loc(classLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_final_loc(finalLoc.o);
  builder.add_colon_loc(colonLoc.o);
  builder.add_base_specifier_list(baseSpecifierListOffsetsVector);
  builder.add_lbrace_loc(lbraceLoc.o);
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);
  builder.add_rbrace_loc(rbraceLoc.o);
  builder.add_class_key(static_cast<std::uint32_t>(ast->classKey));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ClassSpecifier;
}

void ASTEncoder::visit(TypenameSpecifierAST* ast) {
  auto typenameLoc = encodeSourceLocation(ast->typenameLoc);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::TypenameSpecifier::Builder builder{fbb_};
  builder.add_typename_loc(typenameLoc.o);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_TypenameSpecifier;
}

void ASTEncoder::visit(PointerOperatorAST* ast) {
  auto starLoc = encodeSourceLocation(ast->starLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> cvQualifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> cvQualifierListTypes;

  for (auto it = ast->cvQualifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    cvQualifierListOffsets.push_back(offset);
    cvQualifierListTypes.push_back(type);
  }

  auto cvQualifierListOffsetsVector = fbb_.CreateVector(cvQualifierListOffsets);
  auto cvQualifierListTypesVector = fbb_.CreateVector(cvQualifierListTypes);

  io::PointerOperator::Builder builder{fbb_};
  builder.add_star_loc(starLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_cv_qualifier_list(cvQualifierListOffsetsVector);
  builder.add_cv_qualifier_list_type(cvQualifierListTypesVector);

  offset_ = builder.Finish().Union();
  type_ = io::PtrOperator_PointerOperator;
}

void ASTEncoder::visit(ReferenceOperatorAST* ast) {
  auto refLoc = encodeSourceLocation(ast->refLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  io::ReferenceOperator::Builder builder{fbb_};
  builder.add_ref_loc(refLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_ref_op(static_cast<std::uint32_t>(ast->refOp));

  offset_ = builder.Finish().Union();
  type_ = io::PtrOperator_ReferenceOperator;
}

void ASTEncoder::visit(PtrToMemberOperatorAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  auto starLoc = encodeSourceLocation(ast->starLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> cvQualifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> cvQualifierListTypes;

  for (auto it = ast->cvQualifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    cvQualifierListOffsets.push_back(offset);
    cvQualifierListTypes.push_back(type);
  }

  auto cvQualifierListOffsetsVector = fbb_.CreateVector(cvQualifierListOffsets);
  auto cvQualifierListTypesVector = fbb_.CreateVector(cvQualifierListTypes);

  io::PtrToMemberOperator::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_star_loc(starLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_cv_qualifier_list(cvQualifierListOffsetsVector);
  builder.add_cv_qualifier_list_type(cvQualifierListTypesVector);

  offset_ = builder.Finish().Union();
  type_ = io::PtrOperator_PtrToMemberOperator;
}

void ASTEncoder::visit(BitfieldDeclaratorAST* ast) {
  const auto unqualifiedId = accept(ast->unqualifiedId);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  const auto [sizeExpression, sizeExpressionType] =
      acceptExpression(ast->sizeExpression);

  io::BitfieldDeclarator::Builder builder{fbb_};
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_colon_loc(colonLoc.o);
  builder.add_size_expression(sizeExpression);
  builder.add_size_expression_type(
      static_cast<io::Expression>(sizeExpressionType));

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_BitfieldDeclarator;
}

void ASTEncoder::visit(ParameterPackAST* ast) {
  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  const auto [coreDeclarator, coreDeclaratorType] =
      acceptCoreDeclarator(ast->coreDeclarator);

  io::ParameterPack::Builder builder{fbb_};
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_core_declarator(coreDeclarator);
  builder.add_core_declarator_type(
      static_cast<io::CoreDeclarator>(coreDeclaratorType));

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_ParameterPack;
}

void ASTEncoder::visit(IdDeclaratorAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  io::IdDeclarator::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(templateLoc.o);
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_IdDeclarator;
}

void ASTEncoder::visit(NestedDeclaratorAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto declarator = accept(ast->declarator);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::NestedDeclarator::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_declarator(declarator.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_NestedDeclarator;
}

void ASTEncoder::visit(FunctionDeclaratorChunkAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  std::vector<flatbuffers::Offset<>> cvQualifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> cvQualifierListTypes;

  for (auto it = ast->cvQualifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    cvQualifierListOffsets.push_back(offset);
    cvQualifierListTypes.push_back(type);
  }

  auto cvQualifierListOffsetsVector = fbb_.CreateVector(cvQualifierListOffsets);
  auto cvQualifierListTypesVector = fbb_.CreateVector(cvQualifierListTypes);

  auto refLoc = encodeSourceLocation(ast->refLoc);

  const auto [exceptionSpecifier, exceptionSpecifierType] =
      acceptExceptionSpecifier(ast->exceptionSpecifier);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto trailingReturnType = accept(ast->trailingReturnType);

  io::FunctionDeclaratorChunk::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_cv_qualifier_list(cvQualifierListOffsetsVector);
  builder.add_cv_qualifier_list_type(cvQualifierListTypesVector);
  builder.add_ref_loc(refLoc.o);
  builder.add_exception_specifier(exceptionSpecifier);
  builder.add_exception_specifier_type(
      static_cast<io::ExceptionSpecifier>(exceptionSpecifierType));
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_trailing_return_type(trailingReturnType.o);

  offset_ = builder.Finish().Union();
  type_ = io::DeclaratorChunk_FunctionDeclaratorChunk;
}

void ASTEncoder::visit(ArrayDeclaratorChunkAST* ast) {
  auto lbracketLoc = encodeSourceLocation(ast->lbracketLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rbracketLoc = encodeSourceLocation(ast->rbracketLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  io::ArrayDeclaratorChunk::Builder builder{fbb_};
  builder.add_lbracket_loc(lbracketLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rbracket_loc(rbracketLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);

  offset_ = builder.Finish().Union();
  type_ = io::DeclaratorChunk_ArrayDeclaratorChunk;
}

void ASTEncoder::visit(NameIdAST* ast) {
  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::NameId::Builder builder{fbb_};
  builder.add_identifier_loc(identifierLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_NameId;
}

void ASTEncoder::visit(DestructorIdAST* ast) {
  auto tildeLoc = encodeSourceLocation(ast->tildeLoc);

  const auto [id, idType] = acceptUnqualifiedId(ast->id);

  io::DestructorId::Builder builder{fbb_};
  builder.add_tilde_loc(tildeLoc.o);
  builder.add_id(id);
  builder.add_id_type(static_cast<io::UnqualifiedId>(idType));

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_DestructorId;
}

void ASTEncoder::visit(DecltypeIdAST* ast) {
  const auto decltypeSpecifier = accept(ast->decltypeSpecifier);

  io::DecltypeId::Builder builder{fbb_};
  builder.add_decltype_specifier(decltypeSpecifier.o);

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_DecltypeId;
}

void ASTEncoder::visit(OperatorFunctionIdAST* ast) {
  auto operatorLoc = encodeSourceLocation(ast->operatorLoc);

  auto opLoc = encodeSourceLocation(ast->opLoc);

  auto openLoc = encodeSourceLocation(ast->openLoc);

  auto closeLoc = encodeSourceLocation(ast->closeLoc);

  io::OperatorFunctionId::Builder builder{fbb_};
  builder.add_operator_loc(operatorLoc.o);
  builder.add_op_loc(opLoc.o);
  builder.add_open_loc(openLoc.o);
  builder.add_close_loc(closeLoc.o);
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_OperatorFunctionId;
}

void ASTEncoder::visit(LiteralOperatorIdAST* ast) {
  auto operatorLoc = encodeSourceLocation(ast->operatorLoc);

  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::LiteralOperatorId::Builder builder{fbb_};
  builder.add_operator_loc(operatorLoc.o);
  builder.add_literal_loc(literalLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_LiteralOperatorId;
}

void ASTEncoder::visit(ConversionFunctionIdAST* ast) {
  auto operatorLoc = encodeSourceLocation(ast->operatorLoc);

  const auto typeId = accept(ast->typeId);

  io::ConversionFunctionId::Builder builder{fbb_};
  builder.add_operator_loc(operatorLoc.o);
  builder.add_type_id(typeId.o);

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_ConversionFunctionId;
}

void ASTEncoder::visit(SimpleTemplateIdAST* ast) {
  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto lessLoc = encodeSourceLocation(ast->lessLoc);

  std::vector<flatbuffers::Offset<>> templateArgumentListOffsets;
  std::vector<std::underlying_type_t<io::TemplateArgument>>
      templateArgumentListTypes;

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptTemplateArgument(it->value);
    templateArgumentListOffsets.push_back(offset);
    templateArgumentListTypes.push_back(type);
  }

  auto templateArgumentListOffsetsVector =
      fbb_.CreateVector(templateArgumentListOffsets);
  auto templateArgumentListTypesVector =
      fbb_.CreateVector(templateArgumentListTypes);

  auto greaterLoc = encodeSourceLocation(ast->greaterLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::SimpleTemplateId::Builder builder{fbb_};
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_less_loc(lessLoc.o);
  builder.add_template_argument_list(templateArgumentListOffsetsVector);
  builder.add_template_argument_list_type(templateArgumentListTypesVector);
  builder.add_greater_loc(greaterLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_SimpleTemplateId;
}

void ASTEncoder::visit(LiteralOperatorTemplateIdAST* ast) {
  const auto literalOperatorId = accept(ast->literalOperatorId);

  auto lessLoc = encodeSourceLocation(ast->lessLoc);

  std::vector<flatbuffers::Offset<>> templateArgumentListOffsets;
  std::vector<std::underlying_type_t<io::TemplateArgument>>
      templateArgumentListTypes;

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptTemplateArgument(it->value);
    templateArgumentListOffsets.push_back(offset);
    templateArgumentListTypes.push_back(type);
  }

  auto templateArgumentListOffsetsVector =
      fbb_.CreateVector(templateArgumentListOffsets);
  auto templateArgumentListTypesVector =
      fbb_.CreateVector(templateArgumentListTypes);

  auto greaterLoc = encodeSourceLocation(ast->greaterLoc);

  io::LiteralOperatorTemplateId::Builder builder{fbb_};
  builder.add_literal_operator_id(literalOperatorId.o);
  builder.add_less_loc(lessLoc.o);
  builder.add_template_argument_list(templateArgumentListOffsetsVector);
  builder.add_template_argument_list_type(templateArgumentListTypesVector);
  builder.add_greater_loc(greaterLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_LiteralOperatorTemplateId;
}

void ASTEncoder::visit(OperatorFunctionTemplateIdAST* ast) {
  const auto operatorFunctionId = accept(ast->operatorFunctionId);

  auto lessLoc = encodeSourceLocation(ast->lessLoc);

  std::vector<flatbuffers::Offset<>> templateArgumentListOffsets;
  std::vector<std::underlying_type_t<io::TemplateArgument>>
      templateArgumentListTypes;

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptTemplateArgument(it->value);
    templateArgumentListOffsets.push_back(offset);
    templateArgumentListTypes.push_back(type);
  }

  auto templateArgumentListOffsetsVector =
      fbb_.CreateVector(templateArgumentListOffsets);
  auto templateArgumentListTypesVector =
      fbb_.CreateVector(templateArgumentListTypes);

  auto greaterLoc = encodeSourceLocation(ast->greaterLoc);

  io::OperatorFunctionTemplateId::Builder builder{fbb_};
  builder.add_operator_function_id(operatorFunctionId.o);
  builder.add_less_loc(lessLoc.o);
  builder.add_template_argument_list(templateArgumentListOffsetsVector);
  builder.add_template_argument_list_type(templateArgumentListTypesVector);
  builder.add_greater_loc(greaterLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_OperatorFunctionTemplateId;
}

void ASTEncoder::visit(GlobalNestedNameSpecifierAST* ast) {
  auto scopeLoc = encodeSourceLocation(ast->scopeLoc);

  io::GlobalNestedNameSpecifier::Builder builder{fbb_};
  builder.add_scope_loc(scopeLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::NestedNameSpecifier_GlobalNestedNameSpecifier;
}

void ASTEncoder::visit(SimpleNestedNameSpecifierAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  auto scopeLoc = encodeSourceLocation(ast->scopeLoc);

  io::SimpleNestedNameSpecifier::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_identifier_loc(identifierLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }
  builder.add_scope_loc(scopeLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::NestedNameSpecifier_SimpleNestedNameSpecifier;
}

void ASTEncoder::visit(DecltypeNestedNameSpecifierAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto decltypeSpecifier = accept(ast->decltypeSpecifier);

  auto scopeLoc = encodeSourceLocation(ast->scopeLoc);

  io::DecltypeNestedNameSpecifier::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_decltype_specifier(decltypeSpecifier.o);
  builder.add_scope_loc(scopeLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::NestedNameSpecifier_DecltypeNestedNameSpecifier;
}

void ASTEncoder::visit(TemplateNestedNameSpecifierAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  const auto templateId = accept(ast->templateId);

  auto scopeLoc = encodeSourceLocation(ast->scopeLoc);

  io::TemplateNestedNameSpecifier::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(templateLoc.o);
  builder.add_template_id(templateId.o);
  builder.add_scope_loc(scopeLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::NestedNameSpecifier_TemplateNestedNameSpecifier;
}

void ASTEncoder::visit(DefaultFunctionBodyAST* ast) {
  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  auto defaultLoc = encodeSourceLocation(ast->defaultLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::DefaultFunctionBody::Builder builder{fbb_};
  builder.add_equal_loc(equalLoc.o);
  builder.add_default_loc(defaultLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_DefaultFunctionBody;
}

void ASTEncoder::visit(CompoundStatementFunctionBodyAST* ast) {
  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  std::vector<flatbuffers::Offset<>> memInitializerListOffsets;
  std::vector<std::underlying_type_t<io::MemInitializer>>
      memInitializerListTypes;

  for (auto it = ast->memInitializerList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptMemInitializer(it->value);
    memInitializerListOffsets.push_back(offset);
    memInitializerListTypes.push_back(type);
  }

  auto memInitializerListOffsetsVector =
      fbb_.CreateVector(memInitializerListOffsets);
  auto memInitializerListTypesVector =
      fbb_.CreateVector(memInitializerListTypes);

  const auto statement = accept(ast->statement);

  io::CompoundStatementFunctionBody::Builder builder{fbb_};
  builder.add_colon_loc(colonLoc.o);
  builder.add_mem_initializer_list(memInitializerListOffsetsVector);
  builder.add_mem_initializer_list_type(memInitializerListTypesVector);
  builder.add_statement(statement.o);

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_CompoundStatementFunctionBody;
}

void ASTEncoder::visit(TryStatementFunctionBodyAST* ast) {
  auto tryLoc = encodeSourceLocation(ast->tryLoc);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  std::vector<flatbuffers::Offset<>> memInitializerListOffsets;
  std::vector<std::underlying_type_t<io::MemInitializer>>
      memInitializerListTypes;

  for (auto it = ast->memInitializerList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptMemInitializer(it->value);
    memInitializerListOffsets.push_back(offset);
    memInitializerListTypes.push_back(type);
  }

  auto memInitializerListOffsetsVector =
      fbb_.CreateVector(memInitializerListOffsets);
  auto memInitializerListTypesVector =
      fbb_.CreateVector(memInitializerListTypes);

  const auto statement = accept(ast->statement);

  std::vector<flatbuffers::Offset<io::Handler>> handlerListOffsets;
  for (auto it = ast->handlerList; it; it = it->next) {
    if (!it->value) continue;
    handlerListOffsets.emplace_back(accept(it->value).o);
  }

  auto handlerListOffsetsVector = fbb_.CreateVector(handlerListOffsets);

  io::TryStatementFunctionBody::Builder builder{fbb_};
  builder.add_try_loc(tryLoc.o);
  builder.add_colon_loc(colonLoc.o);
  builder.add_mem_initializer_list(memInitializerListOffsetsVector);
  builder.add_mem_initializer_list_type(memInitializerListTypesVector);
  builder.add_statement(statement.o);
  builder.add_handler_list(handlerListOffsetsVector);

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_TryStatementFunctionBody;
}

void ASTEncoder::visit(DeleteFunctionBodyAST* ast) {
  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  auto deleteLoc = encodeSourceLocation(ast->deleteLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::DeleteFunctionBody::Builder builder{fbb_};
  builder.add_equal_loc(equalLoc.o);
  builder.add_delete_loc(deleteLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_DeleteFunctionBody;
}

void ASTEncoder::visit(TypeTemplateArgumentAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::TypeTemplateArgument::Builder builder{fbb_};
  builder.add_type_id(typeId.o);

  offset_ = builder.Finish().Union();
  type_ = io::TemplateArgument_TypeTemplateArgument;
}

void ASTEncoder::visit(ExpressionTemplateArgumentAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::ExpressionTemplateArgument::Builder builder{fbb_};
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::TemplateArgument_ExpressionTemplateArgument;
}

void ASTEncoder::visit(ThrowExceptionSpecifierAST* ast) {
  auto throwLoc = encodeSourceLocation(ast->throwLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::ThrowExceptionSpecifier::Builder builder{fbb_};
  builder.add_throw_loc(throwLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::ExceptionSpecifier_ThrowExceptionSpecifier;
}

void ASTEncoder::visit(NoexceptSpecifierAST* ast) {
  auto noexceptLoc = encodeSourceLocation(ast->noexceptLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::NoexceptSpecifier::Builder builder{fbb_};
  builder.add_noexcept_loc(noexceptLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::ExceptionSpecifier_NoexceptSpecifier;
}

void ASTEncoder::visit(SimpleRequirementAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::SimpleRequirement::Builder builder{fbb_};
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_SimpleRequirement;
}

void ASTEncoder::visit(CompoundRequirementAST* ast) {
  auto lbraceLoc = encodeSourceLocation(ast->lbraceLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto rbraceLoc = encodeSourceLocation(ast->rbraceLoc);

  auto noexceptLoc = encodeSourceLocation(ast->noexceptLoc);

  auto minusGreaterLoc = encodeSourceLocation(ast->minusGreaterLoc);

  const auto typeConstraint = accept(ast->typeConstraint);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::CompoundRequirement::Builder builder{fbb_};
  builder.add_lbrace_loc(lbraceLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rbrace_loc(rbraceLoc.o);
  builder.add_noexcept_loc(noexceptLoc.o);
  builder.add_minus_greater_loc(minusGreaterLoc.o);
  builder.add_type_constraint(typeConstraint.o);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_CompoundRequirement;
}

void ASTEncoder::visit(TypeRequirementAST* ast) {
  auto typenameLoc = encodeSourceLocation(ast->typenameLoc);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::TypeRequirement::Builder builder{fbb_};
  builder.add_typename_loc(typenameLoc.o);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_TypeRequirement;
}

void ASTEncoder::visit(NestedRequirementAST* ast) {
  auto requiresLoc = encodeSourceLocation(ast->requiresLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::NestedRequirement::Builder builder{fbb_};
  builder.add_requires_loc(requiresLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_NestedRequirement;
}

void ASTEncoder::visit(NewParenInitializerAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptExpression(it->value);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::NewParenInitializer::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::NewInitializer_NewParenInitializer;
}

void ASTEncoder::visit(NewBracedInitializerAST* ast) {
  const auto bracedInitList = accept(ast->bracedInitList);

  io::NewBracedInitializer::Builder builder{fbb_};
  builder.add_braced_init_list(bracedInitList.o);

  offset_ = builder.Finish().Union();
  type_ = io::NewInitializer_NewBracedInitializer;
}

void ASTEncoder::visit(ParenMemInitializerAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptExpression(it->value);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  io::ParenMemInitializer::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::MemInitializer_ParenMemInitializer;
}

void ASTEncoder::visit(BracedMemInitializerAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  const auto bracedInitList = accept(ast->bracedInitList);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  io::BracedMemInitializer::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_braced_init_list(bracedInitList.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::MemInitializer_BracedMemInitializer;
}

void ASTEncoder::visit(ThisLambdaCaptureAST* ast) {
  auto thisLoc = encodeSourceLocation(ast->thisLoc);

  io::ThisLambdaCapture::Builder builder{fbb_};
  builder.add_this_loc(thisLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_ThisLambdaCapture;
}

void ASTEncoder::visit(DerefThisLambdaCaptureAST* ast) {
  auto starLoc = encodeSourceLocation(ast->starLoc);

  auto thisLoc = encodeSourceLocation(ast->thisLoc);

  io::DerefThisLambdaCapture::Builder builder{fbb_};
  builder.add_star_loc(starLoc.o);
  builder.add_this_loc(thisLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_DerefThisLambdaCapture;
}

void ASTEncoder::visit(SimpleLambdaCaptureAST* ast) {
  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::SimpleLambdaCapture::Builder builder{fbb_};
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_SimpleLambdaCapture;
}

void ASTEncoder::visit(RefLambdaCaptureAST* ast) {
  auto ampLoc = encodeSourceLocation(ast->ampLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::RefLambdaCapture::Builder builder{fbb_};
  builder.add_amp_loc(ampLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_RefLambdaCapture;
}

void ASTEncoder::visit(RefInitLambdaCaptureAST* ast) {
  auto ampLoc = encodeSourceLocation(ast->ampLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  const auto [initializer, initializerType] =
      acceptExpression(ast->initializer);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::RefInitLambdaCapture::Builder builder{fbb_};
  builder.add_amp_loc(ampLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_RefInitLambdaCapture;
}

void ASTEncoder::visit(InitLambdaCaptureAST* ast) {
  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  const auto [initializer, initializerType] =
      acceptExpression(ast->initializer);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::InitLambdaCapture::Builder builder{fbb_};
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_InitLambdaCapture;
}

void ASTEncoder::visit(EllipsisExceptionDeclarationAST* ast) {
  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  io::EllipsisExceptionDeclaration::Builder builder{fbb_};
  builder.add_ellipsis_loc(ellipsisLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::ExceptionDeclaration_EllipsisExceptionDeclaration;
}

void ASTEncoder::visit(TypeExceptionDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  const auto declarator = accept(ast->declarator);

  io::TypeExceptionDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_declarator(declarator.o);

  offset_ = builder.Finish().Union();
  type_ = io::ExceptionDeclaration_TypeExceptionDeclaration;
}

void ASTEncoder::visit(CxxAttributeAST* ast) {
  auto lbracketLoc = encodeSourceLocation(ast->lbracketLoc);

  auto lbracket2Loc = encodeSourceLocation(ast->lbracket2Loc);

  const auto attributeUsingPrefix = accept(ast->attributeUsingPrefix);

  std::vector<flatbuffers::Offset<io::Attribute>> attributeListOffsets;
  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    attributeListOffsets.emplace_back(accept(it->value).o);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);

  auto rbracketLoc = encodeSourceLocation(ast->rbracketLoc);

  auto rbracket2Loc = encodeSourceLocation(ast->rbracket2Loc);

  io::CxxAttribute::Builder builder{fbb_};
  builder.add_lbracket_loc(lbracketLoc.o);
  builder.add_lbracket2_loc(lbracket2Loc.o);
  builder.add_attribute_using_prefix(attributeUsingPrefix.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_rbracket_loc(rbracketLoc.o);
  builder.add_rbracket2_loc(rbracket2Loc.o);

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_CxxAttribute;
}

void ASTEncoder::visit(GccAttributeAST* ast) {
  auto attributeLoc = encodeSourceLocation(ast->attributeLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  auto lparen2Loc = encodeSourceLocation(ast->lparen2Loc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  auto rparen2Loc = encodeSourceLocation(ast->rparen2Loc);

  io::GccAttribute::Builder builder{fbb_};
  builder.add_attribute_loc(attributeLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_lparen2_loc(lparen2Loc.o);
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_rparen2_loc(rparen2Loc.o);

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_GccAttribute;
}

void ASTEncoder::visit(AlignasAttributeAST* ast) {
  auto alignasLoc = encodeSourceLocation(ast->alignasLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::AlignasAttribute::Builder builder{fbb_};
  builder.add_alignas_loc(alignasLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_AlignasAttribute;
}

void ASTEncoder::visit(AlignasTypeAttributeAST* ast) {
  auto alignasLoc = encodeSourceLocation(ast->alignasLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto typeId = accept(ast->typeId);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::AlignasTypeAttribute::Builder builder{fbb_};
  builder.add_alignas_loc(alignasLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_type_id(typeId.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_AlignasTypeAttribute;
}

void ASTEncoder::visit(AsmAttributeAST* ast) {
  auto asmLoc = encodeSourceLocation(ast->asmLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  auto literalLoc = encodeSourceLocation(ast->literalLoc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::AsmAttribute::Builder builder{fbb_};
  builder.add_asm_loc(asmLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_literal_loc(literalLoc.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_AsmAttribute;
}

void ASTEncoder::visit(ScopedAttributeTokenAST* ast) {
  auto attributeNamespaceLoc = encodeSourceLocation(ast->attributeNamespaceLoc);

  auto scopeLoc = encodeSourceLocation(ast->scopeLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  flatbuffers::Offset<flatbuffers::String> attributeNamespace;
  if (ast->attributeNamespace) {
    if (identifiers_.contains(ast->attributeNamespace)) {
      attributeNamespace = identifiers_.at(ast->attributeNamespace);
    } else {
      attributeNamespace = fbb_.CreateString(ast->attributeNamespace->value());
      identifiers_.emplace(ast->attributeNamespace, attributeNamespace);
    }
  }

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::ScopedAttributeToken::Builder builder{fbb_};
  builder.add_attribute_namespace_loc(attributeNamespaceLoc.o);
  builder.add_scope_loc(scopeLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  if (ast->attributeNamespace) {
    builder.add_attribute_namespace(attributeNamespace);
  }
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::AttributeToken_ScopedAttributeToken;
}

void ASTEncoder::visit(SimpleAttributeTokenAST* ast) {
  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::SimpleAttributeToken::Builder builder{fbb_};
  builder.add_identifier_loc(identifierLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::AttributeToken_SimpleAttributeToken;
}

void ASTEncoder::visit(GlobalModuleFragmentAST* ast) {
  auto moduleLoc = encodeSourceLocation(ast->moduleLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto it = ast->declarationList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptDeclaration(it->value);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  io::GlobalModuleFragment::Builder builder{fbb_};
  builder.add_module_loc(moduleLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(PrivateModuleFragmentAST* ast) {
  auto moduleLoc = encodeSourceLocation(ast->moduleLoc);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  auto privateLoc = encodeSourceLocation(ast->privateLoc);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto it = ast->declarationList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptDeclaration(it->value);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  io::PrivateModuleFragment::Builder builder{fbb_};
  builder.add_module_loc(moduleLoc.o);
  builder.add_colon_loc(colonLoc.o);
  builder.add_private_loc(privateLoc.o);
  builder.add_semicolon_loc(semicolonLoc.o);
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModuleDeclarationAST* ast) {
  auto exportLoc = encodeSourceLocation(ast->exportLoc);

  auto moduleLoc = encodeSourceLocation(ast->moduleLoc);

  const auto moduleName = accept(ast->moduleName);

  const auto modulePartition = accept(ast->modulePartition);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  auto semicolonLoc = encodeSourceLocation(ast->semicolonLoc);

  io::ModuleDeclaration::Builder builder{fbb_};
  builder.add_export_loc(exportLoc.o);
  builder.add_module_loc(moduleLoc.o);
  builder.add_module_name(moduleName.o);
  builder.add_module_partition(modulePartition.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_semicolon_loc(semicolonLoc.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModuleNameAST* ast) {
  const auto moduleQualifier = accept(ast->moduleQualifier);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::ModuleName::Builder builder{fbb_};
  builder.add_module_qualifier(moduleQualifier.o);
  builder.add_identifier_loc(identifierLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModuleQualifierAST* ast) {
  const auto moduleQualifier = accept(ast->moduleQualifier);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto dotLoc = encodeSourceLocation(ast->dotLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::ModuleQualifier::Builder builder{fbb_};
  builder.add_module_qualifier(moduleQualifier.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_dot_loc(dotLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModulePartitionAST* ast) {
  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  const auto moduleName = accept(ast->moduleName);

  io::ModulePartition::Builder builder{fbb_};
  builder.add_colon_loc(colonLoc.o);
  builder.add_module_name(moduleName.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ImportNameAST* ast) {
  auto headerLoc = encodeSourceLocation(ast->headerLoc);

  const auto modulePartition = accept(ast->modulePartition);

  const auto moduleName = accept(ast->moduleName);

  io::ImportName::Builder builder{fbb_};
  builder.add_header_loc(headerLoc.o);
  builder.add_module_partition(modulePartition.o);
  builder.add_module_name(moduleName.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(InitDeclaratorAST* ast) {
  const auto declarator = accept(ast->declarator);

  const auto requiresClause = accept(ast->requiresClause);

  const auto [initializer, initializerType] =
      acceptExpression(ast->initializer);

  io::InitDeclarator::Builder builder{fbb_};
  builder.add_declarator(declarator.o);
  builder.add_requires_clause(requiresClause.o);
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(DeclaratorAST* ast) {
  std::vector<flatbuffers::Offset<>> ptrOpListOffsets;
  std::vector<std::underlying_type_t<io::PtrOperator>> ptrOpListTypes;

  for (auto it = ast->ptrOpList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptPtrOperator(it->value);
    ptrOpListOffsets.push_back(offset);
    ptrOpListTypes.push_back(type);
  }

  auto ptrOpListOffsetsVector = fbb_.CreateVector(ptrOpListOffsets);
  auto ptrOpListTypesVector = fbb_.CreateVector(ptrOpListTypes);

  const auto [coreDeclarator, coreDeclaratorType] =
      acceptCoreDeclarator(ast->coreDeclarator);

  std::vector<flatbuffers::Offset<>> declaratorChunkListOffsets;
  std::vector<std::underlying_type_t<io::DeclaratorChunk>>
      declaratorChunkListTypes;

  for (auto it = ast->declaratorChunkList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptDeclaratorChunk(it->value);
    declaratorChunkListOffsets.push_back(offset);
    declaratorChunkListTypes.push_back(type);
  }

  auto declaratorChunkListOffsetsVector =
      fbb_.CreateVector(declaratorChunkListOffsets);
  auto declaratorChunkListTypesVector =
      fbb_.CreateVector(declaratorChunkListTypes);

  io::Declarator::Builder builder{fbb_};
  builder.add_ptr_op_list(ptrOpListOffsetsVector);
  builder.add_ptr_op_list_type(ptrOpListTypesVector);
  builder.add_core_declarator(coreDeclarator);
  builder.add_core_declarator_type(
      static_cast<io::CoreDeclarator>(coreDeclaratorType));
  builder.add_declarator_chunk_list(declaratorChunkListOffsetsVector);
  builder.add_declarator_chunk_list_type(declaratorChunkListTypesVector);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(UsingDeclaratorAST* ast) {
  auto typenameLoc = encodeSourceLocation(ast->typenameLoc);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  io::UsingDeclarator::Builder builder{fbb_};
  builder.add_typename_loc(typenameLoc.o);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_ellipsis_loc(ellipsisLoc.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(EnumeratorAST* ast) {
  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  auto equalLoc = encodeSourceLocation(ast->equalLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::Enumerator::Builder builder{fbb_};
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_equal_loc(equalLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(TypeIdAST* ast) {
  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptSpecifier(it->value);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  const auto declarator = accept(ast->declarator);

  io::TypeId::Builder builder{fbb_};
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_declarator(declarator.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(HandlerAST* ast) {
  auto catchLoc = encodeSourceLocation(ast->catchLoc);

  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  const auto [exceptionDeclaration, exceptionDeclarationType] =
      acceptExceptionDeclaration(ast->exceptionDeclaration);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  const auto statement = accept(ast->statement);

  io::Handler::Builder builder{fbb_};
  builder.add_catch_loc(catchLoc.o);
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_exception_declaration(exceptionDeclaration);
  builder.add_exception_declaration_type(
      static_cast<io::ExceptionDeclaration>(exceptionDeclarationType));
  builder.add_rparen_loc(rparenLoc.o);
  builder.add_statement(statement.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(BaseSpecifierAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto it = ast->attributeList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptAttributeSpecifier(it->value);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  auto templateLoc = encodeSourceLocation(ast->templateLoc);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::BaseSpecifier::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(templateLoc.o);
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_access_specifier(
      static_cast<std::uint32_t>(ast->accessSpecifier));

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(RequiresClauseAST* ast) {
  auto requiresLoc = encodeSourceLocation(ast->requiresLoc);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::RequiresClause::Builder builder{fbb_};
  builder.add_requires_loc(requiresLoc.o);
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ParameterDeclarationClauseAST* ast) {
  std::vector<flatbuffers::Offset<io::ParameterDeclaration>>
      parameterDeclarationListOffsets;
  for (auto it = ast->parameterDeclarationList; it; it = it->next) {
    if (!it->value) continue;
    parameterDeclarationListOffsets.emplace_back(accept(it->value).o);
  }

  auto parameterDeclarationListOffsetsVector =
      fbb_.CreateVector(parameterDeclarationListOffsets);

  auto commaLoc = encodeSourceLocation(ast->commaLoc);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  io::ParameterDeclarationClause::Builder builder{fbb_};
  builder.add_parameter_declaration_list(parameterDeclarationListOffsetsVector);
  builder.add_comma_loc(commaLoc.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(TrailingReturnTypeAST* ast) {
  auto minusGreaterLoc = encodeSourceLocation(ast->minusGreaterLoc);

  const auto typeId = accept(ast->typeId);

  io::TrailingReturnType::Builder builder{fbb_};
  builder.add_minus_greater_loc(minusGreaterLoc.o);
  builder.add_type_id(typeId.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(LambdaSpecifierAST* ast) {
  auto specifierLoc = encodeSourceLocation(ast->specifierLoc);

  io::LambdaSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(specifierLoc.o);
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(TypeConstraintAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto lessLoc = encodeSourceLocation(ast->lessLoc);

  std::vector<flatbuffers::Offset<>> templateArgumentListOffsets;
  std::vector<std::underlying_type_t<io::TemplateArgument>>
      templateArgumentListTypes;

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptTemplateArgument(it->value);
    templateArgumentListOffsets.push_back(offset);
    templateArgumentListTypes.push_back(type);
  }

  auto templateArgumentListOffsetsVector =
      fbb_.CreateVector(templateArgumentListOffsets);
  auto templateArgumentListTypesVector =
      fbb_.CreateVector(templateArgumentListTypes);

  auto greaterLoc = encodeSourceLocation(ast->greaterLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::TypeConstraint::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_less_loc(lessLoc.o);
  builder.add_template_argument_list(templateArgumentListOffsetsVector);
  builder.add_template_argument_list_type(templateArgumentListTypesVector);
  builder.add_greater_loc(greaterLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(AttributeArgumentClauseAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::AttributeArgumentClause::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(AttributeAST* ast) {
  const auto [attributeToken, attributeTokenType] =
      acceptAttributeToken(ast->attributeToken);

  const auto attributeArgumentClause = accept(ast->attributeArgumentClause);

  auto ellipsisLoc = encodeSourceLocation(ast->ellipsisLoc);

  io::Attribute::Builder builder{fbb_};
  builder.add_attribute_token(attributeToken);
  builder.add_attribute_token_type(
      static_cast<io::AttributeToken>(attributeTokenType));
  builder.add_attribute_argument_clause(attributeArgumentClause.o);
  builder.add_ellipsis_loc(ellipsisLoc.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(AttributeUsingPrefixAST* ast) {
  auto usingLoc = encodeSourceLocation(ast->usingLoc);

  auto attributeNamespaceLoc = encodeSourceLocation(ast->attributeNamespaceLoc);

  auto colonLoc = encodeSourceLocation(ast->colonLoc);

  io::AttributeUsingPrefix::Builder builder{fbb_};
  builder.add_using_loc(usingLoc.o);
  builder.add_attribute_namespace_loc(attributeNamespaceLoc.o);
  builder.add_colon_loc(colonLoc.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(NewPlacementAST* ast) {
  auto lparenLoc = encodeSourceLocation(ast->lparenLoc);

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto it = ast->expressionList; it; it = it->next) {
    if (!it->value) continue;
    const auto [offset, type] = acceptExpression(it->value);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  auto rparenLoc = encodeSourceLocation(ast->rparenLoc);

  io::NewPlacement::Builder builder{fbb_};
  builder.add_lparen_loc(lparenLoc.o);
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(rparenLoc.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(NestedNamespaceSpecifierAST* ast) {
  auto inlineLoc = encodeSourceLocation(ast->inlineLoc);

  auto identifierLoc = encodeSourceLocation(ast->identifierLoc);

  auto scopeLoc = encodeSourceLocation(ast->scopeLoc);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::NestedNamespaceSpecifier::Builder builder{fbb_};
  builder.add_inline_loc(inlineLoc.o);
  builder.add_identifier_loc(identifierLoc.o);
  builder.add_scope_loc(scopeLoc.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
}

}  // namespace cxx
