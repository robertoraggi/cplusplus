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

#include <cxx/private/ast_encoder.h>

// cxx
#include <cxx-ast-flatbuffers/ast_generated.h>
#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/translation_unit.h>

#include <algorithm>
#include <format>

namespace cxx {

auto ASTEncoder::operator()(TranslationUnit* unit)
    -> std::span<const std::uint8_t> {
  if (!unit) return {};
  Table<Identifier> identifiers;
  Table<CharLiteral> charLiterals;
  Table<StringLiteral> stringLiterals;
  Table<IntegerLiteral> integerLiterals;
  Table<FloatLiteral> floatLiterals;

  std::swap(unit_, unit);
  std::swap(identifiers_, identifiers);
  std::swap(charLiterals_, charLiterals);
  std::swap(stringLiterals_, stringLiterals);
  std::swap(integerLiterals_, integerLiterals);
  std::swap(floatLiterals_, floatLiterals);

  std::vector<flatbuffers::Offset<io::Source>> sources;
  for (const auto& source : unit_->preprocessor()->sources()) {
    auto file_name = fbb_.CreateString(source.fileName);
    auto line_offsets =
        fbb_.CreateVector(source.lineOffsets.data(), source.lineOffsets.size());
    sources.push_back(io::CreateSource(fbb_, file_name, line_offsets));
  }

  auto source_list = fbb_.CreateVector(sources);

  std::vector<std::uint64_t> tokens;
  for (std::uint32_t i = 0; i < unit_->tokenCount(); ++i) {
    const auto& token = unit_->tokenAt(SourceLocation(i));
    tokens.push_back(token.raw());
  }

  auto token_list = fbb_.CreateVector(tokens);

  auto [unitOffset, unitType] = acceptUnit(unit_->ast());

  auto file_name = fbb_.CreateString(unit_->fileName());

  io::SerializedUnitBuilder builder{fbb_};
  builder.add_unit(unitOffset);
  builder.add_unit_type(static_cast<io::Unit>(unitType));
  builder.add_file_name(file_name);
  builder.add_source_list(source_list);
  builder.add_token_list(token_list);

  std::swap(unit_, unit);
  std::swap(identifiers_, identifiers);
  std::swap(charLiterals_, charLiterals);
  std::swap(stringLiterals_, stringLiterals);
  std::swap(integerLiterals_, integerLiterals);
  std::swap(floatLiterals_, floatLiterals);

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

  for (auto node : ListView{ast->declarationList}) {
    if (!node) continue;
    const auto [offset, type] = acceptDeclaration(node);
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

  for (auto node : ListView{ast->declarationList}) {
    if (!node) continue;
    const auto [offset, type] = acceptDeclaration(node);
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

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;

  for (auto node : ListView{ast->declSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
    declSpecifierListOffsets.push_back(offset);
    declSpecifierListTypes.push_back(type);
  }

  auto declSpecifierListOffsetsVector =
      fbb_.CreateVector(declSpecifierListOffsets);
  auto declSpecifierListTypesVector = fbb_.CreateVector(declSpecifierListTypes);

  std::vector<flatbuffers::Offset<io::InitDeclarator>>
      initDeclaratorListOffsets;
  for (auto node : ListView{ast->initDeclaratorList}) {
    if (!node) continue;
    initDeclaratorListOffsets.emplace_back(accept(node).o);
  }

  auto initDeclaratorListOffsetsVector =
      fbb_.CreateVector(initDeclaratorListOffsets);

  const auto requiresClause = accept(ast->requiresClause);

  io::SimpleDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_decl_specifier_list(declSpecifierListOffsetsVector);
  builder.add_decl_specifier_list_type(declSpecifierListTypesVector);
  builder.add_init_declarator_list(initDeclaratorListOffsetsVector);
  builder.add_requires_clause(requiresClause.o);
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_SimpleDeclaration;
}

void ASTEncoder::visit(AsmDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<io::AsmQualifier>> asmQualifierListOffsets;
  for (auto node : ListView{ast->asmQualifierList}) {
    if (!node) continue;
    asmQualifierListOffsets.emplace_back(accept(node).o);
  }

  auto asmQualifierListOffsetsVector =
      fbb_.CreateVector(asmQualifierListOffsets);

  std::vector<flatbuffers::Offset<io::AsmOperand>> outputOperandListOffsets;
  for (auto node : ListView{ast->outputOperandList}) {
    if (!node) continue;
    outputOperandListOffsets.emplace_back(accept(node).o);
  }

  auto outputOperandListOffsetsVector =
      fbb_.CreateVector(outputOperandListOffsets);

  std::vector<flatbuffers::Offset<io::AsmOperand>> inputOperandListOffsets;
  for (auto node : ListView{ast->inputOperandList}) {
    if (!node) continue;
    inputOperandListOffsets.emplace_back(accept(node).o);
  }

  auto inputOperandListOffsetsVector =
      fbb_.CreateVector(inputOperandListOffsets);

  std::vector<flatbuffers::Offset<io::AsmClobber>> clobberListOffsets;
  for (auto node : ListView{ast->clobberList}) {
    if (!node) continue;
    clobberListOffsets.emplace_back(accept(node).o);
  }

  auto clobberListOffsetsVector = fbb_.CreateVector(clobberListOffsets);

  std::vector<flatbuffers::Offset<io::AsmGotoLabel>> gotoLabelListOffsets;
  for (auto node : ListView{ast->gotoLabelList}) {
    if (!node) continue;
    gotoLabelListOffsets.emplace_back(accept(node).o);
  }

  auto gotoLabelListOffsetsVector = fbb_.CreateVector(gotoLabelListOffsets);

  io::AsmDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_asm_qualifier_list(asmQualifierListOffsetsVector);
  builder.add_asm_loc(ast->asmLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_literal_loc(ast->literalLoc.index());
  builder.add_output_operand_list(outputOperandListOffsetsVector);
  builder.add_input_operand_list(inputOperandListOffsetsVector);
  builder.add_clobber_list(clobberListOffsetsVector);
  builder.add_goto_label_list(gotoLabelListOffsetsVector);
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmDeclaration;
}

void ASTEncoder::visit(NamespaceAliasDefinitionAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto unqualifiedId = accept(ast->unqualifiedId);

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
  builder.add_namespace_loc(ast->namespaceLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_semicolon_loc(ast->semicolonLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_NamespaceAliasDefinition;
}

void ASTEncoder::visit(UsingDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<io::UsingDeclarator>>
      usingDeclaratorListOffsets;
  for (auto node : ListView{ast->usingDeclaratorList}) {
    if (!node) continue;
    usingDeclaratorListOffsets.emplace_back(accept(node).o);
  }

  auto usingDeclaratorListOffsetsVector =
      fbb_.CreateVector(usingDeclaratorListOffsets);

  io::UsingDeclaration::Builder builder{fbb_};
  builder.add_using_loc(ast->usingLoc.index());
  builder.add_using_declarator_list(usingDeclaratorListOffsetsVector);
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_UsingDeclaration;
}

void ASTEncoder::visit(UsingEnumDeclarationAST* ast) {
  const auto enumTypeSpecifier = accept(ast->enumTypeSpecifier);

  io::UsingEnumDeclaration::Builder builder{fbb_};
  builder.add_using_loc(ast->usingLoc.index());
  builder.add_enum_type_specifier(enumTypeSpecifier.o);
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_UsingEnumDeclaration;
}

void ASTEncoder::visit(UsingDirectiveAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto unqualifiedId = accept(ast->unqualifiedId);

  io::UsingDirective::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_using_loc(ast->usingLoc.index());
  builder.add_namespace_loc(ast->namespaceLoc.index());
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_UsingDirective;
}

void ASTEncoder::visit(StaticAssertDeclarationAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::StaticAssertDeclaration::Builder builder{fbb_};
  builder.add_static_assert_loc(ast->staticAssertLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_comma_loc(ast->commaLoc.index());
  builder.add_literal_loc(ast->literalLoc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_StaticAssertDeclaration;
}

void ASTEncoder::visit(AliasDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> gnuAttributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      gnuAttributeListTypes;

  for (auto node : ListView{ast->gnuAttributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    gnuAttributeListOffsets.push_back(offset);
    gnuAttributeListTypes.push_back(type);
  }

  auto gnuAttributeListOffsetsVector =
      fbb_.CreateVector(gnuAttributeListOffsets);
  auto gnuAttributeListTypesVector = fbb_.CreateVector(gnuAttributeListTypes);

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

  io::AliasDeclaration::Builder builder{fbb_};
  builder.add_using_loc(ast->usingLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_gnu_attribute_list(gnuAttributeListOffsetsVector);
  builder.add_gnu_attribute_list_type(gnuAttributeListTypesVector);
  builder.add_type_id(typeId.o);
  builder.add_semicolon_loc(ast->semicolonLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AliasDeclaration;
}

void ASTEncoder::visit(OpaqueEnumDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto unqualifiedId = accept(ast->unqualifiedId);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto node : ListView{ast->typeSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  io::OpaqueEnumDeclaration::Builder builder{fbb_};
  builder.add_enum_loc(ast->enumLoc.index());
  builder.add_class_loc(ast->classLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_emicolon_loc(ast->emicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_OpaqueEnumDeclaration;
}

void ASTEncoder::visit(FunctionDefinitionAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;

  for (auto node : ListView{ast->declSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
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
  std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
  std::vector<std::underlying_type_t<io::TemplateParameter>>
      templateParameterListTypes;

  for (auto node : ListView{ast->templateParameterList}) {
    if (!node) continue;
    const auto [offset, type] = acceptTemplateParameter(node);
    templateParameterListOffsets.push_back(offset);
    templateParameterListTypes.push_back(type);
  }

  auto templateParameterListOffsetsVector =
      fbb_.CreateVector(templateParameterListOffsets);
  auto templateParameterListTypesVector =
      fbb_.CreateVector(templateParameterListTypes);

  const auto requiresClause = accept(ast->requiresClause);

  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  io::TemplateDeclaration::Builder builder{fbb_};
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_less_loc(ast->lessLoc.index());
  builder.add_template_parameter_list(templateParameterListOffsetsVector);
  builder.add_template_parameter_list_type(templateParameterListTypesVector);
  builder.add_greater_loc(ast->greaterLoc.index());
  builder.add_requires_clause(requiresClause.o);
  builder.add_declaration(declaration);
  builder.add_declaration_type(static_cast<io::Declaration>(declarationType));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_TemplateDeclaration;
}

void ASTEncoder::visit(ConceptDefinitionAST* ast) {
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

  io::ConceptDefinition::Builder builder{fbb_};
  builder.add_concept_loc(ast->conceptLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(ast->semicolonLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ConceptDefinition;
}

void ASTEncoder::visit(DeductionGuideAST* ast) {
  const auto [explicitSpecifier, explicitSpecifierType] =
      acceptSpecifier(ast->explicitSpecifier);

  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  const auto templateId = accept(ast->templateId);

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
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_arrow_loc(ast->arrowLoc.index());
  builder.add_template_id(templateId.o);
  builder.add_semicolon_loc(ast->semicolonLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_DeductionGuide;
}

void ASTEncoder::visit(ExplicitInstantiationAST* ast) {
  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  io::ExplicitInstantiation::Builder builder{fbb_};
  builder.add_extern_loc(ast->externLoc.index());
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_declaration(declaration);
  builder.add_declaration_type(static_cast<io::Declaration>(declarationType));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ExplicitInstantiation;
}

void ASTEncoder::visit(ExportDeclarationAST* ast) {
  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  io::ExportDeclaration::Builder builder{fbb_};
  builder.add_export_loc(ast->exportLoc.index());
  builder.add_declaration(declaration);
  builder.add_declaration_type(static_cast<io::Declaration>(declarationType));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ExportDeclaration;
}

void ASTEncoder::visit(ExportCompoundDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto node : ListView{ast->declarationList}) {
    if (!node) continue;
    const auto [offset, type] = acceptDeclaration(node);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  io::ExportCompoundDeclaration::Builder builder{fbb_};
  builder.add_export_loc(ast->exportLoc.index());
  builder.add_lbrace_loc(ast->lbraceLoc.index());
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);
  builder.add_rbrace_loc(ast->rbraceLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ExportCompoundDeclaration;
}

void ASTEncoder::visit(LinkageSpecificationAST* ast) {
  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto node : ListView{ast->declarationList}) {
    if (!node) continue;
    const auto [offset, type] = acceptDeclaration(node);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

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
  builder.add_extern_loc(ast->externLoc.index());
  builder.add_stringliteral_loc(ast->stringliteralLoc.index());
  builder.add_lbrace_loc(ast->lbraceLoc.index());
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);
  builder.add_rbrace_loc(ast->rbraceLoc.index());
  if (ast->stringLiteral) {
    builder.add_string_literal(stringLiteral);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_LinkageSpecification;
}

void ASTEncoder::visit(NamespaceDefinitionAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<io::NestedNamespaceSpecifier>>
      nestedNamespaceSpecifierListOffsets;
  for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    if (!node) continue;
    nestedNamespaceSpecifierListOffsets.emplace_back(accept(node).o);
  }

  auto nestedNamespaceSpecifierListOffsetsVector =
      fbb_.CreateVector(nestedNamespaceSpecifierListOffsets);

  std::vector<flatbuffers::Offset<>> extraAttributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      extraAttributeListTypes;

  for (auto node : ListView{ast->extraAttributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    extraAttributeListOffsets.push_back(offset);
    extraAttributeListTypes.push_back(type);
  }

  auto extraAttributeListOffsetsVector =
      fbb_.CreateVector(extraAttributeListOffsets);
  auto extraAttributeListTypesVector =
      fbb_.CreateVector(extraAttributeListTypes);

  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto node : ListView{ast->declarationList}) {
    if (!node) continue;
    const auto [offset, type] = acceptDeclaration(node);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

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
  builder.add_inline_loc(ast->inlineLoc.index());
  builder.add_namespace_loc(ast->namespaceLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_namespace_specifier_list(
      nestedNamespaceSpecifierListOffsetsVector);
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_extra_attribute_list(extraAttributeListOffsetsVector);
  builder.add_extra_attribute_list_type(extraAttributeListTypesVector);
  builder.add_lbrace_loc(ast->lbraceLoc.index());
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);
  builder.add_rbrace_loc(ast->rbraceLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_NamespaceDefinition;
}

void ASTEncoder::visit(EmptyDeclarationAST* ast) {
  io::EmptyDeclaration::Builder builder{fbb_};
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_EmptyDeclaration;
}

void ASTEncoder::visit(AttributeDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  io::AttributeDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AttributeDeclaration;
}

void ASTEncoder::visit(ModuleImportDeclarationAST* ast) {
  const auto importName = accept(ast->importName);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  io::ModuleImportDeclaration::Builder builder{fbb_};
  builder.add_import_loc(ast->importLoc.index());
  builder.add_import_name(importName.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ModuleImportDeclaration;
}

void ASTEncoder::visit(ParameterDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto node : ListView{ast->typeSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  const auto declarator = accept(ast->declarator);

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

  io::ParameterDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_this_loc(ast->thisLoc.index());
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_declarator(declarator.o);
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ParameterDeclaration;
}

void ASTEncoder::visit(AccessDeclarationAST* ast) {
  io::AccessDeclaration::Builder builder{fbb_};
  builder.add_access_loc(ast->accessLoc.index());
  builder.add_colon_loc(ast->colonLoc.index());
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

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;

  for (auto node : ListView{ast->declSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
    declSpecifierListOffsets.push_back(offset);
    declSpecifierListTypes.push_back(type);
  }

  auto declSpecifierListOffsetsVector =
      fbb_.CreateVector(declSpecifierListOffsets);
  auto declSpecifierListTypesVector = fbb_.CreateVector(declSpecifierListTypes);

  std::vector<flatbuffers::Offset<io::NameId>> bindingListOffsets;
  for (auto node : ListView{ast->bindingList}) {
    if (!node) continue;
    bindingListOffsets.emplace_back(accept(node).o);
  }

  auto bindingListOffsetsVector = fbb_.CreateVector(bindingListOffsets);

  const auto [initializer, initializerType] =
      acceptExpression(ast->initializer);

  io::StructuredBindingDeclaration::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_decl_specifier_list(declSpecifierListOffsetsVector);
  builder.add_decl_specifier_list_type(declSpecifierListTypesVector);
  builder.add_ref_qualifier_loc(ast->refQualifierLoc.index());
  builder.add_lbracket_loc(ast->lbracketLoc.index());
  builder.add_binding_list(bindingListOffsetsVector);
  builder.add_rbracket_loc(ast->rbracketLoc.index());
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_StructuredBindingDeclaration;
}

void ASTEncoder::visit(AsmOperandAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

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
  builder.add_lbracket_loc(ast->lbracketLoc.index());
  builder.add_symbolic_name_loc(ast->symbolicNameLoc.index());
  builder.add_rbracket_loc(ast->rbracketLoc.index());
  builder.add_constraint_literal_loc(ast->constraintLiteralLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  if (ast->symbolicName) {
    builder.add_symbolic_name(symbolicName);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmOperand;
}

void ASTEncoder::visit(AsmQualifierAST* ast) {
  io::AsmQualifier::Builder builder{fbb_};
  builder.add_qualifier_loc(ast->qualifierLoc.index());
  builder.add_qualifier(static_cast<std::uint32_t>(ast->qualifier));

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmQualifier;
}

void ASTEncoder::visit(AsmClobberAST* ast) {
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
  builder.add_literal_loc(ast->literalLoc.index());
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmClobber;
}

void ASTEncoder::visit(AsmGotoLabelAST* ast) {
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
  builder.add_identifier_loc(ast->identifierLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmGotoLabel;
}

void ASTEncoder::visit(LabeledStatementAST* ast) {
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
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_colon_loc(ast->colonLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_LabeledStatement;
}

void ASTEncoder::visit(CaseStatementAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::CaseStatement::Builder builder{fbb_};
  builder.add_case_loc(ast->caseLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_colon_loc(ast->colonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Statement_CaseStatement;
}

void ASTEncoder::visit(DefaultStatementAST* ast) {
  io::DefaultStatement::Builder builder{fbb_};
  builder.add_default_loc(ast->defaultLoc.index());
  builder.add_colon_loc(ast->colonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Statement_DefaultStatement;
}

void ASTEncoder::visit(ExpressionStatementAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::ExpressionStatement::Builder builder{fbb_};
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ExpressionStatement;
}

void ASTEncoder::visit(CompoundStatementAST* ast) {
  std::vector<flatbuffers::Offset<>> statementListOffsets;
  std::vector<std::underlying_type_t<io::Statement>> statementListTypes;

  for (auto node : ListView{ast->statementList}) {
    if (!node) continue;
    const auto [offset, type] = acceptStatement(node);
    statementListOffsets.push_back(offset);
    statementListTypes.push_back(type);
  }

  auto statementListOffsetsVector = fbb_.CreateVector(statementListOffsets);
  auto statementListTypesVector = fbb_.CreateVector(statementListTypes);

  io::CompoundStatement::Builder builder{fbb_};
  builder.add_lbrace_loc(ast->lbraceLoc.index());
  builder.add_statement_list(statementListOffsetsVector);
  builder.add_statement_list_type(statementListTypesVector);
  builder.add_rbrace_loc(ast->rbraceLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Statement_CompoundStatement;
}

void ASTEncoder::visit(IfStatementAST* ast) {
  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  const auto [elseStatement, elseStatementType] =
      acceptStatement(ast->elseStatement);

  io::IfStatement::Builder builder{fbb_};
  builder.add_if_loc(ast->ifLoc.index());
  builder.add_constexpr_loc(ast->constexprLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));
  builder.add_else_loc(ast->elseLoc.index());
  builder.add_else_statement(elseStatement);
  builder.add_else_statement_type(
      static_cast<io::Statement>(elseStatementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_IfStatement;
}

void ASTEncoder::visit(ConstevalIfStatementAST* ast) {
  const auto [statement, statementType] = acceptStatement(ast->statement);

  const auto [elseStatement, elseStatementType] =
      acceptStatement(ast->elseStatement);

  io::ConstevalIfStatement::Builder builder{fbb_};
  builder.add_if_loc(ast->ifLoc.index());
  builder.add_exclaim_loc(ast->exclaimLoc.index());
  builder.add_constval_loc(ast->constvalLoc.index());
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));
  builder.add_else_loc(ast->elseLoc.index());
  builder.add_else_statement(elseStatement);
  builder.add_else_statement_type(
      static_cast<io::Statement>(elseStatementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ConstevalIfStatement;
}

void ASTEncoder::visit(SwitchStatementAST* ast) {
  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  io::SwitchStatement::Builder builder{fbb_};
  builder.add_switch_loc(ast->switchLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_SwitchStatement;
}

void ASTEncoder::visit(WhileStatementAST* ast) {
  const auto [condition, conditionType] = acceptExpression(ast->condition);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  io::WhileStatement::Builder builder{fbb_};
  builder.add_while_loc(ast->whileLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_WhileStatement;
}

void ASTEncoder::visit(DoStatementAST* ast) {
  const auto [statement, statementType] = acceptStatement(ast->statement);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::DoStatement::Builder builder{fbb_};
  builder.add_do_loc(ast->doLoc.index());
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));
  builder.add_while_loc(ast->whileLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Statement_DoStatement;
}

void ASTEncoder::visit(ForRangeStatementAST* ast) {
  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  const auto [rangeDeclaration, rangeDeclarationType] =
      acceptDeclaration(ast->rangeDeclaration);

  const auto [rangeInitializer, rangeInitializerType] =
      acceptExpression(ast->rangeInitializer);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  io::ForRangeStatement::Builder builder{fbb_};
  builder.add_for_loc(ast->forLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  builder.add_range_declaration(rangeDeclaration);
  builder.add_range_declaration_type(
      static_cast<io::Declaration>(rangeDeclarationType));
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_range_initializer(rangeInitializer);
  builder.add_range_initializer_type(
      static_cast<io::Expression>(rangeInitializerType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ForRangeStatement;
}

void ASTEncoder::visit(ForStatementAST* ast) {
  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  const auto [statement, statementType] = acceptStatement(ast->statement);

  io::ForStatement::Builder builder{fbb_};
  builder.add_for_loc(ast->forLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_semicolon_loc(ast->semicolonLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_statement(statement);
  builder.add_statement_type(static_cast<io::Statement>(statementType));

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ForStatement;
}

void ASTEncoder::visit(BreakStatementAST* ast) {
  io::BreakStatement::Builder builder{fbb_};
  builder.add_break_loc(ast->breakLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Statement_BreakStatement;
}

void ASTEncoder::visit(ContinueStatementAST* ast) {
  io::ContinueStatement::Builder builder{fbb_};
  builder.add_continue_loc(ast->continueLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ContinueStatement;
}

void ASTEncoder::visit(ReturnStatementAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::ReturnStatement::Builder builder{fbb_};
  builder.add_return_loc(ast->returnLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ReturnStatement;
}

void ASTEncoder::visit(CoroutineReturnStatementAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::CoroutineReturnStatement::Builder builder{fbb_};
  builder.add_coreturn_loc(ast->coreturnLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Statement_CoroutineReturnStatement;
}

void ASTEncoder::visit(GotoStatementAST* ast) {
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
  builder.add_goto_loc(ast->gotoLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());
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
  const auto statement = accept(ast->statement);

  std::vector<flatbuffers::Offset<io::Handler>> handlerListOffsets;
  for (auto node : ListView{ast->handlerList}) {
    if (!node) continue;
    handlerListOffsets.emplace_back(accept(node).o);
  }

  auto handlerListOffsetsVector = fbb_.CreateVector(handlerListOffsets);

  io::TryBlockStatement::Builder builder{fbb_};
  builder.add_try_loc(ast->tryLoc.index());
  builder.add_statement(statement.o);
  builder.add_handler_list(handlerListOffsetsVector);

  offset_ = builder.Finish().Union();
  type_ = io::Statement_TryBlockStatement;
}

void ASTEncoder::visit(GeneratedLiteralExpressionAST* ast) {
  io::GeneratedLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(ast->literalLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_GeneratedLiteralExpression;
}

void ASTEncoder::visit(CharLiteralExpressionAST* ast) {
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
  builder.add_literal_loc(ast->literalLoc.index());
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CharLiteralExpression;
}

void ASTEncoder::visit(BoolLiteralExpressionAST* ast) {
  io::BoolLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(ast->literalLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BoolLiteralExpression;
}

void ASTEncoder::visit(IntLiteralExpressionAST* ast) {
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
  builder.add_literal_loc(ast->literalLoc.index());
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_IntLiteralExpression;
}

void ASTEncoder::visit(FloatLiteralExpressionAST* ast) {
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
  builder.add_literal_loc(ast->literalLoc.index());
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_FloatLiteralExpression;
}

void ASTEncoder::visit(NullptrLiteralExpressionAST* ast) {
  io::NullptrLiteralExpression::Builder builder{fbb_};
  builder.add_literal_loc(ast->literalLoc.index());
  builder.add_literal(static_cast<std::uint32_t>(ast->literal));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NullptrLiteralExpression;
}

void ASTEncoder::visit(StringLiteralExpressionAST* ast) {
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
  builder.add_literal_loc(ast->literalLoc.index());
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_StringLiteralExpression;
}

void ASTEncoder::visit(UserDefinedStringLiteralExpressionAST* ast) {
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
  builder.add_literal_loc(ast->literalLoc.index());
  if (ast->literal) {
    builder.add_literal(literal);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_UserDefinedStringLiteralExpression;
}

void ASTEncoder::visit(ThisExpressionAST* ast) {
  io::ThisExpression::Builder builder{fbb_};
  builder.add_this_loc(ast->thisLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ThisExpression;
}

void ASTEncoder::visit(NestedStatementExpressionAST* ast) {
  const auto statement = accept(ast->statement);

  io::NestedStatementExpression::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_statement(statement.o);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NestedStatementExpression;
}

void ASTEncoder::visit(NestedExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::NestedExpression::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NestedExpression;
}

void ASTEncoder::visit(IdExpressionAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::IdExpression::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_IdExpression;
}

void ASTEncoder::visit(LambdaExpressionAST* ast) {
  std::vector<flatbuffers::Offset<>> captureListOffsets;
  std::vector<std::underlying_type_t<io::LambdaCapture>> captureListTypes;

  for (auto node : ListView{ast->captureList}) {
    if (!node) continue;
    const auto [offset, type] = acceptLambdaCapture(node);
    captureListOffsets.push_back(offset);
    captureListTypes.push_back(type);
  }

  auto captureListOffsetsVector = fbb_.CreateVector(captureListOffsets);
  auto captureListTypesVector = fbb_.CreateVector(captureListTypes);

  std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
  std::vector<std::underlying_type_t<io::TemplateParameter>>
      templateParameterListTypes;

  for (auto node : ListView{ast->templateParameterList}) {
    if (!node) continue;
    const auto [offset, type] = acceptTemplateParameter(node);
    templateParameterListOffsets.push_back(offset);
    templateParameterListTypes.push_back(type);
  }

  auto templateParameterListOffsetsVector =
      fbb_.CreateVector(templateParameterListOffsets);
  auto templateParameterListTypesVector =
      fbb_.CreateVector(templateParameterListTypes);

  const auto templateRequiresClause = accept(ast->templateRequiresClause);

  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  std::vector<flatbuffers::Offset<>> gnuAtributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      gnuAtributeListTypes;

  for (auto node : ListView{ast->gnuAtributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    gnuAtributeListOffsets.push_back(offset);
    gnuAtributeListTypes.push_back(type);
  }

  auto gnuAtributeListOffsetsVector = fbb_.CreateVector(gnuAtributeListOffsets);
  auto gnuAtributeListTypesVector = fbb_.CreateVector(gnuAtributeListTypes);

  std::vector<flatbuffers::Offset<io::LambdaSpecifier>>
      lambdaSpecifierListOffsets;
  for (auto node : ListView{ast->lambdaSpecifierList}) {
    if (!node) continue;
    lambdaSpecifierListOffsets.emplace_back(accept(node).o);
  }

  auto lambdaSpecifierListOffsetsVector =
      fbb_.CreateVector(lambdaSpecifierListOffsets);

  const auto [exceptionSpecifier, exceptionSpecifierType] =
      acceptExceptionSpecifier(ast->exceptionSpecifier);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto trailingReturnType = accept(ast->trailingReturnType);

  const auto requiresClause = accept(ast->requiresClause);

  const auto statement = accept(ast->statement);

  io::LambdaExpression::Builder builder{fbb_};
  builder.add_lbracket_loc(ast->lbracketLoc.index());
  builder.add_capture_default_loc(ast->captureDefaultLoc.index());
  builder.add_capture_list(captureListOffsetsVector);
  builder.add_capture_list_type(captureListTypesVector);
  builder.add_rbracket_loc(ast->rbracketLoc.index());
  builder.add_less_loc(ast->lessLoc.index());
  builder.add_template_parameter_list(templateParameterListOffsetsVector);
  builder.add_template_parameter_list_type(templateParameterListTypesVector);
  builder.add_greater_loc(ast->greaterLoc.index());
  builder.add_template_requires_clause(templateRequiresClause.o);
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_gnu_atribute_list(gnuAtributeListOffsetsVector);
  builder.add_gnu_atribute_list_type(gnuAtributeListTypesVector);
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
  const auto [leftExpression, leftExpressionType] =
      acceptExpression(ast->leftExpression);

  const auto [rightExpression, rightExpressionType] =
      acceptExpression(ast->rightExpression);

  io::FoldExpression::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_left_expression(leftExpression);
  builder.add_left_expression_type(
      static_cast<io::Expression>(leftExpressionType));
  builder.add_op_loc(ast->opLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_fold_op_loc(ast->foldOpLoc.index());
  builder.add_right_expression(rightExpression);
  builder.add_right_expression_type(
      static_cast<io::Expression>(rightExpressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_op(static_cast<std::uint32_t>(ast->op));
  builder.add_fold_op(static_cast<std::uint32_t>(ast->foldOp));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_FoldExpression;
}

void ASTEncoder::visit(RightFoldExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::RightFoldExpression::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_op_loc(ast->opLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_RightFoldExpression;
}

void ASTEncoder::visit(LeftFoldExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::LeftFoldExpression::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_op_loc(ast->opLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_LeftFoldExpression;
}

void ASTEncoder::visit(RequiresExpressionAST* ast) {
  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  std::vector<flatbuffers::Offset<>> requirementListOffsets;
  std::vector<std::underlying_type_t<io::Requirement>> requirementListTypes;

  for (auto node : ListView{ast->requirementList}) {
    if (!node) continue;
    const auto [offset, type] = acceptRequirement(node);
    requirementListOffsets.push_back(offset);
    requirementListTypes.push_back(type);
  }

  auto requirementListOffsetsVector = fbb_.CreateVector(requirementListOffsets);
  auto requirementListTypesVector = fbb_.CreateVector(requirementListTypes);

  io::RequiresExpression::Builder builder{fbb_};
  builder.add_requires_loc(ast->requiresLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_lbrace_loc(ast->lbraceLoc.index());
  builder.add_requirement_list(requirementListOffsetsVector);
  builder.add_requirement_list_type(requirementListTypesVector);
  builder.add_rbrace_loc(ast->rbraceLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_RequiresExpression;
}

void ASTEncoder::visit(VaArgExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  const auto typeId = accept(ast->typeId);

  io::VaArgExpression::Builder builder{fbb_};
  builder.add_va_arg_loc(ast->vaArgLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_comma_loc(ast->commaLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_VaArgExpression;
}

void ASTEncoder::visit(SubscriptExpressionAST* ast) {
  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  const auto [indexExpression, indexExpressionType] =
      acceptExpression(ast->indexExpression);

  io::SubscriptExpression::Builder builder{fbb_};
  builder.add_base_expression(baseExpression);
  builder.add_base_expression_type(
      static_cast<io::Expression>(baseExpressionType));
  builder.add_lbracket_loc(ast->lbracketLoc.index());
  builder.add_index_expression(indexExpression);
  builder.add_index_expression_type(
      static_cast<io::Expression>(indexExpressionType));
  builder.add_rbracket_loc(ast->rbracketLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SubscriptExpression;
}

void ASTEncoder::visit(CallExpressionAST* ast) {
  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto node : ListView{ast->expressionList}) {
    if (!node) continue;
    const auto [offset, type] = acceptExpression(node);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  io::CallExpression::Builder builder{fbb_};
  builder.add_base_expression(baseExpression);
  builder.add_base_expression_type(
      static_cast<io::Expression>(baseExpressionType));
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CallExpression;
}

void ASTEncoder::visit(TypeConstructionAST* ast) {
  const auto [typeSpecifier, typeSpecifierType] =
      acceptSpecifier(ast->typeSpecifier);

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto node : ListView{ast->expressionList}) {
    if (!node) continue;
    const auto [offset, type] = acceptExpression(node);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  io::TypeConstruction::Builder builder{fbb_};
  builder.add_type_specifier(typeSpecifier);
  builder.add_type_specifier_type(
      static_cast<io::Specifier>(typeSpecifierType));
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(ast->rparenLoc.index());

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

void ASTEncoder::visit(SpliceMemberExpressionAST* ast) {
  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  const auto splicer = accept(ast->splicer);

  io::SpliceMemberExpression::Builder builder{fbb_};
  builder.add_base_expression(baseExpression);
  builder.add_base_expression_type(
      static_cast<io::Expression>(baseExpressionType));
  builder.add_access_loc(ast->accessLoc.index());
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_splicer(splicer.o);
  builder.add_access_op(static_cast<std::uint32_t>(ast->accessOp));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SpliceMemberExpression;
}

void ASTEncoder::visit(MemberExpressionAST* ast) {
  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::MemberExpression::Builder builder{fbb_};
  builder.add_base_expression(baseExpression);
  builder.add_base_expression_type(
      static_cast<io::Expression>(baseExpressionType));
  builder.add_access_loc(ast->accessLoc.index());
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_access_op(static_cast<std::uint32_t>(ast->accessOp));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_MemberExpression;
}

void ASTEncoder::visit(PostIncrExpressionAST* ast) {
  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  io::PostIncrExpression::Builder builder{fbb_};
  builder.add_base_expression(baseExpression);
  builder.add_base_expression_type(
      static_cast<io::Expression>(baseExpressionType));
  builder.add_op_loc(ast->opLoc.index());
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_PostIncrExpression;
}

void ASTEncoder::visit(CppCastExpressionAST* ast) {
  const auto typeId = accept(ast->typeId);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::CppCastExpression::Builder builder{fbb_};
  builder.add_cast_loc(ast->castLoc.index());
  builder.add_less_loc(ast->lessLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_greater_loc(ast->greaterLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CppCastExpression;
}

void ASTEncoder::visit(BuiltinBitCastExpressionAST* ast) {
  const auto typeId = accept(ast->typeId);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::BuiltinBitCastExpression::Builder builder{fbb_};
  builder.add_cast_loc(ast->castLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_comma_loc(ast->commaLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BuiltinBitCastExpression;
}

void ASTEncoder::visit(BuiltinOffsetofExpressionAST* ast) {
  const auto typeId = accept(ast->typeId);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::BuiltinOffsetofExpression::Builder builder{fbb_};
  builder.add_offsetof_loc(ast->offsetofLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_comma_loc(ast->commaLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BuiltinOffsetofExpression;
}

void ASTEncoder::visit(TypeidExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::TypeidExpression::Builder builder{fbb_};
  builder.add_typeid_loc(ast->typeidLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeidExpression;
}

void ASTEncoder::visit(TypeidOfTypeExpressionAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::TypeidOfTypeExpression::Builder builder{fbb_};
  builder.add_typeid_loc(ast->typeidLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeidOfTypeExpression;
}

void ASTEncoder::visit(SpliceExpressionAST* ast) {
  const auto splicer = accept(ast->splicer);

  io::SpliceExpression::Builder builder{fbb_};
  builder.add_splicer(splicer.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SpliceExpression;
}

void ASTEncoder::visit(GlobalScopeReflectExpressionAST* ast) {
  io::GlobalScopeReflectExpression::Builder builder{fbb_};
  builder.add_caret_loc(ast->caretLoc.index());
  builder.add_scope_loc(ast->scopeLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_GlobalScopeReflectExpression;
}

void ASTEncoder::visit(NamespaceReflectExpressionAST* ast) {
  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::NamespaceReflectExpression::Builder builder{fbb_};
  builder.add_caret_loc(ast->caretLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NamespaceReflectExpression;
}

void ASTEncoder::visit(TypeIdReflectExpressionAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::TypeIdReflectExpression::Builder builder{fbb_};
  builder.add_caret_loc(ast->caretLoc.index());
  builder.add_type_id(typeId.o);

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeIdReflectExpression;
}

void ASTEncoder::visit(ReflectExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::ReflectExpression::Builder builder{fbb_};
  builder.add_caret_loc(ast->caretLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ReflectExpression;
}

void ASTEncoder::visit(UnaryExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::UnaryExpression::Builder builder{fbb_};
  builder.add_op_loc(ast->opLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_UnaryExpression;
}

void ASTEncoder::visit(AwaitExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::AwaitExpression::Builder builder{fbb_};
  builder.add_await_loc(ast->awaitLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AwaitExpression;
}

void ASTEncoder::visit(SizeofExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::SizeofExpression::Builder builder{fbb_};
  builder.add_sizeof_loc(ast->sizeofLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SizeofExpression;
}

void ASTEncoder::visit(SizeofTypeExpressionAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::SizeofTypeExpression::Builder builder{fbb_};
  builder.add_sizeof_loc(ast->sizeofLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SizeofTypeExpression;
}

void ASTEncoder::visit(SizeofPackExpressionAST* ast) {
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
  builder.add_sizeof_loc(ast->sizeofLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SizeofPackExpression;
}

void ASTEncoder::visit(AlignofTypeExpressionAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::AlignofTypeExpression::Builder builder{fbb_};
  builder.add_alignof_loc(ast->alignofLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AlignofTypeExpression;
}

void ASTEncoder::visit(AlignofExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::AlignofExpression::Builder builder{fbb_};
  builder.add_alignof_loc(ast->alignofLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AlignofExpression;
}

void ASTEncoder::visit(NoexceptExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::NoexceptExpression::Builder builder{fbb_};
  builder.add_noexcept_loc(ast->noexceptLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NoexceptExpression;
}

void ASTEncoder::visit(NewExpressionAST* ast) {
  const auto newPlacement = accept(ast->newPlacement);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto node : ListView{ast->typeSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  const auto declarator = accept(ast->declarator);

  const auto [newInitalizer, newInitalizerType] =
      acceptNewInitializer(ast->newInitalizer);

  io::NewExpression::Builder builder{fbb_};
  builder.add_scope_loc(ast->scopeLoc.index());
  builder.add_new_loc(ast->newLoc.index());
  builder.add_new_placement(newPlacement.o);
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_declarator(declarator.o);
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_new_initalizer(newInitalizer);
  builder.add_new_initalizer_type(
      static_cast<io::NewInitializer>(newInitalizerType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NewExpression;
}

void ASTEncoder::visit(DeleteExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::DeleteExpression::Builder builder{fbb_};
  builder.add_scope_loc(ast->scopeLoc.index());
  builder.add_delete_loc(ast->deleteLoc.index());
  builder.add_lbracket_loc(ast->lbracketLoc.index());
  builder.add_rbracket_loc(ast->rbracketLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_DeleteExpression;
}

void ASTEncoder::visit(CastExpressionAST* ast) {
  const auto typeId = accept(ast->typeId);

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::CastExpression::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(ast->rparenLoc.index());
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

  const auto [rightExpression, rightExpressionType] =
      acceptExpression(ast->rightExpression);

  io::BinaryExpression::Builder builder{fbb_};
  builder.add_left_expression(leftExpression);
  builder.add_left_expression_type(
      static_cast<io::Expression>(leftExpressionType));
  builder.add_op_loc(ast->opLoc.index());
  builder.add_right_expression(rightExpression);
  builder.add_right_expression_type(
      static_cast<io::Expression>(rightExpressionType));
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BinaryExpression;
}

void ASTEncoder::visit(ConditionalExpressionAST* ast) {
  const auto [condition, conditionType] = acceptExpression(ast->condition);

  const auto [iftrueExpression, iftrueExpressionType] =
      acceptExpression(ast->iftrueExpression);

  const auto [iffalseExpression, iffalseExpressionType] =
      acceptExpression(ast->iffalseExpression);

  io::ConditionalExpression::Builder builder{fbb_};
  builder.add_condition(condition);
  builder.add_condition_type(static_cast<io::Expression>(conditionType));
  builder.add_question_loc(ast->questionLoc.index());
  builder.add_iftrue_expression(iftrueExpression);
  builder.add_iftrue_expression_type(
      static_cast<io::Expression>(iftrueExpressionType));
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_iffalse_expression(iffalseExpression);
  builder.add_iffalse_expression_type(
      static_cast<io::Expression>(iffalseExpressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ConditionalExpression;
}

void ASTEncoder::visit(YieldExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::YieldExpression::Builder builder{fbb_};
  builder.add_yield_loc(ast->yieldLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_YieldExpression;
}

void ASTEncoder::visit(ThrowExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::ThrowExpression::Builder builder{fbb_};
  builder.add_throw_loc(ast->throwLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ThrowExpression;
}

void ASTEncoder::visit(AssignmentExpressionAST* ast) {
  const auto [leftExpression, leftExpressionType] =
      acceptExpression(ast->leftExpression);

  const auto [rightExpression, rightExpressionType] =
      acceptExpression(ast->rightExpression);

  io::AssignmentExpression::Builder builder{fbb_};
  builder.add_left_expression(leftExpression);
  builder.add_left_expression_type(
      static_cast<io::Expression>(leftExpressionType));
  builder.add_op_loc(ast->opLoc.index());
  builder.add_right_expression(rightExpression);
  builder.add_right_expression_type(
      static_cast<io::Expression>(rightExpressionType));
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AssignmentExpression;
}

void ASTEncoder::visit(PackExpansionExpressionAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::PackExpansionExpression::Builder builder{fbb_};
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_PackExpansionExpression;
}

void ASTEncoder::visit(DesignatedInitializerClauseAST* ast) {
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
  builder.add_dot_loc(ast->dotLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_DesignatedInitializerClause;
}

void ASTEncoder::visit(TypeTraitExpressionAST* ast) {
  std::vector<flatbuffers::Offset<io::TypeId>> typeIdListOffsets;
  for (auto node : ListView{ast->typeIdList}) {
    if (!node) continue;
    typeIdListOffsets.emplace_back(accept(node).o);
  }

  auto typeIdListOffsetsVector = fbb_.CreateVector(typeIdListOffsets);

  io::TypeTraitExpression::Builder builder{fbb_};
  builder.add_type_trait_loc(ast->typeTraitLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id_list(typeIdListOffsetsVector);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeTraitExpression;
}

void ASTEncoder::visit(ConditionExpressionAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;

  for (auto node : ListView{ast->declSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
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
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::EqualInitializer::Builder builder{fbb_};
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
  type_ = io::Expression_EqualInitializer;
}

void ASTEncoder::visit(BracedInitListAST* ast) {
  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto node : ListView{ast->expressionList}) {
    if (!node) continue;
    const auto [offset, type] = acceptExpression(node);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  io::BracedInitList::Builder builder{fbb_};
  builder.add_lbrace_loc(ast->lbraceLoc.index());
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_comma_loc(ast->commaLoc.index());
  builder.add_rbrace_loc(ast->rbraceLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BracedInitList;
}

void ASTEncoder::visit(ParenInitializerAST* ast) {
  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto node : ListView{ast->expressionList}) {
    if (!node) continue;
    const auto [offset, type] = acceptExpression(node);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  io::ParenInitializer::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ParenInitializer;
}

void ASTEncoder::visit(SplicerAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::Splicer::Builder builder{fbb_};
  builder.add_lbracket_loc(ast->lbracketLoc.index());
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_second_colon_loc(ast->secondColonLoc.index());
  builder.add_rbracket_loc(ast->rbracketLoc.index());

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(GlobalModuleFragmentAST* ast) {
  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto node : ListView{ast->declarationList}) {
    if (!node) continue;
    const auto [offset, type] = acceptDeclaration(node);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  io::GlobalModuleFragment::Builder builder{fbb_};
  builder.add_module_loc(ast->moduleLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(PrivateModuleFragmentAST* ast) {
  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto node : ListView{ast->declarationList}) {
    if (!node) continue;
    const auto [offset, type] = acceptDeclaration(node);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  io::PrivateModuleFragment::Builder builder{fbb_};
  builder.add_module_loc(ast->moduleLoc.index());
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_private_loc(ast->privateLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModuleDeclarationAST* ast) {
  const auto moduleName = accept(ast->moduleName);

  const auto modulePartition = accept(ast->modulePartition);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  io::ModuleDeclaration::Builder builder{fbb_};
  builder.add_export_loc(ast->exportLoc.index());
  builder.add_module_loc(ast->moduleLoc.index());
  builder.add_module_name(moduleName.o);
  builder.add_module_partition(modulePartition.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModuleNameAST* ast) {
  const auto moduleQualifier = accept(ast->moduleQualifier);

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
  builder.add_identifier_loc(ast->identifierLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModuleQualifierAST* ast) {
  const auto moduleQualifier = accept(ast->moduleQualifier);

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
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_dot_loc(ast->dotLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModulePartitionAST* ast) {
  const auto moduleName = accept(ast->moduleName);

  io::ModulePartition::Builder builder{fbb_};
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_module_name(moduleName.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ImportNameAST* ast) {
  const auto modulePartition = accept(ast->modulePartition);

  const auto moduleName = accept(ast->moduleName);

  io::ImportName::Builder builder{fbb_};
  builder.add_header_loc(ast->headerLoc.index());
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

  for (auto node : ListView{ast->ptrOpList}) {
    if (!node) continue;
    const auto [offset, type] = acceptPtrOperator(node);
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

  for (auto node : ListView{ast->declaratorChunkList}) {
    if (!node) continue;
    const auto [offset, type] = acceptDeclaratorChunk(node);
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
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::UsingDeclarator::Builder builder{fbb_};
  builder.add_typename_loc(ast->typenameLoc.index());
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(EnumeratorAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

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
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_equal_loc(ast->equalLoc.index());
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

  for (auto node : ListView{ast->typeSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
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
  const auto [exceptionDeclaration, exceptionDeclarationType] =
      acceptExceptionDeclaration(ast->exceptionDeclaration);

  const auto statement = accept(ast->statement);

  io::Handler::Builder builder{fbb_};
  builder.add_catch_loc(ast->catchLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_exception_declaration(exceptionDeclaration);
  builder.add_exception_declaration_type(
      static_cast<io::ExceptionDeclaration>(exceptionDeclarationType));
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_statement(statement.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(BaseSpecifierAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::BaseSpecifier::Builder builder{fbb_};
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_virtual_or_access_loc(ast->virtualOrAccessLoc.index());
  builder.add_other_virtual_or_access_loc(ast->otherVirtualOrAccessLoc.index());
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_access_specifier(
      static_cast<std::uint32_t>(ast->accessSpecifier));

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(RequiresClauseAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::RequiresClause::Builder builder{fbb_};
  builder.add_requires_loc(ast->requiresLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ParameterDeclarationClauseAST* ast) {
  std::vector<flatbuffers::Offset<io::ParameterDeclaration>>
      parameterDeclarationListOffsets;
  for (auto node : ListView{ast->parameterDeclarationList}) {
    if (!node) continue;
    parameterDeclarationListOffsets.emplace_back(accept(node).o);
  }

  auto parameterDeclarationListOffsetsVector =
      fbb_.CreateVector(parameterDeclarationListOffsets);

  io::ParameterDeclarationClause::Builder builder{fbb_};
  builder.add_parameter_declaration_list(parameterDeclarationListOffsetsVector);
  builder.add_comma_loc(ast->commaLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(TrailingReturnTypeAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::TrailingReturnType::Builder builder{fbb_};
  builder.add_minus_greater_loc(ast->minusGreaterLoc.index());
  builder.add_type_id(typeId.o);

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(LambdaSpecifierAST* ast) {
  io::LambdaSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(ast->specifierLoc.index());
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(TypeConstraintAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  std::vector<flatbuffers::Offset<>> templateArgumentListOffsets;
  std::vector<std::underlying_type_t<io::TemplateArgument>>
      templateArgumentListTypes;

  for (auto node : ListView{ast->templateArgumentList}) {
    if (!node) continue;
    const auto [offset, type] = acceptTemplateArgument(node);
    templateArgumentListOffsets.push_back(offset);
    templateArgumentListTypes.push_back(type);
  }

  auto templateArgumentListOffsetsVector =
      fbb_.CreateVector(templateArgumentListOffsets);
  auto templateArgumentListTypesVector =
      fbb_.CreateVector(templateArgumentListTypes);

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
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_less_loc(ast->lessLoc.index());
  builder.add_template_argument_list(templateArgumentListOffsetsVector);
  builder.add_template_argument_list_type(templateArgumentListTypesVector);
  builder.add_greater_loc(ast->greaterLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(AttributeArgumentClauseAST* ast) {
  io::AttributeArgumentClause::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(AttributeAST* ast) {
  const auto [attributeToken, attributeTokenType] =
      acceptAttributeToken(ast->attributeToken);

  const auto attributeArgumentClause = accept(ast->attributeArgumentClause);

  io::Attribute::Builder builder{fbb_};
  builder.add_attribute_token(attributeToken);
  builder.add_attribute_token_type(
      static_cast<io::AttributeToken>(attributeTokenType));
  builder.add_attribute_argument_clause(attributeArgumentClause.o);
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(AttributeUsingPrefixAST* ast) {
  io::AttributeUsingPrefix::Builder builder{fbb_};
  builder.add_using_loc(ast->usingLoc.index());
  builder.add_attribute_namespace_loc(ast->attributeNamespaceLoc.index());
  builder.add_colon_loc(ast->colonLoc.index());

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(NewPlacementAST* ast) {
  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto node : ListView{ast->expressionList}) {
    if (!node) continue;
    const auto [offset, type] = acceptExpression(node);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  io::NewPlacement::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(NestedNamespaceSpecifierAST* ast) {
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
  builder.add_inline_loc(ast->inlineLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_scope_loc(ast->scopeLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(TemplateTypeParameterAST* ast) {
  std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
  std::vector<std::underlying_type_t<io::TemplateParameter>>
      templateParameterListTypes;

  for (auto node : ListView{ast->templateParameterList}) {
    if (!node) continue;
    const auto [offset, type] = acceptTemplateParameter(node);
    templateParameterListOffsets.push_back(offset);
    templateParameterListTypes.push_back(type);
  }

  auto templateParameterListOffsetsVector =
      fbb_.CreateVector(templateParameterListOffsets);
  auto templateParameterListTypesVector =
      fbb_.CreateVector(templateParameterListTypes);

  const auto requiresClause = accept(ast->requiresClause);

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
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_less_loc(ast->lessLoc.index());
  builder.add_template_parameter_list(templateParameterListOffsetsVector);
  builder.add_template_parameter_list_type(templateParameterListTypesVector);
  builder.add_greater_loc(ast->greaterLoc.index());
  builder.add_requires_clause(requiresClause.o);
  builder.add_class_key_loc(ast->classKeyLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_id_expression(idExpression.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::TemplateParameter_TemplateTypeParameter;
}

void ASTEncoder::visit(NonTypeTemplateParameterAST* ast) {
  const auto declaration = accept(ast->declaration);

  io::NonTypeTemplateParameter::Builder builder{fbb_};
  builder.add_declaration(declaration.o);

  offset_ = builder.Finish().Union();
  type_ = io::TemplateParameter_NonTypeTemplateParameter;
}

void ASTEncoder::visit(TypenameTypeParameterAST* ast) {
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
  builder.add_class_key_loc(ast->classKeyLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_type_id(typeId.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::TemplateParameter_TypenameTypeParameter;
}

void ASTEncoder::visit(ConstraintTypeParameterAST* ast) {
  const auto typeConstraint = accept(ast->typeConstraint);

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
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_type_id(typeId.o);
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::TemplateParameter_ConstraintTypeParameter;
}

void ASTEncoder::visit(GeneratedTypeSpecifierAST* ast) {
  io::GeneratedTypeSpecifier::Builder builder{fbb_};
  builder.add_type_loc(ast->typeLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_GeneratedTypeSpecifier;
}

void ASTEncoder::visit(TypedefSpecifierAST* ast) {
  io::TypedefSpecifier::Builder builder{fbb_};
  builder.add_typedef_loc(ast->typedefLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_TypedefSpecifier;
}

void ASTEncoder::visit(FriendSpecifierAST* ast) {
  io::FriendSpecifier::Builder builder{fbb_};
  builder.add_friend_loc(ast->friendLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_FriendSpecifier;
}

void ASTEncoder::visit(ConstevalSpecifierAST* ast) {
  io::ConstevalSpecifier::Builder builder{fbb_};
  builder.add_consteval_loc(ast->constevalLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstevalSpecifier;
}

void ASTEncoder::visit(ConstinitSpecifierAST* ast) {
  io::ConstinitSpecifier::Builder builder{fbb_};
  builder.add_constinit_loc(ast->constinitLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstinitSpecifier;
}

void ASTEncoder::visit(ConstexprSpecifierAST* ast) {
  io::ConstexprSpecifier::Builder builder{fbb_};
  builder.add_constexpr_loc(ast->constexprLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstexprSpecifier;
}

void ASTEncoder::visit(InlineSpecifierAST* ast) {
  io::InlineSpecifier::Builder builder{fbb_};
  builder.add_inline_loc(ast->inlineLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_InlineSpecifier;
}

void ASTEncoder::visit(StaticSpecifierAST* ast) {
  io::StaticSpecifier::Builder builder{fbb_};
  builder.add_static_loc(ast->staticLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_StaticSpecifier;
}

void ASTEncoder::visit(ExternSpecifierAST* ast) {
  io::ExternSpecifier::Builder builder{fbb_};
  builder.add_extern_loc(ast->externLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ExternSpecifier;
}

void ASTEncoder::visit(ThreadLocalSpecifierAST* ast) {
  io::ThreadLocalSpecifier::Builder builder{fbb_};
  builder.add_thread_local_loc(ast->threadLocalLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ThreadLocalSpecifier;
}

void ASTEncoder::visit(ThreadSpecifierAST* ast) {
  io::ThreadSpecifier::Builder builder{fbb_};
  builder.add_thread_loc(ast->threadLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ThreadSpecifier;
}

void ASTEncoder::visit(MutableSpecifierAST* ast) {
  io::MutableSpecifier::Builder builder{fbb_};
  builder.add_mutable_loc(ast->mutableLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_MutableSpecifier;
}

void ASTEncoder::visit(VirtualSpecifierAST* ast) {
  io::VirtualSpecifier::Builder builder{fbb_};
  builder.add_virtual_loc(ast->virtualLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VirtualSpecifier;
}

void ASTEncoder::visit(ExplicitSpecifierAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::ExplicitSpecifier::Builder builder{fbb_};
  builder.add_explicit_loc(ast->explicitLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ExplicitSpecifier;
}

void ASTEncoder::visit(AutoTypeSpecifierAST* ast) {
  io::AutoTypeSpecifier::Builder builder{fbb_};
  builder.add_auto_loc(ast->autoLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_AutoTypeSpecifier;
}

void ASTEncoder::visit(VoidTypeSpecifierAST* ast) {
  io::VoidTypeSpecifier::Builder builder{fbb_};
  builder.add_void_loc(ast->voidLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VoidTypeSpecifier;
}

void ASTEncoder::visit(SizeTypeSpecifierAST* ast) {
  io::SizeTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(ast->specifierLoc.index());
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_SizeTypeSpecifier;
}

void ASTEncoder::visit(SignTypeSpecifierAST* ast) {
  io::SignTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(ast->specifierLoc.index());
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_SignTypeSpecifier;
}

void ASTEncoder::visit(VaListTypeSpecifierAST* ast) {
  io::VaListTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(ast->specifierLoc.index());
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VaListTypeSpecifier;
}

void ASTEncoder::visit(IntegralTypeSpecifierAST* ast) {
  io::IntegralTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(ast->specifierLoc.index());
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_IntegralTypeSpecifier;
}

void ASTEncoder::visit(FloatingPointTypeSpecifierAST* ast) {
  io::FloatingPointTypeSpecifier::Builder builder{fbb_};
  builder.add_specifier_loc(ast->specifierLoc.index());
  builder.add_specifier(static_cast<std::uint32_t>(ast->specifier));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_FloatingPointTypeSpecifier;
}

void ASTEncoder::visit(ComplexTypeSpecifierAST* ast) {
  io::ComplexTypeSpecifier::Builder builder{fbb_};
  builder.add_complex_loc(ast->complexLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ComplexTypeSpecifier;
}

void ASTEncoder::visit(NamedTypeSpecifierAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::NamedTypeSpecifier::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_NamedTypeSpecifier;
}

void ASTEncoder::visit(AtomicTypeSpecifierAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::AtomicTypeSpecifier::Builder builder{fbb_};
  builder.add_atomic_loc(ast->atomicLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_AtomicTypeSpecifier;
}

void ASTEncoder::visit(UnderlyingTypeSpecifierAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::UnderlyingTypeSpecifier::Builder builder{fbb_};
  builder.add_underlying_type_loc(ast->underlyingTypeLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_UnderlyingTypeSpecifier;
}

void ASTEncoder::visit(ElaboratedTypeSpecifierAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::ElaboratedTypeSpecifier::Builder builder{fbb_};
  builder.add_class_loc(ast->classLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_class_key(static_cast<std::uint32_t>(ast->classKey));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ElaboratedTypeSpecifier;
}

void ASTEncoder::visit(DecltypeAutoSpecifierAST* ast) {
  io::DecltypeAutoSpecifier::Builder builder{fbb_};
  builder.add_decltype_loc(ast->decltypeLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_auto_loc(ast->autoLoc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_DecltypeAutoSpecifier;
}

void ASTEncoder::visit(DecltypeSpecifierAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::DecltypeSpecifier::Builder builder{fbb_};
  builder.add_decltype_loc(ast->decltypeLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());

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
  io::ConstQualifier::Builder builder{fbb_};
  builder.add_const_loc(ast->constLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstQualifier;
}

void ASTEncoder::visit(VolatileQualifierAST* ast) {
  io::VolatileQualifier::Builder builder{fbb_};
  builder.add_volatile_loc(ast->volatileLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VolatileQualifier;
}

void ASTEncoder::visit(RestrictQualifierAST* ast) {
  io::RestrictQualifier::Builder builder{fbb_};
  builder.add_restrict_loc(ast->restrictLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_RestrictQualifier;
}

void ASTEncoder::visit(EnumSpecifierAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto unqualifiedId = accept(ast->unqualifiedId);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto node : ListView{ast->typeSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
    typeSpecifierListOffsets.push_back(offset);
    typeSpecifierListTypes.push_back(type);
  }

  auto typeSpecifierListOffsetsVector =
      fbb_.CreateVector(typeSpecifierListOffsets);
  auto typeSpecifierListTypesVector = fbb_.CreateVector(typeSpecifierListTypes);

  std::vector<flatbuffers::Offset<io::Enumerator>> enumeratorListOffsets;
  for (auto node : ListView{ast->enumeratorList}) {
    if (!node) continue;
    enumeratorListOffsets.emplace_back(accept(node).o);
  }

  auto enumeratorListOffsetsVector = fbb_.CreateVector(enumeratorListOffsets);

  io::EnumSpecifier::Builder builder{fbb_};
  builder.add_enum_loc(ast->enumLoc.index());
  builder.add_class_loc(ast->classLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_type_specifier_list(typeSpecifierListOffsetsVector);
  builder.add_type_specifier_list_type(typeSpecifierListTypesVector);
  builder.add_lbrace_loc(ast->lbraceLoc.index());
  builder.add_enumerator_list(enumeratorListOffsetsVector);
  builder.add_comma_loc(ast->commaLoc.index());
  builder.add_rbrace_loc(ast->rbraceLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_EnumSpecifier;
}

void ASTEncoder::visit(ClassSpecifierAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  std::vector<flatbuffers::Offset<io::BaseSpecifier>> baseSpecifierListOffsets;
  for (auto node : ListView{ast->baseSpecifierList}) {
    if (!node) continue;
    baseSpecifierListOffsets.emplace_back(accept(node).o);
  }

  auto baseSpecifierListOffsetsVector =
      fbb_.CreateVector(baseSpecifierListOffsets);

  std::vector<flatbuffers::Offset<>> declarationListOffsets;
  std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;

  for (auto node : ListView{ast->declarationList}) {
    if (!node) continue;
    const auto [offset, type] = acceptDeclaration(node);
    declarationListOffsets.push_back(offset);
    declarationListTypes.push_back(type);
  }

  auto declarationListOffsetsVector = fbb_.CreateVector(declarationListOffsets);
  auto declarationListTypesVector = fbb_.CreateVector(declarationListTypes);

  io::ClassSpecifier::Builder builder{fbb_};
  builder.add_class_loc(ast->classLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_final_loc(ast->finalLoc.index());
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_base_specifier_list(baseSpecifierListOffsetsVector);
  builder.add_lbrace_loc(ast->lbraceLoc.index());
  builder.add_declaration_list(declarationListOffsetsVector);
  builder.add_declaration_list_type(declarationListTypesVector);
  builder.add_rbrace_loc(ast->rbraceLoc.index());
  builder.add_class_key(static_cast<std::uint32_t>(ast->classKey));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ClassSpecifier;
}

void ASTEncoder::visit(TypenameSpecifierAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::TypenameSpecifier::Builder builder{fbb_};
  builder.add_typename_loc(ast->typenameLoc.index());
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_TypenameSpecifier;
}

void ASTEncoder::visit(SplicerTypeSpecifierAST* ast) {
  const auto splicer = accept(ast->splicer);

  io::SplicerTypeSpecifier::Builder builder{fbb_};
  builder.add_typename_loc(ast->typenameLoc.index());
  builder.add_splicer(splicer.o);

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_SplicerTypeSpecifier;
}

void ASTEncoder::visit(PointerOperatorAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> cvQualifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> cvQualifierListTypes;

  for (auto node : ListView{ast->cvQualifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
    cvQualifierListOffsets.push_back(offset);
    cvQualifierListTypes.push_back(type);
  }

  auto cvQualifierListOffsetsVector = fbb_.CreateVector(cvQualifierListOffsets);
  auto cvQualifierListTypesVector = fbb_.CreateVector(cvQualifierListTypes);

  io::PointerOperator::Builder builder{fbb_};
  builder.add_star_loc(ast->starLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_cv_qualifier_list(cvQualifierListOffsetsVector);
  builder.add_cv_qualifier_list_type(cvQualifierListTypesVector);

  offset_ = builder.Finish().Union();
  type_ = io::PtrOperator_PointerOperator;
}

void ASTEncoder::visit(ReferenceOperatorAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  io::ReferenceOperator::Builder builder{fbb_};
  builder.add_ref_loc(ast->refLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_ref_op(static_cast<std::uint32_t>(ast->refOp));

  offset_ = builder.Finish().Union();
  type_ = io::PtrOperator_ReferenceOperator;
}

void ASTEncoder::visit(PtrToMemberOperatorAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> cvQualifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> cvQualifierListTypes;

  for (auto node : ListView{ast->cvQualifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
    cvQualifierListOffsets.push_back(offset);
    cvQualifierListTypes.push_back(type);
  }

  auto cvQualifierListOffsetsVector = fbb_.CreateVector(cvQualifierListOffsets);
  auto cvQualifierListTypesVector = fbb_.CreateVector(cvQualifierListTypes);

  io::PtrToMemberOperator::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_star_loc(ast->starLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);
  builder.add_cv_qualifier_list(cvQualifierListOffsetsVector);
  builder.add_cv_qualifier_list_type(cvQualifierListTypesVector);

  offset_ = builder.Finish().Union();
  type_ = io::PtrOperator_PtrToMemberOperator;
}

void ASTEncoder::visit(BitfieldDeclaratorAST* ast) {
  const auto unqualifiedId = accept(ast->unqualifiedId);

  const auto [sizeExpression, sizeExpressionType] =
      acceptExpression(ast->sizeExpression);

  io::BitfieldDeclarator::Builder builder{fbb_};
  builder.add_unqualified_id(unqualifiedId.o);
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_size_expression(sizeExpression);
  builder.add_size_expression_type(
      static_cast<io::Expression>(sizeExpressionType));

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_BitfieldDeclarator;
}

void ASTEncoder::visit(ParameterPackAST* ast) {
  const auto [coreDeclarator, coreDeclaratorType] =
      acceptCoreDeclarator(ast->coreDeclarator);

  io::ParameterPack::Builder builder{fbb_};
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_core_declarator(coreDeclarator);
  builder.add_core_declarator_type(
      static_cast<io::CoreDeclarator>(coreDeclaratorType));

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_ParameterPack;
}

void ASTEncoder::visit(IdDeclaratorAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  io::IdDeclarator::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_IdDeclarator;
}

void ASTEncoder::visit(NestedDeclaratorAST* ast) {
  const auto declarator = accept(ast->declarator);

  io::NestedDeclarator::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_declarator(declarator.o);
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_NestedDeclarator;
}

void ASTEncoder::visit(FunctionDeclaratorChunkAST* ast) {
  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  std::vector<flatbuffers::Offset<>> cvQualifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> cvQualifierListTypes;

  for (auto node : ListView{ast->cvQualifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
    cvQualifierListOffsets.push_back(offset);
    cvQualifierListTypes.push_back(type);
  }

  auto cvQualifierListOffsetsVector = fbb_.CreateVector(cvQualifierListOffsets);
  auto cvQualifierListTypesVector = fbb_.CreateVector(cvQualifierListTypes);

  const auto [exceptionSpecifier, exceptionSpecifierType] =
      acceptExceptionSpecifier(ast->exceptionSpecifier);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  const auto trailingReturnType = accept(ast->trailingReturnType);

  io::FunctionDeclaratorChunk::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_cv_qualifier_list(cvQualifierListOffsetsVector);
  builder.add_cv_qualifier_list_type(cvQualifierListTypesVector);
  builder.add_ref_loc(ast->refLoc.index());
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
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  io::ArrayDeclaratorChunk::Builder builder{fbb_};
  builder.add_lbracket_loc(ast->lbracketLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rbracket_loc(ast->rbracketLoc.index());
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_attribute_list_type(attributeListTypesVector);

  offset_ = builder.Finish().Union();
  type_ = io::DeclaratorChunk_ArrayDeclaratorChunk;
}

void ASTEncoder::visit(NameIdAST* ast) {
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
  builder.add_identifier_loc(ast->identifierLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_NameId;
}

void ASTEncoder::visit(DestructorIdAST* ast) {
  const auto [id, idType] = acceptUnqualifiedId(ast->id);

  io::DestructorId::Builder builder{fbb_};
  builder.add_tilde_loc(ast->tildeLoc.index());
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
  io::OperatorFunctionId::Builder builder{fbb_};
  builder.add_operator_loc(ast->operatorLoc.index());
  builder.add_op_loc(ast->opLoc.index());
  builder.add_open_loc(ast->openLoc.index());
  builder.add_close_loc(ast->closeLoc.index());
  builder.add_op(static_cast<std::uint32_t>(ast->op));

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_OperatorFunctionId;
}

void ASTEncoder::visit(LiteralOperatorIdAST* ast) {
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
  builder.add_operator_loc(ast->operatorLoc.index());
  builder.add_literal_loc(ast->literalLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_LiteralOperatorId;
}

void ASTEncoder::visit(ConversionFunctionIdAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::ConversionFunctionId::Builder builder{fbb_};
  builder.add_operator_loc(ast->operatorLoc.index());
  builder.add_type_id(typeId.o);

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_ConversionFunctionId;
}

void ASTEncoder::visit(SimpleTemplateIdAST* ast) {
  std::vector<flatbuffers::Offset<>> templateArgumentListOffsets;
  std::vector<std::underlying_type_t<io::TemplateArgument>>
      templateArgumentListTypes;

  for (auto node : ListView{ast->templateArgumentList}) {
    if (!node) continue;
    const auto [offset, type] = acceptTemplateArgument(node);
    templateArgumentListOffsets.push_back(offset);
    templateArgumentListTypes.push_back(type);
  }

  auto templateArgumentListOffsetsVector =
      fbb_.CreateVector(templateArgumentListOffsets);
  auto templateArgumentListTypesVector =
      fbb_.CreateVector(templateArgumentListTypes);

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
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_less_loc(ast->lessLoc.index());
  builder.add_template_argument_list(templateArgumentListOffsetsVector);
  builder.add_template_argument_list_type(templateArgumentListTypesVector);
  builder.add_greater_loc(ast->greaterLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_SimpleTemplateId;
}

void ASTEncoder::visit(LiteralOperatorTemplateIdAST* ast) {
  const auto literalOperatorId = accept(ast->literalOperatorId);

  std::vector<flatbuffers::Offset<>> templateArgumentListOffsets;
  std::vector<std::underlying_type_t<io::TemplateArgument>>
      templateArgumentListTypes;

  for (auto node : ListView{ast->templateArgumentList}) {
    if (!node) continue;
    const auto [offset, type] = acceptTemplateArgument(node);
    templateArgumentListOffsets.push_back(offset);
    templateArgumentListTypes.push_back(type);
  }

  auto templateArgumentListOffsetsVector =
      fbb_.CreateVector(templateArgumentListOffsets);
  auto templateArgumentListTypesVector =
      fbb_.CreateVector(templateArgumentListTypes);

  io::LiteralOperatorTemplateId::Builder builder{fbb_};
  builder.add_literal_operator_id(literalOperatorId.o);
  builder.add_less_loc(ast->lessLoc.index());
  builder.add_template_argument_list(templateArgumentListOffsetsVector);
  builder.add_template_argument_list_type(templateArgumentListTypesVector);
  builder.add_greater_loc(ast->greaterLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_LiteralOperatorTemplateId;
}

void ASTEncoder::visit(OperatorFunctionTemplateIdAST* ast) {
  const auto operatorFunctionId = accept(ast->operatorFunctionId);

  std::vector<flatbuffers::Offset<>> templateArgumentListOffsets;
  std::vector<std::underlying_type_t<io::TemplateArgument>>
      templateArgumentListTypes;

  for (auto node : ListView{ast->templateArgumentList}) {
    if (!node) continue;
    const auto [offset, type] = acceptTemplateArgument(node);
    templateArgumentListOffsets.push_back(offset);
    templateArgumentListTypes.push_back(type);
  }

  auto templateArgumentListOffsetsVector =
      fbb_.CreateVector(templateArgumentListOffsets);
  auto templateArgumentListTypesVector =
      fbb_.CreateVector(templateArgumentListTypes);

  io::OperatorFunctionTemplateId::Builder builder{fbb_};
  builder.add_operator_function_id(operatorFunctionId.o);
  builder.add_less_loc(ast->lessLoc.index());
  builder.add_template_argument_list(templateArgumentListOffsetsVector);
  builder.add_template_argument_list_type(templateArgumentListTypesVector);
  builder.add_greater_loc(ast->greaterLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::UnqualifiedId_OperatorFunctionTemplateId;
}

void ASTEncoder::visit(GlobalNestedNameSpecifierAST* ast) {
  io::GlobalNestedNameSpecifier::Builder builder{fbb_};
  builder.add_scope_loc(ast->scopeLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::NestedNameSpecifier_GlobalNestedNameSpecifier;
}

void ASTEncoder::visit(SimpleNestedNameSpecifierAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  flatbuffers::Offset<flatbuffers::String> identifier;
  if (ast->identifier) {
    if (identifiers_.contains(ast->identifier)) {
      identifier = identifiers_.at(ast->identifier);
    } else {
      identifier = fbb_.CreateString(ast->identifier->value());
      identifiers_.emplace(ast->identifier, identifier);
    }
  }

  io::SimpleNestedNameSpecifier::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_identifier_loc(ast->identifierLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }
  builder.add_scope_loc(ast->scopeLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::NestedNameSpecifier_SimpleNestedNameSpecifier;
}

void ASTEncoder::visit(DecltypeNestedNameSpecifierAST* ast) {
  const auto decltypeSpecifier = accept(ast->decltypeSpecifier);

  io::DecltypeNestedNameSpecifier::Builder builder{fbb_};
  builder.add_decltype_specifier(decltypeSpecifier.o);
  builder.add_scope_loc(ast->scopeLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::NestedNameSpecifier_DecltypeNestedNameSpecifier;
}

void ASTEncoder::visit(TemplateNestedNameSpecifierAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto templateId = accept(ast->templateId);

  io::TemplateNestedNameSpecifier::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_template_id(templateId.o);
  builder.add_scope_loc(ast->scopeLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::NestedNameSpecifier_TemplateNestedNameSpecifier;
}

void ASTEncoder::visit(DefaultFunctionBodyAST* ast) {
  io::DefaultFunctionBody::Builder builder{fbb_};
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_default_loc(ast->defaultLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_DefaultFunctionBody;
}

void ASTEncoder::visit(CompoundStatementFunctionBodyAST* ast) {
  std::vector<flatbuffers::Offset<>> memInitializerListOffsets;
  std::vector<std::underlying_type_t<io::MemInitializer>>
      memInitializerListTypes;

  for (auto node : ListView{ast->memInitializerList}) {
    if (!node) continue;
    const auto [offset, type] = acceptMemInitializer(node);
    memInitializerListOffsets.push_back(offset);
    memInitializerListTypes.push_back(type);
  }

  auto memInitializerListOffsetsVector =
      fbb_.CreateVector(memInitializerListOffsets);
  auto memInitializerListTypesVector =
      fbb_.CreateVector(memInitializerListTypes);

  const auto statement = accept(ast->statement);

  io::CompoundStatementFunctionBody::Builder builder{fbb_};
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_mem_initializer_list(memInitializerListOffsetsVector);
  builder.add_mem_initializer_list_type(memInitializerListTypesVector);
  builder.add_statement(statement.o);

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_CompoundStatementFunctionBody;
}

void ASTEncoder::visit(TryStatementFunctionBodyAST* ast) {
  std::vector<flatbuffers::Offset<>> memInitializerListOffsets;
  std::vector<std::underlying_type_t<io::MemInitializer>>
      memInitializerListTypes;

  for (auto node : ListView{ast->memInitializerList}) {
    if (!node) continue;
    const auto [offset, type] = acceptMemInitializer(node);
    memInitializerListOffsets.push_back(offset);
    memInitializerListTypes.push_back(type);
  }

  auto memInitializerListOffsetsVector =
      fbb_.CreateVector(memInitializerListOffsets);
  auto memInitializerListTypesVector =
      fbb_.CreateVector(memInitializerListTypes);

  const auto statement = accept(ast->statement);

  std::vector<flatbuffers::Offset<io::Handler>> handlerListOffsets;
  for (auto node : ListView{ast->handlerList}) {
    if (!node) continue;
    handlerListOffsets.emplace_back(accept(node).o);
  }

  auto handlerListOffsetsVector = fbb_.CreateVector(handlerListOffsets);

  io::TryStatementFunctionBody::Builder builder{fbb_};
  builder.add_try_loc(ast->tryLoc.index());
  builder.add_colon_loc(ast->colonLoc.index());
  builder.add_mem_initializer_list(memInitializerListOffsetsVector);
  builder.add_mem_initializer_list_type(memInitializerListTypesVector);
  builder.add_statement(statement.o);
  builder.add_handler_list(handlerListOffsetsVector);

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_TryStatementFunctionBody;
}

void ASTEncoder::visit(DeleteFunctionBodyAST* ast) {
  io::DeleteFunctionBody::Builder builder{fbb_};
  builder.add_equal_loc(ast->equalLoc.index());
  builder.add_delete_loc(ast->deleteLoc.index());
  builder.add_semicolon_loc(ast->semicolonLoc.index());

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
  io::ThrowExceptionSpecifier::Builder builder{fbb_};
  builder.add_throw_loc(ast->throwLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::ExceptionSpecifier_ThrowExceptionSpecifier;
}

void ASTEncoder::visit(NoexceptSpecifierAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::NoexceptSpecifier::Builder builder{fbb_};
  builder.add_noexcept_loc(ast->noexceptLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::ExceptionSpecifier_NoexceptSpecifier;
}

void ASTEncoder::visit(SimpleRequirementAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::SimpleRequirement::Builder builder{fbb_};
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_SimpleRequirement;
}

void ASTEncoder::visit(CompoundRequirementAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  const auto typeConstraint = accept(ast->typeConstraint);

  io::CompoundRequirement::Builder builder{fbb_};
  builder.add_lbrace_loc(ast->lbraceLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_rbrace_loc(ast->rbraceLoc.index());
  builder.add_noexcept_loc(ast->noexceptLoc.index());
  builder.add_minus_greater_loc(ast->minusGreaterLoc.index());
  builder.add_type_constraint(typeConstraint.o);
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_CompoundRequirement;
}

void ASTEncoder::visit(TypeRequirementAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  io::TypeRequirement::Builder builder{fbb_};
  builder.add_typename_loc(ast->typenameLoc.index());
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_template_loc(ast->templateLoc.index());
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_TypeRequirement;
}

void ASTEncoder::visit(NestedRequirementAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::NestedRequirement::Builder builder{fbb_};
  builder.add_requires_loc(ast->requiresLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_semicolon_loc(ast->semicolonLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_NestedRequirement;
}

void ASTEncoder::visit(NewParenInitializerAST* ast) {
  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto node : ListView{ast->expressionList}) {
    if (!node) continue;
    const auto [offset, type] = acceptExpression(node);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  io::NewParenInitializer::Builder builder{fbb_};
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(ast->rparenLoc.index());

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

  std::vector<flatbuffers::Offset<>> expressionListOffsets;
  std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;

  for (auto node : ListView{ast->expressionList}) {
    if (!node) continue;
    const auto [offset, type] = acceptExpression(node);
    expressionListOffsets.push_back(offset);
    expressionListTypes.push_back(type);
  }

  auto expressionListOffsetsVector = fbb_.CreateVector(expressionListOffsets);
  auto expressionListTypesVector = fbb_.CreateVector(expressionListTypes);

  io::ParenMemInitializer::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression_list(expressionListOffsetsVector);
  builder.add_expression_list_type(expressionListTypesVector);
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::MemInitializer_ParenMemInitializer;
}

void ASTEncoder::visit(BracedMemInitializerAST* ast) {
  const auto [nestedNameSpecifier, nestedNameSpecifierType] =
      acceptNestedNameSpecifier(ast->nestedNameSpecifier);

  const auto [unqualifiedId, unqualifiedIdType] =
      acceptUnqualifiedId(ast->unqualifiedId);

  const auto bracedInitList = accept(ast->bracedInitList);

  io::BracedMemInitializer::Builder builder{fbb_};
  builder.add_nested_name_specifier(nestedNameSpecifier);
  builder.add_nested_name_specifier_type(
      static_cast<io::NestedNameSpecifier>(nestedNameSpecifierType));
  builder.add_unqualified_id(unqualifiedId);
  builder.add_unqualified_id_type(
      static_cast<io::UnqualifiedId>(unqualifiedIdType));
  builder.add_braced_init_list(bracedInitList.o);
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::MemInitializer_BracedMemInitializer;
}

void ASTEncoder::visit(ThisLambdaCaptureAST* ast) {
  io::ThisLambdaCapture::Builder builder{fbb_};
  builder.add_this_loc(ast->thisLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_ThisLambdaCapture;
}

void ASTEncoder::visit(DerefThisLambdaCaptureAST* ast) {
  io::DerefThisLambdaCapture::Builder builder{fbb_};
  builder.add_star_loc(ast->starLoc.index());
  builder.add_this_loc(ast->thisLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_DerefThisLambdaCapture;
}

void ASTEncoder::visit(SimpleLambdaCaptureAST* ast) {
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
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_SimpleLambdaCapture;
}

void ASTEncoder::visit(RefLambdaCaptureAST* ast) {
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
  builder.add_amp_loc(ast->ampLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_RefLambdaCapture;
}

void ASTEncoder::visit(RefInitLambdaCaptureAST* ast) {
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
  builder.add_amp_loc(ast->ampLoc.index());
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_RefInitLambdaCapture;
}

void ASTEncoder::visit(InitLambdaCaptureAST* ast) {
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
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
  builder.add_initializer(initializer);
  builder.add_initializer_type(static_cast<io::Expression>(initializerType));
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_InitLambdaCapture;
}

void ASTEncoder::visit(EllipsisExceptionDeclarationAST* ast) {
  io::EllipsisExceptionDeclaration::Builder builder{fbb_};
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::ExceptionDeclaration_EllipsisExceptionDeclaration;
}

void ASTEncoder::visit(TypeExceptionDeclarationAST* ast) {
  std::vector<flatbuffers::Offset<>> attributeListOffsets;
  std::vector<std::underlying_type_t<io::AttributeSpecifier>>
      attributeListTypes;

  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    const auto [offset, type] = acceptAttributeSpecifier(node);
    attributeListOffsets.push_back(offset);
    attributeListTypes.push_back(type);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);
  auto attributeListTypesVector = fbb_.CreateVector(attributeListTypes);

  std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
  std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;

  for (auto node : ListView{ast->typeSpecifierList}) {
    if (!node) continue;
    const auto [offset, type] = acceptSpecifier(node);
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
  const auto attributeUsingPrefix = accept(ast->attributeUsingPrefix);

  std::vector<flatbuffers::Offset<io::Attribute>> attributeListOffsets;
  for (auto node : ListView{ast->attributeList}) {
    if (!node) continue;
    attributeListOffsets.emplace_back(accept(node).o);
  }

  auto attributeListOffsetsVector = fbb_.CreateVector(attributeListOffsets);

  io::CxxAttribute::Builder builder{fbb_};
  builder.add_lbracket_loc(ast->lbracketLoc.index());
  builder.add_lbracket2_loc(ast->lbracket2Loc.index());
  builder.add_attribute_using_prefix(attributeUsingPrefix.o);
  builder.add_attribute_list(attributeListOffsetsVector);
  builder.add_rbracket_loc(ast->rbracketLoc.index());
  builder.add_rbracket2_loc(ast->rbracket2Loc.index());

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_CxxAttribute;
}

void ASTEncoder::visit(GccAttributeAST* ast) {
  io::GccAttribute::Builder builder{fbb_};
  builder.add_attribute_loc(ast->attributeLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_lparen2_loc(ast->lparen2Loc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());
  builder.add_rparen2_loc(ast->rparen2Loc.index());

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_GccAttribute;
}

void ASTEncoder::visit(AlignasAttributeAST* ast) {
  const auto [expression, expressionType] = acceptExpression(ast->expression);

  io::AlignasAttribute::Builder builder{fbb_};
  builder.add_alignas_loc(ast->alignasLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_expression(expression);
  builder.add_expression_type(static_cast<io::Expression>(expressionType));
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_AlignasAttribute;
}

void ASTEncoder::visit(AlignasTypeAttributeAST* ast) {
  const auto typeId = accept(ast->typeId);

  io::AlignasTypeAttribute::Builder builder{fbb_};
  builder.add_alignas_loc(ast->alignasLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_type_id(typeId.o);
  builder.add_ellipsis_loc(ast->ellipsisLoc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_AlignasTypeAttribute;
}

void ASTEncoder::visit(AsmAttributeAST* ast) {
  io::AsmAttribute::Builder builder{fbb_};
  builder.add_asm_loc(ast->asmLoc.index());
  builder.add_lparen_loc(ast->lparenLoc.index());
  builder.add_literal_loc(ast->literalLoc.index());
  builder.add_rparen_loc(ast->rparenLoc.index());

  offset_ = builder.Finish().Union();
  type_ = io::AttributeSpecifier_AsmAttribute;
}

void ASTEncoder::visit(ScopedAttributeTokenAST* ast) {
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
  builder.add_attribute_namespace_loc(ast->attributeNamespaceLoc.index());
  builder.add_scope_loc(ast->scopeLoc.index());
  builder.add_identifier_loc(ast->identifierLoc.index());
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
  builder.add_identifier_loc(ast->identifierLoc.index());
  if (ast->identifier) {
    builder.add_identifier(identifier);
  }

  offset_ = builder.Finish().Union();
  type_ = io::AttributeToken_SimpleAttributeToken;
}

}  // namespace cxx
