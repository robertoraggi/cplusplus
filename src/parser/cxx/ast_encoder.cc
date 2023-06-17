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
#include <cxx/translation_unit.h>

#include <algorithm>

namespace cxx {

auto ASTEncoder::operator()(TranslationUnit* unit) -> std::span<std::uint8_t> {
  if (!unit) return {};
  std::swap(unit_, unit);
  auto [unitOffset, unitType] = acceptUnit(unit_->ast());
  io::SerializedUnitBuilder builder{fbb_};
  builder.add_unit(unitOffset);
  builder.add_unit_type(static_cast<io::Unit>(unitType));
  std::swap(unit_, unit);
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

auto ASTEncoder::acceptInitializer(InitializerAST* ast)
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

auto ASTEncoder::acceptName(NameAST* ast)
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

auto ASTEncoder::acceptDeclaratorModifier(DeclaratorModifierAST* ast)
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

auto ASTEncoder::acceptAttribute(AttributeAST* ast)
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

void ASTEncoder::visit(TypeIdAST* ast) {
  io::TypeId::Builder builder{fbb_};

  if (ast->typeSpecifierList) {
    std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      typeSpecifierListOffsets.push_back(offset);
      typeSpecifierListTypes.push_back(type);
    }
    builder.add_type_specifier_list(
        fbb_.CreateVector(typeSpecifierListOffsets));
    builder.add_type_specifier_list_type(
        fbb_.CreateVector(typeSpecifierListTypes));
  }

  const auto declarator = accept(ast->declarator);

  if (!declarator.IsNull()) {
    builder.add_declarator(declarator.o);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(NestedNameSpecifierAST* ast) {
  io::NestedNameSpecifier::Builder builder{fbb_};

  if (ast->nameList) {
    std::vector<flatbuffers::Offset<>> nameListOffsets;
    std::vector<std::underlying_type_t<io::Name>> nameListTypes;
    for (auto it = ast->nameList; it; it = it->next) {
      const auto [offset, type] = acceptName(it->value);
      nameListOffsets.push_back(offset);
      nameListTypes.push_back(type);
    }
    builder.add_name_list(fbb_.CreateVector(nameListOffsets));
    builder.add_name_list_type(fbb_.CreateVector(nameListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(UsingDeclaratorAST* ast) {
  io::UsingDeclarator::Builder builder{fbb_};

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(HandlerAST* ast) {
  io::Handler::Builder builder{fbb_};

  const auto [exceptionDeclaration, exceptionDeclarationType] =
      acceptExceptionDeclaration(ast->exceptionDeclaration);

  if (!exceptionDeclaration.IsNull()) {
    builder.add_exception_declaration(exceptionDeclaration);
    builder.add_exception_declaration_type(
        static_cast<io::ExceptionDeclaration>(exceptionDeclarationType));
  }

  const auto statement = accept(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement.o);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(EnumBaseAST* ast) {
  io::EnumBase::Builder builder{fbb_};

  if (ast->typeSpecifierList) {
    std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      typeSpecifierListOffsets.push_back(offset);
      typeSpecifierListTypes.push_back(type);
    }
    builder.add_type_specifier_list(
        fbb_.CreateVector(typeSpecifierListOffsets));
    builder.add_type_specifier_list_type(
        fbb_.CreateVector(typeSpecifierListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(EnumeratorAST* ast) {
  io::Enumerator::Builder builder{fbb_};

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(DeclaratorAST* ast) {
  io::Declarator::Builder builder{fbb_};

  if (ast->ptrOpList) {
    std::vector<flatbuffers::Offset<>> ptrOpListOffsets;
    std::vector<std::underlying_type_t<io::PtrOperator>> ptrOpListTypes;
    for (auto it = ast->ptrOpList; it; it = it->next) {
      const auto [offset, type] = acceptPtrOperator(it->value);
      ptrOpListOffsets.push_back(offset);
      ptrOpListTypes.push_back(type);
    }
    builder.add_ptr_op_list(fbb_.CreateVector(ptrOpListOffsets));
    builder.add_ptr_op_list_type(fbb_.CreateVector(ptrOpListTypes));
  }

  const auto [coreDeclarator, coreDeclaratorType] =
      acceptCoreDeclarator(ast->coreDeclarator);

  if (!coreDeclarator.IsNull()) {
    builder.add_core_declarator(coreDeclarator);
    builder.add_core_declarator_type(
        static_cast<io::CoreDeclarator>(coreDeclaratorType));
  }

  if (ast->modifiers) {
    std::vector<flatbuffers::Offset<>> modifiersOffsets;
    std::vector<std::underlying_type_t<io::DeclaratorModifier>> modifiersTypes;
    for (auto it = ast->modifiers; it; it = it->next) {
      const auto [offset, type] = acceptDeclaratorModifier(it->value);
      modifiersOffsets.push_back(offset);
      modifiersTypes.push_back(type);
    }
    builder.add_modifiers(fbb_.CreateVector(modifiersOffsets));
    builder.add_modifiers_type(fbb_.CreateVector(modifiersTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(InitDeclaratorAST* ast) {
  io::InitDeclarator::Builder builder{fbb_};

  const auto declarator = accept(ast->declarator);

  if (!declarator.IsNull()) {
    builder.add_declarator(declarator.o);
  }

  const auto requiresClause = accept(ast->requiresClause);

  if (!requiresClause.IsNull()) {
    builder.add_requires_clause(requiresClause.o);
  }

  const auto [initializer, initializerType] =
      acceptInitializer(ast->initializer);

  if (!initializer.IsNull()) {
    builder.add_initializer(initializer);
    builder.add_initializer_type(static_cast<io::Initializer>(initializerType));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(BaseSpecifierAST* ast) {
  io::BaseSpecifier::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(BaseClauseAST* ast) {
  io::BaseClause::Builder builder{fbb_};

  if (ast->baseSpecifierList) {
    std::vector<flatbuffers::Offset<io::BaseSpecifier>>
        baseSpecifierListOffsets;
    for (auto it = ast->baseSpecifierList; it; it = it->next) {
      baseSpecifierListOffsets.emplace_back(accept(it->value).o);
    }
    builder.add_base_specifier_list(
        fbb_.CreateVector(baseSpecifierListOffsets));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(NewTypeIdAST* ast) {
  io::NewTypeId::Builder builder{fbb_};

  if (ast->typeSpecifierList) {
    std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      typeSpecifierListOffsets.push_back(offset);
      typeSpecifierListTypes.push_back(type);
    }
    builder.add_type_specifier_list(
        fbb_.CreateVector(typeSpecifierListOffsets));
    builder.add_type_specifier_list_type(
        fbb_.CreateVector(typeSpecifierListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(RequiresClauseAST* ast) {
  io::RequiresClause::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ParameterDeclarationClauseAST* ast) {
  io::ParameterDeclarationClause::Builder builder{fbb_};

  if (ast->parameterDeclarationList) {
    std::vector<flatbuffers::Offset<io::ParameterDeclaration>>
        parameterDeclarationListOffsets;
    for (auto it = ast->parameterDeclarationList; it; it = it->next) {
      parameterDeclarationListOffsets.emplace_back(accept(it->value).o);
    }
    builder.add_parameter_declaration_list(
        fbb_.CreateVector(parameterDeclarationListOffsets));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ParametersAndQualifiersAST* ast) {
  io::ParametersAndQualifiers::Builder builder{fbb_};

  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  if (!parameterDeclarationClause.IsNull()) {
    builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  }

  if (ast->cvQualifierList) {
    std::vector<flatbuffers::Offset<>> cvQualifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> cvQualifierListTypes;
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      cvQualifierListOffsets.push_back(offset);
      cvQualifierListTypes.push_back(type);
    }
    builder.add_cv_qualifier_list(fbb_.CreateVector(cvQualifierListOffsets));
    builder.add_cv_qualifier_list_type(fbb_.CreateVector(cvQualifierListTypes));
  }

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(LambdaIntroducerAST* ast) {
  io::LambdaIntroducer::Builder builder{fbb_};

  if (ast->captureList) {
    std::vector<flatbuffers::Offset<>> captureListOffsets;
    std::vector<std::underlying_type_t<io::LambdaCapture>> captureListTypes;
    for (auto it = ast->captureList; it; it = it->next) {
      const auto [offset, type] = acceptLambdaCapture(it->value);
      captureListOffsets.push_back(offset);
      captureListTypes.push_back(type);
    }
    builder.add_capture_list(fbb_.CreateVector(captureListOffsets));
    builder.add_capture_list_type(fbb_.CreateVector(captureListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(LambdaDeclaratorAST* ast) {
  io::LambdaDeclarator::Builder builder{fbb_};

  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  if (!parameterDeclarationClause.IsNull()) {
    builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  }

  if (ast->declSpecifierList) {
    std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      declSpecifierListOffsets.push_back(offset);
      declSpecifierListTypes.push_back(type);
    }
    builder.add_decl_specifier_list(
        fbb_.CreateVector(declSpecifierListOffsets));
    builder.add_decl_specifier_list_type(
        fbb_.CreateVector(declSpecifierListTypes));
  }

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto trailingReturnType = accept(ast->trailingReturnType);

  if (!trailingReturnType.IsNull()) {
    builder.add_trailing_return_type(trailingReturnType.o);
  }

  const auto requiresClause = accept(ast->requiresClause);

  if (!requiresClause.IsNull()) {
    builder.add_requires_clause(requiresClause.o);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(TrailingReturnTypeAST* ast) {
  io::TrailingReturnType::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(CtorInitializerAST* ast) {
  io::CtorInitializer::Builder builder{fbb_};

  if (ast->memInitializerList) {
    std::vector<flatbuffers::Offset<>> memInitializerListOffsets;
    std::vector<std::underlying_type_t<io::MemInitializer>>
        memInitializerListTypes;
    for (auto it = ast->memInitializerList; it; it = it->next) {
      const auto [offset, type] = acceptMemInitializer(it->value);
      memInitializerListOffsets.push_back(offset);
      memInitializerListTypes.push_back(type);
    }
    builder.add_mem_initializer_list(
        fbb_.CreateVector(memInitializerListOffsets));
    builder.add_mem_initializer_list_type(
        fbb_.CreateVector(memInitializerListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(RequirementBodyAST* ast) {
  io::RequirementBody::Builder builder{fbb_};

  if (ast->requirementList) {
    std::vector<flatbuffers::Offset<>> requirementListOffsets;
    std::vector<std::underlying_type_t<io::Requirement>> requirementListTypes;
    for (auto it = ast->requirementList; it; it = it->next) {
      const auto [offset, type] = acceptRequirement(it->value);
      requirementListOffsets.push_back(offset);
      requirementListTypes.push_back(type);
    }
    builder.add_requirement_list(fbb_.CreateVector(requirementListOffsets));
    builder.add_requirement_list_type(fbb_.CreateVector(requirementListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(TypeConstraintAST* ast) {
  io::TypeConstraint::Builder builder{fbb_};

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(GlobalModuleFragmentAST* ast) {
  io::GlobalModuleFragment::Builder builder{fbb_};

  if (ast->declarationList) {
    std::vector<flatbuffers::Offset<>> declarationListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;
    for (auto it = ast->declarationList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      declarationListOffsets.push_back(offset);
      declarationListTypes.push_back(type);
    }
    builder.add_declaration_list(fbb_.CreateVector(declarationListOffsets));
    builder.add_declaration_list_type(fbb_.CreateVector(declarationListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(PrivateModuleFragmentAST* ast) {
  io::PrivateModuleFragment::Builder builder{fbb_};

  if (ast->declarationList) {
    std::vector<flatbuffers::Offset<>> declarationListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;
    for (auto it = ast->declarationList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      declarationListOffsets.push_back(offset);
      declarationListTypes.push_back(type);
    }
    builder.add_declaration_list(fbb_.CreateVector(declarationListOffsets));
    builder.add_declaration_list_type(fbb_.CreateVector(declarationListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModuleDeclarationAST* ast) {
  io::ModuleDeclaration::Builder builder{fbb_};

  const auto moduleName = accept(ast->moduleName);

  if (!moduleName.IsNull()) {
    builder.add_module_name(moduleName.o);
  }

  const auto modulePartition = accept(ast->modulePartition);

  if (!modulePartition.IsNull()) {
    builder.add_module_partition(modulePartition.o);
  }

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModuleNameAST* ast) {
  io::ModuleName::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ImportNameAST* ast) {
  io::ImportName::Builder builder{fbb_};

  const auto modulePartition = accept(ast->modulePartition);

  if (!modulePartition.IsNull()) {
    builder.add_module_partition(modulePartition.o);
  }

  const auto moduleName = accept(ast->moduleName);

  if (!moduleName.IsNull()) {
    builder.add_module_name(moduleName.o);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(ModulePartitionAST* ast) {
  io::ModulePartition::Builder builder{fbb_};

  const auto moduleName = accept(ast->moduleName);

  if (!moduleName.IsNull()) {
    builder.add_module_name(moduleName.o);
  }

  offset_ = builder.Finish().Union();
}

void ASTEncoder::visit(SimpleRequirementAST* ast) {
  io::SimpleRequirement::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_SimpleRequirement;
}

void ASTEncoder::visit(CompoundRequirementAST* ast) {
  io::CompoundRequirement::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  const auto typeConstraint = accept(ast->typeConstraint);

  if (!typeConstraint.IsNull()) {
    builder.add_type_constraint(typeConstraint.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_CompoundRequirement;
}

void ASTEncoder::visit(TypeRequirementAST* ast) {
  io::TypeRequirement::Builder builder{fbb_};

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_TypeRequirement;
}

void ASTEncoder::visit(NestedRequirementAST* ast) {
  io::NestedRequirement::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Requirement_NestedRequirement;
}

void ASTEncoder::visit(TypeTemplateArgumentAST* ast) {
  io::TypeTemplateArgument::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::TemplateArgument_TypeTemplateArgument;
}

void ASTEncoder::visit(ExpressionTemplateArgumentAST* ast) {
  io::ExpressionTemplateArgument::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::TemplateArgument_ExpressionTemplateArgument;
}

void ASTEncoder::visit(ParenMemInitializerAST* ast) {
  io::ParenMemInitializer::Builder builder{fbb_};

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  if (ast->expressionList) {
    std::vector<flatbuffers::Offset<>> expressionListOffsets;
    std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;
    for (auto it = ast->expressionList; it; it = it->next) {
      const auto [offset, type] = acceptExpression(it->value);
      expressionListOffsets.push_back(offset);
      expressionListTypes.push_back(type);
    }
    builder.add_expression_list(fbb_.CreateVector(expressionListOffsets));
    builder.add_expression_list_type(fbb_.CreateVector(expressionListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::MemInitializer_ParenMemInitializer;
}

void ASTEncoder::visit(BracedMemInitializerAST* ast) {
  io::BracedMemInitializer::Builder builder{fbb_};

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  const auto bracedInitList = accept(ast->bracedInitList);

  if (!bracedInitList.IsNull()) {
    builder.add_braced_init_list(bracedInitList.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::MemInitializer_BracedMemInitializer;
}

void ASTEncoder::visit(ThisLambdaCaptureAST* ast) {
  io::ThisLambdaCapture::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_ThisLambdaCapture;
}

void ASTEncoder::visit(DerefThisLambdaCaptureAST* ast) {
  io::DerefThisLambdaCapture::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_DerefThisLambdaCapture;
}

void ASTEncoder::visit(SimpleLambdaCaptureAST* ast) {
  io::SimpleLambdaCapture::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_SimpleLambdaCapture;
}

void ASTEncoder::visit(RefLambdaCaptureAST* ast) {
  io::RefLambdaCapture::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_RefLambdaCapture;
}

void ASTEncoder::visit(RefInitLambdaCaptureAST* ast) {
  io::RefInitLambdaCapture::Builder builder{fbb_};

  const auto [initializer, initializerType] =
      acceptInitializer(ast->initializer);

  if (!initializer.IsNull()) {
    builder.add_initializer(initializer);
    builder.add_initializer_type(static_cast<io::Initializer>(initializerType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_RefInitLambdaCapture;
}

void ASTEncoder::visit(InitLambdaCaptureAST* ast) {
  io::InitLambdaCapture::Builder builder{fbb_};

  const auto [initializer, initializerType] =
      acceptInitializer(ast->initializer);

  if (!initializer.IsNull()) {
    builder.add_initializer(initializer);
    builder.add_initializer_type(static_cast<io::Initializer>(initializerType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::LambdaCapture_InitLambdaCapture;
}

void ASTEncoder::visit(EqualInitializerAST* ast) {
  io::EqualInitializer::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Initializer_EqualInitializer;
}

void ASTEncoder::visit(BracedInitListAST* ast) {
  io::BracedInitList::Builder builder{fbb_};

  if (ast->expressionList) {
    std::vector<flatbuffers::Offset<>> expressionListOffsets;
    std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;
    for (auto it = ast->expressionList; it; it = it->next) {
      const auto [offset, type] = acceptExpression(it->value);
      expressionListOffsets.push_back(offset);
      expressionListTypes.push_back(type);
    }
    builder.add_expression_list(fbb_.CreateVector(expressionListOffsets));
    builder.add_expression_list_type(fbb_.CreateVector(expressionListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Initializer_BracedInitList;
}

void ASTEncoder::visit(ParenInitializerAST* ast) {
  io::ParenInitializer::Builder builder{fbb_};

  if (ast->expressionList) {
    std::vector<flatbuffers::Offset<>> expressionListOffsets;
    std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;
    for (auto it = ast->expressionList; it; it = it->next) {
      const auto [offset, type] = acceptExpression(it->value);
      expressionListOffsets.push_back(offset);
      expressionListTypes.push_back(type);
    }
    builder.add_expression_list(fbb_.CreateVector(expressionListOffsets));
    builder.add_expression_list_type(fbb_.CreateVector(expressionListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Initializer_ParenInitializer;
}

void ASTEncoder::visit(NewParenInitializerAST* ast) {
  io::NewParenInitializer::Builder builder{fbb_};

  if (ast->expressionList) {
    std::vector<flatbuffers::Offset<>> expressionListOffsets;
    std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;
    for (auto it = ast->expressionList; it; it = it->next) {
      const auto [offset, type] = acceptExpression(it->value);
      expressionListOffsets.push_back(offset);
      expressionListTypes.push_back(type);
    }
    builder.add_expression_list(fbb_.CreateVector(expressionListOffsets));
    builder.add_expression_list_type(fbb_.CreateVector(expressionListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::NewInitializer_NewParenInitializer;
}

void ASTEncoder::visit(NewBracedInitializerAST* ast) {
  io::NewBracedInitializer::Builder builder{fbb_};

  const auto bracedInit = accept(ast->bracedInit);

  if (!bracedInit.IsNull()) {
    builder.add_braced_init(bracedInit.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::NewInitializer_NewBracedInitializer;
}

void ASTEncoder::visit(EllipsisExceptionDeclarationAST* ast) {
  io::EllipsisExceptionDeclaration::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::ExceptionDeclaration_EllipsisExceptionDeclaration;
}

void ASTEncoder::visit(TypeExceptionDeclarationAST* ast) {
  io::TypeExceptionDeclaration::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  if (ast->typeSpecifierList) {
    std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      typeSpecifierListOffsets.push_back(offset);
      typeSpecifierListTypes.push_back(type);
    }
    builder.add_type_specifier_list(
        fbb_.CreateVector(typeSpecifierListOffsets));
    builder.add_type_specifier_list_type(
        fbb_.CreateVector(typeSpecifierListTypes));
  }

  const auto declarator = accept(ast->declarator);

  if (!declarator.IsNull()) {
    builder.add_declarator(declarator.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::ExceptionDeclaration_TypeExceptionDeclaration;
}

void ASTEncoder::visit(DefaultFunctionBodyAST* ast) {
  io::DefaultFunctionBody::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_DefaultFunctionBody;
}

void ASTEncoder::visit(CompoundStatementFunctionBodyAST* ast) {
  io::CompoundStatementFunctionBody::Builder builder{fbb_};

  const auto ctorInitializer = accept(ast->ctorInitializer);

  if (!ctorInitializer.IsNull()) {
    builder.add_ctor_initializer(ctorInitializer.o);
  }

  const auto statement = accept(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_CompoundStatementFunctionBody;
}

void ASTEncoder::visit(TryStatementFunctionBodyAST* ast) {
  io::TryStatementFunctionBody::Builder builder{fbb_};

  const auto ctorInitializer = accept(ast->ctorInitializer);

  if (!ctorInitializer.IsNull()) {
    builder.add_ctor_initializer(ctorInitializer.o);
  }

  const auto statement = accept(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement.o);
  }

  if (ast->handlerList) {
    std::vector<flatbuffers::Offset<io::Handler>> handlerListOffsets;
    for (auto it = ast->handlerList; it; it = it->next) {
      handlerListOffsets.emplace_back(accept(it->value).o);
    }
    builder.add_handler_list(fbb_.CreateVector(handlerListOffsets));
  }

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_TryStatementFunctionBody;
}

void ASTEncoder::visit(DeleteFunctionBodyAST* ast) {
  io::DeleteFunctionBody::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::FunctionBody_DeleteFunctionBody;
}

void ASTEncoder::visit(TranslationUnitAST* ast) {
  io::TranslationUnit::Builder builder{fbb_};

  if (ast->declarationList) {
    std::vector<flatbuffers::Offset<>> declarationListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;
    for (auto it = ast->declarationList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      declarationListOffsets.push_back(offset);
      declarationListTypes.push_back(type);
    }
    builder.add_declaration_list(fbb_.CreateVector(declarationListOffsets));
    builder.add_declaration_list_type(fbb_.CreateVector(declarationListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Unit_TranslationUnit;
}

void ASTEncoder::visit(ModuleUnitAST* ast) {
  io::ModuleUnit::Builder builder{fbb_};

  const auto globalModuleFragment = accept(ast->globalModuleFragment);

  if (!globalModuleFragment.IsNull()) {
    builder.add_global_module_fragment(globalModuleFragment.o);
  }

  const auto moduleDeclaration = accept(ast->moduleDeclaration);

  if (!moduleDeclaration.IsNull()) {
    builder.add_module_declaration(moduleDeclaration.o);
  }

  if (ast->declarationList) {
    std::vector<flatbuffers::Offset<>> declarationListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;
    for (auto it = ast->declarationList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      declarationListOffsets.push_back(offset);
      declarationListTypes.push_back(type);
    }
    builder.add_declaration_list(fbb_.CreateVector(declarationListOffsets));
    builder.add_declaration_list_type(fbb_.CreateVector(declarationListTypes));
  }

  const auto privateModuleFragment = accept(ast->privateModuleFragment);

  if (!privateModuleFragment.IsNull()) {
    builder.add_private_module_fragment(privateModuleFragment.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Unit_ModuleUnit;
}

void ASTEncoder::visit(ThisExpressionAST* ast) {
  io::ThisExpression::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ThisExpression;
}

void ASTEncoder::visit(CharLiteralExpressionAST* ast) {
  io::CharLiteralExpression::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CharLiteralExpression;
}

void ASTEncoder::visit(BoolLiteralExpressionAST* ast) {
  io::BoolLiteralExpression::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BoolLiteralExpression;
}

void ASTEncoder::visit(IntLiteralExpressionAST* ast) {
  io::IntLiteralExpression::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Expression_IntLiteralExpression;
}

void ASTEncoder::visit(FloatLiteralExpressionAST* ast) {
  io::FloatLiteralExpression::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Expression_FloatLiteralExpression;
}

void ASTEncoder::visit(NullptrLiteralExpressionAST* ast) {
  io::NullptrLiteralExpression::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NullptrLiteralExpression;
}

void ASTEncoder::visit(StringLiteralExpressionAST* ast) {
  io::StringLiteralExpression::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Expression_StringLiteralExpression;
}

void ASTEncoder::visit(UserDefinedStringLiteralExpressionAST* ast) {
  io::UserDefinedStringLiteralExpression::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Expression_UserDefinedStringLiteralExpression;
}

void ASTEncoder::visit(IdExpressionAST* ast) {
  io::IdExpression::Builder builder{fbb_};

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_IdExpression;
}

void ASTEncoder::visit(RequiresExpressionAST* ast) {
  io::RequiresExpression::Builder builder{fbb_};

  const auto parameterDeclarationClause =
      accept(ast->parameterDeclarationClause);

  if (!parameterDeclarationClause.IsNull()) {
    builder.add_parameter_declaration_clause(parameterDeclarationClause.o);
  }

  const auto requirementBody = accept(ast->requirementBody);

  if (!requirementBody.IsNull()) {
    builder.add_requirement_body(requirementBody.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_RequiresExpression;
}

void ASTEncoder::visit(NestedExpressionAST* ast) {
  io::NestedExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NestedExpression;
}

void ASTEncoder::visit(RightFoldExpressionAST* ast) {
  io::RightFoldExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_RightFoldExpression;
}

void ASTEncoder::visit(LeftFoldExpressionAST* ast) {
  io::LeftFoldExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_LeftFoldExpression;
}

void ASTEncoder::visit(FoldExpressionAST* ast) {
  io::FoldExpression::Builder builder{fbb_};

  const auto [leftExpression, leftExpressionType] =
      acceptExpression(ast->leftExpression);

  if (!leftExpression.IsNull()) {
    builder.add_left_expression(leftExpression);
    builder.add_left_expression_type(
        static_cast<io::Expression>(leftExpressionType));
  }

  const auto [rightExpression, rightExpressionType] =
      acceptExpression(ast->rightExpression);

  if (!rightExpression.IsNull()) {
    builder.add_right_expression(rightExpression);
    builder.add_right_expression_type(
        static_cast<io::Expression>(rightExpressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_FoldExpression;
}

void ASTEncoder::visit(LambdaExpressionAST* ast) {
  io::LambdaExpression::Builder builder{fbb_};

  const auto lambdaIntroducer = accept(ast->lambdaIntroducer);

  if (!lambdaIntroducer.IsNull()) {
    builder.add_lambda_introducer(lambdaIntroducer.o);
  }

  if (ast->templateParameterList) {
    std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>>
        templateParameterListTypes;
    for (auto it = ast->templateParameterList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      templateParameterListOffsets.push_back(offset);
      templateParameterListTypes.push_back(type);
    }
    builder.add_template_parameter_list(
        fbb_.CreateVector(templateParameterListOffsets));
    builder.add_template_parameter_list_type(
        fbb_.CreateVector(templateParameterListTypes));
  }

  const auto requiresClause = accept(ast->requiresClause);

  if (!requiresClause.IsNull()) {
    builder.add_requires_clause(requiresClause.o);
  }

  const auto lambdaDeclarator = accept(ast->lambdaDeclarator);

  if (!lambdaDeclarator.IsNull()) {
    builder.add_lambda_declarator(lambdaDeclarator.o);
  }

  const auto statement = accept(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_LambdaExpression;
}

void ASTEncoder::visit(SizeofExpressionAST* ast) {
  io::SizeofExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SizeofExpression;
}

void ASTEncoder::visit(SizeofTypeExpressionAST* ast) {
  io::SizeofTypeExpression::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SizeofTypeExpression;
}

void ASTEncoder::visit(SizeofPackExpressionAST* ast) {
  io::SizeofPackExpression::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SizeofPackExpression;
}

void ASTEncoder::visit(TypeidExpressionAST* ast) {
  io::TypeidExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeidExpression;
}

void ASTEncoder::visit(TypeidOfTypeExpressionAST* ast) {
  io::TypeidOfTypeExpression::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeidOfTypeExpression;
}

void ASTEncoder::visit(AlignofExpressionAST* ast) {
  io::AlignofExpression::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AlignofExpression;
}

void ASTEncoder::visit(TypeTraitsExpressionAST* ast) {
  io::TypeTraitsExpression::Builder builder{fbb_};

  if (ast->typeIdList) {
    std::vector<flatbuffers::Offset<io::TypeId>> typeIdListOffsets;
    for (auto it = ast->typeIdList; it; it = it->next) {
      typeIdListOffsets.emplace_back(accept(it->value).o);
    }
    builder.add_type_id_list(fbb_.CreateVector(typeIdListOffsets));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeTraitsExpression;
}

void ASTEncoder::visit(UnaryExpressionAST* ast) {
  io::UnaryExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_UnaryExpression;
}

void ASTEncoder::visit(BinaryExpressionAST* ast) {
  io::BinaryExpression::Builder builder{fbb_};

  const auto [leftExpression, leftExpressionType] =
      acceptExpression(ast->leftExpression);

  if (!leftExpression.IsNull()) {
    builder.add_left_expression(leftExpression);
    builder.add_left_expression_type(
        static_cast<io::Expression>(leftExpressionType));
  }

  const auto [rightExpression, rightExpressionType] =
      acceptExpression(ast->rightExpression);

  if (!rightExpression.IsNull()) {
    builder.add_right_expression(rightExpression);
    builder.add_right_expression_type(
        static_cast<io::Expression>(rightExpressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BinaryExpression;
}

void ASTEncoder::visit(AssignmentExpressionAST* ast) {
  io::AssignmentExpression::Builder builder{fbb_};

  const auto [leftExpression, leftExpressionType] =
      acceptExpression(ast->leftExpression);

  if (!leftExpression.IsNull()) {
    builder.add_left_expression(leftExpression);
    builder.add_left_expression_type(
        static_cast<io::Expression>(leftExpressionType));
  }

  const auto [rightExpression, rightExpressionType] =
      acceptExpression(ast->rightExpression);

  if (!rightExpression.IsNull()) {
    builder.add_right_expression(rightExpression);
    builder.add_right_expression_type(
        static_cast<io::Expression>(rightExpressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_AssignmentExpression;
}

void ASTEncoder::visit(BracedTypeConstructionAST* ast) {
  io::BracedTypeConstruction::Builder builder{fbb_};

  const auto [typeSpecifier, typeSpecifierType] =
      acceptSpecifier(ast->typeSpecifier);

  if (!typeSpecifier.IsNull()) {
    builder.add_type_specifier(typeSpecifier);
    builder.add_type_specifier_type(
        static_cast<io::Specifier>(typeSpecifierType));
  }

  const auto bracedInitList = accept(ast->bracedInitList);

  if (!bracedInitList.IsNull()) {
    builder.add_braced_init_list(bracedInitList.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_BracedTypeConstruction;
}

void ASTEncoder::visit(TypeConstructionAST* ast) {
  io::TypeConstruction::Builder builder{fbb_};

  const auto [typeSpecifier, typeSpecifierType] =
      acceptSpecifier(ast->typeSpecifier);

  if (!typeSpecifier.IsNull()) {
    builder.add_type_specifier(typeSpecifier);
    builder.add_type_specifier_type(
        static_cast<io::Specifier>(typeSpecifierType));
  }

  if (ast->expressionList) {
    std::vector<flatbuffers::Offset<>> expressionListOffsets;
    std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;
    for (auto it = ast->expressionList; it; it = it->next) {
      const auto [offset, type] = acceptExpression(it->value);
      expressionListOffsets.push_back(offset);
      expressionListTypes.push_back(type);
    }
    builder.add_expression_list(fbb_.CreateVector(expressionListOffsets));
    builder.add_expression_list_type(fbb_.CreateVector(expressionListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_TypeConstruction;
}

void ASTEncoder::visit(CallExpressionAST* ast) {
  io::CallExpression::Builder builder{fbb_};

  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  if (!baseExpression.IsNull()) {
    builder.add_base_expression(baseExpression);
    builder.add_base_expression_type(
        static_cast<io::Expression>(baseExpressionType));
  }

  if (ast->expressionList) {
    std::vector<flatbuffers::Offset<>> expressionListOffsets;
    std::vector<std::underlying_type_t<io::Expression>> expressionListTypes;
    for (auto it = ast->expressionList; it; it = it->next) {
      const auto [offset, type] = acceptExpression(it->value);
      expressionListOffsets.push_back(offset);
      expressionListTypes.push_back(type);
    }
    builder.add_expression_list(fbb_.CreateVector(expressionListOffsets));
    builder.add_expression_list_type(fbb_.CreateVector(expressionListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CallExpression;
}

void ASTEncoder::visit(SubscriptExpressionAST* ast) {
  io::SubscriptExpression::Builder builder{fbb_};

  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  if (!baseExpression.IsNull()) {
    builder.add_base_expression(baseExpression);
    builder.add_base_expression_type(
        static_cast<io::Expression>(baseExpressionType));
  }

  const auto [indexExpression, indexExpressionType] =
      acceptExpression(ast->indexExpression);

  if (!indexExpression.IsNull()) {
    builder.add_index_expression(indexExpression);
    builder.add_index_expression_type(
        static_cast<io::Expression>(indexExpressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_SubscriptExpression;
}

void ASTEncoder::visit(MemberExpressionAST* ast) {
  io::MemberExpression::Builder builder{fbb_};

  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  if (!baseExpression.IsNull()) {
    builder.add_base_expression(baseExpression);
    builder.add_base_expression_type(
        static_cast<io::Expression>(baseExpressionType));
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_MemberExpression;
}

void ASTEncoder::visit(PostIncrExpressionAST* ast) {
  io::PostIncrExpression::Builder builder{fbb_};

  const auto [baseExpression, baseExpressionType] =
      acceptExpression(ast->baseExpression);

  if (!baseExpression.IsNull()) {
    builder.add_base_expression(baseExpression);
    builder.add_base_expression_type(
        static_cast<io::Expression>(baseExpressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_PostIncrExpression;
}

void ASTEncoder::visit(ConditionalExpressionAST* ast) {
  io::ConditionalExpression::Builder builder{fbb_};

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  if (!condition.IsNull()) {
    builder.add_condition(condition);
    builder.add_condition_type(static_cast<io::Expression>(conditionType));
  }

  const auto [iftrueExpression, iftrueExpressionType] =
      acceptExpression(ast->iftrueExpression);

  if (!iftrueExpression.IsNull()) {
    builder.add_iftrue_expression(iftrueExpression);
    builder.add_iftrue_expression_type(
        static_cast<io::Expression>(iftrueExpressionType));
  }

  const auto [iffalseExpression, iffalseExpressionType] =
      acceptExpression(ast->iffalseExpression);

  if (!iffalseExpression.IsNull()) {
    builder.add_iffalse_expression(iffalseExpression);
    builder.add_iffalse_expression_type(
        static_cast<io::Expression>(iffalseExpressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ConditionalExpression;
}

void ASTEncoder::visit(ImplicitCastExpressionAST* ast) {
  io::ImplicitCastExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ImplicitCastExpression;
}

void ASTEncoder::visit(CastExpressionAST* ast) {
  io::CastExpression::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CastExpression;
}

void ASTEncoder::visit(CppCastExpressionAST* ast) {
  io::CppCastExpression::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_CppCastExpression;
}

void ASTEncoder::visit(NewExpressionAST* ast) {
  io::NewExpression::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  const auto [newInitalizer, newInitalizerType] =
      acceptNewInitializer(ast->newInitalizer);

  if (!newInitalizer.IsNull()) {
    builder.add_new_initalizer(newInitalizer);
    builder.add_new_initalizer_type(
        static_cast<io::NewInitializer>(newInitalizerType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NewExpression;
}

void ASTEncoder::visit(DeleteExpressionAST* ast) {
  io::DeleteExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_DeleteExpression;
}

void ASTEncoder::visit(ThrowExpressionAST* ast) {
  io::ThrowExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_ThrowExpression;
}

void ASTEncoder::visit(NoexceptExpressionAST* ast) {
  io::NoexceptExpression::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Expression_NoexceptExpression;
}

void ASTEncoder::visit(LabeledStatementAST* ast) {
  io::LabeledStatement::Builder builder{fbb_};

  const auto [statement, statementType] = acceptStatement(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement);
    builder.add_statement_type(static_cast<io::Statement>(statementType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_LabeledStatement;
}

void ASTEncoder::visit(CaseStatementAST* ast) {
  io::CaseStatement::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  const auto [statement, statementType] = acceptStatement(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement);
    builder.add_statement_type(static_cast<io::Statement>(statementType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_CaseStatement;
}

void ASTEncoder::visit(DefaultStatementAST* ast) {
  io::DefaultStatement::Builder builder{fbb_};

  const auto [statement, statementType] = acceptStatement(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement);
    builder.add_statement_type(static_cast<io::Statement>(statementType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_DefaultStatement;
}

void ASTEncoder::visit(ExpressionStatementAST* ast) {
  io::ExpressionStatement::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ExpressionStatement;
}

void ASTEncoder::visit(CompoundStatementAST* ast) {
  io::CompoundStatement::Builder builder{fbb_};

  if (ast->statementList) {
    std::vector<flatbuffers::Offset<>> statementListOffsets;
    std::vector<std::underlying_type_t<io::Statement>> statementListTypes;
    for (auto it = ast->statementList; it; it = it->next) {
      const auto [offset, type] = acceptStatement(it->value);
      statementListOffsets.push_back(offset);
      statementListTypes.push_back(type);
    }
    builder.add_statement_list(fbb_.CreateVector(statementListOffsets));
    builder.add_statement_list_type(fbb_.CreateVector(statementListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_CompoundStatement;
}

void ASTEncoder::visit(IfStatementAST* ast) {
  io::IfStatement::Builder builder{fbb_};

  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  if (!initializer.IsNull()) {
    builder.add_initializer(initializer);
    builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  }

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  if (!condition.IsNull()) {
    builder.add_condition(condition);
    builder.add_condition_type(static_cast<io::Expression>(conditionType));
  }

  const auto [statement, statementType] = acceptStatement(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement);
    builder.add_statement_type(static_cast<io::Statement>(statementType));
  }

  const auto [elseStatement, elseStatementType] =
      acceptStatement(ast->elseStatement);

  if (!elseStatement.IsNull()) {
    builder.add_else_statement(elseStatement);
    builder.add_else_statement_type(
        static_cast<io::Statement>(elseStatementType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_IfStatement;
}

void ASTEncoder::visit(SwitchStatementAST* ast) {
  io::SwitchStatement::Builder builder{fbb_};

  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  if (!initializer.IsNull()) {
    builder.add_initializer(initializer);
    builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  }

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  if (!condition.IsNull()) {
    builder.add_condition(condition);
    builder.add_condition_type(static_cast<io::Expression>(conditionType));
  }

  const auto [statement, statementType] = acceptStatement(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement);
    builder.add_statement_type(static_cast<io::Statement>(statementType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_SwitchStatement;
}

void ASTEncoder::visit(WhileStatementAST* ast) {
  io::WhileStatement::Builder builder{fbb_};

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  if (!condition.IsNull()) {
    builder.add_condition(condition);
    builder.add_condition_type(static_cast<io::Expression>(conditionType));
  }

  const auto [statement, statementType] = acceptStatement(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement);
    builder.add_statement_type(static_cast<io::Statement>(statementType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_WhileStatement;
}

void ASTEncoder::visit(DoStatementAST* ast) {
  io::DoStatement::Builder builder{fbb_};

  const auto [statement, statementType] = acceptStatement(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement);
    builder.add_statement_type(static_cast<io::Statement>(statementType));
  }

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_DoStatement;
}

void ASTEncoder::visit(ForRangeStatementAST* ast) {
  io::ForRangeStatement::Builder builder{fbb_};

  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  if (!initializer.IsNull()) {
    builder.add_initializer(initializer);
    builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  }

  const auto [rangeDeclaration, rangeDeclarationType] =
      acceptDeclaration(ast->rangeDeclaration);

  if (!rangeDeclaration.IsNull()) {
    builder.add_range_declaration(rangeDeclaration);
    builder.add_range_declaration_type(
        static_cast<io::Declaration>(rangeDeclarationType));
  }

  const auto [rangeInitializer, rangeInitializerType] =
      acceptExpression(ast->rangeInitializer);

  if (!rangeInitializer.IsNull()) {
    builder.add_range_initializer(rangeInitializer);
    builder.add_range_initializer_type(
        static_cast<io::Expression>(rangeInitializerType));
  }

  const auto [statement, statementType] = acceptStatement(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement);
    builder.add_statement_type(static_cast<io::Statement>(statementType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ForRangeStatement;
}

void ASTEncoder::visit(ForStatementAST* ast) {
  io::ForStatement::Builder builder{fbb_};

  const auto [initializer, initializerType] = acceptStatement(ast->initializer);

  if (!initializer.IsNull()) {
    builder.add_initializer(initializer);
    builder.add_initializer_type(static_cast<io::Statement>(initializerType));
  }

  const auto [condition, conditionType] = acceptExpression(ast->condition);

  if (!condition.IsNull()) {
    builder.add_condition(condition);
    builder.add_condition_type(static_cast<io::Expression>(conditionType));
  }

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  const auto [statement, statementType] = acceptStatement(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement);
    builder.add_statement_type(static_cast<io::Statement>(statementType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ForStatement;
}

void ASTEncoder::visit(BreakStatementAST* ast) {
  io::BreakStatement::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Statement_BreakStatement;
}

void ASTEncoder::visit(ContinueStatementAST* ast) {
  io::ContinueStatement::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ContinueStatement;
}

void ASTEncoder::visit(ReturnStatementAST* ast) {
  io::ReturnStatement::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_ReturnStatement;
}

void ASTEncoder::visit(GotoStatementAST* ast) {
  io::GotoStatement::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Statement_GotoStatement;
}

void ASTEncoder::visit(CoroutineReturnStatementAST* ast) {
  io::CoroutineReturnStatement::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_CoroutineReturnStatement;
}

void ASTEncoder::visit(DeclarationStatementAST* ast) {
  io::DeclarationStatement::Builder builder{fbb_};

  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  if (!declaration.IsNull()) {
    builder.add_declaration(declaration);
    builder.add_declaration_type(static_cast<io::Declaration>(declarationType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_DeclarationStatement;
}

void ASTEncoder::visit(TryBlockStatementAST* ast) {
  io::TryBlockStatement::Builder builder{fbb_};

  const auto statement = accept(ast->statement);

  if (!statement.IsNull()) {
    builder.add_statement(statement.o);
  }

  if (ast->handlerList) {
    std::vector<flatbuffers::Offset<io::Handler>> handlerListOffsets;
    for (auto it = ast->handlerList; it; it = it->next) {
      handlerListOffsets.emplace_back(accept(it->value).o);
    }
    builder.add_handler_list(fbb_.CreateVector(handlerListOffsets));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Statement_TryBlockStatement;
}

void ASTEncoder::visit(AccessDeclarationAST* ast) {
  io::AccessDeclaration::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AccessDeclaration;
}

void ASTEncoder::visit(FunctionDefinitionAST* ast) {
  io::FunctionDefinition::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  if (ast->declSpecifierList) {
    std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      declSpecifierListOffsets.push_back(offset);
      declSpecifierListTypes.push_back(type);
    }
    builder.add_decl_specifier_list(
        fbb_.CreateVector(declSpecifierListOffsets));
    builder.add_decl_specifier_list_type(
        fbb_.CreateVector(declSpecifierListTypes));
  }

  const auto declarator = accept(ast->declarator);

  if (!declarator.IsNull()) {
    builder.add_declarator(declarator.o);
  }

  const auto requiresClause = accept(ast->requiresClause);

  if (!requiresClause.IsNull()) {
    builder.add_requires_clause(requiresClause.o);
  }

  const auto [functionBody, functionBodyType] =
      acceptFunctionBody(ast->functionBody);

  if (!functionBody.IsNull()) {
    builder.add_function_body(functionBody);
    builder.add_function_body_type(
        static_cast<io::FunctionBody>(functionBodyType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_FunctionDefinition;
}

void ASTEncoder::visit(ConceptDefinitionAST* ast) {
  io::ConceptDefinition::Builder builder{fbb_};

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ConceptDefinition;
}

void ASTEncoder::visit(ForRangeDeclarationAST* ast) {
  io::ForRangeDeclaration::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ForRangeDeclaration;
}

void ASTEncoder::visit(AliasDeclarationAST* ast) {
  io::AliasDeclaration::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AliasDeclaration;
}

void ASTEncoder::visit(SimpleDeclarationAST* ast) {
  io::SimpleDeclaration::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  if (ast->declSpecifierList) {
    std::vector<flatbuffers::Offset<>> declSpecifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> declSpecifierListTypes;
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      declSpecifierListOffsets.push_back(offset);
      declSpecifierListTypes.push_back(type);
    }
    builder.add_decl_specifier_list(
        fbb_.CreateVector(declSpecifierListOffsets));
    builder.add_decl_specifier_list_type(
        fbb_.CreateVector(declSpecifierListTypes));
  }

  if (ast->initDeclaratorList) {
    std::vector<flatbuffers::Offset<io::InitDeclarator>>
        initDeclaratorListOffsets;
    for (auto it = ast->initDeclaratorList; it; it = it->next) {
      initDeclaratorListOffsets.emplace_back(accept(it->value).o);
    }
    builder.add_init_declarator_list(
        fbb_.CreateVector(initDeclaratorListOffsets));
  }

  const auto requiresClause = accept(ast->requiresClause);

  if (!requiresClause.IsNull()) {
    builder.add_requires_clause(requiresClause.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_SimpleDeclaration;
}

void ASTEncoder::visit(StaticAssertDeclarationAST* ast) {
  io::StaticAssertDeclaration::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_StaticAssertDeclaration;
}

void ASTEncoder::visit(EmptyDeclarationAST* ast) {
  io::EmptyDeclaration::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_EmptyDeclaration;
}

void ASTEncoder::visit(AttributeDeclarationAST* ast) {
  io::AttributeDeclaration::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AttributeDeclaration;
}

void ASTEncoder::visit(OpaqueEnumDeclarationAST* ast) {
  io::OpaqueEnumDeclaration::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  const auto enumBase = accept(ast->enumBase);

  if (!enumBase.IsNull()) {
    builder.add_enum_base(enumBase.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_OpaqueEnumDeclaration;
}

void ASTEncoder::visit(UsingEnumDeclarationAST* ast) {
  io::UsingEnumDeclaration::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_UsingEnumDeclaration;
}

void ASTEncoder::visit(NamespaceDefinitionAST* ast) {
  io::NamespaceDefinition::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  if (ast->extraAttributeList) {
    std::vector<flatbuffers::Offset<>> extraAttributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> extraAttributeListTypes;
    for (auto it = ast->extraAttributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      extraAttributeListOffsets.push_back(offset);
      extraAttributeListTypes.push_back(type);
    }
    builder.add_extra_attribute_list(
        fbb_.CreateVector(extraAttributeListOffsets));
    builder.add_extra_attribute_list_type(
        fbb_.CreateVector(extraAttributeListTypes));
  }

  if (ast->declarationList) {
    std::vector<flatbuffers::Offset<>> declarationListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;
    for (auto it = ast->declarationList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      declarationListOffsets.push_back(offset);
      declarationListTypes.push_back(type);
    }
    builder.add_declaration_list(fbb_.CreateVector(declarationListOffsets));
    builder.add_declaration_list_type(fbb_.CreateVector(declarationListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_NamespaceDefinition;
}

void ASTEncoder::visit(NamespaceAliasDefinitionAST* ast) {
  io::NamespaceAliasDefinition::Builder builder{fbb_};

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_NamespaceAliasDefinition;
}

void ASTEncoder::visit(UsingDirectiveAST* ast) {
  io::UsingDirective::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_UsingDirective;
}

void ASTEncoder::visit(UsingDeclarationAST* ast) {
  io::UsingDeclaration::Builder builder{fbb_};

  if (ast->usingDeclaratorList) {
    std::vector<flatbuffers::Offset<io::UsingDeclarator>>
        usingDeclaratorListOffsets;
    for (auto it = ast->usingDeclaratorList; it; it = it->next) {
      usingDeclaratorListOffsets.emplace_back(accept(it->value).o);
    }
    builder.add_using_declarator_list(
        fbb_.CreateVector(usingDeclaratorListOffsets));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_UsingDeclaration;
}

void ASTEncoder::visit(AsmDeclarationAST* ast) {
  io::AsmDeclaration::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_AsmDeclaration;
}

void ASTEncoder::visit(ExportDeclarationAST* ast) {
  io::ExportDeclaration::Builder builder{fbb_};

  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  if (!declaration.IsNull()) {
    builder.add_declaration(declaration);
    builder.add_declaration_type(static_cast<io::Declaration>(declarationType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ExportDeclaration;
}

void ASTEncoder::visit(ExportCompoundDeclarationAST* ast) {
  io::ExportCompoundDeclaration::Builder builder{fbb_};

  if (ast->declarationList) {
    std::vector<flatbuffers::Offset<>> declarationListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;
    for (auto it = ast->declarationList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      declarationListOffsets.push_back(offset);
      declarationListTypes.push_back(type);
    }
    builder.add_declaration_list(fbb_.CreateVector(declarationListOffsets));
    builder.add_declaration_list_type(fbb_.CreateVector(declarationListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ExportCompoundDeclaration;
}

void ASTEncoder::visit(ModuleImportDeclarationAST* ast) {
  io::ModuleImportDeclaration::Builder builder{fbb_};

  const auto importName = accept(ast->importName);

  if (!importName.IsNull()) {
    builder.add_import_name(importName.o);
  }

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ModuleImportDeclaration;
}

void ASTEncoder::visit(TemplateDeclarationAST* ast) {
  io::TemplateDeclaration::Builder builder{fbb_};

  if (ast->templateParameterList) {
    std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>>
        templateParameterListTypes;
    for (auto it = ast->templateParameterList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      templateParameterListOffsets.push_back(offset);
      templateParameterListTypes.push_back(type);
    }
    builder.add_template_parameter_list(
        fbb_.CreateVector(templateParameterListOffsets));
    builder.add_template_parameter_list_type(
        fbb_.CreateVector(templateParameterListTypes));
  }

  const auto requiresClause = accept(ast->requiresClause);

  if (!requiresClause.IsNull()) {
    builder.add_requires_clause(requiresClause.o);
  }

  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  if (!declaration.IsNull()) {
    builder.add_declaration(declaration);
    builder.add_declaration_type(static_cast<io::Declaration>(declarationType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_TemplateDeclaration;
}

void ASTEncoder::visit(TypenameTypeParameterAST* ast) {
  io::TypenameTypeParameter::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_TypenameTypeParameter;
}

void ASTEncoder::visit(TemplateTypeParameterAST* ast) {
  io::TemplateTypeParameter::Builder builder{fbb_};

  if (ast->templateParameterList) {
    std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>>
        templateParameterListTypes;
    for (auto it = ast->templateParameterList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      templateParameterListOffsets.push_back(offset);
      templateParameterListTypes.push_back(type);
    }
    builder.add_template_parameter_list(
        fbb_.CreateVector(templateParameterListOffsets));
    builder.add_template_parameter_list_type(
        fbb_.CreateVector(templateParameterListTypes));
  }

  const auto requiresClause = accept(ast->requiresClause);

  if (!requiresClause.IsNull()) {
    builder.add_requires_clause(requiresClause.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_TemplateTypeParameter;
}

void ASTEncoder::visit(TemplatePackTypeParameterAST* ast) {
  io::TemplatePackTypeParameter::Builder builder{fbb_};

  if (ast->templateParameterList) {
    std::vector<flatbuffers::Offset<>> templateParameterListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>>
        templateParameterListTypes;
    for (auto it = ast->templateParameterList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      templateParameterListOffsets.push_back(offset);
      templateParameterListTypes.push_back(type);
    }
    builder.add_template_parameter_list(
        fbb_.CreateVector(templateParameterListOffsets));
    builder.add_template_parameter_list_type(
        fbb_.CreateVector(templateParameterListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_TemplatePackTypeParameter;
}

void ASTEncoder::visit(DeductionGuideAST* ast) {
  io::DeductionGuide::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_DeductionGuide;
}

void ASTEncoder::visit(ExplicitInstantiationAST* ast) {
  io::ExplicitInstantiation::Builder builder{fbb_};

  const auto [declaration, declarationType] =
      acceptDeclaration(ast->declaration);

  if (!declaration.IsNull()) {
    builder.add_declaration(declaration);
    builder.add_declaration_type(static_cast<io::Declaration>(declarationType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ExplicitInstantiation;
}

void ASTEncoder::visit(ParameterDeclarationAST* ast) {
  io::ParameterDeclaration::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  if (ast->typeSpecifierList) {
    std::vector<flatbuffers::Offset<>> typeSpecifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> typeSpecifierListTypes;
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      typeSpecifierListOffsets.push_back(offset);
      typeSpecifierListTypes.push_back(type);
    }
    builder.add_type_specifier_list(
        fbb_.CreateVector(typeSpecifierListOffsets));
    builder.add_type_specifier_list_type(
        fbb_.CreateVector(typeSpecifierListTypes));
  }

  const auto declarator = accept(ast->declarator);

  if (!declarator.IsNull()) {
    builder.add_declarator(declarator.o);
  }

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_ParameterDeclaration;
}

void ASTEncoder::visit(LinkageSpecificationAST* ast) {
  io::LinkageSpecification::Builder builder{fbb_};

  if (ast->declarationList) {
    std::vector<flatbuffers::Offset<>> declarationListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;
    for (auto it = ast->declarationList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      declarationListOffsets.push_back(offset);
      declarationListTypes.push_back(type);
    }
    builder.add_declaration_list(fbb_.CreateVector(declarationListOffsets));
    builder.add_declaration_list_type(fbb_.CreateVector(declarationListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Declaration_LinkageSpecification;
}

void ASTEncoder::visit(SimpleNameAST* ast) {
  io::SimpleName::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Name_SimpleName;
}

void ASTEncoder::visit(DestructorNameAST* ast) {
  io::DestructorName::Builder builder{fbb_};

  const auto [id, idType] = acceptName(ast->id);

  if (!id.IsNull()) {
    builder.add_id(id);
    builder.add_id_type(static_cast<io::Name>(idType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Name_DestructorName;
}

void ASTEncoder::visit(DecltypeNameAST* ast) {
  io::DecltypeName::Builder builder{fbb_};

  const auto [decltypeSpecifier, decltypeSpecifierType] =
      acceptSpecifier(ast->decltypeSpecifier);

  if (!decltypeSpecifier.IsNull()) {
    builder.add_decltype_specifier(decltypeSpecifier);
    builder.add_decltype_specifier_type(
        static_cast<io::Specifier>(decltypeSpecifierType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Name_DecltypeName;
}

void ASTEncoder::visit(OperatorNameAST* ast) {
  io::OperatorName::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Name_OperatorName;
}

void ASTEncoder::visit(ConversionNameAST* ast) {
  io::ConversionName::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Name_ConversionName;
}

void ASTEncoder::visit(TemplateNameAST* ast) {
  io::TemplateName::Builder builder{fbb_};

  const auto [id, idType] = acceptName(ast->id);

  if (!id.IsNull()) {
    builder.add_id(id);
    builder.add_id_type(static_cast<io::Name>(idType));
  }

  if (ast->templateArgumentList) {
    std::vector<flatbuffers::Offset<>> templateArgumentListOffsets;
    std::vector<std::underlying_type_t<io::TemplateArgument>>
        templateArgumentListTypes;
    for (auto it = ast->templateArgumentList; it; it = it->next) {
      const auto [offset, type] = acceptTemplateArgument(it->value);
      templateArgumentListOffsets.push_back(offset);
      templateArgumentListTypes.push_back(type);
    }
    builder.add_template_argument_list(
        fbb_.CreateVector(templateArgumentListOffsets));
    builder.add_template_argument_list_type(
        fbb_.CreateVector(templateArgumentListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Name_TemplateName;
}

void ASTEncoder::visit(QualifiedNameAST* ast) {
  io::QualifiedName::Builder builder{fbb_};

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [id, idType] = acceptName(ast->id);

  if (!id.IsNull()) {
    builder.add_id(id);
    builder.add_id_type(static_cast<io::Name>(idType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Name_QualifiedName;
}

void ASTEncoder::visit(TypedefSpecifierAST* ast) {
  io::TypedefSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_TypedefSpecifier;
}

void ASTEncoder::visit(FriendSpecifierAST* ast) {
  io::FriendSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_FriendSpecifier;
}

void ASTEncoder::visit(ConstevalSpecifierAST* ast) {
  io::ConstevalSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstevalSpecifier;
}

void ASTEncoder::visit(ConstinitSpecifierAST* ast) {
  io::ConstinitSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstinitSpecifier;
}

void ASTEncoder::visit(ConstexprSpecifierAST* ast) {
  io::ConstexprSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstexprSpecifier;
}

void ASTEncoder::visit(InlineSpecifierAST* ast) {
  io::InlineSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_InlineSpecifier;
}

void ASTEncoder::visit(StaticSpecifierAST* ast) {
  io::StaticSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_StaticSpecifier;
}

void ASTEncoder::visit(ExternSpecifierAST* ast) {
  io::ExternSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ExternSpecifier;
}

void ASTEncoder::visit(ThreadLocalSpecifierAST* ast) {
  io::ThreadLocalSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ThreadLocalSpecifier;
}

void ASTEncoder::visit(ThreadSpecifierAST* ast) {
  io::ThreadSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ThreadSpecifier;
}

void ASTEncoder::visit(MutableSpecifierAST* ast) {
  io::MutableSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_MutableSpecifier;
}

void ASTEncoder::visit(VirtualSpecifierAST* ast) {
  io::VirtualSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VirtualSpecifier;
}

void ASTEncoder::visit(ExplicitSpecifierAST* ast) {
  io::ExplicitSpecifier::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ExplicitSpecifier;
}

void ASTEncoder::visit(AutoTypeSpecifierAST* ast) {
  io::AutoTypeSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_AutoTypeSpecifier;
}

void ASTEncoder::visit(VoidTypeSpecifierAST* ast) {
  io::VoidTypeSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VoidTypeSpecifier;
}

void ASTEncoder::visit(VaListTypeSpecifierAST* ast) {
  io::VaListTypeSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VaListTypeSpecifier;
}

void ASTEncoder::visit(IntegralTypeSpecifierAST* ast) {
  io::IntegralTypeSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_IntegralTypeSpecifier;
}

void ASTEncoder::visit(FloatingPointTypeSpecifierAST* ast) {
  io::FloatingPointTypeSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_FloatingPointTypeSpecifier;
}

void ASTEncoder::visit(ComplexTypeSpecifierAST* ast) {
  io::ComplexTypeSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ComplexTypeSpecifier;
}

void ASTEncoder::visit(NamedTypeSpecifierAST* ast) {
  io::NamedTypeSpecifier::Builder builder{fbb_};

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_NamedTypeSpecifier;
}

void ASTEncoder::visit(AtomicTypeSpecifierAST* ast) {
  io::AtomicTypeSpecifier::Builder builder{fbb_};

  const auto typeId = accept(ast->typeId);

  if (!typeId.IsNull()) {
    builder.add_type_id(typeId.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_AtomicTypeSpecifier;
}

void ASTEncoder::visit(UnderlyingTypeSpecifierAST* ast) {
  io::UnderlyingTypeSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_UnderlyingTypeSpecifier;
}

void ASTEncoder::visit(ElaboratedTypeSpecifierAST* ast) {
  io::ElaboratedTypeSpecifier::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ElaboratedTypeSpecifier;
}

void ASTEncoder::visit(DecltypeAutoSpecifierAST* ast) {
  io::DecltypeAutoSpecifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_DecltypeAutoSpecifier;
}

void ASTEncoder::visit(DecltypeSpecifierAST* ast) {
  io::DecltypeSpecifier::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_DecltypeSpecifier;
}

void ASTEncoder::visit(PlaceholderTypeSpecifierAST* ast) {
  io::PlaceholderTypeSpecifier::Builder builder{fbb_};

  const auto typeConstraint = accept(ast->typeConstraint);

  if (!typeConstraint.IsNull()) {
    builder.add_type_constraint(typeConstraint.o);
  }

  const auto [specifier, specifierType] = acceptSpecifier(ast->specifier);

  if (!specifier.IsNull()) {
    builder.add_specifier(specifier);
    builder.add_specifier_type(static_cast<io::Specifier>(specifierType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_PlaceholderTypeSpecifier;
}

void ASTEncoder::visit(ConstQualifierAST* ast) {
  io::ConstQualifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ConstQualifier;
}

void ASTEncoder::visit(VolatileQualifierAST* ast) {
  io::VolatileQualifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_VolatileQualifier;
}

void ASTEncoder::visit(RestrictQualifierAST* ast) {
  io::RestrictQualifier::Builder builder{fbb_};

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_RestrictQualifier;
}

void ASTEncoder::visit(EnumSpecifierAST* ast) {
  io::EnumSpecifier::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  const auto enumBase = accept(ast->enumBase);

  if (!enumBase.IsNull()) {
    builder.add_enum_base(enumBase.o);
  }

  if (ast->enumeratorList) {
    std::vector<flatbuffers::Offset<io::Enumerator>> enumeratorListOffsets;
    for (auto it = ast->enumeratorList; it; it = it->next) {
      enumeratorListOffsets.emplace_back(accept(it->value).o);
    }
    builder.add_enumerator_list(fbb_.CreateVector(enumeratorListOffsets));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_EnumSpecifier;
}

void ASTEncoder::visit(ClassSpecifierAST* ast) {
  io::ClassSpecifier::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  const auto baseClause = accept(ast->baseClause);

  if (!baseClause.IsNull()) {
    builder.add_base_clause(baseClause.o);
  }

  if (ast->declarationList) {
    std::vector<flatbuffers::Offset<>> declarationListOffsets;
    std::vector<std::underlying_type_t<io::Declaration>> declarationListTypes;
    for (auto it = ast->declarationList; it; it = it->next) {
      const auto [offset, type] = acceptDeclaration(it->value);
      declarationListOffsets.push_back(offset);
      declarationListTypes.push_back(type);
    }
    builder.add_declaration_list(fbb_.CreateVector(declarationListOffsets));
    builder.add_declaration_list_type(fbb_.CreateVector(declarationListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_ClassSpecifier;
}

void ASTEncoder::visit(TypenameSpecifierAST* ast) {
  io::TypenameSpecifier::Builder builder{fbb_};

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  offset_ = builder.Finish().Union();
  type_ = io::Specifier_TypenameSpecifier;
}

void ASTEncoder::visit(IdDeclaratorAST* ast) {
  io::IdDeclarator::Builder builder{fbb_};

  const auto [name, nameType] = acceptName(ast->name);

  if (!name.IsNull()) {
    builder.add_name(name);
    builder.add_name_type(static_cast<io::Name>(nameType));
  }

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_IdDeclarator;
}

void ASTEncoder::visit(NestedDeclaratorAST* ast) {
  io::NestedDeclarator::Builder builder{fbb_};

  const auto declarator = accept(ast->declarator);

  if (!declarator.IsNull()) {
    builder.add_declarator(declarator.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::CoreDeclarator_NestedDeclarator;
}

void ASTEncoder::visit(PointerOperatorAST* ast) {
  io::PointerOperator::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  if (ast->cvQualifierList) {
    std::vector<flatbuffers::Offset<>> cvQualifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> cvQualifierListTypes;
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      cvQualifierListOffsets.push_back(offset);
      cvQualifierListTypes.push_back(type);
    }
    builder.add_cv_qualifier_list(fbb_.CreateVector(cvQualifierListOffsets));
    builder.add_cv_qualifier_list_type(fbb_.CreateVector(cvQualifierListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::PtrOperator_PointerOperator;
}

void ASTEncoder::visit(ReferenceOperatorAST* ast) {
  io::ReferenceOperator::Builder builder{fbb_};

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::PtrOperator_ReferenceOperator;
}

void ASTEncoder::visit(PtrToMemberOperatorAST* ast) {
  io::PtrToMemberOperator::Builder builder{fbb_};

  const auto nestedNameSpecifier = accept(ast->nestedNameSpecifier);

  if (!nestedNameSpecifier.IsNull()) {
    builder.add_nested_name_specifier(nestedNameSpecifier.o);
  }

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  if (ast->cvQualifierList) {
    std::vector<flatbuffers::Offset<>> cvQualifierListOffsets;
    std::vector<std::underlying_type_t<io::Specifier>> cvQualifierListTypes;
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      const auto [offset, type] = acceptSpecifier(it->value);
      cvQualifierListOffsets.push_back(offset);
      cvQualifierListTypes.push_back(type);
    }
    builder.add_cv_qualifier_list(fbb_.CreateVector(cvQualifierListOffsets));
    builder.add_cv_qualifier_list_type(fbb_.CreateVector(cvQualifierListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::PtrOperator_PtrToMemberOperator;
}

void ASTEncoder::visit(FunctionDeclaratorAST* ast) {
  io::FunctionDeclarator::Builder builder{fbb_};

  const auto parametersAndQualifiers = accept(ast->parametersAndQualifiers);

  if (!parametersAndQualifiers.IsNull()) {
    builder.add_parameters_and_qualifiers(parametersAndQualifiers.o);
  }

  const auto trailingReturnType = accept(ast->trailingReturnType);

  if (!trailingReturnType.IsNull()) {
    builder.add_trailing_return_type(trailingReturnType.o);
  }

  offset_ = builder.Finish().Union();
  type_ = io::DeclaratorModifier_FunctionDeclarator;
}

void ASTEncoder::visit(ArrayDeclaratorAST* ast) {
  io::ArrayDeclarator::Builder builder{fbb_};

  const auto [expression, expressionType] = acceptExpression(ast->expression);

  if (!expression.IsNull()) {
    builder.add_expression(expression);
    builder.add_expression_type(static_cast<io::Expression>(expressionType));
  }

  if (ast->attributeList) {
    std::vector<flatbuffers::Offset<>> attributeListOffsets;
    std::vector<std::underlying_type_t<io::Attribute>> attributeListTypes;
    for (auto it = ast->attributeList; it; it = it->next) {
      const auto [offset, type] = acceptAttribute(it->value);
      attributeListOffsets.push_back(offset);
      attributeListTypes.push_back(type);
    }
    builder.add_attribute_list(fbb_.CreateVector(attributeListOffsets));
    builder.add_attribute_list_type(fbb_.CreateVector(attributeListTypes));
  }

  offset_ = builder.Finish().Union();
  type_ = io::DeclaratorModifier_ArrayDeclarator;
}

}  // namespace cxx
