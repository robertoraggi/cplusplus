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

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>

#include <optional>

namespace cxx {

class TranslationUnit;
class Control;

class ASTInterpreter {
 public:
  explicit ASTInterpreter(TranslationUnit* unit);
  ~ASTInterpreter();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto control() const -> Control*;

  [[nodiscard]] auto evaluate(ExpressionAST* ast) -> std::optional<ConstValue>;

  [[nodiscard]] auto toBool(const ConstValue& value) -> std::optional<bool>;

  [[nodiscard]] auto toInt(const ConstValue& value)
      -> std::optional<std::intmax_t>;

  [[nodiscard]] auto toUInt(const ConstValue& value)
      -> std::optional<std::uintmax_t>;

  [[nodiscard]] auto toFloat(const ConstValue& value) -> std::optional<float>;

  [[nodiscard]] auto toDouble(const ConstValue& value) -> std::optional<double>;

  [[nodiscard]] auto toLongDouble(const ConstValue& value)
      -> std::optional<long double>;

 private:
  using ExpressionResult = std::optional<ConstValue>;

  // base nodes
  struct UnitResult {};
  struct DeclarationResult {};
  struct StatementResult {};
  struct TemplateParameterResult {};
  struct SpecifierResult {};
  struct PtrOperatorResult {};
  struct CoreDeclaratorResult {};
  struct DeclaratorChunkResult {};
  struct UnqualifiedIdResult {};
  struct NestedNameSpecifierResult {};
  struct FunctionBodyResult {};
  struct TemplateArgumentResult {};
  struct ExceptionSpecifierResult {};
  struct RequirementResult {};
  struct NewInitializerResult {};
  struct MemInitializerResult {};
  struct LambdaCaptureResult {};
  struct ExceptionDeclarationResult {};
  struct AttributeSpecifierResult {};
  struct AttributeTokenResult {};
  // misc nodes
  struct GlobalModuleFragmentResult {};
  struct PrivateModuleFragmentResult {};
  struct ModuleDeclarationResult {};
  struct ModuleNameResult {};
  struct ModuleQualifierResult {};
  struct ModulePartitionResult {};
  struct ImportNameResult {};
  struct InitDeclaratorResult {};
  struct DeclaratorResult {};
  struct UsingDeclaratorResult {};
  struct EnumeratorResult {};
  struct TypeIdResult {};
  struct HandlerResult {};
  struct BaseSpecifierResult {};
  struct RequiresClauseResult {};
  struct ParameterDeclarationClauseResult {};
  struct TrailingReturnTypeResult {};
  struct LambdaSpecifierResult {};
  struct TypeConstraintResult {};
  struct AttributeArgumentClauseResult {};
  struct AttributeResult {};
  struct AttributeUsingPrefixResult {};
  struct NewPlacementResult {};
  struct NestedNamespaceSpecifierResult {};
  // visitors
  struct UnitVisitor;
  struct DeclarationVisitor;
  struct StatementVisitor;
  struct ExpressionVisitor;
  struct TemplateParameterVisitor;
  struct SpecifierVisitor;
  struct PtrOperatorVisitor;
  struct CoreDeclaratorVisitor;
  struct DeclaratorChunkVisitor;
  struct UnqualifiedIdVisitor;
  struct NestedNameSpecifierVisitor;
  struct FunctionBodyVisitor;
  struct TemplateArgumentVisitor;
  struct ExceptionSpecifierVisitor;
  struct RequirementVisitor;
  struct NewInitializerVisitor;
  struct MemInitializerVisitor;
  struct LambdaCaptureVisitor;
  struct ExceptionDeclarationVisitor;
  struct AttributeSpecifierVisitor;
  struct AttributeTokenVisitor;

  // ops
  struct ToBool;

  // run on the base nodes
  [[nodiscard]] auto unit(UnitAST* ast) -> UnitResult;
  [[nodiscard]] auto declaration(DeclarationAST* ast) -> DeclarationResult;
  [[nodiscard]] auto statement(StatementAST* ast) -> StatementResult;
  [[nodiscard]] auto expression(ExpressionAST* ast) -> ExpressionResult;
  [[nodiscard]] auto templateParameter(TemplateParameterAST* ast)
      -> TemplateParameterResult;
  [[nodiscard]] auto specifier(SpecifierAST* ast) -> SpecifierResult;
  [[nodiscard]] auto ptrOperator(PtrOperatorAST* ast) -> PtrOperatorResult;
  [[nodiscard]] auto coreDeclarator(CoreDeclaratorAST* ast)
      -> CoreDeclaratorResult;
  [[nodiscard]] auto declaratorChunk(DeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;
  [[nodiscard]] auto unqualifiedId(UnqualifiedIdAST* ast)
      -> UnqualifiedIdResult;
  [[nodiscard]] auto nestedNameSpecifier(NestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;
  [[nodiscard]] auto functionBody(FunctionBodyAST* ast) -> FunctionBodyResult;
  [[nodiscard]] auto templateArgument(TemplateArgumentAST* ast)
      -> TemplateArgumentResult;
  [[nodiscard]] auto exceptionSpecifier(ExceptionSpecifierAST* ast)
      -> ExceptionSpecifierResult;
  [[nodiscard]] auto requirement(RequirementAST* ast) -> RequirementResult;
  [[nodiscard]] auto newInitializer(NewInitializerAST* ast)
      -> NewInitializerResult;
  [[nodiscard]] auto memInitializer(MemInitializerAST* ast)
      -> MemInitializerResult;
  [[nodiscard]] auto lambdaCapture(LambdaCaptureAST* ast)
      -> LambdaCaptureResult;
  [[nodiscard]] auto exceptionDeclaration(ExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;
  [[nodiscard]] auto attributeSpecifier(AttributeSpecifierAST* ast)
      -> AttributeSpecifierResult;
  [[nodiscard]] auto attributeToken(AttributeTokenAST* ast)
      -> AttributeTokenResult;

  // run on the misc nodes
  [[nodiscard]] auto splicer(SplicerAST* ast) -> ExpressionResult;
  [[nodiscard]] auto globalModuleFragment(GlobalModuleFragmentAST* ast)
      -> GlobalModuleFragmentResult;
  [[nodiscard]] auto privateModuleFragment(PrivateModuleFragmentAST* ast)
      -> PrivateModuleFragmentResult;
  [[nodiscard]] auto moduleDeclaration(ModuleDeclarationAST* ast)
      -> ModuleDeclarationResult;
  [[nodiscard]] auto moduleName(ModuleNameAST* ast) -> ModuleNameResult;
  [[nodiscard]] auto moduleQualifier(ModuleQualifierAST* ast)
      -> ModuleQualifierResult;
  [[nodiscard]] auto modulePartition(ModulePartitionAST* ast)
      -> ModulePartitionResult;
  [[nodiscard]] auto importName(ImportNameAST* ast) -> ImportNameResult;
  [[nodiscard]] auto initDeclarator(InitDeclaratorAST* ast)
      -> InitDeclaratorResult;
  [[nodiscard]] auto declarator(DeclaratorAST* ast) -> DeclaratorResult;
  [[nodiscard]] auto usingDeclarator(UsingDeclaratorAST* ast)
      -> UsingDeclaratorResult;
  [[nodiscard]] auto enumerator(EnumeratorAST* ast) -> EnumeratorResult;
  [[nodiscard]] auto typeId(TypeIdAST* ast) -> TypeIdResult;
  [[nodiscard]] auto handler(HandlerAST* ast) -> HandlerResult;
  [[nodiscard]] auto baseSpecifier(BaseSpecifierAST* ast)
      -> BaseSpecifierResult;
  [[nodiscard]] auto requiresClause(RequiresClauseAST* ast)
      -> RequiresClauseResult;
  [[nodiscard]] auto parameterDeclarationClause(
      ParameterDeclarationClauseAST* ast) -> ParameterDeclarationClauseResult;
  [[nodiscard]] auto trailingReturnType(TrailingReturnTypeAST* ast)
      -> TrailingReturnTypeResult;
  [[nodiscard]] auto lambdaSpecifier(LambdaSpecifierAST* ast)
      -> LambdaSpecifierResult;
  [[nodiscard]] auto typeConstraint(TypeConstraintAST* ast)
      -> TypeConstraintResult;
  [[nodiscard]] auto attributeArgumentClause(AttributeArgumentClauseAST* ast)
      -> AttributeArgumentClauseResult;
  [[nodiscard]] auto attribute(AttributeAST* ast) -> AttributeResult;
  [[nodiscard]] auto attributeUsingPrefix(AttributeUsingPrefixAST* ast)
      -> AttributeUsingPrefixResult;
  [[nodiscard]] auto newPlacement(NewPlacementAST* ast) -> NewPlacementResult;
  [[nodiscard]] auto nestedNamespaceSpecifier(NestedNamespaceSpecifierAST* ast)
      -> NestedNamespaceSpecifierResult;
  [[nodiscard]] auto asmOperand(AsmOperandAST* ast) -> DeclarationResult;
  [[nodiscard]] auto asmQualifier(AsmQualifierAST* ast) -> DeclarationResult;
  [[nodiscard]] auto asmClobber(AsmClobberAST* ast) -> DeclarationResult;
  [[nodiscard]] auto asmGotoLabel(AsmGotoLabelAST* ast) -> DeclarationResult;

 private:
  TranslationUnit* unit_ = nullptr;
};

}  // namespace cxx
