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

 private:
  // base nodes
  struct UnitResult;
  struct DeclarationResult;
  struct StatementResult;
  struct ExpressionResult;
  struct TemplateParameterResult;
  struct SpecifierResult;
  struct PtrOperatorResult;
  struct CoreDeclaratorResult;
  struct DeclaratorChunkResult;
  struct UnqualifiedIdResult;
  struct NestedNameSpecifierResult;
  struct FunctionBodyResult;
  struct TemplateArgumentResult;
  struct ExceptionSpecifierResult;
  struct RequirementResult;
  struct NewInitializerResult;
  struct MemInitializerResult;
  struct LambdaCaptureResult;
  struct ExceptionDeclarationResult;
  struct AttributeSpecifierResult;
  struct AttributeTokenResult;
  // misc nodes
  struct SplicerResult;
  struct GlobalModuleFragmentResult;
  struct PrivateModuleFragmentResult;
  struct ModuleDeclarationResult;
  struct ModuleNameResult;
  struct ModuleQualifierResult;
  struct ModulePartitionResult;
  struct ImportNameResult;
  struct InitDeclaratorResult;
  struct DeclaratorResult;
  struct UsingDeclaratorResult;
  struct EnumeratorResult;
  struct TypeIdResult;
  struct HandlerResult;
  struct BaseSpecifierResult;
  struct RequiresClauseResult;
  struct ParameterDeclarationClauseResult;
  struct TrailingReturnTypeResult;
  struct LambdaSpecifierResult;
  struct TypeConstraintResult;
  struct AttributeArgumentClauseResult;
  struct AttributeResult;
  struct AttributeUsingPrefixResult;
  struct NewPlacementResult;
  struct NestedNamespaceSpecifierResult;
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

  // run on the base nodes
  [[nodiscard]] auto operator()(UnitAST* ast) -> UnitResult;
  [[nodiscard]] auto operator()(DeclarationAST* ast) -> DeclarationResult;
  [[nodiscard]] auto operator()(StatementAST* ast) -> StatementResult;
  [[nodiscard]] auto operator()(ExpressionAST* ast) -> ExpressionResult;
  [[nodiscard]] auto operator()(TemplateParameterAST* ast)
      -> TemplateParameterResult;
  [[nodiscard]] auto operator()(SpecifierAST* ast) -> SpecifierResult;
  [[nodiscard]] auto operator()(PtrOperatorAST* ast) -> PtrOperatorResult;
  [[nodiscard]] auto operator()(CoreDeclaratorAST* ast) -> CoreDeclaratorResult;
  [[nodiscard]] auto operator()(DeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;
  [[nodiscard]] auto operator()(UnqualifiedIdAST* ast) -> UnqualifiedIdResult;
  [[nodiscard]] auto operator()(NestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;
  [[nodiscard]] auto operator()(FunctionBodyAST* ast) -> FunctionBodyResult;
  [[nodiscard]] auto operator()(TemplateArgumentAST* ast)
      -> TemplateArgumentResult;
  [[nodiscard]] auto operator()(ExceptionSpecifierAST* ast)
      -> ExceptionSpecifierResult;
  [[nodiscard]] auto operator()(RequirementAST* ast) -> RequirementResult;
  [[nodiscard]] auto operator()(NewInitializerAST* ast) -> NewInitializerResult;
  [[nodiscard]] auto operator()(MemInitializerAST* ast) -> MemInitializerResult;
  [[nodiscard]] auto operator()(LambdaCaptureAST* ast) -> LambdaCaptureResult;
  [[nodiscard]] auto operator()(ExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;
  [[nodiscard]] auto operator()(AttributeSpecifierAST* ast)
      -> AttributeSpecifierResult;
  [[nodiscard]] auto operator()(AttributeTokenAST* ast) -> AttributeTokenResult;

  // run on the misc nodes
  [[nodiscard]] auto operator()(SplicerAST* ast) -> SplicerResult;
  [[nodiscard]] auto operator()(GlobalModuleFragmentAST* ast)
      -> GlobalModuleFragmentResult;
  [[nodiscard]] auto operator()(PrivateModuleFragmentAST* ast)
      -> PrivateModuleFragmentResult;
  [[nodiscard]] auto operator()(ModuleDeclarationAST* ast)
      -> ModuleDeclarationResult;
  [[nodiscard]] auto operator()(ModuleNameAST* ast) -> ModuleNameResult;
  [[nodiscard]] auto operator()(ModuleQualifierAST* ast)
      -> ModuleQualifierResult;
  [[nodiscard]] auto operator()(ModulePartitionAST* ast)
      -> ModulePartitionResult;
  [[nodiscard]] auto operator()(ImportNameAST* ast) -> ImportNameResult;
  [[nodiscard]] auto operator()(InitDeclaratorAST* ast) -> InitDeclaratorResult;
  [[nodiscard]] auto operator()(DeclaratorAST* ast) -> DeclaratorResult;
  [[nodiscard]] auto operator()(UsingDeclaratorAST* ast)
      -> UsingDeclaratorResult;
  [[nodiscard]] auto operator()(EnumeratorAST* ast) -> EnumeratorResult;
  [[nodiscard]] auto operator()(TypeIdAST* ast) -> TypeIdResult;
  [[nodiscard]] auto operator()(HandlerAST* ast) -> HandlerResult;
  [[nodiscard]] auto operator()(BaseSpecifierAST* ast) -> BaseSpecifierResult;
  [[nodiscard]] auto operator()(RequiresClauseAST* ast) -> RequiresClauseResult;
  [[nodiscard]] auto operator()(ParameterDeclarationClauseAST* ast)
      -> ParameterDeclarationClauseResult;
  [[nodiscard]] auto operator()(TrailingReturnTypeAST* ast)
      -> TrailingReturnTypeResult;
  [[nodiscard]] auto operator()(LambdaSpecifierAST* ast)
      -> LambdaSpecifierResult;
  [[nodiscard]] auto operator()(TypeConstraintAST* ast) -> TypeConstraintResult;
  [[nodiscard]] auto operator()(AttributeArgumentClauseAST* ast)
      -> AttributeArgumentClauseResult;
  [[nodiscard]] auto operator()(AttributeAST* ast) -> AttributeResult;
  [[nodiscard]] auto operator()(AttributeUsingPrefixAST* ast)
      -> AttributeUsingPrefixResult;
  [[nodiscard]] auto operator()(NewPlacementAST* ast) -> NewPlacementResult;
  [[nodiscard]] auto operator()(NestedNamespaceSpecifierAST* ast)
      -> NestedNamespaceSpecifierResult;

 private:
  TranslationUnit* unit_ = nullptr;
};

}  // namespace cxx
