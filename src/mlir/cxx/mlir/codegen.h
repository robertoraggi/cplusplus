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
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/source_location.h>
#include <cxx/types_fwd.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir::func {
class FuncOp;
}

namespace cxx {

class TranslationUnit;
class Control;

class Codegen {
 public:
  explicit Codegen(mlir::MLIRContext& context, TranslationUnit* unit);
  ~Codegen();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto control() const -> Control*;

  struct UnitResult {
    mlir::ModuleOp module;
  };

  struct DeclarationResult {};

  struct ExpressionResult {
    mlir::Value value;
  };

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

  struct SplicerResult {};
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

  // run on the base nodes
  [[nodiscard]] auto operator()(UnitAST* ast) -> UnitResult;
  [[nodiscard]] auto operator()(DeclarationAST* ast) -> DeclarationResult;

  void statement(StatementAST* ast);
  [[nodiscard]] auto expression(ExpressionAST* ast) -> ExpressionResult;

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
  auto operator()(SplicerAST* ast) -> SplicerResult;
  auto operator()(GlobalModuleFragmentAST* ast) -> GlobalModuleFragmentResult;
  auto operator()(PrivateModuleFragmentAST* ast) -> PrivateModuleFragmentResult;
  auto operator()(ModuleDeclarationAST* ast) -> ModuleDeclarationResult;
  auto operator()(ModuleNameAST* ast) -> ModuleNameResult;
  auto operator()(ModuleQualifierAST* ast) -> ModuleQualifierResult;
  auto operator()(ModulePartitionAST* ast) -> ModulePartitionResult;
  auto operator()(ImportNameAST* ast) -> ImportNameResult;
  auto operator()(InitDeclaratorAST* ast) -> InitDeclaratorResult;
  auto operator()(DeclaratorAST* ast) -> DeclaratorResult;
  auto operator()(UsingDeclaratorAST* ast) -> UsingDeclaratorResult;
  auto operator()(EnumeratorAST* ast) -> EnumeratorResult;
  auto operator()(TypeIdAST* ast) -> TypeIdResult;
  auto operator()(HandlerAST* ast) -> HandlerResult;
  auto operator()(BaseSpecifierAST* ast) -> BaseSpecifierResult;
  auto operator()(RequiresClauseAST* ast) -> RequiresClauseResult;
  auto operator()(ParameterDeclarationClauseAST* ast)
      -> ParameterDeclarationClauseResult;
  auto operator()(TrailingReturnTypeAST* ast) -> TrailingReturnTypeResult;
  auto operator()(LambdaSpecifierAST* ast) -> LambdaSpecifierResult;
  auto operator()(TypeConstraintAST* ast) -> TypeConstraintResult;
  auto operator()(AttributeArgumentClauseAST* ast)
      -> AttributeArgumentClauseResult;
  auto operator()(AttributeAST* ast) -> AttributeResult;
  auto operator()(AttributeUsingPrefixAST* ast) -> AttributeUsingPrefixResult;
  auto operator()(NewPlacementAST* ast) -> NewPlacementResult;
  auto operator()(NestedNamespaceSpecifierAST* ast)
      -> NestedNamespaceSpecifierResult;

 private:
  [[nodiscard]] auto getLocation(SourceLocation loc) -> mlir::Location;

  [[nodiscard]] auto emitTodoStmt(SourceLocation loc, std::string_view message)
      -> mlir::cxx::TodoStmtOp;

  [[nodiscard]] auto emitTodoExpr(SourceLocation loc, std::string_view message)
      -> mlir::cxx::TodoExprOp;

  [[nodiscard]] auto convertType(const Type* type) -> mlir::Type;

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

  struct ConvertType;

  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  mlir::cxx::FuncOp function_;
  TranslationUnit* unit_ = nullptr;
  int count_ = 0;
};

}  // namespace cxx
