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
#include <cxx/binder.h>
#include <cxx/names_fwd.h>

#include <vector>

namespace cxx {

class TranslationUnit;
class TypeChecker;
class Control;
class Arena;

class ASTRewriter {
 public:
  explicit ASTRewriter(TypeChecker* typeChecker,
                       const std::vector<TemplateArgument>& templateArguments);
  ~ASTRewriter();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto control() const -> Control*;
  [[nodiscard]] auto arena() const -> Arena*;

  // run on the base nodes
  [[nodiscard]] auto operator()(UnitAST* ast) -> UnitAST*;
  [[nodiscard]] auto operator()(DeclarationAST* ast) -> DeclarationAST*;
  [[nodiscard]] auto operator()(StatementAST* ast) -> StatementAST*;
  [[nodiscard]] auto operator()(ExpressionAST* ast) -> ExpressionAST*;
  [[nodiscard]] auto operator()(TemplateParameterAST* ast)
      -> TemplateParameterAST*;
  [[nodiscard]] auto operator()(SpecifierAST* ast) -> SpecifierAST*;
  [[nodiscard]] auto operator()(PtrOperatorAST* ast) -> PtrOperatorAST*;
  [[nodiscard]] auto operator()(CoreDeclaratorAST* ast) -> CoreDeclaratorAST*;
  [[nodiscard]] auto operator()(DeclaratorChunkAST* ast) -> DeclaratorChunkAST*;
  [[nodiscard]] auto operator()(UnqualifiedIdAST* ast) -> UnqualifiedIdAST*;
  [[nodiscard]] auto operator()(NestedNameSpecifierAST* ast)
      -> NestedNameSpecifierAST*;
  [[nodiscard]] auto operator()(FunctionBodyAST* ast) -> FunctionBodyAST*;
  [[nodiscard]] auto operator()(TemplateArgumentAST* ast)
      -> TemplateArgumentAST*;
  [[nodiscard]] auto operator()(ExceptionSpecifierAST* ast)
      -> ExceptionSpecifierAST*;
  [[nodiscard]] auto operator()(RequirementAST* ast) -> RequirementAST*;
  [[nodiscard]] auto operator()(NewInitializerAST* ast) -> NewInitializerAST*;
  [[nodiscard]] auto operator()(MemInitializerAST* ast) -> MemInitializerAST*;
  [[nodiscard]] auto operator()(LambdaCaptureAST* ast) -> LambdaCaptureAST*;
  [[nodiscard]] auto operator()(ExceptionDeclarationAST* ast)
      -> ExceptionDeclarationAST*;
  [[nodiscard]] auto operator()(AttributeSpecifierAST* ast)
      -> AttributeSpecifierAST*;
  [[nodiscard]] auto operator()(AttributeTokenAST* ast) -> AttributeTokenAST*;

  // run on the misc nodes
  [[nodiscard]] auto operator()(SplicerAST* ast) -> SplicerAST*;
  [[nodiscard]] auto operator()(GlobalModuleFragmentAST* ast)
      -> GlobalModuleFragmentAST*;
  [[nodiscard]] auto operator()(PrivateModuleFragmentAST* ast)
      -> PrivateModuleFragmentAST*;
  [[nodiscard]] auto operator()(ModuleDeclarationAST* ast)
      -> ModuleDeclarationAST*;
  [[nodiscard]] auto operator()(ModuleNameAST* ast) -> ModuleNameAST*;
  [[nodiscard]] auto operator()(ModuleQualifierAST* ast) -> ModuleQualifierAST*;
  [[nodiscard]] auto operator()(ModulePartitionAST* ast) -> ModulePartitionAST*;
  [[nodiscard]] auto operator()(ImportNameAST* ast) -> ImportNameAST*;
  [[nodiscard]] auto operator()(InitDeclaratorAST* ast) -> InitDeclaratorAST*;
  [[nodiscard]] auto operator()(DeclaratorAST* ast) -> DeclaratorAST*;
  [[nodiscard]] auto operator()(UsingDeclaratorAST* ast) -> UsingDeclaratorAST*;
  [[nodiscard]] auto operator()(EnumeratorAST* ast) -> EnumeratorAST*;
  [[nodiscard]] auto operator()(TypeIdAST* ast) -> TypeIdAST*;
  [[nodiscard]] auto operator()(HandlerAST* ast) -> HandlerAST*;
  [[nodiscard]] auto operator()(BaseSpecifierAST* ast) -> BaseSpecifierAST*;
  [[nodiscard]] auto operator()(RequiresClauseAST* ast) -> RequiresClauseAST*;
  [[nodiscard]] auto operator()(ParameterDeclarationClauseAST* ast)
      -> ParameterDeclarationClauseAST*;
  [[nodiscard]] auto operator()(TrailingReturnTypeAST* ast)
      -> TrailingReturnTypeAST*;
  [[nodiscard]] auto operator()(LambdaSpecifierAST* ast) -> LambdaSpecifierAST*;
  [[nodiscard]] auto operator()(TypeConstraintAST* ast) -> TypeConstraintAST*;
  [[nodiscard]] auto operator()(AttributeArgumentClauseAST* ast)
      -> AttributeArgumentClauseAST*;
  [[nodiscard]] auto operator()(AttributeAST* ast) -> AttributeAST*;
  [[nodiscard]] auto operator()(AttributeUsingPrefixAST* ast)
      -> AttributeUsingPrefixAST*;
  [[nodiscard]] auto operator()(NewPlacementAST* ast) -> NewPlacementAST*;
  [[nodiscard]] auto operator()(NestedNamespaceSpecifierAST* ast)
      -> NestedNamespaceSpecifierAST*;

 private:
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

 private:
  TypeChecker* typeChecker_ = nullptr;
  const std::vector<TemplateArgument>& templateArguments_;
  TranslationUnit* unit_ = nullptr;
  Binder binder_;
};

}  // namespace cxx
