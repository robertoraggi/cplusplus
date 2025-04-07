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
#include <cxx/source_location.h>

#include <format>
#include <iterator>
#include <ostream>

namespace cxx {

class TranslationUnit;
class Control;

class ASTPrettyPrinter {
 public:
  explicit ASTPrettyPrinter(TranslationUnit* unit, std::ostream& out);
  ~ASTPrettyPrinter();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto control() const -> Control*;

  // run on the base nodes
  void operator()(UnitAST* ast);
  void operator()(DeclarationAST* ast);
  void operator()(StatementAST* ast);
  void operator()(ExpressionAST* ast);
  void operator()(TemplateParameterAST* ast);
  void operator()(SpecifierAST* ast);
  void operator()(PtrOperatorAST* ast);
  void operator()(CoreDeclaratorAST* ast);
  void operator()(DeclaratorChunkAST* ast);
  void operator()(UnqualifiedIdAST* ast);
  void operator()(NestedNameSpecifierAST* ast);
  void operator()(FunctionBodyAST* ast);
  void operator()(TemplateArgumentAST* ast);
  void operator()(ExceptionSpecifierAST* ast);
  void operator()(RequirementAST* ast);
  void operator()(NewInitializerAST* ast);
  void operator()(MemInitializerAST* ast);
  void operator()(LambdaCaptureAST* ast);
  void operator()(ExceptionDeclarationAST* ast);
  void operator()(AttributeSpecifierAST* ast);
  void operator()(AttributeTokenAST* ast);

  // run on the misc nodes
  void operator()(SplicerAST* ast);
  void operator()(GlobalModuleFragmentAST* ast);
  void operator()(PrivateModuleFragmentAST* ast);
  void operator()(ModuleDeclarationAST* ast);
  void operator()(ModuleNameAST* ast);
  void operator()(ModuleQualifierAST* ast);
  void operator()(ModulePartitionAST* ast);
  void operator()(ImportNameAST* ast);
  void operator()(InitDeclaratorAST* ast);
  void operator()(DeclaratorAST* ast);
  void operator()(UsingDeclaratorAST* ast);
  void operator()(EnumeratorAST* ast);
  void operator()(TypeIdAST* ast);
  void operator()(HandlerAST* ast);
  void operator()(BaseSpecifierAST* ast);
  void operator()(RequiresClauseAST* ast);
  void operator()(ParameterDeclarationClauseAST* ast);
  void operator()(TrailingReturnTypeAST* ast);
  void operator()(LambdaSpecifierAST* ast);
  void operator()(TypeConstraintAST* ast);
  void operator()(AttributeArgumentClauseAST* ast);
  void operator()(AttributeAST* ast);
  void operator()(AttributeUsingPrefixAST* ast);
  void operator()(NewPlacementAST* ast);
  void operator()(NestedNamespaceSpecifierAST* ast);

 private:
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

  template <typename... Args>
  void write(std::format_string<Args...> fmt, Args&&... args) {
    if (newline_) {
      std::format_to(output_, "\n");
      if (depth_ > 0) std::format_to(output_, "{:{}}", "", depth_ * 2);
    } else if (space_) {
      std::format_to(output_, " ");
    }
    newline_ = false;
    space_ = false;
    keepSpace_ = false;
    std::format_to(output_, fmt, std::forward<Args>(args)...);
    space();
  }

  void writeToken(SourceLocation loc);

  void space();
  void nospace();
  void keepSpace();
  void newline();
  void nonewline();
  void indent();
  void unindent();

 private:
  TranslationUnit* unit_ = nullptr;
  std::ostream_iterator<char> output_;
  int depth_ = 0;
  bool space_ = false;
  bool keepSpace_ = false;
  bool newline_ = false;
};

}  // namespace cxx
