// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ir/condition_codegen.h>
#include <cxx/ir/expression_codegen.h>
#include <cxx/ir/ir_builder.h>
#include <cxx/ir/statement_codegen.h>
#include <cxx/recursive_ast_visitor.h>

#include <memory>
#include <unordered_map>

namespace ir {
class TranslationUnit;
}

namespace cxx::ir {

class Codegen final : public ir::IRBuilder, RecursiveASTVisitor {
 public:
  Codegen(const Codegen&) = delete;
  auto operator=(const Codegen&) -> Codegen& = delete;

  Codegen();
  ~Codegen() override;

  auto operator()(TranslationUnit* unit) -> std::unique_ptr<ir::Module>;

  auto expression(ExpressionAST* ast) -> ir::Expr*;
  auto reduce(ExpressionAST* ast) -> ir::Expr*;

  void condition(ExpressionAST* ast, ir::Block* iftrue, ir::Block* iffalse);

  void statement(StatementAST* ast);
  void statement(ExpressionAST* ast);

  auto irFactory() -> ir::IRFactory*;

  auto createBlock() -> ir::Block*;

  void place(ir::Block* block);

  [[nodiscard]] auto unit() const -> TranslationUnit* { return unit_; }
  [[nodiscard]] auto function() const -> ir::Function* { return function_; }
  [[nodiscard]] auto entryBlock() const -> ir::Block* { return entryBlock_; }
  [[nodiscard]] auto exitBlock() const -> ir::Block* { return exitBlock_; }
  [[nodiscard]] auto breakBlock() const -> ir::Block* { return breakBlock_; }
  [[nodiscard]] auto continueBlock() const -> ir::Block* {
    return continueBlock_;
  }
  [[nodiscard]] auto result() const -> ir::Local* { return result_; }
  [[nodiscard]] auto currentSwitch() const -> ir::Switch* { return switch_; }

  auto changeBreakBlock(ir::Block* breakBlock) -> ir::Block* {
    std::swap(breakBlock_, breakBlock);
    return breakBlock;
  }

  auto changeContinueBlock(ir::Block* continueBlock) -> ir::Block* {
    std::swap(continueBlock_, continueBlock);
    return continueBlock;
  }

  auto changeCurrentSwitch(ir::Switch* stmt) -> ir::Switch* {
    std::swap(switch_, stmt);
    return stmt;
  }

  auto getLocal(Symbol* symbol) -> ir::Local*;

  auto findOrCreateTargetBlock(const Identifier* id) -> ir::Block*;

 private:
  using RecursiveASTVisitor::visit;

  void visit(FunctionDefinitionAST* ast) override;
  void visit(CompoundStatementFunctionBodyAST* ast) override;

 private:
  ExpressionCodegen expression_{this};
  ConditionCodegen condition_{this};
  StatementCodegen statement_{this};

  TranslationUnit* unit_ = nullptr;
  std::unique_ptr<ir::Module> module_;
  ir::Function* function_ = nullptr;
  ir::Block* entryBlock_ = nullptr;
  ir::Block* exitBlock_ = nullptr;
  ir::Block* breakBlock_ = nullptr;
  ir::Block* continueBlock_ = nullptr;
  ir::Local* result_ = nullptr;
  ir::Switch* switch_ = nullptr;
  std::unordered_map<Symbol*, ir::Local*> locals_;
  std::unordered_map<const Identifier*, ir::Block*> labels_;
};

}  // namespace cxx::ir
