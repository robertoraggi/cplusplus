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

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/ir/codegen.h>
#include <cxx/ir/ir.h>
#include <cxx/ir/ir_factory.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_environment.h>
#include <cxx/types.h>

namespace cxx::ir {

Codegen::Codegen() = default;

Codegen::~Codegen() = default;

auto Codegen::operator()(TranslationUnit* unit) -> std::unique_ptr<ir::Module> {
  auto module = std::make_unique<ir::Module>();
  setModule(module.get());
  std::swap(module_, module);
  std::swap(unit_, unit);
  accept(unit_->ast());
  std::swap(unit_, unit);
  std::swap(module_, module);
  return module;
}

auto Codegen::expression(ExpressionAST* ast) -> ir::Expr* {
  return expression_.gen(ast);
}

auto Codegen::reduce(ExpressionAST* ast) -> ir::Expr* {
  return expression_.reduce(ast);
}

void Codegen::condition(ExpressionAST* ast, ir::Block* iftrue,
                        ir::Block* iffalse) {
  condition_.gen(ast, iftrue, iffalse);
}

void Codegen::statement(StatementAST* ast) { statement_.gen(ast); }

void Codegen::statement(ExpressionAST* ast) { statement_.gen(ast); }

auto Codegen::irFactory() -> ir::IRFactory* { return module_->irFactory(); }

auto Codegen::createBlock() -> ir::Block* {
  return irFactory()->createBlock(function_);
}

void Codegen::place(ir::Block* block) {
  if (!blockHasTerminator()) emitJump(block);
  function_->addBlock(block);
  setInsertionPoint(block);
}

auto Codegen::getLocal(Symbol* symbol) -> ir::Local* {
  auto it = locals_.find(symbol);
  if (it != locals_.end()) return it->second;
  auto local = function_->addLocal(symbol->type());
  locals_.emplace(symbol, local);
  return local;
}

auto Codegen::findOrCreateTargetBlock(const Identifier* id) -> ir::Block* {
  if (auto it = labels_.find(id); it != labels_.end()) return it->second;
  return labels_.emplace(id, createBlock()).first->second;
}

void Codegen::visit(FunctionDefinitionAST* ast) {
  if (!ast->symbol) return;
  ir::Function* function = irFactory()->createFunction(ast->symbol);
  std::unordered_map<Symbol*, ir::Local*> locals;
  std::unordered_map<const Identifier*, ir::Block*> labels;

  module_->addFunction(function);

  std::swap(function_, function);
  std::swap(locals_, locals);
  std::swap(labels_, labels);

  ir::Block* entryBlock = createBlock();
  ir::Block* exitBlock = createBlock();

  auto functionType = Type::cast<FunctionType>(ast->symbol->type());

  ir::Local* result = nullptr;

  if (!Type::is<VoidType>(functionType->returnType())) {
    result = function_->addLocal(functionType->returnType());
  }

  std::swap(result_, result);
  std::swap(entryBlock_, entryBlock);
  std::swap(exitBlock_, exitBlock);

  place(entryBlock_);

  acceptFunctionBody(ast->functionBody);

  place(exitBlock_);

  if (result_) {
    emitRet(createTemp(result_));
  } else {
    emitRetVoid();
  }

  std::swap(entryBlock_, entryBlock);
  std::swap(exitBlock_, exitBlock);
  std::swap(function_, function);
  std::swap(result_, result);
  std::swap(locals_, locals);
  std::swap(labels_, labels);
}

void Codegen::visit(CompoundStatementFunctionBodyAST* ast) {
  statement(ast->statement);
}

}  // namespace cxx::ir
