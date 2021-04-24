// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/codegen.h>
#include <cxx/control.h>
#include <cxx/ir.h>
#include <cxx/ir_factory.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_environment.h>
#include <cxx/types.h>

namespace cxx {

Codegen::Codegen() {}

Codegen::~Codegen() {}

std::unique_ptr<ir::Module> Codegen::operator()(TranslationUnit* unit) {
  auto module = std::make_unique<ir::Module>();
  setModule(module.get());
  std::swap(module_, module);
  std::swap(unit_, unit);
  accept(unit_->ast());
  std::swap(unit_, unit);
  std::swap(module_, module);
  return module;
}

ir::Expr* Codegen::expression(ExpressionAST* ast) {
  return expression_.gen(ast);
}

void Codegen::condition(ExpressionAST* ast, ir::Block* iftrue,
                        ir::Block* iffalse) {
  condition_.gen(ast, iftrue, iffalse);
}

void Codegen::statement(StatementAST* ast) { statement_.gen(ast); }

ir::IRFactory* Codegen::irFactory() { return module_->irFactory(); }

ir::Block* Codegen::createBlock() {
  return irFactory()->createBlock(function_);
}

void Codegen::place(ir::Block* block) {
  if (!blockHasTerminator()) emitJump(block);
  function_->addBlock(block);
  setInsertionPoint(block);
}

void Codegen::visit(FunctionDefinitionAST* ast) {
  ir::Function* function = irFactory()->createFunction(ast->symbol);

  module_->addFunction(function);

  std::swap(function_, function);

  ir::Block* entryBlock = createBlock();
  ir::Block* exitBlock = createBlock();

  auto types = unit_->control()->types();

  auto functionType = ast->symbol->type()->asFunctionType();

  ir::Local* result = nullptr;

  if (!functionType->returnType()->asVoidType()) {
    result = function_->addLocal(functionType->returnType());
  }

  std::swap(result_, result);
  std::swap(entryBlock_, entryBlock);
  std::swap(exitBlock_, exitBlock);

  place(entryBlock_);

  acceptFunctionBody(ast->functionBody);

  place(exitBlock_);

  if (result_) {
    emitRet(createLoad(result_));
  } else {
    emitRetVoid();
  }

  std::swap(entryBlock_, entryBlock);
  std::swap(exitBlock_, exitBlock);
  std::swap(function_, function);
  std::swap(result_, result);
}

void Codegen::visit(CompoundStatementFunctionBodyAST* ast) {
  statement(ast->statement);
}

}  // namespace cxx
