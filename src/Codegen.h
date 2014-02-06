// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CODEGEN_H
#define CODEGEN_H

#include "Globals.h"
#include <map>
#include <vector>

class Codegen {
public:
  Codegen(TranslationUnit* unit);
  ~Codegen();

  Codegen(const Codegen& other) = delete;
  Codegen& operator=(const Codegen& other) = delete;

  TranslationUnit* translationUnit() const { return unit; }
  Control* control() const;

  void operator()(FunctionDefinitionAST* ast);

private:
  enum Format { ex, cx, nx };
  struct Result {
    Format format{nx};
    Format requested{ex};
    IR::BasicBlock* iftrue{nullptr};
    IR::BasicBlock* iffalse{nullptr};
    const IR::Expr* code{nullptr};

    explicit Result(Format requested)
      : requested(requested) {}

    explicit Result(const IR::Expr* code)
      : format(ex), requested(ex), code(code) {}

    explicit Result(IR::BasicBlock* iftrue, IR::BasicBlock* iffalse)
      : requested(cx), iftrue(iftrue), iffalse(iffalse) {}

    const IR::Expr* operator*() const { return code; }
    const IR::Expr* operator->() const { return code; }

    bool is(Format f) const { return format == f; }
    bool isNot(Format f) const { return format != f; }

    bool accept(Format f) {
      if (requested == f) {
        format = f;
        return true;
      }
      return false;
    }
  };

  Result reduce(const Result& expr);
  Result expression(ExpressionAST* ast);
  void condition(ExpressionAST* ast,
                 IR::BasicBlock* iftrue,
                 IR::BasicBlock* iffalse);
  void statement(ExpressionAST* ast);
  void statement(StatementAST* ast);
  void declaration(DeclarationAST* ast);

  void place(IR::BasicBlock* block);

  void accept(AST* ast);

  const IR::Temp* newTemp();

#define VISIT_AST(x) void visit(x##AST* ast);
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST

  int indexOfCase(CaseStatementAST* ast) const {
    for (size_t i = 0; i < _cases.size(); ++i) {
      if (std::get<0>(_cases[i]) == ast)
        return i;
    }
    return -1;
  }

private:
  struct Loop {
    IR::BasicBlock* breakLabel;
    IR::BasicBlock* continueLabel;
    Loop(IR::BasicBlock* breakLabel, IR::BasicBlock* continueLabel)
      : breakLabel(breakLabel), continueLabel(continueLabel) {}
  };

  TranslationUnit* unit;
  IR::Module* _module{nullptr};
  IR::Function* _function{nullptr};
  IR::BasicBlock* _block{nullptr};
  IR::BasicBlock* _exitBlock{nullptr};
  const IR::Expr* _exitValue{nullptr};
  Result _result{nx};
  int _tempCount{0};
  std::map<const Name*, IR::BasicBlock*> _labels;
  std::vector<std::tuple<CaseStatementAST*, IR::BasicBlock*, IR::BasicBlock*>> _cases;
  IR::BasicBlock* _defaultCase{nullptr};
  Loop _loop{nullptr, nullptr};
};

#endif // CODEGEN_H
