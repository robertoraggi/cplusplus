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

class Codegen {
public:
  Codegen(TranslationUnit* unit);
  ~Codegen();

  TranslationUnit* translationUnit() const { return unit; }
  Control* control() const;

  void operator()(TranslationUnitAST* ast);

private:
  enum Format { ex, cx, nx };
  struct Result {
    Format format{nx};
    Format requested{ex};
    IR::BasicBlock* iftrue{0};
    IR::BasicBlock* iffalse{0};
    const IR::Expr* code{0};

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

  void accept(AST* ast);

#define VISIT_AST(x) void visit(x##AST* ast);
FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST

private:
  TranslationUnit* unit;
  IR::Function* function{0};
  IR::BasicBlock* block{0};
  IR::BasicBlock* exitBlock{0};
  const IR::Expr* exitValue{0};
  Result result{ex};
  int tempCount{0};
};

#endif // CODEGEN_H
