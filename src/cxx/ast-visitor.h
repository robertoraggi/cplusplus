// Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "cxx-fwd.h"

namespace cxx {

class ASTVisitor {
  TranslationUnit* unit;

 public:
  ASTVisitor(TranslationUnit* unit) : unit(unit) {}
  virtual ~ASTVisitor() = default;

  TranslationUnit* translationUnit() const { return unit; }
  Control* control() const;

#define VISIT_AST(x) virtual void visit(x##AST*) = 0;
  FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST
};

class RecursiveASTVisitor : public ASTVisitor {
 public:
  RecursiveASTVisitor(TranslationUnit* unit) : ASTVisitor(unit) {}
  ~RecursiveASTVisitor() override;

  virtual bool preVisit(AST*) { return true; }
  virtual void postVisit(AST*) {}

  void accept(AST*);

#define VISIT_AST(x) void visit(x##AST*) override;
  FOR_EACH_AST(VISIT_AST)
#undef VISIT_AST

 private:
  void accept0(AST*);
};

class DumpAST final : protected RecursiveASTVisitor {
 public:
  DumpAST(TranslationUnit* unit);

  void operator()(AST* ast);

 protected:
  bool preVisit(AST*) override;
  void postVisit(AST*) override;

 private:
  int depth{-1};
};

}  // namespace cxx
