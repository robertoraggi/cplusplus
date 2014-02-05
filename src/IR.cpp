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

#include "IR.h"
#include "Names.h"
#include "Token.h"
#include <iostream>
#include <cassert>

namespace IR {

Module::Module() {
}

Module::~Module() {
  for (auto&& fun: functions)
    delete fun;
}

Function* Module::newFunction(FunctionSymbol* symbol) {
  auto fun = new Function(this, symbol);
  functions.push_back(fun);
  return fun;
}

BasicBlock* Function::newBasicBlock() {
  auto block = new BasicBlock(this); // ### fix possible leaks
  return block;
}

Function::~Function() {
  for (auto block: *this)
    delete block;
}

void Function::placeBasicBlock(BasicBlock* basicBlock) {
  assert(basicBlock->index == -1);
  basicBlock->index = size();
  push_back(basicBlock);
}

void Exp::dump(std::ostream& out) const {
  expr()->dump(out);
  out << ';' << std::endl;
}

void Move::dump(std::ostream& out) const {
  target()->dump(out);
  out << ' ' << token_spell[int(op())] << ' ';
  source()->dump(out);
  out << ';' << std::endl;
}

void Ret::dump(std::ostream& out) const {
  out << "return ";
  expr()->dump(out);
  out << ';' << std::endl;
}

void Jump::dump(std::ostream& out) const {
  out << "goto .L" << target()->index << ';' << std::endl;
}

void CJump::dump(std::ostream& out) const {
  out << "if (";
  expr()->dump(out);
  out << ") goto .L" << iftrue()->index << "; else goto .L" << iffalse()->index << ';' << std::endl;
}

void Const::dump(std::ostream& out) const {
  out << value();
}

void Temp::dump(std::ostream& out) const {
  out << '$' << index();
}

void Sym::dump(std::ostream& out) const {
  out << name()->toString();
}

void Cast::dump(std::ostream& out) const {
  TypeToString typeToString;
  out << "reinterpret_cast<" << typeToString(type(), nullptr) << ">(";
  expr()->dump(out);
  out << ')';
}

void Call::dump(std::ostream& out) const {
  expr()->dump(out);
  out << '(';
  bool first = true;
  for (auto arg: args()) {
    if (first)
      first = false;
    else
      out << ", ";
    arg->dump(out);
  }
  out << ')';
}

void Member::dump(std::ostream& out) const {
  expr()->dump(out);
  out << token_spell[op()] << name()->toString();
}

void Subscript::dump(std::ostream& out) const {
  expr()->dump(out);
  out << '[';
  index()->dump(out);
  out << ']';
}

void Unop::dump(std::ostream& out) const {
  out << token_spell[op()];
  expr()->dump(out);
}

void Binop::dump(std::ostream& out) const {
  out << '(';
  left()->dump(out);
  out << ' ' << token_spell[op()] << ' ';
  right()->dump(out);
  out << ')';
}

void Function::dump(std::ostream& out) {
  auto it = begin();
  while (it != end()) {
    auto block = *it++;
    auto nextBlock = it != end() ? *it : nullptr;
    out << ".L" << block->index << ":";
    for (auto s: *block) {
      if (s->asTerminator())
        break;
      out << '\t';
      s->dump(out);
    }
    auto t = block->terminator();
    assert(t);
    auto j = t->asJump();
    auto cj = t->asCJump();
    if (j && j->target() == nextBlock) {
      // nothing to do
      out << std::endl;
    } else if (cj && cj->iffalse() == nextBlock) {
      out << "\tif (";
      cj->expr()->dump(out);
      out << ") goto .L" << cj->iftrue()->index << std::endl;
    } else if (cj && cj->iftrue() == nextBlock) {
      out << "\tiffalse (";
      cj->expr()->dump(out);
      out << ") goto .L" << cj->iffalse()->index << std::endl;
    } else {
      out << '\t';
      t->dump(out);
    }
  }
}

} // end of namespace IR

