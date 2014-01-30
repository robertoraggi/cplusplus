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

#include <gtest/gtest.h>
#include <Control.h>
#include <Symbols.h>
#include <Types.h>
#include <IR.h>
#include <iostream>

FunctionSymbol* newMainFunction(Control* control) {
  QualType funTy{control->getFunctionType(QualType{control->getIntType()},
                                          std::vector<QualType>{},
                                          false, false)};
  auto symbol = control->newFunction();
  symbol->setName(control->getIdentifier("main"));
  symbol->setType(funTy);
  return symbol;
}

TEST(test_ir, BasicBlock) {
  Control control;
  auto symbol = newMainFunction(&control);

  IR::Module module;
  auto fun = module.newFunction(symbol);

  auto entryBlock = fun->newBasicBlock();
  auto exitBlock = fun->newBasicBlock();

  entryBlock->emitJump(exitBlock);
  exitBlock->emitRet(fun->getConst("0"));

  fun->placeBasicBlock(entryBlock);
  fun->placeBasicBlock(exitBlock);
}

TEST(test_ir, expressions) {
  Control control;
  auto symbol = newMainFunction(&control);

  IR::Module module;
  auto fun = module.newFunction(symbol);

  const char* zero = "0";
  const char* one = "1";

  const Name* id_0 = control.getIdentifier("id_0");
  const Name* id_1 = control.getIdentifier("id_1");

  EXPECT_EQ(fun->getConst(one), fun->getConst(one));
  EXPECT_NE(fun->getConst(one), fun->getConst(zero));

  EXPECT_EQ(fun->getTemp(0), fun->getTemp(0));
  EXPECT_NE(fun->getTemp(0), fun->getTemp(1));

  EXPECT_EQ(fun->getSym(id_0), fun->getSym(id_0));
  EXPECT_NE(fun->getSym(id_0), fun->getSym(id_1));

  const QualType intTy{control.getIntType()};
  auto cast1 = fun->getCast(intTy, fun->getConst(zero));
  auto cast2 = fun->getCast(intTy, fun->getConst(zero));
  auto cast3 = fun->getCast(intTy, fun->getConst(one));
  auto cast4 = fun->getCast(intTy, fun->getConst(one));

  EXPECT_EQ(cast1, cast2);
  EXPECT_EQ(cast3, cast4);
  EXPECT_NE(cast1, cast3);
  EXPECT_NE(cast2, cast4);
}
