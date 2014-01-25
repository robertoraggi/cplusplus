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
#include <TranslationUnit.h>
#include <AST.h>
#include <Symbols.h>
#include <sstream>
#include <iostream>

std::string toString(Symbol* symbol) {
  std::ostringstream code;
  symbol->dump(code, 0);
  return code.str();
}

TEST(test_symbols, main_function_prototype) { // ### generalize, e.g. use an external data set.
  Control control;
  TranslationUnit unit(&control);
  unit.setSource("int main(int argc, char* argv[]);");
  unit.parse([&unit] (TranslationUnitAST* ast) {
    ASSERT_TRUE(ast);
    ASSERT_TRUE(ast->globalScope);
    EXPECT_EQ(toString(ast->globalScope), "namespace {\n    int main(int argc, char *argv[]);\n}\n");
  });
}
