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

#include <cxx/ast.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbol_instantiation.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <gtest/gtest.h>

#include <format>
#include <iostream>
#include <sstream>

#include "test_utils.h"

using namespace cxx;

template <typename Node>
auto subst(Source& source, Node* ast, std::vector<TemplateArgument> args) {
  auto control = source.control();
  TypeChecker typeChecker(&source.unit);
  ASTRewriter rewrite{&typeChecker, args};
  return ast_cast<Node>(rewrite(ast));
};

TEST(Rewriter, TypeAlias) {
  auto source = R"(
    template <typename T>
    using Ptr = const T*;

    template <typename T, typename U>
    using Func = void(T, U);
  )"_cxx;

  auto control = source.control();

  auto ptrTypeAlias =
      subst(source, source.getAs<TypeAliasSymbol>("Ptr")->declaration(),
            {control->getIntType()});

  ASSERT_EQ(to_string(ptrTypeAlias->typeId->type), "const int*");

  auto funcTypeAlias =
      subst(source, source.getAs<TypeAliasSymbol>("Func")->declaration(),
            {control->getIntType(), control->getFloatType()});

  ASSERT_EQ(to_string(funcTypeAlias->typeId->type), "void (int, float)");
}
