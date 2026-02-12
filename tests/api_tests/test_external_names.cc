// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/control.h>
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <gtest/gtest.h>

using namespace cxx;

TEST(ExternalNames, BuiltinTypes) {
  Control control;
  ExternalNameEncoder encoder;

  ASSERT_EQ("v", encoder.encode(control.getVoidType()));
  ASSERT_EQ("w", encoder.encode(control.getWideCharType()));
  ASSERT_EQ("b", encoder.encode(control.getBoolType()));
  ASSERT_EQ("c", encoder.encode(control.getCharType()));
  ASSERT_EQ("a", encoder.encode(control.getSignedCharType()));
  ASSERT_EQ("h", encoder.encode(control.getUnsignedCharType()));
  ASSERT_EQ("s", encoder.encode(control.getShortIntType()));
  ASSERT_EQ("t", encoder.encode(control.getUnsignedShortIntType()));
  ASSERT_EQ("i", encoder.encode(control.getIntType()));
  ASSERT_EQ("j", encoder.encode(control.getUnsignedIntType()));
  ASSERT_EQ("l", encoder.encode(control.getLongIntType()));
  ASSERT_EQ("m", encoder.encode(control.getUnsignedLongIntType()));
  ASSERT_EQ("x", encoder.encode(control.getLongLongIntType()));
  ASSERT_EQ("y", encoder.encode(control.getUnsignedLongLongIntType()));
  // ASSERT_EQ("n", encoder.encode(control.getInt128Type()));
  // ASSERT_EQ("o", encoder.encode(control.getUnsignedInt128Type()));
  ASSERT_EQ("f", encoder.encode(control.getFloatType()));
  ASSERT_EQ("d", encoder.encode(control.getDoubleType()));
  ASSERT_EQ("e", encoder.encode(control.getLongDoubleType()));
  // ASSERT_EQ("g", encoder.encode(control.getFloat128Type()));
  // ASSERT_EQ("z", encoder.encode(control.getEllipsisType()));
  ASSERT_EQ("Di", encoder.encode(control.getChar32Type()));
  ASSERT_EQ("Ds", encoder.encode(control.getChar16Type()));
  ASSERT_EQ("Du", encoder.encode(control.getChar8Type()));
  ASSERT_EQ("Da", encoder.encode(control.getAutoType()));
  ASSERT_EQ("Dc", encoder.encode(control.getDecltypeAutoType()));
  ASSERT_EQ("Dn", encoder.encode(control.getNullptrType()));
}

TEST(ExternalNames, CLinkageFunctionName) {
  Control control;
  ExternalNameEncoder encoder;

  auto global = control.newNamespaceSymbol(nullptr, {});
  auto function = control.newFunctionSymbol(global, {});

  function->setName(control.getIdentifier("printf"));
  function->setType(control.getFunctionType(
      control.getIntType(), {control.getPointerType(control.getCharType())},
      true));

  auto cxxName = encoder.encode(function);
  ASSERT_NE("printf", cxxName);

  function->setLanguageLinkage(LanguageKind::kC);
  ASSERT_EQ("printf", encoder.encode(function));
}