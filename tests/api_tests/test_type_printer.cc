// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/type_printer.h>
#include <cxx/types.h>
#include <gtest/gtest.h>

using namespace cxx;

TEST(TypePrinter, BasicTypes) {
  Control control;

  ASSERT_EQ(to_string(control.getNullptrType()), "decltype(nullptr)");
  ASSERT_EQ(to_string(control.getDecltypeAutoType()), "decltype(auto)");
  ASSERT_EQ(to_string(control.getAutoType()), "auto");
  ASSERT_EQ(to_string(control.getVoidType()), "void");
  ASSERT_EQ(to_string(control.getBoolType()), "bool");
  ASSERT_EQ(to_string(control.getCharType()), "char");
  ASSERT_EQ(to_string(control.getSignedCharType()), "signed char");
  ASSERT_EQ(to_string(control.getUnsignedCharType()), "unsigned char");
  ASSERT_EQ(to_string(control.getShortIntType()), "short");
  ASSERT_EQ(to_string(control.getUnsignedShortIntType()), "unsigned short");
  ASSERT_EQ(to_string(control.getIntType()), "int");
  ASSERT_EQ(to_string(control.getUnsignedIntType()), "unsigned int");
  ASSERT_EQ(to_string(control.getLongIntType()), "long");
  ASSERT_EQ(to_string(control.getUnsignedLongIntType()), "unsigned long");
  ASSERT_EQ(to_string(control.getLongLongIntType()), "long long");
  ASSERT_EQ(to_string(control.getUnsignedLongLongIntType()),
            "unsigned long long");
  ASSERT_EQ(to_string(control.getFloatType()), "float");
  ASSERT_EQ(to_string(control.getDoubleType()), "double");
  ASSERT_EQ(to_string(control.getLongDoubleType()), "long double");
}

TEST(TypePrinter, QualTypes) {
  Control control;

  ASSERT_EQ(to_string(control.getConstType(control.getUnsignedCharType())),
            "const unsigned char");

  ASSERT_EQ(to_string(control.getVolatileType(control.getUnsignedCharType())),
            "volatile unsigned char");

  ASSERT_EQ(
      to_string(control.getConstVolatileType(control.getUnsignedCharType())),
      "const volatile unsigned char");
}

TEST(TypePrinter, PointerTypes) {
  Control control;

  ASSERT_EQ(to_string(control.getPointerType(control.getUnsignedCharType())),
            "unsigned char*");

  // test pointer to pointer to unsigned int
  auto pointerToPointerToUnsignedInt = control.getPointerType(
      control.getPointerType(control.getUnsignedIntType()));

  ASSERT_EQ(to_string(pointerToPointerToUnsignedInt), "unsigned int**");

  // test pointer to const char
  auto pointerToConstChar =
      control.getPointerType(control.getConstType(control.getCharType()));

  ASSERT_EQ(to_string(pointerToConstChar), "const char*");

  // const pointer to char
  auto constPointerToChar =
      control.getConstType(control.getPointerType(control.getCharType()));

  ASSERT_EQ(to_string(constPointerToChar), "char* const");

  // pointer to array of 10 ints
  auto pointerToArrayOf10Ints = control.getPointerType(
      control.getBoundedArrayType(control.getIntType(), 10));

  ASSERT_EQ(to_string(pointerToArrayOf10Ints), "int (*)[10]");

  // pointer to function returning int

  auto pointerToFunctionReturningInt =
      control.getPointerType(control.getFunctionType(control.getIntType(), {}));

  ASSERT_EQ(to_string(pointerToFunctionReturningInt), "int (*)()");
}

TEST(TypePrinter, LValueReferences) {
  Control control;

  // lvalue reference to unsigned long
  auto referenceToUnsignedLong =
      control.getLvalueReferenceType(control.getUnsignedLongIntType());

  ASSERT_EQ(to_string(referenceToUnsignedLong), "unsigned long&");

  // lvalue reference to pointer to char
  auto referenceToPointerToChar = control.getLvalueReferenceType(
      control.getPointerType(control.getCharType()));

  ASSERT_EQ(to_string(referenceToPointerToChar), "char*&");

  // lvalue reference to a const pointer to char
  auto referenceToConstPointerToChar = control.getLvalueReferenceType(
      control.getConstType(control.getPointerType(control.getCharType())));

  ASSERT_EQ(to_string(referenceToConstPointerToChar), "char* const&");

  // reference to array of 10 ints
  auto referenceToArrayOf10Ints = control.getLvalueReferenceType(
      control.getBoundedArrayType(control.getIntType(), 10));

  ASSERT_EQ(to_string(referenceToArrayOf10Ints), "int (&)[10]");

  // reference to function returning pointer to const char
  auto referenceToFunctionReturningPointerToConstChar =
      control.getLvalueReferenceType(control.getFunctionType(
          control.getPointerType(control.getConstType(control.getCharType())),
          {}));

  ASSERT_EQ(to_string(referenceToFunctionReturningPointerToConstChar),
            "const char* (&)()");
}

TEST(TypePrinter, RValueReferences) {
  Control control;

  // rvalue reference to unsigned long
  auto referenceToUnsignedLong =
      control.getRvalueReferenceType(control.getUnsignedLongIntType());

  ASSERT_EQ(to_string(referenceToUnsignedLong), "unsigned long&&");

  // rvalue reference to pointer to char
  auto referenceToPointerToChar = control.getRvalueReferenceType(
      control.getPointerType(control.getCharType()));

  ASSERT_EQ(to_string(referenceToPointerToChar), "char*&&");
}

TEST(TypePrinter, Arrays) {
  Control control;

  // array of 10 unsigned longs
  auto arrayOf10UnsignedLongs =
      control.getBoundedArrayType(control.getUnsignedLongIntType(), 10);

  ASSERT_EQ(to_string(arrayOf10UnsignedLongs), "unsigned long [10]");

  // array of 4 arrays of 2 floats
  auto arrayOf4ArraysOf2Floats = control.getBoundedArrayType(
      control.getBoundedArrayType(control.getFloatType(), 2), 4);

  ASSERT_EQ(to_string(arrayOf4ArraysOf2Floats), "float [4][2]");

  // array of 4 arrays of 2 pointers to char
  auto arrayOf4ArraysOf2PointersToChar = control.getBoundedArrayType(
      control.getBoundedArrayType(control.getPointerType(control.getCharType()),
                                  2),
      4);

  ASSERT_EQ(to_string(arrayOf4ArraysOf2PointersToChar), "char* [4][2]");

  // array of 4 arrays of 2 pointers to const char
  auto arrayOf4ArraysOf2PointersToConstChar = control.getBoundedArrayType(
      control.getBoundedArrayType(
          control.getPointerType(control.getConstType(control.getCharType())),
          2),
      4);

  ASSERT_EQ(to_string(arrayOf4ArraysOf2PointersToConstChar),
            "const char* [4][2]");

  // array of 4 pointers to function returning int
  auto arrayOf4PointersToFunctionReturningInt = control.getBoundedArrayType(
      control.getPointerType(control.getFunctionType(control.getIntType(), {})),
      4);

  ASSERT_EQ(to_string(arrayOf4PointersToFunctionReturningInt), "int (*[4])()");
}

TEST(TypePrinter, Functions) {
  Control control;

  // function returning pointer to const char

  auto functionReturningPointerToConstChar = control.getFunctionType(
      control.getPointerType(control.getConstType(control.getCharType())), {});

  ASSERT_EQ(to_string(functionReturningPointerToConstChar), "const char* ()");

  // function returning const pointer to int

  auto functionReturningConstPointerToInt = control.getFunctionType(
      control.getConstType(control.getPointerType(control.getIntType())), {});

  ASSERT_EQ(to_string(functionReturningConstPointerToInt), "int* const ()");

  // variadic function return unsigned long

  auto variadicFunctionReturningUnsignedLong =
      control.getFunctionType(control.getUnsignedLongIntType(), {}, true);

  ASSERT_EQ(to_string(variadicFunctionReturningUnsignedLong),
            "unsigned long (...)");
}