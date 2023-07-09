#include <cxx/control.h>
#include <cxx/types.h>
#include <gtest/gtest.h>

using namespace cxx;

TEST(TypePrinter, BasicTypes) {
  Control control;

  ASSERT_EQ(to_string(control.getNullptrType()), "decltype(nullptr)");
  ASSERT_EQ(to_string(control.getAutoType()), "auto");
  ASSERT_EQ(to_string(control.getVoidType()), "void");
  ASSERT_EQ(to_string(control.getBoolType()), "bool");
  ASSERT_EQ(to_string(control.getCharType()), "char");
  ASSERT_EQ(to_string(control.getSignedCharType()), "signed char");
  ASSERT_EQ(to_string(control.getUnsignedCharType()), "unsigned char");
  ASSERT_EQ(to_string(control.getShortType()), "short");
  ASSERT_EQ(to_string(control.getUnsignedShortType()), "unsigned short");
  ASSERT_EQ(to_string(control.getIntType()), "int");
  ASSERT_EQ(to_string(control.getUnsignedIntType()), "unsigned int");
  ASSERT_EQ(to_string(control.getLongType()), "long");
  ASSERT_EQ(to_string(control.getUnsignedLongType()), "unsigned long");
  ASSERT_EQ(to_string(control.getFloatType()), "float");
  ASSERT_EQ(to_string(control.getDoubleType()), "double");
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
  auto pointerToArrayOf10Ints =
      control.getPointerType(control.getArrayType(control.getIntType(), 10));

  ASSERT_EQ(to_string(pointerToArrayOf10Ints), "int (*)[10]");
}

TEST(TypePrinter, LValueReferences) {
  Control control;

  // lvalue reference to unsigned long
  auto referenceToUnsignedLong =
      control.getLValueReferenceType(control.getUnsignedLongType());

  ASSERT_EQ(to_string(referenceToUnsignedLong), "unsigned long&");

  // lvalue reference to pointer to char
  auto referenceToPointerToChar = control.getLValueReferenceType(
      control.getPointerType(control.getCharType()));

  ASSERT_EQ(to_string(referenceToPointerToChar), "char*&");

  // lvalue reference to a const pointer to char
  auto referenceToConstPointerToChar = control.getLValueReferenceType(
      control.getConstType(control.getPointerType(control.getCharType())));

  ASSERT_EQ(to_string(referenceToConstPointerToChar), "char* const&");

  // reference to array of 10 ints
  auto referenceToArrayOf10Ints = control.getLValueReferenceType(
      control.getArrayType(control.getIntType(), 10));

  ASSERT_EQ(to_string(referenceToArrayOf10Ints), "int (&)[10]");
}

TEST(TypePrinter, RValueReferences) {
  Control control;

  // rvalue reference to unsigned long
  auto referenceToUnsignedLong =
      control.getRValueReferenceType(control.getUnsignedLongType());

  ASSERT_EQ(to_string(referenceToUnsignedLong), "unsigned long&&");

  // rvalue reference to pointer to char
  auto referenceToPointerToChar = control.getRValueReferenceType(
      control.getPointerType(control.getCharType()));

  ASSERT_EQ(to_string(referenceToPointerToChar), "char*&&");
}

TEST(TypePrinter, Arrays) {
  Control control;

  // array of 10 unsigned longs
  auto arrayOf10UnsignedLongs =
      control.getArrayType(control.getUnsignedLongType(), 10);

  ASSERT_EQ(to_string(arrayOf10UnsignedLongs), "unsigned long [10]");

  // array of 4 arrays of 2 floats
  auto arrayOf10ArraysOf2Floats =
      control.getArrayType(control.getArrayType(control.getFloatType(), 2), 4);

  ASSERT_EQ(to_string(arrayOf10ArraysOf2Floats), "float [4][2]");

  // array of 4 arrays of 2 pointers to char
  auto arrayOf10ArraysOf2PointersToChar = control.getArrayType(
      control.getArrayType(control.getPointerType(control.getCharType()), 2),
      4);

  ASSERT_EQ(to_string(arrayOf10ArraysOf2PointersToChar), "char* [4][2]");

  // array of 4 arrays of 2 pointers to const char
  auto arrayOf4ArraysOf2PointersToConstChar = control.getArrayType(
      control.getArrayType(
          control.getPointerType(control.getConstType(control.getCharType())),
          2),
      4);

  ASSERT_EQ(to_string(arrayOf4ArraysOf2PointersToConstChar),
            "const char* [4][2]");
}