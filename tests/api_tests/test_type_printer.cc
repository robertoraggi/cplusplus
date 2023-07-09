#include <cxx/control.h>
#include <cxx/types.h>
#include <gtest/gtest.h>

using namespace cxx;

TEST(TypePrinter, BasicTypes) {
  cxx::Control control;

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
