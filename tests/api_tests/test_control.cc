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

#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/name_printer.h>
#include <cxx/names.h>
#include <gtest/gtest.h>

using namespace cxx;

TEST(Control, integer_literals) {
  Control control;

  auto literal = control.integerLiteral("42");
  EXPECT_EQ(literal->value(), "42");

  EXPECT_EQ(control.integerLiteral("42"), literal);
}

TEST(Control, float_literals) {
  Control control;

  auto literal = control.floatLiteral("42.0");
  EXPECT_EQ(literal->value(), "42.0");

  EXPECT_EQ(control.floatLiteral("42.0"), literal);
}

TEST(Control, string_literals) {
  Control control;

  auto literal = control.stringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.stringLiteral("foo"), literal);
}

TEST(Control, char_literals) {
  Control control;

  auto literal = control.charLiteral("a");
  EXPECT_EQ(literal->value(), "a");

  EXPECT_EQ(control.charLiteral("a"), literal);
}

TEST(Control, wide_string_literals) {
  Control control;

  auto literal = control.wideStringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.wideStringLiteral("foo"), literal);
}

TEST(Control, utf8_string_literals) {
  Control control;

  auto literal = control.utf8StringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.utf8StringLiteral("foo"), literal);
}

TEST(Control, utf16_string_literals) {
  Control control;

  auto literal = control.utf16StringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.utf16StringLiteral("foo"), literal);
}

TEST(Control, utf32_string_literals) {
  Control control;

  auto literal = control.utf32StringLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.utf32StringLiteral("foo"), literal);
}

TEST(Control, comment_literals) {
  Control control;

  auto literal = control.commentLiteral("foo");
  EXPECT_EQ(literal->value(), "foo");

  EXPECT_EQ(control.commentLiteral("foo"), literal);
}

TEST(Control, make_anonymous_id) {
  Control control;

  auto id = control.newAnonymousId("foo");
  EXPECT_TRUE(id->isAnonymous());
  EXPECT_TRUE(id->name().starts_with("$foo"));

  auto otherId = control.newAnonymousId("foo");
  EXPECT_NE(id, otherId);
  EXPECT_NE(id->name(), otherId->name());
}

TEST(Control, get_identifier) {
  Control control;

  auto id = control.getIdentifier("foo");
  EXPECT_EQ(id->name(), "foo");

  EXPECT_EQ(control.getIdentifier("foo"), id);
  EXPECT_NE(name_cast<Identifier>(id), nullptr);
}

TEST(Control, get_operator_id) {
  Control control;

  auto id = control.getOperatorId(TokenKind::T_PLUS);
  EXPECT_EQ(id->op(), TokenKind::T_PLUS);

  EXPECT_EQ(to_string(id), "operator +");

  EXPECT_EQ(control.getOperatorId(TokenKind::T_PLUS), id);
  EXPECT_NE(name_cast<OperatorId>(id), nullptr);
}

TEST(Control, get_destructor_id) {
  Control control;

  auto id = control.getDestructorId(control.getIdentifier("foo"));
  EXPECT_EQ(to_string(id), "~foo");

  EXPECT_EQ(control.getDestructorId(control.getIdentifier("foo")), id);
  EXPECT_NE(name_cast<DestructorId>(id), nullptr);
}
