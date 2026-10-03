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
#include <cxx/diagnostics_client.h>
#include <cxx/translation_unit.h>
#include <gtest/gtest.h>

#include <limits>

using namespace cxx;

TEST(SourceLocations, ValidatesSegmentBounds) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  unit.setSource("int value;", "segment.cc");

  const auto first = unit.locationOfIndex(1);
  const auto last = unit.locationOfIndex(unit.tokenCount() - 1);
  const auto onePast = unit.locationOfIndex(unit.tokenCount());

  ASSERT_TRUE(unit.ownsLocation(first));
  ASSERT_TRUE(unit.ownsLocation(last));
  ASSERT_FALSE(unit.ownsLocation(SourceLocation{}));
  ASSERT_FALSE(unit.ownsLocation(onePast));
  (void)unit.tokenAt(SourceLocation{});

  unit.setTokenSegmentBase(64);

  ASSERT_EQ(unit.locationOfIndex(1).index(), 65);
}

TEST(SourceLocations, RejectsRangeOverflow) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  unit.setSource("int value;", "overflow.cc");

  const auto limit = std::numeric_limits<unsigned>::max();
  const auto largestBase = limit - unit.tokenCount();

  unit.setTokenSegmentBase(largestBase);
  ASSERT_EQ(unit.locationOfIndex(unit.tokenCount()).index(), limit);
}

TEST(MemberInstantiations, PreservesOrderAndDeduplicatesEachBatch) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto first = unit.control()->newClassSymbol(nullptr, {});
  auto second = unit.control()->newClassSymbol(nullptr, {});

  unit.addPendingMemberInstantiation(first);
  unit.addPendingMemberInstantiation(second);
  unit.addPendingMemberInstantiation(first);
  EXPECT_EQ(unit.takePendingMemberInstantiations(),
            (std::vector<ClassSymbol*>{first, second}));
  EXPECT_TRUE(unit.takePendingMemberInstantiations().empty());

  unit.addPendingMemberInstantiation(second);
  EXPECT_EQ(unit.takePendingMemberInstantiations(),
            (std::vector<ClassSymbol*>{second}));
}

TEST(MemberInstantiations, ProcessedClassesRequireReopening) {
  DiagnosticsClient diagnostics;
  TranslationUnit unit{&diagnostics};
  auto instance = unit.control()->newClassSymbol(nullptr, {});

  EXPECT_TRUE(unit.beginMemberInstantiation(instance));
  unit.addPendingMemberInstantiation(instance);
  EXPECT_TRUE(unit.takePendingMemberInstantiations().empty());
  EXPECT_FALSE(unit.beginMemberInstantiation(instance));

  unit.reopenMemberInstantiation(instance);
  unit.reopenMemberInstantiation(instance);
  EXPECT_EQ(unit.takePendingMemberInstantiations(),
            (std::vector<ClassSymbol*>{instance}));
  EXPECT_TRUE(unit.beginMemberInstantiation(instance));
  EXPECT_FALSE(unit.beginMemberInstantiation(instance));
}
