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

#include <cxx/arena.h>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cxx;

namespace {

struct Tracked {
  std::vector<int>* log;
  int id;

  Tracked(std::vector<int>* log, int id) : log(log), id(id) {}

  ~Tracked() { log->push_back(id); }
};

}  // namespace

TEST(Arena, ReusesStorageAfterRewind) {
  Arena arena;

  auto first = arena.make<int>(1);
  const auto mark = arena.mark();
  auto second = arena.make<int>(2);

  arena.rewind(mark);

  auto third = arena.make<int>(3);

  ASSERT_EQ(second, third);
  ASSERT_EQ(*first, 1);
  ASSERT_EQ(*third, 3);
}

TEST(Arena, RunsDestructorsAfterTheMarkInReverseOrder) {
  std::vector<int> destroyed;

  {
    Arena arena;

    (void)arena.make<Tracked>(&destroyed, 1);
    const auto mark = arena.mark();
    (void)arena.make<Tracked>(&destroyed, 2);
    (void)arena.make<Tracked>(&destroyed, 3);

    arena.rewind(mark);

    ASSERT_EQ(destroyed, (std::vector<int>{3, 2}));

    destroyed.clear();
  }

  ASSERT_EQ(destroyed, (std::vector<int>{1}));
}

TEST(Arena, KeepsAlignmentAcrossChunks) {
  Arena arena;

  std::vector<std::string*> strings;

  for (int i = 0; i < 4096; ++i) {
    strings.push_back(arena.make<std::string>(std::string(64, 'x')));
  }

  for (auto string : strings) {
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(string) % alignof(std::string),
              0);
    ASSERT_EQ(string->size(), 64);
  }
}

TEST(Arena, SupportsOveralignedAllocationsAfterGrowthAndRewind) {
  Arena arena;
  const auto mark = arena.mark();

  (void)arena.allocate(64 * 1024, 1);
  auto* first = arena.allocate(1, 1024 * 1024);

  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(first) % (1024 * 1024), 0);

  arena.rewind(mark);

  (void)arena.allocate(64 * 1024, 1);
  auto* second = arena.allocate(1, 1024 * 1024);

  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(second) % (1024 * 1024), 0);
}

TEST(Arena, RejectsAllocationSizeOverflow) {
  Arena arena;
  const auto maxSize = std::numeric_limits<std::size_t>::max();
  ASSERT_THROW((void)arena.allocate(maxSize, 2), std::runtime_error);
}
