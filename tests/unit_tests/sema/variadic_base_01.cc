// RUN: %cxx -verify -fcheck %s

struct TagA {
  int a;
};

struct TagB {
  int b;
};

struct TagC {
  int c;
};

template <typename... Bases>
struct Multi : Bases... {};

static_assert(sizeof(Multi<TagA, TagB>) >= sizeof(int) * 2, "two bases");

static_assert(sizeof(Multi<TagA, TagB, TagC>) >= sizeof(int) * 3,
              "three bases");

static_assert(sizeof(Multi<TagA>) >= sizeof(int), "single base");

static_assert(sizeof(Multi<>) > 0, "empty bases");
