// RUN: %cxx -toolchain macos -fcheck %s

#pragma pack(push, 1)

struct packed1 {
  char a;
  int b;
};

#pragma pack(pop)

_Static_assert(sizeof(struct packed1) == 5, "pack(1) should give 5");

struct normal {
  char a;
  int b;
};

_Static_assert(sizeof(struct normal) == 8, "normal alignment gives 8");

#pragma pack(push, 4)
struct s52 {
  unsigned int f[13];      // 52 bytes
  unsigned long long ctx;  // 8 bytes, capped to 4-byte align
};
#pragma pack(pop)

_Static_assert(sizeof(struct s52) == 60, "pack(4) should give 60");
