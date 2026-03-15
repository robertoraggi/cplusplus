// RUN: %cxx -verify -fcheck %s

typedef int elem_t;
elem_t arr[10];

_Static_assert(sizeof(*arr) == sizeof(elem_t), "sizeof deref array");
_Static_assert(sizeof(arr) / sizeof(*arr) == 10, "array element count");

typedef void (*fp_t)(void);
fp_t fptable[5];
_Static_assert(sizeof(*fptable) == sizeof(fp_t), "sizeof deref fp array");
_Static_assert(sizeof(fptable) / sizeof(*fptable) == 5, "fp array count");
