// RUN: %cxx -E %s -o - | %filecheck %s

#define B

#ifdef A
int a_not_reachable;
#elifdef B
int b_reachable;
#elifdef C
int c_not_reachable;
#else
int else_not_reachable;
#endif

// CHECK: b_reachable;

#ifdef AA
int aa_not_reachable;
#elifndef BB
int bb_reachable;
#elifndef CC
int cc_not_reachable;
#elifdef DD
int dd_not_reachable;
#else
int else_not_reachable;
#endif

// CHECK: bb_reachable;

#ifdef AAA
int aaa_not_reachable;
#elifdef BBB
int bbb_not_reachable;
#elifdef CCC
int ccc_not_reachable;
#else
int else_reachable;
#endif

// CHECK: else_reachable;
