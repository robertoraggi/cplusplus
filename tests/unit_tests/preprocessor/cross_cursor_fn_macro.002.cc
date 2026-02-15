// RUN: %cxx -verify -E -P %s -o - | %filecheck %s

#define CAT(a, b) a##b

#define DETAIL_SELECT(prefix, n) CAT(prefix, n)

#define SELECT(n) DETAIL_SELECT(HANDLER_, n)

#define HANDLER_1(x) one x
#define HANDLER_2(x) two x
#define HANDLER_3(x) three x

int a = SELECT(1)(alpha);
int b = SELECT(2)(beta);
int c = SELECT(3)(gamma);

// CHECK: int a = one alpha;
// CHECK: int b = two beta;
// CHECK: int c = three gamma;
