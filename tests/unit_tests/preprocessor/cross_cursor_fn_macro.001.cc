// RUN: %cxx -verify -E -P %s -o - | %filecheck %s

#define IIF_0(t, f) f
#define IIF_1(t, f) t
#define CAT(a, b) a##b
#define IIF(bit, t, f) CAT(IIF_, bit)(t, f)

#define EXEC(x) x is exec
#define EMPTY(x) x is empty

int a = IIF(1, EXEC, EMPTY)(hello);
int b = IIF(0, EXEC, EMPTY)(world);

// CHECK: int a = hello is exec;
// CHECK: int b = world is empty;
