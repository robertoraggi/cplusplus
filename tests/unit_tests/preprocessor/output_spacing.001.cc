// RUN: %cxx -E -P -nostdinc %s -o - | %filecheck --strict-whitespace %s

#define TYPE __typeof__
#define DECL __decltype
#define ID(x) x
#define PASTE(a, b) a##b
#define EMPTY
#define PLUS +
#define GT >
#define DOT .
#define STR(x) #x
#define REPEAT(x) x, x
#define DECLARE(x) int x;
#define VARIABLES(V) V(first) V(second) V(third)

__typeof__(int) x;
__decltype(x) y;
TYPE(x) z;
DECL(x) w;
bool f() { return true and not false; }
int value = ID(123);
int counter = __COUNTER__;
int pasted = PASTE(12, 34);
int shifted = value >> 2;
int separate = value GT>2;
int increment = value PLUS+2;
int comment = value/EMPTY*2;
a DOT DOT DOT;
a ID(.)ID(.)ID(.);
int number = 1 EMPTY.2;
a L EMPTY"x";
a "x" EMPTY _suffix;
const char* spelling = STR(__typeof__(x) and not false);
const char* file = __FILE__;
const char* date = __DATE__;
const char* time = __TIME__;
int line = __LINE__;
int values[] = {REPEAT(REPEAT(3))};
void variables() { VARIABLES(DECLARE) }
a ID(<)ID(:);
a ID(<)ID(%);
a ID(:)ID(>);
a ID(%)ID(>);
a ID(%)ID(:);
a ID(#)ID(%);
a ID(123)ID(+);
a ID(123)ID(-);
a ID(R)ID('c');
const char* joined = ID("one")ID("two");

// CHECK: {{^}}__typeof__(int) x;{{$}}
// CHECK: {{^}}__decltype(x) y;{{$}}
// CHECK: {{^}}__typeof__(x) z;{{$}}
// CHECK: {{^}}__decltype(x) w;{{$}}
// CHECK: {{^}}bool f() { return true and not false; }{{$}}
// CHECK: {{^}}int value = 123;{{$}}
// CHECK: {{^}}int counter = 0;{{$}}
// CHECK: {{^}}int pasted = 1234;{{$}}
// CHECK: {{^}}int shifted = value >> 2;{{$}}
// CHECK: {{^}}int separate = value > >2;{{$}}
// CHECK: {{^}}int increment = value + +2;{{$}}
// CHECK: {{^}}int comment = value/ *2;{{$}}
// CHECK: {{^}}a . . .;{{$}}
// CHECK: {{^}}a .. .;{{$}}
// CHECK: {{^}}int number = 1 .2;{{$}}
// CHECK: {{^}}a L "x";{{$}}
// CHECK: {{^}}a "x" _suffix;{{$}}
// CHECK: {{^}}const char* spelling = "__typeof__(x) and not false";{{$}}
// CHECK: {{^}}const char* file = "{{.*}}output_spacing.001.cc";{{$}}
// CHECK: {{^}}const char* date = "{{.*}}";{{$}}
// CHECK: {{^}}const char* time = "{{.*}}";{{$}}
// CHECK: {{^}}int line = {{[0-9]+}};{{$}}
// CHECK: {{^}}int values[] = {3, 3, 3, 3};{{$}}
// CHECK: {{^}}void variables() { int first; int second; int third; }{{$}}
// CHECK: {{^}}a < :;{{$}}
// CHECK: {{^}}a < %;{{$}}
// CHECK: {{^}}a : >;{{$}}
// CHECK: {{^}}a % >;{{$}}
// CHECK: {{^}}a % :;{{$}}
// CHECK: {{^}}a # %;{{$}}
// CHECK: {{^}}a 123 +;{{$}}
// CHECK: {{^}}a 123 -;{{$}}
// CHECK: {{^}}a R 'c';{{$}}
// CHECK: {{^}}const char* joined = "one"{{ *}}"two";{{$}}
