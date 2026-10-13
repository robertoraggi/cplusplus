// RUN: %cxx -x c -E -P -nostdinc -U_Thread_local %s -o - | %filecheck --strict-whitespace %s

restrict __restrict __restrict__;
__auto_type auto;
__alignof__(int) __alignof(int) alignof(int) _Alignof(int);
__asm__("") __asm("") asm("") _asm("");
__attribute((unused)) __attribute__((unused));
__decltype__(x) __decltype(x);
__inline__ __inline inline;
__typeof__(x) __typeof(x) typeof(x);
__volatile__ __volatile volatile;
_Alignas(int) _Bool _Static_assert(1) _Thread_local;
and_eq and bitand bitor compl not_eq not or_eq or xor_eq xor;

// CHECK: {{^}}restrict __restrict __restrict__;{{$}}
// CHECK: {{^}}__auto_type auto;{{$}}
// CHECK: {{^}}__alignof__(int) __alignof(int) alignof(int) _Alignof(int);{{$}}
// CHECK: {{^}}__asm__("") __asm("") asm("") _asm("");{{$}}
// CHECK: {{^}}__attribute((unused)) __attribute__((unused));{{$}}
// CHECK: {{^}}__decltype__(x) __decltype(x);{{$}}
// CHECK: {{^}}__inline__ __inline inline;{{$}}
// CHECK: {{^}}__typeof__(x) __typeof(x) typeof(x);{{$}}
// CHECK: {{^}}__volatile__ __volatile volatile;{{$}}
// CHECK: {{^}}_Alignas(int) _Bool _Static_assert(1) _Thread_local;{{$}}
// CHECK: {{^}}and_eq and bitand bitor compl not_eq not or_eq or xor_eq xor;{{$}}
