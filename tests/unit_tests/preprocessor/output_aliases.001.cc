// RUN: %cxx -E -P -nostdinc %s -o - | %filecheck --strict-whitespace %s

__alignof__(int) __alignof(int) alignof(int);
__asm__("") __asm("") asm("");
__attribute((unused)) __attribute__((unused));
__decltype__(x) __decltype(x) decltype(x);
__inline__ __inline inline;
__restrict __restrict__;
__typeof__(x) __typeof(x) typeof(x);
__volatile__ __volatile volatile;
_Alignof(int) _Static_assert(true);
and_eq and bitand bitor compl not_eq not or_eq or xor_eq xor;

// CHECK: {{^}}__alignof__(int) __alignof(int) alignof(int);{{$}}
// CHECK: {{^}}__asm__("") __asm("") asm("");{{$}}
// CHECK: {{^}}__attribute((unused)) __attribute__((unused));{{$}}
// CHECK: {{^}}__decltype__(x) __decltype(x) decltype(x);{{$}}
// CHECK: {{^}}__inline__ __inline inline;{{$}}
// CHECK: {{^}}__restrict __restrict__;{{$}}
// CHECK: {{^}}__typeof__(x) __typeof(x) typeof(x);{{$}}
// CHECK: {{^}}__volatile__ __volatile volatile;{{$}}
// CHECK: {{^}}_Alignof(int) _Static_assert(true);{{$}}
// CHECK: {{^}}and_eq and bitand bitor compl not_eq not or_eq or xor_eq xor;{{$}}
