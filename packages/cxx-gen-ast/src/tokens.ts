// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

export const BASE_TOKENS: string[] = [
  "EOF_SYMBOL",
  "ERROR",
  "COMMENT",
  "BUILTIN",
  "IDENTIFIER",
  "CHARACTER_LITERAL",
  "FLOATING_POINT_LITERAL",
  "INTEGER_LITERAL",
  "STRING_LITERAL",
  "USER_DEFINED_STRING_LITERAL",
  "UTF16_STRING_LITERAL",
  "UTF32_STRING_LITERAL",
  "UTF8_STRING_LITERAL",
  "WIDE_STRING_LITERAL",
];

export const OPERATORS: Array<[kind: string, spelling: string]> = [
  ["AMP_AMP", "&&"],
  ["AMP_EQUAL", "&="],
  ["AMP", "&"],
  ["BAR_BAR", "||"],
  ["BAR_EQUAL", "|="],
  ["BAR", "|"],
  ["CARET_EQUAL", "^="],
  ["CARET", "^"],
  ["COLON_COLON", "::"],
  ["COLON", ":"],
  ["COMMA", ","],
  ["DELETE_ARRAY", "delete[]"],
  ["DOT_DOT_DOT", "..."],
  ["DOT_STAR", ".*"],
  ["DOT", "."],
  ["EQUAL_EQUAL", "=="],
  ["EQUAL", "="],
  ["EXCLAIM_EQUAL", "!="],
  ["EXCLAIM", "!"],
  ["GREATER_EQUAL", ">="],
  ["GREATER_GREATER_EQUAL", ">>="],
  ["GREATER_GREATER", ">>"],
  ["GREATER", ">"],
  ["HASH_HASH", "##"],
  ["HASH", "#"],
  ["LBRACE", "{"],
  ["LBRACKET", "["],
  ["LESS_EQUAL_GREATER", "<=>"],
  ["LESS_EQUAL", "<="],
  ["LESS_LESS_EQUAL", "<<="],
  ["LESS_LESS", "<<"],
  ["LESS", "<"],
  ["LPAREN", "("],
  ["MINUS_EQUAL", "-="],
  ["MINUS_GREATER_STAR", "->*"],
  ["MINUS_GREATER", "->"],
  ["MINUS_MINUS", "--"],
  ["MINUS", "-"],
  ["NEW_ARRAY", "new[]"],
  ["PERCENT_EQUAL", "%="],
  ["PERCENT", "%"],
  ["PLUS_EQUAL", "+="],
  ["PLUS_PLUS", "++"],
  ["PLUS", "+"],
  ["QUESTION", "?"],
  ["RBRACE", "}"],
  ["RBRACKET", "]"],
  ["RPAREN", ")"],
  ["SEMICOLON", ";"],
  ["SLASH_EQUAL", "/="],
  ["SLASH", "/"],
  ["STAR_EQUAL", "*="],
  ["STAR", "*"],
  ["TILDE", "~"],
];

export const KEYWORDS: string[] = [
  "alignas",
  "alignof",
  "asm",
  "auto",
  "bool",
  "break",
  "case",
  "catch",
  "char",
  "char16_t",
  "char32_t",
  "char8_t",
  "class",
  "co_await",
  "co_return",
  "co_yield",
  "concept",
  "const_cast",
  "const",
  "consteval",
  "constexpr",
  "constinit",
  "continue",
  "decltype",
  "default",
  "delete",
  "do",
  "double",
  "dynamic_cast",
  "else",
  "enum",
  "explicit",
  "export",
  "extern",
  "false",
  "float",
  "for",
  "friend",
  "goto",
  "if",
  "import",
  "inline",
  "int",
  "long",
  "module",
  "mutable",
  "namespace",
  "new",
  "noexcept",
  "nullptr",
  "operator",
  "private",
  "protected",
  "public",
  "reinterpret_cast",
  "requires",
  "return",
  "short",
  "signed",
  "sizeof",
  "static_assert",
  "static_cast",
  "static",
  "struct",
  "switch",
  "template",
  "this",
  "thread_local",
  "throw",
  "true",
  "try",
  "typedef",
  "typeid",
  "typename",
  "union",
  "unsigned",
  "using",
  "virtual",
  "void",
  "volatile",
  "wchar_t",
  "while",
  "_Atomic",
  "_Complex",
  "__attribute__",
  "__builtin_va_arg",
  "__builtin_va_list",
  "__complex__",
  "__extension__",
  "__float128",
  "__float80",
  "__imag__",
  "__int128",
  "__int64",
  "__real__",
  "__restrict__",
  "__thread",
  "__underlying_type",
];

export const BUILTIN_CASTS: string[] = ["__builtin_bit_cast"];

export const BUILTIN_TYPE_TRAITS: string[] = [
  "__has_unique_object_representations",
  "__has_virtual_destructor",
  "__is_abstract",
  "__is_aggregate",
  "__is_arithmetic",
  "__is_array",
  "__is_assignable",
  "__is_base_of",
  "__is_bounded_array",
  "__is_class",
  "__is_compound",
  "__is_const",
  "__is_empty",
  "__is_enum",
  "__is_final",
  "__is_floating_point",
  "__is_function",
  "__is_fundamental",
  "__is_integral",
  "__is_layout_compatible",
  "__is_literal_type",
  "__is_lvalue_reference",
  "__is_member_function_pointer",
  "__is_member_object_pointer",
  "__is_member_pointer",
  "__is_null_pointer",
  "__is_object",
  "__is_pod",
  "__is_pointer",
  "__is_polymorphic",
  "__is_reference",
  "__is_rvalue_reference",
  "__is_same_as",
  "__is_same",
  "__is_scalar",
  "__is_scoped_enum",
  "__is_signed",
  "__is_standard_layout",
  "__is_swappable_with",
  "__is_trivial",
  "__is_unbounded_array",
  "__is_union",
  "__is_unsigned",
  "__is_void",
  "__is_volatile",
];

export const TOKEN_ALIASES = [
  ["and_eq", "AMP_EQUAL"],
  ["and", "AMP_AMP"],
  ["bitand", "AMP"],
  ["bitor", "BAR"],
  ["compl", "TILDE"],
  ["not_eq", "EXCLAIM_EQUAL"],
  ["not", "EXCLAIM"],
  ["or_eq", "BAR_EQUAL"],
  ["or", "BAR_BAR"],
  ["xor_eq", "CARET_EQUAL"],
  ["xor", "CARET"],

  ["__alignof__", "ALIGNOF"],
  ["__alignof", "ALIGNOF"],
  ["__asm__", "ASM"],
  ["__asm", "ASM"],
  ["__attribute", "__ATTRIBUTE__"],
  ["__decltype__", "DECLTYPE"],
  ["__decltype", "DECLTYPE"],
  ["__inline__", "INLINE"],
  ["__inline", "INLINE"],
  ["__restrict", "__RESTRICT__"],
  ["__typeof__", "DECLTYPE"],
  ["__typeof", "DECLTYPE"],
  ["_Alignof", "ALIGNOF"],
  ["_Static_assert", "STATIC_ASSERT"],
];
