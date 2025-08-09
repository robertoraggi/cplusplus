// RUN: %cxx %s

char ch = 'a';

_Static_assert(_Generic(ch, int: 123, char: 321) == 321);

_Static_assert(_Generic('a', int: 123, char: 321) == 123);

_Static_assert(_Generic("str", char*: 123, const char*: 321) == 123);

_Static_assert(_Generic(true ? "str" : (const char*)0,
                   char*: 123,
                   const char*: 321) == 321);