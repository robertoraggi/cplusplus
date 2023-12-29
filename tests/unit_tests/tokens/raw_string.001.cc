// RUN: %cxx -verify -dump-tokens %s -o - | %filecheck %s

R"_(\-?(0|[1-9][0-9]*)(\.[0-9]+)?((e|E)(\+|\-)[0-9]+)?)_";

// clang-format off

// CHECK: STRING_LITERAL 'R"_(\-?(0|[1-9][0-9]*)(\.[0-9]+)?((e|E)(\+|\-)[0-9]+)?)_"'
// CHECK: SEMICOLON ';'
