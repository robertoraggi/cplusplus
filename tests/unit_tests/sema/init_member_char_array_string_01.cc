// RUN: %cxx -verify -fcheck %s

struct Token {
  int id;
  char zName[7];
  double start;
  double span;
};

Token t = {3, "day", 5373485.0, 86400.0};
