// RUN: %cxx -verify %s
// expected-no-diagnostics

void use_code() {
  extern char* code0[];
  char** p = code0;
  (void)p;
}

char* code0[] = {"a", "b", 0};

void use_code2() {
  extern char* code0[];
  (void)code0;
}
