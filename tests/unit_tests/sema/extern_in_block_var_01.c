// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

void writer() {
  extern int zoo;
  zoo = 123;
}

int zoo;

void reader() {
  extern int zoo;
  (void)zoo;
}
