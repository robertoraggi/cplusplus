// RUN: %cxx -verify -fcheck %s
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
