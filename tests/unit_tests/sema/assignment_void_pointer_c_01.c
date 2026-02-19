// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

void* malloc(unsigned long);

int main() {
  int* p = 0;
  p = malloc(sizeof(int));
}
