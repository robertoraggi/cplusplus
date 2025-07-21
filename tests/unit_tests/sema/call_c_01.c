// RUN: %cxx -verify -fcheck %s

int f() { return 0; }

int main() {
  int (*fptr)() = f;
  (void)_Generic(fptr(), int: 1);
}