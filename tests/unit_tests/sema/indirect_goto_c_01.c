// RUN: %cxx -verify %s

int main() {
  void* target = 0;

  target = &&L;
  goto* target;

L:
  return 0;
}