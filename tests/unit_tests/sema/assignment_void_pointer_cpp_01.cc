// RUN: %cxx -verify -fcheck %s

void* malloc(unsigned long);

int main() {
  int* p = 0;
  // expected-error@+1 {{cannot assign expression of type 'void*' to 'int*'}}
  p = malloc(sizeof(int));
}
