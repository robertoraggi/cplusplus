// RUN: %cxx -verify -fcheck %s

int main() {
  // expected-error@+1 {{invalid initialization of reference of type 'int&' from expression of type 'int'}}
  int& r = 1;
  (void)r;
}
