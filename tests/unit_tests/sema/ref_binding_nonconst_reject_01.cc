// RUN: %cxx -verify -fcheck %s

void accept(int&);

int main() {
  short s = 0;
  // expected-error@+1 {{invalid argument of type 'short' for parameter of type 'int&'}}
  accept(s);
}
