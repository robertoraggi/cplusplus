// RUN: %cxx -verify -fsyntax-only %s

// expected-note@+1 {{candidate function not viable: no known conversion from 'short' to 'int&' for argument 1}}
void accept(int&);

int main() {
  short s = 0;
  // expected-error@+1 {{no matching function for call to 'accept'}}
  accept(s);
}
