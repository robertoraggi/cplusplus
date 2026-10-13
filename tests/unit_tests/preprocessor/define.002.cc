// RUN: %cxx -verify -E %s -o - | %filecheck %s

#define RESULT 0

#define RESULT 1  // expected-warning {{'RESULT' macro redefined}}

int main() {
  int result = RESULT;
  return result;
}

// CHECK: int result = 1;