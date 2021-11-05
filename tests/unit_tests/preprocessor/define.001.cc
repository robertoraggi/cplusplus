// RUN: %cxx -E %s -o - | filecheck %s

#define RESULT 0

int main() {
  int result = RESULT;
  return result;
}

// CHECK: int result = 0;