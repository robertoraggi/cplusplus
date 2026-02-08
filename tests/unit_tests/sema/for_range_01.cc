// RUN: %cxx -verify -fcheck %s

int sum_array() {
  int arr[4] = {1, 2, 3, 4};
  int total = 0;
  for (int x : arr) {
    total = total + x;
  }
  return total;
}
