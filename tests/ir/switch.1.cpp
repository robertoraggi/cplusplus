#include <stdio.h>
int case_1() { printf("%s\n", __FUNCTION__); return 123; }
int case_2() { printf("%s\n", __FUNCTION__); return 321; }
int case_3() { printf("%s\n", __FUNCTION__); return 999; }
int default_case() { printf("%s\n", __FUNCTION__); return 444; }

int switch_1(int a) {
  switch (a) {
  case 1: return case_1();
  case 2: return case_2();
  case 3: return case_3();
  }
}

int switch_2(int a) {
  switch (a) {
  case 1: case_1();
  case 2: case_2();
  }
}

int switch_3(int a) {
  switch (a) {
  case 1:
  case 2: case_1(); break;
  case 3:
  case 4: case_2(); break;
  }
}

int switch_4(int a) {
  switch (a) {
  case 1:
  case 2: case_1();
  case 3:
  case 4: case_2();
  }
}

int switch_5(int a) {
  switch (a) {
  case 1: return case_1();
  case 2: return case_2();
  case 3: return case_3();
  default: return default_case();
  }
}

int switch_6(int a) {
  switch (a) {
  default: return default_case();
  case 1: return case_1();
  case 2: return case_2();
  case 3: return case_3();
  }
}

int main() {
  printf("switch_1\n");
  switch_1(0);
  switch_1(1);
  switch_1(2);
  switch_1(3);
  printf("switch_2\n");
  switch_2(0);
  switch_2(1);
  switch_2(2);
  switch_2(3);
  printf("switch_3\n");
  switch_3(0);
  switch_3(1);
  switch_3(2);
  switch_3(3);
  printf("switch_4\n");
  switch_4(0);
  switch_4(1);
  switch_4(2);
  switch_4(3);
  printf("switch_5\n");
  switch_5(0);
  switch_5(1);
  switch_5(2);
  switch_5(3);
}
