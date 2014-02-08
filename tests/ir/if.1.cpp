
#include <cstdio>

enum {
  T_ASM,
  T_FOR,
  T_INT,
  T_NEW,
  T_TRY,
  T_IDENTIFIER
};

static inline int classify3(const char *s) {
  if (s[0] == 'a') {
    if (s[1] == 's') {
      if (s[2] == 'm') {
        return T_ASM;
      }
    }
  }
  else if (s[0] == 'f') {
    if (s[1] == 'o') {
      if (s[2] == 'r') {
        return T_FOR;
      }
    }
  }
  else if (s[0] == 'i') {
    if (s[1] == 'n') {
      if (s[2] == 't') {
        return T_INT;
      }
    }
  }
  else if (s[0] == 'n') {
    if (s[1] == 'e') {
      if (s[2] == 'w') {
        return T_NEW;
      }
    }
  }
  else if (s[0] == 't') {
    if (s[1] == 'r') {
      if (s[2] == 'y') {
        return T_TRY;
      }
    }
  }
  return T_IDENTIFIER;
}

int main() {
  printf("asm is keyword %d\n", classify3("asm") == T_ASM);
  printf("for is keyword %d\n", classify3("for") == T_FOR);
  printf("int is keyword %d\n", classify3("int") == T_INT);
  printf("new is keyword %d\n", classify3("new") == T_NEW);
  printf("try is keyword %d\n", classify3("try") == T_TRY);
  printf("oki is identifier %d\n", classify3("oki") == T_IDENTIFIER);
}
