// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s

// CHECK: @macros = internal global [3 x ptr]
// CHECK: @entries = global [4 x %entry_t]
// CHECK: getelementptr inbounds nuw (i8, ptr @macros

typedef struct {
  const char* name;
  int* location;
  int defaultvalue;
  char** text_location;
  const char* default_text_value;
} entry_t;

extern entry_t entries[];

static int g_val = 0;
static char* macros[] = {"No", "Yes", "Help!"};

#define STRING_VALUE 0xFFFF

entry_t entries[] = {
    {"key1", &g_val, 5, 0, 0},
    {"macro0", 0, STRING_VALUE, &macros[0], "No"},
    {"macro1", 0, STRING_VALUE, &macros[1], "Yes"},
    {"macro2", 0, STRING_VALUE, &macros[2], "Help!"},
};

int nentries = sizeof(entries) / sizeof(entry_t);

int main(void) {
  for (int i = 0; i < nentries; i++) {
    if (entries[i].defaultvalue != STRING_VALUE)
      *entries[i].location = entries[i].defaultvalue;
  }
  return g_val == 5 ? 0 : 1;
}
