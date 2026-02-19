// clang-format off
// RUN: %cxx -verify -emit-llvm %s -o %t.ll
// RUN: grep "call void @llvm.va_start" %t.ll
// RUN: grep "call void @llvm.va_end" %t.ll

using va_list = __builtin_va_list;

int vasprintf(char** buf, const char* fmt, va_list ap);

char* format(const char* fmt, ...) {
  va_list ap;
  __builtin_va_start(ap, fmt);
  char* buf = nullptr;
  vasprintf(&buf, fmt, ap);
  __builtin_va_end(ap);
  return buf;
}
