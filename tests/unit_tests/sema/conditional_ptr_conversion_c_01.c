// RUN: %cxx -verify %s

struct Elem {
  struct Elem *next;
  void *data;
};

void *test_null_to_ptr(struct Elem *elem) { return elem ? elem->data : 0; }

int *test_null_to_int_ptr(int cond, int *p) { return cond ? p : 0; }

int *test_null_to_int_ptr2(int cond, int *p) { return cond ? 0 : p; }

void *test_ptr_to_void_ptr(int cond, int *p, void *q) { return cond ? p : q; }
