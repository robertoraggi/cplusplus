
class Class {
 public:
  int a, b;
  int* ptr;
  const char* s;

  struct Private {
    int a, b;
    Class* p;
    Private(Class* p);
    const Class* getClass();
  } *x;

  void dump();
  void init(const char* s);
};
