// RUN: %cxx -toolchain macos -verify -fcheck %s

void test_const_cast() {
  int i = 0;
  const int ci = 0;
  volatile int vi = 0;
  const volatile int cvi = 0;

  int* pi = &i;
  const int* pci = &ci;
  volatile int* pvi = &vi;
  const volatile int* pcvi = &cvi;

  // To pointer
  (void)const_cast<int*>(pi);
  (void)const_cast<int*>(pci);
  (void)const_cast<int*>(pvi);
  (void)const_cast<int*>(pcvi);

  (void)const_cast<const int*>(pi);
  (void)const_cast<const int*>(pci);
  (void)const_cast<const int*>(pvi);
  (void)const_cast<const int*>(pcvi);

  (void)const_cast<volatile int*>(pi);
  (void)const_cast<volatile int*>(pci);
  (void)const_cast<volatile int*>(pvi);
  (void)const_cast<volatile int*>(pcvi);

  (void)const_cast<const volatile int*>(pi);
  (void)const_cast<const volatile int*>(pci);
  (void)const_cast<const volatile int*>(pvi);
  (void)const_cast<const volatile int*>(pcvi);

  // To reference
  (void)const_cast<int&>(i);
  (void)const_cast<int&>(ci);
  (void)const_cast<int&>(vi);
  (void)const_cast<int&>(cvi);

  (void)const_cast<const int&>(i);
  (void)const_cast<const int&>(ci);
  (void)const_cast<const int&>(vi);
  (void)const_cast<const int&>(cvi);
}

struct A {};
struct B : A {};

void test_class_pointers() {
  A* pa = 0;
  const A* pca = 0;
  (void)const_cast<A*>(pca);

  (void)sizeof(int);
}
