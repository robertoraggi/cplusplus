// RUN: %cxx -verify %s

template <typename T>
struct Box {
  Box();
  Box(const Box&);
};

template <typename T>
Box<T>::Box() {}

template <typename T>
Box<T>::Box(const Box&) {}

// expected-no-diagnostics
