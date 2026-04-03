// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

namespace std {
inline namespace __2 {

template <class T>
struct check {
  template <class U, class V>
  static constexpr bool test() {
    return true;
  }
};

template <class T1, class T2>
struct pair {
  T1 first;
  T2 second;

  template <class Dep = check<T1>,
            int = (Dep::template test<T1, T2>() ? 0 : -1)>
  pair(T1 const& t1, T2 const& t2) : first(t1), second(t2) {}
};

}  // namespace __2
}  // namespace std

int main() {
  std::pair<int, int> p{1, 2};
  return 0;
}
