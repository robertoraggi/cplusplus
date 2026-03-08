// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

namespace std {

using size_t = decltype(sizeof(0));

template <typename T>
struct basic_string {
  void append(const T* p);
  auto c_str() const -> const T*;
  auto size() const -> size_t;
  auto operator[](size_t i) const -> const T&;
  auto operator[](size_t i) -> T&;

  basic_string() = default;
  explicit basic_string(const T* p) {}

  struct iterator {
    iterator& operator++();
    iterator operator++(int);
    auto operator*() -> T&;
    auto operator*() const -> const T&;
    bool operator==(const iterator&) const;
    bool operator!=(const iterator&) const;
  };

  auto begin() -> iterator;
  auto end() -> iterator;
};

using string = basic_string<char>;
using u8string = basic_string<char8_t>;

}  // namespace std

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  namespace std
// CHECK-NEXT:    typealias unsigned long size_t
// CHECK-NEXT:    template class basic_string<type-param<0, 0>>
// CHECK-NEXT:      parameter typename<0, 0> T
// CHECK-NEXT:      constructor inline defaulted void basic_string()
// CHECK-NEXT:      constructor inline explicit void basic_string(const type-param<0, 0>*)
// CHECK-NEXT:        parameters
// CHECK-NEXT:          parameter const type-param<0, 0>* p
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[13]
// CHECK-NEXT:      injected class name basic_string
// CHECK-NEXT:      function void append(const type-param<0, 0>*)
// CHECK-NEXT:      function const type-param<0, 0>* c_str() const
// CHECK-NEXT:      function unsigned long size() const
// CHECK-NEXT:      function const type-param<0, 0>& operator [](unsigned long) const
// CHECK-NEXT:      function type-param<0, 0>& operator [](unsigned long)
// CHECK-NEXT:      class iterator
// CHECK-NEXT:        injected class name iterator
// CHECK-NEXT:        function std::basic_string::iterator& operator ++()
// CHECK-NEXT:        function std::basic_string::iterator operator ++(int)
// CHECK-NEXT:        function type-param<0, 0>& operator *()
// CHECK-NEXT:        function const type-param<0, 0>& operator *() const
// CHECK-NEXT:        function bool operator ==(const std::basic_string::iterator&) const
// CHECK-NEXT:        function bool operator !=(const std::basic_string::iterator&) const
// CHECK-NEXT:      function std::basic_string::iterator begin()
// CHECK-NEXT:      function std::basic_string::iterator end()
// CHECK-NEXT:      [specializations]
// CHECK-NEXT:        class basic_string<char>
// CHECK-NEXT:        class basic_string<char8_t>
// CHECK-NEXT:    typealias std::basic_string<char> string
// CHECK-NEXT:    typealias std::basic_string<char8_t> u8string
