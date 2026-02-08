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
// CHECK-NEXT:      constructor defaulted void basic_string()
// CHECK-NEXT:      constructor explicit void basic_string(const type-param<0, 0>*)
// CHECK-NEXT:        parameters
// CHECK-NEXT:          parameter const type-param<0, 0>* p
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[13]
// CHECK-NEXT:      function void append(const type-param<0, 0>*)
// CHECK-NEXT:      function const type-param<0, 0>* c_str() const
// CHECK-NEXT:      function unsigned long size() const
// CHECK-NEXT:      function const type-param<0, 0>& operator [](unsigned long) const
// CHECK-NEXT:      function type-param<0, 0>& operator [](unsigned long)
// CHECK-NEXT:      class iterator
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
// CHECK-NEXT:          constructor void basic_string()
// CHECK-NEXT:          constructor explicit void basic_string(const char*)
// CHECK-NEXT:            parameters
// CHECK-NEXT:              parameter const char* p
// CHECK-NEXT:              block
// CHECK-NEXT:          constructor defaulted void basic_string(const std::basic_string<char>&)
// CHECK-NEXT:          constructor defaulted void basic_string(std::basic_string<char>&&)
// CHECK-NEXT:          function void append(const char*)
// CHECK-NEXT:          function const char* c_str() const
// CHECK-NEXT:          function unsigned long size() const
// CHECK-NEXT:          function const char& operator [](unsigned long) const
// CHECK-NEXT:          function char& operator [](unsigned long)
// CHECK-NEXT:          class iterator
// CHECK-NEXT:            constructor defaulted void iterator()
// CHECK-NEXT:            constructor defaulted void iterator(const std::basic_string<char>::iterator&)
// CHECK-NEXT:            constructor defaulted void iterator(std::basic_string<char>::iterator&&)
// CHECK-NEXT:            function std::basic_string<char>::iterator& operator ++()
// CHECK-NEXT:            function std::basic_string<char>::iterator operator ++(int)
// CHECK-NEXT:            function char& operator *()
// CHECK-NEXT:            function const char& operator *() const
// CHECK-NEXT:            function bool operator ==(const std::basic_string<char>::iterator&) const
// CHECK-NEXT:            function bool operator !=(const std::basic_string<char>::iterator&) const
// CHECK-NEXT:            function defaulted void ~iterator()
// CHECK-NEXT:          function std::basic_string<char>::iterator begin()
// CHECK-NEXT:          function std::basic_string<char>::iterator end()
// CHECK-NEXT:          function defaulted void ~basic_string()
// CHECK-NEXT:        class basic_string<char8_t>
// CHECK-NEXT:          constructor void basic_string()
// CHECK-NEXT:          constructor explicit void basic_string(const char8_t*)
// CHECK-NEXT:            parameters
// CHECK-NEXT:              parameter const char8_t* p
// CHECK-NEXT:              block
// CHECK-NEXT:          constructor defaulted void basic_string(const std::basic_string<char8_t>&)
// CHECK-NEXT:          constructor defaulted void basic_string(std::basic_string<char8_t>&&)
// CHECK-NEXT:          function void append(const char8_t*)
// CHECK-NEXT:          function const char8_t* c_str() const
// CHECK-NEXT:          function unsigned long size() const
// CHECK-NEXT:          function const char8_t& operator [](unsigned long) const
// CHECK-NEXT:          function char8_t& operator [](unsigned long)
// CHECK-NEXT:          class iterator
// CHECK-NEXT:            constructor defaulted void iterator()
// CHECK-NEXT:            constructor defaulted void iterator(const std::basic_string<char8_t>::iterator&)
// CHECK-NEXT:            constructor defaulted void iterator(std::basic_string<char8_t>::iterator&&)
// CHECK-NEXT:            function std::basic_string<char8_t>::iterator& operator ++()
// CHECK-NEXT:            function std::basic_string<char8_t>::iterator operator ++(int)
// CHECK-NEXT:            function char8_t& operator *()
// CHECK-NEXT:            function const char8_t& operator *() const
// CHECK-NEXT:            function bool operator ==(const std::basic_string<char8_t>::iterator&) const
// CHECK-NEXT:            function bool operator !=(const std::basic_string<char8_t>::iterator&) const
// CHECK-NEXT:            function defaulted void ~iterator()
// CHECK-NEXT:          function std::basic_string<char8_t>::iterator begin()
// CHECK-NEXT:          function std::basic_string<char8_t>::iterator end()
// CHECK-NEXT:          function defaulted void ~basic_string()
// CHECK-NEXT:    typealias std::basic_string<char> string
// CHECK-NEXT:    typealias std::basic_string<char8_t> u8string
