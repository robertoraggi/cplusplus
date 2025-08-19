// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

template <typename Key, typename Value>
struct HashMap {
  using iterator = Value*;
};

struct Name;
struct Symbol;

struct ScopeSymbol {
  using Table = HashMap<const Name*, Symbol*>;
  using MemberIterator = Table::iterator;
};

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class HashMap<type-param<0, 0>, type-param<1, 0>>
// CHECK-NEXT:    parameter typename<0, 0> Key
// CHECK-NEXT:    parameter typename<1, 0> Value
// CHECK-NEXT:    typealias type-param<1, 0>* iterator
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class HashMap<const ::Name*, ::Symbol*>
// CHECK-NEXT:        typealias ::Symbol** iterator
// CHECK-NEXT:  class Name
// CHECK-NEXT:  class Symbol
// CHECK-NEXT:  class ScopeSymbol
// CHECK-NEXT:    typealias ::HashMap<const ::Name*, ::Symbol*> Table
// CHECK-NEXT:    typealias ::Symbol** MemberIterator
