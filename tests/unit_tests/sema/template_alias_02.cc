// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

template <typename Key, typename Value>
struct HashMap {
  using iterator = Value*;
};

struct Name;
struct Symbol;

struct Scope {
  using Table = HashMap<const Name*, Symbol*>;
  using MemberIterator = Table::iterator;
};

// clang-format off
// CHECK:namespace
// CHECK:  template class HashMap
// CHECK:    parameter typename<0, 0> Key
// CHECK:    parameter typename<1, 0> Value
// CHECK:    typealias Value* iterator
// CHECK:    [specializations]
// CHECK:      class HashMap<const ::Name*, ::Symbol*>
// CHECK:        typealias ::Symbol** iterator
// CHECK:  class Name
// CHECK:  class Symbol
// CHECK:  class Scope
// CHECK:    typealias ::HashMap<const ::Name*, ::Symbol*> Table
// CHECK:    typealias ::Symbol** MemberIterator
