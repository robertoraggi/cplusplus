// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines --strict-whitespace

namespace {}

inline namespace {}

namespace ns1 {}

inline namespace ns1 {}

namespace n2::n3 {}

namespace n2::inline n4::n5 {}

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      name: simple-name
// CHECK-NEXT:        identifier: ns1
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      name: simple-name
// CHECK-NEXT:        identifier: ns1
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:    namespace-definition
