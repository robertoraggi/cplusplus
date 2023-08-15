// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines
// --strict-whitespace

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
// CHECK-NEXT:      is-inline: false
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      is-inline: true
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      namespace-name: ns1
// CHECK-NEXT:      is-inline: false
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      namespace-name: ns1
// CHECK-NEXT:      is-inline: true
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      nested-namespace-specifier-list
// CHECK-NEXT:        nested-namespace-specifier
// CHECK-NEXT:          namespace-name: n2
// CHECK-NEXT:          is-inline: false
// CHECK-NEXT:      namespace-name: n3
// CHECK-NEXT:      is-inline: false
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      nested-namespace-specifier-list
// CHECK-NEXT:        nested-namespace-specifier
// CHECK-NEXT:          namespace-name: n2
// CHECK-NEXT:          is-inline: false
// CHECK-NEXT:        nested-namespace-specifier
// CHECK-NEXT:          namespace-name: n4
// CHECK-NEXT:          is-inline: true
// CHECK-NEXT:      namespace-name: n5
// CHECK-NEXT:      is-inline: false
