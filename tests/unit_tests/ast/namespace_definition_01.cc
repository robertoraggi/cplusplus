// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

namespace {}

inline namespace {}

namespace ns1 {}

inline namespace ns1 {}

namespace n2::n3 {}

namespace n2::inline n4::n5 {}

namespace n2::inline n4::inline n6 {}

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      is-inline: true
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      identifier: ns1
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      identifier: ns1
// CHECK-NEXT:      is-inline: true
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      identifier: n3
// CHECK-NEXT:      nested-namespace-specifier-list
// CHECK-NEXT:        nested-namespace-specifier
// CHECK-NEXT:          identifier: n2
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      identifier: n5
// CHECK-NEXT:      nested-namespace-specifier-list
// CHECK-NEXT:        nested-namespace-specifier
// CHECK-NEXT:          identifier: n2
// CHECK-NEXT:        nested-namespace-specifier
// CHECK-NEXT:          identifier: n4
// CHECK-NEXT:          is-inline: true
// CHECK-NEXT:    namespace-definition
// CHECK-NEXT:      identifier: n6
// CHECK-NEXT:      is-inline: true
// CHECK-NEXT:      nested-namespace-specifier-list
// CHECK-NEXT:        nested-namespace-specifier
// CHECK-NEXT:          identifier: n2
// CHECK-NEXT:        nested-namespace-specifier
// CHECK-NEXT:          identifier: n4
// CHECK-NEXT:          is-inline: true
