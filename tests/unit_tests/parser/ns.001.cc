// RUN: %cxx -verify -dump-symbols %s -o - | %filecheck %s

// CHECK: {{^}} - namespace:{{$}}

namespace std {
// CHECK: {{^}}   - namespace: std{{$}}

namespace internal {}
// CHECK: {{^}}     - namespace: internal{{$}}

}  // namespace std

namespace aa::bb::cc {}

// CHECK: {{^}}   - namespace: aa{{$}}
// CHECK: {{^}}     - namespace: bb{{$}}
// CHECK: {{^}}       - namespace: cc{{$}}

namespace xx::yy::zz {
namespace ww {}
}  // namespace xx::yy::zz

// CHECK: {{^}}   - namespace: xx{{$}}
// CHECK: {{^}}     - namespace: yy{{$}}
// CHECK: {{^}}       - namespace: zz{{$}}
// CHECK: {{^}}         - namespace: ww{{$}}
