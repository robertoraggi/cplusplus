// RUN: %cxx --target=arm64-apple-macosx14.0.0 -emit-llvm %s -o - | %filecheck %s

template <class T>
auto above(T t)
    -> decltype(t + 1.000000000000000111022302462515654042363166809082031251L) {
  return t;
}

template <class T>
auto below(T t)
    -> decltype(t + 1.0000000000000001110223024625156540423631668090820312499L) {
  return t;
}

template auto above<long double>(long double) -> long double;
template auto below<long double>(long double) -> long double;

// CHECK-DAG: define {{.*}}@_Z5aboveIeEDTplfp_Le3ff0000000000001EET_(
// CHECK-DAG: define {{.*}}@_Z5belowIeEDTplfp_Le3ff0000000000000EET_(
