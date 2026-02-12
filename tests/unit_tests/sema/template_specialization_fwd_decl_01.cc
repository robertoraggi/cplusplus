// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T>
struct reference_wrapper;

struct true_type {
  static constexpr bool value = true;
};
struct false_type {
  static constexpr bool value = false;
};

template <class>
struct __is_identity : false_type {};

struct __identity {};
struct identity {};

template <>
struct __is_identity<__identity> : true_type {};
template <>
struct __is_identity<reference_wrapper<__identity>> : true_type {};
template <>
struct __is_identity<reference_wrapper<const __identity>> : true_type {};

template <>
struct __is_identity<identity> : true_type {};
template <>
struct __is_identity<reference_wrapper<identity>> : true_type {};
template <>
struct __is_identity<reference_wrapper<const identity>> : true_type {};

template <class T>
struct reference_wrapper {
  T* ptr;
};

static_assert(__is_identity<__identity>::value, "identity");
static_assert(__is_identity<identity>::value, "identity");
static_assert(!__is_identity<int>::value, "not identity");

int main() {
  __identity id;
  identity id2;
  reference_wrapper<__identity> rw_id;
  reference_wrapper<const __identity> rw_cid;
  reference_wrapper<identity> rw_id2;
  reference_wrapper<const identity> rw_cid2;
  rw_id.ptr = &id;
  rw_cid.ptr = &id;
  rw_id2.ptr = &id2;
  rw_cid2.ptr = &id2;
  return 0;
}
