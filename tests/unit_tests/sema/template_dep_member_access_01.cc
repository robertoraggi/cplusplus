// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class T>
struct base_traits;

template <class T, class Traits = base_traits<T>>
class base_stream;

template <class T, class Traits>
class save_state {
  typedef base_stream<T, Traits> stream_type;

  stream_type& stream_;

 public:
  explicit save_state(stream_type& s) : stream_(s) {
    (void)stream_.get_state();
  }

  ~save_state() { stream_.set_state(0); }
};
