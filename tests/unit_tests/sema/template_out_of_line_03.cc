// RUN: %cxx -verify -fcheck %s

template <typename K, typename V>
struct Map {
  struct Entry {
    K key;
    V value;
  };

  Entry entry_;

  auto key() const -> K;
  auto value() const -> V;
  void set(const K& k, const V& v);
};

template <typename K, typename V>
auto Map<K, V>::key() const -> K {
  return entry_.key;
}

template <typename K, typename V>
auto Map<K, V>::value() const -> V {
  return entry_.value;
}

template <typename K, typename V>
void Map<K, V>::set(const K& k, const V& v) {
  entry_.key = k;
  entry_.value = v;
}

auto main() -> int {
  Map<int, double> m;
  m.set(1, 3.14);
  auto k = m.key();
  auto v = m.value();
  return 0;
}
