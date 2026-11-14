// RUN: %cxx -std=c++26 -verify -fsyntax-only %s

constexpr int values[] = {2, 4, 6};
constexpr int past = values[3]; // expected-error {{constexpr variable must be initialized}}
constexpr int reversedPast = 3[values]; // expected-error {{constexpr variable must be initialized}}
constexpr int negative = values[-1]; // expected-error {{constexpr variable must be initialized}}

struct Table {
  inline static constexpr int values[] = {2, 4, 6};
};
constexpr int memberNegative = Table::values[-1]; // expected-error {{constexpr variable must be initialized}}
constexpr int memberPast = Table::values[3]; // expected-error {{constexpr variable must be initialized}}
constexpr int memberReversed = 3[Table::values]; // expected-error {{constexpr variable must be initialized}}
constexpr int nullValue = static_cast<const int*>(nullptr)[0]; // expected-error {{constexpr variable must be initialized}}

constexpr int expired() {
  const int* pointer = nullptr;
  {
    int local[] = {1, 2};
    pointer = local;
  }
  return pointer[0];
}
constexpr int expiredValue = expired(); // expected-error {{constexpr variable must be initialized}}
