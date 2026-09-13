// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <iterator>

namespace cxx::utf8 {

namespace detail {

template <typename It>
constexpr auto octet(It it) -> std::uint8_t {
  return static_cast<std::uint8_t>(*it);
}

constexpr auto isTrailOctet(std::uint8_t value) -> bool {
  return (value & 0xc0) == 0x80;
}

constexpr auto sequenceLength(std::uint8_t lead) -> int {
  if (lead < 0x80) return 1;
  if ((lead >> 5) == 0x06) return 2;
  if ((lead >> 4) == 0x0e) return 3;
  if ((lead >> 3) == 0x1e) return 4;
  return 0;
}

}  // namespace detail

template <typename It>
constexpr auto next(It& it, It last) -> std::uint32_t {
  if (it == last) return 0;

  const auto lead = detail::octet(it);
  const auto length = detail::sequenceLength(lead);

  if (length < 2) {
    ++it;
    return lead;
  }

  auto codePoint = std::uint32_t(lead) & (0x7fu >> length);

  auto cursor = it;
  ++cursor;

  for (int i = 1; i < length; ++i) {
    if (cursor == last) break;

    const auto trail = detail::octet(cursor);
    if (!detail::isTrailOctet(trail)) break;

    codePoint = (codePoint << 6) | (trail & 0x3f);
    ++cursor;

    if (i + 1 == length) {
      it = cursor;
      return codePoint;
    }
  }

  ++it;
  return lead;
}

template <typename It>
constexpr auto peekNext(It it, It last) -> std::uint32_t {
  return next(it, last);
}

template <typename It>
constexpr auto prior(It& it, It first) -> std::uint32_t {
  if (it == first) return 0;

  const auto last = it;

  auto lead = last;
  --lead;

  for (int i = 0; i < 3 && lead != first; ++i) {
    if (!detail::isTrailOctet(detail::octet(lead))) break;
    --lead;
  }

  auto cursor = lead;
  const auto codePoint = next(cursor, last);

  if (cursor == last) {
    it = lead;
    return codePoint;
  }

  --it;
  return detail::octet(it);
}

template <typename It>
constexpr auto distance(It first, It last) ->
    typename std::iterator_traits<It>::difference_type {
  typename std::iterator_traits<It>::difference_type count = 0;

  while (first != last) {
    next(first, last);
    ++count;
  }

  return count;
}

}  // namespace cxx::utf8
