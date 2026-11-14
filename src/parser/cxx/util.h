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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

namespace cxx {

[[nodiscard]] auto align_to(std::uint64_t n, std::uint64_t align)
    -> std::uint64_t;

template <typename T>
void hash_combine(std::size_t& seed, const T& val) {
  seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
class InsertionOrderedSet {
 public:
  InsertionOrderedSet() = default;

  explicit InsertionOrderedSet(std::vector<T> distinctElements)
      : elements_(std::move(distinctElements)) {}

  [[nodiscard]] auto begin() const { return elements_.begin(); }
  [[nodiscard]] auto end() const { return elements_.end(); }
  [[nodiscard]] auto empty() const -> bool { return elements_.empty(); }

  [[nodiscard]] auto contains(const T& element) -> bool {
    if (elements_.size() < kLinearScanLimit)
      return std::ranges::contains(elements_, element);
    if (index_.empty()) index_.insert(elements_.begin(), elements_.end());
    return index_.contains(element);
  }

  auto insert(T element) -> bool {
    if (contains(element)) return false;
    if (!index_.empty()) index_.insert(element);
    elements_.push_back(std::move(element));
    return true;
  }

  [[nodiscard]] auto take() && -> std::vector<T> {
    return std::move(elements_);
  }

 private:
  static constexpr std::size_t kLinearScanLimit = 32;

  std::vector<T> elements_;
  std::unordered_set<T> index_;
};

}  // namespace cxx
