// Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <functional>
#include <set>
#include <string>
#include <tuple>

#include "cxx-fwd.h"

namespace cxx {

class Name {
 public:
  virtual ~Name() = default;
  virtual size_t hashCode() const = 0;
};

class Identifier : public Name, public std::tuple<std::string> {
  mutable size_t hashCode_ = static_cast<size_t>(-1);

 public:
  using tuple::tuple;

  const std::string& toString() const { return std::get<0>(*this); }

  size_t hashCode() const {
    if (hashCode_ != static_cast<size_t>(-1)) return hashCode_;
    hashCode_ = std::hash<std::string>()(toString());
    return hashCode_;
  }
};

class Control {
 public:
  Control(const Control& other) = delete;
  Control& operator=(const Control& other) = delete;

  Control();
  ~Control();

  template <typename T>
  const Identifier* getIdentifier(T&& name) {
    return &*identifiers_.emplace(std::forward<T>(name)).first;
  }

 private:
  std::set<Identifier> identifiers_;
};

}  // namespace cxx
