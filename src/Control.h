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

#include <forward_list>
#include <set>

#include "Globals.h"
#include "Names.h"
#include "Symbols.h"
#include "Types.h"

class Control {
 public:
  Control(const Control& other) = delete;
  Control& operator=(const Control& other) = delete;

  Control();
  ~Control();

#define VISIT_TYPE(T) \
  const T##Type* get##T##Type() { return T##Type::get(); }
  FOR_EACH_SINGLETON_TYPE(VISIT_TYPE)
#undef VISIT_TYPE

#define VISIT_TYPE(T, X)                      \
  const IntegerType* get##T##Type() {         \
    return getIntegerType(IntegerKind::k##T); \
  }
  FOR_EACH_INTEGER_TYPE(VISIT_TYPE)
#undef VISIT_TYPE

  template <typename T>
  struct Table final : std::set<T> {
    template <typename... Args>
    const T* operator()(Args&&... args) {
      return &*this->emplace(std::forward<Args>(args)...).first;
    }
  };

  template <typename T>
  struct Sequence final : std::forward_list<T> {
    template <typename... Args>
    T* operator()(Args&&... args) {
      return &*this->emplace_after(this->before_begin(),
                                   std::forward<Args>(args)...);
    }
  };

#define VISIT_TYPE(T) Table<T##Type> get##T##Type;
  FOR_EACH_OTHER_TYPE(VISIT_TYPE)
#undef VISIT_TYPE

#define VISIT_NAME(T) Table<T> get##T;
  FOR_EACH_NAME(VISIT_NAME)
#undef VISIT_NAME

#define VISIT_SYMBOL(T) Sequence<T##Symbol> new##T;
  FOR_EACH_SYMBOL(VISIT_SYMBOL)
#undef VISIT_SYMBOL
};
