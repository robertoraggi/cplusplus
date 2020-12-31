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

#pragma once

#include <cxx/names.h>

#include <set>

namespace cxx {

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

  template <typename T>
  const DestructorId* getDestructorId(T&& name) {
    return &*destructorIds_.emplace(std::forward<T>(name)).first;
  }

  template <typename T>
  const OperatorId* getOperatorId(T&& name) {
    return &*operatorIds_.emplace(std::forward<T>(name)).first;
  }

  template <typename... Args>
  const TemplateId* getTemplateId(Args&&... args) {
    return &*templateIds_.emplace(std::forward<Args>(args)...).first;
  }

 private:
  std::set<Identifier> identifiers_;
  std::set<DestructorId> destructorIds_;
  std::set<OperatorId> operatorIds_;
  std::set<TemplateId> templateIds_;
};

}  // namespace cxx
