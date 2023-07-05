// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

namespace cxx {

template <typename T>
class LinkedList {
 public:
  LinkedList() { zero(); }

  void zero() {
    front_ = nullptr;
    back_ = nullptr;
  }

  auto empty() const -> bool { return front_ == nullptr; }
  auto front() const -> T* { return front_; }
  auto back() const -> T* { return back_; }

  void push_front(T* element) { insert(front_, element); }
  void push_back(T* element) { insert(nullptr, element); }

  void insert(T* next, T* element) {
    if (next != nullptr) {
      element->prev = next->prev;
      element->next = next;
      next->prev = element;

      if (element->prev != nullptr) {
        element->prev->next = element;
      } else {
        front_ = element;
      }
    } else {
      element->prev = back_;
      if (back_ != nullptr) {
        back_->next = element;
      }
      if (front_ == nullptr) {
        front_ = element;
      }
      back_ = element;
    }
  }

  void erase(T* element) {
    if (element->prev != nullptr) {
      element->prev->next = element->next;
    }
    if (element->next != nullptr) {
      element->next->prev = element->prev;
    }
    if (front_ == element) {
      front_ = element->next;
    }
    if (back_ == element) {
      back_ = element->prev;
    }
    element->prev = nullptr;
    element->next = nullptr;
  }

 private:
  T* front_;
  T* back_;
};

}  // namespace cxx
