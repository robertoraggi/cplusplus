// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>

#include <functional>
#include <regex>
#include <vector>

namespace cxx {

class LookupResult final : public std::vector<Symbol*> {
 public:
  using vector::vector;
};

enum class LookupOptions {
  kNone,
};

class Scope {
 public:
  virtual ~Scope();

  Symbol* owner() const;
  void setOwner(Symbol* owner);

  virtual void add(Symbol* symbol);

  LookupResult find(const Name* name,
                    LookupOptions options = LookupOptions::kNone) const;

  using iterator = std::vector<Symbol*>::const_iterator;

  bool empty() const { return members_.empty(); }

  auto begin() const { return members_.begin(); }
  auto end() const { return members_.end(); }

  auto rbegin() const { return members_.rbegin(); }
  auto rend() const { return members_.rend(); }

 private:
  Symbol* owner_ = nullptr;
  std::vector<Symbol*> members_;
};

}  // namespace cxx
