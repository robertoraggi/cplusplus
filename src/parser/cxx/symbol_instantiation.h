// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/types_fwd.h>

#include <unordered_map>
#include <vector>

namespace cxx {

class TranslationUnit;
class Control;

class SymbolInstantiation {
 public:
  explicit SymbolInstantiation(TranslationUnit* unit,
                               const std::vector<TemplateArgument>& arguments);
  ~SymbolInstantiation();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto control() const -> Control*;

  [[nodiscard]] auto operator()(Symbol* symbol) -> Symbol*;

  [[nodiscard]] auto findOrCreateReplacement(Symbol* symbol) -> Symbol*;

  template <typename S>
  [[nodiscard]] auto replacement(S* symbol) -> S* {
    return static_cast<S*>(findOrCreateReplacement(symbol));
  }

 private:
  struct MakeSymbol;
  struct VisitSymbol;
  struct VisitType;

  template <typename S>
  [[nodiscard]] auto instantiate(S* symbol) -> S* {
    return static_cast<S*>(instantiateHelper(symbol));
  }

  [[nodiscard]] auto instantiateHelper(Symbol* symbol) -> Symbol*;

 private:
  TranslationUnit* unit_ = nullptr;
  const std::vector<TemplateArgument>& arguments_;
  std::unordered_map<Symbol*, Symbol*> replacements_;
  Symbol* current_ = nullptr;
};

}  // namespace cxx