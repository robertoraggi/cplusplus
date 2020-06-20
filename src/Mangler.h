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

#ifndef MANGLER_H
#define MANGLER_H

#include <string>

#include "Globals.h"

//
// mangler
//
class Mangler {
 public:
  std::string operator()(FunctionSymbol* symbol) { return encode(symbol); }

  std::string encode(FunctionSymbol* symbol);

 private:
  std::string mangleName(const Name* name, const QualType& type);
  std::string mangleType(const QualType& type);
  std::string mangleBareFunctionType(const FunctionType* funTy);

#define VISIT_NAME(T) std::string mangle##T(const T*, const QualType& type);
  FOR_EACH_NAME(VISIT_NAME)
#undef VISIT_NAME

#define VISIT_TYPE(T) std::string mangle##T##Type(const T##Type*);
  FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE
};

#endif  // MANGLER_H
