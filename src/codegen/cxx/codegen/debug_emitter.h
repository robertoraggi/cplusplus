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
#include <cxx/codegen/emitter_handles.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <string_view>
namespace cxx::ir {
class DebugEmitter {
 public:
  virtual ~DebugEmitter() = default;
  virtual void defineFunction(FunctionSymbol* symbol, FunctionRef function,
                              SourceLocation loc, SourceLocation declaratorLoc,
                              SourceLocation bodyLoc) = 0;
  virtual void localVariable(ValueRef address, Symbol* symbol,
                             std::string_view name, unsigned arg) = 0;
  virtual void objectParameter(ValueRef address, const Type* type,
                               FunctionSymbol* function, std::string_view name,
                               unsigned arg) = 0;
};
}  // namespace cxx::ir
