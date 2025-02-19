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

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

namespace cxx {

class TranslationUnit;

class TypeChecker {
 public:
  explicit TypeChecker(TranslationUnit* unit);

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto reportErrors() const -> bool { return reportErrors_; }
  void setReportErrors(bool reportErrors) { reportErrors_ = reportErrors; }

  void setScope(Scope* scope) { scope_ = scope; }

  void operator()(ExpressionAST* ast);

  void check(ExpressionAST* ast);

  [[nodiscard]] auto ensure_prvalue(ExpressionAST*& expr) -> bool;

  [[nodiscard]] auto implicit_conversion(ExpressionAST*& expr,
                                         const Type* destinationType) -> bool;

 private:
  struct Visitor;

  TranslationUnit* unit_;
  Scope* scope_ = nullptr;
  bool reportErrors_ = false;
};

}  // namespace cxx
