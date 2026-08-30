
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

#include <cxx/symbols_fwd.h>
#include <cxx/token_fwd.h>
#include <cxx/types_fwd.h>

#include <functional>
#include <variant>
#include <vector>

namespace cxx {

class Parser;

struct ScopeCompletionContext {
  ScopeSymbol* scope = nullptr;
  ScopeSymbol* accessingScope = nullptr;
};

struct UnqualifiedCompletionContext {
  ScopeSymbol* scope = nullptr;
};

struct MemberCompletionContext {
  const Type* objectType = nullptr;
  TokenKind accessOp = TokenKind::T_DOT;
  ScopeSymbol* accessingScope = nullptr;
};

struct DesignatorCompletionContext {
  const Type* objectType = nullptr;
  ScopeSymbol* accessingScope = nullptr;
};

struct ArgumentHintsContext {
  std::vector<FunctionSymbol*> candidates;
  int activeParameter = 0;
};

struct TemplateArgumentHintsContext {
  Symbol* templateSymbol = nullptr;
  int activeParameter = 0;
};

using CodeCompletionContext =
    std::variant<ScopeCompletionContext, UnqualifiedCompletionContext,
                 MemberCompletionContext, DesignatorCompletionContext,
                 ArgumentHintsContext, TemplateArgumentHintsContext>;

struct CanContinueParsing {};

struct ParsingComplete {};

using ParsingState = std::variant<CanContinueParsing, ParsingComplete>;

struct ParserConfiguration {
  bool checkTypes = false;
  bool validateAst = false;
  bool allowUnprototypedFunctions = false;
  std::function<bool()> stopParsingPredicate;
  std::function<void(const CodeCompletionContext&)> complete;
};

}  // namespace cxx
