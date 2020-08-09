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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <forward_list>
#include <iostream>
#include <unordered_map>
#include <variant>

#include "arena.h"
#include "control.h"
#include "token.h"
#include "translation-unit.h"

namespace cxx {

// pgen generated parser
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4102)
#endif

#include "parser-priv.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

TokenKind Parser::yytoken(int n) { return unit->tokenKind(yycursor + n); }

bool yyparse(TranslationUnit* unit, const std::function<void()>& consume) {
  Parser p;
  return p.yyparse(unit, consume);
}

bool Parser::yyparse(TranslationUnit* u, const std::function<void()>& consume) {
  unit = u;
  control = unit->control();
  yydepth = -1;
  yycursor = 1;
  yyparsed = yycursor;

  Arena arena;
  pool = &arena;

  module_id = control->getIdentifier("module");
  import_id = control->getIdentifier("import");
  final_id = control->getIdentifier("final");
  override_id = control->getIdentifier("override");

  auto parsed = parse_translation_unit();

  if (consume) consume();

  return parsed;
}

}  // namespace cxx
