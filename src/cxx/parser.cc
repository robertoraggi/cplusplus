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
#include <iostream>

#include "ast-visitor.h"
#include "ast.h"
#include "codegen.h"
#include "control.h"
#include "ir.h"
#include "names.h"
#include "parse-context.h"
#include "symbols.h"
#include "token.h"
#include "translation-unit.h"
#include "types.h"

namespace cxx {

// pgen generated parser
#include "parser-priv.h"

int Parser::yytoken(int n) { return unit->tokenKind(yycursor + n); }

bool yyparse(TranslationUnit* unit,
             const std::function<void(TranslationUnitAST*)>& consume) {
  Parser p;
  return p.yyparse(unit, consume);
}

bool Parser::yyparse(TranslationUnit* u,
                     const std::function<void(TranslationUnitAST*)>& consume) {
  unit = u;
  control = unit->control();
  yydepth = -1;
  yycursor = 1;
  yyparsed = yycursor;

  Arena arena;
  pool = &arena;

  TranslationUnitAST* ast = nullptr;

  auto parsed = parse_translation_unit();

  if (parsed && yytoken() != TokenKind::T_EOF_SYMBOL)
    parsed = false;  // expected EOF

  if (!parsed) unit->error(yyparsed, "syntax error");  // ### remove me

  if (consume) consume(ast);

  return parsed;
}

}  // namespace cxx
