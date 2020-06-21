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

Parser::Assoc Parser::assoc() {
  switch (yytoken()) {
    case T_QUESTION:
    case T_EQUAL:
    case T_PLUS_EQUAL:
    case T_MINUS_EQUAL:
    case T_STAR_EQUAL:
    case T_SLASH_EQUAL:
    case T_PERCENT_EQUAL:
    case T_AMP_EQUAL:
    case T_CARET_EQUAL:
    case T_BAR_EQUAL:
    case T_LESS_LESS_EQUAL:
    case T_GREATER_GREATER_EQUAL:
      return Assoc::Right;

    case T_GREATER:
      if (yytoken(1) == T_GREATER && yytoken(2) == T_EQUAL) return Assoc::Right;
      return Assoc::Left;

    default:
      return Assoc::Left;
  }  // switch
}

int Parser::precedence() {
  switch (yytoken()) {
    case T_DOT_STAR:
    case T_MINUS_GREATER_STAR:
      return 220;

    case T_STAR:
    case T_SLASH:
    case T_PERCENT:
      return 210;

    case T_PLUS:
    case T_MINUS:
      return 200;

    case T_LESS_LESS:
    case T_GREATER_GREATER:
      return 190;

    case T_GREATER:
      if (yytoken(1) == T_GREATER) {
        if (yytoken(2) == T_EQUAL) return 180;  // T_GREATER_GREATER_EQUAL
        return 190;                             // T_GREATER_GREATER
      }
      return 180;  // T_GREATER

    case T_LESS:
    case T_LESS_EQUAL:
    case T_GREATER_EQUAL:
      return 180;

    case T_EQUAL_EQUAL:
    case T_EXCLAIM_EQUAL:
      return 170;

    case T_AMP:
      return 160;

    case T_CARET:
      return 150;

    case T_BAR:
      return 140;

    case T_AMP_AMP:
      return 130;

    case T_BAR_BAR:
      return 120;

    case T_QUESTION:
      return 115;

    case T_EQUAL:
    case T_PLUS_EQUAL:
    case T_MINUS_EQUAL:
    case T_STAR_EQUAL:
    case T_SLASH_EQUAL:
    case T_PERCENT_EQUAL:
    case T_AMP_EQUAL:
    case T_CARET_EQUAL:
    case T_BAR_EQUAL:
    case T_LESS_LESS_EQUAL:
    case T_GREATER_GREATER_EQUAL:
      return 110;

    case T_COMMA:
      return 100;

    default:
      return -1;
  }
}

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

  IR::Module mod;
  module = &mod;
  cg.setModule(module);
  cg.setTranslationUnit(unit);

  globalScope = control->newNamespace();
  globalCode = module->newFunction(nullptr);

  EnterFunction enterFunction;
  enterFunction(this, globalCode);

  scope = globalScope;
  context.unit = unit;

  TranslationUnitAST* ast{nullptr};

  auto parsed = parse_translation_unit(ast);

  if (parsed && yytoken() != T_EOF_SYMBOL) parsed = false;  // expected EOF

  if (!parsed) unit->error(yyparsed, "syntax error");  // ### remove me

  if (consume) consume(ast);

  return parsed;
}

bool Parser::parseBinaryExpression(ParseContext::ExprAttrs& yyast,
                                   bool templArg, int minPrec) {
  if (!parse_cast_expression(yyast)) return false;
  return parseBinaryExpressionHelper(yyast, templArg, minPrec);
}

bool Parser::parseBinaryExpressionHelper(ParseContext::ExprAttrs& yyast,
                                         bool templArg, int minPrec) {
  int prec, nextPrec;
  while (prec = precedence(), prec >= minPrec) {
    auto saved = yycursor;
    auto op = (TokenKind)yytoken();
    if (op == T_GREATER) {
      if (templArg) return true;
      if (yytoken(1) == T_GREATER && yytoken(2) == T_EQUAL)
        yyconsume(), yyconsume(), op = T_GREATER_GREATER_EQUAL;
      else if (yytoken(1) == T_GREATER)
        yyconsume(), op = T_GREATER_GREATER;
      else if (yytoken(1) == T_EQUAL)
        yyconsume(), op = T_GREATER_EQUAL;
    }

    yyconsume();
    unsigned colon_token{0};
    ParseContext::ExprAttrs iftrue_expression{nullptr};
    iftrue_expression.flags = yyast.flags;

    if (op == T_QUESTION) {
      auto parsed = parse_expression(iftrue_expression);
      assert(parsed);
      assert(yytoken() == T_COLON);
      colon_token = yycursor;
      yyconsume();
    }

    ParseContext::ExprAttrs rhs{nullptr};
    rhs.flags = yyast.flags;
    auto e = parse_cast_expression(rhs);
    if (!e) {
      yyrewind(saved);
      return true;
    }

    while (nextPrec = precedence(),
           nextPrec > prec || (nextPrec == prec && assoc() == Assoc::Right)) {
      if (templArg && yytoken() == T_GREATER) break;
      auto pos = yycursor;
      auto parsed = parseBinaryExpressionHelper(rhs, templArg, nextPrec);
      if (yytoken() == T_GREATER && yycursor == pos) return true;
      assert(parsed);
    }

    if (op == T_QUESTION) {
      auto ast = new (pool) ConditionalExpressionAST;
      ast->expression = yyast;
      ast->iftrue_expression = iftrue_expression;
      ast->iffalse_expression = rhs;
      yyast = ast;
    } else {
      auto ast = new (pool) BinaryExpressionAST;
      ast->op = op;
      ast->left_expression = yyast;
      ast->right_expression = rhs;
      yyast = ast;
    }
  }
  return true;
}

bool Parser::implicitCvt(ExpressionAST* ast, const QualType& target) {
  if (ast->type == target)  // identity
    return true;
  // ### TODO: implicit conversions.
  return false;
}

int Parser::compareType(ExpressionAST* source, const QualType& firstTarget,
                        const QualType& secondTarget) {
  if (firstTarget == secondTarget) return 0;
  if (source->type == firstTarget) return -1;
  if (source->type == secondTarget) return +1;
  auto firstConv = implicitCvt(source, firstTarget);
  auto secondConv = implicitCvt(source, secondTarget);
  if (firstConv && !secondConv) return -1;
  if (secondConv && !firstConv) return +1;
  return 0;
}

Symbol* Parser::resolveOverload(Symbol* firstCandidate,
                                List<ExpressionAST*>* actuals) {
  auto bestCandidate = firstCandidate;
  auto functionName = firstCandidate->unqualifiedName();
  size_t argc = 0;
  for (auto it = actuals; it; it = it->next)  // ### TODO remove this loop
    ++argc;
  std::vector<Symbol*> candidates;
  candidates.push_back(bestCandidate);
  for (Symbol* candidate = firstCandidate->next(); candidate;
       candidate = candidate->next()) {
    if (candidate->unqualifiedName() != functionName) continue;
    auto funTy = candidate->type()->asFunctionType();
    if (!funTy) continue;
    auto&& funArgumentTypes = funTy->argumentTypes();
    if (funArgumentTypes.size() != argc)  // ### TODO variadic
      continue;
    auto bestFunTy = bestCandidate->type()->asFunctionType();
    auto&& bestArgumentTypes = bestFunTy->argumentTypes();
    assert(funArgumentTypes.size() ==
           bestArgumentTypes.size());  // ### TODO variadic
    int bestScore = 0, funScore = 0;
    auto it = actuals;
    for (size_t index = 0, end = funArgumentTypes.size(); index != end;
         ++index, it = it->next) {
      assert(it->value);
      auto delta = compareType(it->value, bestArgumentTypes[index],
                               funArgumentTypes[index]);
      if (delta < 0)
        ++bestScore;
      else if (delta > 0)
        ++funScore;
    }
    if (funScore < bestScore) continue;
    if (funScore > bestScore) {
      candidates.clear();
      bestCandidate = candidate;
    }
    candidates.push_back(candidate);
  }
  if (candidates.size() == 1) return candidates.front();
  return nullptr;
}

QualType Parser::unref(const QualType& type, ValueKind* valueKind) {
  if (auto ty = type->asLValueReferenceType()) {
    if (valueKind) *valueKind = ValueKind::kLValue;
    return ty->elementType();
  }

  if (auto ty = type->asRValueReferenceType()) {
    if (valueKind) *valueKind = ValueKind::kXValue;
    return ty->elementType();
  }

  if (valueKind) *valueKind = ValueKind::kPure;

  return type;
}

}  // namespace cxx
