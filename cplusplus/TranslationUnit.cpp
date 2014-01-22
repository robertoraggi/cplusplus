// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TranslationUnit.h"
#include "Control.h"
#include "Names.h"
#include "KeywordsP.h"
#include "AST.h"
#include "Symbols.h"
#include "Names.h"
#include "Types.h"
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <cassert>

#define printf(a...) do {} while (0)

bool yyparse(TranslationUnit* unit);

inline void TranslationUnit::yyinp() {
  if (yychar == '\n')
    lines_.push_back(yypos);
  yychar = *yyptr++;
  ++yypos;
}

TokenKind TranslationUnit::yylex(unsigned* offset, const void** priv) {
again:
  while (isspace(yychar))
    yyinp();
  *offset = yypos;
  *priv = 0;
  if (yychar == 0)
    return T_EOF_SYMBOL;
  auto ch = yychar;
  yyinp();

  // ### TODO: wide and unicode literals
  if (ch == 'L' && (yychar == '\'' || yychar == '"')) {
    ch = yychar;
    yyinp();
  } else if (toupper(ch) == 'U' && (yychar == '\'' || yychar == '"')) {
    ch = yychar;
    yyinp();
  }


  switch (ch) {
  case '\'':
  case '"': {
    const auto quote = ch;
    yytext = quote;
    while (yychar) {
      if (yychar == quote)
        break;
      else if (yychar == '\\') {
        yytext += yychar;
        yyinp();
        if (yychar) {
          yytext += yychar;
          yyinp();
        }
      } else {
        yytext += yychar;
        yyinp();
      }
    }
    assert(yychar == quote);
    yytext += yychar;
    yyinp();
    *priv = control_->getIdentifier(yytext);
    return quote == '"' ? T_STRING_LITERAL : T_CHAR_LITERAL;
  }

  case '=':
    if (yychar == '=') {
      yyinp();
      return T_EQUAL_EQUAL;
    }
    return T_EQUAL;

  case ',':
    return T_COMMA;

  case '~':
    if (yychar == '=') {
      yyinp();
      return T_TILDE_EQUAL;
    }
    return T_TILDE;

  case '{':
    return T_LBRACE;

  case '}':
    return T_RBRACE;

  case '[':
    return T_LBRACKET;

  case ']':
    return T_RBRACKET;

  case '#':
    while (auto ch = yychar) {
      yyinp();
      if (ch == '\n')
        break;
      if (ch == '\\' && yychar)
        yyinp();
    }
    goto again;
#if 0
    if (yychar == '#') {
      yyinp();
      return T_POUND_POUND;
    }
    return T_POUND;
#endif

  case '(':
    return T_LPAREN;

  case ')':
    return T_RPAREN;

  case ';':
    return T_SEMICOLON;

  case ':':
    if (yychar == ':') {
      yyinp();
      return T_COLON_COLON;
    }
    return T_COLON;

  case '.':
    if (yychar == '.') {
      yyinp();
      if (yychar == '.') {
        yyinp();
        return T_DOT_DOT_DOT;
      }
      return T_ERROR;
    } else if (yychar == '*') {
      yyinp();
      return T_DOT_STAR;
    }
    return T_DOT;

  case '?':
    return T_QUESTION;

  case '*':
    if (yychar == '=') {
      yyinp();
      return T_STAR_EQUAL;
    }
    return T_STAR;

  case '%':
    if (yychar == '=') {
      yyinp();
      return T_PERCENT_EQUAL;
    }
    return T_PERCENT;

  case '^':
    if (yychar == '=') {
      yyinp();
      return T_CARET_EQUAL;
    }
    return T_CARET;

  case '&':
    if (yychar == '=') {
      yyinp();
      return T_AMP_EQUAL;
    } else if (yychar == '&') {
      yyinp();
      return T_AMP_AMP;
    }
    return T_AMP;

  case '|':
    if (yychar == '=') {
      yyinp();
      return T_BAR_EQUAL;
    } else if (yychar == '|') {
      yyinp();
      return T_BAR_BAR;
    }
    return T_BAR;

  case '!':
    if (yychar == '=') {
      yyinp();
      return T_EXCLAIM_EQUAL;
    }
    return T_EXCLAIM;

  case '+':
    if (yychar == '+') {
      yyinp();
      return T_PLUS_PLUS;
    } else if (yychar == '=') {
      yyinp();
      return T_PLUS_EQUAL;
    }
    return T_PLUS;

  case '-':
    if (yychar == '-') {
      yyinp();
      return T_MINUS_MINUS;
    } else if (yychar == '=') {
      yyinp();
      return T_MINUS_EQUAL;
    } else if (yychar == '>') {
      yyinp();
      if (yychar == '*') {
        yyinp();
        return T_MINUS_GREATER_STAR;
      } else {
        return T_MINUS_GREATER;
      }
    }
    return T_MINUS;

  case '<':
    if (yychar == '=') {
      yyinp();
      return T_LESS_EQUAL;
    } else if (yychar == '<') {
      yyinp();
      if (yychar == '=') {
        yyinp();
        return T_LESS_LESS_EQUAL;
      }
      return T_LESS_LESS;
    }
    return T_LESS;

  case '>':
#if 0
    if (yychar == '=') {
      yyinp();
      return T_GREATER_EQUAL;
    } else if (yychar == '>') {
      yyinp();
      if (yychar == '=') {
        yyinp();
        return T_GREATER_GREATER_EQUAL;
      }
      return T_GREATER_GREATER;
    }
#endif
    return T_GREATER;

  case '/':
    if (yychar == '/') {
      yyinp();
      while (yychar && yychar != '\n')
        yyinp();
      goto again;
    } else if (yychar == '*') {
      yyinp();
      while (yychar) {
        if (yychar == '*') {
          yyinp();
          if (yychar == '/') {
            yyinp();
            goto again;
          }
        } else {
          yyinp();
        }
      }
      assert(!"unexpected end of file");
    } else if (yychar == '=') {
      yyinp();
      return T_SLASH_EQUAL;
    }
    return T_SLASH;

  default:
    if (std::isalpha(ch) || ch == '_' || ch == '$') {
      yytext = ch;
      while (std::isalnum(yychar) || yychar == '_' || yychar == '$') {
        yytext += yychar;
        yyinp();
      }

      auto k = classify(yytext.c_str(), yytext.size());
      if (k != T_IDENTIFIER)
        return (TokenKind) k;

      *priv = control_->getIdentifier(yytext);
      return T_IDENTIFIER;
    } else if (std::isdigit(ch)) {
      yytext = ch;
      while (std::isalnum(yychar) || yychar == '.') {
        yytext += yychar;
        yyinp();
      }

      *priv = control_->getIdentifier(yytext);
      return T_INT_LITERAL;
    }
  } // switch
  return T_ERROR;
}

void TranslationUnit::warning(unsigned index, const char* format...) {
  unsigned line, column;
  getTokenStartPosition(index, &line, &column);
  fprintf(stderr, "%s:%d:%d: warning: ", yyfilename.c_str(), line, column);
  va_list args, ap;
  va_start(args, format);
  va_copy(ap, args);
  vfprintf(stderr, format, args);
  va_end(ap);
  va_end(args);
  fprintf(stderr, "\n");
}

void TranslationUnit::error(unsigned index, const char* format...) {
  unsigned line, column;
  getTokenStartPosition(index, &line, &column);
  fprintf(stderr, "%s:%d:%d: error: ", yyfilename.c_str(), line, column);
  va_list args, ap;
  va_start(args, format);
  va_copy(ap, args);
  vfprintf(stderr, format, args);
  va_end(ap);
  va_end(args);
  fprintf(stderr, "\n");
}

void TranslationUnit::fatal(unsigned index, const char* format...) {
  unsigned line, column;
  getTokenStartPosition(index, &line, &column);
  fprintf(stderr, "%s:%d:%d: fatal: ", yyfilename.c_str(), line, column);
  va_list args, ap;
  va_start(args, format);
  va_copy(ap, args);
  vfprintf(stderr, format, args);
  va_end(ap);
  va_end(args);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

int TranslationUnit::tokenLength(unsigned index) const
{
  auto&& tk = tokens_[index];
  if (tk.kind() == T_IDENTIFIER) {
    const std::string* id = reinterpret_cast<const std::string*>(tk.priv_);
    return id->size();
  }
  return ::strlen(token_spell[tk.kind()]);
}

const char* TranslationUnit::tokenText(unsigned index) const {
  auto&& tk = tokens_[index];
  switch (tk.kind()) {
  case T_IDENTIFIER:
  case T_STRING_LITERAL:
  case T_CHAR_LITERAL:
  case T_INT_LITERAL: {
    const Identifier* id = reinterpret_cast<const Identifier*>(tk.priv_);
    return id->c_str();
  }
  default:
    return token_spell[tk.kind()];
  }
}

const Identifier* TranslationUnit::identifier(unsigned index) const {
  if (! index)
    return 0;
  auto&& tk = tokens_[index];
  return reinterpret_cast<const Identifier*>(tk.priv_);
}

void TranslationUnit::getTokenStartPosition(unsigned index, unsigned* line, unsigned* column) const {
  auto offset = tokens_[index].offset();
  auto it = std::lower_bound(lines_.cbegin(), lines_.cend(), offset);
  if (it != lines_.cbegin()) {
    --it;
    assert(*it <= offset);
    *line = std::distance(lines_.cbegin(), it) + 1;
    *column = offset - *it;
  } else {
    *line = 1;
    *column = offset + 1;
  }
}

void TranslationUnit::tokenize() {
  TokenKind kind;
  tokens_.emplace_back(T_ERROR, 0, nullptr);
  do {
    unsigned offset = 0;
    const void* value = 0;
    kind = yylex(&offset, &value);
    tokens_.emplace_back(kind, offset, value);
  } while (kind != T_EOF_SYMBOL);
}

bool TranslationUnit::parse() {
  return yyparse(this);
}

namespace {
struct ProcessName {
  TranslationUnit* unit;

  ProcessName(TranslationUnit* unit): unit(unit) {
  }

  const Name* operator()(NameAST* ast) {
    return process(ast);
  }

  const Name* process(NameAST* ast) {
    if (! ast)
      return 0;
    switch (ast->kind()) {
    case ASTKind::kQualifiedName: {
      auto q = ast->asQualifiedName();
      auto base = process(q->base);
      auto name = process(q->name);
      return unit->control()->getQualifiedName(base, name);
    }

    case ASTKind::kPackedName:
      printf("todo packed name\n");
      break;

    case ASTKind::kSimpleName:
      return unit->identifier(ast->asSimpleName()->identifier_token);

    case ASTKind::kDestructorName: {
      auto dtor = ast->asDestructorName();
      auto name = process(dtor->name);
      return unit->control()->getDestructorName(name);
    }

    case ASTKind::kOperatorName: {
      auto op = ast->asOperatorName();
      return unit->control()->getOperatorName(unit->tokenKind(op->op_token));
    }

    case ASTKind::kTemplateId: {
      auto templ = ast->asTemplateId();
      auto name = process(templ->name);
      std::vector<QualType> args;
      for (auto it = templ->expression_list; it; it = it->next) {
        auto arg = it->value;
        TranslationUnit::Decl argDecl;
        if (auto type_id = arg->asTypeId()) {
          auto specs = unit->specifiers(type_id->specifier_list);
          argDecl = unit->declarator(specs, type_id->declarator);
        } else {
          printf("todo process constant expression\n");
        }
        args.push_back(argDecl.specs.type);
      }
      return unit->control()->getTemplateName(name, std::move(args));
    }

    case ASTKind::kDecltypeName:
      printf("todo decltype name\n");
      break;

    case ASTKind::kDecltypeAutoName:
      printf("todo decltype auto name\n");
      break;

    default:
      assert(!"unreachable");
    } // switch

    return 0;
  }
};

class ProcessDeclarator {
  void processSpecifier(SpecifierAST* ast) {
    switch (ast->kind()) {
    case ASTKind::kExceptionSpecification:
      printf("todo exception specification\n");
      break;
    case ASTKind::kAttributeSpecifier:
      printf("todo attribute specifier\n");
      break;
    case ASTKind::kAlignasTypeAttributeSpecifier:
      printf("todo alignas-type specifier\n");
      break;
    case ASTKind::kAlignasAttributeSpecifier:
      printf("todo alignas attribute specifier\n");
      break;

    case ASTKind::kSimpleSpecifier: {
      auto k = unit->tokenKind(ast->asSimpleSpecifier()->specifier_token);
      switch (k) {
      case T_SIGNED:
        // nothing to do (for now).
        break;
      case T_UNSIGNED:
        _decl.setUnsigned(true);
        break;
      case T_CONST:
        _decl.setConst(true);
        break;
      case T_VOLATILE:
        _decl.setVolatile(true);
        break;
      case T_AUTO:
        _decl.setType(control->getAutoType());
        break;
      case T_VOID:
        _decl.setType(control->getVoidType());
        break;
      case T_WCHAR_T:
        _decl.setType(control->getWCharTType());
        break;
      case T_BOOL:
        _decl.setType(control->getBoolType());
        break;
      case T_CHAR:
        _decl.setType(control->getCharType());
        break;
      case T_CHAR16_T:
        _decl.setType(control->getChar16TType());
        break;
      case T_CHAR32_T:
        _decl.setType(control->getChar32TType());
        break;
      case T_SHORT:
        _decl.setType(control->getShortIntType());
        break;
      case T_INT:
        if (! _decl->isIntegerType())
          _decl.setType(control->getIntType());
        break;
      case T_LONG:
        if (_decl->isIntegerType() && _decl->asIntegerType()->isLongInt())
          _decl.setType(control->getLongLongIntType());
        else
          _decl.setType(control->getLongIntType());
        break;
      case T_FLOAT:
        _decl.setType(control->getFloatType(FloatKind::kFloat));
        break;
      case T_DOUBLE:
        if (_decl->isIntegerType() && _decl->asIntegerType()->isLongInt())
          _decl.setType(control->getFloatType(FloatKind::kLongDouble));
        else
          _decl.setType(control->getFloatType(FloatKind::kDouble));
        break;
      case T_EXTERN:
        _decl.specs.isExtern = true;
        break;
      case T_INLINE:
        _decl.specs.isInline = true;
        break;
      case T_STATIC:
        _decl.specs.isStatic = true;
        break;
      case T_TYPEDEF:
        _decl.specs.isTypedef = true;
        break;
      case T_VIRTUAL:
        _decl.specs.isVirtual = true;
        break;
      case T_FRIEND:
        _decl.specs.isFriend = true;
        break;
      case T_EXPLICIT:
        _decl.specs.isExplicit = true;
        break;
      case T_MUTABLE:
        _decl.specs.isMutable = true;
        break;
      case T_CONSTEXPR:
        _decl.specs.isConstexpr = true;
        break;
      default:
        printf("todo simple specifier `%s'\n", token_spell[k]);
        break;
      } // switch
      break;
    }

    case ASTKind::kNamedSpecifier: {
      auto spec = ast->asNamedSpecifier();
      auto name = unit->name(spec->name);
      if (name)
        _decl.setType(control->getNamedType(name));
      else
        printf("todo named specifier\n");
      break;
    }

    case ASTKind::kTypenameSpecifier:
      printf("todo typename specifier\n");
      break;
    case ASTKind::kElaboratedTypeSpecifier:
      printf("todo elaborated specifier\n");
      break;
    case ASTKind::kEnumSpecifier:
      printf("todo enum specifier\n");
      break;
    case ASTKind::kClassSpecifier: {
      auto spec = ast->asClassSpecifier();
      _decl.setType(control->getClassType(spec->symbol));
      break;
    }

    default:
      assert(!"unreachable");
    }
  }

  void coreDeclarator(CoreDeclaratorAST* ast) {
    if (! ast)
      return;

    switch (ast->kind()) {
    case ASTKind::kNestedDeclarator:
      _decl = process(_decl.specs, ast->asNestedDeclarator()->declarator);
      break;

    case ASTKind::kDeclaratorId: {
      auto decl = ast->asDeclaratorId();
      auto name = unit->name(decl->name);
      _decl.name = name;
      break;
    }

    default:
      assert(!"unreachable");
    }
  }

  void postfixDeclarator(PostfixDeclaratorAST* ast) {
    auto elementType = unit->finish(_decl.specs.type);
    switch (ast->kind()) {
    case ASTKind::kArrayDeclarator: {
      auto decl = ast->asArrayDeclarator();
      if (! decl->size_expression) {
        _decl.specs.type = QualType(control->getUnboundedArrayType(elementType));
      } else {
        printf("todo array size_expression\n");
        size_t size = 0;
        _decl.specs.type = QualType(control->getBoundedArrayType(elementType, size));
      }
      break;
    }

    case ASTKind::kFunctionDeclarator: {
      auto decl = ast->asFunctionDeclarator();
      auto fun = control->newFunction();
      fun->setReturnType(unit->finish(_decl.specs.type));
      if (auto params = decl->parameters_and_qualifiers) {
        for (auto it = params->parameter_list; it; it = it->next) {
          auto param = it->value->asParameterDeclaration();
          auto declTy = unit->specifiers(param->specifier_list);
          auto paramDecl = unit->declarator(declTy, param->declarator);
          auto arg = control->newArgument();
          arg->setEnclosingScope(fun);
          arg->setName(paramDecl.name);
          arg->setType(paramDecl.specs.type);
          fun->addArgument(arg);
        }
      }
      _decl.specs.type = QualType(control->getFunctionType(fun));
      break;
    }

    default:
      assert(!"unreachable");
    }
  }

  void ptrOperator(PtrOperatorAST* ast) {
    auto elementType = unit->finish(_decl.specs.type);
    switch (ast->op) {
    case T_STAR:
      _decl.specs.type = QualType(control->getPointerType(elementType));
      break;
    case T_AMP:
      _decl.specs.type = QualType(control->getLValueReferenceType(elementType));
      break;
    case T_AMP_AMP:
      _decl.specs.type = QualType(control->getRValueReferenceType(elementType));
      break;
    default:
      printf("todo ptr operator\n");
      break;
    } // switch
  }

public:
  ProcessDeclarator(TranslationUnit* unit): unit(unit) {
    control = unit->control();
  }

  TranslationUnit::Specs operator()(List<SpecifierAST*>* specifiers) {
    return processSpecifiers(specifiers).specs;
  }

  TranslationUnit::Decl operator()(const TranslationUnit::Specs& specs,
                                   DeclaratorAST* ast) {
    return process(specs, ast);
  }

  TranslationUnit::Decl processSpecifiers(List<SpecifierAST*>* specifiers) {
    TranslationUnit::Decl decl;
    std::swap(_decl, decl);
    for (auto it = specifiers; it; it = it->next)
      processSpecifier(it->value);
    std::swap(_decl, decl);
    return decl;
  }

  TranslationUnit::Decl process(const TranslationUnit::Specs& specs,
                                DeclaratorAST* ast) {
    TranslationUnit::Decl decl;
    decl.specs = specs;
    if (! ast)
      return decl;
    std::swap(_decl, decl);
    for (auto it = ast->ptr_op_list; it; it = it->next)
      ptrOperator(it->value);
    for (auto it = ast->postfix_declarator_list; it; it = it->next)
      postfixDeclarator(it->value);
    coreDeclarator(ast->core_declarator);
    if (_decl.specs.type.isUnsigned() && _decl->isUndefinedType())
      _decl.setType(control->getIntType());
    std::swap(_decl, decl);
    return decl;
  }

private:
  TranslationUnit* unit;
  Control* control;
  TranslationUnit::Decl _decl;
};

} // anonymous namespace

const Name* TranslationUnit::name(NameAST* ast) {
  if (! ast)
    return 0;
  if (! ast->_name) {
    ProcessName process{this};
    ast->_name = process(ast);
  }
  return ast->_name;
}

TranslationUnit::Specs TranslationUnit::specifiers(List<SpecifierAST*>* specifiers) {
  ProcessDeclarator process{this};
  return process(specifiers);
}

TranslationUnit::Decl TranslationUnit::declarator(const Specs& specs, DeclaratorAST* ast) {
  Decl decl;
  decl.specs = specs;
  if (! ast)
    return decl;
  if (! ast->_type) {
    ProcessDeclarator process{this};
    auto r = process(specs, ast);
    ast->_type = r.specs.type; // ### wrong
    ast->_name = r.name;
  }
  decl.name = ast->_name;
  decl.specs.type = ast->_type; // ### wrong
  return decl;
}

QualType TranslationUnit::finish(QualType type) {
  if (type->isUndefinedType() && type.isUnsigned()) {
    type.setType(control()->getIntType());
  }
  return type;
}
