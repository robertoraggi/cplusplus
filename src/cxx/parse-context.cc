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

#include "parse-context.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cassert>

#include "ast.h"
#include "control.h"
#include "translation-unit.h"

namespace cxx {

static const bool debug = false;

class ParseContext::ProcessName {
  ParseContext* context;

 public:
  ProcessName(ParseContext* context) : context(context) {}

  TranslationUnit* unit() const { return context->unit; }
  Control* control() const { return unit()->control(); }

  const Name* operator()(NameAST* ast) { return process(ast); }

  const Name* process(NameAST* ast) {
    if (!ast) return 0;
    switch (ast->kind()) {
      case ASTKind::kQualifiedName: {
        auto q = ast->asQualifiedName();
        auto base = process(q->base);
        auto name = process(q->name);
        return control()->getQualifiedName(base, name);
      }

      case ASTKind::kPackedName:
        if (debug) fmt::print("todo packed name\n");
        break;

      case ASTKind::kSimpleName:
        return unit()->identifier(ast->asSimpleName()->identifier_token);

      case ASTKind::kDestructorName: {
        auto dtor = ast->asDestructorName();
        auto name = process(dtor->name);
        return control()->getDestructorName(name);
      }

      case ASTKind::kOperatorName: {
        auto op = ast->asOperatorName();
        return control()->getOperatorName(op->op);
      }

      case ASTKind::kTemplateId: {
        auto templ = ast->asTemplateId();
        auto name = process(templ->name);
        std::vector<QualType> args;
        for (auto it = templ->expression_list; it; it = it->next) {
          auto arg = it->value;
          ParseContext::Decl argDecl;
          if (auto type_id = arg->asTypeId()) {
            auto specs = context->specifiers(type_id->specifier_list);
            argDecl = context->declarator(specs, type_id->declarator);
          } else {
            if (debug) fmt::print("todo process constant expression\n");
          }
          args.push_back(argDecl.specs.type);
        }
        return control()->getTemplateName(name, std::move(args));
      }

      case ASTKind::kDecltypeName: {
        auto spec = ast->asDecltypeName();
        if (context->unit->resolveSymbols()) {
          return control()->getDecltypeName(spec->type);
        }
        if (debug) fmt::print("todo decltype name\n");
        break;
      }

      case ASTKind::kDecltypeAutoName:
        if (debug) fmt::print("todo decltype auto name\n");
        break;

      case ASTKind::kConversionFunctionId: {
        auto conv = ast->asConversionFunctionId();
        return control()->getConversionName(conv->type);
      }

      default:
        assert(!"unreachable");
    }  // switch

    return 0;
  }
};

class ParseContext::ProcessDeclarator {
  ParseContext* context;
  ParseContext::Decl _decl;

  TranslationUnit* unit() const { return context->unit; }
  Control* control() const { return unit()->control(); }

  void processSpecifier(SpecifierAST* ast) {
    switch (ast->kind()) {
      case ASTKind::kExceptionSpecification:
        fmt::print("todo exception specification\n");
        break;
      case ASTKind::kAttributeSpecifier:
        fmt::print("todo attribute specifier\n");
        break;
      case ASTKind::kAlignasTypeAttributeSpecifier:
        fmt::print("todo alignas-type specifier\n");
        break;
      case ASTKind::kAlignasAttributeSpecifier:
        fmt::print("todo alignas attribute specifier\n");
        break;

      case ASTKind::kSimpleSpecifier: {
        auto spec = ast->asSimpleSpecifier();
        auto k = unit()->tokenKind(spec->specifier_token);
        switch (k) {
          case TokenKind::T_SIGNED:
            // nothing to do (for now).
            break;
          case TokenKind::T_UNSIGNED:
            _decl.setUnsigned(true);
            break;
          case TokenKind::T_CONST:
            _decl.setConst(true);
            break;
          case TokenKind::T_VOLATILE:
            _decl.setVolatile(true);
            break;
          case TokenKind::T_AUTO:
            _decl.setType(control()->getAutoType());
            break;
          case TokenKind::T___INT64:
            _decl.setType(control()->getLongLongIntType());
            break;
          case TokenKind::T___INT128:
            _decl.setType(control()->getInt128Type());
            break;
          case TokenKind::T___FLOAT80:
            _decl.setType(control()->getFloatType(FloatKind::kLongDouble));
            break;
          case TokenKind::T___FLOAT128:
            _decl.setType(control()->getFloatType(FloatKind::kFloat128));
            break;
          case TokenKind::T_VOID:
            _decl.setType(control()->getVoidType());
            break;
          case TokenKind::T_WCHAR_T:
            _decl.setType(control()->getWCharTType());
            break;
          case TokenKind::T_BOOL:
            _decl.setType(control()->getBoolType());
            break;
          case TokenKind::T_CHAR:
            _decl.setType(control()->getCharType());
            break;
          case TokenKind::T_CHAR16_T:
            _decl.setType(control()->getChar16TType());
            break;
          case TokenKind::T_CHAR32_T:
            _decl.setType(control()->getChar32TType());
            break;
          case TokenKind::T_SHORT:
            _decl.setType(control()->getShortIntType());
            break;
          case TokenKind::T_INT:
            if (!_decl->isIntegerType()) _decl.setType(control()->getIntType());
            break;
          case TokenKind::T_LONG:
            if (_decl->isIntegerType() && _decl->asIntegerType()->isLongInt())
              _decl.setType(control()->getLongLongIntType());
            else
              _decl.setType(control()->getLongIntType());
            break;
          case TokenKind::T_FLOAT:
            _decl.setType(control()->getFloatType(FloatKind::kFloat));
            break;
          case TokenKind::T_DOUBLE:
            if (_decl->isIntegerType() && _decl->asIntegerType()->isLongInt())
              _decl.setType(control()->getFloatType(FloatKind::kLongDouble));
            else
              _decl.setType(control()->getFloatType(FloatKind::kDouble));
            break;
          case TokenKind::T_STATIC:
          case TokenKind::T_EXTERN:
          case TokenKind::T_MUTABLE:
          case TokenKind::T_THREAD_LOCAL:
            _decl.specs.storageSpec = k;
            break;
          case TokenKind::T_INLINE:
            _decl.specs.isInline = true;
            break;
          case TokenKind::T_TYPEDEF:
            _decl.specs.isTypedef = true;
            break;
          case TokenKind::T_VIRTUAL:
            _decl.specs.isVirtual = true;
            break;
          case TokenKind::T_FRIEND:
            _decl.specs.isFriend = true;
            break;
          case TokenKind::T_EXPLICIT:
            _decl.specs.isExplicit = true;
            break;
          case TokenKind::T_CONSTEXPR:
            _decl.specs.isConstexpr = true;
            break;
          default:
            fmt::print("todo simple specifier `{}'\n", Token::spell(k));
            break;
        }  // switch
        break;
      }

      case ASTKind::kNamedSpecifier: {
        auto spec = ast->asNamedSpecifier();
        if (context->unit->resolveSymbols()) {
          _decl.setType(*spec->type);  // ### merge the cv qualifiers
        } else if (auto name = context->name(spec->name)) {
          if (auto declTy = name->asDecltypeName()) {
            _decl.setType(*declTy->type());  // ### merge the cv qualifiers
          } else {
            _decl.setType(control()->getNamedType(name));
          }
        } else {
          fmt::print("todo named specifier\n");
        }
        break;
      }

      case ASTKind::kTypenameSpecifier:
        fmt::print("todo typename specifier\n");
        break;

      case ASTKind::kElaboratedTypeSpecifier: {
        auto spec = ast->asElaboratedTypeSpecifier();
        auto classKey = context->unit->tokenKind(spec->class_key_token);
        auto sym = spec->symbol;
        if (!sym) {
          auto name = context->name(spec->name);
          _decl.setType(
              control()->getElaboratedType(name, classKey));  // ### rename.
          break;
        }
        if (auto klass = sym->asClassSymbol()) {
          _decl.setType(control()->getClassType(klass));
        }
        fmt::print("todo elaborated type specifier\n");
        break;
      }

      case ASTKind::kEnumSpecifier:
        fmt::print("todo enum specifier\n");
        break;

      case ASTKind::kClassSpecifier: {
        auto spec = ast->asClassSpecifier();
        _decl.setType(control()->getClassType(spec->symbol));
        break;
      }

      default:
        assert(!"unreachable");
    }
  }

  void coreDeclarator(CoreDeclaratorAST* ast) {
    if (!ast) return;

    switch (ast->kind()) {
      case ASTKind::kNestedDeclarator:
        _decl = process(_decl.specs, ast->asNestedDeclarator()->declarator);
        break;

      case ASTKind::kDeclaratorId: {
        auto decl = ast->asDeclaratorId();
        auto name = context->name(decl->name);
        _decl.name = name;
        break;
      }

      default:
        assert(!"unreachable");
    }
  }

  void postfixDeclarator(PostfixDeclaratorAST* ast) {
    auto elementType = context->finish(_decl.specs.type);
    switch (ast->kind()) {
      case ASTKind::kArrayDeclarator: {
        const IR::Expr* size{nullptr};
        auto decl = ast->asArrayDeclarator();
        if (decl->size_expression) {
          // ### TODO
        }
        _decl.specs.type = QualType(control()->getArrayType(elementType, size));
        break;
      }

      case ASTKind::kFunctionDeclarator: {
        QualType returnType{context->finish(_decl.specs.type)};
        std::vector<QualType> argTypes;
        std::vector<const Name*> formals;
        auto decl = ast->asFunctionDeclarator();
        if (context->unit->resolveSymbols()) {
          if (decl->trailing_return_type) {
            assert(returnType->isAutoType());
          }
          if (returnType->isAutoType()) {
            assert(decl->trailing_return_type);
            returnType = decl->trailing_return_type;
          }
        }
        if (auto params = decl->parameters_and_qualifiers) {
          for (auto it = params->parameter_list; it; it = it->next) {
            auto param = it->value->asParameterDeclaration();
            auto declTy = context->specifiers(param->specifier_list);
            auto paramDecl = context->declarator(declTy, param->declarator);
            auto argTy = paramDecl.specs.type;
            argTypes.push_back(argTy);
            formals.push_back(paramDecl.name);
          }
        }

        bool isVariadic = false;  // ### TODO
        bool isConst = false;     // ### TODO
        QualType funTy(control()->getFunctionType(returnType, argTypes,
                                                  isVariadic, isConst));
        _decl.specs.type = funTy;
        _decl.formals = std::move(formals);
        break;
      }

      default:
        assert(!"unreachable");
    }
  }

  void ptrOperator(PtrOperatorAST* ast) {
    auto elementType = context->finish(_decl.specs.type);
    switch (ast->op) {
      case TokenKind::T_STAR:
        _decl.specs.type = QualType(control()->getPointerType(elementType));
        break;
      case TokenKind::T_AMP:
        _decl.specs.type =
            QualType(control()->getLValueReferenceType(elementType));
        break;
      case TokenKind::T_AMP_AMP:
        _decl.specs.type =
            QualType(control()->getRValueReferenceType(elementType));
        break;
      default:
        fmt::print("todo ptr operator\n");
        break;
    }  // switch
  }

 public:
  ProcessDeclarator(ParseContext* context) : context(context) {}

  ParseContext::Specs operator()(List<SpecifierAST*>* specifiers) {
    return processSpecifiers(specifiers).specs;
  }

  ParseContext::Decl operator()(const ParseContext::Specs& specs,
                                DeclaratorAST* ast) {
    return process(specs, ast);
  }

  ParseContext::Decl processSpecifiers(List<SpecifierAST*>* specifiers) {
    ParseContext::Decl decl;
    std::swap(_decl, decl);
    for (auto it = specifiers; it; it = it->next) processSpecifier(it->value);
    std::swap(_decl, decl);
    return decl;
  }

  ParseContext::Decl process(const ParseContext::Specs& specs,
                             DeclaratorAST* ast) {
    ParseContext::Decl decl;
    decl.specs = specs;
    if (!ast) return decl;
    std::swap(_decl, decl);
    for (auto it = ast->ptr_op_list; it; it = it->next) ptrOperator(it->value);
    for (auto it = ast->postfix_declarator_list; it; it = it->next)
      postfixDeclarator(it->value);
    coreDeclarator(ast->core_declarator);
    if (_decl.specs.type.isUnsigned() && _decl->isUndefinedType())
      _decl.setType(control()->getIntType());
    std::swap(_decl, decl);
    return decl;
  }
};

const Name* ParseContext::name(NameAST* ast) {
  if (!ast) return 0;
  if (!ast->_name) {
    ProcessName process{this};
    ast->_name = process(ast);
  }
  return ast->_name;
}

ParseContext::Specs ParseContext::specifiers(List<SpecifierAST*>* specifiers) {
  ProcessDeclarator process{this};
  return process(specifiers);
}

ParseContext::Decl ParseContext::declarator(const Specs& specs,
                                            DeclaratorAST* ast) {
  Decl decl;
  decl.specs = specs;
  if (!ast) return decl;
  ProcessDeclarator process{this};
  auto r = process(specs, ast);
  return r;
}

QualType ParseContext::finish(QualType type) {
  if (type->isUndefinedType() && type.isUnsigned()) {
    type.setType(unit->control()->getIntType());
  }
  return type;
}

}  // namespace cxx
