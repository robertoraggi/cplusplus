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

#include <cxx/decl.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {

namespace {

class GetFunctionPrototype {
  enum class Kind { Direct, Ptr, Function, Array };

  FunctionDeclaratorChunkAST* prototype_ = nullptr;
  Kind kind_ = Kind::Direct;

 public:
  GetFunctionPrototype() = default;

  FunctionDeclaratorChunkAST* operator()(DeclaratorAST* declarator) {
    makeDirect();
    process(declarator);
    return prototype_;
  }

 private:
  void makeDirect() {
    prototype_ = nullptr;
    kind_ = Kind::Direct;
  }

  void makePtr() {
    prototype_ = nullptr;
    kind_ = Kind::Ptr;
  }

  void makeArray() {
    prototype_ = nullptr;
    kind_ = Kind::Array;
  }

  void makeFunction(FunctionDeclaratorChunkAST* prototype) {
    prototype_ = prototype;
    kind_ = Kind::Function;
  }

  void process(DeclaratorAST* declarator) {
    if (!declarator) return;
    if (declarator->ptrOpList) makePtr();
    if (declarator->declaratorChunkList) {
      auto prototype = ast_cast<FunctionDeclaratorChunkAST>(
          declarator->declaratorChunkList->value);
      if (prototype) makeFunction(prototype);
    }
    auto nested = ast_cast<NestedDeclaratorAST>(declarator->coreDeclarator);
    if (nested) process(nested->declarator);
  }
};

struct GetDeclaratorType {
  TranslationUnit* unit = nullptr;
  const Type* type_ = nullptr;

  struct {
    auto operator()(float value) const -> std::optional<std::size_t> {
      return std::nullopt;
    }

    auto operator()(double value) const -> std::optional<std::size_t> {
      return std::nullopt;
    }

    auto operator()(const StringLiteral* value) const
        -> std::optional<std::size_t> {
      return std::nullopt;
    }

    template <typename T>
    auto operator()(T value) const -> std::optional<std::size_t> {
      return static_cast<std::size_t>(value);
    }
  } get_size_value;

  explicit GetDeclaratorType(TranslationUnit* unit) : unit(unit) {}

  auto control() const -> Control* { return unit->control(); }

  auto operator()(DeclaratorAST* ast, const Type* type) -> const Type* {
    if (!ast) return type;

    std::swap(type_, type);

    std::invoke(*this, ast);

    std::swap(type_, type);

    return type;
  }

  void operator()(DeclaratorAST* ast) {
    for (auto it = ast->ptrOpList; it; it = it->next) visit(*this, it->value);

    auto nestedNameSpecifier = getNestedNameSpecifier(ast->coreDeclarator);

    std::invoke(*this, ast->declaratorChunkList);

    if (ast->coreDeclarator) visit(*this, ast->coreDeclarator);
  }

  auto getNestedNameSpecifier(CoreDeclaratorAST* ast) const
      -> NestedNameSpecifierAST* {
    struct {
      auto operator()(DeclaratorAST* ast) const -> NestedNameSpecifierAST* {
        if (ast->coreDeclarator) return visit(*this, ast->coreDeclarator);
        return nullptr;
      }

      auto operator()(BitfieldDeclaratorAST* ast) const
          -> NestedNameSpecifierAST* {
        return nullptr;
      }

      auto operator()(ParameterPackAST* ast) const -> NestedNameSpecifierAST* {
        return nullptr;
      }

      auto operator()(IdDeclaratorAST* ast) const -> NestedNameSpecifierAST* {
        return ast->nestedNameSpecifier;
      }

      auto operator()(NestedDeclaratorAST* ast) const
          -> NestedNameSpecifierAST* {
        if (!ast->declarator) return nullptr;
        return std::invoke(*this, ast->declarator);
      }
    } v;

    if (!ast) return nullptr;
    return visit(v, ast);
  }

  void operator()(List<DeclaratorChunkAST*>* chunks) {
    if (!chunks) return;
    std::invoke(*this, chunks->next);
    visit(*this, chunks->value);
  }

  void operator()(PointerOperatorAST* ast) {
    type_ = control()->getPointerType(type_);

    for (auto it = ast->cvQualifierList; it; it = it->next) {
      if (ast_cast<ConstQualifierAST>(it->value)) {
        type_ = control()->add_const(type_);
      } else if (ast_cast<VolatileQualifierAST>(it->value)) {
        type_ = control()->add_volatile(type_);
      }
    }
  }

  void operator()(ReferenceOperatorAST* ast) {
    if (ast->refOp == TokenKind::T_AMP_AMP) {
      type_ = control()->add_rvalue_reference(type_);
    } else {
      type_ = control()->add_lvalue_reference(type_);
    }
  }

  void operator()(PtrToMemberOperatorAST* ast) {
    if (!type_) return;

    auto symbol = ast->nestedNameSpecifier->symbol;
    if (!symbol) return;

    auto classType = type_cast<ClassType>(symbol->type());
    if (!classType) return;

    if (auto functionType = type_cast<FunctionType>(type_)) {
      type_ = control()->getMemberFunctionPointerType(classType, functionType);
    } else {
      type_ = control()->getMemberObjectPointerType(classType, type_);
    }

    for (auto it = ast->cvQualifierList; it; it = it->next) {
      if (ast_cast<ConstQualifierAST>(it->value)) {
        type_ = control()->add_const(type_);
      } else if (ast_cast<VolatileQualifierAST>(it->value)) {
        type_ = control()->add_volatile(type_);
      }
    }
  }

  void operator()(BitfieldDeclaratorAST* ast) {}

  void operator()(ParameterPackAST* ast) {
    if (ast->coreDeclarator) visit(*this, ast->coreDeclarator);
  }

  void operator()(IdDeclaratorAST* ast) {}

  void operator()(NestedDeclaratorAST* ast) {
    std::invoke(*this, ast->declarator);
  }

  void operator()(FunctionDeclaratorChunkAST* ast) {
    auto returnType = type_;
    std::vector<const Type*> parameterTypes;
    bool isVariadic = false;

    if (auto params = ast->parameterDeclarationClause) {
      for (auto it = params->parameterDeclarationList; it; it = it->next) {
        auto paramType = it->value->type;
        parameterTypes.push_back(paramType);
      }

      isVariadic = params->isVariadic;
    }

    CvQualifiers cvQualifiers = CvQualifiers::kNone;
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      if (ast_cast<ConstQualifierAST>(it->value)) {
        if (cvQualifiers == CvQualifiers::kVolatile)
          cvQualifiers = CvQualifiers::kConstVolatile;
        else
          cvQualifiers = CvQualifiers::kConst;
      } else if (ast_cast<VolatileQualifierAST>(it->value)) {
        if (cvQualifiers == CvQualifiers::kConst)
          cvQualifiers = CvQualifiers::kConstVolatile;
        else
          cvQualifiers = CvQualifiers::kVolatile;
      }
    }

    RefQualifier refQualifier = RefQualifier::kNone;

    if (ast->refLoc) {
      if (unit->tokenKind(ast->refLoc) == TokenKind::T_AMP_AMP) {
        refQualifier = RefQualifier::kRvalue;
      } else {
        refQualifier = RefQualifier::kLvalue;
      }
    }

    bool isNoexcept = false;

    if (ast->exceptionSpecifier)
      isNoexcept = visit(*this, ast->exceptionSpecifier);

    if (ast->trailingReturnType) {
#if false
        if (!type_cast<AutoType>(returnType)) {
          p->parse_warn(ast->trailingReturnType->firstSourceLocation(),
                        std::format("function with trailing return type must "
                                    "be declared with 'auto', not '{}'",
                                    to_string(returnType)));
        }
#endif

      if (ast->trailingReturnType->typeId) {
        returnType = ast->trailingReturnType->typeId->type;
      }
    }

    type_ = control()->getFunctionType(returnType, std::move(parameterTypes),
                                       isVariadic, cvQualifiers, refQualifier,
                                       isNoexcept);
  }

  void operator()(ArrayDeclaratorChunkAST* ast) {
    if (!ast->expression) {
      type_ = control()->getUnboundedArrayType(type_);
      return;
    }

    ASTInterpreter sem{unit};
    auto constValue = sem.evaluate(ast->expression);

    if (constValue) {
      if (auto size = std::visit(get_size_value, *constValue)) {
        type_ = control()->getBoundedArrayType(type_, *size);
        return;
      }
    }

    type_ =
        control()->getUnresolvedBoundedArrayType(unit, type_, ast->expression);
  }

  auto operator()(ThrowExceptionSpecifierAST* ast) -> bool { return false; }

  auto operator()(NoexceptSpecifierAST* ast) -> bool {
    if (!ast->expression) return true;
    return false;
  }
};

}  // namespace

[[nodiscard]] auto getFunctionPrototype(DeclaratorAST* declarator)
    -> FunctionDeclaratorChunkAST* {
  GetFunctionPrototype prototype;
  return prototype(declarator);
}

[[nodiscard]] auto getDeclaratorType(TranslationUnit* unit,
                                     DeclaratorAST* declarator,
                                     const Type* type) -> const Type* {
  GetDeclaratorType getDeclaratorType{unit};
  return getDeclaratorType(declarator, type);
}

[[nodiscard]] auto getDeclaratorId(DeclaratorAST* declarator)
    -> IdDeclaratorAST* {
  if (!declarator) return nullptr;

  if (auto id = ast_cast<IdDeclaratorAST>(declarator->coreDeclarator)) {
    return id;
  }

  if (auto nested = ast_cast<NestedDeclaratorAST>(declarator->coreDeclarator)) {
    return getDeclaratorId(nested->declarator);
  }

  return nullptr;
}

Decl::Decl(const DeclSpecs& specs, DeclaratorAST* declarator) : specs{specs} {
  declaratorId = getDeclaratorId(declarator);
}

auto Decl::location() const -> SourceLocation {
  if (declaratorId) return declaratorId->firstSourceLocation();
  return {};
}

auto Decl::getName() const -> const Name* {
  auto control = specs.control();
  if (!declaratorId) return nullptr;
  if (!declaratorId->unqualifiedId) return nullptr;
  return get_name(control, declaratorId->unqualifiedId);
}

auto Decl::getNestedNameSpecifier() const -> NestedNameSpecifierAST* {
  if (!declaratorId) return nullptr;
  return declaratorId->nestedNameSpecifier;
}

auto Decl::getScope() const -> Scope* {
  auto nestedNameSpecifier = getNestedNameSpecifier();
  if (!nestedNameSpecifier) return nullptr;

  auto symbol = nestedNameSpecifier->symbol;
  if (!symbol) return nullptr;

  if (auto alias = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType = type_cast<ClassType>(alias->type())) {
      symbol = classType->symbol();
    }
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol))
    return classSymbol->scope();

  if (auto namespaceSymbol = symbol_cast<NamespaceSymbol>(symbol))
    return namespaceSymbol->scope();

  return nullptr;
}

}  // namespace cxx