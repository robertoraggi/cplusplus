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

#include <cxx/names.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/symbols.h>
#include <cxx/util.h>

#include <cstring>

namespace cxx {

namespace {

struct ConstValueHash {
  auto operator()(bool value) const -> std::size_t {
    return std::hash<bool>{}(value);
  }
  auto operator()(std::int32_t value) const -> std::size_t {
    return std::hash<std::int32_t>{}(value);
  }
  auto operator()(std::uint32_t value) const -> std::size_t {
    return std::hash<std::uint32_t>{}(value);
  }
  auto operator()(std::int64_t value) const -> std::size_t {
    return std::hash<std::int64_t>{}(value);
  }
  auto operator()(std::uint64_t value) const -> std::size_t {
    return std::hash<std::uint64_t>{}(value);
  }
  auto operator()(float value) const -> std::size_t {
    return std::hash<float>{}(value);
  }
  auto operator()(double value) const -> std::size_t {
    return std::hash<double>{}(value);
  }
  auto operator()(long double value) const -> std::size_t {
    return std::hash<long double>{}(value);
  }
  auto operator()(const StringLiteral* value) const -> std::size_t {
    return value->hashCode();
  }
};

struct TemplateArgumentHash {
  auto operator()(const Type* arg) const -> std::size_t {
    return std::hash<const void*>{}(arg);
  }

  auto operator()(Symbol* arg) const -> std::size_t {
    return arg->name() ? arg->name()->hashValue() : 0;
  }

  auto operator()(ConstValue arg) const -> std::size_t {
    return std::visit(ConstValueHash{}, arg);
  }

  auto operator()(ExpressionAST* arg) const -> std::size_t {
    return std::hash<const void*>{}(arg);
  }
};

struct ConvertToName {
  Control* control_;

  explicit ConvertToName(Control* control) : control_(control) {}

  auto operator()(NameIdAST* ast) const -> const Name* {
    return ast->identifier;
  }

  auto operator()(DestructorIdAST* ast) const -> const Name* {
    return control_->getDestructorId(visit(*this, ast->id));
  }

  auto operator()(DecltypeIdAST* ast) const -> const Name* {
    cxx_runtime_error("DecltypeIdAST not implemented");
    return {};
  }

  auto operator()(OperatorFunctionIdAST* ast) const -> const Name* {
    return control_->getOperatorId(ast->op);
  }

  auto operator()(LiteralOperatorIdAST* ast) const -> const Name* {
    if (ast->identifier)
      return control_->getLiteralOperatorId(ast->identifier->name());

    auto value = ast->literal->value();
    auto suffix = value.substr(value.find_last_of('"') + 1);
    return control_->getLiteralOperatorId(suffix);
  }

  auto operator()(ConversionFunctionIdAST* ast) const -> const Name* {
    return control_->getConversionFunctionId(ast->typeId->type);
  }

  auto operator()(SimpleTemplateIdAST* ast) const -> const Name* {
    std::vector<TemplateArgument> arguments;
    return control_->getTemplateId(ast->identifier, std::move(arguments));
  }

  auto operator()(LiteralOperatorTemplateIdAST* ast) const -> const Name* {
    std::vector<TemplateArgument> arguments;
    return control_->getTemplateId(operator()(ast->literalOperatorId),
                                   std::move(arguments));
  }

  auto operator()(OperatorFunctionTemplateIdAST* ast) const -> const Name* {
    std::vector<TemplateArgument> arguments;
    return control_->getTemplateId(operator()(ast->operatorFunctionId),
                                   std::move(arguments));
  }
};

}  // namespace

[[nodiscard]] auto get_name(Control* control, UnqualifiedIdAST* id)
    -> const Name* {
  if (!id) return nullptr;
  return visit(ConvertToName{control}, id);
}

Name::~Name() = default;

auto Identifier::isBuiltinTypeTrait() const -> bool {
  return builtinTypeTrait() != BuiltinTypeTraitKind::T_NONE;
}

auto Identifier::builtinTypeTrait() const -> BuiltinTypeTraitKind {
  if (!info_) return BuiltinTypeTraitKind::T_NONE;

  if (info_->kind() != IdentifierInfoKind::kTypeTrait)
    return BuiltinTypeTraitKind::T_NONE;

  return static_cast<const TypeTraitIdentifierInfo*>(info_)->trait();
}

auto TemplateId::hash(const Name* name,
                      const std::vector<TemplateArgument>& args)
    -> std::size_t {
  std::size_t hash = name->hashValue();
  for (const auto& arg : args) {
    hash_combine(hash, std::visit(TemplateArgumentHash{}, arg));
  }
  return hash;
}

}  // namespace cxx
