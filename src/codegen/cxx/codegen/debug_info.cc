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

#include <cxx/ast.h>
#include <cxx/codegen/codegen.h>
#include <cxx/codegen/debug_emitter.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

auto Codegen::debugEmitter() -> ir::DebugEmitter* {
  if (debugEmitter_) return debugEmitter_;
  debugEmitter_ = emitter_.debug();
  if (!debugEmitter_) return nullptr;
  debugCompileUnit_ = debugEmitter_->compileUnit(
      {.file = unit_->fileName(),
       .directory = options_.debugCompilationDirectory,
       .isCxx = unit_->language() == LanguageKind::kCXX});
  return debugEmitter_;
}

auto Codegen::debugLocation(SourceLocation loc) -> ir::DebugLocation {
  if (!loc) return {};
  auto [file, line, column] = unit_->tokenStartPosition(loc);
  return {file, line, column};
}

struct Codegen::DebugScopeVisitor {
  Codegen& gen;

  template <typename T>
  [[nodiscard]] auto operator()(T* symbol) -> ir::DebugScopeRef {
    return gen.debugEmitter_->fileScope(
        gen.debugLocation(symbol->location()).file);
  }

  [[nodiscard]] auto operator()(FunctionParametersSymbol* symbol)
      -> ir::DebugScopeRef {
    return gen.getOrCreateDebugScope(symbol->parent());
  }

  [[nodiscard]] auto operator()(BlockSymbol* block) -> ir::DebugScopeRef {
    auto parent = block->parent();
    auto parentScope = gen.getOrCreateDebugScope(parent);
    if (symbol_cast<FunctionParametersSymbol>(parent)) return parentScope;
    if (symbol_cast<FunctionSymbol>(parent)) return parentScope;
    if (!parentScope) return {};
    return gen.debugEmitter_->lexicalBlock(
        parentScope, gen.debugLocation(block->location()));
  }
};

auto Codegen::getOrCreateDebugScope(Symbol* symbol) -> ir::DebugScopeRef {
  if (!symbol) return {};
  if (auto it = debugScopes_.find(symbol); it != debugScopes_.end())
    return it->second;
  auto scope = visit(DebugScopeVisitor{*this}, symbol);
  debugScopes_.emplace(symbol, scope);
  return scope;
}

auto Codegen::debugSubroutineType(const FunctionType* type,
                                  const Type* objectType) -> ir::DebugTypeRef {
  std::vector<ir::DebugTypeRef> signature{convertDebugType(type->returnType())};
  if (objectType)
    signature.push_back(convertDebugType(traits.add_pointer(objectType)));
  for (auto parameter : type->parameterTypes())
    signature.push_back(convertDebugType(parameter));
  return debugEmitter_->subroutineType(signature);
}

void Codegen::attachDebugInfo(ir::ValueRef address, Symbol* symbol,
                              std::string_view name, unsigned arg) {
  if (!debugInfo_ || !function_) return;
  auto debug = debugEmitter();
  if (!debug) return;
  auto scope = getOrCreateDebugScope(symbol->parent());
  if (!scope) return;
  auto sourceName =
      name.empty() ? to_string(symbol->name()) : std::string{name};
  auto type = convertDebugType(symbol->type());
  if (!type)
    cxx_runtime_error(
        std::format("cannot describe the type '{}' of local variable '{}'",
                    to_string(symbol->type()), sourceName));
  debug->localVariable(address, {.name = sourceName,
                                 .scope = scope,
                                 .type = type,
                                 .location = debugLocation(symbol->location()),
                                 .argument = arg});
}

void Codegen::attachDebugInfo(ir::ValueRef address, const Type* type,
                              std::string_view name, unsigned arg) {
  if (!debugInfo_ || !function_) return;
  auto debug = debugEmitter();
  if (!debug) return;
  auto scope = getOrCreateDebugScope(currentFunctionSymbol_);
  if (!scope) return;
  auto debugType = convertDebugType(type);
  if (!debugType)
    cxx_runtime_error(std::format(
        "cannot describe the type '{}' of the object parameter of '{}'",
        to_string(type), to_string(currentFunctionSymbol_->name())));
  debug->localVariable(address, {.name = name,
                                 .scope = scope,
                                 .type = debugType,
                                 .argument = arg,
                                 .isObjectParameter = true});
}

void Codegen::buildSubprogramAttr(FunctionSymbol* symbol,
                                  FunctionDefinitionAST* ast,
                                  ir::FunctionRef function,
                                  SourceLocation loc) {
  auto debug = debugEmitter();
  if (!debug) return;
  auto declarator = getDeclaratorId(ast->declarator);
  auto location = declarator ? debugLocation(declarator->firstSourceLocation())
                             : ir::DebugLocation{};
  auto scopeLine =
      ast->functionBody
          ? debugLocation(ast->functionBody->firstSourceLocation()).line
          : 0;
  if (location.file.empty()) {
    location = debugLocation(symbol->location());
    if (symbol->location()) scopeLine = location.line;
  }
  ir::DebugScopeRef scope;
  const Type* objectType = nullptr;
  if (symbol->isImplicitObjectMemberFunction()) {
    objectType = symbol->parent()->type();
    scope = debug->typeScope(convertDebugType(objectType));
  }
  auto name = to_string(symbol->name());
  debugScopes_[symbol] = debug->defineFunction(
      function, loc,
      {.name = name,
       .scope = scope,
       .type = debugSubroutineType(type_cast<FunctionType>(symbol->type()),
                                   objectType),
       .location = location,
       .scopeLine = scopeLine});
}

}  // namespace cxx
