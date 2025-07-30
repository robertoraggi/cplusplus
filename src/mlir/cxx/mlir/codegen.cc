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

#include <cxx/mlir/codegen.h>

// cxx
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>

#include <format>

namespace cxx {

Codegen::Codegen(mlir::MLIRContext& context, TranslationUnit* unit)
    : builder_(&context), unit_(unit) {}

Codegen::~Codegen() {}

auto Codegen::control() const -> Control* { return unit_->control(); }

auto Codegen::currentBlockMightHaveTerminator() -> bool {
  auto block = builder_.getInsertionBlock();
  if (!block) return true;
  return block->mightHaveTerminator();
}

auto Codegen::findOrCreateLocal(Symbol* symbol) -> std::optional<mlir::Value> {
  auto var = symbol_cast<VariableSymbol>(symbol);
  if (!var) return std::nullopt;

  if (auto local = locals_.find(var); local != locals_.end()) {
    return local->second;
  }

  auto type = convertType(var->type());
  auto ptrType = builder_.getType<mlir::cxx::PointerType>(type);

  auto loc = getLocation(var->location());
  auto allocaOp = builder_.create<mlir::cxx::AllocaOp>(loc, ptrType);

  locals_.emplace(var, allocaOp);

  return allocaOp;
}

auto Codegen::getLocation(SourceLocation location) -> mlir::Location {
  auto [filename, line, column] = unit_->tokenStartPosition(location);

  auto loc =
      mlir::FileLineColLoc::get(builder_.getContext(), filename, line, column);

  return loc;
}

auto Codegen::emitTodoStmt(SourceLocation location, std::string_view message)
    -> mlir::cxx::TodoStmtOp {
  const auto loc = getLocation(location);
  auto op = builder_.create<mlir::cxx::TodoStmtOp>(loc, message);
  return op;
}

auto Codegen::emitTodoExpr(SourceLocation location, std::string_view message)
    -> mlir::cxx::TodoExprOp {
  const auto loc = getLocation(location);
  auto op = builder_.create<mlir::cxx::TodoExprOp>(loc, message);
  return op;
}

}  // namespace cxx
