// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ir/x64_instruction_selection.h>

// cxx
#include <cxx/ir/ir.h>
#include <cxx/name_printer.h>
#include <cxx/names.h>
#include <cxx/private/format.h>
#include <cxx/symbols.h>

#if defined(_MSVC_LANG) && !defined(__PRETTY_FUNCTION__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

namespace cxx::ir {

X64InstructionSelection::X64InstructionSelection() = default;

X64InstructionSelection::~X64InstructionSelection() = default;

void X64InstructionSelection::operator()(ir::Module* module,
                                         std::ostream& out) {
  auto outP = &out;

  std::map<ir::Block*, std::string> labels;

  std::swap(labels_, labels);
  std::swap(out_, outP);

  for (auto function : module->functions()) {
    // ### TODO: name mangling. Assuming c-linkage for now.
    auto name = fmt::format("{}", *function->symbol()->name());

    emit("\t.text\n");
    emit("\t.globl {}\n", name);

    emit("{}:\n", name);
    emit("\tpushq\t%rbp\n");
    emit("\tmovq\t%rsp, %rbp\n");

    for (auto block : function->blocks()) {
      emit("{}:\n", blockId(block));
    }

    emit("\tleave\n");
    emit("\tret\n");
    emit("\n");
  }

  std::swap(out_, outP);
  std::swap(labels_, labels);
}

auto X64InstructionSelection::blockId(ir::Block* block) -> const std::string& {
  auto it = labels_.find(block);

  if (it == labels_.end()) {
    it = labels_.emplace(block, fmt::format(".L{}", labelCount_++)).first;
  }

  return it->second;
}

void X64InstructionSelection::statement(ir::Stmt* stmt) { stmt->accept(this); }

void X64InstructionSelection::expression(ir::Expr* expr, std::string target) {
  std::swap(target_, target);
  expr->accept(this);
  std::swap(target_, target);
}

void X64InstructionSelection::visit(ir::Jump* stmt) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::CondJump* stmt) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Switch* stmt) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Ret* stmt) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::RetVoid* stmt) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Move* stmt) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::This* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::BoolLiteral* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::CharLiteral* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::IntegerLiteral* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::FloatLiteral* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::NullptrLiteral* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::StringLiteral* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::UserDefinedStringLiteral* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Temp* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Id* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::ExternalId* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Typeid* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Unary* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Binary* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Call* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Subscript* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Access* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Cast* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::StaticCast* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::DynamicCast* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::ReinterpretCast* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::New* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::NewArray* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Delete* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::DeleteArray* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Throw* expr) {
  cxx_runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

}  // namespace cxx::ir
