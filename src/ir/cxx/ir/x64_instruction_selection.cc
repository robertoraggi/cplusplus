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
#include <cxx/symbols.h>

#if defined(_MSVC_LANG) && !defined(__PRETTY_FUNCTION__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

namespace cxx::ir {

X64InstructionSelection::X64InstructionSelection() {}

X64InstructionSelection::~X64InstructionSelection() {}

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

const std::string& X64InstructionSelection::blockId(ir::Block* block) {
  auto it = labels_.find(block);

  if (it == labels_.end())
    it = labels_.emplace(block, fmt::format(".L{}", labelCount_++)).first;

  return it->second;
}

void X64InstructionSelection::statement(ir::Stmt* stmt) { stmt->accept(this); }

void X64InstructionSelection::expression(ir::Expr* expr, std::string target) {
  std::swap(target_, target);
  expr->accept(this);
  std::swap(target_, target);
}

void X64InstructionSelection::visit(ir::Jump* stmt) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::CondJump* stmt) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Switch* stmt) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Ret* stmt) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::RetVoid* stmt) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Move* stmt) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::This* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::BoolLiteral* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::CharLiteral* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::IntegerLiteral* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::FloatLiteral* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::NullptrLiteral* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::StringLiteral* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::UserDefinedStringLiteral* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Temp* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Id* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::ExternalId* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Typeid* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Unary* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Binary* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Call* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Subscript* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Access* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Cast* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::StaticCast* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::DynamicCast* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::ReinterpretCast* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::New* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::NewArray* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Delete* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::DeleteArray* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void X64InstructionSelection::visit(ir::Throw* expr) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

}  // namespace cxx::ir
