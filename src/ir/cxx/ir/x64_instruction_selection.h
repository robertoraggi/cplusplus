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

#pragma once

#include <cxx/ir/ir_visitor.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <map>

namespace cxx::ir {

class X64InstructionSelection : ir::IRVisitor {
 public:
  X64InstructionSelection();
  virtual ~X64InstructionSelection();

  void operator()(ir::Module* module, std::ostream& out);

 private:
  template <typename... Args>
  void emit(const std::string_view& format, const Args&... args) {
    fmt::vprint(*out_, fmt::to_string_view(format),
                fmt::make_format_args(args...));
  }

  void statement(ir::Stmt* stmt);

  void expression(ir::Expr* expr, std::string target);

  const std::string& blockId(ir::Block* block);

  void visit(ir::Jump*) override;
  void visit(ir::CondJump*) override;
  void visit(ir::Switch*) override;
  void visit(ir::Ret*) override;
  void visit(ir::RetVoid*) override;
  void visit(ir::Move*) override;

  void visit(ir::This*) override;
  void visit(ir::BoolLiteral*) override;
  void visit(ir::CharLiteral*) override;
  void visit(ir::IntegerLiteral*) override;
  void visit(ir::FloatLiteral*) override;
  void visit(ir::NullptrLiteral*) override;
  void visit(ir::StringLiteral*) override;
  void visit(ir::UserDefinedStringLiteral*) override;
  void visit(ir::Temp*) override;
  void visit(ir::Id*) override;
  void visit(ir::ExternalId*) override;
  void visit(ir::Typeid*) override;
  void visit(ir::Unary*) override;
  void visit(ir::Binary*) override;
  void visit(ir::Call*) override;
  void visit(ir::Subscript*) override;
  void visit(ir::Access*) override;
  void visit(ir::Cast*) override;
  void visit(ir::StaticCast*) override;
  void visit(ir::DynamicCast*) override;
  void visit(ir::ReinterpretCast*) override;
  void visit(ir::New*) override;
  void visit(ir::NewArray*) override;
  void visit(ir::Delete*) override;
  void visit(ir::DeleteArray*) override;
  void visit(ir::Throw*) override;

 private:
  std::ostream* out_ = nullptr;
  std::string target_;
  std::map<ir::Block*, std::string> labels_;
  uint32_t labelCount_ = 0;
};

}  // namespace cxx::ir
