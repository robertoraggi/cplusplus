// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ir/ir_fwd.h>
#include <cxx/literals_fwd.h>
#include <cxx/qualified_type.h>
#include <cxx/symbols_fwd.h>

#include <list>
#include <memory>
#include <string>
#include <vector>

namespace cxx::ir {

class Module final {
 public:
  Module(const Module&) = delete;
  auto operator=(const Module&) -> Module& = delete;

  Module();
  ~Module();

  [[nodiscard]] auto functions() const -> const std::list<Function*>&;
  void addFunction(Function* function);

  [[nodiscard]] auto globals() const -> const std::list<Global*>&;
  void addGlobal(Global* global);

  auto irFactory() -> IRFactory*;

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

class Global final {
 public:
  explicit Global(Module* module, Symbol* symbol)
      : module_(module), symbol_(symbol) {}

  [[nodiscard]] auto module() const -> Module* { return module_; }
  [[nodiscard]] auto symbol() const -> Symbol* { return symbol_; }

 private:
  Module* module_;
  Symbol* symbol_;
};

class Local final {
 public:
  Local(const QualifiedType& type, int index) : type_(type), index_(index) {}

  [[nodiscard]] auto type() const -> const QualifiedType& { return type_; }
  [[nodiscard]] auto index() const -> int { return index_; }

 private:
  QualifiedType type_;
  int index_;
};

class Function final {
 public:
  Function(Module* module, FunctionSymbol* symbol)
      : module_(module), symbol_(symbol) {}

  [[nodiscard]] auto module() const -> Module* { return module_; }
  [[nodiscard]] auto symbol() const -> FunctionSymbol* { return symbol_; }

  [[nodiscard]] auto blocks() const -> const std::list<Block*>& {
    return blocks_;
  }

  void addBlock(Block* block) { blocks_.push_back(block); }

  [[nodiscard]] auto locals() const -> const std::list<Local>& {
    return locals_;
  }

  auto addLocal(const QualifiedType& type) -> Local*;

 private:
  Module* module_;
  FunctionSymbol* symbol_;
  std::list<Block*> blocks_;
  std::list<Local> locals_;
};

class Block final {
 public:
  explicit Block(Function* function) : function_(function) {}

  [[nodiscard]] auto function() const -> Function* { return function_; }

  auto code() -> std::list<Stmt*>& { return code_; }
  [[nodiscard]] auto code() const -> const std::list<Stmt*>& { return code_; }

  [[nodiscard]] auto id() const -> int;

  [[nodiscard]] auto hasTerminator() const -> bool;

 private:
  Function* function_;
  std::list<Stmt*> code_;
};

class Stmt {
 public:
  Stmt() = default;
  virtual ~Stmt() = default;

  virtual void accept(IRVisitor* visitor) = 0;

  [[nodiscard]] virtual auto isTerminator() const -> bool { return false; }
};

class Expr : public Stmt {
 public:
  Expr() = default;
};

class Move final : public Stmt {
 public:
  Move(Expr* target, Expr* source) : target_(target), source_(source) {}

  [[nodiscard]] auto target() const -> Expr* { return target_; }
  [[nodiscard]] auto source() const -> Expr* { return source_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* target_;
  Expr* source_;
};

class Jump final : public Stmt {
 public:
  explicit Jump(Block* target) : target_(target) {}

  [[nodiscard]] auto target() const -> Block* { return target_; }

  [[nodiscard]] auto isTerminator() const -> bool override { return true; }

  void accept(IRVisitor* visitor) override;

 private:
  Block* target_ = nullptr;
};

class CondJump final : public Stmt {
 public:
  CondJump(Expr* condition, Block* iftrue, Block* iffalse)
      : condition_(condition), iftrue_(iftrue), iffalse_(iffalse) {}

  [[nodiscard]] auto condition() const -> Expr* { return condition_; }
  [[nodiscard]] auto iftrue() const -> Block* { return iftrue_; }
  [[nodiscard]] auto iffalse() const -> Block* { return iffalse_; }

  [[nodiscard]] auto isTerminator() const -> bool override { return true; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* condition_;
  Block* iftrue_;
  Block* iffalse_;
};

class Switch final : public Stmt {
 public:
  using Case = std::tuple<Expr*, Block*>;

  explicit Switch(Expr* condition) : condition_(condition) {}

  [[nodiscard]] auto condition() const -> Expr* { return condition_; }

  auto defaultBlock() -> Block* { return defaultBlock_; }
  void setDefaultBlock(Block* defaultBlock) { defaultBlock_ = defaultBlock; }

  [[nodiscard]] auto cases() const -> const std::vector<Case>& {
    return cases_;
  }
  void addCase(const Case& caseStmt) { cases_.push_back(caseStmt); }

  [[nodiscard]] auto isTerminator() const -> bool override { return true; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* condition_;
  Block* defaultBlock_ = nullptr;
  std::vector<Case> cases_;
};

class Ret final : public Stmt {
 public:
  explicit Ret(Expr* result) : result_(result) {}

  [[nodiscard]] auto result() const -> Expr* { return result_; }

  [[nodiscard]] auto isTerminator() const -> bool override { return true; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* result_;
};

class RetVoid final : public Stmt {
 public:
  RetVoid() = default;

  [[nodiscard]] auto isTerminator() const -> bool override { return true; }

  void accept(IRVisitor* visitor) override;
};

class This final : public Expr {
 public:
  explicit This(const QualifiedType& type) : type_(type) {}

  [[nodiscard]] auto type() const -> const QualifiedType& { return type_; }

  void accept(IRVisitor* visitor) override;

 private:
  QualifiedType type_;
};

class BoolLiteral final : public Expr {
 public:
  explicit BoolLiteral(bool value) : value_(value) {}

  [[nodiscard]] auto value() const -> bool { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  bool value_;
};

class CharLiteral final : public Expr {
 public:
  explicit CharLiteral(const cxx::CharLiteral* value) : value_(value) {}

  [[nodiscard]] auto value() const -> const cxx::CharLiteral* { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  const cxx::CharLiteral* value_;
};

class IntegerLiteral final : public Expr {
 public:
  explicit IntegerLiteral(const IntegerValue& value) : value_(value) {}

  auto value() -> IntegerValue { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  IntegerValue value_;
};

class FloatLiteral final : public Expr {
 public:
  explicit FloatLiteral(const FloatValue& value) : value_(value) {}

  [[nodiscard]] auto value() const -> FloatValue { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  FloatValue value_;
};

class NullptrLiteral final : public Expr {
 public:
  NullptrLiteral() = default;

  void accept(IRVisitor* visitor) override;
};

class StringLiteral final : public Expr {
 public:
  explicit StringLiteral(const cxx::StringLiteral* value) : value_(value) {}

  [[nodiscard]] auto value() const -> const cxx::StringLiteral* {
    return value_;
  }

  void accept(IRVisitor* visitor) override;

 private:
  const cxx::StringLiteral* value_;
};

class UserDefinedStringLiteral final : public Expr {
 public:
  explicit UserDefinedStringLiteral(std::string value)
      : value_(std::move(value)) {}

  [[nodiscard]] auto value() const -> const std::string& { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  std::string value_;
};

class Temp final : public Expr {
 public:
  explicit Temp(Local* local) : local_(local) {}

  [[nodiscard]] auto local() const -> Local* { return local_; }

  void accept(IRVisitor* visitor) override;

 private:
  Local* local_;
};

class Id final : public Expr {
 public:
  explicit Id(Symbol* symbol) : symbol_(symbol) {}

  [[nodiscard]] auto symbol() const -> Symbol* { return symbol_; }

  void accept(IRVisitor* visitor) override;

 private:
  Symbol* symbol_;
};

class ExternalId final : public Expr {
 public:
  explicit ExternalId(std::string name) : name_(std::move(name)) {}

  [[nodiscard]] auto name() const -> const std::string& { return name_; }

  void accept(IRVisitor* visitor) override;

 private:
  std::string name_;
};

class Typeid final : public Expr {
 public:
  explicit Typeid(Expr* expr) : expr_(expr) {}

  [[nodiscard]] auto expr() const -> Expr* { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* expr_;
};

class Unary final : public Expr {
 public:
  Unary(UnaryOp op, Expr* expr) : op_(op), expr_(expr) {}

  [[nodiscard]] auto op() const -> UnaryOp { return op_; }
  [[nodiscard]] auto expr() const -> Expr* { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  UnaryOp op_;
  Expr* expr_;
};

class Binary final : public Expr {
 public:
  Binary(BinaryOp op, Expr* left, Expr* right)
      : op_(op), left_(left), right_(right) {}

  [[nodiscard]] auto op() const -> BinaryOp { return op_; }
  [[nodiscard]] auto left() const -> Expr* { return left_; }
  [[nodiscard]] auto right() const -> Expr* { return right_; }

  void accept(IRVisitor* visitor) override;

 private:
  BinaryOp op_;
  Expr* left_;
  Expr* right_;
};

class Call final : public Expr {
 public:
  Call(Expr* base, std::vector<Expr*> args)
      : base_(base), args_(std::move(args)) {}

  [[nodiscard]] auto base() const -> Expr* { return base_; }
  [[nodiscard]] auto args() const -> const std::vector<Expr*>& { return args_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* base_;
  std::vector<Expr*> args_;
};

class Subscript final : public Expr {
 public:
  Subscript(Expr* base, Expr* index) : base_(base), index_(index) {}

  [[nodiscard]] auto base() const -> Expr* { return base_; }
  [[nodiscard]] auto index() const -> Expr* { return index_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* base_;
  Expr* index_;
};

class Access final : public Expr {
 public:
  Access(Expr* base, Symbol* member) : base_(base), member_(member) {}

  [[nodiscard]] auto base() const -> Expr* { return base_; }
  [[nodiscard]] auto member() const -> Symbol* { return member_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* base_;
  Symbol* member_;
};

class Cast final : public Expr {
 public:
  Cast(const QualifiedType& type, Expr* expr) : type_(type), expr_(expr) {}

  [[nodiscard]] auto type() const -> const QualifiedType& { return type_; }
  [[nodiscard]] auto expr() const -> Expr* { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  QualifiedType type_;
  Expr* expr_;
};

class StaticCast final : public Expr {
 public:
  StaticCast(const QualifiedType& type, Expr* expr)
      : type_(type), expr_(expr) {}

  [[nodiscard]] auto type() const -> const QualifiedType& { return type_; }
  [[nodiscard]] auto expr() const -> Expr* { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  QualifiedType type_;
  Expr* expr_;
};

class DynamicCast final : public Expr {
 public:
  DynamicCast(const QualifiedType& type, Expr* expr)
      : type_(type), expr_(expr) {}

  [[nodiscard]] auto type() const -> const QualifiedType& { return type_; }
  [[nodiscard]] auto expr() const -> Expr* { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  QualifiedType type_;
  Expr* expr_;
};

class ReinterpretCast final : public Expr {
 public:
  ReinterpretCast(const QualifiedType& type, Expr* expr)
      : type_(type), expr_(expr) {}

  [[nodiscard]] auto type() const -> const QualifiedType& { return type_; }
  [[nodiscard]] auto expr() const -> Expr* { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  QualifiedType type_;
  Expr* expr_;
};

class New final : public Expr {
 public:
  New(const QualifiedType& type, std::vector<Expr*> args)
      : type_(type), args_(std::move(args)) {}

  [[nodiscard]] auto type() const -> const QualifiedType& { return type_; }
  [[nodiscard]] auto args() const -> const std::vector<Expr*>& { return args_; }

  void accept(IRVisitor* visitor) override;

 private:
  QualifiedType type_;
  std::vector<Expr*> args_;
};

class NewArray final : public Expr {
 public:
  NewArray(const QualifiedType& type, Expr* size) : type_(type), size_(size) {}

  [[nodiscard]] auto type() const -> const QualifiedType& { return type_; }
  [[nodiscard]] auto size() const -> Expr* { return size_; }

  void accept(IRVisitor* visitor) override;

 private:
  QualifiedType type_;
  Expr* size_;
};

class Delete final : public Expr {
 public:
  explicit Delete(Expr* expr) : expr_(expr) {}

  [[nodiscard]] auto expr() const -> Expr* { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* expr_;
};

class DeleteArray final : public Expr {
 public:
  explicit DeleteArray(Expr* expr) : expr_(expr) {}

  [[nodiscard]] auto expr() const -> Expr* { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* expr_;
};

class Throw final : public Expr {
 public:
  explicit Throw(Expr* expr) : expr_(expr) {}

  [[nodiscard]] auto expr() const -> Expr* { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* expr_;
};

}  // namespace cxx::ir
