// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ir_fwd.h>
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
  Module& operator=(const Module&) = delete;

  Module();
  ~Module();

  const std::list<Function*>& functions() const;
  void addFunction(Function* function);

  const std::list<Global*>& globals() const;
  void addGlobal(Global* global);

  IRFactory* irFactory();

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

class Global final {
 public:
  explicit Global(Module* module, Symbol* symbol)
      : module_(module), symbol_(symbol) {}

  Module* module() const { return module_; }
  Symbol* symbol() const { return symbol_; }

 private:
  Module* module_;
  Symbol* symbol_;
};

class Local final {
 public:
  Local(const QualifiedType& type, int index) : type_(type), index_(index) {}

  const QualifiedType& type() const { return type_; }
  int index() const { return index_; }

 private:
  QualifiedType type_;
  int index_;
};

class Function final {
 public:
  Function(Module* module, FunctionSymbol* symbol)
      : module_(module), symbol_(symbol) {}

  Module* module() const { return module_; }
  FunctionSymbol* symbol() const { return symbol_; }

  const std::list<Block*>& blocks() const { return blocks_; }

  void addBlock(Block* block) { blocks_.push_back(block); }

  const std::list<Local>& locals() const { return locals_; }

  Local* addLocal(const QualifiedType& type) {
    int index = int(locals_.size());
    return &locals_.emplace_back(type, index);
  }

 private:
  Module* module_;
  FunctionSymbol* symbol_;
  std::list<Block*> blocks_;
  std::list<Local> locals_;
};

class Block final {
 public:
  explicit Block(Function* function) : function_(function) {}

  Function* function() const { return function_; }

  std::list<Stmt*>& code() { return code_; }
  const std::list<Stmt*>& code() const { return code_; }

  int id() const;

  bool hasTerminator() const;

 private:
  Function* function_;
  std::list<Stmt*> code_;
};

class Stmt {
 public:
  Stmt() = default;
  virtual ~Stmt() = default;

  virtual void accept(IRVisitor* visitor) = 0;

  virtual bool isTerminator() const { return false; }
};

class Expr : public Stmt {
 public:
  Expr() = default;
};

class Store final : public Stmt {
 public:
  Store(Expr* target, Expr* source) : target_(target), source_(source) {}

  Expr* target() const { return target_; }
  Expr* source() const { return source_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* target_;
  Expr* source_;
};

class Jump final : public Stmt {
 public:
  explicit Jump(Block* target) : target_(target) {}

  Block* target() const { return target_; }

  bool isTerminator() const override { return true; }

  void accept(IRVisitor* visitor) override;

 private:
  Block* target_ = nullptr;
};

class CondJump final : public Stmt {
 public:
  CondJump(Expr* condition, Block* iftrue, Block* iffalse)
      : condition_(condition), iftrue_(iftrue), iffalse_(iffalse) {}

  Expr* condition() const { return condition_; }
  Block* iftrue() const { return iftrue_; }
  Block* iffalse() const { return iffalse_; }

  bool isTerminator() const override { return true; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* condition_;
  Block* iftrue_;
  Block* iffalse_;
};

class Ret final : public Stmt {
 public:
  explicit Ret(Expr* result) : result_(result) {}

  Expr* result() const { return result_; }

  bool isTerminator() const override { return true; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* result_;
};

class RetVoid final : public Stmt {
 public:
  RetVoid() = default;

  bool isTerminator() const override { return true; }

  void accept(IRVisitor* visitor) override;
};

class This final : public Expr {
 public:
  explicit This(Expr* type) : type_(type) {}

  Expr* type() const { return type_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* type_;
};

class BoolLiteral final : public Expr {
 public:
  explicit BoolLiteral(bool value) : value_(value) {}

  bool value() const { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  bool value_;
};

class IntegerLiteral final : public Expr {
 public:
  explicit IntegerLiteral(const IntegerValue& value) : value_(value) {}

  IntegerValue value() { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  IntegerValue value_;
};

class FloatLiteral final : public Expr {
 public:
  explicit FloatLiteral(const FloatValue& value) : value_(value) {}

  FloatValue value() const { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  FloatValue value_;
};

class NullptrLiteral final : public Expr {
 public:
  NullptrLiteral() {}

  void accept(IRVisitor* visitor) override;
};

class StringLiteral final : public Expr {
 public:
  explicit StringLiteral(std::string value) : value_(std::move(value)) {}

  const std::string& value() const { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  std::string value_;
};

class UserDefinedStringLiteral final : public Expr {
 public:
  explicit UserDefinedStringLiteral(std::string value)
      : value_(std::move(value)) {}

  const std::string& value() const { return value_; }

  void accept(IRVisitor* visitor) override;

 private:
  std::string value_;
};

class Load final : public Expr {
 public:
  explicit Load(Local* local) : local_(local) {}

  Local* local() const { return local_; }

  void accept(IRVisitor* visitor) override;

 private:
  Local* local_;
};

class Id final : public Expr {
 public:
  explicit Id(Symbol* symbol) : symbol_(symbol) {}

  Symbol* symbol() const { return symbol_; }

  void accept(IRVisitor* visitor) override;

 private:
  Symbol* symbol_;
};

class ExternalId final : public Expr {
 public:
  explicit ExternalId(std::string name) : name_(std::move(name)) {}

  const std::string& name() const { return name_; }

  void accept(IRVisitor* visitor) override;

 private:
  std::string name_;
};

class Typeid final : public Expr {
 public:
  explicit Typeid(Expr* expr) : expr_(expr) {}

  Expr* expr() const { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* expr_;
};

class Unary final : public Expr {
 public:
  Unary(UnaryOp op, Expr* expr) : op_(op), expr_(expr) {}

  UnaryOp op() const { return op_; }
  Expr* expr() const { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  UnaryOp op_;
  Expr* expr_;
};

class Binary final : public Expr {
 public:
  Binary(BinaryOp op, Expr* left, Expr* right)
      : op_(op), left_(left), right_(right) {}

  BinaryOp op() const { return op_; }
  Expr* left() const { return left_; }
  Expr* right() const { return right_; }

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

  Expr* base() const { return base_; }
  const std::vector<Expr*>& args() const { return args_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* base_;
  std::vector<Expr*> args_;
};

class Subscript final : public Expr {
 public:
  Subscript(Expr* base, Expr* index) : base_(base), index_(index) {}

  Expr* base() const { return base_; }
  Expr* index() const { return index_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* base_;
  Expr* index_;
};

class Access final : public Expr {
 public:
  Access(Expr* base, Expr* member) : base_(base), member_(member) {}

  Expr* base() const { return base_; }
  Expr* member() const { return member_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* base_;
  Expr* member_;
};

class Cast final : public Expr {
 public:
  Cast(Expr* type, Expr* expr) : type_(type), expr_(expr) {}

  Expr* type() const { return type_; }
  Expr* expr() const { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* type_;
  Expr* expr_;
};

class StaticCast final : public Expr {
 public:
  StaticCast(Expr* type, Expr* expr) : type_(type), expr_(expr) {}

  Expr* type() const { return type_; }
  Expr* expr() const { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* type_;
  Expr* expr_;
};

class DynamicCast final : public Expr {
 public:
  DynamicCast(Expr* type, Expr* expr) : type_(type), expr_(expr) {}

  Expr* type() const { return type_; }
  Expr* expr() const { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* type_;
  Expr* expr_;
};

class ReinterpretCast final : public Expr {
 public:
  ReinterpretCast(Expr* type, Expr* expr) : type_(type), expr_(expr) {}

  Expr* type() const { return type_; }
  Expr* expr() const { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* type_;
  Expr* expr_;
};

class New final : public Expr {
 public:
  New(Expr* type, std::vector<Expr*> args)
      : type_(type), args_(std::move(args)) {}

  Expr* type() const { return type_; }
  const std::vector<Expr*>& args() const { return args_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* type_;
  std::vector<Expr*> args_;
};

class NewArray final : public Expr {
 public:
  NewArray(Expr* type, Expr* size) : type_(type), size_(size) {}

  Expr* type() const { return type_; }
  Expr* size() const { return size_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* type_;
  Expr* size_;
};

class Delete final : public Expr {
 public:
  explicit Delete(Expr* expr) : expr_(expr) {}

  Expr* expr() const { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* expr_;
};

class DeleteArray final : public Expr {
 public:
  explicit DeleteArray(Expr* expr) : expr_(expr) {}

  Expr* expr() const { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* expr_;
};

class Throw final : public Expr {
 public:
  explicit Throw(Expr* expr) : expr_(expr) {}

  Expr* expr() const { return expr_; }

  void accept(IRVisitor* visitor) override;

 private:
  Expr* expr_;
};

}  // namespace cxx::ir
