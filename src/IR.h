// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
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

#ifndef IR_H
#define IR_H

#include <forward_list>
#include <set>
#include <tuple>
#include <vector>

#include "Globals.h"
#include "Types.h"  // ### remove

namespace IR {

struct Stmt {
  Stmt(StmtKind kind) : _kind(kind) {}
  virtual ~Stmt() = default;

  inline StmtKind kind() const { return _kind; }

#define VISIT_IR_STMT(T)                                         \
  inline bool is##T() const { return _kind == StmtKind::k##T; }  \
  inline const T* as##T() const {                                \
    return is##T() ? reinterpret_cast<const T*>(this) : nullptr; \
  }                                                              \
  inline T* as##T() { return is##T() ? reinterpret_cast<T*>(this) : nullptr; }
  FOR_EACH_IR_STMT(VISIT_IR_STMT)
#undef VISIT_IR_STMT

  virtual Terminator* asTerminator() { return nullptr; }
  virtual void dump(std::ostream& out) const = 0;

 private:
  StmtKind _kind;
};

struct Expr {
  Expr(ExprKind kind) : _kind(kind) {}
  virtual ~Expr() = default;

  inline ExprKind kind() const { return _kind; }

#define VISIT_IR_EXPR(T)                                         \
  inline bool is##T() const { return _kind == ExprKind::k##T; }  \
  inline const T* as##T() const {                                \
    return is##T() ? reinterpret_cast<const T*>(this) : nullptr; \
  }
  FOR_EACH_IR_EXPR(VISIT_IR_EXPR)
#undef VISIT_IR_EXPR

  virtual void dump(std::ostream& out) const = 0;

 private:
  ExprKind _kind;
};

template <StmtKind K, typename Base = Stmt>
struct ExtendsStmt : Base {
  ExtendsStmt() : Base(K) {}
};

template <ExprKind K, typename Base = Expr>
struct ExtendsExpr : Expr {
  ExtendsExpr() : Expr(K) {}
};

//
// statements
//
struct Terminator : Stmt {
  using Stmt::Stmt;
  Terminator* asTerminator() override { return this; }
};

struct Exp final : ExtendsStmt<StmtKind::kExp>, std::tuple<const Expr*> {
  using tuple::tuple;

  const Expr* expr() const { return std::get<0>(*this); }
  void setExpr(const Expr* expr) { std::get<0>(*this) = expr; }

  void dump(std::ostream& out) const override;
};

struct Move final : ExtendsStmt<StmtKind::kMove>,
                    std::tuple<const Expr*, const Expr*, TokenKind> {
  Move(const Expr* target, const Expr* source, TokenKind op = T_EQUAL)
      : tuple(target, source, op) {}

  const Expr* target() const { return std::get<0>(*this); }
  void setTarget(const Expr* target) { std::get<0>(*this) = target; }

  const Expr* source() const { return std::get<1>(*this); }
  void setSource(const Expr* source) { std::get<1>(*this) = source; }

  TokenKind op() const {
    auto op = std::get<2>(*this);
    return op ? op : T_EQUAL;
  }
  void setOp(TokenKind op) { std::get<2>(*this) = op; }

  void dump(std::ostream& out) const override;
};

struct Ret final : ExtendsStmt<StmtKind::kRet, Terminator>,
                   std::tuple<const Expr*> {
  using tuple::tuple;

  const Expr* expr() const { return std::get<0>(*this); }
  void setExpr(const Expr* expr) { std::get<0>(*this) = expr; }

  void dump(std::ostream& out) const override;
};

struct Jump final : ExtendsStmt<StmtKind::kJump, Terminator>,
                    std::tuple<BasicBlock*> {
  using tuple::tuple;

  BasicBlock* target() const { return std::get<0>(*this); }
  void setTarget(BasicBlock* target) { std::get<0>(*this) = target; }

  void dump(std::ostream& out) const override;
};

struct CJump final : ExtendsStmt<StmtKind::kCJump, Terminator>,
                     std::tuple<const Expr*, BasicBlock*, BasicBlock*> {
  using tuple::tuple;

  const Expr* expr() const { return std::get<0>(*this); }
  void setExpr(const Expr* expr) { std::get<0>(*this) = expr; }

  BasicBlock* iftrue() const { return std::get<1>(*this); }
  void setIftrue(BasicBlock* iftrue) { std::get<1>(*this) = iftrue; }

  BasicBlock* iffalse() const { return std::get<2>(*this); }
  void setIffalse(BasicBlock* iffalse) { std::get<2>(*this) = iffalse; }

  void dump(std::ostream& out) const override;
};

//
// expressions
//
struct This final : ExtendsExpr<ExprKind::kThis> {
  This() = default;
  bool operator<(const This&) const { return false; }
  void dump(std::ostream& out) const override;
};

struct Const final : ExtendsExpr<ExprKind::kConst>,
                     std::tuple<const char*> {  // ### TODO: LiteralValue*.
  using tuple::tuple;
  const char* value() const { return std::get<0>(*this); }
  void dump(std::ostream& out) const override;
};

struct Temp final : ExtendsExpr<ExprKind::kTemp>, std::tuple<int> {
  using tuple::tuple;
  int index() const { return std::get<0>(*this); }
  void dump(std::ostream& out) const override;
};

struct Sym final : ExtendsExpr<ExprKind::kSym>, std::tuple<const Name*> {
  using tuple::tuple;
  const Name* name() const { return std::get<0>(*this); }
  void dump(std::ostream& out) const override;
};

template <ExprKind K>
struct ExtendsCast : ExtendsExpr<K>, std::tuple<QualType, const Expr*> {
  using tuple::tuple;
  QualType type() const { return std::get<0>(*this); }
  const Expr* expr() const { return std::get<1>(*this); }
};

struct Cast final : ExtendsCast<ExprKind::kCast> {
  using ExtendsCast::ExtendsCast;
  void dump(std::ostream& out) const override;
};

struct DynamicCast final : ExtendsCast<ExprKind::kDynamicCast> {
  using ExtendsCast::ExtendsCast;
  void dump(std::ostream& out) const override;
};

struct StaticCast final : ExtendsCast<ExprKind::kStaticCast> {
  using ExtendsCast::ExtendsCast;
  void dump(std::ostream& out) const override;
};

struct ReinterpretCast final : ExtendsCast<ExprKind::kReinterpretCast> {
  using ExtendsCast::ExtendsCast;
  void dump(std::ostream& out) const override;
};

struct ConstCast final : ExtendsCast<ExprKind::kConstCast> {
  using ExtendsCast::ExtendsCast;
  void dump(std::ostream& out) const override;
};

struct Call final : ExtendsExpr<ExprKind::kCall>,
                    std::tuple<const Expr*, std::vector<const Expr*>> {
  using tuple::tuple;
  const Expr* expr() const { return std::get<0>(*this); }
  const std::vector<const Expr*>& args() const { return std::get<1>(*this); }
  void dump(std::ostream& out) const override;
};

struct Member final : ExtendsExpr<ExprKind::kMember>,
                      std::tuple<TokenKind, const Expr*, const Name*> {
  using tuple::tuple;
  TokenKind op() const { return std::get<0>(*this); }
  const Expr* expr() const { return std::get<1>(*this); }
  const Name* name() const { return std::get<2>(*this); }
  void dump(std::ostream& out) const override;
};

struct Subscript final : ExtendsExpr<ExprKind::kSubscript>,
                         std::tuple<const Expr*, const Expr*> {
  using tuple::tuple;
  const Expr* expr() const { return std::get<0>(*this); }
  const Expr* index() const { return std::get<1>(*this); }
  void dump(std::ostream& out) const override;
};

struct Unop final : ExtendsExpr<ExprKind::kUnop>,
                    std::tuple<TokenKind, const Expr*> {
  using tuple::tuple;
  TokenKind op() const { return std::get<0>(*this); }
  const Expr* expr() const { return std::get<1>(*this); }
  void dump(std::ostream& out) const override;
};

struct Binop final : ExtendsExpr<ExprKind::kBinop>,
                     std::tuple<TokenKind, const Expr*, const Expr*> {
  using tuple::tuple;
  TokenKind op() const { return std::get<0>(*this); }
  const Expr* left() const { return std::get<1>(*this); }
  const Expr* right() const { return std::get<2>(*this); }
  void dump(std::ostream& out) const override;
};

struct Module {
  std::vector<Function*> functions;

  Module();
  ~Module();

  Function* newFunction(FunctionSymbol* symbol);
};

struct Function final : std::vector<BasicBlock*> {
  Module* module;
  FunctionSymbol* symbol;

  Function(Module* module, FunctionSymbol* symbol)
      : module(module), symbol(symbol) {}
  ~Function();

  BasicBlock* newBasicBlock();
  void placeBasicBlock(BasicBlock* basicBlock);

  void removeUnreachableBasicBlocks();

  void dump(std::ostream& out);

  template <typename T>
  struct Table final : std::set<T> {
    template <typename... Args>
    const T* operator()(Args&&... args) {
      return &*this->emplace(std::forward<Args>(args)...).first;
    }
  };

  template <typename T>
  struct Sequence final : std::forward_list<T> {
    template <typename... Args>
    T* operator()(Args&&... args) {
      auto node = &*this->emplace_after(this->before_begin(),
                                        std::forward<Args>(args)...);
      return node;
    }
  };

#define VISIT_IR_EXPR(T) Table<T> get##T;
  FOR_EACH_IR_EXPR(VISIT_IR_EXPR)
#undef VISIT_IR_EXPR

#define VISIT_IR_STMT(T) Sequence<T> new##T;
  FOR_EACH_IR_STMT(VISIT_IR_STMT)
#undef VISIT_IR_STMT
};

struct BasicBlock final : std::vector<Stmt*> {
  Function* function;
  int index{-1};

  BasicBlock(Function* function) : function(function) {}

  Terminator* terminator() const {
    return empty() ? nullptr : back()->asTerminator();
  }

  bool isTerminated() const { return terminator() != nullptr; }

#define VISIT_IR_STMT(T)                                       \
  template <typename... Args>                                  \
  void emit##T(Args&&... args) {                               \
    if (isTerminated()) return;                                \
    auto node = function->new##T(std::forward<Args>(args)...); \
    push_back(node);                                           \
  }
  FOR_EACH_IR_STMT(VISIT_IR_STMT)
#undef VISIT_IR_STMT
};

}  // end of namespace IR

#endif  // IR_H
