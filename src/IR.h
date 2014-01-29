// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef IR_H
#define IR_H

#include "Globals.h"
#include "Types.h" // ### remove
#include <vector>
#include <tuple>
#include <set>
#include <forward_list>

#define FOR_EACH_IR_STMT(V) \
  V(Exp) \
  V(Move) \
  V(Ret) \
  V(Jump) \
  V(CJump)

#define FOR_EACH_IR_EXPR(V) \
  V(Const) \
  V(Temp) \
  V(Sym) \
  V(Cast) \
  V(Call) \
  V(Member) \
  V(Subscript) \
  V(Unop) \
  V(Binop)

namespace IR {

#define VISIT_IR_STMT(x) struct x;
FOR_EACH_IR_STMT(VISIT_IR_STMT)
#undef VISIT_IR_STMT

#define VISIT_IR_EXPR(x) struct x;
FOR_EACH_IR_EXPR(VISIT_IR_EXPR)
#undef VISIT_IR_EXPR

enum struct StmtKind {
#define VISIT_IR_STMT(x) k##x,
FOR_EACH_IR_STMT(VISIT_IR_STMT)
#undef VISIT_IR_STMT
};

enum struct ExprKind {
#define VISIT_IR_EXPR(x) k##x,
FOR_EACH_IR_EXPR(VISIT_IR_EXPR)
#undef VISIT_IR_EXPR
};

struct Stmt {
  Stmt(StmtKind kind): _kind(kind) {}
  virtual ~Stmt() = default;

#define VISIT_IR_STMT(T) \
  inline bool is##T() const { \
    return _kind == StmtKind::k##T; \
  } \
  inline const T* as##T() const { \
    return is##T() ? reinterpret_cast<const T*>(this) : nullptr; \
  }
  FOR_EACH_IR_STMT(VISIT_IR_STMT)
#undef VISIT_IR_STMT

  virtual Terminator* asTerminator() { return nullptr; }
  virtual void dump(std::ostream& out) const = 0;
private:
  StmtKind _kind;
};

struct Expr {
  Expr(ExprKind kind): _kind(kind) {}
  virtual ~Expr() = default;

#define VISIT_IR_EXPR(T) \
  inline bool is##T() const { \
    return _kind == ExprKind::k##T; \
  } \
  inline const T* as##T() const { \
    return is##T() ? reinterpret_cast<const T*>(this) : nullptr; \
  }
  FOR_EACH_IR_EXPR(VISIT_IR_EXPR)
#undef VISIT_IR_EXPR

  virtual void dump(std::ostream& out) const = 0;
private:
  ExprKind _kind;
};

template <StmtKind K, typename Base, typename..._Args>
struct ExtendsStmt: Base, std::tuple<_Args...> {
  template <typename...Args>
  ExtendsStmt(Args&&...args)
    : Base(K), std::tuple<_Args...>(std::forward<Args>(args)...) {}
};

template <ExprKind K, typename..._Args>
struct ExtendsExpr: Expr, std::tuple<_Args...> {
  template <typename...Args>
  ExtendsExpr(Args&&...args)
    : Expr(K), std::tuple<_Args...>(std::forward<Args>(args)...) {}
};

//
// statements
//
struct Terminator: Stmt {
  using Stmt::Stmt;
  Terminator* asTerminator() override { return this; }
};

struct Exp final: ExtendsStmt<StmtKind::kExp, Stmt, const Expr*> {
  using ExtendsStmt::ExtendsStmt;
  const Expr* expr() const { return std::get<0>(*this); }
  void dump(std::ostream& out) const override;
};

struct Move final: ExtendsStmt<StmtKind::kMove, Stmt, const Expr*, const Expr*, TokenKind> {
  using ExtendsStmt::ExtendsStmt;
  Move(const Expr* target, const Expr* source)
    : ExtendsStmt(target, source, T_EQUAL) {}
  const Expr* target() const { return std::get<0>(*this); }
  const Expr* source() const { return std::get<1>(*this); }
  TokenKind op() const { return std::get<2>(*this); }
  void dump(std::ostream& out) const override;
};

struct Ret final: ExtendsStmt<StmtKind::kRet, Terminator, const Expr*> {
  using ExtendsStmt::ExtendsStmt;
  const Expr* expr() const { return std::get<0>(*this); }
  void dump(std::ostream& out) const override;
};

struct Jump final: ExtendsStmt<StmtKind::kJump, Terminator, BasicBlock*> {
  using ExtendsStmt::ExtendsStmt;
  BasicBlock* target() const { return std::get<0>(*this); }
  void dump(std::ostream& out) const override;
};

struct CJump final: ExtendsStmt<StmtKind::kCJump, Terminator, const Expr*, BasicBlock*, BasicBlock*> {
  using ExtendsStmt::ExtendsStmt;
  const Expr* expr() const { return std::get<0>(*this); }
  BasicBlock* iftrue() const { return std::get<1>(*this); }
  BasicBlock* iffalse() const { return std::get<2>(*this); }
  void dump(std::ostream& out) const override;
};

//
// expressions
//
struct Const final: ExtendsExpr<ExprKind::kConst, const char*> { // ### TODO: LiteralValue*.
  using ExtendsExpr::ExtendsExpr;
  const char* value() const { return std::get<0>(*this); }
  void dump(std::ostream& out) const override;
};

struct Temp final: ExtendsExpr<ExprKind::kTemp, int> {
  using ExtendsExpr::ExtendsExpr;
  int index() const { return std::get<0>(*this); }
  void dump(std::ostream& out) const override;
};

struct Sym final: ExtendsExpr<ExprKind::kSym, const Name*> {
  using ExtendsExpr::ExtendsExpr;
  const Name* name() const { return std::get<0>(*this); }
  void dump(std::ostream& out) const override;
};

struct Cast final: ExtendsExpr<ExprKind::kCast, QualType, const Expr*> {
  using ExtendsExpr::ExtendsExpr;
  QualType type() const { return std::get<0>(*this); }
  const Expr* expr() const { return std::get<1>(*this); }
  void dump(std::ostream& out) const override;
};

struct Call final: ExtendsExpr<ExprKind::kCall, const Expr*, std::vector<const Expr*>> {
  using ExtendsExpr::ExtendsExpr;
  const Expr* expr() const { return std::get<0>(*this); }
  const std::vector<const Expr*>& args() const { return std::get<1>(*this); }
  void dump(std::ostream& out) const override;
};

struct Member final: ExtendsExpr<ExprKind::kMember, TokenKind, const Expr*, const Name*> {
  using ExtendsExpr::ExtendsExpr;
  TokenKind op() const { return std::get<0>(*this); }
  const Expr* expr() const { return std::get<1>(*this); }
  const Name* name() const { return std::get<2>(*this); }
  void dump(std::ostream& out) const override;
};

struct Subscript final: ExtendsExpr<ExprKind::kSubscript, const Expr*, const Expr*> {
  using ExtendsExpr::ExtendsExpr;
  const Expr* expr() const { return std::get<0>(*this); }
  const Expr* index() const { return std::get<1>(*this); }
  void dump(std::ostream& out) const override;
};

struct Unop final: ExtendsExpr<ExprKind::kUnop, TokenKind, const Expr*> {
  using ExtendsExpr::ExtendsExpr;
  TokenKind op() const { return std::get<0>(*this); }
  const Expr* expr() const { return std::get<1>(*this); }
  void dump(std::ostream& out) const override;
};

struct Binop final: ExtendsExpr<ExprKind::kBinop, TokenKind, const Expr*, const Expr*> {
  using ExtendsExpr::ExtendsExpr;
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

struct Function final: std::vector<BasicBlock*> {
  Module* module;
  FunctionSymbol* symbol;

  Function(Module* module, FunctionSymbol* symbol)
    : module(module)
    , symbol(symbol) {}

  BasicBlock* newBasicBlock();
  void placeBasicBlock(BasicBlock* basicBlock);

  void dump(std::ostream& out);

  template <typename T>
  struct Table final: std::set<T> {
    template <typename...Args>
    const T* operator()(Args&&...args) {
      return &*this->emplace(std::forward<Args>(args)...).first;
    }
  };

#define VISIT_IR_EXPR(T) Table<T> get##T;
  FOR_EACH_IR_EXPR(VISIT_IR_EXPR)
#undef VISIT_IR_EXPR
};

struct BasicBlock final: std::vector<Stmt*> {
  Function* function;
  int index{-1};

  BasicBlock(Function* function)
    : function(function) {}

  Terminator* terminator() const {
    return empty() ? nullptr : back()->asTerminator();
  }

  bool isTerminated() const {
    return terminator() != nullptr;
  }

  template <typename T>
  struct Sequence final: std::forward_list<T> {
    BasicBlock* basicBlock;

    Sequence(BasicBlock* basicBlock)
      : basicBlock(basicBlock) {}

    template <typename...Args>
    void operator()(Args&&...args) {
      if (basicBlock->isTerminated())
        return;
      auto node = &*this->emplace_after(this->before_begin(), std::forward<Args>(args)...);
      basicBlock->push_back(node);
    }
  };

#define VISIT_IR_STMT(T) Sequence<T> emit##T{this};
  FOR_EACH_IR_STMT(VISIT_IR_STMT)
#undef VISIT_IR_STMT
};

} // end of namespace IR

#endif // IR_H
