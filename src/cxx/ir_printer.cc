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

#include <cxx/ir.h>
#include <cxx/ir_printer.h>
#include <cxx/name_printer.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/type_printer.h>
#include <cxx/types.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace cxx::ir {

namespace {

TypePrinter typePrinter;

}  // namespace

void IRPrinter::print(Module* module, std::ostream& out) {
  for (auto function : module->functions()) {
    fmt::print(out, "\n");
    print(function, out);
  }
}

void IRPrinter::print(Function* function, std::ostream& out) {
  auto symbol = function->symbol();
  auto name = fmt::format("{}", *symbol->name());

  fmt::print(out, "{} {{\n", typePrinter.toString(symbol->type(), name));

  int tempIndex = 0;

  for (const auto& local : function->locals()) {
    std::string t = fmt::format("t{}", tempIndex++);
    fmt::print(out, "\t{};\n", typePrinter.toString(local.type(), t));
  }

  for (auto block : function->blocks()) {
    print(block, out);
  }

  fmt::print(out, "}}\n");
}

void IRPrinter::print(Block* block, std::ostream& out) {
  fmt::print("{}:", toString(block));
  for (auto stmt : block->code()) {
    print(stmt, out);
  }
}

void IRPrinter::print(Stmt* stmt, std::ostream& out) {
  if (!stmt) return;
  fmt::print(out, "\t{};\n", toString(stmt));
}

std::string IRPrinter::toString(Stmt* stmt) {
  std::string text;
  if (stmt) {
    std::swap(text_, text);
    stmt->accept(this);
    std::swap(text_, text);
  }
  return text;
}

std::string IRPrinter::toString(Block* block) const {
  return fmt::format("L{}", block->id());
}

std::string_view IRPrinter::toString(UnaryOp op) const {
  switch (op) {
    case UnaryOp::kStar:
      return "*";
    case UnaryOp::kAmp:
      return "&";
    case UnaryOp::kPlus:
      return "+";
    case UnaryOp::kMinus:
      return "-";
    case UnaryOp::kExclaim:
      return "!";
    case UnaryOp::kTilde:
      return "~";
    case UnaryOp::kPlusPlus:
      return "++";
    case UnaryOp::kMinusMinus:
      return "--";
    case UnaryOp::kPostPlusPlus:
      return "++";
    case UnaryOp::kPostMinusMinus:
      return "++";
    default:
      throw std::runtime_error("invalid operator");
  }  // switch
}

std::string_view IRPrinter::toString(BinaryOp op) const {
  switch (op) {
    case BinaryOp::kStar:
      return "*";
    case BinaryOp::kSlash:
      return "/";
    case BinaryOp::kPercent:
      return "%";
    case BinaryOp::kPlus:
      return "+";
    case BinaryOp::kMinus:
      return "-";
    case BinaryOp::kGreaterGreater:
      return ">>";
    case BinaryOp::kLessLess:
      return "<<";
    case BinaryOp::kGreater:
      return ">";
    case BinaryOp::kLess:
      return "<";
    case BinaryOp::kGreaterEqual:
      return ">=";
    case BinaryOp::kLessEqual:
      return "<=";
    case BinaryOp::kEqualEqual:
      return "==";
    case BinaryOp::kExclaimEqual:
      return "!=";
    case BinaryOp::kAmp:
      return "&";
    case BinaryOp::kCaret:
      return "^";
    case BinaryOp::kBar:
      return "|";
    default:
      throw std::runtime_error("invalid operator");
  }  // switch
}

std::string IRPrinter::quote(const std::string& s) const {
  std::string result;
  for (auto c : s) {
    if (c == '"' || c == '\\') result += '\\';
    result += c;
  }
  return result;
}

void IRPrinter::visit(Move* expr) {
  text_ = fmt::format("{} = {}", toString(expr->target()),
                      toString(expr->source()));
}

void IRPrinter::visit(Jump* stmt) {
  text_ = fmt::format("goto {}", toString(stmt->target()));
}

void IRPrinter::visit(CondJump* stmt) {
  text_ =
      fmt::format("if ({}) goto {}; else goto {}", toString(stmt->condition()),
                  toString(stmt->iftrue()), toString(stmt->iffalse()));
}

void IRPrinter::visit(Ret* stmt) {
  text_ = fmt::format("return {}", toString(stmt->result()));
}

void IRPrinter::visit(RetVoid*) { text_ = "return"; }

void IRPrinter::visit(This*) { text_ = "this"; }

void IRPrinter::visit(BoolLiteral* expr) {
  text_ = expr->value() ? "true" : "false";
}

void IRPrinter::visit(IntegerLiteral* expr) {
  struct Print {
    std::string operator()(std::int8_t v) const { return fmt::format("{}", v); }

    std::string operator()(std::int16_t v) const {
      return fmt::format("{}", v);
    }

    std::string operator()(std::int32_t v) const {
      return fmt::format("{}", v);
    }

    std::string operator()(std::int64_t v) const {
      return fmt::format("{}", v);
    }

    std::string operator()(std::uint8_t v) const {
      return fmt::format("{}", v);
    }

    std::string operator()(std::uint16_t v) const {
      return fmt::format("{}", v);
    }

    std::string operator()(std::uint32_t v) const {
      return fmt::format("{}", v);
    }

    std::string operator()(std::uint64_t v) const {
      return fmt::format("{}", v);
    }
  };

  text_ = std::visit(Print(), expr->value());
}

void IRPrinter::visit(FloatLiteral* expr) {
  struct Print {
    std::string operator()(float v) const { return fmt::format("{}", v); }
    std::string operator()(double v) const { return fmt::format("{}", v); }
    std::string operator()(long double v) const { return fmt::format("{}", v); }
  };

  text_ = std::visit(Print(), expr->value());
}

void IRPrinter::visit(NullptrLiteral*) { text_ = "nullptr"; }

void IRPrinter::visit(StringLiteral* expr) {
  text_ = fmt::format("\"{}\"", quote(expr->value()));
}

void IRPrinter::visit(UserDefinedStringLiteral* expr) {
  text_ = fmt::format("\"{}\"", quote(expr->value()));
}

void IRPrinter::visit(Load* expr) {
  text_ = fmt::format("t{}", expr->local()->index());
}

void IRPrinter::visit(Id* expr) {
  text_ = fmt::format("{}", *expr->symbol()->name());
}

void IRPrinter::visit(ExternalId* expr) { text_ = expr->name(); }

void IRPrinter::visit(Typeid* expr) {
  text_ = fmt::format("typeid({})", toString(expr->expr()));
}

void IRPrinter::visit(Unary* expr) {
  switch (expr->op()) {
    case UnaryOp::kPostMinusMinus:
    case UnaryOp::kPostPlusPlus:
      text_ =
          fmt::format("({} {})", toString(expr->expr()), toString(expr->op()));
      break;
    default:
      text_ =
          fmt::format("({} {})", toString(expr->op()), toString(expr->expr()));
  }  // switch
}

void IRPrinter::visit(Binary* expr) {
  text_ = fmt::format("{} {} {}", toString(expr->left()), toString(expr->op()),
                      toString(expr->right()));
}

void IRPrinter::visit(Call* expr) {
  text_ += toString(expr->base());
  text_ += '(';
  for (size_t i = 0; i < expr->args().size(); ++i) {
    if (i) text_ += ", ";
    text_ += toString(expr->args()[i]);
  }
  text_ += ')';
}

void IRPrinter::visit(Subscript* expr) {
  text_ =
      fmt::format("{}[{}]", toString(expr->base()), toString(expr->index()));
}

void IRPrinter::visit(Access* expr) {
  text_ =
      fmt::format("{}.{}", toString(expr->base()), toString(expr->member()));
}

void IRPrinter::visit(Cast* expr) {
  text_ = fmt::format("({}){}", toString(expr->type()), toString(expr->expr()));
}

void IRPrinter::visit(StaticCast* expr) {
  text_ = fmt::format("static_cast<{}>({})", toString(expr->type()),
                      toString(expr->expr()));
}

void IRPrinter::visit(DynamicCast* expr) {
  text_ = fmt::format("dynamic_cast<{}>({})", toString(expr->type()),
                      toString(expr->expr()));
}

void IRPrinter::visit(ReinterpretCast* expr) {
  text_ = fmt::format("reinterpret_cast<{}>({})", toString(expr->type()),
                      toString(expr->expr()));
}

void IRPrinter::visit(New* expr) {
  text_ = "new ";
  text_ += toString(expr->type());
  text_ += '(';
  for (size_t i = 0; i < expr->args().size(); ++i) {
    if (i) text_ += ", ";
    text_ += toString(expr->args()[i]);
  }
  text_ += ')';
}

void IRPrinter::visit(NewArray* expr) {
  text_ =
      fmt::format("new {}[{}]", toString(expr->type()), toString(expr->size()));
}

void IRPrinter::visit(Delete* expr) {
  text_ = fmt::format("delete {}", toString(expr->expr()));
}

void IRPrinter::visit(DeleteArray* expr) {
  text_ = fmt::format("delete[] {}", toString(expr->expr()));
}

void IRPrinter::visit(Throw* expr) {
  text_ = fmt::format("throw {}", toString(expr->expr()));
}

}  // namespace cxx::ir
