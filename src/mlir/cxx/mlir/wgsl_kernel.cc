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

#include <cxx/device_profile_limits.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/device_kernel.h>
#include <cxx/mlir/kernel_manifest.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include <array>
#include <cmath>
#include <string>

namespace cxx {
namespace {

using namespace mlir;

constexpr auto kWGSLReservedWords = std::to_array<llvm::StringLiteral>({
    "alias",
    "break",
    "case",
    "const",
    "const_assert",
    "continue",
    "continuing",
    "default",
    "diagnostic",
    "discard",
    "else",
    "enable",
    "false",
    "fn",
    "for",
    "if",
    "let",
    "loop",
    "override",
    "requires",
    "return",
    "struct",
    "switch",
    "true",
    "var",
    "while",
    "NULL",
    "Self",
    "abstract",
    "active",
    "alignas",
    "alignof",
    "as",
    "asm",
    "asm_fragment",
    "async",
    "attribute",
    "auto",
    "await",
    "become",
    "cast",
    "catch",
    "class",
    "co_await",
    "co_return",
    "co_yield",
    "coherent",
    "column_major",
    "common",
    "compile",
    "compile_fragment",
    "concept",
    "const_cast",
    "consteval",
    "constexpr",
    "constinit",
    "crate",
    "debugger",
    "decltype",
    "delete",
    "demote",
    "demote_to_helper",
    "do",
    "dynamic_cast",
    "enum",
    "explicit",
    "export",
    "extends",
    "extern",
    "external",
    "fallthrough",
    "filter",
    "final",
    "finally",
    "friend",
    "from",
    "fxgroup",
    "get",
    "goto",
    "groupshared",
    "highp",
    "impl",
    "implements",
    "import",
    "inline",
    "instanceof",
    "interface",
    "layout",
    "lowp",
    "macro",
    "macro_rules",
    "match",
    "mediump",
    "meta",
    "mod",
    "module",
    "move",
    "mut",
    "mutable",
    "namespace",
    "new",
    "nil",
    "noexcept",
    "noinline",
    "nointerpolation",
    "non_coherent",
    "noncoherent",
    "noperspective",
    "null",
    "nullptr",
    "of",
    "operator",
    "package",
    "packoffset",
    "partition",
    "pass",
    "patch",
    "pixelfragment",
    "precise",
    "precision",
    "premerge",
    "priv",
    "protected",
    "pub",
    "public",
    "readonly",
    "ref",
    "regardless",
    "register",
    "reinterpret_cast",
    "require",
    "resource",
    "restrict",
    "self",
    "set",
    "shared",
    "sizeof",
    "smooth",
    "snorm",
    "static",
    "static_assert",
    "static_cast",
    "std",
    "subroutine",
    "super",
    "target",
    "template",
    "this",
    "thread_local",
    "throw",
    "trait",
    "try",
    "type",
    "typedef",
    "typeid",
    "typename",
    "typeof",
    "union",
    "unless",
    "unorm",
    "unsafe",
    "unsized",
    "use",
    "using",
    "varying",
    "virtual",
    "volatile",
    "wgsl",
    "where",
    "with",
    "writeonly",
    "yield",
});

constexpr auto kWGSLGeneratedNames = std::to_array<llvm::StringLiteral>({
    "abs",   "acos",  "array",  "asin", "atan", "atan2", "bitcast",
    "bool",  "ceil",  "cos",    "cosh", "exp",  "exp2",  "f32",
    "floor", "fma",   "i32",    "log",  "log2", "max",   "min",
    "pow",   "round", "select", "sign", "sin",  "sinh",  "sqrt",
    "tan",   "tanh",  "trunc",  "u32",  "vec2", "vec3",  "vec4",
});

[[nodiscard]] auto isUsableEntryName(llvm::StringRef name) -> bool {
  if (name.empty() || name == "_" || name.starts_with("__")) return false;
  if (name.starts_with("cxx_") || name.starts_with("Cxx")) return false;
  if (llvm::is_contained(kWGSLReservedWords, name)) return false;
  return !llvm::is_contained(kWGSLGeneratedNames, name);
}

struct BufferView {
  std::string name;
  bool isUniform = false;
  bool isUnsigned = false;
};

[[nodiscard]] auto bufferValueTypeName(const mlir::cxx::KernelBuffer& buffer)
    -> std::string {
  auto scalar = mlir::cxx::kernelBufferScalarTypeName(buffer).str();
  auto components = mlir::cxx::kernelBufferComponents(buffer);
  if (components == 1) return scalar;
  return "vec" + std::to_string(components) + "<" + scalar + ">";
}

class WGSLWriter {
 public:
  WGSLWriter(gpu::GPUFuncOp function, const KernelEntry& entry)
      : function_(function), entry_(entry), out_(text_) {}

  [[nodiscard]] auto write() -> FailureOr<std::string> {
    writeBindings();
    const auto& workgroup = entry_.contract.workgroup;
    out_ << "@compute @workgroup_size(" << workgroup[0] << ", " << workgroup[1]
         << ", " << workgroup[2] << ")\nfn " << entry_.name << "("
         << entryParameters() << ") {\n";
    auto terminator = writeBody(function_.getBody().front(), 1);
    if (failed(terminator)) return failure();
    if (!isa<gpu::ReturnOp>(*terminator))
      return (*terminator)->emitError("unexpected WGSL function terminator");
    out_ << "}\n";
    return text_;
  }

 private:
  void writeBindings() {
    for (auto [buffer, argument] :
         llvm::zip(entry_.contract.buffers, function_.getArguments())) {
      auto value = bufferValueTypeName(buffer);
      auto name = "cxx_buffer" + std::to_string(buffer.binding);
      auto isUniform = buffer.kind == mlir::cxx::KernelBufferKind::Uniform;
      out_ << "@group(" << ::cxx::kBufferDescriptorSet << ") @binding("
           << buffer.binding << ") ";
      if (isUniform) {
        out_ << "var<uniform> " << name << ": " << value << ";\n\n";
      } else {
        out_ << "var<storage, " << (buffer.readOnly ? "read" : "read_write")
             << "> " << name << ": array<" << value << ">;\n\n";
      }
      buffers_[argument] = {name, isUniform, buffer.unsignedElements};
    }
  }

  [[nodiscard]] auto entryParameters() -> std::string {
    constexpr std::array<std::pair<llvm::StringRef, llvm::StringRef>, 3>
        builtins = {{{"global_invocation_id", "cxx_global_id"},
                     {"local_invocation_id", "cxx_local_id"},
                     {"workgroup_id", "cxx_workgroup_id"}}};
    std::array<bool, 3> used{};
    function_.walk([&](Operation* op) {
      if (isa<gpu::GlobalIdOp>(op)) used[0] = true;
      if (isa<gpu::ThreadIdOp>(op)) used[1] = true;
      if (isa<gpu::BlockIdOp>(op)) used[2] = true;
    });
    std::string parameters;
    for (auto [isUsed, builtin] : llvm::zip(used, builtins)) {
      if (!isUsed) continue;
      if (!parameters.empty()) parameters += ", ";
      parameters += "@builtin(" + builtin.first.str() + ") " +
                    builtin.second.str() + ": vec3<u32>";
    }
    return parameters;
  }

  [[nodiscard]] auto scalarTypeName(Type type) -> std::optional<std::string> {
    if (type.isF32()) return "f32";
    if (type.isIndex() || type.isSignlessInteger(32)) return "i32";
    if (type.isSignlessInteger(1)) return "bool";
    return std::nullopt;
  }

  [[nodiscard]] auto typeName(Operation* op, Type type)
      -> FailureOr<std::string> {
    if (auto vector = dyn_cast<VectorType>(type)) {
      auto lanes = vector.getNumElements();
      auto element = scalarTypeName(vector.getElementType());
      if (vector.getRank() == 1 && lanes >= 2 && lanes <= 4 && element &&
          !vector.getElementType().isIndex())
        return "vec" + std::to_string(lanes) + "<" + *element + ">";
    } else if (auto scalar = scalarTypeName(type)) {
      return *scalar;
    }
    return op->emitError("WGSL emission does not support type ") << type;
  }

  [[nodiscard]] static auto withLanesOf(Type type, llvm::StringRef element)
      -> std::string {
    auto vector = dyn_cast<VectorType>(type);
    if (!vector) return element.str();
    return "vec" + std::to_string(vector.getNumElements()) + "<" +
           element.str() + ">";
  }

  [[nodiscard]] auto splat(Type type, llvm::StringRef scalar) -> std::string {
    if (!isa<VectorType>(type)) return scalar.str();
    return withLanesOf(type, "i32") + "(" + scalar.str() + ")";
  }

  void indent(unsigned depth) {
    for (unsigned i = 0; i < depth; ++i) out_ << "  ";
  }

  [[nodiscard]] auto use(Value value) -> std::string {
    return names_.lookup(value);
  }

  [[nodiscard]] auto unsignedUse(Value value) -> std::string {
    return "bitcast<" + withLanesOf(value.getType(), "u32") + ">(" +
           use(value) + ")";
  }

  [[nodiscard]] auto signedResult(Value value, const std::string& expression)
      -> std::string {
    return "bitcast<" + withLanesOf(value.getType(), "i32") + ">(" +
           expression + ")";
  }

  [[nodiscard]] auto define(Operation* op, Value result,
                            const std::string& expression, unsigned depth)
      -> LogicalResult {
    auto type = typeName(op, result.getType());
    if (failed(type)) return failure();
    auto name = "cxx_v" + std::to_string(next_++);
    indent(depth);
    out_ << "let " << name << ": " << *type << " = " << expression << ";\n";
    names_[result] = name;
    return success();
  }

  [[nodiscard]] auto declareResults(Operation* op, unsigned depth)
      -> LogicalResult {
    for (auto result : op->getResults()) {
      auto type = typeName(op, result.getType());
      if (failed(type)) return failure();
      auto name = "cxx_r" + std::to_string(next_++);
      indent(depth);
      out_ << "var " << name << ": " << *type << ";\n";
      names_[result] = name;
    }
    return success();
  }

  void assign(ValueRange targets, ValueRange values, unsigned depth) {
    llvm::SmallVector<std::string> snapshots;
    for (auto value : values) {
      auto name = "cxx_n" + std::to_string(next_++);
      indent(depth);
      out_ << "let " << name << " = " << use(value) << ";\n";
      snapshots.push_back(name);
    }
    for (auto [target, snapshot] : llvm::zip(targets, snapshots)) {
      indent(depth);
      out_ << use(target) << " = " << snapshot << ";\n";
    }
  }

  [[nodiscard]] auto writeBody(Block& block, unsigned depth)
      -> FailureOr<Operation*> {
    for (auto& op : block.without_terminator())
      if (failed(writeOperation(&op, depth))) return failure();
    return block.getTerminator();
  }

  [[nodiscard]] auto writeYieldingRegion(Region& region, Operation* parent,
                                         unsigned depth) -> LogicalResult {
    auto terminator = writeBody(region.front(), depth);
    if (failed(terminator)) return failure();
    auto yield = dyn_cast<scf::YieldOp>(*terminator);
    if (!yield) return (*terminator)->emitError("expected scf.yield");
    assign(parent->getResults(), yield.getOperands(), depth);
    return success();
  }

  [[nodiscard]] auto literal(Operation* op, Attribute value)
      -> FailureOr<std::string> {
    if (auto dense = dyn_cast<DenseElementsAttr>(value)) {
      auto type = typeName(op, dense.getType());
      if (failed(type)) return failure();
      std::string text = *type + "(";
      auto elements = llvm::to_vector(dense.getValues<Attribute>());
      for (auto [index, element] : llvm::enumerate(elements)) {
        auto scalar = literal(op, element);
        if (failed(scalar)) return failure();
        if (index) text += ", ";
        text += *scalar;
      }
      return text + ")";
    }
    if (auto boolean = dyn_cast<BoolAttr>(value))
      return std::string(boolean.getValue() ? "true" : "false");
    if (auto integer = dyn_cast<IntegerAttr>(value)) {
      if (integer.getType().isSignlessInteger(1))
        return std::string(integer.getValue().isZero() ? "false" : "true");
      auto number = integer.getValue().sextOrTrunc(32).getSExtValue();
      if (number == INT32_MIN) return std::string("(-2147483647i - 1i)");
      if (number < 0) return "(-" + std::to_string(-number) + "i)";
      return std::to_string(number) + "i";
    }
    if (auto real = dyn_cast<FloatAttr>(value)) {
      auto number = real.getValue();
      if (!number.isFinite())
        return op->emitError(
            "WGSL cannot represent a non-finite floating-point constant");
      char buffer[64];
      number.convertToHexString(buffer, 0, false,
                                llvm::APFloat::rmNearestTiesToEven);
      return "(" + std::string(buffer) + "f)";
    }
    return op->emitError("WGSL emission does not support constant ") << value;
  }

  [[nodiscard]] auto zeroValue(Operation* op, Type type)
      -> FailureOr<std::string> {
    auto name = typeName(op, type);
    if (failed(name)) return failure();
    if (isa<VectorType>(type)) return *name + "()";
    if (*name == "bool") return std::string("false");
    if (*name == "f32") return std::string("0.0f");
    return std::string("0i");
  }

  [[nodiscard]] auto integerComparison(arith::CmpIPredicate predicate)
      -> std::pair<llvm::StringRef, bool> {
    switch (predicate) {
      case arith::CmpIPredicate::eq:
        return {"==", false};
      case arith::CmpIPredicate::ne:
        return {"!=", false};
      case arith::CmpIPredicate::slt:
        return {"<", false};
      case arith::CmpIPredicate::sle:
        return {"<=", false};
      case arith::CmpIPredicate::sgt:
        return {">", false};
      case arith::CmpIPredicate::sge:
        return {">=", false};
      case arith::CmpIPredicate::ult:
        return {"<", true};
      case arith::CmpIPredicate::ule:
        return {"<=", true};
      case arith::CmpIPredicate::ugt:
        return {">", true};
      case arith::CmpIPredicate::uge:
        return {">=", true};
    }
    return {"", false};
  }

  [[nodiscard]] auto floatComparison(arith::CmpFPredicate predicate)
      -> llvm::StringRef {
    switch (predicate) {
      case arith::CmpFPredicate::OEQ:
      case arith::CmpFPredicate::UEQ:
        return "==";
      case arith::CmpFPredicate::ONE:
      case arith::CmpFPredicate::UNE:
        return "!=";
      case arith::CmpFPredicate::OLT:
      case arith::CmpFPredicate::ULT:
        return "<";
      case arith::CmpFPredicate::OLE:
      case arith::CmpFPredicate::ULE:
        return "<=";
      case arith::CmpFPredicate::OGT:
      case arith::CmpFPredicate::UGT:
        return ">";
      case arith::CmpFPredicate::OGE:
      case arith::CmpFPredicate::UGE:
        return ">=";
      default:
        return "";
    }
  }

  [[nodiscard]] auto infix(Operation* op, llvm::StringRef symbol,
                           unsigned depth) -> LogicalResult {
    return define(op, op->getResult(0),
                  "(" + use(op->getOperand(0)) + " " + symbol.str() + " " +
                      use(op->getOperand(1)) + ")",
                  depth);
  }

  [[nodiscard]] auto unsignedInfix(Operation* op, llvm::StringRef symbol,
                                   unsigned depth) -> LogicalResult {
    return define(
        op, op->getResult(0),
        signedResult(op->getResult(0), unsignedUse(op->getOperand(0)) + " " +
                                           symbol.str() + " " +
                                           unsignedUse(op->getOperand(1))),
        depth);
  }

  [[nodiscard]] auto call(Operation* op, llvm::StringRef function,
                          unsigned depth) -> LogicalResult {
    std::string arguments;
    for (auto operand : op->getOperands()) {
      if (!arguments.empty()) arguments += ", ";
      arguments += use(operand);
    }
    return define(op, op->getResult(0), function.str() + "(" + arguments + ")",
                  depth);
  }

  [[nodiscard]] auto writeRound(math::RoundOp round, unsigned depth)
      -> LogicalResult {
    auto type = typeName(round, round.getType());
    if (failed(type)) return failure();
    auto value = use(round.getOperand());
    auto truncated = "cxx_v" + std::to_string(next_++);
    indent(depth);
    out_ << "let " << truncated << ": " << *type << " = trunc(" << value
         << ");\n";
    auto half = withLanesOf(round.getType(), "f32") + "(0.5f)";
    return define(round, round.getResult(),
                  "select(" + truncated + ", " + truncated + " + sign(" +
                      value + "), abs(" + value + " - " + truncated +
                      ") >= " + half + ")",
                  depth);
  }

  [[nodiscard]] auto writeCopySign(math::CopySignOp copySign, unsigned depth)
      -> LogicalResult {
    auto type = typeName(copySign, copySign.getType());
    if (failed(type)) return failure();
    auto bits = withLanesOf(copySign.getType(), "u32");
    auto magnitude = "(bitcast<" + bits + ">(" + use(copySign.getLhs()) +
                     ") & " + bits + "(0x7fffffffu))";
    auto sign = "(bitcast<" + bits + ">(" + use(copySign.getRhs()) + ") & " +
                bits + "(0x80000000u))";
    return define(copySign, copySign.getResult(),
                  "bitcast<" + *type + ">(" + magnitude + " | " + sign + ")",
                  depth);
  }

  [[nodiscard]] auto writeLog10(math::Log10Op log10, unsigned depth)
      -> LogicalResult {
    auto scale =
        literal(log10, FloatAttr::get(Float32Type::get(log10.getContext()),
                                      1.0 / std::log(10.0)));
    if (failed(scale)) return failure();
    return define(log10, log10.getResult(),
                  "(log(" + use(log10.getOperand()) + ") * " +
                      withLanesOf(log10.getType(), "f32") + "(" + *scale + "))",
                  depth);
  }

  [[nodiscard]] auto unsignedCall(Operation* op, llvm::StringRef function,
                                  unsigned depth) -> LogicalResult {
    return define(
        op, op->getResult(0),
        signedResult(op->getResult(0),
                     function.str() + "(" + unsignedUse(op->getOperand(0)) +
                         ", " + unsignedUse(op->getOperand(1)) + ")"),
        depth);
  }

  [[nodiscard]] auto isBoolean(Value value) -> bool {
    return getElementTypeOrSelf(value.getType()).isSignlessInteger(1);
  }

  [[nodiscard]] auto alias(Value result, Value source) -> LogicalResult {
    names_[result] = use(source);
    return success();
  }

  [[nodiscard]] auto writeOperation(Operation* op, unsigned depth)
      -> LogicalResult {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case([&](arith::ConstantOp constant) -> LogicalResult {
          auto value = literal(op, constant.getValue());
          if (failed(value)) return failure();
          names_[constant.getResult()] = *value;
          return success();
        })
        .Case([&](ub::PoisonOp poison) -> LogicalResult {
          auto value = zeroValue(op, poison.getType());
          if (failed(value)) return failure();
          names_[poison.getResult()] = *value;
          return success();
        })
        .Case([&](gpu::GlobalIdOp id) {
          return writeInvocationId(op, "cxx_global_id", id.getDimension(),
                                   depth);
        })
        .Case([&](gpu::ThreadIdOp id) {
          return writeInvocationId(op, "cxx_local_id", id.getDimension(),
                                   depth);
        })
        .Case([&](gpu::BlockIdOp id) {
          return writeInvocationId(op, "cxx_workgroup_id", id.getDimension(),
                                   depth);
        })
        .Case([&](vector::BroadcastOp broadcast) -> LogicalResult {
          auto type = typeName(op, broadcast.getType());
          if (failed(type)) return failure();
          if (isa<VectorType>(broadcast.getSourceType()))
            return op->emitError("WGSL emission broadcasts scalars only");
          return define(op, broadcast.getResult(),
                        *type + "(" + use(broadcast.getSource()) + ")", depth);
        })
        .Case([&](vector::FromElementsOp fromElements) -> LogicalResult {
          auto type = typeName(op, fromElements.getType());
          if (failed(type)) return failure();
          std::string elements;
          for (auto element : fromElements.getElements()) {
            if (!elements.empty()) elements += ", ";
            elements += use(element);
          }
          return define(op, fromElements.getResult(),
                        *type + "(" + elements + ")", depth);
        })
        .Case([&](vector::ExtractOp extract) -> LogicalResult {
          auto position = lanePosition(op, extract.getMixedPosition());
          if (failed(position)) return failure();
          return define(op, extract.getResult(),
                        use(extract.getSource()) + "[" + *position + "]",
                        depth);
        })
        .Case(
            [&](vector::InsertOp insert) { return writeInsert(insert, depth); })
        .Case([&](vector::ShuffleOp shuffle) {
          return writeShuffle(shuffle, depth);
        })
        .Case([&](memref::AllocaOp allocation) {
          return writePrivateVariable(allocation, depth);
        })
        .Case([&](memref::LoadOp load) { return writeLoad(load, depth); })
        .Case([&](memref::StoreOp store) { return writeStore(store, depth); })
        .Case([&](scf::IfOp ifOp) { return writeIf(ifOp, depth); })
        .Case([&](scf::WhileOp loop) { return writeWhile(loop, depth); })
        .Case([&](scf::ForOp loop) { return writeFor(loop, depth); })
        .Case([&](scf::IndexSwitchOp switchOp) {
          return writeSwitch(switchOp, depth);
        })
        .Case([&](arith::AddIOp) { return infix(op, "+", depth); })
        .Case([&](arith::SubIOp) { return infix(op, "-", depth); })
        .Case([&](arith::MulIOp) { return infix(op, "*", depth); })
        .Case([&](arith::DivSIOp) { return infix(op, "/", depth); })
        .Case([&](arith::RemSIOp) { return infix(op, "%", depth); })
        .Case([&](arith::DivUIOp) { return unsignedInfix(op, "/", depth); })
        .Case([&](arith::RemUIOp) { return unsignedInfix(op, "%", depth); })
        .Case([&](arith::AndIOp) { return infix(op, "&", depth); })
        .Case([&](arith::OrIOp) { return infix(op, "|", depth); })
        .Case([&](arith::XOrIOp xorOp) {
          return infix(op, isBoolean(xorOp.getResult()) ? "!=" : "^", depth);
        })
        .Case([&](arith::ShLIOp shift) {
          return define(op, shift.getResult(),
                        "(" + use(shift.getLhs()) + " << " +
                            unsignedUse(shift.getRhs()) + ")",
                        depth);
        })
        .Case([&](arith::ShRSIOp shift) {
          return define(op, shift.getResult(),
                        "(" + use(shift.getLhs()) + " >> " +
                            unsignedUse(shift.getRhs()) + ")",
                        depth);
        })
        .Case([&](arith::ShRUIOp) { return unsignedInfix(op, ">>", depth); })
        .Case([&](arith::MaxSIOp) { return call(op, "max", depth); })
        .Case([&](arith::MinSIOp) { return call(op, "min", depth); })
        .Case([&](arith::MaxUIOp) { return unsignedCall(op, "max", depth); })
        .Case([&](arith::MinUIOp) { return unsignedCall(op, "min", depth); })
        .Case([&](math::AbsFOp) { return call(op, "abs", depth); })
        .Case([&](math::CeilOp) { return call(op, "ceil", depth); })
        .Case([&](math::FloorOp) { return call(op, "floor", depth); })
        .Case([&](math::TruncOp) { return call(op, "trunc", depth); })
        .Case([&](math::RoundEvenOp) { return call(op, "round", depth); })
        .Case([&](math::RoundOp round) { return writeRound(round, depth); })
        .Case([&](math::CopySignOp copySign) {
          return writeCopySign(copySign, depth);
        })
        .Case([&](math::FmaOp) { return call(op, "fma", depth); })
        .Case([&](math::SqrtOp) { return call(op, "sqrt", depth); })
        .Case([&](math::SinOp) { return call(op, "sin", depth); })
        .Case([&](math::CosOp) { return call(op, "cos", depth); })
        .Case([&](math::TanOp) { return call(op, "tan", depth); })
        .Case([&](math::AsinOp) { return call(op, "asin", depth); })
        .Case([&](math::AcosOp) { return call(op, "acos", depth); })
        .Case([&](math::AtanOp) { return call(op, "atan", depth); })
        .Case([&](math::Atan2Op) { return call(op, "atan2", depth); })
        .Case([&](math::SinhOp) { return call(op, "sinh", depth); })
        .Case([&](math::CoshOp) { return call(op, "cosh", depth); })
        .Case([&](math::TanhOp) { return call(op, "tanh", depth); })
        .Case([&](math::ExpOp) { return call(op, "exp", depth); })
        .Case([&](math::Exp2Op) { return call(op, "exp2", depth); })
        .Case([&](math::LogOp) { return call(op, "log", depth); })
        .Case([&](math::Log2Op) { return call(op, "log2", depth); })
        .Case([&](math::Log10Op log10) { return writeLog10(log10, depth); })
        .Case([&](math::PowFOp) { return call(op, "pow", depth); })
        .Case([&](arith::AddFOp) { return infix(op, "+", depth); })
        .Case([&](arith::SubFOp) { return infix(op, "-", depth); })
        .Case([&](arith::MulFOp) { return infix(op, "*", depth); })
        .Case([&](arith::DivFOp) { return infix(op, "/", depth); })
        .Case([&](arith::RemFOp) { return infix(op, "%", depth); })
        .Case<arith::MaximumFOp, arith::MaxNumFOp>(
            [&](Operation*) { return call(op, "max", depth); })
        .Case<arith::MinimumFOp, arith::MinNumFOp>(
            [&](Operation*) { return call(op, "min", depth); })
        .Case([&](arith::NegFOp negate) {
          return define(op, negate.getResult(),
                        "(-" + use(negate.getOperand()) + ")", depth);
        })
        .Case([&](arith::CmpIOp compare) { return writeCmpI(compare, depth); })
        .Case([&](arith::CmpFOp compare) { return writeCmpF(compare, depth); })
        .Case([&](arith::SelectOp select) {
          return define(op, select.getResult(),
                        "select(" + use(select.getFalseValue()) + ", " +
                            use(select.getTrueValue()) + ", " +
                            use(select.getCondition()) + ")",
                        depth);
        })
        .Case([&](arith::ExtUIOp extend) {
          return writeBooleanExtension(extend, extend.getIn(), "1i", depth);
        })
        .Case([&](arith::ExtSIOp extend) {
          return writeBooleanExtension(extend, extend.getIn(), "(-1i)", depth);
        })
        .Case([&](arith::TruncIOp truncate) -> LogicalResult {
          if (!isBoolean(truncate.getResult()))
            return op->emitError(
                "WGSL emission supports truncation to i1 only");
          auto type = truncate.getIn().getType();
          return define(op, truncate.getResult(),
                        "((" + use(truncate.getIn()) + " & " +
                            splat(type, "1i") + ") != " + splat(type, "0i") +
                            ")",
                        depth);
        })
        .Case<arith::IndexCastOp, arith::IndexCastUIOp>([&](Operation*) {
          return alias(op->getResult(0), op->getOperand(0));
        })
        .Case([&](arith::SIToFPOp convert) {
          return define(op, convert.getResult(),
                        withLanesOf(convert.getType(), "f32") + "(" +
                            use(convert.getIn()) + ")",
                        depth);
        })
        .Case([&](arith::UIToFPOp convert) {
          return define(op, convert.getResult(),
                        withLanesOf(convert.getType(), "f32") + "(" +
                            unsignedUse(convert.getIn()) + ")",
                        depth);
        })
        .Case([&](arith::FPToSIOp convert) {
          return define(op, convert.getResult(),
                        withLanesOf(convert.getType(), "i32") + "(" +
                            use(convert.getIn()) + ")",
                        depth);
        })
        .Case([&](arith::FPToUIOp convert) {
          return define(op, convert.getResult(),
                        signedResult(convert.getResult(),
                                     withLanesOf(convert.getType(), "u32") +
                                         "(" + use(convert.getIn()) + ")"),
                        depth);
        })
        .Case([&](arith::BitcastOp cast) -> LogicalResult {
          auto type = typeName(op, cast.getType());
          if (failed(type)) return failure();
          return define(op, cast.getResult(),
                        "bitcast<" + *type + ">(" + use(cast.getIn()) + ")",
                        depth);
        })
        .Default([&](Operation*) -> LogicalResult {
          return op->emitError("WGSL emission does not support operation '")
                 << op->getName() << "'";
        });
  }

  [[nodiscard]] auto writeBooleanExtension(Operation* op, Value input,
                                           llvm::StringRef one, unsigned depth)
      -> LogicalResult {
    if (!isBoolean(input))
      return op->emitError("WGSL emission supports extension from i1 only");
    auto type = op->getResult(0).getType();
    return define(op, op->getResult(0),
                  "select(" + splat(type, "0i") + ", " + splat(type, one) +
                      ", " + use(input) + ")",
                  depth);
  }

  [[nodiscard]] auto writeInvocationId(Operation* op, llvm::StringRef builtin,
                                       gpu::Dimension dimension, unsigned depth)
      -> LogicalResult {
    auto component = "xyz"[static_cast<unsigned>(dimension)];
    return define(op, op->getResult(0),
                  "bitcast<i32>(" + builtin.str() + "." + component + ")",
                  depth);
  }

  [[nodiscard]] auto lanePosition(Operation* op,
                                  llvm::ArrayRef<OpFoldResult> position)
      -> FailureOr<std::string> {
    if (position.size() != 1)
      return op->emitError("WGSL emission indexes one vector lane");
    if (auto value = dyn_cast<Value>(position.front())) return use(value);
    auto lane = cast<IntegerAttr>(cast<Attribute>(position.front()));
    return std::to_string(lane.getInt()) + "i";
  }

  [[nodiscard]] auto writeInsert(vector::InsertOp insert, unsigned depth)
      -> LogicalResult {
    auto position = lanePosition(insert, insert.getMixedPosition());
    if (failed(position)) return failure();
    auto type = typeName(insert, insert.getType());
    if (failed(type)) return failure();
    auto name = "cxx_v" + std::to_string(next_++);
    indent(depth);
    out_ << "var " << name << ": " << *type << " = " << use(insert.getDest())
         << ";\n";
    indent(depth);
    out_ << name << "[" << *position << "] = " << use(insert.getValueToStore())
         << ";\n";
    names_[insert.getResult()] = name;
    return success();
  }

  [[nodiscard]] auto writeShuffle(vector::ShuffleOp shuffle, unsigned depth)
      -> LogicalResult {
    auto type = typeName(shuffle, shuffle.getType());
    if (failed(type)) return failure();
    auto firstLanes = shuffle.getV1VectorType().getNumElements();
    std::string lanes;
    for (auto lane : shuffle.getMask()) {
      if (!lanes.empty()) lanes += ", ";
      if (lane == vector::ShuffleOp::kPoisonIndex) {
        auto zero =
            zeroValue(shuffle, shuffle.getResultVectorType().getElementType());
        if (failed(zero)) return failure();
        lanes += *zero;
      } else if (lane < firstLanes) {
        lanes += use(shuffle.getV1()) + "[" + std::to_string(lane) + "i]";
      } else {
        lanes += use(shuffle.getV2()) + "[" +
                 std::to_string(lane - firstLanes) + "i]";
      }
    }
    return define(shuffle, shuffle.getResult(), *type + "(" + lanes + ")",
                  depth);
  }

  [[nodiscard]] auto writeCmpI(arith::CmpIOp compare, unsigned depth)
      -> LogicalResult {
    auto [symbol, isUnsigned] = integerComparison(compare.getPredicate());
    if (isBoolean(compare.getLhs()) && isUnsigned)
      return compare.emitError("WGSL emission does not order booleans");
    if (!isUnsigned) return infix(compare, symbol, depth);
    return define(compare, compare.getResult(),
                  "(" + unsignedUse(compare.getLhs()) + " " + symbol.str() +
                      " " + unsignedUse(compare.getRhs()) + ")",
                  depth);
  }

  [[nodiscard]] auto writeCmpF(arith::CmpFOp compare, unsigned depth)
      -> LogicalResult {
    auto predicate = compare.getPredicate();
    if (predicate == arith::CmpFPredicate::AlwaysTrue)
      return define(compare, compare.getResult(), "true", depth);
    if (predicate == arith::CmpFPredicate::AlwaysFalse)
      return define(compare, compare.getResult(), "false", depth);
    auto symbol = floatComparison(predicate);
    if (symbol.empty())
      return compare.emitError(
          "WGSL assumes finite floating point and cannot test for NaN");
    return infix(compare, symbol, depth);
  }

  [[nodiscard]] auto bufferOf(Operation* op, Value memref)
      -> FailureOr<BufferView> {
    auto found = buffers_.find(memref);
    if (found == buffers_.end())
      return op->emitError("WGSL access has no declared resource root");
    return found->second;
  }

  [[nodiscard]] auto privateElement(memref::LoadOp load) -> std::string {
    return privateElement(load.getMemRef(), load.getIndices());
  }

  [[nodiscard]] auto privateElement(Value memref, ValueRange indices)
      -> std::string {
    auto element = names_.lookup(memref);
    for (auto index : indices) element += "[" + use(index) + "]";
    return element;
  }

  [[nodiscard]] auto writePrivateVariable(memref::AllocaOp allocation,
                                          unsigned depth) -> LogicalResult {
    auto type = allocation.getType();
    auto element = typeName(allocation, type.getElementType());
    if (failed(element)) return failure();
    auto spelled = *element;
    for (auto extent : llvm::reverse(type.getShape()))
      spelled = "array<" + spelled + ", " + std::to_string(extent) + ">";
    auto name = "cxx_p" + std::to_string(next_++);
    indent(depth);
    out_ << "var " << name << ": " << spelled << ";\n";
    names_[allocation.getResult()] = name;
    private_.insert(allocation.getResult());
    return success();
  }

  [[nodiscard]] auto writeFor(scf::ForOp loop, unsigned depth)
      -> LogicalResult {
    auto& body = *loop.getBody();
    auto induction = "cxx_i" + std::to_string(next_++);
    names_[loop.getInductionVar()] = induction;
    for (auto [argument, initial] :
         llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs())) {
      auto type = typeName(loop, argument.getType());
      if (failed(type)) return failure();
      auto name = "cxx_l" + std::to_string(next_++);
      indent(depth);
      out_ << "var " << name << ": " << *type << " = " << use(initial) << ";\n";
      names_[argument] = name;
    }
    indent(depth);
    out_ << "for (var " << induction << ": i32 = " << use(loop.getLowerBound())
         << "; " << induction << " < " << use(loop.getUpperBound()) << "; "
         << induction << " += " << use(loop.getStep()) << ") {\n";
    auto yield = writeBody(body, depth + 1);
    if (failed(yield)) return failure();
    auto yieldOp = dyn_cast<scf::YieldOp>(*yield);
    if (!yieldOp) return (*yield)->emitError("expected scf.yield");
    assign(loop.getRegionIterArgs(), yieldOp.getOperands(), depth + 1);
    indent(depth);
    out_ << "}\n";
    for (auto [result, argument] :
         llvm::zip(loop.getResults(), loop.getRegionIterArgs()))
      names_[result] = use(argument);
    return success();
  }

  [[nodiscard]] auto writeLoad(memref::LoadOp load, unsigned depth)
      -> LogicalResult {
    if (private_.contains(load.getMemRef()))
      return define(load, load.getResult(), privateElement(load), depth);
    auto buffer = bufferOf(load, load.getMemRef());
    if (failed(buffer)) return failure();
    auto element = buffer->name;
    if (!buffer->isUniform)
      element += "[" + use(load.getIndices().front()) + "]";
    if (buffer->isUnsigned) {
      auto type = typeName(load, load.getType());
      if (failed(type)) return failure();
      element = "bitcast<" + *type + ">(" + element + ")";
    }
    return define(load, load.getResult(), element, depth);
  }

  [[nodiscard]] auto writeStore(memref::StoreOp store, unsigned depth)
      -> LogicalResult {
    if (private_.contains(store.getMemRef())) {
      indent(depth);
      out_ << privateElement(store.getMemRef(), store.getIndices()) << " = "
           << use(store.getValueToStore()) << ";\n";
      return success();
    }
    auto buffer = bufferOf(store, store.getMemRef());
    if (failed(buffer)) return failure();
    if (buffer->isUniform)
      return store.emitError("WGSL cannot write a uniform resource");
    auto value = use(store.getValueToStore());
    if (buffer->isUnsigned) value = "bitcast<u32>(" + value + ")";
    indent(depth);
    out_ << buffer->name << "[" << use(store.getIndices().front())
         << "] = " << value << ";\n";
    return success();
  }

  [[nodiscard]] auto writeIf(scf::IfOp ifOp, unsigned depth) -> LogicalResult {
    if (failed(declareResults(ifOp, depth))) return failure();
    indent(depth);
    out_ << "if (" << use(ifOp.getCondition()) << ") {\n";
    if (failed(writeYieldingRegion(ifOp.getThenRegion(), ifOp, depth + 1)))
      return failure();
    indent(depth);
    if (ifOp.getElseRegion().empty()) {
      out_ << "}\n";
      return success();
    }
    out_ << "} else {\n";
    if (failed(writeYieldingRegion(ifOp.getElseRegion(), ifOp, depth + 1)))
      return failure();
    indent(depth);
    out_ << "}\n";
    return success();
  }

  [[nodiscard]] auto writeSwitch(scf::IndexSwitchOp switchOp, unsigned depth)
      -> LogicalResult {
    if (failed(declareResults(switchOp, depth))) return failure();
    indent(depth);
    out_ << "switch (" << use(switchOp.getArg()) << ") {\n";
    for (auto [value, region] :
         llvm::zip(switchOp.getCases(), switchOp.getCaseRegions())) {
      indent(depth + 1);
      out_ << "case " << value << "i: {\n";
      if (failed(writeYieldingRegion(region, switchOp, depth + 2)))
        return failure();
      indent(depth + 1);
      out_ << "}\n";
    }
    indent(depth + 1);
    out_ << "default: {\n";
    if (failed(writeYieldingRegion(switchOp.getDefaultRegion(), switchOp,
                                   depth + 2)))
      return failure();
    indent(depth + 1);
    out_ << "}\n";
    indent(depth);
    out_ << "}\n";
    return success();
  }

  [[nodiscard]] auto writeWhile(scf::WhileOp loop, unsigned depth)
      -> LogicalResult {
    auto& before = loop.getBefore().front();
    auto& after = loop.getAfter().front();
    for (auto [argument, initial] :
         llvm::zip(before.getArguments(), loop.getInits())) {
      auto type = typeName(loop, argument.getType());
      if (failed(type)) return failure();
      auto name = "cxx_l" + std::to_string(next_++);
      indent(depth);
      out_ << "var " << name << ": " << *type << " = " << use(initial) << ";\n";
      names_[argument] = name;
    }
    if (failed(declareResults(loop, depth))) return failure();
    for (auto [argument, result] :
         llvm::zip(after.getArguments(), loop.getResults()))
      names_[argument] = use(result);
    indent(depth);
    out_ << "loop {\n";
    auto condition = writeBody(before, depth + 1);
    if (failed(condition)) return failure();
    auto conditionOp = dyn_cast<scf::ConditionOp>(*condition);
    if (!conditionOp) return (*condition)->emitError("expected scf.condition");
    assign(loop.getResults(), conditionOp.getArgs(), depth + 1);
    indent(depth + 1);
    out_ << "if (!" << use(conditionOp.getCondition()) << ") {\n";
    indent(depth + 2);
    out_ << "break;\n";
    indent(depth + 1);
    out_ << "}\n";
    auto yield = writeBody(after, depth + 1);
    if (failed(yield)) return failure();
    auto yieldOp = dyn_cast<scf::YieldOp>(*yield);
    if (!yieldOp) return (*yield)->emitError("expected scf.yield");
    assign(before.getArguments(), yieldOp.getOperands(), depth + 1);
    indent(depth);
    out_ << "}\n";
    return success();
  }

  gpu::GPUFuncOp function_;
  const KernelEntry& entry_;
  std::string text_;
  llvm::raw_string_ostream out_;
  DenseMap<Value, std::string> names_;
  DenseMap<Value, BufferView> buffers_;
  DenseSet<Value> private_;
  unsigned next_ = 0;
};

}  // namespace

auto compileWGSLKernel(mlir::ModuleOp source, llvm::StringRef kernel)
    -> mlir::FailureOr<KernelArtifact> {
  using namespace mlir;
  auto entry = kernelEntryOf(source, kernel);
  if (failed(entry)) return failure();
  if (!isUsableEntryName(kernel)) {
    source.emitError("kernel name '")
        << kernel << "' is reserved in WGSL output";
    return failure();
  }
  auto working = lowerKernelToGPU(source, kernel);
  if (failed(working)) return failure();
  gpu::GPUFuncOp function;
  working->get()->walk([&](gpu::GPUFuncOp candidate) {
    if (candidate.getName() == kernel) function = candidate;
  });
  if (!function) {
    source.emitError("selected kernel was not lowered to a GPU function");
    return failure();
  }
  auto wgsl = WGSLWriter(function, *entry).write();
  if (failed(wgsl)) return failure();
  KernelArtifact artifact;
  artifact.bytes.assign(wgsl->begin(), wgsl->end());
  artifact.manifest =
      kernelManifest(*entry, "wgsl", artifact.bytes,
                     llvm::json::Object{{"targetEnvironment", "webgpu"}});
  return artifact;
}

}  // namespace cxx
