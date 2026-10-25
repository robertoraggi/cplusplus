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
#include <cxx/mlir/atomic_ops.h>
#include <cxx/mlir/buffer_layout.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/device_kernel.h>
#include <cxx/mlir/kernel_manifest.h>
#include <cxx/mlir/storage_address.h>
#include <cxx/mlir/webgpu_policy.h>
#include <cxx/mlir/wgsl_name_encoder.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/SymbolTable.h>

#include <array>
#include <cmath>
#include <string>

namespace cxx {
namespace {

using namespace mlir;

[[nodiscard]] auto legalizeWGSLFunctionNames(gpu::GPUModuleOp module)
    -> LogicalResult {
  SymbolTable symbols(module);
  for (auto function : module.getOps<func::FuncOp>()) {
    auto original = function.getSymName();
    auto name = WGSLNameEncoder::encodeFunction(original);
    if (name == original) continue;
    llvm::SmallString<64> unique(name);
    if (symbols.lookup(name)) {
      unsigned counter = 0;
      unique = SymbolTable::generateSymbolName<64>(
          name, [&](StringRef candidate) { return symbols.lookup(candidate); },
          counter);
    }
    if (failed(symbols.rename(function, unique)))
      return function.emitError("cannot encode the WGSL function name");
  }
  return success();
}

struct BufferView {
  std::string name;
  bool isUniform = false;
  bool isUnsigned = false;
  bool readOnly = false;
};

struct StorageView {
  std::string expression;
  Type type;
  bool isUnsigned = false;
  bool readOnly = false;
  bool atomic = false;
};

[[nodiscard]] auto bufferValueTypeName(const mlir::cxx::KernelResource& buffer)
    -> std::string {
  auto scalar = mlir::cxx::kernelBufferScalarTypeName(buffer).str();
  auto components = mlir::cxx::kernelBufferComponents(buffer);
  if (components == 1) return scalar;
  return "vec" + std::to_string(components) + "<" + scalar + ">";
}

class WGSLWriter {
 public:
  WGSLWriter(gpu::GPUModuleOp module,
             const mlir::cxx::KernelModuleContract& contract)
      : module_(module), contract_(contract), out_(text_) {}

  [[nodiscard]] auto write() -> FailureOr<std::string> {
    if (failed(collectAtomicStorage())) return failure();
    if (failed(writeBindings())) return failure();
    if (failed(writeWorkgroupStorage())) return failure();
    for (auto helper : module_.getOps<func::FuncOp>())
      if (failed(writeFunction(helper))) return failure();
    for (const auto& entry : contract_.entries) {
      function_ = module_.lookupSymbol<gpu::GPUFuncOp>(entry.name);
      names_.clear();
      objects_.clear();
      const auto& workgroup = entry.contract.workgroup;
      out_ << "@compute @workgroup_size(" << workgroup[0] << ", "
           << workgroup[1] << ", " << workgroup[2] << ")\nfn " << entry.name
           << "(" << entryParameters() << ") {\n";
      auto terminator = writeBody(function_.getBody().front(), 1);
      if (failed(terminator)) return failure();
      if (!isa<gpu::ReturnOp>(*terminator))
        return (*terminator)->emitError("unexpected WGSL function terminator");
      out_ << "}\n\n";
    }
    if (workgroupPointers_)
      languageFeatures_.push_back("unrestricted_pointer_parameters");
    if (mlir::cxx::declaresAggregateUniforms(contract_))
      languageFeatures_.push_back(mlir::cxx::kWebGPUStandardUniformLayout);
    std::string requirements;
    if (!languageFeatures_.empty())
      requirements =
          "requires " + llvm::join(languageFeatures_, ", ") + ";\n\n";
    return requirements + typeDeclarations_ + text_;
  }

  [[nodiscard]] auto languageFeatures() const -> ArrayRef<llvm::StringRef> {
    return languageFeatures_;
  }

 private:
  [[nodiscard]] static auto isWorkgroupStorage(MemRefType type) -> bool {
    auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace());
    return space && space.getValue() == gpu::AddressSpace::Workgroup;
  }

  [[nodiscard]] auto isAtomicStorage(Value root) const -> bool {
    if (auto index = mlir::cxx::deviceResourceIndex(root))
      return atomicResources_.contains(index.getInt());
    return atomicRoots_.contains(root);
  }

  auto markAtomicStorage(Value root) -> bool {
    if (auto index = mlir::cxx::deviceResourceIndex(root))
      return atomicResources_.insert(index.getInt()).second;
    return atomicRoots_.insert(root).second;
  }

  [[nodiscard]] auto collectAtomicStorage() -> LogicalResult {
    auto result = module_.walk([&](Operation* op) {
      auto pointer = mlir::cxx::atomicAddress(op);
      if (!pointer) return WalkResult::advance();
      auto address = mlir::cxx::resolveStorageAddress(pointer);
      if (failed(address)) {
        op->emitError("WGSL atomic object has no storage root");
        return WalkResult::interrupt();
      }
      markAtomicStorage(address->root);
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
    bool changed;
    do {
      changed = false;
      module_.walk([&](func::CallOp call) {
        auto target = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        if (!target || target.getNumArguments() != call.getNumOperands())
          return;
        for (auto [argument, operand] :
             llvm::zip(target.getArguments(), call.getOperands())) {
          if (!isa<MemRefType>(argument.getType())) continue;
          if (!isAtomicStorage(argument) && !isAtomicStorage(operand)) continue;
          changed |= markAtomicStorage(argument);
          changed |= markAtomicStorage(operand);
        }
      });
    } while (changed);
    return success();
  }

  [[nodiscard]] auto storageTypeName(Operation* op, Type type, bool atomic)
      -> FailureOr<std::string> {
    if (!atomic) return typeName(op, type);
    if (type.isInteger(32)) return std::string("atomic<i32>");
    if (auto array = dyn_cast<mlir::cxx::ArrayType>(type)) {
      auto element = storageTypeName(op, array.getElementType(), true);
      if (failed(element)) return failure();
      return "array<" + *element + ", " + std::to_string(array.getSize()) + ">";
    }
    return op->emitError(
        "WGSL atomic storage requires 32-bit integer elements");
  }

  [[nodiscard]] auto writeWorkgroupStorage() -> LogicalResult {
    unsigned index = 0;
    auto result = module_.walk([&](memref::AllocOp allocation) {
      auto type = allocation.getType();
      if (!isWorkgroupStorage(type) || !type.hasStaticShape()) {
        allocation.emitError(
            "WGSL allocation requires statically sized workgroup storage");
        return WalkResult::interrupt();
      }
      auto spelling = storageTypeName(allocation, type.getElementType(),
                                      isAtomicStorage(allocation));
      if (failed(spelling)) return WalkResult::interrupt();
      for (auto extent : llvm::reverse(type.getShape()))
        *spelling = "array<" + *spelling + ", " + std::to_string(extent) + ">";
      auto name = WGSLNameEncoder::encodeWorkgroupVariable(index++);
      while (SymbolTable::lookupSymbolIn(module_, name)) name += "_";
      sharedNames_[allocation] = name;
      out_ << "var<workgroup> " << name << ": " << *spelling << ";\n\n";
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  [[nodiscard]] auto writeBindings() -> LogicalResult {
    for (auto [index, buffer] : llvm::enumerate(contract_.resources)) {
      auto spelled = bindingValueTypeName(buffer);
      if (failed(spelled)) return failure();
      auto value = *spelled;
      if (atomicResources_.contains(index)) {
        if (buffer.readOnly ||
            buffer.kind == mlir::cxx::KernelBufferKind::Uniform)
          return module_.emitError(
              "WGSL atomics require a read-write storage resource");
        if (mlir::cxx::kernelBufferComponents(buffer) != 1 ||
            !buffer.elementType.isInteger(32))
          return module_.emitError(
              "WGSL atomics require scalar 32-bit integer storage");
        value = "atomic<" + value + ">";
      }
      auto name = WGSLNameEncoder::encodeBinding(index);
      auto isUniform = buffer.kind == mlir::cxx::KernelBufferKind::Uniform;
      if (hasUniformBlockMember(buffer)) {
        auto block = encoder_.encodeRecord();
        typeDeclarations_ += "struct " + block + " {\n  ";
        typeDeclarations_ +=
            WGSLNameEncoder::encodeField(0) + ": " + value + ",\n}\n\n";
        value = block;
      }
      out_ << "@group(" << ::cxx::kBufferDescriptorSet << ") @binding("
           << buffer.binding << ") ";
      if (isUniform) {
        out_ << "var<uniform> " << name << ": " << value << ";\n\n";
      } else {
        out_ << "var<storage, " << (buffer.readOnly ? "read" : "read_write")
             << "> " << name << ": array<" << value << ">;\n\n";
      }
    }
    for (const auto& entry : contract_.entries) {
      auto function = module_.lookupSymbol<gpu::GPUFuncOp>(entry.name);
      if (!function || function.getNumArguments() != entry.resources.size())
        return module_.emitError(
                   "WGSL entry resources differ from the source contract: ")
               << entry.name;
      for (auto [argument, index] :
           llvm::zip(function.getArguments(), entry.resources)) {
        const auto& buffer = contract_.resources[index];
        buffers_[argument] = {
            bindingExpression(index),
            buffer.kind == mlir::cxx::KernelBufferKind::Uniform,
            buffer.unsignedElements, buffer.readOnly};
      }
    }
    return success();
  }

  [[nodiscard]] static auto hasUniformBlockMember(
      const mlir::cxx::KernelResource& buffer) -> bool {
    return buffer.kind == mlir::cxx::KernelBufferKind::Uniform &&
           !mlir::cxx::isBufferRecordType(buffer.elementType);
  }

  [[nodiscard]] auto bindingExpression(unsigned index) const -> std::string {
    auto name = WGSLNameEncoder::encodeBinding(index);
    if (hasUniformBlockMember(contract_.resources[index]))
      name += "." + WGSLNameEncoder::encodeField(0);
    return name;
  }

  [[nodiscard]] auto entryParameters() -> std::string {
    const std::array<std::pair<llvm::StringRef, llvm::StringRef>, 3> builtins =
        {{{"global_invocation_id",
           WGSLNameEncoder::encodeBuiltin(WGSLNameEncoder::Builtin::kGlobalId)},
          {"local_invocation_id",
           WGSLNameEncoder::encodeBuiltin(WGSLNameEncoder::Builtin::kLocalId)},
          {"workgroup_id", WGSLNameEncoder::encodeBuiltin(
                               WGSLNameEncoder::Builtin::kWorkgroupId)}}};
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

  [[nodiscard]] auto writeFunction(func::FuncOp function) -> LogicalResult {
    if (!llvm::hasSingleElement(function.getBody()))
      return function.emitError("WGSL helpers require structured control flow");
    names_.clear();
    objects_.clear();
    std::string parameters;
    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      auto binding = dyn_cast_or_null<spirv::InterfaceVarABIAttr>(
          mlir::cxx::deviceResourceBinding(argument));
      if (binding) {
        auto resource = mlir::cxx::deviceResourceIndex(argument);
        if (!resource || resource.getInt() < 0 ||
            std::uint64_t(resource.getInt()) >= contract_.resources.size())
          return function.emitError("WGSL helper has no resource identity");
        const auto& buffer = contract_.resources[resource.getInt()];
        if (binding.getDescriptorSet() != ::cxx::kBufferDescriptorSet ||
            binding.getBinding() != buffer.binding)
          return function.emitError(
              "WGSL helper resource differs from its source contract");
        buffers_[argument] = {
            bindingExpression(resource.getInt()),
            buffer.kind == mlir::cxx::KernelBufferKind::Uniform,
            buffer.unsignedElements, buffer.readOnly};
        continue;
      }
      auto memory = dyn_cast<MemRefType>(argument.getType());
      auto type = memory ? memory.getElementType() : argument.getType();
      auto spelling =
          storageTypeName(function, type, memory && isAtomicStorage(argument));
      if (failed(spelling)) return failure();
      if (memory) {
        workgroupPointers_ |= isWorkgroupStorage(memory);
        if (!memory.hasStaticShape())
          return function.emitError(
              "WGSL resource arguments require binding specialization");
        for (auto extent : llvm::reverse(memory.getShape()))
          *spelling =
              "array<" + *spelling + ", " + std::to_string(extent) + ">";
        auto space = isWorkgroupStorage(memory) ? "workgroup" : "function";
        *spelling = "ptr<" + std::string(space) + ", " + *spelling + ">";
        objects_.insert(argument);
      }
      auto name = WGSLNameEncoder::encodeArgument(index);
      names_[argument] = memory ? "(*" + name + ")" : name;
      if (!parameters.empty()) parameters += ", ";
      parameters += name + ": " + *spelling;
    }
    std::string result;
    if (function.getNumResults()) {
      if (function.getNumResults() != 1)
        return function.emitError("WGSL helpers return one value");
      auto type = typeName(function, function.getResultTypes().front());
      if (failed(type)) return failure();
      result = " -> " + *type;
    }
    out_ << "fn " << function.getSymName() << "(" << parameters << ")" << result
         << " {\n";
    auto terminator = writeBody(function.getBody().front(), 1);
    if (failed(terminator)) return failure();
    auto returned = dyn_cast<func::ReturnOp>(*terminator);
    if (!returned)
      return (*terminator)->emitError("expected a function return");
    if (returned.getNumOperands())
      out_ << "  return " << use(returned.getOperand(0)) << ";\n";
    out_ << "}\n\n";
    return success();
  }

  [[nodiscard]] auto writeCall(func::CallOp call, unsigned depth)
      -> LogicalResult {
    auto target = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!target || target.getNumArguments() != call.getNumOperands())
      return call.emitError("WGSL call has no matching helper definition");
    std::string arguments;
    for (auto [parameter, value] :
         llvm::zip(target.getArguments(), call.getOperands())) {
      if (mlir::cxx::deviceResourceBinding(parameter)) continue;
      if (!arguments.empty()) arguments += ", ";
      arguments +=
          isa<MemRefType>(value.getType()) ? "&" + use(value) : use(value);
    }
    auto expression = call.getCallee().str() + "(" + arguments + ")";
    if (call.getNumResults()) {
      if (call.getNumResults() != 1)
        return call.emitError("WGSL helpers return one value");
      return define(call, call.getResult(0), expression, depth);
    }
    indent(depth);
    out_ << expression << ";\n";
    return success();
  }

  [[nodiscard]] auto aggregateElement(Value aggregate, std::int64_t index)
      -> std::string {
    auto base = use(aggregate);
    if (isa<mlir::cxx::ArrayType>(aggregate.getType()))
      return base + "[" + std::to_string(index) + "i]";
    return base + "." + WGSLNameEncoder::encodeField(index);
  }

  [[nodiscard]] auto storageElement(Value pointer, Operation* op)
      -> FailureOr<StorageView> {
    auto address = mlir::cxx::resolveStorageAddress(pointer);
    if (failed(address))
      return op->emitError("WGSL reference has no storage root");
    StorageView result{"", mlir::cxx::getStorageElementType(*address)};
    result.atomic = isAtomicStorage(address->root);
    auto buffer = buffers_.find(address->root);
    if (buffer != buffers_.end()) {
      result.expression = buffer->second.name;
      result.isUnsigned = buffer->second.isUnsigned;
      result.readOnly = buffer->second.readOnly;
    } else {
      result.expression = names_.lookup(address->root);
    }
    auto& name = result.expression;
    if (name.empty())
      return op->emitError("WGSL storage root has no declaration");
    for (const auto& index : address->path) {
      if (auto member = std::get_if<mlir::cxx::StorageMember>(&index)) {
        name += "." + WGSLNameEncoder::encodeField(member->position);
      } else {
        std::string total;
        for (auto term : std::get<mlir::cxx::StorageElement>(index).terms) {
          if (!total.empty()) total += " + ";
          total += use(term);
        }
        name += "[" + (total.empty() ? "0i" : total) + "]";
      }
    }
    return result;
  }

  [[nodiscard]] auto scalarTypeName(Type type) -> std::optional<std::string> {
    if (type.isF32()) return "f32";
    if (type.isSignlessInteger(1)) return "bool";
    if (type.isIndex()) return "i32";
    if (auto integer = dyn_cast<IntegerType>(type))
      if (integer.isSignless() && integer.getWidth() <= 32) return "i32";
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
    if (auto array = dyn_cast<mlir::cxx::ArrayType>(type)) {
      auto element = typeName(op, array.getElementType());
      if (failed(element) || !array.getSize()) return failure();
      return "array<" + *element + ", " + std::to_string(array.getSize()) + ">";
    }
    if (mlir::cxx::isBufferRecordType(type)) return bufferRecordName(op, type);
    if (isa<mlir::cxx::ClassType, TupleType>(type)) {
      auto found = records_.find(type);
      if (found != records_.end()) return found->second;
      if (!activeRecords_.insert(type).second)
        return op->emitError("WGSL record contains itself by value");
      auto name = encoder_.encodeRecord();
      std::string declaration = "struct " + name + " {\n";
      auto body = mlir::cxx::getRecordElementTypes(type);
      for (auto position : mlir::cxx::valueElementPositions(type)) {
        auto field = typeName(op, body[position]);
        if (failed(field)) return failure();
        declaration += "  " + WGSLNameEncoder::encodeField(position) + ": " +
                       *field + ",\n";
      }
      typeDeclarations_ += declaration + "}\n\n";
      activeRecords_.erase(type);
      records_[type] = name;
      return name;
    }
    return op->emitError("WGSL emission does not support type ") << type;
  }

  [[nodiscard]] static auto naturalSize(Type type) -> std::uint32_t {
    auto vector = dyn_cast<VectorType>(type);
    if (vector && vector.getNumElements() == 3) return 12;
    return mlir::cxx::bufferLayoutSize(type);
  }

  [[nodiscard]] auto bufferRecordName(Operation* op, Type type)
      -> FailureOr<std::string> {
    auto found = records_.find(type);
    if (found != records_.end()) return found->second;
    std::string declaration;
    for (const auto& member : mlir::cxx::bufferLayoutMembers(type)) {
      auto field = typeName(op, member.type);
      if (failed(field)) return failure();
      declaration += "  ";
      if (member.span != naturalSize(member.type))
        declaration += "@size(" + std::to_string(member.span) + ") ";
      declaration +=
          WGSLNameEncoder::encodeField(member.position) + ": " + *field + ",\n";
    }
    auto name = encoder_.encodeRecord();
    typeDeclarations_ += "struct " + name + " {\n" + declaration + "}\n\n";
    records_[type] = name;
    return name;
  }

  [[nodiscard]] auto bindingValueTypeName(
      const mlir::cxx::KernelResource& buffer) -> FailureOr<std::string> {
    if (mlir::cxx::isBufferValueType(buffer.elementType))
      return bufferValueTypeName(buffer);
    return typeName(module_, buffer.elementType);
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
    auto type = value.getType();
    if (isBoolean(value)) {
      auto lanes = withLanesOf(type, "u32");
      return "select(" + lanes + "(0u), " + lanes + "(1u), " + use(value) + ")";
    }
    auto result =
        "bitcast<" + withLanesOf(type, "u32") + ">(" + use(value) + ")";
    if (auto width = narrowIntegerWidth(type)) {
      auto mask = std::to_string((std::uint32_t(1) << width) - 1) + "u";
      if (isa<VectorType>(type))
        mask = withLanesOf(type, "u32") + "(" + mask + ")";
      result = "(" + result + " & " + mask + ")";
    }
    return result;
  }

  [[nodiscard]] static auto narrowIntegerWidth(Type type) -> unsigned {
    if (auto vector = dyn_cast<VectorType>(type))
      type = vector.getElementType();
    auto integer = dyn_cast<IntegerType>(type);
    if (!integer || integer.getWidth() <= 1 || integer.getWidth() >= 32)
      return 0;
    return integer.getWidth();
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
    auto name = encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kValue);
    auto value = expression;
    if (auto width = narrowIntegerWidth(result.getType())) {
      auto shift = std::to_string(32 - width) + "u";
      if (isa<VectorType>(result.getType()))
        shift = withLanesOf(result.getType(), "u32") + "(" + shift + ")";
      value = "((" + value + " << " + shift + ") >> " + shift + ")";
    }
    indent(depth);
    out_ << "let " << name << ": " << *type << " = " << value << ";\n";
    names_[result] = name;
    return success();
  }

  [[nodiscard]] auto declareResults(Operation* op, unsigned depth)
      -> LogicalResult {
    for (auto result : op->getResults()) {
      auto type = typeName(op, result.getType());
      if (failed(type)) return failure();
      auto name = encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kResult);
      indent(depth);
      out_ << "var " << name << ": " << *type << ";\n";
      names_[result] = name;
    }
    return success();
  }

  void assign(ValueRange targets, ValueRange values, unsigned depth) {
    llvm::SmallVector<std::string> snapshots;
    for (auto value : values) {
      auto name =
          encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kSnapshot);
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
    if (isa<VectorType, mlir::cxx::ArrayType, mlir::cxx::ClassType, TupleType>(
            type))
      return *name + "()";
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
    auto truncated =
        encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kValue);
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
          return writeInvocationId(op,
                                   WGSLNameEncoder::encodeBuiltin(
                                       WGSLNameEncoder::Builtin::kGlobalId),
                                   id.getDimension(), depth);
        })
        .Case([&](gpu::ThreadIdOp id) {
          return writeInvocationId(op,
                                   WGSLNameEncoder::encodeBuiltin(
                                       WGSLNameEncoder::Builtin::kLocalId),
                                   id.getDimension(), depth);
        })
        .Case([&](gpu::BlockIdOp id) {
          return writeInvocationId(op,
                                   WGSLNameEncoder::encodeBuiltin(
                                       WGSLNameEncoder::Builtin::kWorkgroupId),
                                   id.getDimension(), depth);
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
        .Case([&](func::CallOp call) { return writeCall(call, depth); })
        .Case([&](mlir::cxx::UndefOp undef) -> LogicalResult {
          auto value = zeroValue(op, undef.getType());
          if (failed(value)) return failure();
          names_[undef.getResult()] = *value;
          return success();
        })
        .Case([&](mlir::cxx::ExtractValueOp extract) {
          return define(
              op, extract.getResult(),
              aggregateElement(extract.getContainer(), extract.getPosition()),
              depth);
        })
        .Case([&](mlir::cxx::InsertValueOp insert) -> LogicalResult {
          auto name = encoder_.encodeTemporary(
              WGSLNameEncoder::Temporary::kInsertedValue);
          indent(depth);
          out_ << "var " << name << " = " << use(insert.getContainer())
               << ";\n";
          names_[insert.getResult()] = name;
          indent(depth);
          out_ << aggregateElement(insert.getResult(), insert.getPosition())
               << " = " << use(insert.getValue()) << ";\n";
          return success();
        })
        .Case([&](mlir::cxx::LoadOp load) -> LogicalResult {
          auto element = storageElement(load.getAddr(), op);
          if (failed(element)) return failure();
          auto value = element->expression;
          if (element->atomic) value = "atomicLoad(&" + value + ")";
          if (element->isUnsigned || element->type != load.getType()) {
            auto type = typeName(op, load.getType());
            if (failed(type)) return failure();
            value = "bitcast<" + *type + ">(" + value + ")";
          }
          return define(op, load.getResult(), value, depth);
        })
        .Case([&](mlir::cxx::StoreOp store) -> LogicalResult {
          auto element = storageElement(store.getAddr(), op);
          if (failed(element)) return failure();
          if (element->readOnly)
            return store.emitError("WGSL cannot write a read-only resource");
          auto value = use(store.getValue());
          if (element->isUnsigned ||
              element->type != store.getValue().getType()) {
            auto type = typeName(op, element->type);
            if (failed(type)) return failure();
            if (element->isUnsigned) *type = withLanesOf(element->type, "u32");
            value = "bitcast<" + *type + ">(" + value + ")";
          }
          indent(depth);
          if (element->atomic)
            out_ << "atomicStore(&" << element->expression << ", " << value
                 << ");\n";
          else
            out_ << element->expression << " = " << value << ";\n";
          return success();
        })
        .Case([&](mlir::cxx::BitcastOp cast) -> LogicalResult {
          if (isa<mlir::cxx::PointerType>(cast.getType()))
            return success(
                succeeded(mlir::cxx::resolveStorageAddress(cast.getResult())));
          auto type = typeName(op, cast.getType());
          if (failed(type)) return failure();
          return define(op, cast.getResult(),
                        "bitcast<" + *type + ">(" + use(cast.getValue()) + ")",
                        depth);
        })
        .Case<ptr::ToPtrOp>([](auto) { return success(); })
        .Case<mlir::cxx::StorageRefOp>([](auto) { return success(); })
        .Case<ptr::FromPtrOp, mlir::cxx::MemberOp, mlir::cxx::SubscriptOp,
              mlir::cxx::ArrayToPointerOp, mlir::cxx::PtrAddOp>([](auto op) {
          if (succeeded(mlir::cxx::resolveStorageAddress(op->getResult(0))))
            return success();
          return LogicalResult(
              op->emitError("WGSL reference has no storage root"));
        })
        .Case([&](memref::AllocOp allocation) -> LogicalResult {
          auto found = sharedNames_.find(allocation.getResult());
          if (found == sharedNames_.end())
            return allocation.emitError(
                "WGSL shared allocation has no declaration");
          names_[allocation] = found->second;
          objects_.insert(allocation);
          return success();
        })
        .Case([&](mlir::cxx::AtomicLoadOp atomic) {
          return writeAtomicLoad(atomic, depth);
        })
        .Case([&](mlir::cxx::AtomicStoreOp atomic) {
          return writeAtomicStore(atomic, depth);
        })
        .Case([&](mlir::cxx::AtomicRMWOp atomic) {
          return writeAtomicReadModifyWrite(atomic, depth);
        })
        .Case([&](mlir::cxx::AtomicCompareExchangeOp atomic) {
          return writeAtomicCompareExchange(atomic, depth);
        })
        .Case([](mlir::cxx::AtomicFenceOp) { return success(); })
        .Case([&](mlir::cxx::ControlBarrierOp barrier) {
          return writeControlBarrier(barrier, depth);
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
          if (!isBoolean(extend.getIn()))
            return define(
                op, extend.getResult(),
                signedResult(extend.getResult(), unsignedUse(extend.getIn())),
                depth);
          return writeBooleanExtension(extend, extend.getIn(), "1i", depth);
        })
        .Case([&](arith::ExtSIOp extend) {
          if (!isBoolean(extend.getIn()))
            return define(op, extend.getResult(), use(extend.getIn()), depth);
          return writeBooleanExtension(extend, extend.getIn(), "(-1i)", depth);
        })
        .Case([&](arith::TruncIOp truncate) -> LogicalResult {
          if (!isBoolean(truncate.getResult()))
            return define(op, truncate.getResult(), use(truncate.getIn()),
                          depth);
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
    auto name = encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kValue);
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
    auto name =
        encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kPrivateVariable);
    indent(depth);
    out_ << "var " << name << ": " << spelled << ";\n";
    names_[allocation.getResult()] = name;
    objects_.insert(allocation.getResult());
    return success();
  }

  [[nodiscard]] auto writeFor(scf::ForOp loop, unsigned depth)
      -> LogicalResult {
    auto& body = *loop.getBody();
    auto induction =
        encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kInduction);
    const bool isUnsigned = loop.getUnsignedCmp();
    names_[loop.getInductionVar()] =
        isUnsigned ? signedResult(loop.getInductionVar(), induction)
                   : induction;
    for (auto [argument, initial] :
         llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs())) {
      auto type = typeName(loop, argument.getType());
      if (failed(type)) return failure();
      auto name =
          encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kLoopValue);
      indent(depth);
      out_ << "var " << name << ": " << *type << " = " << use(initial) << ";\n";
      names_[argument] = name;
    }
    indent(depth);
    auto bound = [&](Value value) {
      return isUnsigned ? unsignedUse(value) : use(value);
    };
    out_ << "for (var " << induction << ": " << (isUnsigned ? "u32" : "i32")
         << " = " << bound(loop.getLowerBound()) << "; " << induction << " < "
         << bound(loop.getUpperBound()) << "; " << induction
         << " += " << bound(loop.getStep()) << ") {\n";
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

  [[nodiscard]] auto atomicResult(Operation* op, Value result,
                                  const StorageView& element,
                                  std::string expression, unsigned depth)
      -> LogicalResult {
    if (element.isUnsigned) expression = "bitcast<i32>(" + expression + ")";
    return define(op, result, expression, depth);
  }

  [[nodiscard]] auto atomicElement(Operation* op, Value address)
      -> FailureOr<StorageView> {
    auto element = storageElement(address, op);
    if (failed(element)) return failure();
    if (element->readOnly || !element->atomic)
      return op->emitError("WGSL atomic requires read-write atomic storage");
    return element;
  }

  [[nodiscard]] auto atomicOperand(const StorageView& element, Value value)
      -> std::string {
    if (element.isUnsigned) return unsignedUse(value);
    return use(value);
  }

  [[nodiscard]] static auto atomicFunction(mlir::cxx::AtomicRMWKind kind)
      -> std::optional<StringRef> {
    switch (kind) {
      case mlir::cxx::AtomicRMWKind::Exchange:
        return StringRef("atomicExchange");
      case mlir::cxx::AtomicRMWKind::Add:
        return StringRef("atomicAdd");
      case mlir::cxx::AtomicRMWKind::Sub:
        return StringRef("atomicSub");
      case mlir::cxx::AtomicRMWKind::And:
        return StringRef("atomicAnd");
      case mlir::cxx::AtomicRMWKind::Or:
        return StringRef("atomicOr");
      case mlir::cxx::AtomicRMWKind::Xor:
        return StringRef("atomicXor");
      case mlir::cxx::AtomicRMWKind::Nand:
        return std::nullopt;
    }
    return std::nullopt;
  }

  [[nodiscard]] auto writeAtomicLoad(mlir::cxx::AtomicLoadOp op, unsigned depth)
      -> LogicalResult {
    auto element = atomicElement(op, op.getAddr());
    if (failed(element)) return failure();
    return atomicResult(op, op.getResult(), *element,
                        "atomicLoad(&" + element->expression + ")", depth);
  }

  [[nodiscard]] auto writeAtomicStore(mlir::cxx::AtomicStoreOp op,
                                      unsigned depth) -> LogicalResult {
    auto element = atomicElement(op, op.getAddr());
    if (failed(element)) return failure();
    indent(depth);
    out_ << "atomicStore(&" << element->expression << ", "
         << atomicOperand(*element, op.getValue()) << ");\n";
    return success();
  }

  [[nodiscard]] auto writeAtomicReadModifyWrite(mlir::cxx::AtomicRMWOp op,
                                                unsigned depth)
      -> LogicalResult {
    auto function = atomicFunction(op.getKind());
    if (!function)
      return op.emitError("WGSL has no atomic for this modification");
    auto element = atomicElement(op, op.getAddr());
    if (failed(element)) return failure();
    return atomicResult(op, op.getResult(), *element,
                        function->str() + "(&" + element->expression + ", " +
                            atomicOperand(*element, op.getValue()) + ")",
                        depth);
  }

  [[nodiscard]] auto writeControlBarrier(mlir::cxx::ControlBarrierOp barrier,
                                         unsigned depth) -> LogicalResult {
    auto regions = barrier.getMemoryRegions();
    if (mlir::cxx::bitEnumContainsAll(regions,
                                      mlir::cxx::MemoryRegions::Workgroup)) {
      indent(depth);
      out_ << "workgroupBarrier();\n";
    }
    if (mlir::cxx::bitEnumContainsAll(regions,
                                      mlir::cxx::MemoryRegions::Storage)) {
      indent(depth);
      out_ << "storageBarrier();\n";
    }
    return success();
  }

  [[nodiscard]] auto writeAtomicCompareExchange(
      mlir::cxx::AtomicCompareExchangeOp op, unsigned depth) -> LogicalResult {
    auto element = atomicElement(op, op.getAddr());
    if (failed(element)) return failure();
    auto expected = atomicOperand(*element, op.getExpected());
    auto desired = atomicOperand(*element, op.getDesired());
    auto call = "atomicCompareExchangeWeak(&" + element->expression + ", " +
                expected + ", " + desired + ")";
    auto old = encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kOldValue);
    auto exchanged =
        encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kExchanged);
    auto attempt =
        encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kAttempt);
    auto scalar = element->isUnsigned ? "u32" : "i32";
    indent(depth);
    out_ << "var " << old << ": " << scalar << ";\n";
    indent(depth);
    out_ << "var " << exchanged << ": bool;\n";
    unsigned body = depth;
    if (!op.getWeak()) {
      indent(depth);
      out_ << "loop {\n";
      ++body;
    }
    indent(body);
    out_ << "let " << attempt << " = " << call << ";\n";
    indent(body);
    out_ << old << " = " << attempt << ".old_value;\n";
    indent(body);
    out_ << exchanged << " = " << attempt << ".exchanged;\n";
    if (!op.getWeak()) {
      indent(body);
      out_ << "if (" << exchanged << " || " << old << " != " << expected
           << ") { break; }\n";
      indent(depth);
      out_ << "}\n";
    }
    names_[op.getOld()] =
        element->isUnsigned ? "bitcast<i32>(" + old + ")" : old;
    names_[op.getExchanged()] = exchanged;
    return success();
  }

  [[nodiscard]] auto writeLoad(memref::LoadOp load, unsigned depth)
      -> LogicalResult {
    if (objects_.contains(load.getMemRef())) {
      auto element = privateElement(load);
      if (isAtomicStorage(load.getMemRef()))
        element = "atomicLoad(&" + element + ")";
      return define(load, load.getResult(), element, depth);
    }
    auto buffer = bufferOf(load, load.getMemRef());
    if (failed(buffer)) return failure();
    auto element = buffer->name;
    if (!buffer->isUniform)
      element += "[" + use(load.getIndices().front()) + "]";
    if (isAtomicStorage(load.getMemRef()))
      element = "atomicLoad(&" + element + ")";
    if (buffer->isUnsigned) {
      auto type = typeName(load, load.getType());
      if (failed(type)) return failure();
      element = "bitcast<" + *type + ">(" + element + ")";
    }
    return define(load, load.getResult(), element, depth);
  }

  [[nodiscard]] auto writeStore(memref::StoreOp store, unsigned depth)
      -> LogicalResult {
    auto value = use(store.getValueToStore());
    std::string element;
    if (objects_.contains(store.getMemRef())) {
      element = privateElement(store.getMemRef(), store.getIndices());
    } else {
      auto buffer = bufferOf(store, store.getMemRef());
      if (failed(buffer)) return failure();
      if (buffer->readOnly)
        return store.emitError("WGSL cannot write a read-only resource");
      if (buffer->isUnsigned) value = unsignedUse(store.getValueToStore());
      element = buffer->name + "[" + use(store.getIndices().front()) + "]";
    }
    indent(depth);
    if (isAtomicStorage(store.getMemRef()))
      out_ << "atomicStore(&" << element << ", " << value << ");\n";
    else
      out_ << element << " = " << value << ";\n";
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
      auto name =
          encoder_.encodeTemporary(WGSLNameEncoder::Temporary::kLoopValue);
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

  gpu::GPUModuleOp module_;
  const mlir::cxx::KernelModuleContract& contract_;
  gpu::GPUFuncOp function_;
  std::string text_;
  std::string typeDeclarations_;
  DenseMap<Type, std::string> records_;
  DenseSet<Type> activeRecords_;
  llvm::raw_string_ostream out_;
  DenseMap<Value, std::string> names_;
  DenseMap<Value, BufferView> buffers_;
  DenseSet<Value> objects_;
  bool workgroupPointers_ = false;
  SmallVector<llvm::StringRef> languageFeatures_;
  DenseSet<Value> atomicRoots_;
  DenseSet<unsigned> atomicResources_;
  DenseMap<Value, std::string> sharedNames_;
  WGSLNameEncoder encoder_;
};

}  // namespace

auto compileWGSLModule(mlir::ModuleOp source, llvm::StringRef kernel)
    -> mlir::FailureOr<KernelArtifact> {
  using namespace mlir;
  auto contract = mlir::cxx::kernelModuleContractOf(source, kernel);
  if (failed(contract)) return failure();
  for (const auto& entry : contract->entries)
    if (!WGSLNameEncoder::isUsableEntryName(entry.name)) {
      source.lookupSymbol<mlir::cxx::FuncOp>(entry.name)
              .emitError("kernel name '")
          << entry.name << "' is reserved in WGSL output";
      return failure();
    }
  auto working = lowerDeviceModuleToGPU(source, kernel);
  if (failed(working)) return failure();
  auto modules = working->get().getOps<gpu::GPUModuleOp>();
  if (!llvm::hasSingleElement(modules))
    return source.emitError("expected one lowered GPU module");
  auto module = *modules.begin();
  if (failed(mlir::cxx::verifyWebGPUOperations(module)) ||
      failed(legalizeWGSLFunctionNames(module)))
    return failure();
  WGSLWriter writer(module, *contract);
  auto wgsl = writer.write();
  if (failed(wgsl)) return failure();
  KernelArtifact artifact;
  artifact.bytes.assign(wgsl->begin(), wgsl->end());
  llvm::json::Object target{{"targetEnvironment", "webgpu"}};
  if (!writer.languageFeatures().empty())
    target["requiredFeatures"] = llvm::json::Array(writer.languageFeatures());
  artifact.manifest =
      kernelManifest(*contract, "wgsl", artifact.bytes, std::move(target));
  return artifact;
}

}  // namespace cxx
