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

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/device_profile_limits.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <array>
#include <format>

namespace cxx {
namespace {

constexpr std::string_view kKernelTemplateMessage =
    "a kernel cannot be a function template or a specialization of one";

[[nodiscard]] auto isParameterAttribute(AttributeKind kind) -> bool {
  switch (kind) {
    case AttributeKind::kBuffer:
    case AttributeKind::kGlobalId:
    case AttributeKind::kLocalId:
    case AttributeKind::kWorkgroupId:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] auto invocationIdKind(AttributeKind kind)
    -> KernelInvocationIdKind {
  switch (kind) {
    case AttributeKind::kLocalId:
      return KernelInvocationIdKind::kLocal;
    case AttributeKind::kWorkgroupId:
      return KernelInvocationIdKind::kWorkgroup;
    default:
      return KernelInvocationIdKind::kGlobal;
  }
}

[[nodiscard]] auto namesParameterAttribute(const Attribute& attribute) -> bool {
  return std::ranges::any_of(attributeInfos(), [&](const AttributeInfo& info) {
    return info.availability == AttributeAvailability::kDeviceProfile &&
           isParameterAttribute(info.kind) &&
           namesAttribute(attribute, info.kind);
  });
}

[[nodiscard]] auto declaresParameterAttribute(const AttributeMap* attributes)
    -> bool {
  if (!attributes) return false;
  return std::ranges::any_of(*attributes, namesParameterAttribute);
}

[[nodiscard]] auto conflicts(const KernelParameterBinding& lhs,
                             const KernelParameterBinding& rhs) -> bool {
  return lhs == rhs;
}

[[nodiscard]] auto conflictMessage(const KernelParameterBinding& binding)
    -> std::string {
  if (std::holds_alternative<KernelBufferBinding>(binding))
    return "kernel parameters cannot share a buffer index";
  return "kernel parameters cannot share an invocation id";
}

class DeviceAttributeBinder {
 public:
  explicit DeviceAttributeBinder(TranslationUnit* unit)
      : unit_(unit), control_(unit->control()), traits_(unit->typeTraits()) {}

  void bindParameter(ParameterSymbol* parameter,
                     std::span<const AttributeRef> attributes) {
    std::optional<AttributeRef> binding;
    for (auto attribute : attributes) {
      if (!isParameterAttribute(attribute.kind())) {
        unit_->error(attribute.location,
                     "kernel attributes require a function");
        continue;
      }
      if (binding) {
        unit_->error(attribute.location,
                     "a kernel parameter takes one 'buffer' or invocation id "
                     "attribute");
        continue;
      }
      binding = attribute;
    }
    if (!binding) return;
    if (binding->kind() == AttributeKind::kBuffer) {
      if (auto buffer = bufferBinding(parameter, *binding))
        parameter->setKernelParameterBinding(*buffer);
      return;
    }
    if (auto id = invocationId(parameter, *binding))
      parameter->setKernelParameterBinding(*id);
  }

  void bindFunction(FunctionSymbol* function,
                    std::span<const AttributeRef> attributes) {
    bool isKernel = false;
    std::optional<KernelWorkgroupSize> workgroup;
    for (auto attribute : attributes) {
      switch (attribute.kind()) {
        case AttributeKind::kKernel:
          if (isKernel) {
            unit_->error(attribute.location, "duplicate 'kernel' attribute");
            continue;
          }
          isKernel = true;
          (void)arguments(attribute, 0);
          continue;
        case AttributeKind::kWorkgroupSize:
          if (workgroup) {
            unit_->error(attribute.location,
                         "duplicate 'workgroup_size' attribute");
            continue;
          }
          workgroup = workgroupSize(attribute);
          continue;
        default:
          unit_->error(attribute.location,
                       std::format("'{}' attribute requires a function "
                                   "parameter",
                                   attribute.spelling->name));
      }
    }
    if (!declareKernel(function, isKernel, workgroup)) return;
    if (auto previous = conflictingEntryName(function)) {
      unit_->error(function->location(),
                   std::format("kernel entry name '{}' conflicts with another "
                               "function",
                               to_diag_string(function->name())));
      unit_->note(previous->location(), "previous declaration is here");
    }
    bindKernelParameters(function);
  }

  void rejectAttributes(std::span<const AttributeRef> attributes) {
    for (auto attribute : attributes) {
      if (isParameterAttribute(attribute.kind()))
        unit_->error(attribute.location,
                     std::format("'{}' attribute requires a function parameter",
                                 attribute.spelling->name));
      else
        unit_->error(attribute.location,
                     "kernel attributes require a function");
    }
  }

 private:
  [[nodiscard]] auto conflictingEntryName(FunctionSymbol* function)
      -> FunctionSymbol* {
    auto canonical = function->canonical();
    const auto kernel = canonical->kernelWorkgroupSize().has_value();
    if (!kernel && !canonical->hasCLinkage()) return nullptr;

    std::vector<ScopeSymbol*> pending{unit_->globalScope()};
    while (!pending.empty()) {
      auto scope = pending.back();
      pending.pop_back();
      for (auto symbol : scope->find(function->name())) {
        for (auto other : views::each_function(symbol)) {
          other = other->canonical();
          if (other == canonical) continue;
          const auto otherKernel = other->kernelWorkgroupSize().has_value();
          if (!kernel && !otherKernel) continue;
          if (!otherKernel && !other->hasCLinkage()) continue;
          if (canonical->hasCLinkage() && other->hasCLinkage() &&
              canonical->type() == other->type())
            continue;
          return other;
        }
      }
      for (auto ns : scope->members() | views::namespaces)
        pending.push_back(ns);
    }
    return nullptr;
  }

  [[nodiscard]] auto arguments(AttributeRef attribute, std::size_t count)
      -> std::optional<std::vector<std::uint32_t>> {
    std::vector<std::uint32_t> values;
    auto clause = attribute.argumentClause;
    ASTInterpreter interpreter{unit_};
    for (auto expression :
         ListView{clause ? clause->expressionList : nullptr}) {
      auto value = interpreter.evaluate(expression);
      auto integer = value ? interpreter.toUInt(*value) : std::nullopt;
      if (!traits_.is_integral(expression->type) || !integer ||
          *integer > 0x7fffffff) {
        unit_->error(attribute.location,
                     "device attribute arguments must be nonnegative integer "
                     "constants");
        return std::nullopt;
      }
      values.push_back(static_cast<std::uint32_t>(*integer));
    }
    if (values.size() != count) {
      unit_->error(
          attribute.location,
          std::format("'{}' attribute requires {} argument{}",
                      attribute.spelling->name, count, count == 1 ? "" : "s"));
      return std::nullopt;
    }
    return values;
  }

  [[nodiscard]] auto bufferBinding(ParameterSymbol* parameter,
                                   AttributeRef attribute)
      -> std::optional<KernelBufferBinding> {
    auto args = arguments(attribute, 1);
    if (!args) return std::nullopt;
    auto binding = args->front();
    if (binding > kMaxBufferBinding) {
      unit_->error(
          attribute.location,
          std::format("'buffer' index must be 0 to {}", kMaxBufferBinding));
      return std::nullopt;
    }
    if (!isStorageBufferType(parameter->type()) &&
        !isUniformBufferType(parameter->type())) {
      unit_->error(attribute.location,
                   "'buffer' requires an address-space-1 or address-space-2 "
                   "pointer, or an address-space-2 reference, to a scalar, a "
                   "vector, or an array or class of them");
      return std::nullopt;
    }
    return KernelBufferBinding{binding};
  }

  [[nodiscard]] auto isStorageBufferType(const Type* type) -> bool {
    auto pointer = type_cast<PointerType>(unqualified_type(type));
    if (!pointer) return false;
    auto element = pointer->elementType();
    auto space = address_space(element);
    if (space != kDeviceAddressSpace && space != kConstantAddressSpace)
      return false;
    return !traits_.is_volatile(element) && isBufferObjectType(element);
  }

  [[nodiscard]] auto isUniformBufferType(const Type* type) -> bool {
    if (!traits_.is_lvalue_reference(type)) return false;
    auto object = traits_.remove_reference(type);
    if (address_space(object) != kConstantAddressSpace) return false;
    if (traits_.is_volatile(object)) return false;
    return isBufferObjectType(object);
  }

  [[nodiscard]] auto isBufferObjectType(const Type* type) -> bool {
    if (traits_.is_device_value_type(type)) return true;
    if (traits_.is_array(type))
      return traits_.is_bounded_array(type) &&
             isBufferObjectType(traits_.remove_extent(type));
    auto classType = type_cast<ClassType>(unqualified_type(type));
    if (!classType || !classType->symbol()) return false;
    return isBufferClass(classType->symbol()->resolvedDefinition());
  }

  [[nodiscard]] auto isBufferClass(ClassSymbol* symbol) -> bool {
    if (!symbol->isComplete() || symbol->isUnion()) return false;
    if (symbol->isPolymorphic() || symbol->hasVirtualBaseClasses())
      return false;
    auto hasData = false;
    for (auto base : symbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) return false;
      baseClass = baseClass->resolvedDefinition();
      if (baseClass->layout() && baseClass->layout()->isAbiEmpty()) continue;
      if (!isBufferClass(baseClass)) return false;
      hasData = true;
    }
    for (auto field : symbol->members() | views::non_static_fields) {
      if (field->bitFieldWidth() || !isBufferObjectType(field->type()))
        return false;
      hasData = true;
    }
    return hasData;
  }

  [[nodiscard]] auto invocationId(ParameterSymbol* parameter,
                                  AttributeRef attribute)
      -> std::optional<KernelInvocationId> {
    if (!arguments(attribute, 0)) return std::nullopt;
    if (!isInvocationIdType(parameter->type())) {
      unit_->error(attribute.location,
                   std::format("'{}' requires an unsigned int parameter or a "
                               "vector of 2 or 3 unsigned ints",
                               attribute.spelling->name));
      return std::nullopt;
    }
    return KernelInvocationId{invocationIdKind(attribute.kind())};
  }

  [[nodiscard]] auto isInvocationIdType(const Type* type) -> bool {
    auto unsignedInt = control_->getUnsignedIntType();
    auto unqualified = unqualified_type(type);
    if (unqualified == unsignedInt) return true;
    auto vector = type_cast<VectorType>(unqualified);
    if (!vector || unqualified_type(vector->elementType()) != unsignedInt)
      return false;
    return vector->elementCount() == 2 || vector->elementCount() == 3;
  }

  [[nodiscard]] auto workgroupSize(AttributeRef attribute)
      -> std::optional<KernelWorkgroupSize> {
    auto args = arguments(attribute, 3);
    if (!args) return std::nullopt;
    auto [x, y, z] = std::tie((*args)[0], (*args)[1], (*args)[2]);
    auto limits = deviceProfileLimits(
        to_string(control_->memoryLayout()->deviceProfile()));
    if (!limits || !limits->admitsWorkgroup(x, y, z)) {
      unit_->error(attribute.location,
                   "'workgroup_size' exceeds the compute limits of the device "
                   "profile");
      return std::nullopt;
    }
    return KernelWorkgroupSize{x, y, z};
  }

  [[nodiscard]] auto isKernelSignature(FunctionSymbol* function) -> bool {
    auto type = type_cast<FunctionType>(function->type());
    return !function->enclosingClass() && !function->isStatic() &&
           !type->isVariadic() && traits_.is_void(type->returnType());
  }

  [[nodiscard]] auto declareKernel(
      FunctionSymbol* function, bool isKernel,
      const std::optional<KernelWorkgroupSize>& workgroup) -> bool {
    auto canonical = function->canonical();
    auto previous = canonical->kernelWorkgroupSize();
    if (workgroup && !isKernel && !previous) {
      unit_->error(function->location(),
                   "'workgroup_size' requires a kernel declaration");
      return false;
    }
    if (isKernel || workgroup) {
      if (is_templated_function(function)) {
        unit_->error(function->location(), std::string{kKernelTemplateMessage});
        return false;
      }
      if (!isKernelSignature(function)) {
        unit_->error(
            function->location(),
            "a kernel must be a non-static, non-variadic free function "
            "returning void");
        return false;
      }
      auto requested =
          workgroup.value_or(previous.value_or(KernelWorkgroupSize{}));
      if (previous && *previous != requested) {
        unit_->error(
            function->location(),
            "kernel workgroup size differs from a previous declaration");
        return false;
      }
      canonical->setKernelWorkgroupSize(requested);
    }
    function->setKernelWorkgroupSize(canonical->kernelWorkgroupSize());
    return true;
  }

  void bindKernelParameters(FunctionSymbol* function) {
    auto canonical = function->canonical();
    auto parameters = function->parameters();
    auto previous = canonical->parameters();
    auto isKernel = function->kernelWorkgroupSize().has_value();
    for (std::size_t index = 0; index < parameters.size(); ++index) {
      auto parameter = parameters[index];
      auto binding = parameter->kernelParameterBinding();
      if (canonical != function && index < previous.size()) {
        auto prior = previous[index]->kernelParameterBinding();
        if (binding && prior && binding != prior)
          unit_->error(parameter->location(),
                       "kernel parameter binding differs from a previous "
                       "declaration");
        if (!binding) binding = prior;
        previous[index]->setKernelParameterBinding(binding);
        parameter->setKernelParameterBinding(binding);
      }
      auto declared =
          binding || declaresParameterAttribute(parameter->attributes());
      if (isKernel && !declared)
        unit_->error(parameter->location(),
                     "every kernel parameter requires a 'buffer' or "
                     "invocation id attribute");
      if (!isKernel && binding)
        unit_->error(parameter->location(),
                     "a kernel parameter binding requires a kernel "
                     "declaration");
      if (!binding) continue;
      for (std::size_t prior = 0; prior < index; ++prior) {
        auto other = parameters[prior]->kernelParameterBinding();
        if (other && conflicts(*other, *binding))
          unit_->error(parameter->location(), conflictMessage(*binding));
      }
    }
  }

  TranslationUnit* unit_;
  Control* control_;
  TypeTraits traits_;
};

}  // namespace

void Binder::rejectKernelTemplate(FunctionSymbol* function) {
  if (!function->kernelWorkgroupSize()) return;
  if (!is_templated_function(function)) return;
  error(function->location(), std::string{kKernelTemplateMessage});
}

void Binder::applyDeviceAttributes(Symbol* symbol,
                                   std::span<const AttributeRef> attributes) {
  if (attributes.empty() && !symbol->isFunction()) return;
  if (control()->memoryLayout()->deviceProfile() !=
      DeviceProfile::kVulkan11F32) {
    if (!attributes.empty())
      error(attributes.front().location,
            "device attributes require the 'vulkan1.1-f32' device profile");
    return;
  }
  DeviceAttributeBinder device{unit_};
  if (auto parameter = symbol_cast<ParameterSymbol>(symbol))
    return device.bindParameter(parameter, attributes);
  if (auto function = symbol_cast<FunctionSymbol>(symbol))
    return device.bindFunction(function, attributes);
  device.rejectAttributes(attributes);
}

}  // namespace cxx
