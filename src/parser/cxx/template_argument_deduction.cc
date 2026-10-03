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
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>

namespace cxx {

namespace {

[[nodiscard]] auto isClassTemplateId(const Type* type) -> bool {
  if (type_cast<TemplateTypeParameterSpecializationType>(type)) return true;
  auto classType = type_cast<ClassType>(type);
  return classType && classType->symbol() &&
         classType->symbol()->isSpecialization();
}

[[nodiscard]] auto isTemplateTemplateParameter(TemplateParameterAST* parameter)
    -> bool {
  return ast_cast<TemplateTypeParameterAST>(parameter) != nullptr;
}

}  // namespace

TemplateArgumentDeduction::TemplateArgumentDeduction(TranslationUnit* unit)
    : unit_(unit),
      traits(unit),
      control_(unit->control()),
      arena_(unit->arena()) {}

void TemplateArgumentDeduction::begin(TemplateDeclarationAST* templateDecl) {
  templateDecl_ = templateDecl;
  deduction_.emplace(unit_, templateDecl);
  explicitArguments_.assign(deduction_->parameterCount(), nullptr);
}

auto TemplateArgumentDeduction::deduce(
    FunctionSymbol* func, List<ExpressionAST*>* args,
    List<TemplateArgumentAST*>* explicitTemplateArgs)
    -> std::optional<List<TemplateArgumentAST*>*> {
  auto templateDecl = func->templateDeclaration();
  if (!templateDecl) return std::nullopt;

  auto functionType = type_cast<FunctionType>(func->type());
  if (!functionType) return std::nullopt;

  begin(templateDecl);
  if (!specifyExplicitArguments(explicitTemplateArgs)) return std::nullopt;
  if (!deduceFromCall(functionType, args)) return std::nullopt;
  return deducedArguments();
}

auto TemplateArgumentDeduction::deduceForGuide(
    TemplateDeclarationAST* templateDecl, const FunctionType* functionType,
    List<ExpressionAST*>* args) -> std::optional<List<TemplateArgumentAST*>*> {
  if (!templateDecl || !functionType) return std::nullopt;

  begin(templateDecl);
  if (!deduceFromCall(functionType, args)) return std::nullopt;
  return deducedArguments();
}

auto TemplateArgumentDeduction::deduceFromTargetType(
    FunctionSymbol* func, const FunctionType* targetType,
    List<TemplateArgumentAST*>* explicitTemplateArgs, bool matchReturnType)
    -> std::optional<List<TemplateArgumentAST*>*> {
  auto templateDecl = func->templateDeclaration();
  if (!templateDecl) return std::nullopt;

  auto functionType = type_cast<FunctionType>(func->type());
  if (!functionType) return std::nullopt;

  begin(templateDecl);
  if (!specifyExplicitArguments(explicitTemplateArgs)) return std::nullopt;

  if (!matchReturnType) {
    functionType = control_->getFunctionType(
        targetType->returnType(), functionType->parameterTypes(),
        functionType->isVariadic(), functionType->cvQualifiers(),
        functionType->refQualifier(), functionType->exceptionSpecification());
  }

  if (!deduceFromTypes(functionType, targetType)) return std::nullopt;
  return deducedArguments();
}

auto TemplateArgumentDeduction::deduceFromConversionTarget(
    FunctionSymbol* func, const Type* targetType)
    -> std::optional<List<TemplateArgumentAST*>*> {
  auto templateDecl = func->templateDeclaration();
  if (!templateDecl) return std::nullopt;

  auto functionType = type_cast<FunctionType>(func->type());
  if (!functionType) return std::nullopt;

  begin(templateDecl);

  auto P = traits.remove_reference(functionType->returnType());
  auto A = targetType;

  if (traits.is_reference(A)) {
    A = traits.remove_reference(A);
  } else if (traits.is_array(P) || traits.is_function(P)) {
    P = traits.decay(P);
  } else {
    P = traits.remove_cv(P);
  }

  A = traits.remove_cv(A);

  if (!isDependent(unit_, P)) {
    auto conversionType = P;
    if (auto pointer = type_cast<PointerType>(P)) {
      if (auto targetPointer = type_cast<PointerType>(A)) {
        auto targetFunction =
            type_cast<FunctionType>(targetPointer->elementType());
        if (targetFunction && !targetFunction->isNoexcept())
          conversionType = control_->getPointerType(
              traits.remove_noexcept(pointer->elementType()));
      }
    }
    if (auto pointer = type_cast<MemberFunctionPointerType>(P)) {
      if (auto targetPointer = type_cast<MemberFunctionPointerType>(A)) {
        if (!targetPointer->functionType()->isNoexcept())
          conversionType = control_->getMemberFunctionPointerType(
              pointer->classType(),
              type_cast<FunctionType>(
                  traits.remove_noexcept(pointer->functionType())));
      }
    }
    if (!traits.is_qualification_convertible(conversionType, A))
      return std::nullopt;
  }

  if (deduction_->mentionsDeducibleParameter(P) && !deduceFromTypes(P, A))
    return std::nullopt;
  return deducedArguments();
}

auto TemplateArgumentDeduction::specifyExplicitArguments(
    List<TemplateArgumentAST*>* explicitTemplateArgs) -> bool {
  if (!explicitTemplateArgs) return true;

  std::vector<TemplateArgumentAST*> written;
  for (auto argument : ListView{explicitTemplateArgs})
    written.push_back(argument);

  std::optional<std::vector<TemplateArgument>> arguments;
  {
    SilentDiagnosticsScope silent{unit_};
    arguments =
        Substitution::writtenTemplateArguments(unit_, explicitTemplateArgs);
    if (silent.hadError()) return false;
  }
  if (!arguments) return false;

  std::size_t index = 0;
  for (int slot = 0; slot < deduction_->parameterCount(); ++slot) {
    if (index == written.size()) break;

    auto parameter = deduction_->parameter(slot);

    if (isPackParameter(parameter)) {
      std::vector<Symbol*> elements;
      for (; index < written.size(); ++index) {
        if (!matchesTemplateParameterKind(parameter, written[index]))
          return false;
        for (const auto& element :
             expand_template_arguments(std::span{&(*arguments)[index], 1}))
          elements.push_back(std::get<Symbol*>(element));
      }
      deduction_->specifyPackPrefix(slot,
                                    control_->getPackArgumentSymbol(elements));
      break;
    }

    if (!matchesTemplateParameterKind(parameter, written[index])) return false;
    explicitArguments_[slot] = written[index];
    deduction_->specify(slot, std::get<Symbol*>((*arguments)[index]));
    ++index;
  }

  return index == written.size();
}

auto TemplateArgumentDeduction::isForwardingReference(const Type* P) const
    -> bool {
  auto reference = type_cast<RvalueReferenceType>(P);
  if (!reference) return false;
  auto parameter = type_cast<TypeParameterType>(reference->elementType());
  return parameter && deduction_->slotOf(parameter) >= 0;
}

auto call_deduction_parameter_type(const TypeTraits& traits, const Type* P)
    -> const Type* {
  if (traits.is_reference(P)) return traits.remove_reference(P);
  return traits.remove_cv(P);
}

auto call_deduction_argument_type(const TypeTraits& traits, const Type* P,
                                  const Type* A, bool forwardsLvalue)
    -> const Type* {
  if (forwardsLvalue)
    return traits.add_lvalue_reference(traits.remove_reference(A));

  A = traits.remove_reference(A);
  if (traits.is_reference(P)) return A;
  if (traits.is_array(A) || traits.is_function(A)) return traits.decay(A);
  return traits.remove_cv(A);
}

auto TemplateArgumentDeduction::bindsLvalueToForwardingReference(
    const Type* P, ExpressionAST* argument) const -> bool {
  if (!argument) return false;
  if (argument->valueCategory != ValueCategory::kLValue) return false;
  return isForwardingReference(P);
}

auto TemplateArgumentDeduction::callArgumentType(const Type* P, const Type* A,
                                                 ExpressionAST* argument) const
    -> const Type* {
  return call_deduction_argument_type(
      traits, P, A, bindsLvalueToForwardingReference(P, argument));
}

auto TemplateArgumentDeduction::deduceFromCall(const FunctionType* functionType,
                                               List<ExpressionAST*>* args)
    -> bool {
  const auto& parameters = functionType->parameterTypes();
  std::size_t index = 0;

  for (auto it = args; it;) {
    if (!it->value) return false;
    if (index == parameters.size()) return functionType->isVariadic();

    auto P = parameters[index];

    if (auto expansion = type_cast<PackExpansionType>(P)) {
      if (index + 1 == parameters.size())
        return deduceFromFunctionParameterPack(expansion->pattern(), it);
      auto slots = deduction_->packSlots(expansion->pattern());
      auto count =
          slots.empty() ? 0uz : deduction_->specifiedPackLength(slots.front());
      for (; count && it; --count) it = it->next;
      ++index;
      continue;
    }

    if (!deduceFromCallArgument(P, it->value)) return false;
    it = it->next;
    ++index;
  }

  return true;
}

auto TemplateArgumentDeduction::deduceFromFunctionParameterPack(
    const Type* P, List<ExpressionAST*>* args) -> bool {
  auto expansion = deduction_->beginExpansion(P);
  if (expansion.slots.empty()) return deduction_->endExpansion(expansion);

  std::size_t index = 0;
  for (auto it = args; it; it = it->next, ++index) {
    if (!it->value) return false;
    if (deduction_->beginElement(expansion, index) &&
        !deduceFromCallArgument(P, it->value))
      return false;
    if (!deduction_->endElement(expansion)) return false;
  }

  return deduction_->endExpansion(expansion);
}

auto TemplateArgumentDeduction::deduceFromCallArgument(const Type* P,
                                                       ExpressionAST* argument)
    -> bool {
  if (auto list = ast_cast<BracedInitListAST>(argument))
    return deduceFromInitializerList(P, list);

  auto A = argument->type;
  if (!A) return false;

  if (!deduction_->mentionsDeducibleParameter(P)) return true;

  if (auto overloadSet = type_cast<OverloadSetType>(A))
    return deduceFromOverloadSet(P, overloadSet, /*takesAddress=*/false);

  if (auto pointer = type_cast<PointerType>(A)) {
    if (auto overloadSet = type_cast<OverloadSetType>(pointer->elementType()))
      return deduceFromOverloadSet(P, overloadSet, /*takesAddress=*/true);
  }

  return deduceFromArgumentType(call_deduction_parameter_type(traits, P),
                                callArgumentType(P, A, argument));
}

auto TemplateArgumentDeduction::deduceFromInitializerList(
    const Type* P, BracedInitListAST* list) -> bool {
  if (!deduction_->mentionsDeducibleParameter(P)) return true;
  if (!list->expressionList) return true;

  auto parameter = traits.remove_cvref(P);
  auto elementType = traits.initializer_list_element_type(parameter);
  ExpressionAST* bound = nullptr;

  if (auto array = type_cast<BoundedArrayType>(parameter)) {
    elementType = array->elementType();
  } else if (auto array = type_cast<UnresolvedBoundedArrayType>(parameter)) {
    elementType = array->elementType();
    bound = array->size();
  }

  if (!elementType) return true;

  std::size_t count = 0;
  for (auto element : ListView{list->expressionList}) {
    ++count;
    if (!element) continue;
    if (!ast_cast<BracedInitListAST>(element) && !element->type) continue;
    if (!deduceFromCallArgument(elementType, element)) return false;
  }

  if (!bound) return true;
  return deduction_->deduceArrayBound(bound, count);
}

auto TemplateArgumentDeduction::deduceFromOverloadSet(const Type* P,
                                                      const OverloadSetType* A,
                                                      bool takesAddress)
    -> bool {
  auto functions = A->symbol()->functions();
  if (std::ranges::any_of(functions, &FunctionSymbol::templateDeclaration))
    return true;

  auto saved = deduction_->state();
  std::optional<TypeDeduction::State> deduced;
  const Type* deducedA = nullptr;

  for (auto function : functions) {
    if (ASTRewriter::evaluateAssociatedConstraints(unit_, function) == false)
      continue;
    deduction_->restore(saved);
    auto argumentType = takesAddress
                            ? traits.address_of_function(function)
                            : callArgumentType(P, function->type(), nullptr);
    if (!deduction_->deduce(call_deduction_parameter_type(traits, P),
                            argumentType))
      continue;
    if (deduced && !traits.is_same(deducedA, argumentType)) {
      deduction_->restore(std::move(saved));
      return true;
    }
    deduced = deduction_->state();
    deducedA = argumentType;
  }

  deduction_->restore(deduced ? std::move(*deduced) : std::move(saved));
  return true;
}

auto TemplateArgumentDeduction::deduceFromTypes(const Type* P, const Type* A)
    -> bool {
  auto saved = deduction_->state();
  if (deduction_->deduce(P, A)) return true;
  deduction_->restore(saved);
  if (deduction_->deduceAllowingConversions(P, A)) return true;
  deduction_->restore(std::move(saved));
  return false;
}

auto TemplateArgumentDeduction::deduceFromArgumentType(const Type* P,
                                                       const Type* A) -> bool {
  if (deduceFromTypes(P, A)) return true;
  return deduceFromBaseClass(P, A);
}

auto TemplateArgumentDeduction::baseClassesOf(ClassSymbol* classSymbol) const
    -> std::vector<ClassSymbol*> {
  std::vector<ClassSymbol*> bases;
  std::vector<ClassSymbol*> worklist{classSymbol->resolvedDefinition()};

  while (!worklist.empty()) {
    auto derived = worklist.back();
    worklist.pop_back();
    if (!derived) continue;
    for (auto base : derived->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      baseClass = baseClass->resolvedDefinition();
      if (std::ranges::contains(bases, baseClass)) continue;
      bases.push_back(baseClass);
      worklist.push_back(baseClass);
    }
  }

  return bases;
}

auto TemplateArgumentDeduction::deduceFromBaseClass(const Type* P,
                                                    const Type* A) -> bool {
  auto patternClass = P;
  auto argumentClass = A;

  if (auto pointer = type_cast<PointerType>(P)) {
    auto argumentPointer = type_cast<PointerType>(A);
    if (!argumentPointer) return false;
    patternClass = pointer->elementType();
    argumentClass = argumentPointer->elementType();
  }

  if (!isClassTemplateId(unqualified_type(patternClass))) return false;

  auto classType = type_cast<ClassType>(unqualified_type(argumentClass));
  if (!classType || !classType->symbol()) return false;

  traits.requireCompleteClass(classType->symbol());

  struct Candidate {
    ClassSymbol* base = nullptr;
    TypeDeduction::State state;
  };

  auto saved = deduction_->state();
  std::vector<Candidate> candidates;

  for (auto base : baseClassesOf(classType->symbol())) {
    deduction_->restore(saved);
    if (!deduction_->deduceAllowingConversions(patternClass, base->type()))
      continue;
    candidates.push_back({base, deduction_->state()});
  }

  auto isHiddenByDerivedCandidate = [&](const Candidate& candidate) {
    return std::ranges::any_of(candidates, [&](const Candidate& other) {
      return other.base != candidate.base &&
             traits.is_base_of(candidate.base->type(), other.base->type());
    });
  };

  std::erase_if(candidates, isHiddenByDerivedCandidate);

  if (candidates.size() != 1) {
    deduction_->restore(std::move(saved));
    return false;
  }

  deduction_->restore(std::move(candidates.front().state));
  return true;
}

auto TemplateArgumentDeduction::collectDeducedSoFar(
    List<TemplateArgumentAST*>* argumentsSoFar)
    -> std::optional<std::vector<TemplateArgument>> {
  if (!argumentsSoFar) return std::vector<TemplateArgument>{};

  SilentDiagnosticsScope silent{unit_};
  auto substitution =
      Substitution::makePartial(unit_, templateDecl_, argumentsSoFar);

  if (!substitution.has_value() || substitution->hadError())
    return std::nullopt;

  return std::move(*substitution).templateArguments();
}

auto TemplateArgumentDeduction::nonTypeParameterType(int slot) const
    -> const Type* {
  auto parameter =
      ast_cast<NonTypeTemplateParameterAST>(deduction_->parameter(slot));
  if (!parameter || !parameter->declaration) return nullptr;

  auto declaredType = parameter->declaration->type;
  if (!declaredType) return nullptr;
  if (containsPlaceholderType(declaredType)) return nullptr;

  auto typeSlot = deduction_->slotOf(declaredType);
  if (typeSlot < 0) return declaredType;

  auto deduced = deduction_->deduced(typeSlot);
  return deduced ? deduced->type() : nullptr;
}

auto TemplateArgumentDeduction::valueSymbol(Symbol* value,
                                            const Type* valueType) const
    -> Symbol* {
  auto variable = symbol_cast<VariableSymbol>(value);
  if (!variable || !variable->constValue()) return nullptr;
  if (!valueType) valueType = variable->type();

  auto converted =
      traits.converted_constant_value(valueType, *variable->constValue());
  if (!converted) return nullptr;

  return control_->getConstantArgumentSymbol(valueType, *converted);
}

auto TemplateArgumentDeduction::symbolArgument(Symbol* symbol,
                                               const Type* type) const
    -> TemplateArgumentAST* {
  auto namedSpec = NamedTypeSpecifierAST::create(arena_);
  namedSpec->symbol = symbol;

  auto typeId = TypeIdAST::create(arena_);
  typeId->typeSpecifierList = make_list_node<SpecifierAST>(arena_, namedSpec);
  typeId->type = type;

  auto argument = TypeTemplateArgumentAST::create(arena_);
  argument->typeId = typeId;
  return argument;
}

auto TemplateArgumentDeduction::typeArgument(const Type* type) const
    -> TemplateArgumentAST* {
  auto typeId = TypeIdAST::create(arena_);
  typeId->type = type;

  auto argument = TypeTemplateArgumentAST::create(arena_);
  argument->typeId = typeId;
  return argument;
}

auto TemplateArgumentDeduction::packArgument(int slot) -> TemplateArgumentAST* {
  auto deduced = symbol_cast<ParameterPackSymbol>(deduction_->deduced(slot));
  if (!deduced)
    return symbolArgument(control_->getPackArgumentSymbol({}), nullptr);

  const auto isValuePack = static_cast<bool>(
      ast_cast<NonTypeTemplateParameterAST>(deduction_->parameter(slot)));
  auto elementType = nonTypeParameterType(slot);

  std::vector<Symbol*> elements;
  elements.reserve(deduced->elements().size());
  for (auto element : deduced->elements()) {
    if (isValuePack) element = valueSymbol(element, elementType);
    if (!element) return nullptr;
    elements.push_back(element);
  }

  return symbolArgument(control_->getPackArgumentSymbol(elements), nullptr);
}

auto TemplateArgumentDeduction::deducedArgument(
    int slot, List<TemplateArgumentAST*>* argumentsSoFar)
    -> TemplateArgumentAST* {
  auto parameter = deduction_->parameter(slot);
  if (isPackParameter(parameter)) return packArgument(slot);
  if (auto argument = explicitArguments_[slot]) return argument;

  auto deduced = deduction_->deduced(slot);
  if (!deduced) {
    if (!hasDefaultTemplateArgument(parameter)) return nullptr;
    auto deducedSoFar = collectDeducedSoFar(argumentsSoFar);
    if (!deducedSoFar) return nullptr;
    return TemplateArguments{unit_}.defaultArgument(templateDecl_, parameter,
                                                    *deducedSoFar);
  }

  if (isTemplateTemplateParameter(parameter))
    return TemplateArguments{unit_}.templateName(deduced);

  if (ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    auto valueType = nonTypeParameterType(slot);
    auto value = valueSymbol(deduced, valueType);
    if (!value) return nullptr;
    return symbolArgument(value, value->type());
  }

  return typeArgument(deduced->type());
}

auto TemplateArgumentDeduction::deducedArguments()
    -> std::optional<List<TemplateArgumentAST*>*> {
  List<TemplateArgumentAST*>* arguments = nullptr;
  auto it = &arguments;

  for (int slot = 0; slot < deduction_->parameterCount(); ++slot) {
    auto argument = deducedArgument(slot, arguments);
    if (!argument) return std::nullopt;
    *it = make_list_node<TemplateArgumentAST>(arena_, argument);
    it = &(*it)->next;
  }

  return arguments;
}

}  // namespace cxx
