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
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/external_name_encoder.h>
#include <cxx/initialization.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <filesystem>
#include <format>

namespace cxx {
struct Codegen::DeclarationVisitor {
  Codegen& gen;

  void allocateLocals(ScopeSymbol* block);

  auto operator()(SimpleDeclarationAST* ast) -> DeclarationResult;
  auto operator()(AsmDeclarationAST* ast) -> DeclarationResult;
  auto operator()(NamespaceAliasDefinitionAST* ast) -> DeclarationResult;
  auto operator()(UsingDeclarationAST* ast) -> DeclarationResult;
  auto operator()(UsingEnumDeclarationAST* ast) -> DeclarationResult;
  auto operator()(UsingDirectiveAST* ast) -> DeclarationResult;
  auto operator()(StaticAssertDeclarationAST* ast) -> DeclarationResult;
  auto operator()(AliasDeclarationAST* ast) -> DeclarationResult;
  auto operator()(OpaqueEnumDeclarationAST* ast) -> DeclarationResult;
  auto operator()(FunctionDefinitionAST* ast) -> DeclarationResult;
  auto operator()(TemplateDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ConceptDefinitionAST* ast) -> DeclarationResult;
  auto operator()(DeductionGuideAST* ast) -> DeclarationResult;
  auto operator()(ExplicitInstantiationAST* ast) -> DeclarationResult;
  auto operator()(ExportDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ExportCompoundDeclarationAST* ast) -> DeclarationResult;
  auto operator()(LinkageSpecificationAST* ast) -> DeclarationResult;
  auto operator()(NamespaceDefinitionAST* ast) -> DeclarationResult;
  auto operator()(EmptyDeclarationAST* ast) -> DeclarationResult;
  auto operator()(AttributeDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ModuleImportDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ParameterDeclarationAST* ast) -> DeclarationResult;
  auto operator()(AccessDeclarationAST* ast) -> DeclarationResult;
  auto operator()(ForRangeDeclarationAST* ast) -> DeclarationResult;
  auto operator()(StructuredBindingDeclarationAST* ast) -> DeclarationResult;
};

struct Codegen::FunctionBodyVisitor {
  Codegen& gen;

  [[nodiscard]] auto emitAnonymousUnionInitializer(SourceLocation sourceLoc,
                                                   ir::ValueRef thisPtr,
                                                   ClassSymbol* classSymbol,
                                                   FieldSymbol* field) -> bool;

  auto operator()(DefaultFunctionBodyAST* ast) -> FunctionBodyResult;
  auto operator()(CompoundStatementFunctionBodyAST* ast) -> FunctionBodyResult;
  auto operator()(TryStatementFunctionBodyAST* ast) -> FunctionBodyResult;
  auto operator()(DeleteFunctionBodyAST* ast) -> FunctionBodyResult;
};

struct Codegen::TemplateParameterVisitor {
  Codegen& gen;

  auto operator()(TemplateTypeParameterAST* ast) -> TemplateParameterResult;
  auto operator()(NonTypeTemplateParameterAST* ast) -> TemplateParameterResult;
  auto operator()(TypenameTypeParameterAST* ast) -> TemplateParameterResult;
  auto operator()(ConstraintTypeParameterAST* ast) -> TemplateParameterResult;
};

auto Codegen::declaration(DeclarationAST* ast) -> DeclarationResult {
  if (ast) return visit(DeclarationVisitor{*this}, ast);
  return {};
}

auto Codegen::templateParameter(TemplateParameterAST* ast)
    -> TemplateParameterResult {
  if (ast) return visit(TemplateParameterVisitor{*this}, ast);
  return {};
}

auto Codegen::functionBody(FunctionBodyAST* ast) -> FunctionBodyResult {
  if (ast) return visit(FunctionBodyVisitor{*this}, ast);
  return {};
}

auto Codegen::nestedNamespaceSpecifier(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierResult {
  if (!ast) return {};

  return {};
}

auto Codegen::typeConstraint(TypeConstraintAST* ast) -> TypeConstraintResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult =
      nestedNameSpecifier(ast->nestedNameSpecifier);

  for (auto node : ListView{ast->templateArgumentList}) {
    auto value = templateArgument(node);
  }

  return {};
}

auto Codegen::usingDeclarator(UsingDeclaratorAST* ast)
    -> UsingDeclaratorResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult =
      nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = unqualifiedId(ast->unqualifiedId);

  return {};
}

auto Codegen::lambdaSpecifier(LambdaSpecifierAST* ast)
    -> LambdaSpecifierResult {
  if (!ast) return {};

  return {};
}

void Codegen::DeclarationVisitor::allocateLocals(ScopeSymbol* block) {
  for (auto symbol : views::members(block)) {
    if (auto nestedBlock = symbol_cast<BlockSymbol>(symbol)) {
      allocateLocals(nestedBlock);
      continue;
    }
    if (auto params = symbol_cast<FunctionParametersSymbol>(symbol)) {
      allocateLocals(params);
      continue;
    }

    if (auto var = symbol_cast<VariableSymbol>(symbol)) {
      if (var->isStatic()) continue;
      if (type_cast<UnresolvedBoundedArrayType>(var->type())) continue;

      auto local = gen.findOrCreateLocal(var);
      if (!local.has_value()) {
        gen.unit_->error(var->location(),
                         std::format("cannot allocate local variable '{}'",
                                     to_string(var->name())));
      }
    }
  }
}

auto Codegen::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationResult {
  if (!gen.function_) {
    for (auto node : ListView{ast->initDeclaratorList}) {
      auto var = symbol_cast<VariableSymbol>(node->symbol);
      if (!var) continue;

      auto glo = gen.findOrCreateGlobal(var);
      if (!glo) {
        gen.unit_->error(node->initializer->firstSourceLocation(),
                         std::format("cannot create global variable '{}'",
                                     to_string(var->name())));
        continue;
      }

      gen.emitGlobalVarInit(var, *glo);
    }

    return {};
  }

  for (auto node : ListView{ast->initDeclaratorList}) {
    auto var = symbol_cast<VariableSymbol>(node->symbol);
    if (!var) continue;
    gen.emitLocalVariableInit(var, node->initializer);
  }

  return {};
}

void Codegen::emitLocalVariableInit(VariableSymbol* var,
                                    ExpressionAST* initializer) {
  const bool isVLA =
      type_cast<UnresolvedBoundedArrayType>(var->type()) != nullptr;
  if (!isVLA && !initializer && !traits.is_class(var->type())) return;

  const auto loc = var->location();

  if (var->isStatic()) {
    auto glo = findOrCreateGlobal(var);
    if (!glo) {
      unit_->error(
          initializer ? initializer->firstSourceLocation() : var->location(),
          std::format("cannot create static local variable '{}'",
                      to_string(var->name())));
      return;
    }
    emitStaticLocalVarInit(var, *glo, initializer);
    return;
  }

  auto local = findOrCreateLocal(var);

  if (!local.has_value()) {
    unit_->error(
        initializer ? initializer->firstSourceLocation() : var->location(),
        std::format("cannot find local variable '{}'", to_string(var->name())));
    return;
  }

  if (traits.is_array(var->type())) {
    arrayInit(local.value(), var->type(), initializer);
    return;
  }

  if (traits.is_class(var->type())) {
    auto registerCleanup = [&] {
      if (traits.has_trivial_destructor(var->type())) return;
      auto classType = unqualified_cast<ClassType>(var->type());
      if (!classType || !classType->symbol()) return;
      auto dtor = classType->symbol()->destructor();
      if (!dtor) return;
      addCleanup(local.value(), completeObjectDtor(dtor));
    };

    auto singleInitExpr = initializerExpression(initializer);
    if (ast_cast<BracedInitListAST>(singleInitExpr)) singleInitExpr = nullptr;

    if (singleInitExpr &&
        singleInitExpr->valueCategory == ValueCategory::kPrValue &&
        singleInitExpr->type &&
        traits.is_same(traits.remove_cv(singleInitExpr->type),
                       traits.remove_cv(var->type()))) {
      if (emitPrvalueInto(local.value(), var->type(), singleInitExpr,
                          initializer->firstSourceLocation())) {
        registerCleanup();
        return;
      }
    }

    if (auto ctor = var->constructor()) {
      auto arguments = constructorArguments(initializer);

      if (initializer && arguments.empty() &&
          requiresZeroInitialization(var->type(), ctor)) {
        emitZeroInitialization(loc, local.value(), var->type());
      }

      (void)emitCtorCall(
          initializer ? initializer->firstSourceLocation() : var->location(),
          ctor, local.value(), std::move(arguments), true);
      registerCleanup();
      return;
    }

    auto braced =
        ast_cast<BracedInitListAST>(Initializer{initializer}.clause());

    if (braced) {
      braced->type = var->type();
      emitAggregateInit(local.value(), var->type(), braced);
      registerCleanup();
      return;
    }
  }

  if (initializer) {
    auto initExpr = initializerExpression(initializer);

    if (traits.is_reference(var->type())) {
      emitReferenceInit(var, local.value(), initExpr, loc);
      return;
    }

    auto expressionResult = expression(initExpr);

    emitter_.store(loc, expressionResult.value, local.value(),
                   getAlignment(var->type()));
  }
}

void Codegen::emitReferenceInit(VariableSymbol* var, ir::ValueRef local,
                                ExpressionAST* initExpr, SourceLocation loc) {
  auto temporary = materializedTemporary(traits, initExpr);

  if (!temporary || temporary.conditional) {
    auto expressionResult = expression(initExpr);
    emitter_.store(loc, expressionResult.value, local, 8);
    return;
  }

  auto temporaryType = traits.remove_cv(temporary.expression->type);
  auto tempPtrType = emitter_.pointerType(convertType(temporaryType));
  auto extendedTemporary =
      emitter_.allocate(loc, tempPtrType, getAlignment(temporaryType));

  ir::ValueRef address = extendedTemporary;

  if (temporary.expression == initExpr) {
    (void)emitPrvalueInto(address, temporaryType, initExpr,
                          initExpr->firstSourceLocation());
  } else {
    auto resultObject = ResultObject{*this, temporary.expression, address};
    address = expression(initExpr).value;
  }

  if (auto classType = unqualified_cast<ClassType>(temporaryType)) {
    if (classType->symbol()) {
      if (auto dtor = classType->symbol()->resolvedDefinition()->destructor())
        addCleanup(extendedTemporary, completeObjectDtor(dtor));
    }
  }

  emitter_.store(loc, address, local, 8);
}

auto Codegen::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->asmQualifierList}) {
    gen.asmQualifier(node);
  }

  for (auto node : ListView{ast->outputOperandList}) {
    gen.asmOperand(node);
  }

  for (auto node : ListView{ast->inputOperandList}) {
    gen.asmOperand(node);
  }

  for (auto node : ListView{ast->clobberList}) {
    gen.asmClobber(node);
  }

  for (auto node : ListView{ast->gotoLabelList}) {
    gen.asmGotoLabel(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(NamespaceAliasDefinitionAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingEnumDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(StaticAssertDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(OpaqueEnumDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationResult {
  auto functionSymbol = ast->symbol;
  if (functionSymbol &&
      (functionSymbol->isConsteval() || functionSymbol->isDeleted()))
    return {};
  const auto isUninstantiatedTemplate = functionSymbol &&
                                        functionSymbol->templateDeclaration() &&
                                        !functionSymbol->isSpecialization();
  if (isUninstantiatedTemplate) return {};

  const auto functionType =
      functionSymbol ? type_cast<FunctionType>(functionSymbol->type())
                     : nullptr;

  if (!functionType) {
    gen.unit_->error(ast->firstSourceLocation(),
                     "unable to generate code for this function definition");
    return {};
  }

  const auto returnType = functionType->returnType();

  auto func = gen.findOrCreateFunction(functionSymbol);

  if (gen.emitter_.functionHasBody(func)) return {};

  const auto needsExitValue = !gen.traits.is_void(returnType);

  const auto returnAbi =
      gen.classifyClassValueAbi(returnType, ClassValueAbiContext::Return);
  const bool sretReturn = returnAbi.kind == ClassValueAbi::Kind::Indirect;

  auto loc = ast->firstSourceLocation();

  if (gen.debugInfo_) {
    gen.buildSubprogramAttr(functionSymbol, ast, func, loc);
  }

  gen.returnType_ = returnType;

  auto functionBodyGuard = ir::FunctionBodyGuard{gen.emitter_, func};

  auto entryBlock = gen.emitter_.createBlock(func);
  auto inputs = gen.emitter_.functionParameterTypes(func);

  for (const auto& input : inputs) {
    (void)gen.emitter_.addBlockParameter(entryBlock, input, loc);
  }

  auto exitBlock = gen.emitter_.createBlock(func);
  ir::ValueRef exitValue;

  gen.emitter_.setInsertionBlock(entryBlock);

  if (needsExitValue) {
    auto exitValueLoc = ast->functionBody
                            ? ast->functionBody->firstSourceLocation()
                            : ast->firstSourceLocation();
    auto exitValueType = gen.convertType(returnType);
    auto ptrType = gen.emitter_.pointerType(exitValueType);
    exitValue = gen.emitter_.allocate(exitValueLoc, ptrType,
                                      gen.getAlignment(returnType));

    auto id = name_cast<Identifier>(functionSymbol->name());
    if (id && id->name() == "main" &&
        is_global_namespace(functionSymbol->parent())) {
      auto intTy = gen.convertType(gen.control()->getIntType());
      auto zeroOp = gen.emitter_.constantInt(loc, intTy, 0);

      gen.emitter_.store(exitValueLoc, zeroOp, exitValue,
                         gen.getAlignment(gen.control()->getIntType()));
    }
  }

  std::unordered_map<Symbol*, ir::ValueRef> locals;
  std::unordered_map<const Name*, int> staticLocalCounts;
  std::vector<Codegen::CleanupScope> cleanupStack;

  std::swap(gen.function_, func);
  std::swap(gen.entryBlock_, entryBlock);
  std::swap(gen.exitBlock_, exitBlock);
  std::swap(gen.exitValue_, exitValue);
  std::swap(gen.locals_, locals);
  std::swap(gen.staticLocalCounts_, staticLocalCounts);
  std::swap(gen.cleanupStack_, cleanupStack);

  ir::ValueRef structorVTTValue;
  std::swap(gen.structorVTTValue_, structorVTTValue);

  FunctionSymbol* prevFunctionSymbol = nullptr;
  std::swap(gen.currentFunctionSymbol_, prevFunctionSymbol);
  gen.currentFunctionSymbol_ = functionSymbol;

  ir::ValueRef thisValue;

  if (functionSymbol->isImplicitObjectMemberFunction()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    auto thisType = gen.convertType(classSymbol->type());
    auto ptrType = gen.emitter_.pointerType(thisType);

    auto allocaOp = gen.newTemp(gen.traits.add_pointer(classSymbol->type()),
                                ast->firstSourceLocation());
    thisValue = allocaOp;

    if (gen.unit_->language() == LanguageKind::kCXX) {
      gen.attachDebugInfo(allocaOp, gen.traits.add_pointer(classSymbol->type()),
                          "this", 1);
    }

    gen.emitter_.store(
        loc,

        gen.emitter_.blockParameter(gen.entryBlock_, sretReturn ? 1 : 0),
        thisValue,
        gen.getAlignment(gen.traits.add_pointer(classSymbol->type())));
  }

  if ((functionSymbol->isConstructor() || functionSymbol->isDestructor()) &&
      gen.requiresVTT(symbol_cast<ClassSymbol>(functionSymbol->parent()))) {
    std::size_t ordinaryArguments = 1;
    if (auto functionType = type_cast<FunctionType>(functionSymbol->type())) {
      for (auto parameterType : functionType->parameterTypes()) {
        if (gen.classifyClassValueAbi(parameterType,
                                      ClassValueAbiContext::Argument)
                .kind != ClassValueAbi::Kind::Empty)
          ++ordinaryArguments;
      }
    }
    const auto entryArgumentCount =
        gen.emitter_.blockParameterCount(gen.entryBlock_);
    if (entryArgumentCount > ordinaryArguments)
      gen.structorVTTValue_ =
          gen.emitter_.blockParameter(gen.entryBlock_, entryArgumentCount - 1);
  }

  FunctionParametersSymbol* params = nullptr;
  for (auto member : views::members(ast->symbol)) {
    params = symbol_cast<FunctionParametersSymbol>(member);
    if (!params) continue;

    const auto argumentCount =
        gen.emitter_.blockParameterCount(gen.entryBlock_);
    unsigned argc = sretReturn ? 1 : 0;
    unsigned debugArgc = 0;
    if (thisValue) {
      ++argc;
      ++debugArgc;
    }
    for (auto param : views::members(params)) {
      auto arg = symbol_cast<ParameterSymbol>(param);
      if (!arg) continue;

      ++debugArgc;

      const auto paramAbi = gen.classifyClassValueAbi(
          arg->type(), ClassValueAbiContext::Argument);
      auto loc = arg->location();

      if (paramAbi.kind == ClassValueAbi::Kind::Indirect) {
        if (argc >= argumentCount) {
          gen.unit_->error(arg->location(),
                           std::format("unexpected argument for function '{}'",
                                       to_string(functionSymbol->name())));
          break;
        }
        gen.locals_.emplace(arg,
                            gen.emitter_.blockParameter(gen.entryBlock_, argc));
        ++argc;
        continue;
      }

      auto type = gen.convertType(arg->type());
      auto ptrType = gen.emitter_.pointerType(type);

      auto allocaOp =
          gen.emitter_.allocate(loc, ptrType, gen.getAlignment(arg->type()));

      gen.attachDebugInfo(allocaOp, arg, {}, debugArgc);

      if (paramAbi.kind == ClassValueAbi::Kind::Empty) {
        gen.locals_.emplace(arg, allocaOp);
        continue;
      }

      if (argc >= argumentCount) {
        gen.unit_->error(arg->location(),
                         std::format("unexpected argument for function '{}'",
                                     to_string(functionSymbol->name())));
        break;
      }

      if (paramAbi.kind == ClassValueAbi::Kind::Coerce) {
        std::vector<ir::ValueRef> values;
        for (std::size_t slot = 0; slot < paramAbi.slots.size(); ++slot) {
          values.push_back(gen.emitter_.blockParameter(gen.entryBlock_, argc));
          ++argc;
        }
        gen.abiStoreClassValue({}, arg->type(), paramAbi, values, allocaOp);
      } else {
        auto value = gen.emitter_.blockParameter(gen.entryBlock_, argc);
        ++argc;
        gen.emitter_.store({}, value, allocaOp, gen.getAlignment(arg->type()));
      }

      gen.locals_.emplace(arg, allocaOp);
    }
  }

  if (params) {
    std::vector<std::pair<ir::ValueRef, FunctionSymbol*>> destroyedInCallee;
    for (auto param : views::members(params)) {
      auto arg = symbol_cast<ParameterSymbol>(param);
      if (!arg || !isClassValueDestroyedInCallee(arg->type())) continue;
      auto classType = unqualified_cast<ClassType>(arg->type());
      auto destructor = classType->symbol()->resolvedDefinition()->destructor();
      if (!destructor) continue;
      destroyedInCallee.emplace_back(gen.locals_[arg],
                                     gen.completeObjectDtor(destructor));
    }

    if (!destroyedInCallee.empty()) {
      gen.pushCleanup();
      for (auto& [storage, destructor] : destroyedInCallee)
        gen.addCleanup(storage, destructor);
    }
  }

  if (auto principal = functionSymbol->structorPrincipal();
      principal && params) {
    std::vector<ir::ValueRef> paramStorage;
    for (auto param : views::members(params)) {
      if (symbol_cast<ParameterSymbol>(param))
        paramStorage.push_back(gen.locals_[param]);
    }

    auto bindAliases = [&](FunctionSymbol* fn) {
      if (!fn) return;
      auto fnParams = fn->functionParameters();
      if (!fnParams) return;
      std::size_t index = 0;
      for (auto param : views::members(fnParams)) {
        if (!symbol_cast<ParameterSymbol>(param)) continue;
        if (index >= paramStorage.size()) break;
        gen.locals_.emplace(param, paramStorage[index]);
        ++index;
      }
    };

    bindAliases(principal);
    if (principal->definition() != principal)
      bindAliases(principal->definition());
  }

  std::swap(gen.thisValue_, thisValue);
  std::swap(gen.structorVTTValue_, structorVTTValue);

  allocateLocals(functionSymbol);

  auto functionBodyResult = gen.functionBody(ast->functionBody);

  const auto endLoc = lastTokenLocation(ast);

  gen.emitBranchWithCleanups(lastTokenLocation(ast), gen.exitBlock_, 0);

  gen.emitter_.setInsertionBlock(gen.exitBlock_);

  if (name_cast<DestructorId>(functionSymbol->name()) && gen.thisValue_ &&
      !functionSymbol->isStructorVariant()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    if (classSymbol) {
      auto thisPtr = gen.loadThisPointer(endLoc, classSymbol);
      auto subobjects = gen.subobjectsInDeclarationOrder(classSymbol);

      for (auto it = subobjects.rbegin(); it != subobjects.rend(); ++it) {
        gen.emitSubobjectDestruction(lastTokenLocation(ast), thisPtr,
                                     classSymbol, *it);
      }
    }
  }

  if (gen.exitValue_) {
    if (sretReturn) {
      auto elementType =
          gen.emitter_.elementType(gen.emitter_.typeOf(gen.exitValue_));
      auto value = gen.emitter_.load(endLoc, elementType, gen.exitValue_,
                                     gen.getAlignment(returnType));
      gen.emitter_.store(endLoc, value,
                         gen.emitter_.blockParameter(gen.entryBlock_, 0),
                         gen.getAlignment(returnType));
      gen.emitter_.ret(endLoc, {});
    } else if (returnAbi.kind == ClassValueAbi::Kind::Coerce) {
      std::vector<ir::ValueRef> values;
      gen.abiLoadClassValue(endLoc, returnType, returnAbi, gen.exitValue_,
                            values);
      gen.emitter_.ret(endLoc, values);
    } else if (returnAbi.kind == ClassValueAbi::Kind::Empty) {
      gen.emitter_.ret(endLoc, {});
    } else {
      auto elementType =
          gen.emitter_.elementType(gen.emitter_.typeOf(gen.exitValue_));

      auto value = gen.emitter_.load(endLoc, elementType, gen.exitValue_,
                                     gen.getAlignment(returnType));

      gen.emitter_.ret(endLoc, {&value, 1});
    }
  } else if (gen.structorReturnsThis(functionSymbol) && gen.thisValue_) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());

    auto thisPtr = gen.loadThisPointer(endLoc, classSymbol);

    gen.emitter_.ret(endLoc, {&thisPtr, 1});
  } else {
    gen.emitter_.ret(endLoc, {});
  }

  gen.emitter_.resolveFunctionControlFlow(gen.function_);

  std::swap(gen.thisValue_, thisValue);
  gen.currentFunctionSymbol_ = prevFunctionSymbol;

  std::swap(gen.function_, func);
  std::swap(gen.entryBlock_, entryBlock);
  std::swap(gen.exitBlock_, exitBlock);
  std::swap(gen.exitValue_, exitValue);
  std::swap(gen.locals_, locals);
  std::swap(gen.staticLocalCounts_, staticLocalCounts);
  std::swap(gen.cleanupStack_, cleanupStack);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExplicitInstantiationAST* ast)
    -> DeclarationResult {
  auto declarationResult = gen.declaration(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationResult {
  auto declarationResult = gen.declaration(ast->declaration);

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ExportCompoundDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(LinkageSpecificationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    auto value = gen.nestedNamespaceSpecifier(node);
  }

  for (auto node : ListView{ast->extraAttributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(AttributeDeclarationAST* ast)
    -> DeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ModuleImportDeclarationAST* ast)
    -> DeclarationResult {
  auto importNameResult = gen.importName(ast->importName);

  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ParameterDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto Codegen::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationResult {
  if (!gen.function_) return {};

  if (ast->hiddenVariable) {
    if (auto var = symbol_cast<VariableSymbol>(ast->hiddenVariable->symbol)) {
      gen.emitLocalVariableInit(var, ast->hiddenVariable->initializer);
    }
  }

  for (auto node : ListView{ast->bindingDeclaratorList}) {
    auto var = symbol_cast<VariableSymbol>(node->symbol);
    if (!var) continue;
    gen.emitLocalVariableInit(var, node->initializer);
  }

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(DefaultFunctionBodyAST* ast)
    -> FunctionBodyResult {
  auto functionSymbol = gen.currentFunctionSymbol_;
  if (!functionSymbol) return {};

  auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
  if (!classSymbol) return {};

  auto sourceLoc = ast->firstSourceLocation();
  if (!sourceLoc) sourceLoc = functionSymbol->location();
  auto loc = sourceLoc;

  if (functionSymbol->isDestructor()) {
    gen.emitCtorVtableInit(functionSymbol, loc);
    return {};
  }

  const bool isCopyAssign =
      functionSymbol == classSymbol->copyAssignmentOperator();
  const bool isMoveAssign =
      functionSymbol == classSymbol->moveAssignmentOperator();

  if (!functionSymbol->isConstructor() && !isCopyAssign && !isMoveAssign)
    return {};

  auto thisPtr = gen.loadThisPointer(loc, classSymbol);

  auto layout = classSymbol->layout();

  if (isCopyAssign || isMoveAssign) {
    if (classSymbol->isUnion()) {
      auto otherPtr = gen.emitter_.blockParameter(gen.entryBlock_, 1);
      auto objectType = gen.convertType(classSymbol->type());
      auto value = gen.emitter_.load(loc, objectType, otherPtr,
                                     gen.getAlignment(classSymbol->type()));
      gen.emitter_.store(loc, value, thisPtr,
                         gen.getAlignment(classSymbol->type()));

      if (gen.exitValue_) {
        gen.emitter_.store(loc, thisPtr, gen.exitValue_,
                           gen.getAlignment(gen.control()->getPointerType(
                               classSymbol->type())));
      }
    }

    return {};
  }

  bool isCopyCtor = (functionSymbol == classSymbol->copyConstructor());
  bool isMoveCtor = (functionSymbol == classSymbol->moveConstructor());

  if (isCopyCtor || isMoveCtor) {
    if (classSymbol->isUnion()) {
      auto otherPtr = gen.emitter_.blockParameter(gen.entryBlock_, 1);
      auto objectType = gen.convertType(classSymbol->type());
      auto value = gen.emitter_.load(loc, objectType, otherPtr,
                                     gen.getAlignment(classSymbol->type()));
      gen.emitter_.store(loc, value, thisPtr,
                         gen.getAlignment(classSymbol->type()));
    }

    return {};
  }

  for (auto subobject : gen.subobjectsInDeclarationOrder(classSymbol)) {
    auto field = symbol_cast<FieldSymbol>(subobject);

    if (field) {
      if (auto initializer = field->initializer()) {
        gen.emitFieldInitializer(
            ast->firstSourceLocation(), field,
            gen.subobjectAddress(loc, thisPtr, classSymbol, field),
            initializer);
        continue;
      }

      if (emitAnonymousUnionInitializer(ast->firstSourceLocation(), thisPtr,
                                        classSymbol, field))
        continue;
    }

    gen.emitSubobjectDefaultConstruction(ast->firstSourceLocation(), thisPtr,
                                         classSymbol, subobject);
  }

  gen.emitCtorVtableInit(functionSymbol, loc);

  return {};
}

void Codegen::emitFieldInitializer(SourceLocation sourceLoc, FieldSymbol* field,
                                   ir::ValueRef fieldPtr,
                                   ExpressionAST* initializer) {
  if (!fieldPtr || !initializer) return;

  auto loc = sourceLoc;

  auto fieldType = traits.remove_cv(field->type());
  auto classType = type_cast<ClassType>(fieldType);

  if (!classType) {
    if (traits.is_array(fieldType)) {
      arrayInit(fieldPtr, field->type(), initializer);
      return;
    }

    if (auto braced = ast_cast<BracedInitListAST>(initializer)) {
      if (!braced->type) braced->type = field->type();
    }

    auto initResult = expression(initializer);
    emitter_.store(loc, initResult.value, fieldPtr,
                   getAlignment(field->type()));
    return;
  }

  auto expr = Initializer{initializer}.clause();

  if (auto paren = ast_cast<ParenInitializerAST>(expr)) {
    auto ctor = field->constructor();
    if (!ctor) return;
    std::vector<ExpressionResult> args;
    for (auto it = paren->expressionList; it; it = it->next)
      args.push_back(expression(it->value));
    (void)emitCtorCall(sourceLoc, ctor, fieldPtr, args,
                       /*completeObject=*/true);
    return;
  }

  if (auto braced = ast_cast<BracedInitListAST>(expr)) {
    if (auto ctor = field->constructor()) {
      const auto passListWhole =
          braced->type &&
          !traits.is_same(traits.remove_cv(braced->type), fieldType);
      std::vector<ExpressionResult> args;
      if (passListWhole) {
        args.push_back(expression(braced));
      } else {
        for (auto it = braced->expressionList; it; it = it->next)
          args.push_back(expression(it->value));
      }
      (void)emitCtorCall(sourceLoc, ctor, fieldPtr, args,
                         /*completeObject=*/true);
    } else {
      if (!braced->type) braced->type = field->type();
      emitAggregateInit(fieldPtr, field->type(), braced);
    }
    return;
  }

  while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
    if (cast->castKind != ImplicitCastKind::kTemporaryMaterializationConversion)
      break;
    expr = cast->expression;
  }

  const auto isSameClassPrvalue =
      expr->valueCategory == ValueCategory::kPrValue && expr->type &&
      traits.is_same(traits.remove_cv(expr->type), fieldType);

  if (!isSameClassPrvalue) {
    if (auto ctor = field->constructor()) {
      std::vector<ExpressionResult> args;
      args.push_back(expression(expr));
      (void)emitCtorCall(sourceLoc, ctor, fieldPtr, args,
                         /*completeObject=*/true);
      return;
    }
  }

  (void)emitPrvalueInto(fieldPtr, field->type(), expr, sourceLoc);
}

auto Codegen::FunctionBodyVisitor::emitAnonymousUnionInitializer(
    SourceLocation sourceLoc, ir::ValueRef thisPtr, ClassSymbol* classSymbol,
    FieldSymbol* field) -> bool {
  auto unionType = unqualified_cast<ClassType>(field->type());
  if (!unionType || !unionType->symbol()) return false;

  auto unionSymbol = unionType->symbol()->resolvedDefinition();
  if (!unionSymbol->isUnion() || unionSymbol->name()) return false;

  auto loc = sourceLoc;

  auto unionPtr = gen.subobjectAddress(loc, thisPtr, classSymbol, field);
  if (!unionPtr) return false;

  for (auto member : views::members(unionSymbol) | views::non_static_fields) {
    auto initializer = member->initializer();
    if (!initializer) continue;

    gen.emitFieldInitializer(
        sourceLoc, member,
        gen.subobjectAddress(loc, unionPtr, unionSymbol, member), initializer);
    return true;
  }

  return false;
}

auto Codegen::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyResult {
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = gen.memInitializer(node);
  }

  if (gen.currentFunctionSymbol_) {
    auto loc = ast->firstSourceLocation();
    gen.emitCtorVtableInit(gen.currentFunctionSymbol_, loc);
  }

  gen.statement(ast->statement);

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(TryStatementFunctionBodyAST* ast)
    -> FunctionBodyResult {
  gen.statement(ast->statement);

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyResult {
  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterResult {
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = gen.templateParameter(node);
  }

  auto requiresClauseResult = gen.requiresClause(ast->requiresClause);

  auto idExpressionResult = gen.expression(ast->idExpression);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterResult {
  auto declarationResult = gen.declaration(ast->declaration);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeIdResult = gen.typeId(ast->typeId);

  return {};
}

auto Codegen::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeConstraintResult = gen.typeConstraint(ast->typeConstraint);
  auto typeIdResult = gen.typeId(ast->typeId);

  return {};
}

void Codegen::asmOperand(AsmOperandAST* ast) {
  auto expressionResult = expression(ast->expression);
}

void Codegen::asmQualifier(AsmQualifierAST* ast) {}

void Codegen::asmClobber(AsmClobberAST* ast) {}

void Codegen::asmGotoLabel(AsmGotoLabelAST* ast) {}
}  // namespace cxx
