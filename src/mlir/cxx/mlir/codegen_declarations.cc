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
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/external_name_encoder.h>
#include <cxx/mlir/codegen.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

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

#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen.specifier(node);
  }

  for (auto node : ListView{ast->initDeclaratorList}) {
    auto value = gen.initDeclarator(node);
  }

  auto requiresClauseResult = gen.requiresClause(ast->requiresClause);
#endif

  for (auto node : ListView{ast->initDeclaratorList}) {
    auto var = symbol_cast<VariableSymbol>(node->symbol);
    if (!var) continue;
    gen.emitLocalVariableInit(var, node);
  }

  return {};
}

void Codegen::emitLocalVariableInit(VariableSymbol* var,
                                    InitDeclaratorAST* node) {
  const bool isVLA =
      type_cast<UnresolvedBoundedArrayType>(var->type()) != nullptr;
  if (!isVLA && !node->initializer &&
      !unit_->typeTraits().is_class(var->type()))
    return;

  const auto loc = getLocation(var->location());

  if (var->isStatic()) {
    auto glo = findOrCreateGlobal(var);
    if (!glo) {
      unit_->error(node->initializer ? node->initializer->firstSourceLocation()
                                     : var->location(),
                   std::format("cannot create static local variable '{}'",
                               to_string(var->name())));
    }
    return;
  }

  auto local = findOrCreateLocal(var);

  if (!local.has_value()) {
    unit_->error(
        node->initializer ? node->initializer->firstSourceLocation()
                          : var->location(),
        std::format("cannot find local variable '{}'", to_string(var->name())));
    return;
  }

  if (unit_->typeTraits().is_array(var->type())) {
    arrayInit(local.value(), var->type(), node->initializer);
    return;
  }

  if (unit_->typeTraits().is_class(var->type())) {
    auto registerCleanup = [&] {
      auto classType =
          type_cast<ClassType>(unit_->typeTraits().remove_cv(var->type()));
      if (!classType || !classType->symbol()) return;
      auto dtor = classType->symbol()->destructor();
      if (!dtor) return;
      addCleanup(local.value(), completeObjectDtor(dtor));
    };

    auto singleInitExpr = [&]() -> ExpressionAST* {
      if (!node->initializer) return nullptr;
      if (auto equal = ast_cast<EqualInitializerAST>(node->initializer)) {
        if (ast_cast<BracedInitListAST>(equal->expression)) return nullptr;
        return equal->expression;
      }
      if (auto paren = ast_cast<ParenInitializerAST>(node->initializer)) {
        if (paren->expressionList && !paren->expressionList->next)
          return paren->expressionList->value;
      }
      return nullptr;
    }();

    if (singleInitExpr &&
        singleInitExpr->valueCategory == ValueCategory::kPrValue &&
        singleInitExpr->type &&
        unit_->typeTraits().is_same(
            unit_->typeTraits().remove_cv(singleInitExpr->type),
            unit_->typeTraits().remove_cv(var->type()))) {
      auto loc = getLocation(node->initializer->firstSourceLocation());
      auto value = expression(singleInitExpr);
      if (value.value) {
        auto objectType = convertType(var->type());
        auto resultValue = value.value;
        if (resultValue.getType() != objectType &&
            mlir::isa<mlir::cxx::PointerType>(resultValue.getType())) {
          resultValue =
              mlir::cxx::LoadOp::create(builder_, loc, objectType, resultValue,
                                        getAlignment(var->type()));
        }
        mlir::cxx::StoreOp::create(builder_, loc, resultValue, local.value(),
                                   getAlignment(var->type()));
        registerCleanup();
        return;
      }
    }

    if (auto ctor = var->constructor()) {
      std::vector<ExpressionResult> args;

      auto pushBracedArgs = [&](BracedInitListAST* braced) {
        if (braced->type) {
          args.push_back(expression(braced));
          return;
        }
        for (auto it = braced->expressionList; it; it = it->next) {
          args.push_back(expression(it->value));
        }
      };

      if (node->initializer) {
        if (auto paren = ast_cast<ParenInitializerAST>(node->initializer)) {
          for (auto it = paren->expressionList; it; it = it->next) {
            args.push_back(expression(it->value));
          }
        } else if (auto braced =
                       ast_cast<BracedInitListAST>(node->initializer)) {
          pushBracedArgs(braced);
        } else if (auto equal =
                       ast_cast<EqualInitializerAST>(node->initializer)) {
          if (auto braced = ast_cast<BracedInitListAST>(equal->expression)) {
            pushBracedArgs(braced);
          } else {
            args.push_back(expression(equal->expression));
          }
        }
      }
      (void)emitCtorCall(node->initializer
                             ? node->initializer->firstSourceLocation()
                             : var->location(),
                         ctor, local.value(), args, /*completeObject=*/true);
      registerCleanup();
      return;
    }

    BracedInitListAST* braced = nullptr;
    if (node->initializer) {
      if (auto b = ast_cast<BracedInitListAST>(node->initializer)) {
        braced = b;
      } else if (auto equal =
                     ast_cast<EqualInitializerAST>(node->initializer)) {
        braced = ast_cast<BracedInitListAST>(equal->expression);
      }
    }

    if (braced) {
      braced->type = var->type();
      emitAggregateInit(local.value(), var->type(), braced);
      registerCleanup();
      return;
    }
  }

  if (node->initializer) {
    ExpressionAST* initExpr = nullptr;
    if (auto equal = ast_cast<EqualInitializerAST>(node->initializer)) {
      initExpr = equal->expression;
    } else if (auto paren = ast_cast<ParenInitializerAST>(node->initializer)) {
      if (paren->expressionList && !paren->expressionList->next)
        initExpr = paren->expressionList->value;
    } else {
      initExpr = node->initializer;
    }

    auto expressionResult = expression(initExpr);

    if (unit_->typeTraits().is_reference(var->type())) {
      mlir::Value addressToStore = expressionResult.value;

      if (initExpr && initExpr->valueCategory == ValueCategory::kPrValue) {
        auto refType = type_cast<LvalueReferenceType>(var->type());
        auto elementType = refType->elementType();
        auto mlirElementType = convertType(elementType);
        auto tempPtrType =
            mlir::cxx::PointerType::get(context_, mlirElementType);
        auto tempAlloca = mlir::cxx::AllocaOp::create(
            builder_, loc, tempPtrType, getAlignment(elementType));

        mlir::cxx::StoreOp::create(builder_, loc, expressionResult.value,
                                   tempAlloca, getAlignment(elementType));

        addressToStore = tempAlloca;
      }

      mlir::cxx::StoreOp::create(builder_, loc, addressToStore, local.value(),
                                 8);
    } else {
      mlir::cxx::StoreOp::create(builder_, loc, expressionResult.value,
                                 local.value(), getAlignment(var->type()));
    }
  }
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
#if false
  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->usingDeclaratorList}) {
    auto value = gen.usingDeclarator(node);
  }
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingEnumDeclarationAST* ast)
    -> DeclarationResult {
#if false
  auto enumTypeSpecifierResult = gen.specifier(ast->enumTypeSpecifier);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(StaticAssertDeclarationAST* ast)
    -> DeclarationResult {
#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->gnuAttributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto typeIdResult = gen.typeId(ast->typeId);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(OpaqueEnumDeclarationAST* ast)
    -> DeclarationResult {
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  auto nestedNameSpecifierResult =
      gen.nestedNameSpecifier(ast->nestedNameSpecifier);

  auto unqualifiedIdResult = gen.unqualifiedId(ast->unqualifiedId);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen.specifier(node);
  }
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationResult {
  auto functionSymbol = ast->symbol;
  if (functionSymbol && functionSymbol->templateDeclaration()) return {};

  auto ctx = gen.context_;

  const auto functionType = type_cast<FunctionType>(functionSymbol->type());
  const auto returnType = functionType->returnType();

  auto func = gen.findOrCreateFunction(functionSymbol);

  if (!func.getBody().empty()) return {};

  const auto needsExitValue = !gen.unit_->typeTraits().is_void(returnType);

  const auto returnAbi = gen.classifyClassValueAbi(returnType);
  const bool sretReturn =
      returnAbi.kind == Codegen::ClassValueAbi::Kind::Indirect;

  auto loc = gen.getLocation(ast->firstSourceLocation());

  if (gen.debugInfo_) {
    gen.buildSubprogramAttr(functionSymbol, ast, func, loc);
  }

  gen.returnType_ = returnType;

  auto entryBlock = gen.builder_.createBlock(&func.getBody());
  auto inputs = func.getFunctionType().getInputs();

  for (const auto& input : inputs) {
    entryBlock->addArgument(input, loc);
  }

  auto exitBlock = gen.builder_.createBlock(&func.getBody());
  mlir::cxx::AllocaOp exitValue;

  gen.builder_.setInsertionPointToEnd(entryBlock);

  if (needsExitValue) {
    auto exitValueLoc =
        gen.getLocation(ast->functionBody->firstSourceLocation());
    auto exitValueType = gen.convertType(returnType);
    auto ptrType = mlir::cxx::PointerType::get(gen.context_, exitValueType);
    exitValue = mlir::cxx::AllocaOp::create(gen.builder_, exitValueLoc, ptrType,
                                            gen.getAlignment(returnType));

    auto id = name_cast<Identifier>(functionSymbol->name());
    if (id && id->name() == "main" &&
        is_global_namespace(functionSymbol->parent())) {
      auto intTy = gen.convertType(gen.control()->getIntType());
      auto zeroOp = mlir::arith::ConstantOp::create(
          gen.builder_, loc, intTy, gen.builder_.getIntegerAttr(intTy, 0));

      mlir::cxx::StoreOp::create(gen.builder_, exitValueLoc, zeroOp, exitValue,
                                 gen.getAlignment(gen.control()->getIntType()));
    }
  }

  std::unordered_map<Symbol*, mlir::Value> locals;
  std::unordered_map<const Name*, int> staticLocalCounts;
  std::vector<Codegen::CleanupScope> cleanupStack;

  std::swap(gen.function_, func);
  std::swap(gen.entryBlock_, entryBlock);
  std::swap(gen.exitBlock_, exitBlock);
  std::swap(gen.exitValue_, exitValue);
  std::swap(gen.locals_, locals);
  std::swap(gen.staticLocalCounts_, staticLocalCounts);
  std::swap(gen.cleanupStack_, cleanupStack);

  FunctionSymbol* prevFunctionSymbol = nullptr;
  std::swap(gen.currentFunctionSymbol_, prevFunctionSymbol);
  gen.currentFunctionSymbol_ = functionSymbol;

  mlir::Value thisValue;

  if (functionSymbol->isImplicitObjectMemberFunction()) {
    auto classSymbol = symbol_cast<ClassSymbol>(
        functionSymbol->enclosingNonTemplateParametersScope());
    auto thisType = gen.convertType(classSymbol->type());
    auto ptrType = mlir::cxx::PointerType::get(gen.context_, thisType);

    auto allocaOp =
        gen.newTemp(gen.unit_->typeTraits().add_pointer(classSymbol->type()),
                    ast->firstSourceLocation());
    thisValue = allocaOp;

    if (gen.unit_->language() == LanguageKind::kCXX) {
      gen.attachDebugInfo(
          allocaOp, gen.unit_->typeTraits().add_pointer(classSymbol->type()),
          "this", 1,
          mlir::LLVM::DIFlags::Artificial | mlir::LLVM::DIFlags::ObjectPointer);
    }

    mlir::cxx::StoreOp::create(
        gen.builder_, loc, gen.entryBlock_->getArgument(sretReturn ? 1 : 0),
        thisValue,
        gen.getAlignment(
            gen.unit_->typeTraits().add_pointer(classSymbol->type())));
  }

  FunctionParametersSymbol* params = nullptr;
  for (auto member : views::members(ast->symbol)) {
    params = symbol_cast<FunctionParametersSymbol>(member);
    if (!params) continue;

    auto args = gen.entryBlock_->getArguments();
    int argc = sretReturn ? 1 : 0;
    if (thisValue) {
      ++argc;
    }
    for (auto param : views::members(params)) {
      auto arg = symbol_cast<ParameterSymbol>(param);
      if (!arg) continue;

      const auto paramAbi = gen.classifyClassValueAbi(arg->type());
      auto loc = gen.getLocation(arg->location());

      if (paramAbi.kind == Codegen::ClassValueAbi::Kind::Indirect) {
        if (argc >= args.size()) {
          gen.unit_->error(arg->location(),
                           std::format("unexpected argument for function '{}'",
                                       to_string(functionSymbol->name())));
          break;
        }
        gen.locals_.emplace(arg, args[argc]);
        ++argc;
        continue;
      }

      auto type = gen.convertType(arg->type());
      auto ptrType = mlir::cxx::PointerType::get(gen.context_, type);

      auto allocaOp = mlir::cxx::AllocaOp::create(
          gen.builder_, loc, ptrType, gen.getAlignment(arg->type()));

      gen.attachDebugInfo(allocaOp, arg, {}, argc + 1);

      if (paramAbi.kind == Codegen::ClassValueAbi::Kind::Empty) {
        gen.locals_.emplace(arg, allocaOp);
        continue;
      }

      if (argc >= args.size()) {
        gen.unit_->error(arg->location(),
                         std::format("unexpected argument for function '{}'",
                                     to_string(functionSymbol->name())));
        break;
      }

      auto value = args[argc];
      ++argc;

      if (paramAbi.kind == Codegen::ClassValueAbi::Kind::Scalar) {
        auto scalarType = gen.convertType(paramAbi.scalarType);
        auto scalarPtrType =
            mlir::cxx::PointerType::get(gen.context_, scalarType);
        auto castOp = mlir::cxx::BitcastOp::create(gen.builder_, loc,
                                                   scalarPtrType, allocaOp);
        mlir::cxx::StoreOp::create(gen.builder_, loc, value, castOp,
                                   gen.getAlignment(paramAbi.scalarType));
      } else {
        mlir::cxx::StoreOp::create(gen.builder_, loc, value, allocaOp,
                                   gen.getAlignment(arg->type()));
      }

      gen.locals_.emplace(arg, allocaOp);
    }
  }

  if (auto principal = functionSymbol->structorPrincipal();
      principal && params) {
    std::vector<mlir::Value> paramStorage;
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

  allocateLocals(functionSymbol);

  auto functionBodyResult = gen.functionBody(ast->functionBody);

  const auto endLoc = gen.getLocation(ast->lastSourceLocation());

  gen.emitBranchWithCleanups(ast->lastSourceLocation(), gen.exitBlock_, 0);

  gen.builder_.setInsertionPointToEnd(gen.exitBlock_);

  if (name_cast<DestructorId>(functionSymbol->name()) && gen.thisValue_ &&
      !functionSymbol->isStructorVariant()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    if (classSymbol) {
      auto layout = classSymbol->layout();

      auto thisPtrType = mlir::cxx::PointerType::get(
          gen.context_, gen.convertType(classSymbol->type()));

      auto thisPtr = mlir::cxx::LoadOp::create(
          gen.builder_, endLoc, thisPtrType, gen.thisValue_,
          gen.getAlignment(
              gen.unit_->typeTraits().add_pointer(classSymbol->type())));

      auto bases = classSymbol->baseClasses();
      for (auto it = bases.rbegin(); it != bases.rend(); ++it) {
        if ((*it)->isVirtual()) continue;
        auto baseClassSymbol = symbol_cast<ClassSymbol>((*it)->symbol());
        if (!baseClassSymbol) continue;

        auto baseDtor = baseClassSymbol->destructor();
        if (!baseDtor) continue;

        int index = 0;
        if (layout) {
          if (auto bi = layout->getBaseInfo(baseClassSymbol)) {
            index = bi->index;
          }
        }

        auto basePtr =
            gen.memberAddress(endLoc, thisPtr, baseClassSymbol->type(), index);

        (void)gen.emitCall(ast->lastSourceLocation(), baseDtor, {basePtr}, {});
      }
    }
  }

  if (gen.exitValue_) {
    if (sretReturn) {
      auto elementType = gen.exitValue_.getType().getElementType();
      auto value = mlir::cxx::LoadOp::create(gen.builder_, endLoc, elementType,
                                             gen.exitValue_,
                                             gen.getAlignment(returnType));
      mlir::cxx::StoreOp::create(gen.builder_, endLoc, value,
                                 gen.entryBlock_->getArgument(0),
                                 gen.getAlignment(returnType));
      mlir::cxx::ReturnOp::create(gen.builder_, endLoc);
    } else if (returnAbi.kind == Codegen::ClassValueAbi::Kind::Scalar) {
      auto scalarType = gen.convertType(returnAbi.scalarType);
      auto scalarPtrType =
          mlir::cxx::PointerType::get(gen.context_, scalarType);
      auto castOp = mlir::cxx::BitcastOp::create(gen.builder_, endLoc,
                                                 scalarPtrType, gen.exitValue_);
      auto value =
          mlir::cxx::LoadOp::create(gen.builder_, endLoc, scalarType, castOp,
                                    gen.getAlignment(returnAbi.scalarType));
      mlir::cxx::ReturnOp::create(gen.builder_, endLoc, value->getResults());
    } else if (returnAbi.kind == Codegen::ClassValueAbi::Kind::Empty) {
      mlir::cxx::ReturnOp::create(gen.builder_, endLoc);
    } else {
      auto elementType = gen.exitValue_.getType().getElementType();

      auto value = mlir::cxx::LoadOp::create(gen.builder_, endLoc, elementType,
                                             gen.exitValue_,
                                             gen.getAlignment(returnType));

      mlir::cxx::ReturnOp::create(gen.builder_, endLoc, value->getResults());
    }
  } else if (gen.structorReturnsThis(functionSymbol) && gen.thisValue_) {
    auto classSymbol = symbol_cast<ClassSymbol>(
        functionSymbol->enclosingNonTemplateParametersScope());

    auto thisPtrType = mlir::cxx::PointerType::get(
        gen.context_, gen.convertType(classSymbol->type()));

    auto thisPtr = mlir::cxx::LoadOp::create(
        gen.builder_, endLoc, thisPtrType, gen.thisValue_,
        gen.getAlignment(
            gen.unit_->typeTraits().add_pointer(classSymbol->type())));

    mlir::cxx::ReturnOp::create(gen.builder_, endLoc, thisPtr->getResults());
  } else {
    mlir::cxx::ReturnOp::create(gen.builder_, endLoc);
  }

  gen.resolveLabels();

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
#if false
  for (auto node : ListView{ast->templateParameterList}) {
    auto value = gen.templateParameter(node);
  }

  auto requiresClauseResult = gen.requiresClause(ast->requiresClause);

  auto declarationResult = gen.declaration(ast->declaration);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationResult {
#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  return {};
}

auto Codegen::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationResult {
#if false
  auto explicitSpecifierResult = gen.specifier(ast->explicitSpecifier);

  auto parameterDeclarationClauseResult =
      gen.parameterDeclarationClause(ast->parameterDeclarationClause);

  auto templateIdResult = gen.unqualifiedId(ast->templateId);
#endif
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
#if false
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen.specifier(node);
  }

  auto declaratorResult = gen.declarator(ast->declarator);
  auto expressionResult = gen.expression(ast->expression);
#endif
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
      gen.emitLocalVariableInit(var, ast->hiddenVariable);
    }
  }

  for (auto node : ListView{ast->bindingDeclaratorList}) {
    auto var = symbol_cast<VariableSymbol>(node->symbol);
    if (!var) continue;
    gen.emitLocalVariableInit(var, node);
  }

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(DefaultFunctionBodyAST* ast)
    -> FunctionBodyResult {
  auto functionSymbol = gen.currentFunctionSymbol_;
  if (!functionSymbol) return {};

  auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
  if (!classSymbol) return {};

  const bool isCopyAssign =
      functionSymbol == classSymbol->copyAssignmentOperator();
  const bool isMoveAssign =
      functionSymbol == classSymbol->moveAssignmentOperator();

  if (!functionSymbol->isConstructor() && !isCopyAssign && !isMoveAssign)
    return {};

  auto sourceLoc = ast->firstSourceLocation();
  if (!sourceLoc) sourceLoc = functionSymbol->location();
  auto loc = gen.getLocation(sourceLoc);

  auto thisPtrType = mlir::cxx::PointerType::get(
      gen.context_, gen.convertType(classSymbol->type()));

  auto thisPtr = mlir::cxx::LoadOp::create(
      gen.builder_, loc, thisPtrType, gen.thisValue_,
      gen.getAlignment(gen.control()->getPointerType(classSymbol->type())));

  auto layout = classSymbol->layout();

  if (isCopyAssign || isMoveAssign) {
    if (classSymbol->isUnion()) {
      auto otherPtr = gen.entryBlock_->getArgument(1);
      auto objectType = gen.convertType(classSymbol->type());
      auto value =
          mlir::cxx::LoadOp::create(gen.builder_, loc, objectType, otherPtr,
                                    gen.getAlignment(classSymbol->type()));
      mlir::cxx::StoreOp::create(gen.builder_, loc, value, thisPtr,
                                 gen.getAlignment(classSymbol->type()));

      if (gen.exitValue_) {
        mlir::cxx::StoreOp::create(
            gen.builder_, loc, thisPtr, gen.exitValue_.getResult(),
            gen.getAlignment(
                gen.control()->getPointerType(classSymbol->type())));
      }
    }

    return {};
  }

  bool isCopyCtor = (functionSymbol == classSymbol->copyConstructor());
  bool isMoveCtor = (functionSymbol == classSymbol->moveConstructor());

  if (isCopyCtor || isMoveCtor) {
    if (classSymbol->isUnion()) {
      auto otherPtr = gen.entryBlock_->getArgument(1);
      auto objectType = gen.convertType(classSymbol->type());
      auto value =
          mlir::cxx::LoadOp::create(gen.builder_, loc, objectType, otherPtr,
                                    gen.getAlignment(classSymbol->type()));
      mlir::cxx::StoreOp::create(gen.builder_, loc, value, thisPtr,
                                 gen.getAlignment(classSymbol->type()));
    }

    return {};
  }

  if (classSymbol->isClosureType()) {
    int argIndex = 1;
    const auto& blockArgs = gen.entryBlock_->getArguments();

    for (auto member : views::members(classSymbol)) {
      auto field = symbol_cast<FieldSymbol>(member);
      if (!field || field->isStatic()) continue;

      int index = 0;
      if (layout) {
        if (auto fi = layout->getFieldInfo(field)) {
          index = fi->index;
        }
      }

      auto fieldType = field->type();

      const auto paramAbi = gen.classifyClassValueAbi(fieldType);

      if (paramAbi.kind == Codegen::ClassValueAbi::Kind::Empty) {
        continue;
      }

      if (argIndex >= static_cast<int>(blockArgs.size())) break;

      auto thisFieldPtr = gen.memberAddress(loc, thisPtr, fieldType, index);

      auto argValue = blockArgs[argIndex++];

      switch (paramAbi.kind) {
        case Codegen::ClassValueAbi::Kind::Indirect: {
          auto value = mlir::cxx::LoadOp::create(
              gen.builder_, loc, gen.convertType(fieldType), argValue,
              gen.getAlignment(fieldType));
          mlir::cxx::StoreOp::create(gen.builder_, loc, value, thisFieldPtr,
                                     gen.getAlignment(fieldType));
          break;
        }
        case Codegen::ClassValueAbi::Kind::Scalar: {
          auto scalarType = gen.convertType(paramAbi.scalarType);
          auto scalarPtrType =
              mlir::cxx::PointerType::get(gen.context_, scalarType);
          auto castOp = mlir::cxx::BitcastOp::create(
              gen.builder_, loc, scalarPtrType, thisFieldPtr);
          mlir::cxx::StoreOp::create(gen.builder_, loc, argValue, castOp,
                                     gen.getAlignment(paramAbi.scalarType));
          break;
        }
        default:
          mlir::cxx::StoreOp::create(gen.builder_, loc, argValue, thisFieldPtr,
                                     gen.getAlignment(fieldType));
          break;
      }
    }

    return {};
  }

  auto defaultCtorArgs =
      [&](FunctionSymbol* ctor) -> std::vector<ExpressionResult> {
    std::vector<ExpressionResult> args;
    if (auto fpScope = ctor->functionParameters()) {
      for (auto member : fpScope->members()) {
        auto param = symbol_cast<ParameterSymbol>(member);
        if (!param || !param->defaultArgument()) continue;
        args.push_back(gen.expression(param->defaultArgument()));
      }
    }
    return args;
  };

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;

    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;

    auto defaultCtor = baseClassSymbol->defaultConstructor();
    if (!defaultCtor) continue;

    int index = 0;
    if (layout) {
      if (auto bi = layout->getBaseInfo(baseClassSymbol)) {
        index = bi->index;
      }
    }

    auto fieldPtr =
        gen.memberAddress(loc, thisPtr, baseClassSymbol->type(), index);

    (void)gen.emitCtorCall(ast->firstSourceLocation(), defaultCtor, fieldPtr,
                           defaultCtorArgs(defaultCtor),
                           /*completeObject=*/false);
  }

  for (auto member : views::members(classSymbol)) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;

    int index = 0;
    if (layout) {
      if (auto fi = layout->getFieldInfo(field)) {
        index = fi->index;
      }
    }

    auto fieldType = gen.unit_->typeTraits().remove_cv(field->type());
    auto classType = type_cast<ClassType>(fieldType);

    if (auto initializer = field->initializer()) {
      auto fieldPtr = gen.memberAddress(loc, thisPtr, field->type(), index);

      if (!classType) {
        if (auto braced = ast_cast<BracedInitListAST>(initializer)) {
          if (!braced->type) {
            braced->type = field->type();
          }
        }

        auto initResult = gen.expression(initializer);
        mlir::cxx::StoreOp::create(gen.builder_, loc, initResult.value,
                                   fieldPtr, gen.getAlignment(field->type()));
        continue;
      }

      ExpressionAST* expr = initializer;
      if (auto equal = ast_cast<EqualInitializerAST>(expr))
        expr = equal->expression;

      if (auto paren = ast_cast<ParenInitializerAST>(expr)) {
        auto ctor = field->constructor();
        if (!ctor) continue;
        std::vector<ExpressionResult> args;
        for (auto it = paren->expressionList; it; it = it->next)
          args.push_back(gen.expression(it->value));
        (void)gen.emitCtorCall(ast->firstSourceLocation(), ctor, fieldPtr, args,
                               /*completeObject=*/false);
        continue;
      }

      if (auto braced = ast_cast<BracedInitListAST>(expr)) {
        if (auto ctor = field->constructor()) {
          const auto passListWhole =
              braced->type &&
              !gen.unit_->typeTraits().is_same(
                  gen.unit_->typeTraits().remove_cv(braced->type), fieldType);
          std::vector<ExpressionResult> args;
          if (passListWhole) {
            args.push_back(gen.expression(braced));
          } else {
            for (auto it = braced->expressionList; it; it = it->next)
              args.push_back(gen.expression(it->value));
          }
          (void)gen.emitCtorCall(ast->firstSourceLocation(), ctor, fieldPtr,
                                 args, /*completeObject=*/false);
        } else {
          if (!braced->type) braced->type = field->type();
          gen.emitAggregateInit(fieldPtr, field->type(), braced);
        }
        continue;
      }

      while (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
        if (cast->castKind !=
            ImplicitCastKind::kTemporaryMaterializationConversion)
          break;
        expr = cast->expression;
      }

      const auto isSameClassPrvalue =
          expr->valueCategory == ValueCategory::kPrValue && expr->type &&
          gen.unit_->typeTraits().is_same(
              gen.unit_->typeTraits().remove_cv(expr->type), fieldType);

      if (!isSameClassPrvalue) {
        if (auto ctor = field->constructor()) {
          std::vector<ExpressionResult> args;
          args.push_back(gen.expression(expr));
          (void)gen.emitCtorCall(ast->firstSourceLocation(), ctor, fieldPtr,
                                 args, /*completeObject=*/false);
          continue;
        }
      }

      auto value = gen.expression(expr).value;
      if (value && mlir::isa<mlir::cxx::PointerType>(value.getType())) {
        value = mlir::cxx::LoadOp::create(gen.builder_, loc,
                                          gen.convertType(fieldType), value,
                                          gen.getAlignment(fieldType));
      }
      if (value) {
        mlir::cxx::StoreOp::create(gen.builder_, loc, value, fieldPtr,
                                   gen.getAlignment(field->type()));
      }
    } else if (classType) {
      auto fieldClassSymbol = classType->symbol();
      if (!fieldClassSymbol) continue;

      auto defaultCtor = fieldClassSymbol->defaultConstructor();
      if (!defaultCtor) continue;

      auto fieldPtr = gen.memberAddress(loc, thisPtr, field->type(), index);

      (void)gen.emitCtorCall(ast->firstSourceLocation(), defaultCtor, fieldPtr,
                             defaultCtorArgs(defaultCtor),
                             /*completeObject=*/false);
    }
  }

  gen.emitCtorVtableInit(functionSymbol, loc);

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyResult {
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = gen.memInitializer(node);
  }

  if (gen.currentFunctionSymbol_) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    gen.emitCtorVtableInit(gen.currentFunctionSymbol_, loc);
  }

  gen.statement(ast->statement);

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(TryStatementFunctionBodyAST* ast)
    -> FunctionBodyResult {
#if false
  for (auto node : ListView{ast->memInitializerList}) {
    auto value = gen(node);
  }

#endif

  gen.statement(ast->statement);

#if false
  for (auto node : ListView{ast->handlerList}) {
    auto value = gen(node);
  }
#endif

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
