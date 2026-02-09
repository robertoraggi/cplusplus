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

#include <cxx/mlir/codegen.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/external_name_encoder.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

// mlir
#include <llvm/BinaryFormat/Dwarf.h>
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
    // skip for now, as we only look for local variable declarations

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
    if (!node->initializer && !gen.control()->is_class(var->type())) continue;

    const auto loc = gen.getLocation(var->location());

    if (var->isStatic()) {
      auto glo = gen.findOrCreateGlobal(var);
      if (!glo) {
        gen.unit_->error(node->initializer
                             ? node->initializer->firstSourceLocation()
                             : var->location(),
                         std::format("cannot create static local variable '{}'",
                                     to_string(var->name())));
      }
      continue;
    }

    auto local = gen.findOrCreateLocal(var);

    if (!local.has_value()) {
      gen.unit_->error(node->initializer
                           ? node->initializer->firstSourceLocation()
                           : var->location(),
                       std::format("cannot find local variable '{}'",
                                   to_string(var->name())));
      continue;
    }

    if (gen.control()->is_array(var->type())) {
      gen.arrayInit(local.value(), var->type(), node->initializer);
      continue;
    }

    if (gen.control()->is_class(var->type())) {
      if (auto ctor = var->constructor()) {
        std::vector<ExpressionResult> args;
        if (node->initializer) {
          if (auto paren = ast_cast<ParenInitializerAST>(node->initializer)) {
            for (auto it = paren->expressionList; it; it = it->next) {
              args.push_back(gen.expression(it->value));
            }
          } else if (auto braced =
                         ast_cast<BracedInitListAST>(node->initializer)) {
            for (auto it = braced->expressionList; it; it = it->next) {
              args.push_back(gen.expression(it->value));
            }
          } else if (auto equal =
                         ast_cast<EqualInitializerAST>(node->initializer)) {
            args.push_back(gen.expression(equal->expression));
          }
        }
        gen.emitCall(node->initializer
                         ? node->initializer->firstSourceLocation()
                         : var->location(),
                     ctor, {local.value()}, args);
        continue;
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
        gen.emitAggregateInit(local.value(), var->type(), braced);
        continue;
      }
    }

    if (node->initializer) {
      ExpressionAST* initExpr = nullptr;
      if (auto equal = ast_cast<EqualInitializerAST>(node->initializer)) {
        initExpr = equal->expression;
      } else {
        initExpr = node->initializer;
      }

      auto expressionResult = gen.expression(initExpr);

      if (gen.control()->is_reference(var->type())) {
        mlir::Value addressToStore = expressionResult.value;

        if (initExpr && initExpr->valueCategory == ValueCategory::kPrValue) {
          auto refType = type_cast<LvalueReferenceType>(var->type());
          auto elementType = refType->elementType();
          auto mlirElementType = gen.convertType(elementType);
          auto tempPtrType =
              gen.builder_.getType<mlir::cxx::PointerType>(mlirElementType);
          auto tempAlloca = mlir::cxx::AllocaOp::create(
              gen.builder_, loc, tempPtrType, gen.getAlignment(elementType));

          mlir::cxx::StoreOp::create(gen.builder_, loc, expressionResult.value,
                                     tempAlloca, gen.getAlignment(elementType));

          addressToStore = tempAlloca;
        }

        mlir::cxx::StoreOp::create(gen.builder_, loc, addressToStore,
                                   local.value(), 8 /* pointer alignment */);
      } else {
        mlir::cxx::StoreOp::create(gen.builder_, loc, expressionResult.value,
                                   local.value(),
                                   gen.getAlignment(var->type()));
      }
    }
  }

  return {};
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
  auto ctx = gen.builder_.getContext();

  auto functionSymbol = ast->symbol;
  const auto functionType = type_cast<FunctionType>(functionSymbol->type());
  const auto returnType = functionType->returnType();

  auto func = gen.findOrCreateFunction(functionSymbol);

  if (!func.getBody().empty()) return {};

  mlir::DistinctAttr id =
      mlir::DistinctAttr::create(gen.builder_.getUnitAttr());

  mlir::LLVM::DIScopeAttr scope;

  if (!functionSymbol->isStatic() && functionSymbol->parent()->isClass()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    if (classSymbol) {
      scope = mlir::dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(
          gen.convertDebugType(classSymbol->type()));
    }
  }

  mlir::StringAttr name = mlir::StringAttr::get(ctx, func.getSymName());

  mlir::StringAttr linkageName = name;

  auto declaratorId = getDeclaratorId(ast->declarator);

  mlir::LLVM::DIFileAttr fileAttr;
  unsigned line = 0;
  unsigned scopeLine = 0;
  std::string_view fileName;

  if (declaratorId && declaratorId->firstSourceLocation()) {
    auto funcLoc =
        gen.unit_->tokenStartPosition(declaratorId->firstSourceLocation());
    fileAttr = gen.getFileAttr(funcLoc.fileName);
    line = funcLoc.line;
    fileName = funcLoc.fileName;
  }

  if (ast->functionBody) {
    auto bodyLoc = ast->functionBody->firstSourceLocation();
    if (bodyLoc) {
      scopeLine = gen.unit_->tokenStartPosition(bodyLoc).line;
    }
  }

  if (!fileAttr) {
    auto classLoc = functionSymbol->location();
    if (classLoc) {
      auto pos = gen.unit_->tokenStartPosition(classLoc);
      fileAttr = gen.getFileAttr(pos.fileName);
      line = pos.line;
      scopeLine = pos.line;
      fileName = pos.fileName;
    } else {
      fileAttr = gen.getFileAttr(std::string_view{""});
      fileName = "";
    }
  }

  mlir::LLVM::DISubprogramFlags subprogramFlags =
      mlir::LLVM::DISubprogramFlags::Definition;

  mlir::SmallVector<mlir::LLVM::DITypeAttr> signatureType;
  signatureType.push_back(gen.convertDebugType(functionType->returnType()));
  if (auto classType = type_cast<ClassType>(functionSymbol->parent()->type());
      classType && !functionSymbol->isStatic()) {
    signatureType.push_back(
        gen.convertDebugType(gen.control()->add_pointer(classType)));
  }
  for (auto paramType : functionType->parameterTypes()) {
    signatureType.push_back(gen.convertDebugType(paramType));
  }

  mlir::LLVM::DISubroutineTypeAttr type =
      mlir::LLVM::DISubroutineTypeAttr::get(ctx, signatureType);

  mlir::SmallVector<mlir::LLVM::DINodeAttr> retainedNodes;
  mlir::SmallVector<mlir::LLVM::DINodeAttr> annotations;

  auto compileUnitAttr = gen.getCompileUnitAttr(fileName);

  auto subprogram = mlir::LLVM::DISubprogramAttr::get(
      ctx, id, compileUnitAttr, scope, name, linkageName,
      compileUnitAttr.getFile(), line, scopeLine, subprogramFlags, type,
      retainedNodes, annotations);

  const auto needsExitValue = !gen.control()->is_void(returnType);

  auto loc = gen.getLocation(ast->firstSourceLocation());
  func->setLoc(mlir::FusedLoc::get({loc}, subprogram, ctx));

  gen.diScopes_[functionSymbol] = subprogram;

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
    auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(exitValueType);
    exitValue = mlir::cxx::AllocaOp::create(gen.builder_, exitValueLoc, ptrType,
                                            gen.getAlignment(returnType));

    auto id = name_cast<Identifier>(functionSymbol->name());
    if (id && id->name() == "main" &&
        is_global_namespace(functionSymbol->parent())) {
      auto zeroOp = mlir::cxx::IntConstantOp::create(
          gen.builder_, loc, gen.convertType(gen.control()->getIntType()), 0);

      mlir::cxx::StoreOp::create(gen.builder_, exitValueLoc, zeroOp, exitValue,
                                 gen.getAlignment(gen.control()->getIntType()));
    }
  }

  std::unordered_map<Symbol*, mlir::Value> locals;
  std::unordered_map<const Name*, int> staticLocalCounts;

  // function state
  std::swap(gen.function_, func);
  std::swap(gen.entryBlock_, entryBlock);
  std::swap(gen.exitBlock_, exitBlock);
  std::swap(gen.exitValue_, exitValue);
  std::swap(gen.locals_, locals);
  std::swap(gen.staticLocalCounts_, staticLocalCounts);

  FunctionSymbol* prevFunctionSymbol = nullptr;
  std::swap(gen.currentFunctionSymbol_, prevFunctionSymbol);
  gen.currentFunctionSymbol_ = functionSymbol;

  mlir::Value thisValue;

  if (!functionSymbol->isStatic() && functionSymbol->parent()->isClass()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());

    auto thisType = gen.convertType(classSymbol->type());
    auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(thisType);

    auto allocaOp = gen.newTemp(gen.control()->add_pointer(classSymbol->type()),
                                ast->firstSourceLocation());
    thisValue = allocaOp;

    if (gen.unit_->language() == LanguageKind::kCXX) {
      gen.attachDebugInfo(
          allocaOp, gen.control()->add_pointer(classSymbol->type()), "this", 1,
          mlir::LLVM::DIFlags::Artificial | mlir::LLVM::DIFlags::ObjectPointer);
    }

    mlir::cxx::StoreOp::create(
        gen.builder_, loc, gen.entryBlock_->getArgument(0), thisValue,
        gen.getAlignment(gen.control()->add_pointer(classSymbol->type())));
  }

  FunctionParametersSymbol* params = nullptr;
  for (auto member : views::members(ast->symbol)) {
    params = symbol_cast<FunctionParametersSymbol>(member);
    if (!params) continue;

    auto args = gen.entryBlock_->getArguments();
    int argc = 0;
    if (thisValue) {
      ++argc;
    }
    for (auto param : views::members(params)) {
      auto arg = symbol_cast<ParameterSymbol>(param);
      if (!arg) continue;

      auto type = gen.convertType(arg->type());
      auto ptrType = gen.builder_.getType<mlir::cxx::PointerType>(type);

      auto loc = gen.getLocation(arg->location());
      auto allocaOp = mlir::cxx::AllocaOp::create(
          gen.builder_, loc, ptrType, gen.getAlignment(arg->type()));

      gen.attachDebugInfo(allocaOp, arg, {}, argc + 1);

      auto value = args[argc];
      ++argc;
      mlir::cxx::StoreOp::create(gen.builder_, loc, value, allocaOp,
                                 gen.getAlignment(arg->type()));

      gen.locals_.emplace(arg, allocaOp);
    }
  }

  std::swap(gen.thisValue_, thisValue);

  allocateLocals(functionSymbol);

  auto functionBodyResult = gen.functionBody(ast->functionBody);

  const auto endLoc = gen.getLocation(ast->lastSourceLocation());

  if (!gen.builder_.getBlock()->mightHaveTerminator()) {
    mlir::cf::BranchOp::create(gen.builder_, endLoc, gen.exitBlock_);
  }

  gen.builder_.setInsertionPointToEnd(gen.exitBlock_);

  if (name_cast<DestructorId>(functionSymbol->name()) && gen.thisValue_) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    if (classSymbol) {
      auto layout = classSymbol->layout();

      auto thisPtrType = gen.builder_.getType<mlir::cxx::PointerType>(
          gen.convertType(classSymbol->type()));

      auto thisPtr = mlir::cxx::LoadOp::create(
          gen.builder_, endLoc, thisPtrType, gen.thisValue_,
          gen.getAlignment(gen.control()->add_pointer(classSymbol->type())));

      auto bases = classSymbol->baseClasses();
      for (auto it = bases.rbegin(); it != bases.rend(); ++it) {
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

        auto basePtrType = gen.builder_.getType<mlir::cxx::PointerType>(
            gen.convertType(baseClassSymbol->type()));

        auto basePtr = mlir::cxx::MemberOp::create(gen.builder_, endLoc,
                                                   basePtrType, thisPtr, index);

        gen.emitCall(ast->lastSourceLocation(), baseDtor, {basePtr}, {});
      }
    }
  }

  if (gen.exitValue_) {
    auto elementType = gen.exitValue_.getType().getElementType();

    auto value =
        mlir::cxx::LoadOp::create(gen.builder_, endLoc, elementType,
                                  gen.exitValue_, gen.getAlignment(returnType));

    mlir::cxx::ReturnOp::create(gen.builder_, endLoc, value->getResults());
  } else {
    mlir::cxx::ReturnOp::create(gen.builder_, endLoc);
  }

  // restore the state
  std::swap(gen.thisValue_, thisValue);
  gen.currentFunctionSymbol_ = prevFunctionSymbol;

  std::swap(gen.function_, func);
  std::swap(gen.entryBlock_, entryBlock);
  std::swap(gen.exitBlock_, exitBlock);
  std::swap(gen.exitValue_, exitValue);
  std::swap(gen.locals_, locals);
  std::swap(gen.staticLocalCounts_, staticLocalCounts);

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
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = gen.specifier(node);
  }

  for (auto node : ListView{ast->bindingList}) {
    auto value = gen.unqualifiedId(node);
  }

  auto initializerResult = gen.expression(ast->initializer);

  return {};
}

auto Codegen::FunctionBodyVisitor::operator()(DefaultFunctionBodyAST* ast)
    -> FunctionBodyResult {
  auto functionSymbol = gen.currentFunctionSymbol_;
  if (!functionSymbol || !functionSymbol->isConstructor()) return {};

  auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
  if (!classSymbol) return {};

  auto sourceLoc = ast->firstSourceLocation();
  if (!sourceLoc) sourceLoc = functionSymbol->location();
  auto loc = gen.getLocation(sourceLoc);

  auto thisPtrType = gen.builder_.getType<mlir::cxx::PointerType>(
      gen.convertType(classSymbol->type()));

  auto thisPtr = mlir::cxx::LoadOp::create(
      gen.builder_, loc, thisPtrType, gen.thisValue_,
      gen.getAlignment(gen.control()->getPointerType(classSymbol->type())));

  auto layout = classSymbol->layout();

  // Check if this is a copy or move constructor
  bool isCopyCtor = (functionSymbol == classSymbol->copyConstructor());
  bool isMoveCtor = (functionSymbol == classSymbol->moveConstructor());

  if (isCopyCtor || isMoveCtor) {
    auto otherPtr = gen.entryBlock_->getArgument(1);

    for (auto base : classSymbol->baseClasses()) {
      auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClassSymbol) continue;

      FunctionSymbol* ctor = isCopyCtor ? baseClassSymbol->copyConstructor()
                                        : baseClassSymbol->moveConstructor();
      if (!ctor) continue;

      int index = 0;
      if (layout) {
        if (auto bi = layout->getBaseInfo(baseClassSymbol)) {
          index = bi->index;
        }
      }

      auto basePtrType = gen.builder_.getType<mlir::cxx::PointerType>(
          gen.convertType(baseClassSymbol->type()));

      auto thisBasePtr = mlir::cxx::MemberOp::create(
          gen.builder_, loc, basePtrType, thisPtr, index);
      auto otherBasePtr = mlir::cxx::MemberOp::create(
          gen.builder_, loc, basePtrType, otherPtr, index);

      gen.emitCall(ast->firstSourceLocation(), ctor, {thisBasePtr},
                   {{otherBasePtr}});
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

      auto fieldType = field->type();
      auto memberPtrType = gen.builder_.getType<mlir::cxx::PointerType>(
          gen.convertType(fieldType));

      auto thisFieldPtr = mlir::cxx::MemberOp::create(
          gen.builder_, loc, memberPtrType, thisPtr, index);
      auto otherFieldPtr = mlir::cxx::MemberOp::create(
          gen.builder_, loc, memberPtrType, otherPtr, index);

      auto unqualFieldType = gen.control()->remove_cv(fieldType);
      auto fieldClassType = type_cast<ClassType>(unqualFieldType);

      if (fieldClassType && fieldClassType->symbol()) {
        // For class type fields, call their copy/move constructor
        auto fieldClassSymbol = fieldClassType->symbol();
        FunctionSymbol* ctor = isCopyCtor ? fieldClassSymbol->copyConstructor()
                                          : fieldClassSymbol->moveConstructor();
        if (ctor) {
          gen.emitCall(ast->firstSourceLocation(), ctor, {thisFieldPtr},
                       {{otherFieldPtr}});
        }
      } else {
        auto mlirFieldType = gen.convertType(fieldType);
        auto value = mlir::cxx::LoadOp::create(gen.builder_, loc, mlirFieldType,
                                               otherFieldPtr,
                                               gen.getAlignment(fieldType));
        mlir::cxx::StoreOp::create(gen.builder_, loc, value, thisFieldPtr,
                                   gen.getAlignment(fieldType));
      }
    }

    gen.emitCtorVtableInit(functionSymbol, loc);
    return {};
  }

  for (auto base : classSymbol->baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;

    FunctionSymbol* defaultCtor = nullptr;
    for (auto ctor : baseClassSymbol->constructors()) {
      auto funcType = type_cast<FunctionType>(ctor->type());
      if (funcType && funcType->parameterTypes().empty()) {
        defaultCtor = ctor;
        break;
      }
    }
    if (!defaultCtor) continue;

    int index = 0;
    if (layout) {
      if (auto bi = layout->getBaseInfo(baseClassSymbol)) {
        index = bi->index;
      }
    }

    auto memberPtrType = gen.builder_.getType<mlir::cxx::PointerType>(
        gen.convertType(baseClassSymbol->type()));

    auto fieldPtr = mlir::cxx::MemberOp::create(gen.builder_, loc,
                                                memberPtrType, thisPtr, index);

    gen.emitCall(ast->firstSourceLocation(), defaultCtor, {fieldPtr}, {});
  }

  for (auto member : views::members(classSymbol)) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;

    auto fieldType = gen.control()->remove_cv(field->type());
    auto classType = type_cast<ClassType>(fieldType);
    if (!classType) continue;

    auto fieldClassSymbol = classType->symbol();
    if (!fieldClassSymbol) continue;

    FunctionSymbol* defaultCtor = nullptr;
    for (auto ctor : fieldClassSymbol->constructors()) {
      auto funcType = type_cast<FunctionType>(ctor->type());
      if (funcType && funcType->parameterTypes().empty()) {
        defaultCtor = ctor;
        break;
      }
    }
    if (!defaultCtor) continue;

    int index = 0;
    if (layout) {
      if (auto fi = layout->getFieldInfo(field)) {
        index = fi->index;
      }
    }

    auto memberPtrType = gen.builder_.getType<mlir::cxx::PointerType>(
        gen.convertType(field->type()));

    auto fieldPtr = mlir::cxx::MemberOp::create(gen.builder_, loc,
                                                memberPtrType, thisPtr, index);

    gen.emitCall(ast->firstSourceLocation(), defaultCtor, {fieldPtr}, {});
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
