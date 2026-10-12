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
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

namespace cxx {

auto Codegen::findOrCreateRuntimeFunction(SourceLocation loc,
                                          std::string_view name,
                                          std::vector<ir::TypeRef> parameters,
                                          std::vector<ir::TypeRef> results)
    -> ir::FunctionRef {
  if (auto existing = findFunction(name)) return existing;

  auto guard = ir::InsertionGuard(emitter_);
  emitter_.setModuleInsertionPoint(true);

  auto functionType = emitter_.functionType(parameters, results, false);

  return declareFunction(
      loc,
      {.name = name, .type = functionType, .linkage = ir::Linkage::External});
}

auto Codegen::exceptionObjectDestructor(SourceLocation loc,
                                        const Type* exceptionType)
    -> ir::ValueRef {
  auto bytePointer = emitter_.pointerType(emitter_.integerType(8));

  auto classType = unqualified_cast<ClassType>(exceptionType);
  if (!classType || traits.has_trivial_destructor(exceptionType))
    return emitter_.nullPointer(loc, bytePointer);

  auto destructor = classType->symbol()->resolvedDefinition()->destructor();
  if (!destructor) return emitter_.nullPointer(loc, bytePointer);

  auto function = findOrCreateFunction(completeObjectDtor(destructor));
  return emitter_.addressOfSymbol(loc, bytePointer, functionName(function));
}

void Codegen::emitThrow(ThrowExpressionAST* ast) {
  auto loc = ast->throwLoc;
  auto bytePointer = emitter_.pointerType(emitter_.integerType(8));

  if (!ast->expression) {
    auto rethrow = findOrCreateRuntimeFunction(loc, "__cxa_rethrow", {}, {});
    (void)emitCallWithUnwind(loc, {.callee = functionName(rethrow)});
    emitter_.unreachable(loc);
    emitter_.setInsertionBlock(newBlock());
    return;
  }

  auto exceptionType = ast->expression->type;
  auto sizeType = convertType(control()->getSizeType());

  auto allocate = findOrCreateRuntimeFunction(loc, "__cxa_allocate_exception",
                                              {sizeType}, {bytePointer});
  std::vector<ir::ValueRef> allocationArguments{emitter_.constantInt(
      loc, sizeType, static_cast<std::int64_t>(getSize(exceptionType)))};
  std::vector<ir::TypeRef> allocationResults{bytePointer};
  auto exceptionObject = emitter_
                             .call(loc, {.callee = functionName(allocate),
                                         .arguments = allocationArguments,
                                         .results = allocationResults})
                             .front();

  auto objectAddress = emitter_.bitcast(
      loc, emitter_.pointerType(convertType(exceptionType)), exceptionObject);
  pushCleanup();
  auto freeException = findOrCreateRuntimeFunction(loc, "__cxa_free_exception",
                                                   {bytePointer}, {});
  cleanupStack_.back().entries.push_back({.address = exceptionObject,
                                          .destructor = freeException,
                                          .exceptionalOnly = true});
  (void)emitPrvalueInto(objectAddress, exceptionType, ast->expression, loc);
  cancelCleanup(exceptionObject);
  popCleanup(loc);

  auto throwFunction = findOrCreateRuntimeFunction(
      loc, "__cxa_throw", {bytePointer, bytePointer, bytePointer}, {});
  std::vector<ir::ValueRef> throwArguments{
      exceptionObject, typeInfoAddress(loc, exceptionType),
      exceptionObjectDestructor(loc, exceptionType)};
  (void)emitCallWithUnwind(loc, {.callee = functionName(throwFunction),
                                 .arguments = throwArguments});
  emitter_.unreachable(loc);
  emitter_.setInsertionBlock(newBlock());
}

void Codegen::ensureExceptionPersonality(SourceLocation loc) {
  constexpr std::string_view name = "__gxx_personality_v0";
  if (!findFunction(name)) {
    auto guard = ir::InsertionGuard{emitter_};
    emitter_.setModuleInsertionPoint(true);
    auto type = emitter_.functionType({}, {emitter_.integerType(32)}, true);
    (void)declareFunction(loc, {.name = name, .type = type});
  }
  emitter_.functionPersonality(function_, name);
}

auto Codegen::emitCallWithUnwind(SourceLocation loc, ir::CallInfo info)
    -> std::vector<ir::ValueRef> {
  if (!unit_->config().exceptionsEnabled || info.kind == ir::CallKind::Builtin)
    return emitter_.call(loc, info);
  auto depth =
      exceptionScopes_.empty() ? 0 : exceptionScopes_.back().cleanupDepth;
  auto cleanups = collectCleanupSnapshot(depth, true);
  if (exceptionScopes_.empty() && cleanups.empty())
    return emitter_.call(loc, info);
  ensureExceptionPersonality(loc);
  auto pointer = emitter_.pointerType(emitter_.integerType(8));
  std::vector<ir::ValueRef> clauses;
  bool catchesAll = false;
  for (auto scope = exceptionScopes_.rbegin(); scope != exceptionScopes_.rend();
       ++scope) {
    if (scope->terminates) {
      clauses.push_back(emitter_.nullPointer(loc, pointer));
      break;
    }
    for (auto handler : ListView{scope->handlers}) {
      if (auto declaration = ast_cast<TypeExceptionDeclarationAST>(
              handler->exceptionDeclaration)) {
        auto type = traits.remove_cvref(declaration->symbol->type());
        clauses.push_back(typeInfoAddress(loc, type));
      } else {
        clauses.push_back(emitter_.nullPointer(loc, pointer));
        catchesAll = true;
      }
    }
    if (catchesAll) break;
  }
  auto normal = newBlock();
  auto unwind = newBlock();
  info.normalDestination = normal;
  info.unwindDestination = unwind;
  auto results = emitter_.call(loc, info);
  emitter_.setInsertionBlock(unwind);
  auto landing = emitter_.landingPad(loc, clauses, !cleanups.empty());
  if (!exceptionScopes_.empty()) {
    auto scope = exceptionScopes_.back();
    emitter_.store(
        loc, landing[0], scope.exception,
        getAlignment(control()->getPointerType(control()->getVoidType())));
    emitter_.store(loc, landing[1], scope.selector, 4);
    emitCleanupSnapshot(loc, cleanups, true);
    branch(loc, scope.dispatch);
  } else {
    auto resume = newBlock();
    emitCleanupSnapshot(loc, cleanups, true);
    branch(loc, resume);
    emitter_.setInsertionBlock(resume);
    emitter_.resumeException(loc, landing[0], landing[1]);
  }
  emitter_.setInsertionBlock(normal);
  return results;
}

void Codegen::emitTerminateOnException(const std::function<void()>& body,
                                       SourceLocation loc) {
  if (!unit_->config().exceptionsEnabled) {
    body();
    return;
  }
  auto pointerType = control()->getPointerType(control()->getVoidType());
  auto pointer = convertType(pointerType);
  auto dispatch = newBlock();
  auto merge = newBlock();
  ExceptionScope scope{nullptr,
                       dispatch,
                       newTemp(pointerType, loc),
                       newTemp(control()->getIntType(), loc),
                       cleanupStack_.size(),
                       true};
  exceptionScopes_.push_back(scope);
  body();
  branch(loc, merge);
  exceptionScopes_.pop_back();
  emitter_.setInsertionBlock(dispatch);
  auto exception =
      emitter_.load(loc, pointer, scope.exception, getAlignment(pointerType));
  auto beginCatch = findOrCreateRuntimeFunction(loc, "__cxa_begin_catch",
                                                {pointer}, {pointer});
  std::vector<ir::ValueRef> arguments{exception};
  std::vector<ir::TypeRef> results{pointer};
  (void)emitter_.call(loc, {.callee = functionName(beginCatch),
                            .arguments = arguments,
                            .results = results});
  auto terminate = findOrCreateRuntimeFunction(loc, "_ZSt9terminatev", {}, {});
  (void)emitter_.call(loc, {.callee = functionName(terminate)});
  emitter_.unreachable(loc);
  emitter_.setInsertionBlock(merge);
}

void Codegen::emitTry(const std::function<void()>& body,
                      List<HandlerAST*>* handlers, SourceLocation loc,
                      bool rethrowOnFallthrough) {
  auto pointer = emitter_.pointerType(emitter_.integerType(8));
  auto exceptionType = control()->getPointerType(control()->getVoidType());
  auto dispatch = newBlock();
  auto merge = newBlock();
  ExceptionScope scope{handlers, dispatch, newTemp(exceptionType, loc),
                       newTemp(control()->getIntType(), loc),
                       cleanupStack_.size()};
  exceptionScopes_.push_back(scope);
  body();
  branch(loc, merge);
  exceptionScopes_.pop_back();
  emitter_.setInsertionBlock(dispatch);
  auto exception =
      emitter_.load(loc, pointer, scope.exception, getAlignment(exceptionType));
  auto selector =
      emitter_.load(loc, emitter_.integerType(32), scope.selector, 4);
  for (auto handler : ListView{handlers}) {
    auto declaration =
        ast_cast<TypeExceptionDeclarationAST>(handler->exceptionDeclaration);
    auto handlerBlock = newBlock();
    auto next = newBlock();
    if (declaration) {
      auto type = traits.remove_cvref(declaration->symbol->type());
      auto typeId = emitter_.exceptionTypeId(loc, typeInfoAddress(loc, type));
      auto matches =
          emitter_.compareInt(loc, ir::IntPredicate::Equal, selector, typeId);
      emitter_.condBranch(loc, matches, handlerBlock, next);
    } else {
      branch(loc, handlerBlock);
    }
    emitter_.setInsertionBlock(handlerBlock);
    auto beginCatch = findOrCreateRuntimeFunction(loc, "__cxa_begin_catch",
                                                  {pointer}, {pointer});
    std::vector<ir::ValueRef> arguments{exception};
    std::vector<ir::TypeRef> results{pointer};
    auto object = emitter_
                      .call(loc, {.callee = functionName(beginCatch),
                                  .arguments = arguments,
                                  .results = results})
                      .front();
    pushCleanup();
    auto endCatch = findOrCreateRuntimeFunction(loc, "__cxa_end_catch", {}, {});
    cleanupStack_.back().entries.push_back({object, endCatch});
    if (declaration && declaration->symbol) {
      auto symbol = declaration->symbol;
      auto type = traits.remove_reference(symbol->type());
      auto local = findOrCreateLocal(symbol).value();
      if (traits.is_reference(symbol->type())) {
        auto address = object;
        if (traits.is_pointer(type) && traits.is_const(type)) {
          address = newTemp(type, loc);
          emitter_.store(loc, object, address, getAlignment(type));
        } else if (traits.is_pointer(type)) {
          auto size = emitter_.constantInt(
              loc, convertType(control()->getSizeType()),
              control()->memoryLayout()->unwindExceptionSize());
          address = emitter_.pointerAdd(loc, pointer, exception, size);
        }
        emitter_.store(loc, address, local, getAlignment(symbol->type()));
      } else if (traits.is_class_or_union(type)) {
        locals_.insert_or_assign(declaration->exceptionObject, object);
        emitTerminateOnException(
            [&] {
              if (auto constructor = symbol->constructor()) {
                (void)emitCtorCall(
                    loc, constructor, local,
                    constructorArguments(
                        declaration->exceptionObject->initializer()),
                    true);
              } else {
                (void)emitPrvalueInto(
                    local, type, declaration->exceptionObject->initializer(),
                    loc);
              }
            },
            loc);
        addArrayCleanup(local, type);
      } else {
        auto value = object;
        if (!traits.is_pointer(type))
          value =
              emitter_.load(loc, convertType(type), object, getAlignment(type));
        emitter_.store(loc, value, local, getAlignment(type));
      }
    }
    statement(handler->statement);
    if (rethrowOnFallthrough && !currentBlockMightHaveTerminator()) {
      auto rethrow = ThrowExpressionAST::create(unit_->arena());
      rethrow->throwLoc = loc;
      emitThrow(rethrow);
    }
    popCleanup(loc);
    branch(loc, merge);
    emitter_.setInsertionBlock(next);
  }
  if (!exceptionScopes_.empty()) {
    auto parent = exceptionScopes_.back();
    emitter_.store(loc, exception, parent.exception,
                   getAlignment(exceptionType));
    emitter_.store(loc, selector, parent.selector, 4);
    emitCleanupSnapshot(loc, collectCleanupSnapshot(parent.cleanupDepth, true),
                        true);
    branch(loc, parent.dispatch);
  } else {
    emitter_.resumeException(loc, exception, selector);
  }
  emitter_.setInsertionBlock(merge);
}

}  // namespace cxx
