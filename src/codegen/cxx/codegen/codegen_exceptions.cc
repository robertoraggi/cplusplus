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

  auto function = findOrCreateFunction(destructor);
  return emitter_.addressOfSymbol(loc, bytePointer, functionName(function));
}

void Codegen::emitThrow(ThrowExpressionAST* ast) {
  auto loc = ast->throwLoc;
  auto bytePointer = emitter_.pointerType(emitter_.integerType(8));

  if (!ast->expression) {
    auto rethrow = findOrCreateRuntimeFunction(loc, "__cxa_rethrow", {}, {});
    (void)emitter_.call(loc, {.callee = functionName(rethrow)});
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
  (void)emitPrvalueInto(objectAddress, exceptionType, ast->expression, loc);

  auto throwFunction = findOrCreateRuntimeFunction(
      loc, "__cxa_throw", {bytePointer, bytePointer, bytePointer}, {});
  std::vector<ir::ValueRef> throwArguments{
      exceptionObject, typeInfoAddress(loc, exceptionType),
      exceptionObjectDestructor(loc, exceptionType)};
  (void)emitter_.call(loc, {.callee = functionName(throwFunction),
                            .arguments = throwArguments});
  emitter_.unreachable(loc);
  emitter_.setInsertionBlock(newBlock());
}

}  // namespace cxx
