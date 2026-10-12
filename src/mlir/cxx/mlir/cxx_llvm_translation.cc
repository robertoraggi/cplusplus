#include <cxx/mlir/cxx_dialect.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

namespace cxx {
namespace {

[[nodiscard]] auto translateInlineAsm(
    mlir::LLVM::ModuleTranslation& translation,
    llvm::ArrayRef<llvm::Value*> arguments, mlir::TypeRange results,
    llvm::StringRef assembly, llvm::StringRef constraints, bool hasSideEffects)
    -> llvm::InlineAsm* {
  llvm::SmallVector<llvm::Type*> parameterTypes;
  for (auto argument : arguments) parameterTypes.push_back(argument->getType());
  llvm::SmallVector<llvm::Type*> resultTypes;
  for (auto type : results)
    resultTypes.push_back(translation.convertType(type));
  auto& context = translation.getLLVMContext();
  llvm::Type* resultType = llvm::Type::getVoidTy(context);
  if (resultTypes.size() == 1) resultType = resultTypes.front();
  if (resultTypes.size() > 1)
    resultType = llvm::StructType::get(context, resultTypes);
  auto functionType =
      llvm::FunctionType::get(resultType, parameterTypes, false);
  return llvm::InlineAsm::get(functionType, assembly, constraints,
                              hasSideEffects);
}

void translateAsmElementTypes(llvm::CallBase& call,
                              mlir::ArrayAttr elementTypes,
                              mlir::LLVM::ModuleTranslation& translation) {
  for (auto [index, element] : llvm::enumerate(elementTypes))
    if (auto type = mlir::dyn_cast<mlir::TypeAttr>(element))
      call.addParamAttr(
          index, llvm::Attribute::get(
                     translation.getLLVMContext(), llvm::Attribute::ElementType,
                     translation.convertType(type.getValue())));
}

class CxxLLVMTranslation final : public mlir::LLVMTranslationDialectInterface {
 public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  auto convertOperation(mlir::Operation* operation,
                        llvm::IRBuilderBase& builder,
                        mlir::LLVM::ModuleTranslation& translation) const
      -> mlir::LogicalResult override {
    if (auto op = mlir::dyn_cast<mlir::cxx::InlineAsmOp>(operation))
      return convertInlineAsm(op, builder, translation);
    auto op = mlir::dyn_cast<mlir::cxx::AsmGotoOp>(operation);
    if (!op) return mlir::failure();
    auto arguments = translation.lookupValues(op.getInputs());
    auto assembly =
        translateInlineAsm(translation, arguments, op.getResultTypes(),
                           op.getAsmString(), op.getConstraints(), true);
    llvm::SmallVector<llvm::BasicBlock*> destinations;
    for (auto target : op.getDestinations())
      destinations.push_back(translation.lookupBlock(target));
    auto call = builder.CreateCallBr(
        assembly->getFunctionType(), assembly,
        translation.lookupBlock(op.getFallthrough()), destinations, arguments);
    translateAsmElementTypes(*call, op.getArgumentElementTypes(), translation);
    translation.mapBranch(op, call);
    if (op.getResult()) translation.mapValue(op.getResult(), call);
    return mlir::success();
  }

 private:
  [[nodiscard]] auto convertInlineAsm(
      mlir::cxx::InlineAsmOp op, llvm::IRBuilderBase& builder,
      mlir::LLVM::ModuleTranslation& translation) const -> mlir::LogicalResult {
    auto arguments = translation.lookupValues(op.getInputs());
    auto assembly = translateInlineAsm(
        translation, arguments, op.getResultTypes(), op.getAsmString(),
        op.getConstraints(), op.getHasSideEffects());
    auto call =
        builder.CreateCall(assembly->getFunctionType(), assembly, arguments);
    translateAsmElementTypes(*call, op.getArgumentElementTypes(), translation);
    for (auto [index, result] : llvm::enumerate(op.getResults())) {
      llvm::Value* value = call;
      if (op.getNumResults() > 1)
        value = builder.CreateExtractValue(call, index);
      translation.mapValue(result, value);
    }
    return mlir::success();
  }
};

}  // namespace

void registerCxxLLVMTranslation(mlir::MLIRContext& context) {
  auto dialect = context.getOrLoadDialect<mlir::cxx::CxxDialect>();
  if (dialect->getRegisteredInterface<mlir::LLVMTranslationDialectInterface>())
    return;
  dialect->addInterfaces<CxxLLVMTranslation>();
}

}  // namespace cxx
