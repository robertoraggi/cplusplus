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

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/class_value_abi.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types_fwd.h>

// mlir
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace mlir::func {
class FuncOp;
}

namespace cxx {
class TranslationUnit;
class Control;

class Codegen {
 public:
  explicit Codegen(mlir::MLIRContext& context, TranslationUnit* unit,
                   bool debugInfo = true);
  ~Codegen();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto control() const -> Control*;

  struct UnitResult {
    mlir::ModuleOp module;
  };

  struct DeclarationResult {};

  struct ExpressionResult {
    mlir::Value value;
    ValueCategory category = ValueCategory::kNone;
    bool isRValueMaterialized = false;
  };

  struct TemplateParameterResult {};
  struct SpecifierResult {};
  struct PtrOperatorResult {};
  struct CoreDeclaratorResult {};
  struct DeclaratorChunkResult {};
  struct UnqualifiedIdResult {};
  struct NestedNameSpecifierResult {};
  struct FunctionBodyResult {};
  struct TemplateArgumentResult {};
  struct ExceptionSpecifierResult {};
  struct RequirementResult {};
  struct NewInitializerResult {};
  struct MemInitializerResult {};
  struct LambdaCaptureResult {};
  struct ExceptionDeclarationResult {};
  struct AttributeSpecifierResult {};
  struct AttributeTokenResult {};

  struct SplicerResult {};
  struct GlobalModuleFragmentResult {};
  struct PrivateModuleFragmentResult {};
  struct ModuleDeclarationResult {};
  struct ModuleNameResult {};
  struct ModuleQualifierResult {};
  struct ModulePartitionResult {};
  struct ImportNameResult {};
  struct InitDeclaratorResult {};
  struct DeclaratorResult {};
  struct UsingDeclaratorResult {};
  struct EnumeratorResult {};
  struct TypeIdResult {};
  struct HandlerResult {};
  struct BaseSpecifierResult {};
  struct RequiresClauseResult {};
  struct ParameterDeclarationClauseResult {};
  struct TrailingReturnTypeResult {};
  struct LambdaSpecifierResult {};
  struct TypeConstraintResult {};
  struct AttributeArgumentClauseResult {};
  struct AttributeResult {};
  struct AttributeUsingPrefixResult {};
  struct NewPlacementResult {};
  struct NestedNamespaceSpecifierResult {};

  [[nodiscard]] auto operator()(UnitAST* ast) -> UnitResult;

  [[nodiscard]] auto declaration(DeclarationAST* ast) -> DeclarationResult;

  void statement(StatementAST* ast);

  enum struct ExpressionFormat {
    kValue,
    kSideEffect,
  };

  [[nodiscard]] auto expression(
      ExpressionAST* ast, ExpressionFormat format = ExpressionFormat::kValue)
      -> ExpressionResult;

  void condition(ExpressionAST* ast, mlir::Block* trueBlock,
                 mlir::Block* falseBlock);

  void conditionWithCleanups(ExpressionAST* ast, mlir::Block* trueBlock,
                             mlir::Block* falseBlock);

  [[nodiscard]] auto templateParameter(TemplateParameterAST* ast)
      -> TemplateParameterResult;

  [[nodiscard]] auto specifier(SpecifierAST* ast) -> SpecifierResult;

  [[nodiscard]] auto ptrOperator(PtrOperatorAST* ast) -> PtrOperatorResult;

  [[nodiscard]] auto coreDeclarator(CoreDeclaratorAST* ast)
      -> CoreDeclaratorResult;

  [[nodiscard]] auto declaratorChunk(DeclaratorChunkAST* ast)
      -> DeclaratorChunkResult;

  [[nodiscard]] auto unqualifiedId(UnqualifiedIdAST* ast)
      -> UnqualifiedIdResult;

  [[nodiscard]] auto nestedNameSpecifier(NestedNameSpecifierAST* ast)
      -> NestedNameSpecifierResult;

  [[nodiscard]] auto functionBody(FunctionBodyAST* ast) -> FunctionBodyResult;

  [[nodiscard]] auto templateArgument(TemplateArgumentAST* ast)
      -> TemplateArgumentResult;

  [[nodiscard]] auto exceptionSpecifier(ExceptionSpecifierAST* ast)
      -> ExceptionSpecifierResult;

  [[nodiscard]] auto requirement(RequirementAST* ast) -> RequirementResult;

  [[nodiscard]] auto newInitializer(NewInitializerAST* ast)
      -> NewInitializerResult;

  [[nodiscard]] auto memInitializer(MemInitializerAST* ast)
      -> MemInitializerResult;

  [[nodiscard]] auto lambdaCapture(LambdaCaptureAST* ast)
      -> LambdaCaptureResult;

  [[nodiscard]] auto exceptionDeclaration(ExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;

  [[nodiscard]] auto attributeSpecifier(AttributeSpecifierAST* ast)
      -> AttributeSpecifierResult;

  [[nodiscard]] auto attributeToken(AttributeTokenAST* ast)
      -> AttributeTokenResult;

  [[nodiscard]] auto splicer(SplicerAST* ast) -> SplicerResult;

  [[nodiscard]] auto globalModuleFragment(GlobalModuleFragmentAST* ast)
      -> GlobalModuleFragmentResult;

  [[nodiscard]] auto privateModuleFragment(PrivateModuleFragmentAST* ast)
      -> PrivateModuleFragmentResult;

  [[nodiscard]] auto moduleDeclaration(ModuleDeclarationAST* ast)
      -> ModuleDeclarationResult;

  [[nodiscard]] auto moduleName(ModuleNameAST* ast) -> ModuleNameResult;

  [[nodiscard]] auto moduleQualifier(ModuleQualifierAST* ast)
      -> ModuleQualifierResult;

  [[nodiscard]] auto modulePartition(ModulePartitionAST* ast)
      -> ModulePartitionResult;

  [[nodiscard]] auto importName(ImportNameAST* ast) -> ImportNameResult;

  [[nodiscard]] auto initDeclarator(InitDeclaratorAST* ast)
      -> InitDeclaratorResult;

  [[nodiscard]] auto declarator(DeclaratorAST* ast) -> DeclaratorResult;

  [[nodiscard]] auto usingDeclarator(UsingDeclaratorAST* ast)
      -> UsingDeclaratorResult;

  [[nodiscard]] auto enumerator(EnumeratorAST* ast) -> EnumeratorResult;

  [[nodiscard]] auto typeId(TypeIdAST* ast) -> TypeIdResult;

  [[nodiscard]] auto handler(HandlerAST* ast) -> HandlerResult;

  [[nodiscard]] auto baseSpecifier(BaseSpecifierAST* ast)
      -> BaseSpecifierResult;

  [[nodiscard]] auto requiresClause(RequiresClauseAST* ast)
      -> RequiresClauseResult;

  [[nodiscard]] auto parameterDeclarationClause(
      ParameterDeclarationClauseAST* ast) -> ParameterDeclarationClauseResult;

  [[nodiscard]] auto trailingReturnType(TrailingReturnTypeAST* ast)
      -> TrailingReturnTypeResult;

  [[nodiscard]] auto lambdaSpecifier(LambdaSpecifierAST* ast)
      -> LambdaSpecifierResult;

  [[nodiscard]] auto typeConstraint(TypeConstraintAST* ast)
      -> TypeConstraintResult;

  [[nodiscard]] auto attributeArgumentClause(AttributeArgumentClauseAST* ast)
      -> AttributeArgumentClauseResult;

  [[nodiscard]] auto attribute(AttributeAST* ast) -> AttributeResult;

  [[nodiscard]] auto attributeUsingPrefix(AttributeUsingPrefixAST* ast)
      -> AttributeUsingPrefixResult;

  [[nodiscard]] auto newPlacement(NewPlacementAST* ast) -> NewPlacementResult;

  [[nodiscard]] auto nestedNamespaceSpecifier(NestedNamespaceSpecifierAST* ast)
      -> NestedNamespaceSpecifierResult;

  void asmOperand(AsmOperandAST* ast);
  void asmQualifier(AsmQualifierAST* ast);
  void asmClobber(AsmClobberAST* ast);
  void asmGotoLabel(AsmGotoLabelAST* ast);
  void arrayInit(mlir::Value address, const Type* type, ExpressionAST* init);
  [[nodiscard]] auto emitInPlaceConstruction(mlir::Value address,
                                             ExpressionAST* ast) -> bool;

  void emitAggregateInit(mlir::Value address, const Type* type,
                         BracedInitListAST* ast);
  void emitLocalVariableInit(VariableSymbol* var, ExpressionAST* initializer);
  void emitReferenceInit(VariableSymbol* var, mlir::Value local,
                         ExpressionAST* initExpr, mlir::Location loc);
  void emitDesignatedInit(mlir::Value address, const Type* type,
                          DesignatedInitializerClauseAST* ast);

 private:
  [[nodiscard]] auto getCompileUnitAttr() -> mlir::LLVM::DICompileUnitAttr;

  [[nodiscard]] auto getOrCreateFileAttr(const std::string& filename)
      -> mlir::LLVM::DIFileAttr;

  [[nodiscard]] auto getFileAttr(const std::string& filename)
      -> mlir::LLVM::DIFileAttr;

  [[nodiscard]] auto getFileAttr(std::string_view filename)
      -> mlir::LLVM::DIFileAttr;

  [[nodiscard]] auto getFileAttrAt(SourceLocation location)
      -> mlir::LLVM::DIFileAttr;

  [[nodiscard]] auto getLocation(SourceLocation loc) -> mlir::Location;

  [[nodiscard]] auto emitTodoStmt(SourceLocation loc, std::string_view message)
      -> mlir::cxx::TodoStmtOp;

  [[nodiscard]] auto emitTodoExpr(SourceLocation loc, std::string_view message)
      -> mlir::cxx::TodoExprOp;

  [[nodiscard]] auto convertType(const Type* type) -> mlir::Type;

  [[nodiscard]] auto convertBaseSubobjectType(ClassSymbol* classSymbol)
      -> mlir::Type;

  [[nodiscard]] auto buildClassMemberTypes(ClassSymbol* classSymbol,
                                           bool includeVirtualBases)
      -> std::vector<mlir::Type>;

  [[nodiscard]] auto convertBaseEmbedding(ClassSymbol* baseSymbol,
                                          std::uint64_t availableBytes)
      -> mlir::Type;

  [[nodiscard]] auto convertDebugType(const Type* type)
      -> mlir::LLVM::DITypeAttr;

  [[nodiscard]] auto currentBlockMightHaveTerminator() -> bool;

  [[nodiscard]] auto getAlignment(const Type* type) -> uint64_t;

  void reportDeferredBodyDiagnostics(FunctionSymbol* functionSymbol);

  [[nodiscard]] auto findOrCreateFunction(FunctionSymbol* functionSymbol)
      -> mlir::cxx::FuncOp;

  [[nodiscard]] auto computeFunctionSignature(FunctionSymbol* functionSymbol)
      -> mlir::cxx::FunctionType;

  [[nodiscard]] auto computeFunctionSignature(const FunctionType* functionType,
                                              FunctionSymbol* functionSymbol)
      -> mlir::cxx::FunctionType;

  [[nodiscard]] auto structorReturnsThis(FunctionSymbol* symbol) -> bool;

  [[nodiscard]] auto classifyClassValueAbi(const Type* type) -> ClassValueAbi;

  [[nodiscard]] auto classValueAddress(SourceLocation loc, const Type* type,
                                       mlir::Value value) -> mlir::Value;

  [[nodiscard]] auto abiLowerClassArgument(SourceLocation loc,
                                           const Type* paramType,
                                           mlir::Value value) -> mlir::Value;

  [[nodiscard]] auto abiPrepareResult(
      SourceLocation loc, const Type* returnType,
      mlir::SmallVector<mlir::Type>& resultTypes, mlir::Value resultObject = {})
      -> mlir::Value;

  [[nodiscard]] auto abiFinishResult(SourceLocation loc, const Type* returnType,
                                     mlir::cxx::CallOp callOp,
                                     mlir::Value sretTemp) -> ExpressionResult;

  [[nodiscard]] auto emitVirtualBaseAddress(mlir::Location loc,
                                            mlir::Value objectPtr,
                                            ClassSymbol* fromClass,
                                            ClassSymbol* vbaseClass)
      -> mlir::Value;

  [[nodiscard]] auto adjustByVtableWord(mlir::Location loc,
                                        mlir::Value objectPtrI8,
                                        std::int64_t byteOffset) -> mlir::Value;

  [[nodiscard]] auto emitBaseClassAddress(mlir::Location loc,
                                          mlir::Value objectPtr,
                                          ClassSymbol* fromClass,
                                          ClassSymbol* targetClass)
      -> mlir::Value;

  [[nodiscard]] auto emitDerivedClassAddress(mlir::Location loc,
                                             mlir::Value objectPtr,
                                             ClassSymbol* fromClass,
                                             ClassSymbol* targetClass)
      -> mlir::Value;

  [[nodiscard]] auto navigateToClass(mlir::Location loc, mlir::Value value,
                                     ClassSymbol* from, ClassSymbol* to)
      -> mlir::Value;

  [[nodiscard]] auto memberAddress(mlir::Location loc, mlir::Value objectPtr,
                                   const Type* memberType, std::uint32_t index)
      -> mlir::Value;
  [[nodiscard]] auto memberAddress(mlir::Location loc, mlir::Value objectPtr,
                                   mlir::Type memberType, std::uint32_t index)
      -> mlir::Value;

  [[nodiscard]] auto loadThisPointer(mlir::Location loc,
                                     ClassSymbol* classSymbol) -> mlir::Value;

  [[nodiscard]] auto loadEnclosingObject(mlir::Location loc,
                                         ClassSymbol* targetClass,
                                         ClassSymbol*& objectClass)
      -> mlir::Value;

  struct ClassSubobjectShape {
    ClassSymbol* classSymbol = nullptr;
    const Type* elementType = nullptr;
    std::uint64_t elementCount = 1;
  };

  [[nodiscard]] auto classSubobjectShape(const Type* type) const
      -> std::optional<ClassSubobjectShape>;

  [[nodiscard]] auto subobjectType(Symbol* subobject) const -> const Type*;

  [[nodiscard]] auto subobjectIndex(ClassSymbol* classSymbol,
                                    Symbol* subobject) const
      -> std::optional<int>;

  [[nodiscard]] auto subobjectAddress(mlir::Location loc, mlir::Value objectPtr,
                                      ClassSymbol* classSymbol,
                                      Symbol* subobject) -> mlir::Value;

  [[nodiscard]] auto subobjectElementAddresses(mlir::Location loc,
                                               mlir::Value subobjectPtr,
                                               const ClassSubobjectShape& shape)
      -> std::vector<mlir::Value>;

  [[nodiscard]] auto subobjectsInDeclarationOrder(
      ClassSymbol* classSymbol) const -> std::vector<Symbol*>;

  [[nodiscard]] auto isImplicitlyInitializedSubobject(ClassSymbol* classSymbol,
                                                      Symbol* subobject) const
      -> bool;

  [[nodiscard]] auto defaultConstructorArguments(FunctionSymbol* constructor)
      -> std::vector<ExpressionResult>;

  void emitSubobjectDestruction(SourceLocation loc, mlir::Value objectPtr,
                                ClassSymbol* classSymbol, Symbol* subobject);

  void emitSubobjectDefaultConstruction(SourceLocation loc,
                                        mlir::Value objectPtr,
                                        ClassSymbol* classSymbol,
                                        Symbol* subobject);

  void enqueueFunctionBody(FunctionSymbol* symbol);
  void processPendingFunctions();
  void resolveLabels();

  [[nodiscard]] auto findOrCreateGlobal(Symbol* symbol)
      -> std::optional<mlir::cxx::GlobalOp>;

  [[nodiscard]] auto findOrCreateExternField(FieldSymbol* field)
      -> mlir::cxx::GlobalOp;

  void emitGlobalVarInit(VariableSymbol* var, mlir::cxx::GlobalOp global);

  void generateVTable(ClassSymbol* classSymbol);

  [[nodiscard]] auto findOrCreateCxaAtexit(mlir::Location loc)
      -> mlir::cxx::FuncOp;

  [[nodiscard]] auto findOrCreateDsoHandle(mlir::Location loc)
      -> mlir::cxx::GlobalOp;

  void emitGlobalVarDtorRegistration(VariableSymbol* defVar,
                                     FunctionSymbol* dtor,
                                     mlir::cxx::GlobalOp global,
                                     mlir::Location loc);

  void emitCtorVtableInit(FunctionSymbol* functionSymbol, mlir::Location loc);

  [[nodiscard]] auto vtableSlotIndex(FunctionSymbol* function) -> int;

  void emitVTableGroupOp(mlir::Location loc, llvm::StringRef name,
                         const VTableLayout::Group& group);

  void generateSecondaryVTables(ClassSymbol* classSymbol);

  [[nodiscard]] auto encodeSecondaryVTableName(ClassSymbol* classSymbol,
                                               ClassSymbol* base)
      -> std::string;

  [[nodiscard]] auto findOrCreateThunk(
      FunctionSymbol* target, llvm::StringRef thunkName,
      const std::function<mlir::Value(
          mlir::Value rawThisI8, mlir::Location loc)>& computeAdjustedThisI8)
      -> mlir::cxx::FuncOp;

  [[nodiscard]] auto findOrCreateThisAdjustingThunk(FunctionSymbol* target,
                                                    std::int64_t offset)
      -> mlir::cxx::FuncOp;

  [[nodiscard]] auto findOrCreateVirtualThunk(FunctionSymbol* target,
                                              std::int64_t vcallSlotByteOffset)
      -> mlir::cxx::FuncOp;

  [[nodiscard]] auto resolveVptrField(mlir::Value basePtr,
                                      ClassSymbol* baseClassSym,
                                      mlir::Location loc) -> mlir::Value;

  [[nodiscard]] auto newTemp(const Type* type, SourceLocation loc)
      -> mlir::cxx::AllocaOp;

  [[nodiscard]] auto findOrCreateLocal(Symbol* symbol)
      -> std::optional<mlir::Value>;

  [[nodiscard]] auto emitCall(SourceLocation loc, FunctionSymbol* symbol,
                              ExpressionResult thisValue,
                              std::vector<ExpressionResult> arguments,
                              bool isVirtualDispatch = false)
      -> ExpressionResult;

  [[nodiscard]] auto emitCall(SourceLocation loc,
                              const FunctionType* functionType,
                              FunctionSymbol* symbol, bool isVirtualDispatch,
                              ExpressionResult thisValue,
                              std::vector<ExpressionResult> arguments,
                              mlir::Value resultObject = {})
      -> ExpressionResult;

  [[nodiscard]] auto emitCtorCall(SourceLocation loc, FunctionSymbol* ctor,
                                  mlir::Value thisPtr,
                                  std::vector<ExpressionResult> args,
                                  bool completeObject) -> ExpressionResult;

  [[nodiscard]] static auto completeObjectDtor(FunctionSymbol* dtor)
      -> FunctionSymbol*;

  [[nodiscard]] auto newBlock() -> mlir::Block*;

  [[nodiscard]] auto newUniqueSymbolName(std::string_view prefix)
      -> std::string;

  [[nodiscard]] auto getFloatAttr(const std::optional<ConstValue>& value,
                                  const Type* type)
      -> std::optional<mlir::FloatAttr>;

  [[nodiscard]] auto constValueToAttr(const ConstValue& value, const Type* type)
      -> std::optional<mlir::Attribute>;

  [[nodiscard]] auto emitConstInitValue(mlir::OpBuilder& builder,
                                        mlir::Location loc, const Type* type,
                                        const ConstValue& value) -> mlir::Value;

  void branch(mlir::Location loc, mlir::Block* block,
              mlir::ValueRange operands = {});

  struct Loop {
    mlir::Block* continueBlock = nullptr;
    mlir::Block* breakBlock = nullptr;
    std::size_t continueCleanupDepth = 0;
    std::size_t breakCleanupDepth = 0;
  };

  struct CleanupScope {
    struct Entry {
      mlir::Value address;
      FunctionSymbol* destructor;
      mlir::Value activeFlag;
    };
    std::vector<Entry> entries;
    bool isFullExpression = false;
    mlir::Block* startBlock = nullptr;
    mlir::Operation* startAnchor = nullptr;
  };

  void pushCleanup();
  void pushFullExpressionCleanup();
  void popCleanup(SourceLocation loc);
  void emitBranchWithCleanups(SourceLocation loc, mlir::Block* target,
                              std::size_t targetDepth);
  void addCleanup(mlir::Value address, FunctionSymbol* dtor);
  void addTemporaryCleanup(mlir::Value address, const Type* type);

  [[nodiscard]] auto allocaInEntryBlock(mlir::Location loc,
                                        mlir::Type elementType,
                                        std::uint64_t alignment) -> mlir::Value;

  void hoistAllocaToEntryBlock(mlir::Value address);

  [[nodiscard]] auto createConditionalCleanupFlag(mlir::Location loc,
                                                  CleanupScope& scope)
      -> mlir::Value;

  class FullExpression {
   public:
    FullExpression(Codegen& gen, SourceLocation endLoc);
    ~FullExpression();

   private:
    Codegen& gen_;
    SourceLocation endLoc_;
  };

  class ConditionalEvaluation {
   public:
    explicit ConditionalEvaluation(Codegen& gen) : gen_(gen) {
      ++gen_.conditionalEvaluationDepth_;
    }
    ~ConditionalEvaluation() { --gen_.conditionalEvaluationDepth_; }

   private:
    Codegen& gen_;
  };

  [[nodiscard]] auto takeResultObject(ExpressionAST* ast) -> mlir::Value;

  auto emitPrvalueInto(mlir::Value object, const Type* objectType,
                       ExpressionAST* ast, SourceLocation loc) -> bool;

  class ResultObject {
   public:
    ResultObject(Codegen& gen, ExpressionAST* ast, mlir::Value address);
    ~ResultObject();

    [[nodiscard]] auto wasConsumed() const -> bool;

   private:
    Codegen& gen_;
    ExpressionAST* savedOwner_;
    mlir::Value savedAddress_;
    bool savedInitialized_;
  };

  struct CleanupSnapshot {
    mlir::SmallVector<mlir::Value> addresses;
    mlir::SmallVector<mlir::Attribute> destructors;
    mlir::SmallVector<mlir::Attribute> depths;
    mlir::SmallVector<mlir::Value> activeFlags;
    mlir::SmallVector<std::int32_t> activeFlagIndices;
  };

  [[nodiscard]] auto collectCleanupSnapshot(std::size_t targetDepth = 0)
      -> CleanupSnapshot;

  struct Switch {
    std::vector<std::int64_t> caseValues;
    std::vector<mlir::Block*> caseDestinations;
    mlir::Block* defaultDestination = nullptr;
  };

  struct UnitVisitor;
  struct DeclarationVisitor;
  struct StatementVisitor;
  struct ExpressionVisitor;
  struct TemplateParameterVisitor;
  struct SpecifierVisitor;
  struct PtrOperatorVisitor;
  struct CoreDeclaratorVisitor;
  struct DeclaratorChunkVisitor;
  struct UnqualifiedIdVisitor;
  struct NestedNameSpecifierVisitor;
  struct FunctionBodyVisitor;
  struct TemplateArgumentVisitor;
  struct ExceptionSpecifierVisitor;
  struct RequirementVisitor;
  struct NewInitializerVisitor;
  struct MemInitializerVisitor;
  struct LambdaCaptureVisitor;
  struct ExceptionDeclarationVisitor;
  struct AttributeSpecifierVisitor;
  struct AttributeTokenVisitor;

  struct ConvertType;
  struct ConvertDebugType;

  void attachDebugInfo(mlir::cxx::AllocaOp allocaOp, Symbol* symbol,
                       std::string_view name = {}, unsigned arg = 0);

  void attachDebugInfo(mlir::cxx::AllocaOp allocaOp, const Type* type,
                       std::string_view name, unsigned arg,
                       mlir::LLVM::DIFlags flags);

  [[nodiscard]] auto getOrCreateDIScope(Symbol* symbol)
      -> mlir::LLVM::DIScopeAttr;

  void buildSubprogramAttr(FunctionSymbol* functionSymbol,
                           FunctionDefinitionAST* ast, mlir::cxx::FuncOp func,
                           mlir::Location loc);

  [[nodiscard]] auto buildSubroutineTypeAttr(FunctionSymbol* functionSymbol)
      -> mlir::LLVM::DISubroutineTypeAttr;

  mlir::MLIRContext* context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  mlir::cxx::FuncOp function_;
  TranslationUnit* unit_ = nullptr;
  TypeTraits traits;
  mlir::Block* entryBlock_ = nullptr;
  mlir::Block* exitBlock_ = nullptr;
  mlir::cxx::AllocaOp exitValue_;
  const Type* returnType_ = nullptr;
  mlir::Value thisValue_;
  mlir::Value targetValue_;
  FunctionSymbol* currentFunctionSymbol_ = nullptr;
  std::unordered_map<ClassSymbol*, mlir::Type> classNames_;
  std::unordered_map<ClassSymbol*, mlir::Type> baseSubobjectTypeNames_;
  std::unordered_map<Symbol*, mlir::Value> locals_;
  std::unordered_map<FunctionSymbol*, mlir::cxx::FuncOp> funcOps_;
  std::vector<FunctionSymbol*> pendingFunctions_;
  std::unordered_set<FunctionSymbol*> enqueuedFunctions_;
  std::unordered_set<ClassSymbol*> emittedVTables_;
  std::unordered_map<VariableSymbol*, mlir::cxx::GlobalOp> globalOps_;
  std::unordered_map<FieldSymbol*, mlir::cxx::GlobalOp> externFieldGlobalOps_;
  std::unordered_set<VariableSymbol*> emittedGlobalVarInits_;
  std::unordered_map<std::string_view, int> uniqueSymbolNames_;
  std::unordered_map<const StringLiteral*, mlir::StringAttr> stringLiterals_;
  std::unordered_map<std::string, mlir::LLVM::DIFileAttr> fileAttrs_;
  mlir::LLVM::DICompileUnitAttr compileUnitAttr_;
  Loop loop_;
  Switch switch_;
  std::vector<CleanupScope> cleanupStack_;
  ExpressionAST* resultObjectOwner_ = nullptr;
  bool resultObjectInitialized_ = false;
  int conditionalEvaluationDepth_ = 0;
  mlir::Value resultObjectAddress_;
  int count_ = 0;
  int globalVarInitCount_ = 0;
  int globalVarDtorCount_ = 0;
  std::unordered_map<const Type*, mlir::LLVM::DITypeAttr> debugTypeCache_;
  std::unordered_map<Symbol*, mlir::LLVM::DIScopeAttr> diScopes_;
  std::unordered_map<const Name*, int> staticLocalCounts_;
  bool debugInfo_ = true;
  bool isWasmTarget_ = false;
};
}  // namespace cxx
