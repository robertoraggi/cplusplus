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
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

// mlir
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

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

  // run on the base nodes
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

  // run on the misc nodes
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
  void emitAggregateInit(mlir::Value address, const Type* type,
                         BracedInitListAST* ast);
  void emitDesignatedInit(mlir::Value address, const Type* type,
                          DesignatedInitializerClauseAST* ast);

 private:
  [[nodiscard]] auto getCompileUnitAttr(std::string_view filename)
      -> mlir::LLVM::DICompileUnitAttr;

  [[nodiscard]] auto getFileAttr(const std::string& filename)
      -> mlir::LLVM::DIFileAttr;

  [[nodiscard]] auto getFileAttr(std::string_view filename)
      -> mlir::LLVM::DIFileAttr;

  [[nodiscard]] auto getLocation(SourceLocation loc) -> mlir::Location;

  [[nodiscard]] auto emitTodoStmt(SourceLocation loc, std::string_view message)
      -> mlir::cxx::TodoStmtOp;

  [[nodiscard]] auto emitTodoExpr(SourceLocation loc, std::string_view message)
      -> mlir::cxx::TodoExprOp;

  [[nodiscard]] auto convertType(const Type* type) -> mlir::Type;

  [[nodiscard]] auto convertDebugType(const Type* type)
      -> mlir::LLVM::DITypeAttr;

  [[nodiscard]] auto currentBlockMightHaveTerminator() -> bool;

  [[nodiscard]] auto getAlignment(const Type* type) -> uint64_t;

  [[nodiscard]] auto findOrCreateFunction(FunctionSymbol* functionSymbol)
      -> mlir::cxx::FuncOp;

  void enqueueFunctionBody(FunctionSymbol* symbol);
  void processPendingFunctions();

  [[nodiscard]] auto findOrCreateGlobal(Symbol* symbol)
      -> std::optional<mlir::cxx::GlobalOp>;

  void generateVTable(ClassSymbol* classSymbol);

  void emitCtorVtableInit(FunctionSymbol* functionSymbol, mlir::Location loc);

  [[nodiscard]] static auto computeVtableSlots(ClassSymbol* classSymbol)
      -> std::vector<FunctionSymbol*>;

  [[nodiscard]] auto newTemp(const Type* type, SourceLocation loc)
      -> mlir::cxx::AllocaOp;

  [[nodiscard]] auto findOrCreateLocal(Symbol* symbol)
      -> std::optional<mlir::Value>;

  [[nodiscard]] auto emitCall(SourceLocation loc, FunctionSymbol* symbol,
                              ExpressionResult thisValue,
                              std::vector<ExpressionResult> arguments)
      -> ExpressionResult;

  [[nodiscard]] auto newBlock() -> mlir::Block*;

  [[nodiscard]] auto newUniqueSymbolName(std::string_view prefix)
      -> std::string;

  [[nodiscard]] auto getFloatAttr(const std::optional<ConstValue>& value,
                                  const Type* type)
      -> std::optional<mlir::FloatAttr>;

  [[nodiscard]] auto constValueToAttr(const ConstValue& value, const Type* type)
      -> std::optional<mlir::Attribute>;

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
    };
    std::vector<Entry> entries;
  };

  void pushCleanup();
  void popCleanup(SourceLocation loc);
  void emitBranchWithCleanups(SourceLocation loc, mlir::Block* target,
                              std::size_t targetDepth);
  void addCleanup(mlir::Value address, FunctionSymbol* dtor);

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

  mlir::MLIRContext* context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  mlir::cxx::FuncOp function_;
  TranslationUnit* unit_ = nullptr;
  mlir::Block* entryBlock_ = nullptr;
  mlir::Block* exitBlock_ = nullptr;
  mlir::cxx::AllocaOp exitValue_;
  const Type* returnType_ = nullptr;
  mlir::Value thisValue_;
  mlir::Value targetValue_;
  FunctionSymbol* currentFunctionSymbol_ = nullptr;
  std::unordered_map<ClassSymbol*, mlir::Type> classNames_;
  std::unordered_map<Symbol*, mlir::Value> locals_;
  std::unordered_map<FunctionSymbol*, mlir::cxx::FuncOp> funcOps_;
  std::vector<FunctionSymbol*> pendingFunctions_;
  std::unordered_set<FunctionSymbol*> enqueuedFunctions_;
  std::unordered_set<ClassSymbol*> emittedVTables_;
  std::unordered_map<VariableSymbol*, mlir::cxx::GlobalOp> globalOps_;
  std::unordered_map<std::string_view, int> uniqueSymbolNames_;
  std::unordered_map<const StringLiteral*, mlir::StringAttr> stringLiterals_;
  std::unordered_map<std::string, mlir::LLVM::DIFileAttr> fileAttrs_;
  std::unordered_map<std::string_view, mlir::LLVM::DICompileUnitAttr>
      compileUnitAttrs_;
  Loop loop_;
  Switch switch_;
  std::vector<CleanupScope> cleanupStack_;
  int count_ = 0;
  std::unordered_map<const Type*, mlir::LLVM::DITypeAttr> debugTypeCache_;
  std::unordered_map<Symbol*, mlir::LLVM::DIScopeAttr> diScopes_;
  std::unordered_map<const Name*, int> staticLocalCounts_;
  bool debugInfo_ = true;
};

}  // namespace cxx
