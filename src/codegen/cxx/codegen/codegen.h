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
#include <cxx/codegen/emitter.h>
#include <cxx/codegen/source_context_observer.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types_fwd.h>

#include <functional>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace cxx {
class TranslationUnit;
class Control;

class Codegen {
 public:
  struct Options {
    bool debugInfo = true;
    std::string debugCompilationDirectory;
    SourceContextObserver* sourceObserver = nullptr;
  };

  explicit Codegen(ir::Emitter& emitter, TranslationUnit* unit,
                   Options options);
  ~Codegen();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit_;
  }

  [[nodiscard]] auto control() const -> Control*;

  [[nodiscard]] auto debugCompilationDirectory() const -> std::string_view {
    return options_.debugCompilationDirectory;
  }

  [[nodiscard]] auto nullMemberObjectPointer() const -> std::int64_t;

  struct UnitResult {
    ir::ModuleRef module;
  };

  struct DeclarationResult {};

  enum class AtomicResult { kOld, kUpdated };

  struct ExpressionResult {
    ir::ValueRef value;
    ValueCategory category = ValueCategory::kNone;
    bool isRValueMaterialized = false;
    const Type* type = nullptr;
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

  void condition(ExpressionAST* ast, ir::BlockRef trueBlock,
                 ir::BlockRef falseBlock);

  void conditionWithCleanups(ExpressionAST* ast, ir::BlockRef trueBlock,
                             ir::BlockRef falseBlock);

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

  void inlineAsm(AsmDeclarationAST* ast);
  void arrayInit(ir::ValueRef address, const Type* type, ExpressionAST* init);

  void emitArrayInitialization(SourceLocation loc, ir::ValueRef address,
                               const Type* type, FunctionSymbol* constructor,
                               ExpressionAST* initializer);

  void emitArrayCopy(SourceLocation loc, ir::ValueRef destination,
                     ir::ValueRef source, const Type* arrayType,
                     FunctionSymbol* elementConstructor);
  void emitArrayDestruction(SourceLocation loc, ir::ValueRef address,
                            const Type* elementType, ir::ValueRef count);

  [[nodiscard]] auto emitWholeArrayCopy(SourceLocation loc,
                                        ir::ValueRef address, const Type* type,
                                        FunctionSymbol* elementConstructor,
                                        ExpressionAST* initializer) -> bool;

  [[nodiscard]] auto arrayElementAddress(SourceLocation loc,
                                         ir::ValueRef address,
                                         const Type* elementType,
                                         std::size_t index) -> ir::ValueRef;
  [[nodiscard]] auto emitInPlaceConstruction(ir::ValueRef address,
                                             ExpressionAST* ast) -> bool;

  void emitAggregateInit(ir::ValueRef address, const Type* type,
                         BracedInitListAST* ast,
                         ir::ValueRef elementCount = {});
  void emitAggregateInit(ir::ValueRef address, const Type* type,
                         List<ExpressionAST*>* initializerList,
                         SourceLocation location);
  void emitAggregateElements(ir::ValueRef address, const Type* type,
                             List<ExpressionAST*>* initializerList,
                             SourceLocation location,
                             ir::ValueRef initializedCount,
                             ir::ValueRef elementCount = {});
  void emitConstructorInitialization(SourceLocation loc, ir::ValueRef address,
                                     const Type* type,
                                     FunctionSymbol* constructor,
                                     ExpressionAST* initializer);
  void emitArrayElementInit(ir::ValueRef elementAddress,
                            const Type* elementType, ExpressionAST* node);
  void emitImplicitArrayElements(ir::ValueRef address, const Type* type,
                                 BracedInitListAST* ast,
                                 ir::ValueRef initializedCount,
                                 ir::ValueRef elementCount);
  void emitLocalVariableInit(VariableSymbol* var, ExpressionAST* initializer);
  void emitReferenceInit(Symbol* reference, ir::ValueRef address,
                         ExpressionAST* initExpr, SourceLocation loc,
                         ir::GlobalRef staticStorage = {});
  [[nodiscard]] auto emitsOnlyWhenReferenced(
      Symbol* variable, const std::optional<ConstValue>& constantValue) -> bool;
  [[nodiscard]] static auto referenceTemporaryLinkage(ir::Linkage ownerLinkage)
      -> ir::Linkage;
  [[nodiscard]] auto findOrCreateReferenceTemporary(
      Symbol* reference, const Type* temporaryType, ir::GlobalRef storage,
      std::size_t index, SourceLocation loc) -> ir::GlobalRef;
  void emitDesignatedInit(ir::ValueRef address, const Type* type,
                          DesignatedInitializerClauseAST* ast);
  [[nodiscard]] auto emitVectorInitializer(const Type* type,
                                           BracedInitListAST* ast)
      -> ir::ValueRef;
  void emitBitFieldInit(SourceLocation loc, ir::ValueRef address,
                        FieldSymbol* field, ExpressionAST* init);

 private:
  struct InlineAsmBuilder;
  struct BitFieldAccess {
    ir::ValueRef address;
    ClassLayout::MemberInfo info;
    FieldSymbol* field = nullptr;
    bool isVolatile = false;
  };
  struct DynamicLane {
    ir::ValueRef index;
    bool isSigned = false;
  };
  using VectorLanes = std::variant<DynamicLane, std::vector<std::int64_t>>;
  struct VectorElementAccess {
    ir::ValueRef vectorAddress;
    const Type* vectorType = nullptr;
    ir::Access access;
    VectorLanes lanes;
  };
  using NonAddressableAccess =
      std::variant<BitFieldAccess, VectorElementAccess>;
  [[nodiscard]] auto bitFieldAccess(ExpressionAST* expression)
      -> std::optional<BitFieldAccess>;
  [[nodiscard]] auto uncapturedStructuredBindingAlias(Symbol* symbol) const
      -> VariableSymbol*;
  [[nodiscard]] static auto bitfieldInfo(const ClassSymbol* owner,
                                         const ClassLayout::MemberInfo& info)
      -> ir::BitfieldInfo;
  struct VaArgBuilder;
  struct AtomicBuiltin;
  [[nodiscard]] auto emitAtomicBuiltin(CallExpressionAST* ast)
      -> ExpressionResult;
  [[nodiscard]] auto emitAtomicUpdate(
      SourceLocation loc, ir::AtomicOperation operation, const Type* objectType,
      ir::ValueRef address, ir::ValueRef operand,
      const ir::AtomicAccess& access, AtomicResult result) -> ir::ValueRef;
  [[nodiscard]] auto atomicUpdatedValue(SourceLocation loc,
                                        ir::AtomicOperation operation,
                                        const Type* objectType,
                                        ir::ValueRef old, ir::ValueRef operand)
      -> ir::ValueRef;
  [[nodiscard]] auto scaledAtomicOperand(SourceLocation loc,
                                         const Type* objectType,
                                         ir::ValueRef operand) -> ir::ValueRef;
  [[nodiscard]] auto builtinOperand(ExpressionAST* expression,
                                    ir::ValueRef value) -> ir::ValueRef;
  [[nodiscard]] auto emitPointerVaArg(SourceLocation loc, ir::ValueRef list,
                                      const Type* type) -> ExpressionResult;
  [[nodiscard]] auto emitAArch64VaArg(SourceLocation loc, ir::ValueRef list,
                                      const Type* type) -> ExpressionResult;
  [[nodiscard]] auto emitX86VaArg(SourceLocation loc, ir::ValueRef list,
                                  const Type* type) -> ExpressionResult;
  [[nodiscard]] auto emitTodoStmt(SourceLocation loc, std::string_view message)
      -> ir::ValueRef;

  [[nodiscard]] auto emitTodoExpr(SourceLocation loc, std::string_view message)
      -> ir::ValueRef;

  [[nodiscard]] auto convertType(const Type* type) -> ir::TypeRef;

  [[nodiscard]] auto convertBaseSubobjectType(ClassSymbol* classSymbol)
      -> ir::TypeRef;

  struct ClassMemberTypes {
    std::vector<ir::TypeRef> members;
    std::vector<std::uint32_t> padding;
    bool packed = false;
  };

  [[nodiscard]] auto buildClassMemberTypes(ClassSymbol* classSymbol,
                                           bool includeVirtualBases)
      -> ClassMemberTypes;

  [[nodiscard]] auto arrayType(ir::TypeRef elementType, std::uint64_t size)
      -> ir::TypeRef;

  [[nodiscard]] auto arraySize(ir::TypeRef type) const -> std::uint64_t;
  [[nodiscard]] auto vectorType(ir::TypeRef elementType,
                                std::uint64_t elementCount) -> ir::TypeRef;

  [[nodiscard]] auto declareClassType(std::string_view name, bool isUnion)
      -> ir::TypeRef;

  void defineClassType(ir::TypeRef classType,
                       std::span<const ir::TypeRef> members, bool isPacked,
                       std::uint64_t alignment,
                       std::span<const std::uint32_t> padding = {});

  [[nodiscard]] auto classMembers(ir::TypeRef classType) const
      -> std::span<const ir::TypeRef>;

  [[nodiscard]] auto isClassTypeDefined(ir::TypeRef classType) const -> bool;

  [[nodiscard]] auto isUnionClassType(ir::TypeRef classType) const -> bool;

  [[nodiscard]] auto className(ir::TypeRef classType) const -> std::string_view;

  [[nodiscard]] auto storageAlignment(ir::TypeRef type,
                                      std::uint64_t pointerSize)
      -> std::uint64_t;

  [[nodiscard]] auto bitFieldStorageType(std::uint64_t sizeInBytes)
      -> ir::TypeRef;

  [[nodiscard]] auto declareFunction(SourceLocation loc,
                                     const ir::FunctionInfo& info)
      -> ir::FunctionRef;

  [[nodiscard]] auto findFunction(std::string_view name) -> ir::FunctionRef;

  [[nodiscard]] auto functionName(ir::FunctionRef function) const
      -> std::string_view;

  [[nodiscard]] auto functionType(ir::FunctionRef function) const
      -> ir::TypeRef;

  [[nodiscard]] auto declareGlobal(SourceLocation loc,
                                   const ir::GlobalInfo& info) -> ir::GlobalRef;

  [[nodiscard]] auto findGlobal(std::string_view name) -> ir::GlobalRef;

  [[nodiscard]] auto globalName(ir::GlobalRef global) const -> std::string_view;

  [[nodiscard]] auto convertBaseEmbedding(ClassSymbol* baseSymbol,
                                          std::uint64_t availableBytes)
      -> ir::TypeRef;

  [[nodiscard]] auto currentBlockMightHaveTerminator() -> bool;

  [[nodiscard]] auto getAlignment(const Type* type) -> uint64_t;
  [[nodiscard]] auto getAlignment(VariableSymbol* var) -> uint64_t;
  [[nodiscard]] auto getAlignment(FieldSymbol* field) -> uint64_t;
  [[nodiscard]] auto variableAlignment(const Type* type, int explicitAlignment)
      -> uint64_t;
  [[nodiscard]] auto lvalueAlignment(ExpressionAST* expression) -> uint64_t;
  [[nodiscard]] auto lvalueAccess(ExpressionAST* expression) -> ir::Access;
  [[nodiscard]] auto objectAccess(const Type* type, std::uint64_t alignment)
      -> ir::Access;
  [[nodiscard]] auto objectVariable(Symbol* symbol) -> VariableSymbol*;
  [[nodiscard]] auto memberAlignment(MemberExpressionAST* member) -> uint64_t;
  [[nodiscard]] auto fieldAlignment(FieldSymbol* field,
                                    uint64_t objectAlignment) -> uint64_t;
  [[nodiscard]] auto elementAlignment(SubscriptExpressionAST* subscript)
      -> uint64_t;

  [[nodiscard]] auto pointerSize() const -> std::int64_t;

  [[nodiscard]] auto pointerSizedIntType() -> ir::TypeRef;

  [[nodiscard]] auto hasInternalLinkage(Symbol* symbol) const -> bool;
  [[nodiscard]] auto hasVagueFunctionEmission(FunctionSymbol* function) const
      -> bool;
  [[nodiscard]] auto hasVagueEmission(Symbol* symbol) const -> bool;
  [[nodiscard]] auto symbolLinkage(Symbol* symbol) const -> ir::Linkage;
  [[nodiscard]] static auto isMaterializedTemporary(Symbol* symbol) -> bool;
  struct ReferenceTemporary;
  void assignReferenceTemporaries(const ReferenceTemporary& owner,
                                  const ConstValue& value, std::size_t& count);
  void assignReferenceTemporaries(Symbol* owner, std::string_view ownerSuffix,
                                  ir::Linkage linkage,
                                  const std::optional<ConstValue>& value);
  [[nodiscard]] auto globalSymbolName(VariableSymbol* variable,
                                      std::string_view suffix) -> std::string;
  [[nodiscard]] auto globalLinkage(VariableSymbol* variable) const
      -> ir::Linkage;

  [[nodiscard]] auto findOrCreateBaseObjectStructor(
      FunctionSymbol* functionSymbol) -> ir::FunctionRef;

  [[nodiscard]] auto duplicatesBaseObjectStructorBody(
      FunctionSymbol* functionSymbol) -> bool;
  void emitBaseObjectStructor(FunctionSymbol* functionSymbol,
                              ir::FunctionRef completeObjectFunc);

  [[nodiscard]] auto baseObjectStructorName(FunctionSymbol* functionSymbol)
      -> std::optional<std::string>;

  [[nodiscard]] auto aliasNameOf(FunctionSymbol* emittedSymbol)
      -> std::optional<std::string>;

  auto findOrCreateSecondaryFunctionName(FunctionSymbol* functionSymbol,
                                         std::string_view name,
                                         std::string_view aliaseeName)
      -> ir::FunctionRef;

  [[nodiscard]] auto emittedFunctionSymbol(FunctionSymbol* functionSymbol)
      -> FunctionSymbol*;

  [[nodiscard]] auto findOrCreateFunction(FunctionSymbol* functionSymbol)
      -> ir::FunctionRef;

  [[nodiscard]] auto computeFunctionSignature(FunctionSymbol* functionSymbol)
      -> ir::TypeRef;

  [[nodiscard]] auto computeFunctionSignature(const FunctionType* functionType,
                                              FunctionSymbol* functionSymbol)
      -> ir::TypeRef;

  [[nodiscard]] auto structorReturnsThis(FunctionSymbol* symbol) -> bool;

  [[nodiscard]] auto classifyClassValueAbi(const Type* type,
                                           ClassValueAbiContext context)
      -> ClassValueAbi;

  [[nodiscard]] auto getSize(const Type* type) -> std::uint64_t;

  [[nodiscard]] auto abiSlotAddress(SourceLocation loc, ir::ValueRef address,
                                    const ClassValueAbiSlot& slot)
      -> ir::ValueRef;

  [[nodiscard]] auto abiCoerceStorage(SourceLocation loc, const Type* valueType,
                                      const ClassValueAbi& abi,
                                      ir::ValueRef address) -> ir::ValueRef;

  void abiLoadClassValue(SourceLocation loc, const Type* valueType,
                         const ClassValueAbi& abi, ir::ValueRef address,
                         std::vector<ir::ValueRef>& values);

  void abiStoreClassValue(SourceLocation loc, const Type* valueType,
                          const ClassValueAbi& abi,
                          std::span<const ir::ValueRef> values,
                          ir::ValueRef address);

  [[nodiscard]] auto hasNoValueRepresentation(const Type* type) -> bool;

  [[nodiscard]] auto classValueAddress(SourceLocation loc, const Type* type,
                                       ir::ValueRef value) -> ir::ValueRef;

  struct FunctionAbi {
    ir::TypeRef signature;
    std::vector<ir::ParameterAbi> parameters;
    std::vector<ClassValueAbi> arguments;
    ir::ParameterAbi resultAbi;
  };

  [[nodiscard]] auto scalarExtensionAbi(const Type* type) const
      -> ir::ParameterAbi;

  [[nodiscard]] auto computeFunctionAbi(const FunctionType* functionType,
                                        FunctionSymbol* functionSymbol)
      -> FunctionAbi;

  [[nodiscard]] auto variadicCalleeType(std::span<const ir::ValueRef> args,
                                        std::size_t ellipsisArgumentCount,
                                        std::span<const ir::TypeRef> results)
      -> ir::TypeRef;

  [[nodiscard]] auto classValueLoad(SourceLocation loc, const Type* type,
                                    ir::ValueRef value) -> ir::ValueRef;

  void abiLowerClassArgument(SourceLocation loc, const Type* paramType,
                             const ClassValueAbi& abi, ir::ValueRef value,
                             std::vector<ir::ValueRef>& args);

  void appendCallArgument(SourceLocation loc, const Type* paramType,
                          const ClassValueAbi& abi, ir::ValueRef value,
                          std::vector<ir::ValueRef>& args);

  struct CallArguments {
    std::vector<ir::ValueRef> values;
    std::vector<ir::ParameterAbi> parameters;
    std::size_t ellipsisCount = 0;
  };

  [[nodiscard]] auto lowerCallArguments(
      SourceLocation loc, const FunctionType* functionType,
      const FunctionAbi& functionAbi,
      std::span<const ExpressionResult> arguments,
      std::vector<ir::ValueRef> implicitArguments = {}) -> CallArguments;

  [[nodiscard]] auto abiPrepareResult(SourceLocation loc,
                                      const Type* returnType,
                                      std::vector<ir::TypeRef>& resultTypes,
                                      ir::ValueRef resultObject = {})
      -> ir::ValueRef;

  [[nodiscard]] auto abiFinishResult(SourceLocation loc, const Type* returnType,
                                     std::span<const ir::ValueRef> callResults,
                                     ir::ValueRef sretTemp) -> ExpressionResult;

  [[nodiscard]] auto emitVirtualBaseAddress(SourceLocation loc,
                                            ir::ValueRef objectPtr,
                                            ClassSymbol* fromClass,
                                            ClassSymbol* vbaseClass)
      -> ir::ValueRef;

  [[nodiscard]] auto adjustByVtableWord(SourceLocation loc,
                                        ir::ValueRef objectPtrI8,
                                        std::int64_t byteOffset)
      -> ir::ValueRef;

  [[nodiscard]] auto emitBaseClassAddress(SourceLocation loc,
                                          ir::ValueRef objectPtr,
                                          ClassSymbol* fromClass,
                                          ClassSymbol* targetClass)
      -> ir::ValueRef;

  [[nodiscard]] auto emitDerivedClassAddress(SourceLocation loc,
                                             ir::ValueRef objectPtr,
                                             ClassSymbol* fromClass,
                                             ClassSymbol* targetClass)
      -> ir::ValueRef;

  enum class ClassPointerConversion { kDerivedToBase, kBaseToDerived };

  [[nodiscard]] auto baseClassPathChangesAddress(ClassSymbol* derivedClass,
                                                 ClassSymbol* baseClass)
      -> bool;

  [[nodiscard]] auto emitNullPreservingClassAddress(
      SourceLocation loc, ir::ValueRef pointer, ClassSymbol* fromClass,
      ClassSymbol* targetClass, ClassPointerConversion conversion)
      -> ir::ValueRef;

  [[nodiscard]] auto emitMemberFunctionPointerValue(
      SourceLocation loc, const MemberFunctionPointerType* pointerType,
      FunctionSymbol* function, std::int64_t adjustmentBytes) -> ir::ValueRef;

  [[nodiscard]] auto makeMemberFunctionPointer(
      SourceLocation loc, const MemberFunctionPointerType* pointerType,
      ir::ValueRef pointerField, ir::ValueRef adjustmentField) -> ir::ValueRef;

  [[nodiscard]] auto memberFunctionPointerFields(
      SourceLocation loc, const MemberFunctionPointerType* pointerType,
      ir::ValueRef value) -> std::pair<ir::ValueRef, ir::ValueRef>;

  [[nodiscard]] auto navigateToClass(SourceLocation loc, ir::ValueRef value,
                                     ClassSymbol* from, ClassSymbol* to)
      -> ir::ValueRef;

  [[nodiscard]] auto subobjectAddress(SourceLocation loc,
                                      ir::ValueRef objectPtr,
                                      ClassSymbol* subobjectClass,
                                      std::uint64_t byteOffset) -> ir::ValueRef;

  [[nodiscard]] auto baseSubobjectAddress(SourceLocation loc,
                                          ir::ValueRef objectPtr,
                                          ClassSymbol* derivedClass,
                                          ClassSymbol* baseClass)
      -> ir::ValueRef;

  [[nodiscard]] auto classBodyElement(ClassSymbol* classSymbol,
                                      std::uint32_t index) -> ir::TypeRef;

  [[nodiscard]] auto memberAddress(SourceLocation loc, ir::ValueRef objectPtr,
                                   const Type* memberType, std::uint32_t index)
      -> ir::ValueRef;
  [[nodiscard]] auto memberAddress(SourceLocation loc, ir::ValueRef objectPtr,
                                   ir::TypeRef memberType, std::uint32_t index)
      -> ir::ValueRef;

  [[nodiscard]] auto loadThisPointer(SourceLocation loc,
                                     ClassSymbol* classSymbol) -> ir::ValueRef;

  [[nodiscard]] auto loadEnclosingObject(SourceLocation loc,
                                         ClassSymbol* targetClass,
                                         ClassSymbol*& objectClass)
      -> ir::ValueRef;

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

  [[nodiscard]] auto subobjectAddress(SourceLocation loc,
                                      ir::ValueRef objectPtr,
                                      ClassSymbol* classSymbol,
                                      Symbol* subobject) -> ir::ValueRef;

  void forEachSubobjectElement(SourceLocation loc, ir::ValueRef subobjectPtr,
                               const ClassSubobjectShape& shape, bool reverse,
                               const std::function<void(ir::ValueRef)>& body);

  [[nodiscard]] auto subobjectsInDeclarationOrder(
      ClassSymbol* classSymbol) const -> std::vector<Symbol*>;

  [[nodiscard]] auto isImplicitlyInitializedSubobject(ClassSymbol* classSymbol,
                                                      Symbol* subobject) const
      -> bool;

  [[nodiscard]] auto defaultConstructorArguments(FunctionSymbol* constructor)
      -> std::vector<ExpressionResult>;

  void emitSubobjectDefaultConstruction(SourceLocation loc,
                                        ir::ValueRef objectPtr,
                                        ClassSymbol* classSymbol,
                                        Symbol* subobject);

  void enqueueFunctionBody(FunctionSymbol* symbol);
  void processPendingFunctions();

  [[nodiscard]] auto findOrCreateGlobal(Symbol* symbol)
      -> std::optional<ir::GlobalRef>;

  [[nodiscard]] auto staticStorageAddress(SourceLocation loc, Symbol* symbol)
      -> ir::ValueRef;
  [[nodiscard]] auto findOrCreateStaticField(FieldSymbol* field)
      -> ir::GlobalRef;

  void emitGlobalVarInit(VariableSymbol* var, ir::GlobalRef global);
  [[nodiscard]] auto globalObjectAddress(SourceLocation loc, Symbol* symbol,
                                         ir::GlobalRef global) -> ir::ValueRef;
  [[nodiscard]] auto isNonLocalThreadLocal(Symbol* symbol) const -> bool;
  [[nodiscard]] auto threadLocalCallingConvention() const
      -> ir::CallingConvention;
  [[nodiscard]] auto findOrCreateThreadLocalWrapper(Symbol* symbol,
                                                    ir::GlobalRef global)
      -> ir::FunctionRef;
  [[nodiscard]] auto findOrCreateThreadLocalInitializer() -> ir::FunctionRef;
  void emitThreadLocalInitializers();
  void emitStaticLocalVarInit(VariableSymbol* var, ir::GlobalRef global,
                              ExpressionAST* initializer);

  [[nodiscard]] auto findOrCreateGuardVariable(Symbol* symbol,
                                               ir::GlobalRef global,
                                               ir::Linkage linkage,
                                               SourceLocation loc)
      -> ir::GlobalRef;

  void emitGlobalInit(Symbol* symbol, const Type* type,
                      ExpressionAST* initializer, FunctionSymbol* constructor,
                      FunctionSymbol* destructor, ir::GlobalRef global,
                      bool guarded);
  void emitGlobalObjectInitialization(Symbol* symbol, const Type* type,
                                      ExpressionAST* initializer,
                                      FunctionSymbol* constructor,
                                      FunctionSymbol* destructor,
                                      ir::GlobalRef global, SourceLocation loc);

  [[nodiscard]] auto constructorArgumentList(BracedInitListAST* bracedInitList)
      -> List<ExpressionAST*>*;

  [[nodiscard]] auto constructorArguments(ExpressionAST* initializer)
      -> std::vector<ExpressionResult>;
  [[nodiscard]] auto initializerExpression(ExpressionAST* initializer)
      -> ExpressionAST*;

  void generateVTable(ClassSymbol* classSymbol);
  struct VTableEmission;

  [[nodiscard]] auto requiresVTT(ClassSymbol* classSymbol) const -> bool;
  [[nodiscard]] auto takesVTTParameter(FunctionSymbol* function) const -> bool;
  [[nodiscard]] auto vttParameter(FunctionSymbol* function) -> ir::ValueRef;
  [[nodiscard]] auto constructionVTableName(ClassSymbol* completeClass,
                                            const VTableLayout::Group& group)
      -> std::string;
  void generateVTT(ClassSymbol* completeClass, const VTableEmission& emission);
  [[nodiscard]] auto vttAddress(SourceLocation loc, ClassSymbol* completeClass,
                                std::size_t index) -> ir::ValueRef;

  struct VTableEmission {
    bool emitDefinition = true;
    ir::Linkage linkage = ir::Linkage::LinkOnceODR;
  };

  [[nodiscard]] auto vtableEmission(ClassSymbol* classSymbol) -> VTableEmission;

  void declareExternalVTable(SourceLocation loc, std::string_view name,
                             std::size_t wordCount);

  [[nodiscard]] auto findOrCreateTypeInfo(const Type* type) -> std::string;

  [[nodiscard]] auto findOrCreateTypeInfoName(const Type* type) -> std::string;

  [[nodiscard]] auto typeInfoHasIncompleteClass(const Type* type) -> bool;

  [[nodiscard]] auto typeInfoHasInternalLinkage(const Type* type) const -> bool;

  [[nodiscard]] auto typeInfoEmission(const Type* type) -> VTableEmission;

  [[nodiscard]] auto findOrCreateAbiTypeInfoVTable(
      std::string_view abiClassName) -> ir::GlobalRef;

  [[nodiscard]] auto emitTypeInfoObject(
      SourceLocation loc, std::string_view name, std::string_view abiClassName,
      std::string_view typeInfoNameSymbol, ir::Linkage linkage,
      const std::function<void(std::vector<ir::TypeRef>& fieldTypes,
                               std::vector<ir::ValueRef>& fields)>&
          emitTrailingFields) -> ir::GlobalRef;

  struct TypeInfoBaseDescriptor {
    std::string typeInfo;
    std::int64_t offsetFlags = 0;
  };

  [[nodiscard]] auto classTypeInfoBaseDescriptors(ClassSymbol* classSymbol)
      -> std::vector<TypeInfoBaseDescriptor>;

  void emitClassTypeInfoBases(
      ClassSymbol* classSymbol,
      const std::vector<TypeInfoBaseDescriptor>& descriptors,
      std::vector<ir::TypeRef>& fieldTypes, std::vector<ir::ValueRef>& fields,
      SourceLocation loc);

  [[nodiscard]] auto virtualBaseOffsetSlotOffset(ClassSymbol* classSymbol,
                                                 ClassSymbol* virtualBase)
      -> std::optional<std::int64_t>;

  [[nodiscard]] auto typeInfoAddress(SourceLocation loc, const Type* type)
      -> ir::ValueRef;

  [[nodiscard]] auto findOrCreateRuntimeFunction(
      SourceLocation loc, std::string_view name,
      std::vector<ir::TypeRef> parameters, std::vector<ir::TypeRef> results)
      -> ir::FunctionRef;

  [[nodiscard]] auto exceptionObjectDestructor(SourceLocation loc,
                                               const Type* exceptionType)
      -> ir::ValueRef;

  void emitThrow(ThrowExpressionAST* ast);
  void emitTry(const std::function<void()>& body, List<HandlerAST*>* handlers,
               SourceLocation loc, bool rethrowOnFallthrough = false);
  [[nodiscard]] auto emitCallWithUnwind(SourceLocation loc, ir::CallInfo info)
      -> std::vector<ir::ValueRef>;
  void ensureExceptionPersonality(SourceLocation loc);
  void emitTerminateOnException(const std::function<void()>& body,
                                SourceLocation loc);
  struct ExceptionScope {
    List<HandlerAST*>* handlers;
    ir::BlockRef dispatch;
    ir::ValueRef exception;
    ir::ValueRef selector;
    std::size_t cleanupDepth;
    bool terminates = false;
  };
  std::vector<ExceptionScope> exceptionScopes_;

  [[nodiscard]] auto findOrCreateNoreturnRuntimeCall(SourceLocation loc,
                                                     std::string_view name)
      -> ir::FunctionRef;

  [[nodiscard]] auto findOrCreateDynamicCast(SourceLocation loc)
      -> ir::FunctionRef;

  [[nodiscard]] auto dynamicCastOffsetHint(ClassSymbol* sourceClass,
                                           ClassSymbol* targetClass)
      -> std::int64_t;

  [[nodiscard]] auto emitPointerIsNull(SourceLocation loc, ir::ValueRef pointer)
      -> ir::ValueRef;

  [[nodiscard]] auto dynamicCastNeedsRuntimeCheck(CppCastExpressionAST* ast)
      -> bool;

  [[nodiscard]] auto emitDynamicCast(CppCastExpressionAST* ast) -> ir::ValueRef;

  [[nodiscard]] auto emitDynamicCastAddress(SourceLocation loc,
                                            const Type* sourceType,
                                            const Type* targetType,
                                            ir::ValueRef sourceAddress)
      -> ir::ValueRef;

  [[nodiscard]] auto emitTypeidOfPolymorphicGlvalue(SourceLocation loc,
                                                    ir::ValueRef objectPtr)
      -> ir::ValueRef;

  [[nodiscard]] auto emitTypeid(TypeidExpressionAST* ast) -> ir::ValueRef;

  [[nodiscard]] auto findOrCreateUnimplementedVirtual(SourceLocation loc,
                                                      std::string_view name)
      -> ir::FunctionRef;

  [[nodiscard]] auto findOrCreateCxaAtexit(SourceLocation loc,
                                           bool threadLocal = false)
      -> ir::FunctionRef;

  [[nodiscard]] auto findOrCreateCxaGuardFunction(SourceLocation loc,
                                                  bool acquire)
      -> ir::FunctionRef;

  [[nodiscard]] auto findOrCreateDsoHandle(SourceLocation loc) -> ir::GlobalRef;

  void emitStaticStorageInitializer(Symbol* symbol, const Type* type,
                                    ExpressionAST* initializer,
                                    FunctionSymbol* constructor,
                                    ir::GlobalRef global, ir::ValueRef address,
                                    SourceLocation initLoc);
  void emitGlobalVarDtorRegistration(Symbol* symbol, const Type* type,
                                     FunctionSymbol* dtor, ir::GlobalRef global,
                                     SourceLocation loc);

  void emitCtorVtableInit(FunctionSymbol* functionSymbol, SourceLocation loc);

  [[nodiscard]] auto vtableSlotIndex(FunctionSymbol* function) -> int;

  void emitVTableOp(SourceLocation loc, std::string_view name,
                    ClassSymbol* classSymbol, const VTableLayout::Group& group,
                    ir::Linkage linkage);

  void emitVTableGroup(SourceLocation loc, std::string_view name,
                       ClassSymbol* classSymbol,
                       const VTableLayout::Group& group,
                       const VTableEmission& emission);

  [[nodiscard]] auto vtableEntryTarget(const VTableLayout::Slot& slot)
      -> FunctionSymbol*;

  [[nodiscard]] auto vtableEntry(SourceLocation loc,
                                 const VTableLayout::Slot& slot)
      -> ir::FunctionRef;

  [[nodiscard]] auto definesFunctionBody(FunctionSymbol* function) -> bool;

  [[nodiscard]] auto applyCallOffset(SourceLocation loc, ir::ValueRef pointerI8,
                                     const VTableLayout::CallOffset& callOffset,
                                     bool virtualFirst) -> ir::ValueRef;

  void emitForwardingBody(ir::FunctionRef function, FunctionSymbol* target,
                          ir::FunctionRef targetFuncOp,
                          const VTableLayout::CallOffset& thisAdjustment,
                          const VTableLayout::CallOffset& returnAdjustment);

  [[nodiscard]] auto findOrCreateThunk(
      FunctionSymbol* target, const VTableLayout::CallOffset& thisAdjustment,
      const VTableLayout::CallOffset& returnAdjustment) -> ir::FunctionRef;

  void emitAdjustingEntryPoints(FunctionSymbol* function);

  [[nodiscard]] auto classRequiringVTable(FunctionSymbol* function)
      -> ClassSymbol*;

  [[nodiscard]] auto tableSubobjectAddress(SourceLocation loc,
                                           ir::ValueRef thisPtr,
                                           ClassSymbol* classSymbol,
                                           const VTableLayout::Table& table,
                                           bool usesVTT) -> ir::ValueRef;

  [[nodiscard]] auto offsetInVirtualBase(ClassSymbol* classSymbol,
                                         const VTableLayout::Table& table)
      -> std::uint64_t;

  [[nodiscard]] auto tableVPtrAlignment(ClassSymbol* classSymbol,
                                        const VTableLayout::Table& table,
                                        bool usesVTT) -> std::uint64_t;

  [[nodiscard]] auto vptrAddress(SourceLocation loc, ir::ValueRef objectPtr)
      -> ir::ValueRef;

  [[nodiscard]] auto newTemp(const Type* type, SourceLocation loc)
      -> ir::ValueRef;

  [[nodiscard]] auto findOrCreateLocal(Symbol* symbol)
      -> std::optional<ir::ValueRef>;

  [[nodiscard]] auto implicitLocation(SourceLocation loc) const
      -> SourceLocation;

  [[nodiscard]] auto emitCall(
      SourceLocation loc, FunctionSymbol* symbol, ExpressionResult thisValue,
      std::vector<ExpressionResult> arguments, bool isVirtualDispatch = false,
      ExpressionAST* resultOwner = nullptr, bool baseObjectStructor = false,
      ir::ValueRef structorVTT = {}) -> ExpressionResult;

  [[nodiscard]] auto baseStructorVTTArgument(SourceLocation loc,
                                             ClassSymbol* targetClass,
                                             bool virtualBase = false)
      -> ir::ValueRef;

  [[nodiscard]] auto emitCall(
      SourceLocation loc, const FunctionType* functionType,
      FunctionSymbol* symbol, bool isVirtualDispatch,
      ExpressionResult thisValue, std::vector<ExpressionResult> arguments,
      ir::ValueRef resultObject = {}, ir::ValueRef calleeValue = {},
      bool baseObjectStructor = false, ir::ValueRef structorVTT = {})
      -> ExpressionResult;

  [[nodiscard]] auto uniqueClassTypeName(std::string name) -> std::string;

  [[nodiscard]] auto loadReferenceBinding(SourceLocation loc, const Type* type,
                                          ir::ValueRef value) -> ir::ValueRef;

  [[nodiscard]] auto requiresZeroInitialization(const Type* type,
                                                FunctionSymbol* constructor)
      -> bool;

  void emitZeroInitialization(SourceLocation loc, ir::ValueRef address,
                              const Type* type);

  [[nodiscard]] auto arrayCookieSize() -> std::uint64_t;

  [[nodiscard]] auto arrayCookiePrefixSize(const Type* elementType,
                                           bool cookieRequired)
      -> std::uint64_t;

  [[nodiscard]] auto arrayCookieCountOffset(std::uint64_t prefixSize) const
      -> std::uint64_t;

  [[nodiscard]] auto arrayElementCount(SourceLocation loc,
                                       const Type* allocatedType,
                                       ir::TypeRef countType) -> ir::ValueRef;

  void emitArrayLoop(SourceLocation loc, ir::ValueRef base,
                     const Type* elementType, ir::ValueRef count, bool reverse,
                     const std::function<void(ir::ValueRef)>& body);

  void emitIndexLoop(SourceLocation loc, ir::ValueRef count, bool reverse,
                     const std::function<void(ir::ValueRef)>& body);

  void emitFieldInitializer(SourceLocation sourceLoc, FieldSymbol* field,
                            ir::ValueRef fieldPtr, ExpressionAST* initializer);

  void emitCaptureInit(ClassSymbol* classSymbol, ir::ValueRef closure,
                       LambdaCaptureAST* capture, ExpressionAST* initializer);

  [[nodiscard]] auto emitCtorCall(SourceLocation loc, FunctionSymbol* ctor,
                                  ir::ValueRef thisPtr,
                                  std::vector<ExpressionResult> args,
                                  bool completeObject, ir::ValueRef vtt = {})
      -> ExpressionResult;

  [[nodiscard]] static auto completeObjectDtor(FunctionSymbol* dtor)
      -> FunctionSymbol*;

  [[nodiscard]] auto newBlock() -> ir::BlockRef;

  [[nodiscard]] auto newUniqueSymbolName(std::string_view prefix)
      -> std::string;

  [[nodiscard]] auto constantStringRecord(SourceLocation loc,
                                          const StringLiteral* literal)
      -> std::string;

  [[nodiscard]] auto constantStringData(SourceLocation loc,
                                        const StringLiteral* literal,
                                        bool isUtf16) -> std::string;

  [[nodiscard]] auto makeFloatInitializer(const Type* type, double value)
      -> ir::Initializer;

  [[nodiscard]] auto getFloatAttr(const std::optional<ConstValue>& value,
                                  const Type* type)
      -> std::optional<ir::Initializer>;

  struct ClassConstantSlot {
    ConstValue value;
    const Type* type = nullptr;
    bool isBaseSubobject = false;
  };

  struct BitFieldStorageSlot {
    std::vector<std::uint8_t> bytes;
  };

  using ConstantSlotValue =
      std::variant<ClassConstantSlot, BitFieldStorageSlot>;
  using ConstantSlot = std::optional<ConstantSlotValue>;

  [[nodiscard]] auto bitFieldStorageInitializer(
      std::span<const std::uint8_t> storage) -> ir::Initializer;

  [[nodiscard]] auto emitBitFieldStorageValue(
      SourceLocation loc, std::span<const std::uint8_t> storage)
      -> ir::ValueRef;

  [[nodiscard]] auto emitConstantSlotValue(SourceLocation loc,
                                           const ConstantSlotValue& slot,
                                           ir::TypeRef layoutType = {})
      -> ir::ValueRef;

  [[nodiscard]] auto constantSlotInitializer(const ConstantSlotValue& slot,
                                             ir::TypeRef layoutType = {})
      -> std::optional<ir::Initializer>;

  void completeUnionStorage(ClassMemberTypes& storage, ClassSymbol* symbol,
                            std::uint64_t payloadSize);

  struct ConstantStorageTypeVisitor;

  [[nodiscard]] auto constantStorageType(const ConstValue& value,
                                         const Type* type,
                                         ir::TypeRef layoutType = {})
      -> ir::TypeRef;

  [[nodiscard]] auto classConstantSlots(const ConstValue& value,
                                        const ClassType* classType)
      -> std::optional<std::vector<ConstantSlot>>;

  [[nodiscard]] auto emitConstantVptrs(SourceLocation loc, ir::ValueRef value,
                                       ClassSymbol* classSymbol,
                                       ClassSymbol* completeClass,
                                       std::uint64_t offset = 0)
      -> ir::ValueRef;

  [[nodiscard]] auto vectorInitializer(const ConstValue& value,
                                       const VectorType* vector)
      -> std::optional<ir::Initializer>;
  [[nodiscard]] auto arrayInitializer(const InitializerList& list,
                                      ir::TypeRef storage)
      -> std::optional<ir::Initializer>;

  [[nodiscard]] auto constValueToInitializer(const ConstValue& value,
                                             const Type* type,
                                             ir::TypeRef layoutType = {})
      -> std::optional<ir::Initializer>;

  [[nodiscard]] auto pointerCast(SourceLocation loc, ir::TypeRef pointerType,
                                 ir::ValueRef address) -> ir::ValueRef;

  [[nodiscard]] auto displacedAddress(SourceLocation loc,
                                      ir::TypeRef pointerType,
                                      ir::ValueRef address, std::intmax_t bytes)
      -> ir::ValueRef;

  [[nodiscard]] auto stringLiteralInitializer(const StringLiteral* literal,
                                              const Type* arrayType)
      -> std::optional<ir::Initializer>;

  [[nodiscard]] auto emitConstantAggregate(SourceLocation loc,
                                           ir::TypeRef irType, const Type* type,
                                           const ConstValue& value)
      -> ir::ValueRef;

  [[nodiscard]] auto emitStringLiteralAddress(SourceLocation loc,
                                              ir::TypeRef pointerType,
                                              const StringLiteral* literal)
      -> ir::ValueRef;
  [[nodiscard]] auto emitConstInitValue(SourceLocation loc, const Type* type,
                                        const ConstValue& value,
                                        ir::TypeRef layoutType = {})
      -> ir::ValueRef;

  void branch(SourceLocation loc, ir::BlockRef block,
              std::vector<ir::ValueRef> operands = {});

  struct Loop {
    ir::BlockRef continueBlock;
    ir::BlockRef breakBlock;
    std::size_t continueCleanupDepth = 0;
    std::size_t breakCleanupDepth = 0;
  };

  struct CleanupScope {
    std::uint64_t sourceScope = 0;
    struct Entry {
      ir::ValueRef address;
      ir::FunctionRef destructor;
      ir::ValueRef activeFlag;
      std::int64_t elementCount = 1;
      ir::ValueRef initializedCount;
      std::vector<ir::ValueRef> additionalArguments;
      std::vector<ir::ParameterAbi> parameters;
      ir::TypeRef variadicCalleeType;
      bool exceptionalOnly = false;
    };
    std::vector<Entry> entries;
    bool isFullExpression = false;
    ir::CleanupRegionRef region;
  };

  using CleanupSnapshot = std::vector<CleanupScope::Entry>;

  struct CleanupLabel {
    std::string_view name;
    ir::BlockRef block;
    CleanupSnapshot cleanups;
  };

  struct CleanupGoto {
    SourceLocation location;
    std::string_view label;
    ir::BlockRef block;
    std::vector<CleanupScope> cleanups;
    std::vector<ExceptionScope> exceptions;
    std::uint64_t sourceContext = 0;
  };

  void defineCleanupLabel(SourceLocation loc, std::string_view name);
  void emitGotoLabel(SourceLocation loc, std::string_view name);
  void resolveCleanupGotos();

  void pushCleanup();
  void pushFullExpressionCleanup();
  void popCleanup(SourceLocation loc);
  void emitBranchWithCleanups(SourceLocation loc, ir::BlockRef target,
                              std::size_t targetDepth);
  void emitCleanupSnapshot(SourceLocation loc, const CleanupSnapshot& snapshot,
                           bool exceptional);
  void emitCleanupCall(SourceLocation loc, const CleanupScope::Entry& cleanup,
                       bool exceptional);
  void emitCleanupDestructorCall(SourceLocation loc,
                                 const CleanupScope::Entry& cleanup,
                                 ir::ValueRef address);
  void removeCleanup(ir::ValueRef address, ir::FunctionRef destructor);
  void addCleanup(ir::ValueRef address, FunctionSymbol* dtor,
                  std::int64_t elementCount = 1, bool exceptionalOnly = false);
  void addArrayCleanup(ir::ValueRef address, const Type* arrayType);
  void addSubobjectCleanup(SourceLocation loc, ir::ValueRef address,
                           const Type* type, bool completeObject,
                           bool exceptionalOnly = true,
                           bool virtualBase = false);
  void addSubobjectCleanup(SourceLocation loc, ir::ValueRef address,
                           const ClassSubobjectShape& shape,
                           bool completeObject, bool exceptionalOnly = true,
                           bool virtualBase = false);
  [[nodiscard]] auto subobjectCleanup(SourceLocation loc, ir::ValueRef address,
                                      const ClassSubobjectShape& shape,
                                      bool completeObject,
                                      bool virtualBase = false)
      -> std::optional<CleanupScope::Entry>;
  void addDestructorSubobjectCleanups(SourceLocation loc,
                                      FunctionSymbol* function);
  void emitSubobjectConstruction(SourceLocation loc, ir::ValueRef address,
                                 const ClassSubobjectShape& shape,
                                 bool completeObject,
                                 const std::function<void(ir::ValueRef)>& body);
  [[nodiscard]] auto beginArrayConstruction(SourceLocation loc,
                                            ir::ValueRef address,
                                            const Type* type) -> ir::ValueRef;
  [[nodiscard]] auto beginConstructionRange(SourceLocation loc,
                                            ir::ValueRef address,
                                            const ClassSubobjectShape& shape,
                                            bool completeObject)
      -> ir::ValueRef;
  void advanceArrayConstruction(SourceLocation loc, ir::ValueRef count,
                                const Type* type);
  [[nodiscard]] auto objectDestructor(const Type* type) const
      -> FunctionSymbol*;
  void addTemporaryCleanup(ir::ValueRef address, const Type* type);

  void cancelCleanup(ir::ValueRef address);

  class LifetimeExtension {
   public:
    LifetimeExtension(Codegen& gen, std::span<ExpressionAST* const> temporaries,
                      Symbol* reference = nullptr,
                      ir::GlobalRef staticStorage = {});
    ~LifetimeExtension();

    void extend(ExpressionAST* expression, ir::ValueRef address);
    [[nodiscard]] auto extends(ExpressionAST* expression) const -> bool;
    [[nodiscard]] auto extendsStatically(ExpressionAST* expression) const
        -> bool;
    [[nodiscard]] auto temporaryStorage(ExpressionAST* temporary,
                                        const Type* type, SourceLocation loc)
        -> ir::ValueRef;
    void extendObject(ir::ValueRef address, const Type* type,
                      SourceLocation loc);

   private:
    void relocateCleanup(ir::ValueRef address);

    Codegen& gen_;
    std::span<ExpressionAST* const> temporaries_;
    Symbol* reference_ = nullptr;
    ir::GlobalRef staticStorage_;
    std::size_t staticCount_ = 0;
    std::vector<std::pair<ir::ValueRef, ir::GlobalRef>> staticTemporaries_;
    std::optional<std::size_t> targetScope_;
    LifetimeExtension* previous_;
  };

  class FullExpression {
   public:
    FullExpression(Codegen& gen, SourceLocation endLoc);
    ~FullExpression();

   private:
    Codegen& gen_;
    SourceLocation endLoc_;
  };

  class CleanupScopeGuard {
   public:
    CleanupScopeGuard(Codegen& gen, SourceLocation endLoc)
        : gen_(gen), endLoc_(endLoc) {
      gen_.pushCleanup();
    }
    ~CleanupScopeGuard() { gen_.popCleanup(endLoc_); }

   private:
    Codegen& gen_;
    SourceLocation endLoc_;
  };

  class DefaultInitializerObjectGuard {
   public:
    DefaultInitializerObjectGuard(Codegen& gen, ir::ValueRef object)
        : gen_(gen), object_(object) {
      std::swap(gen_.defaultInitializerObject_, object_);
    }

    ~DefaultInitializerObjectGuard() {
      std::swap(gen_.defaultInitializerObject_, object_);
    }

   private:
    Codegen& gen_;
    ir::ValueRef object_;
  };

  class ThisValueGuard {
   public:
    ThisValueGuard(Codegen& gen, ir::ValueRef thisValue)
        : gen_(gen), thisValue_(thisValue) {
      std::swap(gen_.thisValue_, thisValue_);
    }

    ~ThisValueGuard() { std::swap(gen_.thisValue_, thisValue_); }

   private:
    Codegen& gen_;
    ir::ValueRef thisValue_;
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

  [[nodiscard]] auto takeResultObject(ExpressionAST* ast) -> ir::ValueRef;

  [[nodiscard]] auto emitIntoResultObject(ExpressionAST* owner,
                                          ExpressionAST* operand)
      -> ir::ValueRef;
  [[nodiscard]] auto emitAggregateObject(ExpressionAST* owner, const Type* type,
                                         BracedInitListAST* initializer,
                                         SourceLocation loc) -> ir::ValueRef;
  [[nodiscard]] auto takeIndirectResultObject(ExpressionAST* ast,
                                              const FunctionType* functionType)
      -> ir::ValueRef;

  auto emitPrvalueInto(ir::ValueRef object, const Type* objectType,
                       ExpressionAST* ast, SourceLocation loc) -> bool;

  class ResultObject {
   public:
    ResultObject(Codegen& gen, ExpressionAST* ast, ir::ValueRef address);
    ~ResultObject();

    [[nodiscard]] auto wasConsumed() const -> bool;

   private:
    Codegen& gen_;
    ExpressionAST* savedOwner_;
    ir::ValueRef savedAddress_;
    bool savedInitialized_;
  };

  [[nodiscard]] auto collectCleanupSnapshot(std::size_t targetDepth = 0,
                                            bool exceptional = false)
      -> CleanupSnapshot;

  struct Switch {
    struct Range {
      ConstInt lower;
      ConstInt upper;
      ir::BlockRef destination;
    };
    const Type* conditionType = nullptr;
    std::vector<Range> ranges;
    std::vector<ConstInt> caseValues;
    std::vector<ir::BlockRef> caseDestinations;
    ir::BlockRef defaultDestination;
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
  struct DebugScopeVisitor;

  [[nodiscard]] auto debugEmitter() -> ir::DebugEmitter*;
  [[nodiscard]] auto debugLocation(SourceLocation loc) -> ir::DebugLocation;
  [[nodiscard]] auto convertDebugType(const Type* type) -> ir::DebugTypeRef;
  [[nodiscard]] auto getOrCreateDebugScope(Symbol* symbol) -> ir::DebugScopeRef;
  [[nodiscard]] auto debugSubroutineType(const FunctionType* type,
                                         const Type* objectType = nullptr)
      -> ir::DebugTypeRef;
  struct TypeInfoIncompleteClassVisitor;
  struct TypeInfoInternalLinkageVisitor;
  struct ConstructorArgumentsVisitor;
  struct InitializerExpressionVisitor;

  void attachDebugInfo(ir::ValueRef address, Symbol* symbol,
                       std::string_view name = {}, unsigned arg = 0);

  void attachDebugInfo(ir::ValueRef address, const Type* type,
                       std::string_view name, unsigned arg);

  void buildSubprogramAttr(FunctionSymbol* functionSymbol,
                           FunctionDefinitionAST* ast, ir::FunctionRef func,
                           SourceLocation loc);

  ir::DebugEmitter* debugEmitter_ = nullptr;
  ir::DebugScopeRef debugCompileUnit_;
  std::unordered_map<const Type*, ir::DebugTypeRef> debugTypeCache_;
  std::unordered_map<Symbol*, ir::DebugScopeRef> debugScopes_;
  ir::Emitter& emitter_;
  ir::FunctionRef function_;
  TranslationUnit* unit_ = nullptr;
  TypeTraits traits;
  ir::BlockRef entryBlock_;
  ir::BlockRef exitBlock_;
  ir::ValueRef exitValue_;
  const Type* returnType_ = nullptr;
  ir::ValueRef thisValue_;
  ir::ValueRef defaultInitializerObject_;
  ir::ValueRef structorVTTValue_;
  ir::ValueRef targetValue_;
  ExpressionAST* targetExpression_ = nullptr;
  FunctionSymbol* currentFunctionSymbol_ = nullptr;
  struct ClassTypeInfo {
    std::string name;
    std::vector<ir::TypeRef> members;
    std::vector<std::uint32_t> padding;
    bool isUnion = false;
    bool isPacked = false;
    bool isDefined = false;
    std::uint64_t alignment = 0;
  };

  [[nodiscard]] auto naturalStorageAlignment(const ClassTypeInfo& info,
                                             std::uint64_t pointerSize)
      -> std::uint64_t;

  std::unordered_map<ir::TypeRef, ClassTypeInfo> classTypes_;
  std::unordered_map<ir::FunctionRef, std::string> functionNames_;
  std::unordered_map<ir::FunctionRef, ir::TypeRef> functionTypes_;
  std::unordered_map<ir::GlobalRef, std::string> globalNames_;
  std::unordered_map<ir::TypeRef, std::uint64_t> arraySizes_;
  std::unordered_map<ir::TypeRef, std::uint64_t> vectorSizes_;
  std::unordered_map<ClassSymbol*, ir::TypeRef> classNames_;
  std::unordered_map<ClassSymbol*, ir::TypeRef> baseSubobjectTypeNames_;
  std::unordered_set<std::string> classTypeNames_;
  std::unordered_map<Symbol*, ir::ValueRef> locals_;
  std::unordered_map<FunctionSymbol*, ir::FunctionRef> funcOps_;
  std::vector<FunctionSymbol*> pendingFunctions_;
  std::unordered_set<FunctionSymbol*> enqueuedFunctions_;
  std::unordered_set<ClassSymbol*> emittedVTables_;
  std::unordered_set<std::string> emittedTypeInfos_;
  std::unordered_map<VariableSymbol*, ir::GlobalRef> globalOps_;
  std::unordered_map<FieldSymbol*, ir::GlobalRef> staticFieldGlobalOps_;
  std::unordered_set<Symbol*> emittedGlobalInits_;
  std::unordered_map<Symbol*, ir::FunctionRef> threadLocalWrappers_;
  std::unordered_map<Symbol*, ir::FunctionRef> threadLocalInitFunctions_;
  std::vector<ir::FunctionRef> threadLocalInitializers_;
  ir::FunctionRef threadLocalInitializer_;
  std::unordered_map<std::string_view, int> uniqueSymbolNames_;
  std::unordered_map<const StringLiteral*, std::string> stringLiterals_;
  std::unordered_map<std::string_view, std::string> constantStrings_;
  Loop loop_;
  Switch switch_;
  std::vector<CleanupScope> cleanupStack_;
  LifetimeExtension* lifetimeExtension_ = nullptr;
  std::vector<CleanupLabel> cleanupLabels_;
  std::vector<CleanupGoto> cleanupGotos_;
  ExpressionAST* resultObjectOwner_ = nullptr;
  bool resultObjectInitialized_ = false;
  int conditionalEvaluationDepth_ = 0;
  ir::ValueRef resultObjectAddress_;
  int count_ = 0;
  int globalVarInitCount_ = 0;
  int globalVarDtorCount_ = 0;
  std::unordered_map<const Name*, int> staticLocalCounts_;
  struct ReferenceTemporary {
    Symbol* owner = nullptr;
    std::string ownerSuffix;
    std::size_t index = 0;
    ir::Linkage linkage = ir::Linkage::Internal;
  };
  std::unordered_map<Symbol*, ReferenceTemporary> referenceTemporaries_;
  Options options_;
  bool debugInfo_ = true;
  bool isWasmTarget_ = false;
};
}  // namespace cxx
