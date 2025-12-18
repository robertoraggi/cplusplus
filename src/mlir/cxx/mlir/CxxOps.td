// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Cxx_Dialect : Dialect {
  let name = "cxx";
  let cppNamespace = "mlir::cxx";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["mlir::cf::ControlFlowDialect", "mlir::DLTIDialect", "mlir::LLVM::LLVMDialect"];
}

class Cxx_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Cxx_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Cxx_Op<string mnemonic, list<Trait> traits = []>
    : Op<Cxx_Dialect, mnemonic, traits> {}

// enums
class Cxx_I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "mlir::cxx";
}

def Cxx_InlineKind : Cxx_I32EnumAttr<"InlineKind", "inlineKind", [
  I32EnumAttrCase<"NoInline", 0, "no_inline">,
  I32EnumAttrCase<"InlineHint", 1, "inline">
]>;

def Cxx_LinkageKind : Cxx_I32EnumAttr<"LinkageKind", "linkageKind", [
  I32EnumAttrCase<"External", 0, "external">,
  I32EnumAttrCase<"Internal", 1, "internal">
]>;

// types

def Cxx_ExprType : Cxx_Type<"Expr", "expr">;

def Cxx_VoidType : Cxx_Type<"Void", "void">;

def Cxx_BoolType : Cxx_Type<"Bool", "bool">;

def Cxx_IntegerType : Cxx_Type<"Integer", "int"> {
  let parameters = (ins "unsigned":$width, "bool":$isSigned);

  let assemblyFormat = "`<` $width `,` $isSigned `>`";
}

def Cxx_FloatType : Cxx_Type<"Float", "float"> {
  let parameters = (ins "unsigned":$width);

  let assemblyFormat = "`<` $width `>`";
}

def Cxx_PointerType : Cxx_Type<"Pointer", "ptr"> {
  let parameters = (ins "Type":$elementType);

  let assemblyFormat = "`<` $elementType `>`";
}

def Cxx_ArrayType : Cxx_Type<"Array", "array"> {
  let parameters = (ins "Type":$elementType, "unsigned":$size);

  let assemblyFormat = "`<` $elementType `,` $size `>`";
}

def Cxx_ClassType : Cxx_Type<"Class", "class", [MutableType]> {

  let storageClass = "ClassTypeStorage";
  let genStorageClass = 0;

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let parameters = (ins
    StringRefParameter<"class name", [{ "" }]>:$name,
    OptionalArrayRefParameter<"mlir::Type">:$body
  );

  let extraClassDeclaration = [{
    static auto getNamed(MLIRContext *context, StringRef name) -> ClassType;
    auto setBody(ArrayRef<Type> types) -> LogicalResult;
  }];

}

def Cxx_FunctionType : Cxx_Type<"Function", "function"> {
  let parameters = (ins
      ArrayRefParameter<"mlir::Type">:$inputs
    , ArrayRefParameter<"mlir::Type">:$results
    , "bool":$variadic
  );

  let assemblyFormat = "`<` $inputs `,` $results `,` $variadic `>`";

  let extraClassDeclaration = [{
    auto clone(mlir::TypeRange inputs, mlir::TypeRange results) const -> FunctionType;
  }];
}

def Cxx_IntOrPointerType : AnyTypeOf<[Cxx_IntegerType, Cxx_PointerType]>;

// ops

def Cxx_FuncOp : Cxx_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<Cxx_FunctionType>:$function_type,
      OptionalAttr<Cxx_LinkageKind>:$linkage_kind,
      OptionalAttr<Cxx_InlineKind>:$inline_kind,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    auto getArgumentTypes() -> ArrayRef<Type> { return getFunctionType().getInputs(); }
    auto getResultTypes() -> ArrayRef<Type> { return getFunctionType().getResults(); }
    auto getCallableRegion() -> Region* { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Cxx_GlobalOp : Cxx_Op<"global"> {
  let arguments = (ins
      TypeAttrOf<AnyType>:$global_type
    , UnitAttr:$constant
    , SymbolNameAttr:$sym_name
    , OptionalAttr<AnyAttr>:$value
  );
}

def Cxx_ReturnOp : Cxx_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$input);

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
}

def Cxx_CallOp : Cxx_Op<"call"> {
  let arguments = (ins
      FlatSymbolRefAttr:$callee
    , Variadic<AnyType>:$inputs
    , OptionalAttr<TypeAttrOf<Cxx_FunctionType>>:$var_callee_type
  );

  let results = (outs Optional<AnyType>:$result);
}

def Cxx_AllocaOp : Cxx_Op<"alloca"> {
  let arguments = (ins);

  let results = (outs Cxx_PointerType:$result);
}

def Cxx_LoadOp : Cxx_Op<"load"> {
  let arguments = (ins Cxx_PointerType:$addr);

  let results = (outs AnyType:$result);
}

def Cxx_StoreOp : Cxx_Op<"store"> {
  let arguments = (ins AnyType:$value, Cxx_PointerType:$addr);

  let hasVerifier = 1;
}

def Cxx_SubscriptOp : Cxx_Op<"subscript"> {
  let arguments = (ins Cxx_PointerType:$base, AnyType:$index);

  let results = (outs Cxx_PointerType:$result);
}

def Cxx_MemberOp : Cxx_Op<"member"> {
  let arguments = (ins Cxx_PointerType:$base, I32Prop:$member_index);

  let results = (outs Cxx_PointerType:$result);
}

def Cxx_PtrAddOp : Cxx_Op<"ptr_add"> {
  let arguments = (ins Cxx_PointerType:$base, Cxx_IntegerType:$offset);

  let results = (outs Cxx_PointerType:$result);
}

def Cxx_AddressOfOp : Cxx_Op<"addressof"> {
  let arguments = (ins FlatSymbolRefAttr:$sym_name);

  let results = (outs AnyType:$result);
}

def Cxx_BoolConstantOp : Cxx_Op<"constant.bool", [
  Pure
]> {
  let arguments = (ins BoolProp:$value);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_IntConstantOp : Cxx_Op<"constant.int", [
  Pure
]> {
  let arguments = (ins I64Prop:$value);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_FloatConstantOp : Cxx_Op<"constant.float", [
  Pure
]> {
  let arguments = (ins TypedAttrInterface:$value);

  let results = (outs Cxx_FloatType:$result);
}

def Cxx_NullPtrConstantOp : Cxx_Op<"constant.null", [
  Pure
]> {
  let arguments = (ins);

  let results = (outs Cxx_PointerType:$result);
}

def Cxx_IntToBoolOp : Cxx_Op<"int_to_bool"> {
  let arguments = (ins AnyType:$value);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_FloatToBoolOp : Cxx_Op<"float_to_bool"> {
  let arguments = (ins Cxx_FloatType:$value);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_PtrToBoolOp : Cxx_Op<"ptr_to_bool"> {
  let arguments = (ins Cxx_PointerType:$value);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_BoolToIntOp : Cxx_Op<"bool_to_int"> {
  let arguments = (ins Cxx_BoolType:$value);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_IntegralCastOp : Cxx_Op<"integral_cast"> {
  let arguments = (ins Cxx_IntegerType:$value);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_ArrayToPointerOp : Cxx_Op<"array_to_pointer"> {
  let arguments = (ins Cxx_PointerType:$value);

  let results = (outs Cxx_PointerType:$result);
}

def Cxx_NotOp : Cxx_Op<"not"> {
  let arguments = (ins AnyType:$value);

  let results = (outs AnyType:$result);
}

def Cxx_AddIOp : Cxx_Op<"addi"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_SubIOp : Cxx_Op<"subi"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_MulIOp : Cxx_Op<"muli"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_DivIOp : Cxx_Op<"divi"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_ModIOp : Cxx_Op<"mod"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_ShiftLeftOp : Cxx_Op<"shl"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_ShiftRightOp : Cxx_Op<"shr"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_EqualOp : Cxx_Op<"eq"> {
  let arguments = (ins Cxx_IntOrPointerType:$lhs, Cxx_IntOrPointerType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_NotEqualOp : Cxx_Op<"ne"> {
  let arguments = (ins Cxx_IntOrPointerType:$lhs, Cxx_IntOrPointerType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_LessThanOp : Cxx_Op<"lt"> {
  let arguments = (ins Cxx_IntOrPointerType:$lhs, Cxx_IntOrPointerType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_LessEqualOp : Cxx_Op<"le"> {
  let arguments = (ins Cxx_IntOrPointerType:$lhs, Cxx_IntOrPointerType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_GreaterThanOp : Cxx_Op<"gt"> {
  let arguments = (ins Cxx_IntOrPointerType:$lhs, Cxx_IntOrPointerType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_GreaterEqualOp : Cxx_Op<"ge"> {
  let arguments = (ins Cxx_IntOrPointerType:$lhs, Cxx_IntOrPointerType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

//
// float operations
//

def Cxx_AddFOp : Cxx_Op<"addf"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_FloatType:$result);
}

def Cxx_SubFOp : Cxx_Op<"subf"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_FloatType:$result);
}

def Cxx_MulFOp : Cxx_Op<"mulf"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_FloatType:$result);
}

def Cxx_DivFOp : Cxx_Op<"divf"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_FloatType:$result);
}

def Cxx_FloatingPointCastOp : Cxx_Op<"floating_point_cast"> {
  let arguments = (ins Cxx_FloatType:$value);

  let results = (outs Cxx_FloatType:$result);
}

def Cxx_FloatToIntOp : Cxx_Op<"float_to_int"> {
  let arguments = (ins Cxx_FloatType:$value);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_IntToFloatOp : Cxx_Op<"int_to_float"> {
  let arguments = (ins Cxx_IntegerType:$value);

  let results = (outs Cxx_FloatType:$result);
}

def Cxx_LessThanFOp : Cxx_Op<"ltf"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_LessEqualFOp : Cxx_Op<"lef"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_GreaterThanFOp : Cxx_Op<"gtf"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_GreaterEqualFOp : Cxx_Op<"gef"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_EqualFOp : Cxx_Op<"eqf"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

def Cxx_NotEqualFOp : Cxx_Op<"nef"> {
  let arguments = (ins Cxx_FloatType:$lhs, Cxx_FloatType:$rhs);

  let results = (outs Cxx_BoolType:$result);
}

// bitwise ops
def Cxx_AndOp : Cxx_Op<"and"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_OrOp : Cxx_Op<"or"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

def Cxx_XorOp : Cxx_Op<"xor"> {
  let arguments = (ins Cxx_IntegerType:$lhs, Cxx_IntegerType:$rhs);

  let results = (outs Cxx_IntegerType:$result);
}

//
// control flow ops
//

def Cxx_LabelOp : Cxx_Op<"label"> {
  let arguments = (ins StringProp:$name);
}

def Cxx_GotoOp : Cxx_Op<"goto"> {
  let arguments = (ins StringProp:$label);
}

def Cxx_CondBranchOp : Cxx_Op<"cond_br", [ AttrSizedOperandSegments, Terminator ]> {
  let arguments = (ins Cxx_BoolType:$condition, Variadic<AnyType>:$trueDestOperands, Variadic<AnyType>:$falseDestOperands);

  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
}

def Cxx_SwitchOp : Cxx_Op<"switch", [ AttrSizedOperandSegments, Terminator ]> {
  let arguments = (ins
    Cxx_IntegerType:$value,
    Variadic<AnyType>:$defaultOperands,
    VariadicOfVariadic<AnyType, "case_operand_segments">:$caseOperands,
    OptionalAttr<AnyIntElementsAttr>:$case_values,
    DenseI32ArrayAttr:$case_operand_segments
  );

  let successors = (successor
    AnySuccessor:$defaultDestination,
    VariadicSuccessor<AnySuccessor>:$caseDestinations
  );

  let builders = [
    OpBuilder<(ins "Value":$value,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"ArrayRef<std::int64_t>", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands)>,
    OpBuilder<(ins "Value":$value,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"DenseIntElementsAttr", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands)>
  ];
}

//
// todo ops
//

def Cxx_TodoExprOp : Cxx_Op<"todo.expr"> {
  let arguments = (ins StringProp:$message);
  let results = (outs Cxx_ExprType:$result);
  let assemblyFormat = "$message attr-dict `:` type($result)";
  let builders =
      [OpBuilder<
           (ins "::llvm::StringRef":$message),
           [{ build($_builder, $_state, $_builder.getType<ExprType>(), message); }]>,
  ];
}

def Cxx_TodoStmtOp : Cxx_Op<"todo.stmt"> {
  let arguments = (ins StringProp:$message);
  let results = (outs);
  let assemblyFormat = "$message attr-dict";
}
