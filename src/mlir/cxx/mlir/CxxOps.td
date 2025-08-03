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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Cxx_Dialect : Dialect {
  let name = "cxx";
  let cppNamespace = "mlir::cxx";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["mlir::cf::ControlFlowDialect" ];
}

class Cxx_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Cxx_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Cxx_Op<string mnemonic, list<Trait> traits = []>
    : Op<Cxx_Dialect, mnemonic, traits> {}


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

// ops

def Cxx_FuncOp : Cxx_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    auto getArgumentTypes() -> ArrayRef<Type> { return getFunctionType().getInputs(); }
    auto getResultTypes() -> ArrayRef<Type> { return getFunctionType().getResults(); }
    auto getCallableRegion() -> Region* { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def Cxx_ReturnOp : Cxx_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$input);

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 0;
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

def Cxx_IntToBoolOp : Cxx_Op<"int_to_bool"> {
  let arguments = (ins AnyType:$value);

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

def Cxx_NotOp : Cxx_Op<"not"> {
  let arguments = (ins Cxx_BoolType:$value);

  let results = (outs Cxx_BoolType:$result);
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

def CondBranchOp : Cxx_Op<"cond_br", [ AttrSizedOperandSegments, Terminator ]> {
  let arguments = (ins Cxx_BoolType:$condition, Variadic<AnyType>:$trueDestOperands, Variadic<AnyType>:$falseDestOperands);

  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
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