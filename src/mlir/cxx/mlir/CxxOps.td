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

def Cxx_Dialect : Dialect {
  let name = "cxx";
  let cppNamespace = "mlir::cxx";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::scf::SCFDialect",
  ];
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

def Cxx_PointerType : Cxx_Type<"Pointer", "ptr"> {
  let parameters = (ins "Type":$elementType);

  let assemblyFormat = "`<` $elementType `>`";
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

  let extraClassDeclaration =
      [{
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
}

def Cxx_TodoExprOp : Cxx_Op<"todo.expr"> {
  let arguments = (ins StrAttr:$message);
  let results = (outs Cxx_ExprType:$result);
  let assemblyFormat = "$message attr-dict `:` type($result)";
  let builders =
      [OpBuilder<
           (ins "::llvm::StringRef":$message),
           [{ build($_builder, $_state, $_builder.getType<ExprType>(), message); }]>,
  ];
}

def Cxx_TodoStmtOp : Cxx_Op<"todo.stmt"> {
  let arguments = (ins StrAttr:$message);
  let results = (outs);
  let assemblyFormat = "$message attr-dict";
}