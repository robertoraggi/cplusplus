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

def Cxx_Dialect : Dialect {
  let name = "cxx";
  let cppNamespace = "mlir::cxx";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
  ];
}

class Cxx_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Cxx_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

class Cxx_Op<string mnemonic, list<Trait> traits = []> : Op<Cxx_Dialect, mnemonic, traits> {
}

def Cxx_ExprType : Cxx_Type<"Expr", "expr">;

def TodoExprOp : Cxx_Op<"todo.expr"> {
  let arguments = (ins StrAttr:$message);
  let results = (outs Cxx_ExprType:$result);
  let assemblyFormat = "$message attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$message),
      [{ build($_builder, $_state, $_builder.getType<ExprType>(), message); }]>,
  ];
}

def TodoStmtOp : Cxx_Op<"todo.stmt"> {
  let arguments = (ins StrAttr:$message);
  let results = (outs);
  let assemblyFormat = "$message attr-dict";
}

def ToBoolOp : Cxx_Op<"to.bool"> {
  let arguments = (ins Cxx_ExprType:$value);
  let results = (outs I1:$result);
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
}

def ImplicitCastOp : Cxx_Op<"implicit.cast"> {
  let arguments = (ins StrAttr:$cast, Cxx_ExprType:$value);
  let results = (outs Cxx_ExprType:$result);
  let assemblyFormat = "$cast `(` $value `)` attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$cast, "mlir::Value":$value),
      [{ build($_builder, $_state, $_builder.getType<ExprType>(), cast, value); }]>,
  ];
}

def IntLiteralOp : Cxx_Op<"int.literal"> {
  let arguments = (ins I64Attr:$value);
  let results = (outs Cxx_ExprType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "int64_t":$value),
      [{ build($_builder, $_state, $_builder.getType<ExprType>(), value); }]>,
  ];
}

def IdOp : Cxx_Op<"id"> {
  let arguments = (ins StrAttr:$name);
  let results = (outs Cxx_ExprType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name),
      [{ build($_builder, $_state, $_builder.getType<ExprType>(), name); }]>,
  ];
}

def BinOp : Cxx_Op<"binary"> {
  let arguments = (ins StrAttr:$op, Cxx_ExprType:$lhs, Cxx_ExprType:$rhs);
  let results = (outs Cxx_ExprType:$result);
  let assemblyFormat = "`op` $op `(` $lhs `,` $rhs `)` attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$op, "mlir::Value":$lhs, "mlir::Value":$rhs),
      [{ build($_builder, $_state, $_builder.getType<ExprType>(), op, lhs, rhs); }]>,
  ];
}

def CallOp : Cxx_Op<"call"> {
  let arguments = (ins Cxx_ExprType:$callee, Variadic<AnyType>:$arguments);
  let results = (outs Cxx_ExprType:$result);
  let assemblyFormat = "$callee `(` $arguments `:` type($arguments) `)` attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "mlir::Value":$callee, "mlir::ValueRange":$arguments),
      [{ build($_builder, $_state, $_builder.getType<ExprType>(), callee, arguments); }]>,
  ];
}

def ReturnOp : Cxx_Op<"return"> {
  let arguments = (ins Optional<Cxx_ExprType>:$value);
  let results = (outs);
}