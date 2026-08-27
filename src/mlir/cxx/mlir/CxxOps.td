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

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Cxx_Dialect : Dialect {
  let name = "cxx";
  let cppNamespace = "mlir::cxx";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect", "mlir::DLTIDialect", "mlir::LLVM::LLVMDialect"];
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
  I32EnumAttrCase<"Internal", 1, "internal">,
  I32EnumAttrCase<"LinkOnceODR", 2, "linkonce_odr">,
  I32EnumAttrCase<"WeakODR", 3, "weak_odr">,
  I32EnumAttrCase<"AvailableExternally", 4, "available_externally">,
  I32EnumAttrCase<"Appending", 5, "appending">
]>;

def Cxx_Visibility : Cxx_I32EnumAttr<"Visibility", "visibility", [
  I32EnumAttrCase<"Default", 0, "default">,
  I32EnumAttrCase<"Hidden", 1, "hidden">,
  I32EnumAttrCase<"Protected", 2, "protected">
]>;

// types

def Cxx_ExprType : Cxx_Type<"Expr", "expr">;

def Cxx_VoidType : Cxx_Type<"Void", "void">;

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

// ops

def Cxx_FuncOp : Cxx_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<Cxx_FunctionType>:$function_type,
      OptionalAttr<Cxx_LinkageKind>:$linkage_kind,
      OptionalAttr<Cxx_InlineKind>:$inline_kind,
      OptionalAttr<Cxx_Visibility>:$visibility_,
      OptionalAttr<StrAttr>:$alias_name,
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
    , OptionalAttr<Cxx_LinkageKind>:$linkage_kind
    , OptionalAttr<I64Attr>:$alignment
  );

  let regions = (region AnyRegion:$initializer);

  let hasCustomAssemblyFormat = 1;
}

def Cxx_GlobalCtorOp : Cxx_Op<"global_ctor"> {
  let summary = "run a function before main, in module order";

  let arguments = (ins FlatSymbolRefAttr:$ctor);

  let assemblyFormat = "$ctor attr-dict";
}

def Cxx_VTableOp : Cxx_Op<"vtable"> {
  let summary = "one Itanium vtable group";
  let description = [{
    Storage shape (one pointer-sized word per entry):

      [vbase_offsets...] [vcall_offsets...] [offset_to_top] [type_info] [slots...]

    `vbase_offsets`, `vcall_offsets`, and `offset_to_top` are byte offsets
    stored as pointer-sized integers; `type_info` names the `_ZTI` object for
    the class this group belongs to, and is null when absent; each slot is
    either a function symbol or a unit attribute (a null slot). The lowering
    is purely mechanical -- every field's meaning is explicit here, it never
    guesses from attribute shapes.
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name
    , I64ArrayAttr:$vbase_offsets
    , I64ArrayAttr:$vcall_offsets
    , I64Attr:$offset_to_top
    , OptionalAttr<FlatSymbolRefAttr>:$type_info
    , ArrayAttr:$slots
    , OptionalAttr<Cxx_LinkageKind>:$linkage_kind
  );

  let hasCustomAssemblyFormat = 1;
}

def Cxx_ReturnOp : Cxx_Op<"return", [Pure, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$input);

  let assemblyFormat = "($input^ `:` type($input))? attr-dict";

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
}

def Cxx_UnreachableOp : Cxx_Op<"unreachable", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

def Cxx_CallOp : Cxx_Op<"call", [AttrSizedOperandSegments]> {
  let arguments = (ins
      OptionalAttr<FlatSymbolRefAttr>:$callee
    , Optional<Cxx_PointerType>:$callee_operand
    , Variadic<AnyType>:$inputs
    , OptionalAttr<TypeAttrOf<Cxx_FunctionType>>:$var_callee_type
  );

  let results = (outs Optional<AnyType>:$result);

  let assemblyFormat = [{
    ($callee^)?
    (`indirect` $callee_operand^ `:` type($callee_operand))?
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "::mlir::FlatSymbolRefAttr":$callee,
                   "::mlir::ValueRange":$inputs), [{
      build($_builder, $_state, resultTypes, callee, ::mlir::Value{}, inputs,
            ::mlir::TypeAttr{});
    }]>,
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes, "::llvm::StringRef":$callee,
                   "::mlir::ValueRange":$inputs), [{
      build($_builder, $_state, resultTypes,
            ::mlir::FlatSymbolRefAttr::get($_builder.getContext(), callee),
            inputs);
    }]>,
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes, "::mlir::Value":$callee,
                   "::mlir::ValueRange":$inputs), [{
      build($_builder, $_state, resultTypes, ::mlir::FlatSymbolRefAttr{},
            callee, inputs, ::mlir::TypeAttr{});
    }]>
  ];

  let hasVerifier = 1;
}

def Cxx_AllocaOp : Cxx_Op<"alloca"> {
  let arguments = (ins I64Prop:$alignment);

  let results = (outs Cxx_PointerType:$result);

  let assemblyFormat = "attr-dict `:` type($result) `align` $alignment";
}

def Cxx_DynAllocaOp : Cxx_Op<"dyn_alloca"> {
  let arguments = (ins AnyInteger:$size, I64Prop:$alignment);

  let results = (outs Cxx_PointerType:$result);

  let assemblyFormat = "$size attr-dict `:` type($size) `->` type($result) `align` $alignment";
}

def Cxx_LoadOp : Cxx_Op<"load"> {
  let arguments = (ins Cxx_PointerType:$addr, I64Prop:$alignment);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($result) `align` $alignment";
}

def Cxx_StoreOp : Cxx_Op<"store"> {
  let arguments = (ins AnyType:$value, Cxx_PointerType:$addr, I64Prop:$alignment);

  let assemblyFormat = "$value `,` $addr attr-dict `:` type($value) `,` type($addr) `align` $alignment";

  let hasVerifier = 1;
}

def Cxx_MemSetZeroOp : Cxx_Op<"memzero"> {
  let arguments = (ins Cxx_PointerType:$addr, I64Prop:$size);

  let assemblyFormat = "$addr attr-dict `:` type($addr) `size` $size";
}

def Cxx_MemCpyOp : Cxx_Op<"memcpy"> {
  let arguments = (ins Cxx_PointerType:$dest, Cxx_PointerType:$src, I64Prop:$size);

  let assemblyFormat = "$dest `,` $src attr-dict `:` type($dest) `,` type($src) `size` $size";
}

def Cxx_SubscriptOp : Cxx_Op<"subscript"> {
  let arguments = (ins Cxx_PointerType:$base, AnyType:$index);

  let results = (outs Cxx_PointerType:$result);

  let assemblyFormat = "$base `[` $index `]` attr-dict `:` functional-type(operands, results)";
}

def Cxx_MemberOp : Cxx_Op<"member"> {
  let arguments = (ins Cxx_PointerType:$base, I32Prop:$member_index);

  let results = (outs Cxx_PointerType:$result);

  let assemblyFormat = "$base `[` $member_index `]` attr-dict `:` type($base) `->` type($result)";
}

def Cxx_PtrAddOp : Cxx_Op<"ptradd"> {
  let arguments = (ins Cxx_PointerType:$base, AnySignlessInteger:$offset);

  let results = (outs Cxx_PointerType:$result);

  let assemblyFormat = "$base `,` $offset attr-dict `:` functional-type(operands, results)";
}

def Cxx_PtrDiffOp : Cxx_Op<"ptrdiff"> {
  let arguments = (ins Cxx_PointerType:$lhs, Cxx_PointerType:$rhs);

  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Cxx_AddressOfOp : Cxx_Op<"address_of"> {
  let arguments = (ins FlatSymbolRefAttr:$sym_name);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$sym_name attr-dict `:` type($result)";
}

def Cxx_NullPtrConstantOp : Cxx_Op<"constant.null", [
  Pure
]> {
  let arguments = (ins);

  let results = (outs Cxx_PointerType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Cxx_ArrayToPointerOp : Cxx_Op<"array_to_ptr"> {
  let arguments = (ins Cxx_PointerType:$value);

  let results = (outs Cxx_PointerType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
}

def Cxx_BitcastOp : Cxx_Op<"bitcast"> {
  let arguments = (ins AnyType:$value);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
}

def Cxx_ZeroOp : Cxx_Op<"zero", [Pure]> {
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Cxx_UndefOp : Cxx_Op<"undef", [Pure]> {
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Cxx_ReshapeOp : Cxx_Op<"reshape", [Pure]> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
}

def Cxx_InsertValueOp : Cxx_Op<"insertvalue", [AllTypesMatch<["container", "result"]>]> {
  let arguments = (ins AnyType:$container, AnyType:$value, I64Attr:$position);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$value `,` $container `[` $position `]` attr-dict `:` type($container) `,` type($value)";

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$container, "Value":$value, "int64_t":$position),
    [{
      build($_builder, $_state, resultType, container, value,
            $_builder.getI64IntegerAttr(position));
    }]>
  ];
}

def Cxx_PtrToIntOp : Cxx_Op<"ptrtoint"> {
  let arguments = (ins Cxx_PointerType:$value);

  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
}

def Cxx_IntToPtrOp : Cxx_Op<"inttoptr"> {
  let arguments = (ins AnySignlessInteger:$value);

  let results = (outs Cxx_PointerType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
}

//
// control flow ops
//

def Cxx_LabelOp : Cxx_Op<"label"> {
  let arguments = (ins StringProp:$name, I64Attr:$cleanup_depth);

  let assemblyFormat = "$name `cleanup_depth` $cleanup_depth attr-dict";
}

def Cxx_GotoOp : Cxx_Op<"goto", [AttrSizedOperandSegments]> {
  let arguments = (ins
      Variadic<Cxx_PointerType>:$addresses
    , Variadic<Cxx_PointerType>:$activeFlags
    , ArrayAttr:$destructors
    , ArrayAttr:$depths
    , DenseI32ArrayAttr:$activeFlagIndices
    , StringProp:$label
  );

  let assemblyFormat = "$label `(` $addresses `:` type($addresses) `)` `destructors` $destructors `depths` $depths `flags` `(` $activeFlags `:` type($activeFlags) `)` `active` $activeFlagIndices attr-dict";
}

def Cxx_LabelAddressOp : Cxx_Op<"label_address"> {
  let arguments = (ins StringProp:$label_name,
                       OptionalAttr<I32Attr>:$tag_id,
                       OptionalAttr<StrAttr>:$func_name);
  let results   = (outs Cxx_PointerType:$result);

  let assemblyFormat = "$label_name attr-dict `:` type($result)";
}

def Cxx_IndirectGotoOp : Cxx_Op<"indirect_goto", [Terminator]> {
  let arguments = (ins AnyType:$target);
  let successors = (successor VariadicSuccessor<AnySuccessor>:$destinations);

  let assemblyFormat = "$target `[` $destinations `]` attr-dict `:` type($target)";
}

def Cxx_CleanupBranchOp : Cxx_Op<"cleanup_branch",
    [Terminator, AttrSizedOperandSegments]> {
  let arguments = (ins
      Variadic<Cxx_PointerType>:$addresses
    , Variadic<Cxx_PointerType>:$activeFlags
    , ArrayAttr:$destructors
    , DenseI32ArrayAttr:$activeFlagIndices
  );

  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = "`(` $addresses `:` type($addresses) `)` `destructors` $destructors `flags` `(` $activeFlags `:` type($activeFlags) `)` `active` $activeFlagIndices $dest attr-dict";
}

//
// builtin ops
//

def Cxx_BuiltinCallOp : Cxx_Op<"builtin.call"> {
  let arguments = (ins
      StringProp:$builtin_name
    , Variadic<AnyType>:$inputs
  );

  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = "$builtin_name `(` $inputs `)` attr-dict `:` functional-type($inputs, results)";
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

def BitfieldLoadOp : Cxx_Op<"bitfield.load"> {
  let arguments = (ins Cxx_PointerType:$addr,
                       I32Attr:$bit_offset, I32Attr:$bit_width,
                       I64Attr:$alignment, BoolAttr:$is_signed);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($result)";
}

def BitfieldStoreOp : Cxx_Op<"bitfield.store"> {
  let arguments = (ins AnySignlessInteger:$value, Cxx_PointerType:$addr,
                       I32Attr:$bit_offset, I32Attr:$bit_width,
                       I64Attr:$alignment);
  let assemblyFormat = "$value `,` $addr attr-dict `:` type($value) `,` type($addr)";
}
