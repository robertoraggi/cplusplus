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

#include <cxx/mlir/wgsl_name_encoder.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringExtras.h>

#include <array>

namespace cxx {

namespace {

constexpr llvm::StringLiteral kMarker = "__";
constexpr llvm::StringLiteral kValue = "v";
constexpr llvm::StringLiteral kResult = "r";
constexpr llvm::StringLiteral kSnapshot = "n";
constexpr llvm::StringLiteral kInsertedValue = "value";
constexpr llvm::StringLiteral kPrivateVariable = "p";
constexpr llvm::StringLiteral kInduction = "i";
constexpr llvm::StringLiteral kLoopValue = "l";
constexpr llvm::StringLiteral kOldValue = "old";
constexpr llvm::StringLiteral kExchanged = "exchanged";
constexpr llvm::StringLiteral kAttempt = "attempt";
constexpr llvm::StringLiteral kRecord = "Record";
constexpr llvm::StringLiteral kField = "field";
constexpr llvm::StringLiteral kArgument = "arg";
constexpr llvm::StringLiteral kBinding = "buffer";
constexpr llvm::StringLiteral kWorkgroupVariable = "shared";
constexpr llvm::StringLiteral kGlobalId = "global_id__";
constexpr llvm::StringLiteral kLocalId = "local_id__";
constexpr llvm::StringLiteral kWorkgroupId = "workgroup_id__";
constexpr llvm::StringLiteral kRenamedFunction = "function";

constexpr auto kWGSLReservedWords = std::to_array<llvm::StringLiteral>({
    "alias",
    "break",
    "case",
    "const",
    "const_assert",
    "continue",
    "continuing",
    "default",
    "diagnostic",
    "discard",
    "else",
    "enable",
    "false",
    "fn",
    "for",
    "if",
    "let",
    "loop",
    "override",
    "requires",
    "return",
    "struct",
    "switch",
    "true",
    "var",
    "while",
    "NULL",
    "Self",
    "abstract",
    "active",
    "alignas",
    "alignof",
    "as",
    "asm",
    "asm_fragment",
    "async",
    "attribute",
    "auto",
    "await",
    "become",
    "cast",
    "catch",
    "class",
    "co_await",
    "co_return",
    "co_yield",
    "coherent",
    "column_major",
    "common",
    "compile",
    "compile_fragment",
    "concept",
    "const_cast",
    "consteval",
    "constexpr",
    "constinit",
    "crate",
    "debugger",
    "decltype",
    "delete",
    "demote",
    "demote_to_helper",
    "do",
    "dynamic_cast",
    "enum",
    "explicit",
    "export",
    "extends",
    "extern",
    "external",
    "fallthrough",
    "filter",
    "final",
    "finally",
    "friend",
    "from",
    "fxgroup",
    "get",
    "goto",
    "groupshared",
    "highp",
    "impl",
    "implements",
    "import",
    "inline",
    "instanceof",
    "interface",
    "layout",
    "lowp",
    "macro",
    "macro_rules",
    "match",
    "mediump",
    "meta",
    "mod",
    "module",
    "move",
    "mut",
    "mutable",
    "namespace",
    "new",
    "nil",
    "noexcept",
    "noinline",
    "nointerpolation",
    "non_coherent",
    "noncoherent",
    "noperspective",
    "null",
    "nullptr",
    "of",
    "operator",
    "package",
    "packoffset",
    "partition",
    "pass",
    "patch",
    "pixelfragment",
    "precise",
    "precision",
    "premerge",
    "priv",
    "protected",
    "pub",
    "public",
    "readonly",
    "ref",
    "regardless",
    "register",
    "reinterpret_cast",
    "require",
    "resource",
    "restrict",
    "self",
    "set",
    "shared",
    "sizeof",
    "smooth",
    "snorm",
    "static",
    "static_assert",
    "static_cast",
    "std",
    "subroutine",
    "super",
    "target",
    "template",
    "this",
    "thread_local",
    "throw",
    "trait",
    "try",
    "type",
    "typedef",
    "typeid",
    "typename",
    "typeof",
    "union",
    "unless",
    "unorm",
    "unsafe",
    "unsized",
    "use",
    "using",
    "varying",
    "virtual",
    "volatile",
    "wgsl",
    "where",
    "with",
    "writeonly",
    "yield",
});

constexpr auto kWGSLPredeclaredNames = std::to_array<llvm::StringLiteral>({
    "abs",
    "acos",
    "acosh",
    "all",
    "any",
    "array",
    "arrayLength",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "atomic",
    "atomicAdd",
    "atomicAnd",
    "atomicCompareExchangeWeak",
    "atomicExchange",
    "atomicLoad",
    "atomicMax",
    "atomicMin",
    "atomicOr",
    "atomicStore",
    "atomicStoreMax",
    "atomicStoreMin",
    "atomicSub",
    "atomicXor",
    "bar",
    "bgra8unorm",
    "bool",
    "bufferLength",
    "ceil",
    "clamp",
    "cos",
    "cosh",
    "countLeadingZeros",
    "countOneBits",
    "countTrailingZeros",
    "cross",
    "degrees",
    "determinant",
    "distance",
    "dot",
    "dot4I8Packed",
    "dot4U8Packed",
    "dpdx",
    "dpdxCoarse",
    "dpdxFine",
    "dpdy",
    "dpdyCoarse",
    "dpdyFine",
    "exp",
    "exp2",
    "extractBits",
    "f16",
    "f32",
    "faceForward",
    "firstLeadingBit",
    "firstTrailingBit",
    "floor",
    "fma",
    "foo",
    "foobar",
    "fract",
    "frexp",
    "func",
    "function",
    "fwidth",
    "fwidthCoarse",
    "fwidthFine",
    "gather_depth_compare",
    "gather_depth_components",
    "gather_x_components",
    "gather_y_components",
    "gather_z_components",
    "i32",
    "insertBits",
    "inverseSqrt",
    "ldexp",
    "length",
    "log",
    "log2",
    "main",
    "mat2x2",
    "mat2x2f",
    "mat2x2h",
    "mat2x3",
    "mat2x3f",
    "mat2x3h",
    "mat2x4",
    "mat2x4f",
    "mat2x4h",
    "mat3x2",
    "mat3x2f",
    "mat3x2h",
    "mat3x3",
    "mat3x3f",
    "mat3x3h",
    "mat3x4",
    "mat3x4f",
    "mat3x4h",
    "mat4x2",
    "mat4x2f",
    "mat4x2h",
    "mat4x3",
    "mat4x3f",
    "mat4x3h",
    "mat4x4",
    "mat4x4f",
    "mat4x4h",
    "max",
    "min",
    "mix",
    "modf",
    "normalize",
    "num_point_lights",
    "pack2x16float",
    "pack2x16snorm",
    "pack2x16unorm",
    "pack4x8snorm",
    "pack4x8unorm",
    "pack4xI8",
    "pack4xI8Clamp",
    "pack4xU8",
    "pack4xU8Clamp",
    "pow",
    "private",
    "ptr",
    "quadBroadcast",
    "quadSwapDiagonal",
    "quadSwapX",
    "quadSwapY",
    "quantizeToF16",
    "r16float",
    "r16sint",
    "r16snorm",
    "r16uint",
    "r16unorm",
    "r32float",
    "r32sint",
    "r32uint",
    "r8sint",
    "r8snorm",
    "r8uint",
    "r8unorm",
    "radians",
    "read",
    "read_write",
    "reflect",
    "refract",
    "reverseBits",
    "rg11b10ufloat",
    "rg16float",
    "rg16sint",
    "rg16snorm",
    "rg16uint",
    "rg16unorm",
    "rg32float",
    "rg32sint",
    "rg32uint",
    "rg8sint",
    "rg8snorm",
    "rg8uint",
    "rg8unorm",
    "rgb10a2uint",
    "rgb10a2unorm",
    "rgba16float",
    "rgba16sint",
    "rgba16snorm",
    "rgba16uint",
    "rgba16unorm",
    "rgba32float",
    "rgba32sint",
    "rgba32uint",
    "rgba8sint",
    "rgba8snorm",
    "rgba8uint",
    "rgba8unorm",
    "round",
    "sampler",
    "sampler_comparison",
    "saturate",
    "select",
    "sign",
    "sin",
    "sinh",
    "smoothstep",
    "sqrt",
    "step",
    "storage",
    "storageBarrier",
    "subgroupAdd",
    "subgroupAll",
    "subgroupAnd",
    "subgroupAny",
    "subgroupBallot",
    "subgroupBroadcast",
    "subgroupBroadcastFirst",
    "subgroupElect",
    "subgroupExclusiveAdd",
    "subgroupExclusiveMul",
    "subgroupInclusiveAdd",
    "subgroupInclusiveMul",
    "subgroupMax",
    "subgroupMin",
    "subgroupMul",
    "subgroupOr",
    "subgroupShuffle",
    "subgroupShuffleDown",
    "subgroupShuffleUp",
    "subgroupShuffleXor",
    "subgroupXor",
    "tan",
    "tanh",
    "textureBarrier",
    "textureDimensions",
    "textureGather",
    "textureGatherCompare",
    "textureLoad",
    "textureNumLayers",
    "textureNumLevels",
    "textureNumSamples",
    "textureSample",
    "textureSampleBaseClampToEdge",
    "textureSampleBias",
    "textureSampleCompare",
    "textureSampleCompareLevel",
    "textureSampleGrad",
    "textureSampleLevel",
    "textureStore",
    "texture_1d",
    "texture_2d",
    "texture_2d_array",
    "texture_3d",
    "texture_cube",
    "texture_cube_array",
    "texture_depth_2d",
    "texture_depth_2d_array",
    "texture_depth_cube",
    "texture_depth_cube_array",
    "texture_depth_multisampled_2d",
    "texture_external",
    "texture_multisampled_2d",
    "texture_storage_1d",
    "texture_storage_2d",
    "texture_storage_2d_array",
    "texture_storage_3d",
    "transpose",
    "trunc",
    "u32",
    "uniform",
    "unpack2x16float",
    "unpack2x16snorm",
    "unpack2x16unorm",
    "unpack4x8snorm",
    "unpack4x8unorm",
    "unpack4xI8",
    "unpack4xU8",
    "vec2",
    "vec2f",
    "vec2h",
    "vec2i",
    "vec2u",
    "vec3",
    "vec3f",
    "vec3h",
    "vec3i",
    "vec3u",
    "vec4",
    "vec4f",
    "vec4h",
    "vec4i",
    "vec4u",
    "workgroup",
    "workgroupBarrier",
    "workgroupUniformLoad",
    "write",
});

[[nodiscard]] auto marked(llvm::StringRef role, llvm::StringRef suffix)
    -> std::string {
  return (role + kMarker + suffix).str();
}

[[nodiscard]] auto numbered(llvm::StringRef role, std::uint32_t index)
    -> std::string {
  return marked(role, std::to_string(index));
}

[[nodiscard]] auto temporarySpelling(WGSLNameEncoder::Temporary kind)
    -> llvm::StringRef {
  switch (kind) {
    case WGSLNameEncoder::Temporary::kValue:
      return kValue;
    case WGSLNameEncoder::Temporary::kResult:
      return kResult;
    case WGSLNameEncoder::Temporary::kSnapshot:
      return kSnapshot;
    case WGSLNameEncoder::Temporary::kInsertedValue:
      return kInsertedValue;
    case WGSLNameEncoder::Temporary::kPrivateVariable:
      return kPrivateVariable;
    case WGSLNameEncoder::Temporary::kInduction:
      return kInduction;
    case WGSLNameEncoder::Temporary::kLoopValue:
      return kLoopValue;
    case WGSLNameEncoder::Temporary::kOldValue:
      return kOldValue;
    case WGSLNameEncoder::Temporary::kExchanged:
      return kExchanged;
    case WGSLNameEncoder::Temporary::kAttempt:
      return kAttempt;
  }
  return kValue;
}

}  // namespace

auto WGSLNameEncoder::encodeTemporary(Temporary kind) -> std::string {
  return numbered(temporarySpelling(kind), nextTemporary_++);
}

auto WGSLNameEncoder::encodeRecord() -> std::string {
  return numbered(kRecord, nextRecord_++);
}

auto WGSLNameEncoder::encodeField(std::uint32_t position) -> std::string {
  return numbered(kField, position);
}

auto WGSLNameEncoder::encodeArgument(std::uint32_t index) -> std::string {
  return numbered(kArgument, index);
}

auto WGSLNameEncoder::encodeBinding(std::uint32_t resource) -> std::string {
  return numbered(kBinding, resource);
}

auto WGSLNameEncoder::encodeWorkgroupVariable(std::uint32_t index)
    -> std::string {
  return numbered(kWorkgroupVariable, index);
}

auto WGSLNameEncoder::encodeBuiltin(Builtin builtin) -> llvm::StringRef {
  switch (builtin) {
    case Builtin::kGlobalId:
      return kGlobalId;
    case Builtin::kLocalId:
      return kLocalId;
    case Builtin::kWorkgroupId:
      return kWorkgroupId;
  }
  return kGlobalId;
}

auto WGSLNameEncoder::encodeFunction(llvm::StringRef name) -> std::string {
  std::string spelled;
  for (unsigned char ch : name)
    spelled.push_back(llvm::isAlnum(ch) || ch == '_' ? ch : '_');
  if (isUsableEntryName(spelled) && !llvm::StringRef(spelled).contains(kMarker))
    return spelled;
  return marked(kRenamedFunction, spelled);
}

auto WGSLNameEncoder::isUsableEntryName(llvm::StringRef name) -> bool {
  if (name.empty() || name == "_" || name.starts_with(kMarker)) return false;
  if (llvm::is_contained(kWGSLReservedWords, name)) return false;
  return !llvm::is_contained(kWGSLPredeclaredNames, name);
}

}  // namespace cxx
