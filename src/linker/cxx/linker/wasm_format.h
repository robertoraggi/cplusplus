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

#include <cstdint>
#include <vector>

namespace cxx::linker {

enum SectionId : std::uint8_t {
  kCustomSection = 0,
  kTypeSection = 1,
  kImportSection = 2,
  kFunctionSection = 3,
  kTableSection = 4,
  kMemorySection = 5,
  kGlobalSection = 6,
  kExportSection = 7,
  kElementSection = 9,
  kCodeSection = 10,
  kDataSection = 11,
  kTagSection = 13,
};

enum ValueType : std::uint8_t {
  kI32 = 0x7f,
  kI64 = 0x7e,
  kF32 = 0x7d,
  kF64 = 0x7c,
  kFuncRef = 0x70,
};

enum ExternalKind : std::uint8_t {
  kExternalFunction = 0,
  kExternalTable = 1,
  kExternalMemory = 2,
  kExternalGlobal = 3,
  kExternalTag = 4,
};

enum SymbolKindId : std::uint8_t {
  kSymbolFunction = 0,
  kSymbolData = 1,
  kSymbolGlobal = 2,
  kSymbolSection = 3,
  kSymbolTag = 4,
  kSymbolTable = 5,
};

constexpr ExternalKind kImportKindOfSymbol[] = {
    kExternalFunction, kExternalMemory, kExternalGlobal,
    kExternalMemory,   kExternalTag,    kExternalTable,
};

enum SymbolFlag : std::uint32_t {
  kBindingWeak = 0x01,
  kBindingLocal = 0x02,
  kUndefined = 0x10,
  kExported = 0x20,
  kExplicitName = 0x40,
  kNoStrip = 0x80,
};

enum LinkingSubsection : std::uint8_t {
  kSegmentInfo = 5,
  kInitFunctions = 6,
  kSymbolTableSubsection = 8,
};

enum SegmentFlag : std::uint32_t {
  kSegmentTls = 0x02,
};

enum RelocationType : std::uint8_t {
  kFunctionIndexLeb = 0,
  kTableIndexSleb = 1,
  kTableIndexI32 = 2,
  kMemoryAddrLeb = 3,
  kMemoryAddrSleb = 4,
  kMemoryAddrI32 = 5,
  kTypeIndexLeb = 6,
  kGlobalIndexLeb = 7,
  kFunctionOffsetI32 = 8,
  kSectionOffsetI32 = 9,
  kTagIndexLeb = 10,
  kMemoryAddrRelSleb = 11,
  kTableIndexRelSleb = 12,
  kGlobalIndexI32 = 13,
  kTableNumberLeb = 20,
  kMemoryAddrLocrelI32 = 23,
  kFunctionIndexI32 = 26,
};

enum Opcode : std::uint8_t {
  kOpUnreachable = 0x00,
  kOpEnd = 0x0b,
  kOpCall = 0x10,
  kOpGlobalGet = 0x23,
  kOpI32Const = 0x41,
  kOpI64Const = 0x42,
  kOpF32Const = 0x43,
  kOpF64Const = 0x44,
};

constexpr std::uint8_t kFunctionTypeForm = 0x60;
constexpr std::uint8_t kLimitsHasMax = 0x01;
constexpr std::uint32_t kPageSize = 65536;

struct FuncType {
  std::vector<std::uint8_t> params;
  std::vector<std::uint8_t> results;

  [[nodiscard]] auto operator==(const FuncType&) const -> bool = default;
};

}  // namespace cxx::linker
