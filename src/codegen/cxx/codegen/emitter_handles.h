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
#include <functional>

namespace cxx::ir {

template <typename Tag>
class Handle {
 public:
  constexpr Handle() = default;

  [[nodiscard]] constexpr explicit operator bool() const { return id_ != 0; }

  friend constexpr auto operator==(Handle, Handle) -> bool = default;

 private:
  constexpr explicit Handle(std::uint32_t id) : id_(id) {}

  [[nodiscard]] constexpr auto id() const -> std::uint32_t { return id_; }

  std::uint32_t id_ = 0;

  friend class HandleAccess;
};

class HandleAccess {
 public:
  template <typename Tag>
  [[nodiscard]] static constexpr auto make(std::uint32_t id) -> Handle<Tag> {
    return Handle<Tag>{id};
  }

  template <typename Tag>
  [[nodiscard]] static constexpr auto id(Handle<Tag> handle) -> std::uint32_t {
    return handle.id();
  }
};

inline constexpr std::uint32_t kHandleIndexBits = 24;
inline constexpr std::uint32_t kHandleIndexMask = (1u << kHandleIndexBits) - 1;
inline constexpr std::uint32_t kHandleMaxIndex = kHandleIndexMask;

[[nodiscard]] constexpr auto handleId(std::uint32_t index,
                                      std::uint8_t generation)
    -> std::uint32_t {
  return index | (static_cast<std::uint32_t>(generation) << kHandleIndexBits);
}

[[nodiscard]] constexpr auto handleIndex(std::uint32_t id) -> std::uint32_t {
  return id & kHandleIndexMask;
}

[[nodiscard]] constexpr auto handleGeneration(std::uint32_t id)
    -> std::uint8_t {
  return static_cast<std::uint8_t>(id >> kHandleIndexBits);
}

using TypeRef = Handle<struct TypeTag>;
using ValueRef = Handle<struct ValueTag>;
using BlockRef = Handle<struct BlockTag>;
using FunctionRef = Handle<struct FunctionTag>;
using GlobalRef = Handle<struct GlobalTag>;
using InsertionPointRef = Handle<struct InsertionPointTag>;
using ModuleRef = Handle<struct ModuleTag>;
using CleanupRegionRef = Handle<struct CleanupRegionTag>;

}  // namespace cxx::ir

template <typename Tag>
struct std::hash<cxx::ir::Handle<Tag>> {
  [[nodiscard]] auto operator()(cxx::ir::Handle<Tag> handle) const noexcept
      -> std::size_t {
    return std::hash<std::uint32_t>{}(cxx::ir::HandleAccess::id(handle));
  }
};
