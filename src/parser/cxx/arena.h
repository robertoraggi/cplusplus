// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/cxx_fwd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

namespace cxx {

class Arena {
  struct Block {
    static const std::size_t SIZE = 16 * 1024;

    char* data;
    char* ptr;

    Block(const Block&) = delete;
    auto operator=(const Block&) -> Block& = delete;

    Block() { ptr = data = static_cast<char*>(std::malloc(SIZE)); }

    ~Block() {
      if (data) std::free(data);
    }
  };

  std::vector<Block*> blocks;

 public:
  Arena(const Arena& other) = delete;
  auto operator=(const Arena& other) -> Arena& = delete;
  Arena() = default;
  ~Arena() { reset(); }

  void reset() {
    for (auto&& block : blocks) delete block;
    blocks.clear();
  }

  auto allocate(std::size_t size) noexcept -> void* {
    if (!blocks.empty()) {
      constexpr auto align = alignof(std::max_align_t) - 1;

      auto block = blocks.back();
      block->ptr = (char*)((std::intptr_t(block->ptr) + align) & ~align);
      void* addr = block->ptr;
      block->ptr += size;
      if (block->ptr < block->data + Block::SIZE) return addr;
    }
    blocks.push_back(::new Block());
    return allocate(size);
  }
};

struct Managed {
  auto operator new(std::size_t size, Arena* arena) noexcept -> void* {
    return arena->allocate(size);
  }
  void operator delete(void* ptr, std::size_t) {}
  void operator delete(void* ptr, Arena*) noexcept {}
};

}  // namespace cxx
