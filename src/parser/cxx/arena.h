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

#include <cxx/cxx_fwd.h>

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <memory_resource>
#include <type_traits>
#include <utility>
#include <vector>

namespace cxx {

class Arena : public std::pmr::memory_resource {
 public:
  struct Mark {
    std::size_t chunk = 0;
    std::size_t offset = 0;
    std::size_t cleanupCount = 0;
  };

  Arena() = default;

  Arena(const Arena&) = delete;
  auto operator=(const Arena&) -> Arena& = delete;

  ~Arena() override {
    runCleanups(0);
    for (auto& chunk : chunks_) std::free(chunk.data);
  }

  template <typename T, typename... Args>
  [[nodiscard]] auto make(Args&&... args) -> T* {
    auto object = static_cast<T*>(allocate(sizeof(T), alignof(T)));
    std::construct_at(object, std::forward<Args>(args)...);
    if constexpr (!std::is_trivially_destructible_v<T>) {
      cleanups_.push_back(
          {[](void* ptr) { std::destroy_at(static_cast<T*>(ptr)); }, object});
    }
    return object;
  }

  /**
   * A bump allocation without the virtual dispatch of `memory_resource`.
   * Over-aligned and chunk-crossing requests fall back to `do_allocate`.
   */
  [[nodiscard]] auto allocate(std::size_t bytes,
                              std::size_t alignment = alignof(std::max_align_t))
      -> void* {
    if (alignment <= alignof(std::max_align_t) &&
        currentChunk_ < chunks_.size()) {
      const auto& chunk = chunks_[currentChunk_];
      const auto aligned = (currentOffset_ + alignment - 1) & ~(alignment - 1);
      if (aligned <= chunk.size && bytes <= chunk.size - aligned) {
        currentOffset_ = aligned + bytes;
        return chunk.data + aligned;
      }
    }
    return do_allocate(bytes, alignment);
  }

  [[nodiscard]] auto mark() const -> Mark {
    return Mark{currentChunk_, currentOffset_, cleanups_.size()};
  }

  void rewind(const Mark& mark) {
    runCleanups(mark.cleanupCount);
    currentChunk_ = mark.chunk;
    currentOffset_ = mark.offset;
  }

 protected:
  auto do_allocate(std::size_t bytes, std::size_t alignment) -> void* override {
    while (currentChunk_ < chunks_.size()) {
      auto& chunk = chunks_[currentChunk_];
      auto* address = static_cast<void*>(chunk.data + currentOffset_);
      auto space = chunk.size - currentOffset_;
      if (std::align(alignment, bytes, address, space)) {
        const auto aligned = static_cast<char*>(address) - chunk.data;
        currentOffset_ = aligned + bytes;
        return address;
      }
      ++currentChunk_;
      currentOffset_ = 0;
    }

    const auto maxSize = std::numeric_limits<std::size_t>::max();
    if (bytes > maxSize - alignment) cxx_runtime_error("out of memory");
    allocateChunk(bytes + alignment);

    auto& chunk = chunks_[currentChunk_];
    auto* address = static_cast<void*>(chunk.data);
    auto space = chunk.size;
    if (!std::align(alignment, bytes, address, space)) {
      cxx_runtime_error("out of memory");
    }
    const auto aligned = static_cast<char*>(address) - chunk.data;
    currentOffset_ = aligned + bytes;
    return address;
  }

  void do_deallocate(void*, std::size_t, std::size_t) override {}

  [[nodiscard]] auto do_is_equal(
      const std::pmr::memory_resource& other) const noexcept -> bool override {
    return this == &other;
  }

 private:
  struct Chunk {
    char* data = nullptr;
    std::size_t size = 0;
  };

  struct Cleanup {
    void (*destroy)(void*);
    void* object;
  };

  static constexpr std::size_t kInitialChunkSize = 64 * 1024;

  void allocateChunk(std::size_t leastSize) {
    const auto maxSize = std::numeric_limits<std::size_t>::max();
    auto size = kInitialChunkSize;
    if (!chunks_.empty()) {
      const auto previousSize = chunks_.back().size;
      if (previousSize > maxSize / 2) {
        size = maxSize;
      } else {
        size = previousSize * 2;
      }
    }

    while (size < leastSize) {
      if (size > maxSize / 2) {
        size = leastSize;
        break;
      }
      size *= 2;
    }

    auto data = static_cast<char*>(std::malloc(size));
    if (!data) cxx_runtime_error("out of memory");

    chunks_.push_back({data, size});
    currentChunk_ = chunks_.size() - 1;
    currentOffset_ = 0;
  }

  void runCleanups(std::size_t count) {
    while (cleanups_.size() > count) {
      auto cleanup = cleanups_.back();
      cleanups_.pop_back();
      cleanup.destroy(cleanup.object);
    }
  }

  std::vector<Chunk> chunks_;
  std::vector<Cleanup> cleanups_;
  std::size_t currentChunk_ = 0;
  std::size_t currentOffset_ = 0;
};

struct Managed {
  auto operator new(std::size_t size, Arena* arena) noexcept -> void* {
    return arena->allocate(size);
  }
  void operator delete(void* ptr, std::size_t) {}
  void operator delete(void* ptr, Arena*) noexcept {}
};

}  // namespace cxx
