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

#include <cxx/linker/archive.h>
#include <cxx/linker/bytes.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cxx::linker {

struct LinkOptions {
  std::string entry = "_start";
  std::vector<std::string> exports;
  std::uint32_t globalBase = 0;
  std::uint32_t stackSize = 65536;
  std::uint32_t initialMemory = 0;
  std::uint32_t maxMemory = 0;
  bool stackFirst = true;
  bool gcSections = true;
  bool noEntry = false;
  bool allowUndefined = false;
  bool stripDebug = false;
  bool stripAll = false;
};

class Linker {
 public:
  explicit Linker(LinkOptions options);
  Linker(const Linker&) = delete;
  auto operator=(const Linker&) -> Linker& = delete;
  ~Linker();

  [[nodiscard]] auto addObject(std::string name,
                               std::shared_ptr<const Bytes> content,
                               std::string& error) -> bool;

  void addArchive(std::shared_ptr<const Archive> archive);

  [[nodiscard]] auto link(Bytes& output, std::string& error) -> bool;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cxx::linker
