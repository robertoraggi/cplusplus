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

#include <cxx/toolchain.h>

// cxx
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>

namespace cxx {

Toolchain::Toolchain(Preprocessor* preprocessor) : preprocessor_(preprocessor) {
  setMemoryLayout(std::make_unique<MemoryLayout>(64));
}

Toolchain::~Toolchain() = default;

auto Toolchain::language() const -> LanguageKind {
  return preprocessor_->language();
}

void Toolchain::setMemoryLayout(std::unique_ptr<MemoryLayout> memoryLayout) {
  memoryLayout_ = std::move(memoryLayout);
}

void Toolchain::defineMacro(const std::string& name,
                            const std::string& definition) {
  preprocessor_->defineMacro(name, definition);
}

void Toolchain::addSystemIncludePath(std::string path) {
  preprocessor_->addSystemIncludePath(std::move(path));
}

}  // namespace cxx
