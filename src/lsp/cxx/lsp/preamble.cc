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

#include "preamble.h"

#include <cxx/lexer.h>

namespace cxx::lsp {

namespace {

[[nodiscard]] auto isSplicedNewline(std::string_view source,
                                    std::size_t newline) -> bool {
  auto index = newline;
  if (index > 0 && source[index - 1] == '\r') --index;
  return index > 0 && source[index - 1] == '\\';
}

[[nodiscard]] auto endOfDirectiveLine(std::string_view source,
                                      std::size_t offset) -> std::size_t {
  if (offset == 0 || offset >= source.size()) return offset;
  if (source[offset - 1] == '\n') return offset;
  while (offset < source.size()) {
    const auto newline = source.find('\n', offset);
    if (newline == std::string_view::npos) break;
    if (!isSplicedNewline(source, newline)) return newline + 1;
    offset = newline + 1;
  }
  return source.size();
}

}  // namespace

void maskPreamble(std::string& source, std::size_t size) {
  const auto end = endOfDirectiveLine(source, size);
  for (std::size_t i = 0; i < end; ++i) {
    if (source[i] == '\n') continue;
    if (source[i] == '\r') continue;
    source[i] = ' ';
  }
}

auto preambleKeys() -> PrecompiledHeaderKeys {
  return {.serializationAbi = precompiledHeaderSerializationAbi(),
          .targetKey = "lsp-session",
          .languageKey = "lsp-session",
          .optionDigest = "lsp-session"};
}

auto PreambleCache::get(std::string_view source) const
    -> std::shared_ptr<const Preamble> {
#ifndef CXX_NO_THREADS
  auto lock = std::lock_guard(mutex_);
#endif
  if (!preamble_) return {};
  if (!source.starts_with(preamble_->source)) return {};
  auto suffix = source.substr(preamble_->source.size());
  if (!suffix.empty()) {
    Lexer lexer(suffix);
    lexer.setPreprocessing(true);
    if (lexer.next() == TokenKind::T_HASH) return {};
  }
  return preamble_;
}

void PreambleCache::put(std::shared_ptr<const Preamble> preamble) {
#ifndef CXX_NO_THREADS
  auto lock = std::lock_guard(mutex_);
#endif
  preamble_ = std::move(preamble);
}

}  // namespace cxx::lsp
