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

#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#ifdef _WIN32
#include <io.h>
#endif

namespace cxx::detail {

[[nodiscard]] inline auto readAll(std::FILE* file)
    -> std::optional<std::string> {
#ifdef _WIN32
  struct _stat64 info;
  const auto status = ::_fstat64(::_fileno(file), &info);
  const bool regular = status == 0 && (info.st_mode & _S_IFMT) == _S_IFREG;
  const auto position = regular ? ::_ftelli64(file) : -1;
#else
  struct stat info;
  const auto status = ::fstat(::fileno(file), &info);
  const bool regular = status == 0 && S_ISREG(info.st_mode);
  const auto position = regular ? ::ftello(file) : -1;
#endif
  std::string text;
  if (regular && position >= 0 && info.st_size >= position) {
    const auto size = static_cast<std::uintmax_t>(info.st_size - position);
    if (size > text.max_size()) return std::nullopt;
    text.resize(static_cast<std::size_t>(size));
    const auto count = std::fread(text.data(), 1, text.size(), file);
    if (std::ferror(file)) return std::nullopt;
    text.resize(count);
    return text;
  }
  for (;;) {
    auto offset = text.size();
    text.resize(offset + 4096);
    auto count = std::fread(text.data() + offset, 1, 4096, file);
    text.resize(offset + count);
    if (count == 4096) continue;
    if (std::ferror(file)) return std::nullopt;
    return text;
  }
}

}  // namespace cxx::detail
