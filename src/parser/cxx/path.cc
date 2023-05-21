// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/private/path.h>

#ifndef CXX_NO_FILESYSTEM

namespace cxx::fs {}

#else

#include <unistd.h>

namespace cxx::fs {

path& path::remove_filename() {
  auto pos = path_.find_last_of('/');
  if (pos != std::string::npos) path_.resize(pos);
  return *this;
}

bool exists(const path& p) {
  const auto& fn = p.string();
  return access(fn.c_str(), F_OK) == 0;
}

path operator/(path lhs, const path& rhs) {
  auto sep = lhs.string().back() == '/' ? "" : "/";
  return path(lhs.string() + "/" + rhs.string());
}

path operator/(path lhs, const std::string& rhs) {
  if (lhs.string().ends_with('/')) return path(lhs.string() + rhs);
  return path(lhs.string() + "/" + rhs);
}

path current_path() { return {}; }

}  // namespace cxx::fs

#endif
