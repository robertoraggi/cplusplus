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

#ifndef CXX_NO_FILESYSTEM
#include <filesystem>

namespace cxx::fs {

using path = std::filesystem::path;

using std::filesystem::current_path;
using std::filesystem::exists;
using std::filesystem::is_symlink;
using std::filesystem::read_symlink;

}  // namespace cxx::fs

#else

#include <string>
#include <tuple>

namespace cxx::fs {

class path {
  std::string path_;

 public:
  path() = default;
  path(std::string p) : path_(std::move(p)) {}

  const std::string& string() const { return path_; }
  operator const std::string&() const { return path_; }

  path& remove_filename();
};

bool exists(const path& path);
path operator/(path lhs, const path& rhs);
path operator/(path lhs, const std::string& rhs);
path current_path();
path absolute(const path& p);

}  // namespace cxx::fs

#endif
