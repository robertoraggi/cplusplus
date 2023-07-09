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

#pragma once

#include <cxx/names_fwd.h>

#include <cstdint>
#include <string>

namespace cxx {

class Name {
 public:
  explicit Name(NameKind kind) : kind_(kind) {}
  virtual ~Name();

  auto kind() const -> NameKind { return kind_; }

 private:
  NameKind kind_;
};

class Identifier : public Name {
  std::string name_;

 public:
  static constexpr auto Kind = NameKind::kIdentifier;

  explicit Identifier(std::string name) : Name(Kind), name_(std::move(name)) {}

  auto isAnonymous() const -> bool { return name_.at(0) == '$'; }

  auto value() const -> const std::string& { return name_; }
  auto name() const -> const std::string& { return name_; }
  auto length() const -> std::uint32_t { return name_.size(); }
};

class OperatorId : public Name {
  std::string name_;

 public:
  static constexpr auto Kind = NameKind::kOperatorId;

  explicit OperatorId(std::string name) : Name(Kind), name_(std::move(name)) {}

  auto name() const -> const std::string& { return name_; }
  auto length() const -> std::uint32_t { return name_.size(); }
};

class DestructorId : public Name {
  std::string name_;

 public:
  static constexpr auto Kind = NameKind::kDestructorId;

  explicit DestructorId(std::string name)
      : Name(Kind), name_(std::move(name)) {}

  auto name() const -> const std::string& { return name_; }
  auto length() const -> std::uint32_t { return name_.size(); }
};

template <typename T>
auto name_cast(const Name* name) -> const T* {
  return name && name->kind() == T::Kind ? static_cast<const T*>(name)
                                         : nullptr;
}

}  // namespace cxx