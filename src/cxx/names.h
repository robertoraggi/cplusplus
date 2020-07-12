// Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <string>
#include <tuple>
#include <vector>

#include "types.h"

namespace cxx {

class Name {
  NameKind _kind;

 public:
  Name(NameKind kind) : _kind(kind) {}
  virtual ~Name() = default;

  inline NameKind kind() const { return _kind; }

#define VISIT_NAME(T)                                            \
  inline bool is##T() const { return _kind == NameKind::k##T; }  \
  inline const T* as##T() const {                                \
    return is##T() ? reinterpret_cast<const T*>(this) : nullptr; \
  }
  FOR_EACH_NAME(VISIT_NAME)
#undef VISIT_NAME

  virtual std::string toString() const = 0;
};

template <NameKind K, typename Base = Name>
struct ExtendsName : Base {
  inline ExtendsName() : Base(K) {}
};

class Identifier final : public ExtendsName<NameKind::kIdentifier>,
                         public std::tuple<std::string> {
 public:
  using tuple::tuple;
  inline const std::string& string() const { return std::get<0>(*this); }
  inline const char* c_str() const { return string().c_str(); }
  inline unsigned size() const { return unsigned(string().size()); }
  std::string toString() const override;
};

class OperatorName final : public ExtendsName<NameKind::kOperatorName>,
                           public std::tuple<TokenKind> {
 public:
  using tuple::tuple;
  inline TokenKind op() const { return std::get<0>(*this); }
  std::string toString() const override;
};

class ConversionName final : public ExtendsName<NameKind::kConversionName>,
                             public std::tuple<QualType> {
 public:
  using tuple::tuple;
  inline QualType type() const { return std::get<0>(*this); }
  std::string toString() const override;
};

class DestructorName final : public ExtendsName<NameKind::kDestructorName>,
                             public std::tuple<const Name*> {
 public:
  using tuple::tuple;
  inline const Name* name() const { return std::get<0>(*this); }
  std::string toString() const override;
};

class QualifiedName final : public ExtendsName<NameKind::kQualifiedName>,
                            public std::tuple<const Name*, const Name*> {
 public:
  using tuple::tuple;
  inline const Name* base() const { return std::get<0>(*this); }
  inline const Name* name() const { return std::get<1>(*this); }
  std::string toString() const override;
};

class TemplateName final
    : public ExtendsName<NameKind::kTemplateName>,
      public std::tuple<const Name*, std::vector<QualType>> {
 public:
  using tuple::tuple;
  inline const Name* name() const { return std::get<0>(*this); }
  inline const std::vector<QualType>& argumentTypes() const {
    return std::get<1>(*this);
  }
  std::string toString() const override;
};

class DecltypeName final : public ExtendsName<NameKind::kDecltypeName>,
                           public std::tuple<QualType> {
 public:
  using tuple::tuple;
  inline QualType type() const { return std::get<0>(*this); }
  std::string toString() const override;
};

}  // namespace cxx
