// Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TYPES_H
#define TYPES_H

#include <tuple>
#include <string>
#include <cassert>

// ### move
class Symbol;
class FunctionSymbol;
class ClassSymbol;
class Name;

#define FOR_EACH_SINGLETON_TYPE(V) \
  V(Undefined) \
  V(Auto) \
  V(Void) \
  V(Nullptr)

#define FOR_EACH_OTHER_TYPE(V) \
  V(Integer) \
  V(Float) \
  V(Pointer) \
  V(LValueReference) \
  V(RValueReference) \
  V(BoundedArray) \
  V(UnboundedArray) \
  V(Function) \
  V(Class) \
  V(Named)

#define FOR_EACH_TYPE(V) \
  FOR_EACH_SINGLETON_TYPE(V) \
  FOR_EACH_OTHER_TYPE(V)

#define VISIT_TYPE(T) class T##Type;
  FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE

enum struct TypeKind {
#define VISIT_TYPE(T) k##T,
  FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE
};

#define FOR_EACH_INTEGER_TYPE(V) \
  V(SignedChar, "signed char") \
  V(ShortInt, "short int") \
  V(Int, "int") \
  V(LongInt, "long int") \
  V(LongLongInt, "long long int") \
  V(UnsignedChar, "unsigned char") \
  V(UnsignedShortInt, "unsigned short int") \
  V(UnsignedInt, "unsigned int") \
  V(UnsignedLongInt, "unsigned long int") \
  V(UnsignedLongLongInt, "unsigned long long int") \
  V(WCharT, "wchar_t") \
  V(Char, "char") \
  V(Char16T, "char16_t") \
  V(Char32T, "char32_t") \
  V(Bool, "bool")

#define FOR_EACH_FLOAT_TYPE(V) \
  V(Float, "float") \
  V(Double, "double") \
  V(LongDouble, "long double")

enum struct IntegerKind {
#define VISIT_INTEGER_TYPE(T,N) k##T,
  FOR_EACH_INTEGER_TYPE(VISIT_INTEGER_TYPE)
#undef VISIT_INTEGER_TYPE
};

enum struct FloatKind {
#define VISIT_FLOAT_TYPE(T,N) k##T,
  FOR_EACH_FLOAT_TYPE(VISIT_FLOAT_TYPE)
#undef VISIT_FLOAT_TYPE
};

class Type;
class ArrayType;
class ReferenceType;

class QualType {
  const Type* _type;
  union {
    unsigned _flags;
    struct {
      unsigned _isConst: 1;
      unsigned _isVolatile: 1;
      unsigned _isUnsigned: 1;
    };
  };
public:
  explicit QualType(const Type* type = 0);
  void setType(const Type* type) { assert(type); _type = type; }
  const Type* operator->() const { return _type; }
  const Type* operator*() const { return _type; }
  bool isConst() const { return _isConst; }
  void setConst(bool isConst) { _isConst = isConst; }
  bool isVolatile() const{ return _isVolatile; }
  void setVolatile(bool isVolatile) { _isVolatile = isVolatile; }
  bool isUnsigned() const { return _isUnsigned; }
  void setUnsigned(bool isUnsigned) { _isUnsigned = isUnsigned; }
  explicit operator bool() const;
  inline bool operator<(const QualType& other) const {
    if (_type == other._type)
      return _flags < other._flags;
    return _type < other._type;
  }
  inline bool operator==(const QualType& other) const {
    return _type == other._type && _flags == other._flags;
  }
  inline bool operator!=(const QualType& other) const {
    return !operator==(other);
  }
};

class Type {
  TypeKind _kind;
public:
  explicit Type(TypeKind kind): _kind(kind) {}
  virtual ~Type() = default;

  inline TypeKind kind() const { return _kind; }

#define VISIT_TYPE(T) \
  inline bool is##T##Type() const { \
    return _kind == TypeKind::k##T; \
  } \
  inline const T##Type* as##T##Type() const { \
    return is##T##Type() ? reinterpret_cast<const T##Type*>(this) : nullptr; \
  }
  FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE

  virtual const ArrayType* asArrayType() const { return 0; }
  virtual const ReferenceType* asReferenceType() const { return 0; }  
};

class ReferenceType: public Type {
public:
  using Type::Type;
  const ReferenceType* asReferenceType() const override { return this; }
};

class ArrayType: public Type {
public:
  using Type::Type;
  const ArrayType* asArrayType() const override { return this; }
};

template <TypeKind K, typename Base = Type>
struct ExtendsType: Base {
  inline ExtendsType(): Base(K) {}
};

class UndefinedType final: public ExtendsType<TypeKind::kUndefined> {
public:
  static const UndefinedType* get() {
    static UndefinedType u;
    return &u;
  }
};

class VoidType final: public ExtendsType<TypeKind::kVoid> {
public:
  static const VoidType* get() {
    static VoidType u;
    return &u;
  }
};

class AutoType final: public ExtendsType<TypeKind::kAuto> {
public:
  static const AutoType* get() {
    static AutoType u;
    return &u;
  }
};

class NullptrType final: public ExtendsType<TypeKind::kNullptr> {
public:
  static const NullptrType* get() {
    static NullptrType u;
    return &u;
  }
};

class IntegerType final: public ExtendsType<TypeKind::kInteger>, public std::tuple<IntegerKind> {
public:
  using tuple::tuple;
  inline IntegerKind integerKind() const { return std::get<0>(*this); }
#define VISIT_INTEGER_TYPE(T,N) inline bool is##T() const { return integerKind() == IntegerKind::k##T; }
  FOR_EACH_INTEGER_TYPE(VISIT_INTEGER_TYPE)
#undef VISIT_INTEGER_TYPE
};

class FloatType final: public ExtendsType<TypeKind::kFloat>, public std::tuple<FloatKind> {
public:
  using tuple::tuple;
  inline FloatKind floatKind() const { return std::get<0>(*this); }
#define VISIT_FLOAT_TYPE(T,N) inline bool is##T() const { return floatKind() == FloatKind::k##T; }
  FOR_EACH_FLOAT_TYPE(VISIT_FLOAT_TYPE)
#undef VISIT_FLOAT_TYPE
};

class PointerType final: public ExtendsType<TypeKind::kPointer>,
                         public std::tuple<QualType> {
public:
  using tuple::tuple;
  QualType elementType() const { return std::get<0>(*this); }
};

class LValueReferenceType final: public ExtendsType<TypeKind::kLValueReference, ReferenceType>,
                                 public std::tuple<QualType> {
public:
  using tuple::tuple;
  QualType elementType() const { return std::get<0>(*this); }
};

class RValueReferenceType final: public ExtendsType<TypeKind::kRValueReference, ReferenceType>,
                                 public std::tuple<QualType> {
public:
  using tuple::tuple;
  QualType elementType() const { return std::get<0>(*this); }
};

class BoundedArrayType final: public ExtendsType<TypeKind::kBoundedArray, ArrayType>,
                              public std::tuple<QualType, size_t> {
public:
  using tuple::tuple;
  QualType elementType() const { return std::get<0>(*this); }
  size_t size() const { return std::get<1>(*this); }
};

class UnboundedArrayType final: public ExtendsType<TypeKind::kUnboundedArray, ArrayType>,
                                public std::tuple<QualType> {
public:
  using tuple::tuple;
  QualType elementType() const { return std::get<0>(*this); }
};

class FunctionType final: public ExtendsType<TypeKind::kFunction>, public std::tuple<FunctionSymbol*> {
public:
  using tuple::tuple;
  FunctionSymbol* symbol() const { return std::get<0>(*this); }
  QualType returnType() const;
  unsigned argumentCount() const;
  QualType argumentAt(unsigned index) const;
  bool isVariadic() const;
};

class ClassType final: public ExtendsType<TypeKind::kClass>, public std::tuple<ClassSymbol*> {
public:
  using tuple::tuple;
  ClassSymbol* symbol() const { return std::get<0>(*this); }
};

class NamedType final: public ExtendsType<TypeKind::kNamed>, public std::tuple<const Name*> {
public:
  using tuple::tuple;
  const Name* name() const { return std::get<0>(*this); }
};

//
// implementation
//
inline QualType::QualType(const Type* type)
  : _type(type ? type : UndefinedType::get())
  , _flags(0) {
}

inline QualType::operator bool() const {
  assert(_type != 0);
  return ! _type->asUndefinedType();
}

class TypeToString {
public:
  std::string operator()(QualType type, const Name* name = 0);
  std::string operator()(QualType type, std::string decl) { return print(type, std::move(decl)); }
private:
  std::string text;
  std::string decl;

#define VISIT_TYPE(T) void visit(const T##Type*);
  FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE

  void accept(QualType type);

  std::string print(QualType type, std::string&& decl);
  std::string print(QualType type, const Name* name);
};


#endif // TYPES_H
