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

#ifndef TRANSLATIONUNIT_H
#define TRANSLATIONUNIT_H

#include "Token.h"
#include "Types.h"
#include <string>
#include <vector>

class TranslationUnit {
  Control* control_;
  std::vector<Token> tokens_;
  std::vector<int> lines_;
  int yychar{'\n'};
  int yypos{-1};
  std::string yyfilename;
  std::string yytext;
  std::string yycode;
  const char* yyptr{0};

public:
  TranslationUnit(Control* control): control_(control) {}
  ~TranslationUnit() = default;
  Control* control() const { return control_; }

  const std::string& fileName() const { return yyfilename; }
  template <typename T>
  void setFileName(T&& fileName) { yyfilename = std::forward<T>(fileName); }

  const std::string& source() const { return yycode; }
  template <typename T>
  void setSource(T&& source) {
    yycode = std::forward<T>(source);
    yyptr = yycode.c_str();
  }

  void warning(unsigned index, const char* format...);
  void error(unsigned index, const char* format...);
  void fatal(unsigned index, const char* format...);

  // tokens
  inline unsigned tokenCount() const { return tokens_.size(); }
  inline const Token& tokenAt(unsigned index) const { return tokens_[index]; }
  inline TokenKind tokenKind(unsigned index) const { return tokens_[index].kind(); }
  int tokenLength(unsigned index) const;
  const char* tokenText(unsigned index) const;
  const Identifier* identifier(unsigned index) const;
  void getTokenStartPosition(unsigned index, unsigned* line, unsigned* column) const;

  // front end
  void tokenize();
  bool parse();

  struct Specs {
    Specs() = default;
    QualType type;
    union {
      unsigned _flags{0};
      unsigned isExtern: 1;
      unsigned isInline: 1;
      unsigned isStatic: 1;
      unsigned isTypedef: 1;
      unsigned isVirtual: 1;
      unsigned isFriend: 1;
      unsigned isExplicit: 1;
      unsigned isMutable: 1;
      unsigned isConstexpr: 1;
    };
  };

  struct Decl {
    Specs specs;
    const Name* name{0};
    const Type* operator->() const { return *specs.type; }
    void setType(const Type* type) {
      specs.type.setType(type);
    }
    void setUnsigned(bool isUnsigned) {
      specs.type.setUnsigned(isUnsigned);
    }
    void setConst(bool isConst) {
      specs.type.setConst(isConst);
    }
    void setVolatile(bool isVolatile) {
      specs.type.setVolatile(isVolatile);
    }
  };

  const Name* name(NameAST* ast);
  Specs specifiers(List<SpecifierAST*>* specifiers);
  Decl declarator(const Specs& specs, DeclaratorAST* decl);
  QualType finish(QualType type);

private:
  void yyinp();
  TokenKind yylex(unsigned* offset, const void** priv);
};

#endif // TRANSLATIONUNIT_H
