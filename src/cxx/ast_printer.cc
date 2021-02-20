// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast.h>
#include <cxx/ast_printer.h>
#include <cxx/translation_unit.h>

#if defined(__has_include)
#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define WITH_CXXABI
#endif
#endif

namespace cxx {

nlohmann::json ASTPrinter::accept(AST* ast) {
  nlohmann::json json;
  if (ast) {
    std::swap(json_, json);
    ast->accept(this);
    std::swap(json_, json);
  }
  return json;
}

void ASTPrinter::visit(TypeIdAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["typeSpecifierList"] = elements;
  }

  if (ast->declarator) {
    json_["declarator"] = accept(ast->declarator);
  }
}

void ASTPrinter::visit(NestedNameSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->nameList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->nameList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["nameList"] = elements;
  }
}

void ASTPrinter::visit(UsingDeclaratorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->nestedNameSpecifier) {
    json_["nestedNameSpecifier"] = accept(ast->nestedNameSpecifier);
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }
}

void ASTPrinter::visit(HandlerAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->exceptionDeclaration) {
    json_["exceptionDeclaration"] = accept(ast->exceptionDeclaration);
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }
}

void ASTPrinter::visit(TemplateArgumentAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(EnumBaseAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["typeSpecifierList"] = elements;
  }
}

void ASTPrinter::visit(EnumeratorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(DeclaratorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->ptrOpList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->ptrOpList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["ptrOpList"] = elements;
  }

  if (ast->coreDeclarator) {
    json_["coreDeclarator"] = accept(ast->coreDeclarator);
  }

  if (ast->modifiers) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->modifiers; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["modifiers"] = elements;
  }
}

void ASTPrinter::visit(BaseSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }
}

void ASTPrinter::visit(BaseClauseAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->baseSpecifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->baseSpecifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["baseSpecifierList"] = elements;
  }
}

void ASTPrinter::visit(NewTypeIdAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["typeSpecifierList"] = elements;
  }
}

void ASTPrinter::visit(ParameterDeclarationClauseAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->parameterDeclarationList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->parameterDeclarationList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["parameterDeclarationList"] = elements;
  }
}

void ASTPrinter::visit(ParametersAndQualifiersAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->parameterDeclarationClause) {
    json_["parameterDeclarationClause"] =
        accept(ast->parameterDeclarationClause);
  }

  if (ast->cvQualifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["cvQualifierList"] = elements;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }
}

void ASTPrinter::visit(LambdaIntroducerAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(LambdaDeclaratorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->parameterDeclarationClause) {
    json_["parameterDeclarationClause"] =
        accept(ast->parameterDeclarationClause);
  }

  if (ast->declSpecifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["declSpecifierList"] = elements;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->trailingReturnType) {
    json_["trailingReturnType"] = accept(ast->trailingReturnType);
  }
}

void ASTPrinter::visit(TrailingReturnTypeAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->typeId) {
    json_["typeId"] = accept(ast->typeId);
  }
}

void ASTPrinter::visit(EqualInitializerAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(BracedInitListAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->expressionList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["expressionList"] = elements;
  }
}

void ASTPrinter::visit(ParenInitializerAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->expressionList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["expressionList"] = elements;
  }
}

void ASTPrinter::visit(NewParenInitializerAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->expressionList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["expressionList"] = elements;
  }
}

void ASTPrinter::visit(NewBracedInitializerAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->bracedInit) {
    json_["bracedInit"] = accept(ast->bracedInit);
  }
}

void ASTPrinter::visit(EllipsisExceptionDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(TypeExceptionDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["typeSpecifierList"] = elements;
  }

  if (ast->declarator) {
    json_["declarator"] = accept(ast->declarator);
  }
}

void ASTPrinter::visit(TranslationUnitAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->declarationList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["declarationList"] = elements;
  }
}

void ASTPrinter::visit(ModuleUnitAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ThisExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(CharLiteralExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(BoolLiteralExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(IntLiteralExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(FloatLiteralExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(NullptrLiteralExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(StringLiteralExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(UserDefinedStringLiteralExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(IdExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }
}

void ASTPrinter::visit(NestedExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(LambdaExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->lambdaIntroducer) {
    json_["lambdaIntroducer"] = accept(ast->lambdaIntroducer);
  }

  if (ast->templateParameterList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->templateParameterList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["templateParameterList"] = elements;
  }

  if (ast->lambdaDeclarator) {
    json_["lambdaDeclarator"] = accept(ast->lambdaDeclarator);
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }
}

void ASTPrinter::visit(UnaryExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(BinaryExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->leftExpression) {
    json_["leftExpression"] = accept(ast->leftExpression);
  }

  if (ast->rightExpression) {
    json_["rightExpression"] = accept(ast->rightExpression);
  }
}

void ASTPrinter::visit(AssignmentExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->leftExpression) {
    json_["leftExpression"] = accept(ast->leftExpression);
  }

  if (ast->rightExpression) {
    json_["rightExpression"] = accept(ast->rightExpression);
  }
}

void ASTPrinter::visit(CallExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->baseExpression) {
    json_["baseExpression"] = accept(ast->baseExpression);
  }

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->expressionList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["expressionList"] = elements;
  }
}

void ASTPrinter::visit(SubscriptExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->baseExpression) {
    json_["baseExpression"] = accept(ast->baseExpression);
  }

  if (ast->indexExpression) {
    json_["indexExpression"] = accept(ast->indexExpression);
  }
}

void ASTPrinter::visit(MemberExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->baseExpression) {
    json_["baseExpression"] = accept(ast->baseExpression);
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }
}

void ASTPrinter::visit(ConditionalExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->condition) {
    json_["condition"] = accept(ast->condition);
  }

  if (ast->iftrueExpression) {
    json_["iftrueExpression"] = accept(ast->iftrueExpression);
  }

  if (ast->iffalseExpression) {
    json_["iffalseExpression"] = accept(ast->iffalseExpression);
  }
}

void ASTPrinter::visit(CppCastExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->typeId) {
    json_["typeId"] = accept(ast->typeId);
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(NewExpressionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->typeId) {
    json_["typeId"] = accept(ast->typeId);
  }

  if (ast->newInitalizer) {
    json_["newInitalizer"] = accept(ast->newInitalizer);
  }
}

void ASTPrinter::visit(LabeledStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }
}

void ASTPrinter::visit(CaseStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }
}

void ASTPrinter::visit(DefaultStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }
}

void ASTPrinter::visit(ExpressionStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(CompoundStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->statementList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->statementList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["statementList"] = elements;
  }
}

void ASTPrinter::visit(IfStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->initializer) {
    json_["initializer"] = accept(ast->initializer);
  }

  if (ast->condition) {
    json_["condition"] = accept(ast->condition);
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }

  if (ast->elseStatement) {
    json_["elseStatement"] = accept(ast->elseStatement);
  }
}

void ASTPrinter::visit(SwitchStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->initializer) {
    json_["initializer"] = accept(ast->initializer);
  }

  if (ast->condition) {
    json_["condition"] = accept(ast->condition);
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }
}

void ASTPrinter::visit(WhileStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->condition) {
    json_["condition"] = accept(ast->condition);
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }
}

void ASTPrinter::visit(DoStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(ForRangeStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->initializer) {
    json_["initializer"] = accept(ast->initializer);
  }

  if (ast->rangeDeclaration) {
    json_["rangeDeclaration"] = accept(ast->rangeDeclaration);
  }

  if (ast->rangeInitializer) {
    json_["rangeInitializer"] = accept(ast->rangeInitializer);
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }
}

void ASTPrinter::visit(ForStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->initializer) {
    json_["initializer"] = accept(ast->initializer);
  }

  if (ast->condition) {
    json_["condition"] = accept(ast->condition);
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }
}

void ASTPrinter::visit(BreakStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ContinueStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ReturnStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(GotoStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(CoroutineReturnStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(DeclarationStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->declaration) {
    json_["declaration"] = accept(ast->declaration);
  }
}

void ASTPrinter::visit(TryBlockStatementAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->statement) {
    json_["statement"] = accept(ast->statement);
  }

  if (ast->handlerList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->handlerList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["handlerList"] = elements;
  }
}

void ASTPrinter::visit(FunctionDefinitionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->declSpecifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["declSpecifierList"] = elements;
  }

  if (ast->declarator) {
    json_["declarator"] = accept(ast->declarator);
  }

  if (ast->functionBody) {
    json_["functionBody"] = accept(ast->functionBody);
  }
}

void ASTPrinter::visit(ConceptDefinitionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(ForRangeDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(AliasDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->typeId) {
    json_["typeId"] = accept(ast->typeId);
  }
}

void ASTPrinter::visit(SimpleDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributes) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributes; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributes"] = elements;
  }

  if (ast->declSpecifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["declSpecifierList"] = elements;
  }

  if (ast->declaratorList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->declaratorList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["declaratorList"] = elements;
  }
}

void ASTPrinter::visit(StaticAssertDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(EmptyDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(AttributeDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }
}

void ASTPrinter::visit(OpaqueEnumDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->nestedNameSpecifier) {
    json_["nestedNameSpecifier"] = accept(ast->nestedNameSpecifier);
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }

  if (ast->enumBase) {
    json_["enumBase"] = accept(ast->enumBase);
  }
}

void ASTPrinter::visit(UsingEnumDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(NamespaceDefinitionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->nestedNameSpecifier) {
    json_["nestedNameSpecifier"] = accept(ast->nestedNameSpecifier);
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }

  if (ast->extraAttributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->extraAttributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["extraAttributeList"] = elements;
  }

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->declarationList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["declarationList"] = elements;
  }
}

void ASTPrinter::visit(NamespaceAliasDefinitionAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->nestedNameSpecifier) {
    json_["nestedNameSpecifier"] = accept(ast->nestedNameSpecifier);
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }
}

void ASTPrinter::visit(UsingDirectiveAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(UsingDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->usingDeclaratorList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->usingDeclaratorList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["usingDeclaratorList"] = elements;
  }
}

void ASTPrinter::visit(AsmDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }
}

void ASTPrinter::visit(ExportDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ModuleImportDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(TemplateDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->templateParameterList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->templateParameterList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["templateParameterList"] = elements;
  }

  if (ast->declaration) {
    json_["declaration"] = accept(ast->declaration);
  }
}

void ASTPrinter::visit(DeductionGuideAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ExplicitInstantiationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->declaration) {
    json_["declaration"] = accept(ast->declaration);
  }
}

void ASTPrinter::visit(ParameterDeclarationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["typeSpecifierList"] = elements;
  }

  if (ast->declarator) {
    json_["declarator"] = accept(ast->declarator);
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(LinkageSpecificationAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->declarationList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["declarationList"] = elements;
  }
}

void ASTPrinter::visit(SimpleNameAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(DestructorNameAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }
}

void ASTPrinter::visit(DecltypeNameAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->decltypeSpecifier) {
    json_["decltypeSpecifier"] = accept(ast->decltypeSpecifier);
  }
}

void ASTPrinter::visit(OperatorNameAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(TemplateNameAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }

  if (ast->templateArgumentList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->templateArgumentList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["templateArgumentList"] = elements;
  }
}

void ASTPrinter::visit(QualifiedNameAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->nestedNameSpecifier) {
    json_["nestedNameSpecifier"] = accept(ast->nestedNameSpecifier);
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }
}

void ASTPrinter::visit(TypedefSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(FriendSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ConstevalSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ConstinitSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ConstexprSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(InlineSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(StaticSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ExternSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ThreadLocalSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ThreadSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(MutableSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(SimpleSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ExplicitSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(NamedTypeSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }
}

void ASTPrinter::visit(UnderlyingTypeSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(AtomicTypeSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(ElaboratedTypeSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(DecltypeAutoSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(DecltypeSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }
}

void ASTPrinter::visit(PlaceholderTypeSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(CvQualifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(EnumSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->nestedNameSpecifier) {
    json_["nestedNameSpecifier"] = accept(ast->nestedNameSpecifier);
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }

  if (ast->enumBase) {
    json_["enumBase"] = accept(ast->enumBase);
  }

  if (ast->enumeratorList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->enumeratorList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["enumeratorList"] = elements;
  }
}

void ASTPrinter::visit(ClassSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }

  if (ast->baseClause) {
    json_["baseClause"] = accept(ast->baseClause);
  }

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->declarationList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["declarationList"] = elements;
  }
}

void ASTPrinter::visit(TypenameSpecifierAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }
}

void ASTPrinter::visit(IdDeclaratorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->name) {
    json_["name"] = accept(ast->name);
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }
}

void ASTPrinter::visit(NestedDeclaratorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->declarator) {
    json_["declarator"] = accept(ast->declarator);
  }
}

void ASTPrinter::visit(PointerOperatorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->cvQualifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["cvQualifierList"] = elements;
  }
}

void ASTPrinter::visit(ReferenceOperatorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }
}

void ASTPrinter::visit(PtrToMemberOperatorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->nestedNameSpecifier) {
    json_["nestedNameSpecifier"] = accept(ast->nestedNameSpecifier);
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }

  if (ast->cvQualifierList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["cvQualifierList"] = elements;
  }
}

void ASTPrinter::visit(FunctionDeclaratorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->parametersAndQualifiers) {
    json_["parametersAndQualifiers"] = accept(ast->parametersAndQualifiers);
  }

  if (ast->trailingReturnType) {
    json_["trailingReturnType"] = accept(ast->trailingReturnType);
  }
}

void ASTPrinter::visit(ArrayDeclaratorAST* ast) {
  json_ = nlohmann::json::object();

#ifdef WITH_CXXABI
  char name[1024];
  std::size_t nameSize = sizeof(name);
  abi::__cxa_demangle(typeid(*ast).name(), name, &nameSize, nullptr);
  json_["$id"] = name;
#else
  json_["$id"] = typeid(*ast).name();
#endif

  auto [startLoc, endLoc] = ast->sourceLocationRange();
  if (startLoc && endLoc) {
    unsigned startLine = 0, startColumn = 0;
    unsigned endLine = 0, endColumn = 0;

    unit_->getTokenStartPosition(startLoc, &startLine, &startColumn);
    unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn);

    auto range = nlohmann::json::object();
    range["startLine"] = startLine;
    range["startColumn"] = startColumn;
    range["endLine"] = endLine;
    range["endColumn"] = endColumn;

    json_["$range"] = range;
  }

  if (ast->expression) {
    json_["expression"] = accept(ast->expression);
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    for (auto it = ast->attributeList; it; it = it->next) {
      elements.push_back(accept(it->value));
    }
    json_["attributeList"] = elements;
  }
}

}  // namespace cxx