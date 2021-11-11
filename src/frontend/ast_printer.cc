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

#include "ast_printer.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/translation_unit.h>

namespace cxx {

nlohmann::json ASTPrinter::operator()(AST* ast, bool printLocations) {
  std::vector<std::string_view> fileNames;
  std::swap(fileNames_, fileNames);
  std::swap(printLocations_, printLocations);
  auto result = accept(ast);
  std::swap(printLocations_, printLocations);
  std::swap(fileNames_, fileNames);
  result.push_back(std::vector<nlohmann::json>{"$files", std::move(fileNames)});
  return result;
}

nlohmann::json ASTPrinter::accept(AST* ast) {
  nlohmann::json json;

  if (ast) {
    std::swap(json_, json);
    ast->accept(this);
    std::swap(json_, json);

    if (!json.is_null() && printLocations_) {
      auto [startLoc, endLoc] = ast->sourceLocationRange();
      if (startLoc && endLoc) {
        unsigned startLine = 0, startColumn = 0;
        unsigned endLine = 0, endColumn = 0;
        std::string_view fileName, endFileName;

        unit_->getTokenStartPosition(startLoc, &startLine, &startColumn,
                                     &fileName);
        unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn,
                                   &endFileName);

        if (fileName == endFileName && !fileName.empty()) {
          auto it = std::find(begin(fileNames_), end(fileNames_), fileName);
          auto fileId = std::distance(begin(fileNames_), it);
          if (it == fileNames_.end()) fileNames_.push_back(fileName);
          json.push_back(std::vector<nlohmann::json>{
              "$range", fileId, startLine, startColumn, endLine, endColumn});
        }
      }
    }
  }

  return json;
}

void ASTPrinter::visit(TypeIdAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypeId");

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeSpecifierList", elements});
  }

  if (ast->declarator) {
    if (auto childNode = accept(ast->declarator); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarator", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(NestedNameSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NestedNameSpecifier");

  if (ast->nameList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->nameList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(std::vector<nlohmann::json>{"attr:nameList", elements});
  }
}

void ASTPrinter::visit(UsingDeclaratorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:UsingDeclarator");

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(HandlerAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:Handler");

  if (ast->exceptionDeclaration) {
    if (auto childNode = accept(ast->exceptionDeclaration);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:exceptionDeclaration",
                                                  std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(EnumBaseAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:EnumBase");

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeSpecifierList", elements});
  }
}

void ASTPrinter::visit(EnumeratorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:Enumerator");

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(DeclaratorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:Declarator");

  if (ast->ptrOpList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->ptrOpList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(std::vector<nlohmann::json>{"attr:ptrOpList", elements});
  }

  if (ast->coreDeclarator) {
    if (auto childNode = accept(ast->coreDeclarator); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:coreDeclarator",
                                                  std::move(childNode)});
    }
  }

  if (ast->modifiers) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->modifiers; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(std::vector<nlohmann::json>{"attr:modifiers", elements});
  }
}

void ASTPrinter::visit(InitDeclaratorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:InitDeclarator");

  if (ast->declarator) {
    if (auto childNode = accept(ast->declarator); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarator", std::move(childNode)});
    }
  }

  if (ast->requiresClause) {
    if (auto childNode = accept(ast->requiresClause); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:requiresClause",
                                                  std::move(childNode)});
    }
  }

  if (ast->initializer) {
    if (auto childNode = accept(ast->initializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:initializer",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(BaseSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:BaseSpecifier");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(BaseClauseAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:BaseClause");

  if (ast->baseSpecifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->baseSpecifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:baseSpecifierList", elements});
  }
}

void ASTPrinter::visit(NewTypeIdAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NewTypeId");

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeSpecifierList", elements});
  }
}

void ASTPrinter::visit(RequiresClauseAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:RequiresClause");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ParameterDeclarationClauseAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ParameterDeclarationClause");

  if (ast->parameterDeclarationList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->parameterDeclarationList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(std::vector<nlohmann::json>{
          "attr:parameterDeclarationList", elements});
  }
}

void ASTPrinter::visit(ParametersAndQualifiersAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ParametersAndQualifiers");

  if (ast->parameterDeclarationClause) {
    if (auto childNode = accept(ast->parameterDeclarationClause);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{
          "attr:parameterDeclarationClause", std::move(childNode)});
    }
  }

  if (ast->cvQualifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:cvQualifierList", elements});
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }
}

void ASTPrinter::visit(LambdaIntroducerAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:LambdaIntroducer");

  if (ast->captureList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->captureList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:captureList", elements});
  }
}

void ASTPrinter::visit(LambdaDeclaratorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:LambdaDeclarator");

  if (ast->parameterDeclarationClause) {
    if (auto childNode = accept(ast->parameterDeclarationClause);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{
          "attr:parameterDeclarationClause", std::move(childNode)});
    }
  }

  if (ast->declSpecifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declSpecifierList", elements});
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->trailingReturnType) {
    if (auto childNode = accept(ast->trailingReturnType);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:trailingReturnType",
                                                  std::move(childNode)});
    }
  }

  if (ast->requiresClause) {
    if (auto childNode = accept(ast->requiresClause); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:requiresClause",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TrailingReturnTypeAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TrailingReturnType");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(CtorInitializerAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CtorInitializer");

  if (ast->memInitializerList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->memInitializerList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:memInitializerList", elements});
  }
}

void ASTPrinter::visit(RequirementBodyAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:RequirementBody");

  if (ast->requirementList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->requirementList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:requirementList", elements});
  }
}

void ASTPrinter::visit(TypeConstraintAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypeConstraint");

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(GlobalModuleFragmentAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:GlobalModuleFragment");

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declarationList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarationList", elements});
  }
}

void ASTPrinter::visit(PrivateModuleFragmentAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:PrivateModuleFragment");

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declarationList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarationList", elements});
  }
}

void ASTPrinter::visit(ModuleDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ModuleDeclaration");

  if (ast->moduleName) {
    if (auto childNode = accept(ast->moduleName); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:moduleName", std::move(childNode)});
    }
  }

  if (ast->modulePartition) {
    if (auto childNode = accept(ast->modulePartition); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:modulePartition",
                                                  std::move(childNode)});
    }
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }
}

void ASTPrinter::visit(ModuleNameAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ModuleName");
}

void ASTPrinter::visit(ImportNameAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ImportName");

  if (ast->modulePartition) {
    if (auto childNode = accept(ast->modulePartition); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:modulePartition",
                                                  std::move(childNode)});
    }
  }

  if (ast->moduleName) {
    if (auto childNode = accept(ast->moduleName); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:moduleName", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ModulePartitionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ModulePartition");

  if (ast->moduleName) {
    if (auto childNode = accept(ast->moduleName); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:moduleName", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(SimpleRequirementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:SimpleRequirement");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(CompoundRequirementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CompoundRequirement");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }

  if (ast->typeConstraint) {
    if (auto childNode = accept(ast->typeConstraint); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:typeConstraint",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TypeRequirementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypeRequirement");

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(NestedRequirementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NestedRequirement");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TypeTemplateArgumentAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypeTemplateArgument");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ExpressionTemplateArgumentAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ExpressionTemplateArgument");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ParenMemInitializerAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ParenMemInitializer");

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->expressionList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expressionList", elements});
  }
}

void ASTPrinter::visit(BracedMemInitializerAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:BracedMemInitializer");

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->bracedInitList) {
    if (auto childNode = accept(ast->bracedInitList); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:bracedInitList",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ThisLambdaCaptureAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ThisLambdaCapture");
}

void ASTPrinter::visit(DerefThisLambdaCaptureAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DerefThisLambdaCapture");
}

void ASTPrinter::visit(SimpleLambdaCaptureAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:SimpleLambdaCapture");

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(RefLambdaCaptureAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:RefLambdaCapture");

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(RefInitLambdaCaptureAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:RefInitLambdaCapture");

  if (ast->initializer) {
    if (auto childNode = accept(ast->initializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:initializer",
                                                  std::move(childNode)});
    }
  }

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(InitLambdaCaptureAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:InitLambdaCapture");

  if (ast->initializer) {
    if (auto childNode = accept(ast->initializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:initializer",
                                                  std::move(childNode)});
    }
  }

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(EqualInitializerAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:EqualInitializer");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(BracedInitListAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:BracedInitList");

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->expressionList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expressionList", elements});
  }
}

void ASTPrinter::visit(ParenInitializerAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ParenInitializer");

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->expressionList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expressionList", elements});
  }
}

void ASTPrinter::visit(NewParenInitializerAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NewParenInitializer");

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->expressionList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expressionList", elements});
  }
}

void ASTPrinter::visit(NewBracedInitializerAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NewBracedInitializer");

  if (ast->bracedInit) {
    if (auto childNode = accept(ast->bracedInit); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:bracedInit", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(EllipsisExceptionDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:EllipsisExceptionDeclaration");
}

void ASTPrinter::visit(TypeExceptionDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypeExceptionDeclaration");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeSpecifierList", elements});
  }

  if (ast->declarator) {
    if (auto childNode = accept(ast->declarator); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarator", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(DefaultFunctionBodyAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DefaultFunctionBody");
}

void ASTPrinter::visit(CompoundStatementFunctionBodyAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CompoundStatementFunctionBody");

  if (ast->ctorInitializer) {
    if (auto childNode = accept(ast->ctorInitializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:ctorInitializer",
                                                  std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TryStatementFunctionBodyAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TryStatementFunctionBody");

  if (ast->ctorInitializer) {
    if (auto childNode = accept(ast->ctorInitializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:ctorInitializer",
                                                  std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }

  if (ast->handlerList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->handlerList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:handlerList", elements});
  }
}

void ASTPrinter::visit(DeleteFunctionBodyAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DeleteFunctionBody");
}

void ASTPrinter::visit(TranslationUnitAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TranslationUnit");

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declarationList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarationList", elements});
  }
}

void ASTPrinter::visit(ModuleUnitAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ModuleUnit");

  if (ast->globalModuleFragment) {
    if (auto childNode = accept(ast->globalModuleFragment);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:globalModuleFragment",
                                                  std::move(childNode)});
    }
  }

  if (ast->moduleDeclaration) {
    if (auto childNode = accept(ast->moduleDeclaration); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:moduleDeclaration",
                                                  std::move(childNode)});
    }
  }

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declarationList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarationList", elements});
  }

  if (ast->privateModuleFragmentAST) {
    if (auto childNode = accept(ast->privateModuleFragmentAST);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{
          "attr:privateModuleFragmentAST", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ThisExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ThisExpression");
}

void ASTPrinter::visit(CharLiteralExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CharLiteralExpression");

  if (ast->literal)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:literal",
        std::vector<nlohmann::json>{"literal", ast->literal->value()}});
}

void ASTPrinter::visit(BoolLiteralExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:BoolLiteralExpression");

  if (ast->literal != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:literal",
        std::vector<nlohmann::json>{"token", Token::spell(ast->literal)}});
  }
}

void ASTPrinter::visit(IntLiteralExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:IntLiteralExpression");

  if (ast->literal)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:literal",
        std::vector<nlohmann::json>{"literal", ast->literal->value()}});
}

void ASTPrinter::visit(FloatLiteralExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:FloatLiteralExpression");

  if (ast->literal)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:literal",
        std::vector<nlohmann::json>{"literal", ast->literal->value()}});
}

void ASTPrinter::visit(NullptrLiteralExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NullptrLiteralExpression");

  if (ast->literal != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:literal",
        std::vector<nlohmann::json>{"token", Token::spell(ast->literal)}});
  }
}

void ASTPrinter::visit(StringLiteralExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:StringLiteralExpression");

  if (ast->literal)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:literal",
        std::vector<nlohmann::json>{"literal", ast->literal->value()}});
}

void ASTPrinter::visit(UserDefinedStringLiteralExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:UserDefinedStringLiteralExpression");

  if (ast->literal)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:literal",
        std::vector<nlohmann::json>{"literal", ast->literal->value()}});
}

void ASTPrinter::visit(IdExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:IdExpression");

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(RequiresExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:RequiresExpression");

  if (ast->parameterDeclarationClause) {
    if (auto childNode = accept(ast->parameterDeclarationClause);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{
          "attr:parameterDeclarationClause", std::move(childNode)});
    }
  }

  if (ast->requirementBody) {
    if (auto childNode = accept(ast->requirementBody); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:requirementBody",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(NestedExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NestedExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(RightFoldExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:RightFoldExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }

  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:op",
        std::vector<nlohmann::json>{"token", Token::spell(ast->op)}});
  }
}

void ASTPrinter::visit(LeftFoldExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:LeftFoldExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }

  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:op",
        std::vector<nlohmann::json>{"token", Token::spell(ast->op)}});
  }
}

void ASTPrinter::visit(FoldExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:FoldExpression");

  if (ast->leftExpression) {
    if (auto childNode = accept(ast->leftExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:leftExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->rightExpression) {
    if (auto childNode = accept(ast->rightExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:rightExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:op",
        std::vector<nlohmann::json>{"token", Token::spell(ast->op)}});
  }

  if (ast->foldOp != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:foldOp",
        std::vector<nlohmann::json>{"token", Token::spell(ast->foldOp)}});
  }
}

void ASTPrinter::visit(LambdaExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:LambdaExpression");

  if (ast->lambdaIntroducer) {
    if (auto childNode = accept(ast->lambdaIntroducer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:lambdaIntroducer",
                                                  std::move(childNode)});
    }
  }

  if (ast->templateParameterList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:templateParameterList", elements});
  }

  if (ast->requiresClause) {
    if (auto childNode = accept(ast->requiresClause); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:requiresClause",
                                                  std::move(childNode)});
    }
  }

  if (ast->lambdaDeclarator) {
    if (auto childNode = accept(ast->lambdaDeclarator); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:lambdaDeclarator",
                                                  std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(SizeofExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:SizeofExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(SizeofTypeExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:SizeofTypeExpression");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(SizeofPackExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:SizeofPackExpression");

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(TypeidExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypeidExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TypeidOfTypeExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypeidOfTypeExpression");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(AlignofExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:AlignofExpression");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(UnaryExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:UnaryExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }

  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:op",
        std::vector<nlohmann::json>{"token", Token::spell(ast->op)}});
  }
}

void ASTPrinter::visit(BinaryExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:BinaryExpression");

  if (ast->leftExpression) {
    if (auto childNode = accept(ast->leftExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:leftExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->rightExpression) {
    if (auto childNode = accept(ast->rightExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:rightExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:op",
        std::vector<nlohmann::json>{"token", Token::spell(ast->op)}});
  }
}

void ASTPrinter::visit(AssignmentExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:AssignmentExpression");

  if (ast->leftExpression) {
    if (auto childNode = accept(ast->leftExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:leftExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->rightExpression) {
    if (auto childNode = accept(ast->rightExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:rightExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:op",
        std::vector<nlohmann::json>{"token", Token::spell(ast->op)}});
  }
}

void ASTPrinter::visit(BracedTypeConstructionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:BracedTypeConstruction");

  if (ast->typeSpecifier) {
    if (auto childNode = accept(ast->typeSpecifier); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:typeSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->bracedInitList) {
    if (auto childNode = accept(ast->bracedInitList); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:bracedInitList",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TypeConstructionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypeConstruction");

  if (ast->typeSpecifier) {
    if (auto childNode = accept(ast->typeSpecifier); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:typeSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->expressionList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expressionList", elements});
  }
}

void ASTPrinter::visit(CallExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CallExpression");

  if (ast->baseExpression) {
    if (auto childNode = accept(ast->baseExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:baseExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->expressionList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->expressionList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expressionList", elements});
  }
}

void ASTPrinter::visit(SubscriptExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:SubscriptExpression");

  if (ast->baseExpression) {
    if (auto childNode = accept(ast->baseExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:baseExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->indexExpression) {
    if (auto childNode = accept(ast->indexExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:indexExpression",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(MemberExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:MemberExpression");

  if (ast->baseExpression) {
    if (auto childNode = accept(ast->baseExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:baseExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->accessOp != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:accessOp",
        std::vector<nlohmann::json>{"token", Token::spell(ast->accessOp)}});
  }
}

void ASTPrinter::visit(PostIncrExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:PostIncrExpression");

  if (ast->baseExpression) {
    if (auto childNode = accept(ast->baseExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:baseExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:op",
        std::vector<nlohmann::json>{"token", Token::spell(ast->op)}});
  }
}

void ASTPrinter::visit(ConditionalExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ConditionalExpression");

  if (ast->condition) {
    if (auto childNode = accept(ast->condition); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:condition", std::move(childNode)});
    }
  }

  if (ast->iftrueExpression) {
    if (auto childNode = accept(ast->iftrueExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:iftrueExpression",
                                                  std::move(childNode)});
    }
  }

  if (ast->iffalseExpression) {
    if (auto childNode = accept(ast->iffalseExpression); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:iffalseExpression",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ImplicitCastExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ImplicitCastExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(CastExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CastExpression");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(CppCastExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CppCastExpression");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(NewExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NewExpression");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }

  if (ast->newInitalizer) {
    if (auto childNode = accept(ast->newInitalizer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:newInitalizer",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(DeleteExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DeleteExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ThrowExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ThrowExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(NoexceptExpressionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NoexceptExpression");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(LabeledStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:LabeledStatement");

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(CaseStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CaseStatement");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(DefaultStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DefaultStatement");

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ExpressionStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ExpressionStatement");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(CompoundStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CompoundStatement");

  if (ast->statementList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->statementList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statementList", elements});
  }
}

void ASTPrinter::visit(IfStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:IfStatement");

  if (ast->initializer) {
    if (auto childNode = accept(ast->initializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:initializer",
                                                  std::move(childNode)});
    }
  }

  if (ast->condition) {
    if (auto childNode = accept(ast->condition); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:condition", std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }

  if (ast->elseStatement) {
    if (auto childNode = accept(ast->elseStatement); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:elseStatement",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(SwitchStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:SwitchStatement");

  if (ast->initializer) {
    if (auto childNode = accept(ast->initializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:initializer",
                                                  std::move(childNode)});
    }
  }

  if (ast->condition) {
    if (auto childNode = accept(ast->condition); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:condition", std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(WhileStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:WhileStatement");

  if (ast->condition) {
    if (auto childNode = accept(ast->condition); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:condition", std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(DoStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DoStatement");

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ForRangeStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ForRangeStatement");

  if (ast->initializer) {
    if (auto childNode = accept(ast->initializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:initializer",
                                                  std::move(childNode)});
    }
  }

  if (ast->rangeDeclaration) {
    if (auto childNode = accept(ast->rangeDeclaration); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:rangeDeclaration",
                                                  std::move(childNode)});
    }
  }

  if (ast->rangeInitializer) {
    if (auto childNode = accept(ast->rangeInitializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:rangeInitializer",
                                                  std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ForStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ForStatement");

  if (ast->initializer) {
    if (auto childNode = accept(ast->initializer); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:initializer",
                                                  std::move(childNode)});
    }
  }

  if (ast->condition) {
    if (auto childNode = accept(ast->condition); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:condition", std::move(childNode)});
    }
  }

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(BreakStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:BreakStatement");
}

void ASTPrinter::visit(ContinueStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ContinueStatement");
}

void ASTPrinter::visit(ReturnStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ReturnStatement");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(GotoStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:GotoStatement");

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(CoroutineReturnStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:CoroutineReturnStatement");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(DeclarationStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DeclarationStatement");

  if (ast->declaration) {
    if (auto childNode = accept(ast->declaration); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:declaration",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TryBlockStatementAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TryBlockStatement");

  if (ast->statement) {
    if (auto childNode = accept(ast->statement); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:statement", std::move(childNode)});
    }
  }

  if (ast->handlerList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->handlerList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:handlerList", elements});
  }
}

void ASTPrinter::visit(AccessDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:AccessDeclaration");
}

void ASTPrinter::visit(FunctionDefinitionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:FunctionDefinition");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->declSpecifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declSpecifierList", elements});
  }

  if (ast->declarator) {
    if (auto childNode = accept(ast->declarator); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarator", std::move(childNode)});
    }
  }

  if (ast->requiresClause) {
    if (auto childNode = accept(ast->requiresClause); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:requiresClause",
                                                  std::move(childNode)});
    }
  }

  if (ast->functionBody) {
    if (auto childNode = accept(ast->functionBody); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:functionBody",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ConceptDefinitionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ConceptDefinition");

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ForRangeDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ForRangeDeclaration");
}

void ASTPrinter::visit(AliasDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:AliasDeclaration");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(SimpleDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:SimpleDeclaration");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->declSpecifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declSpecifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declSpecifierList", elements});
  }

  if (ast->initDeclaratorList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->initDeclaratorList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:initDeclaratorList", elements});
  }

  if (ast->requiresClause) {
    if (auto childNode = accept(ast->requiresClause); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:requiresClause",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(StaticAssertDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:StaticAssertDeclaration");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(EmptyDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:EmptyDeclaration");
}

void ASTPrinter::visit(AttributeDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:AttributeDeclaration");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }
}

void ASTPrinter::visit(OpaqueEnumDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:OpaqueEnumDeclaration");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->enumBase) {
    if (auto childNode = accept(ast->enumBase); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:enumBase", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(UsingEnumDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:UsingEnumDeclaration");
}

void ASTPrinter::visit(NamespaceDefinitionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NamespaceDefinition");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->extraAttributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->extraAttributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:extraAttributeList", elements});
  }

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declarationList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarationList", elements});
  }
}

void ASTPrinter::visit(NamespaceAliasDefinitionAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NamespaceAliasDefinition");

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(UsingDirectiveAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:UsingDirective");
}

void ASTPrinter::visit(UsingDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:UsingDeclaration");

  if (ast->usingDeclaratorList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->usingDeclaratorList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:usingDeclaratorList", elements});
  }
}

void ASTPrinter::visit(AsmDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:AsmDeclaration");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }
}

void ASTPrinter::visit(ExportDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ExportDeclaration");

  if (ast->declaration) {
    if (auto childNode = accept(ast->declaration); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:declaration",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ExportCompoundDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ExportCompoundDeclaration");

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declarationList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarationList", elements});
  }
}

void ASTPrinter::visit(ModuleImportDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ModuleImportDeclaration");

  if (ast->importName) {
    if (auto childNode = accept(ast->importName); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:importName", std::move(childNode)});
    }
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }
}

void ASTPrinter::visit(TemplateDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TemplateDeclaration");

  if (ast->templateParameterList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:templateParameterList", elements});
  }

  if (ast->requiresClause) {
    if (auto childNode = accept(ast->requiresClause); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:requiresClause",
                                                  std::move(childNode)});
    }
  }

  if (ast->declaration) {
    if (auto childNode = accept(ast->declaration); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:declaration",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TypenameTypeParameterAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypenameTypeParameter");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(TypenamePackTypeParameterAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypenamePackTypeParameter");

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(TemplateTypeParameterAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TemplateTypeParameter");

  if (ast->templateParameterList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:templateParameterList", elements});
  }

  if (ast->requiresClause) {
    if (auto childNode = accept(ast->requiresClause); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:requiresClause",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(TemplatePackTypeParameterAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TemplatePackTypeParameter");

  if (ast->templateParameterList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->templateParameterList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:templateParameterList", elements});
  }

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(DeductionGuideAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DeductionGuide");
}

void ASTPrinter::visit(ExplicitInstantiationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ExplicitInstantiation");

  if (ast->declaration) {
    if (auto childNode = accept(ast->declaration); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:declaration",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ParameterDeclarationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ParameterDeclaration");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->typeSpecifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->typeSpecifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeSpecifierList", elements});
  }

  if (ast->declarator) {
    if (auto childNode = accept(ast->declarator); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarator", std::move(childNode)});
    }
  }

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(LinkageSpecificationAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:LinkageSpecification");

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declarationList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarationList", elements});
  }

  if (ast->stringLiteral)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:stringLiteral",
        std::vector<nlohmann::json>{"literal", ast->stringLiteral->value()}});
}

void ASTPrinter::visit(SimpleNameAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:SimpleName");

  if (ast->identifier)
    json_.push_back(std::vector<nlohmann::json>{
        "attr:identifier",
        std::vector<nlohmann::json>{"identifier", ast->identifier->name()}});
}

void ASTPrinter::visit(DestructorNameAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DestructorName");

  if (ast->id) {
    if (auto childNode = accept(ast->id); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:id", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(DecltypeNameAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DecltypeName");

  if (ast->decltypeSpecifier) {
    if (auto childNode = accept(ast->decltypeSpecifier); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:decltypeSpecifier",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(OperatorNameAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:OperatorName");

  if (ast->op != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:op",
        std::vector<nlohmann::json>{"token", Token::spell(ast->op)}});
  }
}

void ASTPrinter::visit(ConversionNameAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ConversionName");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TemplateNameAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TemplateName");

  if (ast->id) {
    if (auto childNode = accept(ast->id); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:id", std::move(childNode)});
    }
  }

  if (ast->templateArgumentList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->templateArgumentList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:templateArgumentList", elements});
  }
}

void ASTPrinter::visit(QualifiedNameAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:QualifiedName");

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->id) {
    if (auto childNode = accept(ast->id); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:id", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TypedefSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypedefSpecifier");
}

void ASTPrinter::visit(FriendSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:FriendSpecifier");
}

void ASTPrinter::visit(ConstevalSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ConstevalSpecifier");
}

void ASTPrinter::visit(ConstinitSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ConstinitSpecifier");
}

void ASTPrinter::visit(ConstexprSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ConstexprSpecifier");
}

void ASTPrinter::visit(InlineSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:InlineSpecifier");
}

void ASTPrinter::visit(StaticSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:StaticSpecifier");
}

void ASTPrinter::visit(ExternSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ExternSpecifier");
}

void ASTPrinter::visit(ThreadLocalSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ThreadLocalSpecifier");
}

void ASTPrinter::visit(ThreadSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ThreadSpecifier");
}

void ASTPrinter::visit(MutableSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:MutableSpecifier");
}

void ASTPrinter::visit(VirtualSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:VirtualSpecifier");
}

void ASTPrinter::visit(ExplicitSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ExplicitSpecifier");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(AutoTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:AutoTypeSpecifier");
}

void ASTPrinter::visit(VoidTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:VoidTypeSpecifier");
}

void ASTPrinter::visit(VaListTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:VaListTypeSpecifier");

  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:specifier",
        std::vector<nlohmann::json>{"token", Token::spell(ast->specifier)}});
  }
}

void ASTPrinter::visit(IntegralTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:IntegralTypeSpecifier");

  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:specifier",
        std::vector<nlohmann::json>{"token", Token::spell(ast->specifier)}});
  }
}

void ASTPrinter::visit(FloatingPointTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:FloatingPointTypeSpecifier");

  if (ast->specifier != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:specifier",
        std::vector<nlohmann::json>{"token", Token::spell(ast->specifier)}});
  }
}

void ASTPrinter::visit(ComplexTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ComplexTypeSpecifier");
}

void ASTPrinter::visit(NamedTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NamedTypeSpecifier");

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(AtomicTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:AtomicTypeSpecifier");

  if (ast->typeId) {
    if (auto childNode = accept(ast->typeId); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:typeId", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(UnderlyingTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:UnderlyingTypeSpecifier");
}

void ASTPrinter::visit(ElaboratedTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ElaboratedTypeSpecifier");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(DecltypeAutoSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DecltypeAutoSpecifier");
}

void ASTPrinter::visit(DecltypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:DecltypeSpecifier");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(TypeofSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypeofSpecifier");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(PlaceholderTypeSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:PlaceholderTypeSpecifier");

  if (ast->typeConstraint) {
    if (auto childNode = accept(ast->typeConstraint); !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:typeConstraint",
                                                  std::move(childNode)});
    }
  }

  if (ast->specifier) {
    if (auto childNode = accept(ast->specifier); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:specifier", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ConstQualifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ConstQualifier");
}

void ASTPrinter::visit(VolatileQualifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:VolatileQualifier");
}

void ASTPrinter::visit(RestrictQualifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:RestrictQualifier");
}

void ASTPrinter::visit(EnumSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:EnumSpecifier");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->enumBase) {
    if (auto childNode = accept(ast->enumBase); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:enumBase", std::move(childNode)});
    }
  }

  if (ast->enumeratorList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->enumeratorList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:enumeratorList", elements});
  }
}

void ASTPrinter::visit(ClassSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ClassSpecifier");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->baseClause) {
    if (auto childNode = accept(ast->baseClause); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:baseClause", std::move(childNode)});
    }
  }

  if (ast->declarationList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->declarationList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarationList", elements});
  }
}

void ASTPrinter::visit(TypenameSpecifierAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:TypenameSpecifier");

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(IdDeclaratorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:IdDeclarator");

  if (ast->name) {
    if (auto childNode = accept(ast->name); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:name", std::move(childNode)});
    }
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }
}

void ASTPrinter::visit(NestedDeclaratorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:NestedDeclarator");

  if (ast->declarator) {
    if (auto childNode = accept(ast->declarator); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:declarator", std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(PointerOperatorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:PointerOperator");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->cvQualifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:cvQualifierList", elements});
  }
}

void ASTPrinter::visit(ReferenceOperatorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ReferenceOperator");

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->refOp != TokenKind::T_EOF_SYMBOL) {
    json_.push_back(std::vector<nlohmann::json>{
        "attr:refOp",
        std::vector<nlohmann::json>{"token", Token::spell(ast->refOp)}});
  }
}

void ASTPrinter::visit(PtrToMemberOperatorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:PtrToMemberOperator");

  if (ast->nestedNameSpecifier) {
    if (auto childNode = accept(ast->nestedNameSpecifier);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:nestedNameSpecifier",
                                                  std::move(childNode)});
    }
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }

  if (ast->cvQualifierList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:cvQualifierList", elements});
  }
}

void ASTPrinter::visit(FunctionDeclaratorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:FunctionDeclarator");

  if (ast->parametersAndQualifiers) {
    if (auto childNode = accept(ast->parametersAndQualifiers);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{
          "attr:parametersAndQualifiers", std::move(childNode)});
    }
  }

  if (ast->trailingReturnType) {
    if (auto childNode = accept(ast->trailingReturnType);
        !childNode.is_null()) {
      json_.push_back(std::vector<nlohmann::json>{"attr:trailingReturnType",
                                                  std::move(childNode)});
    }
  }
}

void ASTPrinter::visit(ArrayDeclaratorAST* ast) {
  json_ = nlohmann::json::array();

  json_.push_back("ast:ArrayDeclarator");

  if (ast->expression) {
    if (auto childNode = accept(ast->expression); !childNode.is_null()) {
      json_.push_back(
          std::vector<nlohmann::json>{"attr:expression", std::move(childNode)});
    }
  }

  if (ast->attributeList) {
    auto elements = nlohmann::json::array();
    elements.push_back("array");
    for (auto it = ast->attributeList; it; it = it->next) {
      if (auto childNode = accept(it->value); !childNode.is_null()) {
        elements.push_back(std::move(childNode));
      }
    }
    if (elements.size() > 1)
      json_.push_back(
          std::vector<nlohmann::json>{"attr:attributeList", elements});
  }
}

}  // namespace cxx
