// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast_pretty_printer.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/translation_unit.h>

namespace cxx {

struct ASTPrettyPrinter::UnitVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(TranslationUnitAST* ast);

  void operator()(ModuleUnitAST* ast);
};

struct ASTPrettyPrinter::DeclarationVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(SimpleDeclarationAST* ast);

  void operator()(AsmDeclarationAST* ast);

  void operator()(NamespaceAliasDefinitionAST* ast);

  void operator()(UsingDeclarationAST* ast);

  void operator()(UsingEnumDeclarationAST* ast);

  void operator()(UsingDirectiveAST* ast);

  void operator()(StaticAssertDeclarationAST* ast);

  void operator()(AliasDeclarationAST* ast);

  void operator()(OpaqueEnumDeclarationAST* ast);

  void operator()(FunctionDefinitionAST* ast);

  void operator()(TemplateDeclarationAST* ast);

  void operator()(ConceptDefinitionAST* ast);

  void operator()(DeductionGuideAST* ast);

  void operator()(ExplicitInstantiationAST* ast);

  void operator()(ExportDeclarationAST* ast);

  void operator()(ExportCompoundDeclarationAST* ast);

  void operator()(LinkageSpecificationAST* ast);

  void operator()(NamespaceDefinitionAST* ast);

  void operator()(EmptyDeclarationAST* ast);

  void operator()(AttributeDeclarationAST* ast);

  void operator()(ModuleImportDeclarationAST* ast);

  void operator()(ParameterDeclarationAST* ast);

  void operator()(AccessDeclarationAST* ast);

  void operator()(ForRangeDeclarationAST* ast);

  void operator()(StructuredBindingDeclarationAST* ast);

  void operator()(AsmOperandAST* ast);

  void operator()(AsmQualifierAST* ast);

  void operator()(AsmClobberAST* ast);

  void operator()(AsmGotoLabelAST* ast);
};

struct ASTPrettyPrinter::StatementVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(LabeledStatementAST* ast);

  void operator()(CaseStatementAST* ast);

  void operator()(DefaultStatementAST* ast);

  void operator()(ExpressionStatementAST* ast);

  void operator()(CompoundStatementAST* ast);

  void operator()(IfStatementAST* ast);

  void operator()(ConstevalIfStatementAST* ast);

  void operator()(SwitchStatementAST* ast);

  void operator()(WhileStatementAST* ast);

  void operator()(DoStatementAST* ast);

  void operator()(ForRangeStatementAST* ast);

  void operator()(ForStatementAST* ast);

  void operator()(BreakStatementAST* ast);

  void operator()(ContinueStatementAST* ast);

  void operator()(ReturnStatementAST* ast);

  void operator()(CoroutineReturnStatementAST* ast);

  void operator()(GotoStatementAST* ast);

  void operator()(DeclarationStatementAST* ast);

  void operator()(TryBlockStatementAST* ast);
};

struct ASTPrettyPrinter::ExpressionVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(GeneratedLiteralExpressionAST* ast);

  void operator()(CharLiteralExpressionAST* ast);

  void operator()(BoolLiteralExpressionAST* ast);

  void operator()(IntLiteralExpressionAST* ast);

  void operator()(FloatLiteralExpressionAST* ast);

  void operator()(NullptrLiteralExpressionAST* ast);

  void operator()(StringLiteralExpressionAST* ast);

  void operator()(UserDefinedStringLiteralExpressionAST* ast);

  void operator()(ThisExpressionAST* ast);

  void operator()(NestedStatementExpressionAST* ast);

  void operator()(NestedExpressionAST* ast);

  void operator()(IdExpressionAST* ast);

  void operator()(LambdaExpressionAST* ast);

  void operator()(FoldExpressionAST* ast);

  void operator()(RightFoldExpressionAST* ast);

  void operator()(LeftFoldExpressionAST* ast);

  void operator()(RequiresExpressionAST* ast);

  void operator()(VaArgExpressionAST* ast);

  void operator()(SubscriptExpressionAST* ast);

  void operator()(CallExpressionAST* ast);

  void operator()(TypeConstructionAST* ast);

  void operator()(BracedTypeConstructionAST* ast);

  void operator()(SpliceMemberExpressionAST* ast);

  void operator()(MemberExpressionAST* ast);

  void operator()(PostIncrExpressionAST* ast);

  void operator()(CppCastExpressionAST* ast);

  void operator()(BuiltinBitCastExpressionAST* ast);

  void operator()(BuiltinOffsetofExpressionAST* ast);

  void operator()(TypeidExpressionAST* ast);

  void operator()(TypeidOfTypeExpressionAST* ast);

  void operator()(SpliceExpressionAST* ast);

  void operator()(GlobalScopeReflectExpressionAST* ast);

  void operator()(NamespaceReflectExpressionAST* ast);

  void operator()(TypeIdReflectExpressionAST* ast);

  void operator()(ReflectExpressionAST* ast);

  void operator()(UnaryExpressionAST* ast);

  void operator()(AwaitExpressionAST* ast);

  void operator()(SizeofExpressionAST* ast);

  void operator()(SizeofTypeExpressionAST* ast);

  void operator()(SizeofPackExpressionAST* ast);

  void operator()(AlignofTypeExpressionAST* ast);

  void operator()(AlignofExpressionAST* ast);

  void operator()(NoexceptExpressionAST* ast);

  void operator()(NewExpressionAST* ast);

  void operator()(DeleteExpressionAST* ast);

  void operator()(CastExpressionAST* ast);

  void operator()(ImplicitCastExpressionAST* ast);

  void operator()(BinaryExpressionAST* ast);

  void operator()(ConditionalExpressionAST* ast);

  void operator()(YieldExpressionAST* ast);

  void operator()(ThrowExpressionAST* ast);

  void operator()(AssignmentExpressionAST* ast);

  void operator()(PackExpansionExpressionAST* ast);

  void operator()(DesignatedInitializerClauseAST* ast);

  void operator()(TypeTraitExpressionAST* ast);

  void operator()(ConditionExpressionAST* ast);

  void operator()(EqualInitializerAST* ast);

  void operator()(BracedInitListAST* ast);

  void operator()(ParenInitializerAST* ast);
};

struct ASTPrettyPrinter::TemplateParameterVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(TemplateTypeParameterAST* ast);

  void operator()(NonTypeTemplateParameterAST* ast);

  void operator()(TypenameTypeParameterAST* ast);

  void operator()(ConstraintTypeParameterAST* ast);
};

struct ASTPrettyPrinter::SpecifierVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(GeneratedTypeSpecifierAST* ast);

  void operator()(TypedefSpecifierAST* ast);

  void operator()(FriendSpecifierAST* ast);

  void operator()(ConstevalSpecifierAST* ast);

  void operator()(ConstinitSpecifierAST* ast);

  void operator()(ConstexprSpecifierAST* ast);

  void operator()(InlineSpecifierAST* ast);

  void operator()(StaticSpecifierAST* ast);

  void operator()(ExternSpecifierAST* ast);

  void operator()(ThreadLocalSpecifierAST* ast);

  void operator()(ThreadSpecifierAST* ast);

  void operator()(MutableSpecifierAST* ast);

  void operator()(VirtualSpecifierAST* ast);

  void operator()(ExplicitSpecifierAST* ast);

  void operator()(AutoTypeSpecifierAST* ast);

  void operator()(VoidTypeSpecifierAST* ast);

  void operator()(SizeTypeSpecifierAST* ast);

  void operator()(SignTypeSpecifierAST* ast);

  void operator()(VaListTypeSpecifierAST* ast);

  void operator()(IntegralTypeSpecifierAST* ast);

  void operator()(FloatingPointTypeSpecifierAST* ast);

  void operator()(ComplexTypeSpecifierAST* ast);

  void operator()(NamedTypeSpecifierAST* ast);

  void operator()(AtomicTypeSpecifierAST* ast);

  void operator()(UnderlyingTypeSpecifierAST* ast);

  void operator()(ElaboratedTypeSpecifierAST* ast);

  void operator()(DecltypeAutoSpecifierAST* ast);

  void operator()(DecltypeSpecifierAST* ast);

  void operator()(PlaceholderTypeSpecifierAST* ast);

  void operator()(ConstQualifierAST* ast);

  void operator()(VolatileQualifierAST* ast);

  void operator()(RestrictQualifierAST* ast);

  void operator()(EnumSpecifierAST* ast);

  void operator()(ClassSpecifierAST* ast);

  void operator()(TypenameSpecifierAST* ast);

  void operator()(SplicerTypeSpecifierAST* ast);
};

struct ASTPrettyPrinter::PtrOperatorVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(PointerOperatorAST* ast);

  void operator()(ReferenceOperatorAST* ast);

  void operator()(PtrToMemberOperatorAST* ast);
};

struct ASTPrettyPrinter::CoreDeclaratorVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(BitfieldDeclaratorAST* ast);

  void operator()(ParameterPackAST* ast);

  void operator()(IdDeclaratorAST* ast);

  void operator()(NestedDeclaratorAST* ast);
};

struct ASTPrettyPrinter::DeclaratorChunkVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(FunctionDeclaratorChunkAST* ast);

  void operator()(ArrayDeclaratorChunkAST* ast);
};

struct ASTPrettyPrinter::UnqualifiedIdVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(NameIdAST* ast);

  void operator()(DestructorIdAST* ast);

  void operator()(DecltypeIdAST* ast);

  void operator()(OperatorFunctionIdAST* ast);

  void operator()(LiteralOperatorIdAST* ast);

  void operator()(ConversionFunctionIdAST* ast);

  void operator()(SimpleTemplateIdAST* ast);

  void operator()(LiteralOperatorTemplateIdAST* ast);

  void operator()(OperatorFunctionTemplateIdAST* ast);
};

struct ASTPrettyPrinter::NestedNameSpecifierVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(GlobalNestedNameSpecifierAST* ast);

  void operator()(SimpleNestedNameSpecifierAST* ast);

  void operator()(DecltypeNestedNameSpecifierAST* ast);

  void operator()(TemplateNestedNameSpecifierAST* ast);
};

struct ASTPrettyPrinter::FunctionBodyVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(DefaultFunctionBodyAST* ast);

  void operator()(CompoundStatementFunctionBodyAST* ast);

  void operator()(TryStatementFunctionBodyAST* ast);

  void operator()(DeleteFunctionBodyAST* ast);
};

struct ASTPrettyPrinter::TemplateArgumentVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(TypeTemplateArgumentAST* ast);

  void operator()(ExpressionTemplateArgumentAST* ast);
};

struct ASTPrettyPrinter::ExceptionSpecifierVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(ThrowExceptionSpecifierAST* ast);

  void operator()(NoexceptSpecifierAST* ast);
};

struct ASTPrettyPrinter::RequirementVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(SimpleRequirementAST* ast);

  void operator()(CompoundRequirementAST* ast);

  void operator()(TypeRequirementAST* ast);

  void operator()(NestedRequirementAST* ast);
};

struct ASTPrettyPrinter::NewInitializerVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(NewParenInitializerAST* ast);

  void operator()(NewBracedInitializerAST* ast);
};

struct ASTPrettyPrinter::MemInitializerVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(ParenMemInitializerAST* ast);

  void operator()(BracedMemInitializerAST* ast);
};

struct ASTPrettyPrinter::LambdaCaptureVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(ThisLambdaCaptureAST* ast);

  void operator()(DerefThisLambdaCaptureAST* ast);

  void operator()(SimpleLambdaCaptureAST* ast);

  void operator()(RefLambdaCaptureAST* ast);

  void operator()(RefInitLambdaCaptureAST* ast);

  void operator()(InitLambdaCaptureAST* ast);
};

struct ASTPrettyPrinter::ExceptionDeclarationVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(EllipsisExceptionDeclarationAST* ast);

  void operator()(TypeExceptionDeclarationAST* ast);
};

struct ASTPrettyPrinter::AttributeSpecifierVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(CxxAttributeAST* ast);

  void operator()(GccAttributeAST* ast);

  void operator()(AlignasAttributeAST* ast);

  void operator()(AlignasTypeAttributeAST* ast);

  void operator()(AsmAttributeAST* ast);
};

struct ASTPrettyPrinter::AttributeTokenVisitor {
  ASTPrettyPrinter& accept;
  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return accept.unit_;
  }
  void space() { accept.space(); }
  void nospace() { accept.nospace(); }
  void newline() { accept.newline(); }
  void nonewline() { accept.nonewline(); }
  void indent() { accept.indent(); }
  void unindent() { accept.unindent(); }

  void operator()(ScopedAttributeTokenAST* ast);

  void operator()(SimpleAttributeTokenAST* ast);
};

void ASTPrettyPrinter::operator()(UnitAST* ast) {
  if (!ast) return;
  visit(UnitVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(DeclarationAST* ast) {
  if (!ast) return;
  visit(DeclarationVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(StatementAST* ast) {
  if (!ast) return;
  visit(StatementVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(ExpressionAST* ast) {
  if (!ast) return;
  visit(ExpressionVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(TemplateParameterAST* ast) {
  if (!ast) return;
  visit(TemplateParameterVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(SpecifierAST* ast) {
  if (!ast) return;
  visit(SpecifierVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(PtrOperatorAST* ast) {
  if (!ast) return;
  visit(PtrOperatorVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(CoreDeclaratorAST* ast) {
  if (!ast) return;
  visit(CoreDeclaratorVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(DeclaratorChunkAST* ast) {
  if (!ast) return;
  visit(DeclaratorChunkVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(UnqualifiedIdAST* ast) {
  if (!ast) return;
  visit(UnqualifiedIdVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(NestedNameSpecifierAST* ast) {
  if (!ast) return;
  visit(NestedNameSpecifierVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(FunctionBodyAST* ast) {
  if (!ast) return;
  visit(FunctionBodyVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(TemplateArgumentAST* ast) {
  if (!ast) return;
  visit(TemplateArgumentVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(ExceptionSpecifierAST* ast) {
  if (!ast) return;
  visit(ExceptionSpecifierVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(RequirementAST* ast) {
  if (!ast) return;
  visit(RequirementVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(NewInitializerAST* ast) {
  if (!ast) return;
  visit(NewInitializerVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(MemInitializerAST* ast) {
  if (!ast) return;
  visit(MemInitializerVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(LambdaCaptureAST* ast) {
  if (!ast) return;
  visit(LambdaCaptureVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(ExceptionDeclarationAST* ast) {
  if (!ast) return;
  visit(ExceptionDeclarationVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(AttributeSpecifierAST* ast) {
  if (!ast) return;
  visit(AttributeSpecifierVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(AttributeTokenAST* ast) {
  if (!ast) return;
  visit(AttributeTokenVisitor{*this}, ast);
}

void ASTPrettyPrinter::operator()(SplicerAST* ast) {
  if (!ast) return;

  if (ast->lbracketLoc) {
    nospace();
    writeToken(ast->lbracketLoc);
    nospace();
  }
  if (ast->colonLoc) {
    nospace();
    writeToken(ast->colonLoc);
  }
  if (ast->ellipsisLoc) {
    writeToken(ast->ellipsisLoc);
  }
  operator()(ast->expression);
  if (ast->secondColonLoc) {
    writeToken(ast->secondColonLoc);
  }
  if (ast->rbracketLoc) {
    nospace();
    writeToken(ast->rbracketLoc);
  }
}

void ASTPrettyPrinter::operator()(GlobalModuleFragmentAST* ast) {
  if (!ast) return;

  if (ast->moduleLoc) {
    writeToken(ast->moduleLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    writeToken(ast->semicolonLoc);
    newline();
  }

  for (auto it = ast->declarationList; it; it = it->next) {
    operator()(it->value);
  }
}

void ASTPrettyPrinter::operator()(PrivateModuleFragmentAST* ast) {
  if (!ast) return;

  if (ast->moduleLoc) {
    writeToken(ast->moduleLoc);
  }
  if (ast->colonLoc) {
    nospace();
    writeToken(ast->colonLoc);
  }
  if (ast->privateLoc) {
    writeToken(ast->privateLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    writeToken(ast->semicolonLoc);
    newline();
  }

  for (auto it = ast->declarationList; it; it = it->next) {
    operator()(it->value);
  }
}

void ASTPrettyPrinter::operator()(ModuleDeclarationAST* ast) {
  if (!ast) return;

  if (ast->exportLoc) {
    writeToken(ast->exportLoc);
  }
  if (ast->moduleLoc) {
    writeToken(ast->moduleLoc);
  }
  operator()(ast->moduleName);
  operator()(ast->modulePartition);

  for (auto it = ast->attributeList; it; it = it->next) {
    operator()(it->value);
  }

  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::operator()(ModuleNameAST* ast) {
  if (!ast) return;

  operator()(ast->moduleQualifier);
  if (ast->identifierLoc) {
    writeToken(ast->identifierLoc);
  }
}

void ASTPrettyPrinter::operator()(ModuleQualifierAST* ast) {
  if (!ast) return;

  operator()(ast->moduleQualifier);
  if (ast->identifierLoc) {
    writeToken(ast->identifierLoc);
  }
  if (ast->dotLoc) {
    nospace();
    writeToken(ast->dotLoc);
    nospace();
  }
}

void ASTPrettyPrinter::operator()(ModulePartitionAST* ast) {
  if (!ast) return;

  if (ast->colonLoc) {
    nospace();
    writeToken(ast->colonLoc);
  }
  operator()(ast->moduleName);
}

void ASTPrettyPrinter::operator()(ImportNameAST* ast) {
  if (!ast) return;

  if (ast->headerLoc) {
    writeToken(ast->headerLoc);
  }
  operator()(ast->modulePartition);
  operator()(ast->moduleName);
}

void ASTPrettyPrinter::operator()(InitDeclaratorAST* ast) {
  if (!ast) return;

  operator()(ast->declarator);
  operator()(ast->requiresClause);
  operator()(ast->initializer);
}

void ASTPrettyPrinter::operator()(DeclaratorAST* ast) {
  if (!ast) return;

  for (auto it = ast->ptrOpList; it; it = it->next) {
    operator()(it->value);
  }

  operator()(ast->coreDeclarator);

  for (auto it = ast->declaratorChunkList; it; it = it->next) {
    operator()(it->value);
  }
}

void ASTPrettyPrinter::operator()(UsingDeclaratorAST* ast) {
  if (!ast) return;

  if (ast->typenameLoc) {
    writeToken(ast->typenameLoc);
  }
  operator()(ast->nestedNameSpecifier);
  operator()(ast->unqualifiedId);
  if (ast->ellipsisLoc) {
    writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::operator()(EnumeratorAST* ast) {
  if (!ast) return;

  if (ast->identifierLoc) {
    writeToken(ast->identifierLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    operator()(it->value);
  }

  if (ast->equalLoc) {
    space();
    writeToken(ast->equalLoc);
  }
  operator()(ast->expression);
}

void ASTPrettyPrinter::operator()(TypeIdAST* ast) {
  if (!ast) return;

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    operator()(it->value);
  }

  operator()(ast->declarator);
}

void ASTPrettyPrinter::operator()(HandlerAST* ast) {
  if (!ast) return;

  if (ast->catchLoc) {
    writeToken(ast->catchLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    writeToken(ast->lparenLoc);
    nospace();
  }
  operator()(ast->exceptionDeclaration);
  if (ast->rparenLoc) {
    nospace();
    writeToken(ast->rparenLoc);
  }
  operator()(ast->statement);
}

void ASTPrettyPrinter::operator()(BaseSpecifierAST* ast) {
  if (!ast) return;

  for (auto it = ast->attributeList; it; it = it->next) {
    operator()(it->value);
  }

  if (ast->virtualOrAccessLoc) {
    writeToken(ast->virtualOrAccessLoc);
  }
  if (ast->otherVirtualOrAccessLoc) {
    writeToken(ast->otherVirtualOrAccessLoc);
  }
  operator()(ast->nestedNameSpecifier);
  if (ast->templateLoc) {
    writeToken(ast->templateLoc);
  }
  operator()(ast->unqualifiedId);
  if (ast->ellipsisLoc) {
    writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::operator()(RequiresClauseAST* ast) {
  if (!ast) return;

  if (ast->requiresLoc) {
    writeToken(ast->requiresLoc);
  }
  operator()(ast->expression);
}

void ASTPrettyPrinter::operator()(ParameterDeclarationClauseAST* ast) {
  if (!ast) return;

  for (auto it = ast->parameterDeclarationList; it; it = it->next) {
    operator()(it->value);
    if (it->next) {
      nospace();
      write(",");
    }
  }

  if (ast->commaLoc) {
    nospace();
    writeToken(ast->commaLoc);
  }
  if (ast->ellipsisLoc) {
    writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::operator()(TrailingReturnTypeAST* ast) {
  if (!ast) return;

  if (ast->minusGreaterLoc) {
    writeToken(ast->minusGreaterLoc);
    space();
  }
  operator()(ast->typeId);
}

void ASTPrettyPrinter::operator()(LambdaSpecifierAST* ast) {
  if (!ast) return;

  if (ast->specifierLoc) {
    writeToken(ast->specifierLoc);
  }
}

void ASTPrettyPrinter::operator()(TypeConstraintAST* ast) {
  if (!ast) return;

  operator()(ast->nestedNameSpecifier);
  if (ast->identifierLoc) {
    writeToken(ast->identifierLoc);
  }
  if (ast->lessLoc) {
    writeToken(ast->lessLoc);
    nospace();
  }

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    operator()(it->value);
    if (it->next) {
      nospace();
      write(",");
    }
  }

  if (ast->greaterLoc) {
    nospace();
    writeToken(ast->greaterLoc);
  }
}

void ASTPrettyPrinter::operator()(AttributeArgumentClauseAST* ast) {
  if (!ast) return;

  if (ast->lparenLoc) {
    nospace();
    writeToken(ast->lparenLoc);
    nospace();
  }
  if (ast->rparenLoc) {
    nospace();
    writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::operator()(AttributeAST* ast) {
  if (!ast) return;

  operator()(ast->attributeToken);
  operator()(ast->attributeArgumentClause);
  if (ast->ellipsisLoc) {
    writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::operator()(AttributeUsingPrefixAST* ast) {
  if (!ast) return;

  if (ast->usingLoc) {
    writeToken(ast->usingLoc);
  }
  if (ast->attributeNamespaceLoc) {
    writeToken(ast->attributeNamespaceLoc);
  }
  if (ast->colonLoc) {
    nospace();
    writeToken(ast->colonLoc);
  }
}

void ASTPrettyPrinter::operator()(NewPlacementAST* ast) {
  if (!ast) return;

  if (ast->lparenLoc) {
    nospace();
    writeToken(ast->lparenLoc);
    nospace();
  }

  for (auto it = ast->expressionList; it; it = it->next) {
    operator()(it->value);
    if (it->next) {
      nospace();
      write(",");
    }
  }

  if (ast->rparenLoc) {
    nospace();
    writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::operator()(NestedNamespaceSpecifierAST* ast) {
  if (!ast) return;

  if (ast->inlineLoc) {
    writeToken(ast->inlineLoc);
  }
  if (ast->identifierLoc) {
    writeToken(ast->identifierLoc);
  }
  if (ast->scopeLoc) {
    nospace();
    writeToken(ast->scopeLoc);
    nospace();
  }
}

void ASTPrettyPrinter::UnitVisitor::operator()(TranslationUnitAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next) {
    accept(it->value);
  }
}

void ASTPrettyPrinter::UnitVisitor::operator()(ModuleUnitAST* ast) {
  accept(ast->globalModuleFragment);
  accept(ast->moduleDeclaration);

  for (auto it = ast->declarationList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->privateModuleFragment);
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    SimpleDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->initDeclaratorList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  accept(ast->requiresClause);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(AsmDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->asmQualifierList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->asmLoc) {
    accept.writeToken(ast->asmLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
  if (ast->outputOperandList) accept.write(":");

  for (auto it = ast->outputOperandList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->inputOperandList) accept.write(":");

  for (auto it = ast->inputOperandList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->clobberList) accept.write(":");

  for (auto it = ast->clobberList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->gotoLabelList) accept.write(":");

  for (auto it = ast->gotoLabelList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    NamespaceAliasDefinitionAST* ast) {
  if (ast->namespaceLoc) {
    accept.writeToken(ast->namespaceLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    UsingDeclarationAST* ast) {
  if (ast->usingLoc) {
    accept.writeToken(ast->usingLoc);
  }

  for (auto it = ast->usingDeclaratorList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    UsingEnumDeclarationAST* ast) {
  if (ast->usingLoc) {
    accept.writeToken(ast->usingLoc);
  }
  accept(ast->enumTypeSpecifier);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(UsingDirectiveAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->usingLoc) {
    accept.writeToken(ast->usingLoc);
  }
  if (ast->namespaceLoc) {
    accept.writeToken(ast->namespaceLoc);
  }
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    StaticAssertDeclarationAST* ast) {
  if (ast->staticAssertLoc) {
    accept.writeToken(ast->staticAssertLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->commaLoc) {
    nospace();
    accept.writeToken(ast->commaLoc);
  }
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    AliasDeclarationAST* ast) {
  if (ast->usingLoc) {
    accept.writeToken(ast->usingLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }

  for (auto it = ast->gnuAttributeList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->typeId);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    OpaqueEnumDeclarationAST* ast) {
  if (ast->enumLoc) {
    accept.writeToken(ast->enumLoc);
  }
  if (ast->classLoc) {
    accept.writeToken(ast->classLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
  }

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->emicolonLoc) {
    accept.writeToken(ast->emicolonLoc);
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    FunctionDefinitionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->declarator);
  accept(ast->requiresClause);
  accept(ast->functionBody);
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    TemplateDeclarationAST* ast) {
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  if (ast->lessLoc) {
    accept.writeToken(ast->lessLoc);
    nospace();
  }

  for (auto it = ast->templateParameterList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->greaterLoc) {
    nospace();
    accept.writeToken(ast->greaterLoc);
    newline();
  }
  accept(ast->requiresClause);
  accept(ast->declaration);
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    ConceptDefinitionAST* ast) {
  if (ast->conceptLoc) {
    accept.writeToken(ast->conceptLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }
  accept(ast->expression);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(DeductionGuideAST* ast) {
  accept(ast->explicitSpecifier);
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->parameterDeclarationClause);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  if (ast->arrowLoc) {
    accept.writeToken(ast->arrowLoc);
  }
  accept(ast->templateId);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    ExplicitInstantiationAST* ast) {
  if (ast->externLoc) {
    accept.writeToken(ast->externLoc);
  }
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->declaration);
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    ExportDeclarationAST* ast) {
  if (ast->exportLoc) {
    accept.writeToken(ast->exportLoc);
  }
  accept(ast->declaration);
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    ExportCompoundDeclarationAST* ast) {
  if (ast->exportLoc) {
    accept.writeToken(ast->exportLoc);
  }
  if (ast->lbraceLoc) {
    space();
    accept.writeToken(ast->lbraceLoc);
    indent();
    newline();
  }

  for (auto it = ast->declarationList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->rbraceLoc) {
    unindent();
    newline();
    accept.writeToken(ast->rbraceLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    LinkageSpecificationAST* ast) {
  if (ast->externLoc) {
    accept.writeToken(ast->externLoc);
  }
  if (ast->stringliteralLoc) {
    accept.writeToken(ast->stringliteralLoc);
  }
  if (ast->lbraceLoc) {
    space();
    accept.writeToken(ast->lbraceLoc);
    indent();
    newline();
  }

  for (auto it = ast->declarationList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->rbraceLoc) {
    unindent();
    newline();
    accept.writeToken(ast->rbraceLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    NamespaceDefinitionAST* ast) {
  if (ast->inlineLoc) {
    accept.writeToken(ast->inlineLoc);
  }
  if (ast->namespaceLoc) {
    accept.writeToken(ast->namespaceLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->nestedNamespaceSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }

  for (auto it = ast->extraAttributeList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->lbraceLoc) {
    space();
    accept.writeToken(ast->lbraceLoc);
    indent();
    newline();
  }

  for (auto it = ast->declarationList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->rbraceLoc) {
    unindent();
    newline();
    accept.writeToken(ast->rbraceLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    EmptyDeclarationAST* ast) {
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    AttributeDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    ModuleImportDeclarationAST* ast) {
  if (ast->importLoc) {
    accept.writeToken(ast->importLoc);
  }
  accept(ast->importName);

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    ParameterDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->thisLoc) {
    accept.writeToken(ast->thisLoc);
  }

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->declarator);
  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    AccessDeclarationAST* ast) {
  if (ast->accessLoc) {
    nospace();
    accept.writeToken(ast->accessLoc);
    nospace();
  }
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    ForRangeDeclarationAST* ast) {}

void ASTPrettyPrinter::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->refQualifierLoc) {
    accept.writeToken(ast->refQualifierLoc);
  }
  if (ast->lbracketLoc) {
    nospace();
    accept.writeToken(ast->lbracketLoc);
    nospace();
  }

  for (auto it = ast->bindingList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rbracketLoc) {
    nospace();
    accept.writeToken(ast->rbracketLoc);
  }
  accept(ast->initializer);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(AsmOperandAST* ast) {
  if (ast->lbracketLoc) {
    nospace();
    accept.writeToken(ast->lbracketLoc);
    nospace();
  }
  if (ast->symbolicNameLoc) {
    accept.writeToken(ast->symbolicNameLoc);
  }
  if (ast->rbracketLoc) {
    nospace();
    accept.writeToken(ast->rbracketLoc);
  }
  if (ast->constraintLiteralLoc) {
    accept.writeToken(ast->constraintLiteralLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(AsmQualifierAST* ast) {
  if (ast->qualifierLoc) {
    accept.writeToken(ast->qualifierLoc);
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(AsmClobberAST* ast) {
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
}

void ASTPrettyPrinter::DeclarationVisitor::operator()(AsmGotoLabelAST* ast) {
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(LabeledStatementAST* ast) {
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(CaseStatementAST* ast) {
  if (ast->caseLoc) {
    accept.writeToken(ast->caseLoc);
  }
  accept(ast->expression);
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(DefaultStatementAST* ast) {
  if (ast->defaultLoc) {
    accept.writeToken(ast->defaultLoc);
  }
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(
    ExpressionStatementAST* ast) {
  accept(ast->expression);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(CompoundStatementAST* ast) {
  if (ast->lbraceLoc) {
    space();
    accept.writeToken(ast->lbraceLoc);
    indent();
    newline();
  }

  for (auto it = ast->statementList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->rbraceLoc) {
    unindent();
    newline();
    accept.writeToken(ast->rbraceLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(IfStatementAST* ast) {
  if (ast->ifLoc) {
    accept.writeToken(ast->ifLoc);
  }
  if (ast->constexprLoc) {
    accept.writeToken(ast->constexprLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->initializer);
  accept(ast->condition);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  accept(ast->statement);
  if (ast->elseLoc) {
    accept.writeToken(ast->elseLoc);
  }
  accept(ast->elseStatement);
}

void ASTPrettyPrinter::StatementVisitor::operator()(
    ConstevalIfStatementAST* ast) {
  if (ast->ifLoc) {
    accept.writeToken(ast->ifLoc);
  }
  if (ast->exclaimLoc) {
    accept.writeToken(ast->exclaimLoc);
  }
  if (ast->constvalLoc) {
    accept.writeToken(ast->constvalLoc);
  }
  accept(ast->statement);
  if (ast->elseLoc) {
    accept.writeToken(ast->elseLoc);
  }
  accept(ast->elseStatement);
}

void ASTPrettyPrinter::StatementVisitor::operator()(SwitchStatementAST* ast) {
  if (ast->switchLoc) {
    accept.writeToken(ast->switchLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->initializer);
  accept(ast->condition);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  accept(ast->statement);
}

void ASTPrettyPrinter::StatementVisitor::operator()(WhileStatementAST* ast) {
  if (ast->whileLoc) {
    accept.writeToken(ast->whileLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->condition);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  accept(ast->statement);
}

void ASTPrettyPrinter::StatementVisitor::operator()(DoStatementAST* ast) {
  if (ast->doLoc) {
    accept.writeToken(ast->doLoc);
  }
  accept(ast->statement);
  if (ast->whileLoc) {
    accept.writeToken(ast->whileLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(ForRangeStatementAST* ast) {
  if (ast->forLoc) {
    accept.writeToken(ast->forLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->initializer);
  accept(ast->rangeDeclaration);
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
  }
  accept(ast->rangeInitializer);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  accept(ast->statement);
}

void ASTPrettyPrinter::StatementVisitor::operator()(ForStatementAST* ast) {
  if (ast->forLoc) {
    accept.writeToken(ast->forLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->initializer);
  accept(ast->condition);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  accept(ast->statement);
}

void ASTPrettyPrinter::StatementVisitor::operator()(BreakStatementAST* ast) {
  if (ast->breakLoc) {
    accept.writeToken(ast->breakLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(ContinueStatementAST* ast) {
  if (ast->continueLoc) {
    accept.writeToken(ast->continueLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(ReturnStatementAST* ast) {
  if (ast->returnLoc) {
    accept.writeToken(ast->returnLoc);
  }
  accept(ast->expression);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(
    CoroutineReturnStatementAST* ast) {
  if (ast->coreturnLoc) {
    accept.writeToken(ast->coreturnLoc);
  }
  accept(ast->expression);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(GotoStatementAST* ast) {
  if (ast->gotoLoc) {
    accept.writeToken(ast->gotoLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::StatementVisitor::operator()(
    DeclarationStatementAST* ast) {
  accept(ast->declaration);
}

void ASTPrettyPrinter::StatementVisitor::operator()(TryBlockStatementAST* ast) {
  if (ast->tryLoc) {
    accept.writeToken(ast->tryLoc);
  }
  accept(ast->statement);

  for (auto it = ast->handlerList; it; it = it->next) {
    accept(it->value);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    GeneratedLiteralExpressionAST* ast) {
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    CharLiteralExpressionAST* ast) {
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    BoolLiteralExpressionAST* ast) {
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    IntLiteralExpressionAST* ast) {
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    FloatLiteralExpressionAST* ast) {
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    NullptrLiteralExpressionAST* ast) {
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    StringLiteralExpressionAST* ast) {
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) {
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(ThisExpressionAST* ast) {
  if (ast->thisLoc) {
    accept.writeToken(ast->thisLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    NestedStatementExpressionAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->statement);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(NestedExpressionAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(IdExpressionAST* ast) {
  accept(ast->nestedNameSpecifier);
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->unqualifiedId);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(LambdaExpressionAST* ast) {
  if (ast->lbracketLoc) {
    nospace();
    accept.writeToken(ast->lbracketLoc);
    nospace();
  }
  if (ast->captureDefaultLoc) {
    accept.writeToken(ast->captureDefaultLoc);
    if (ast->captureList) accept.write(",");
  }

  for (auto it = ast->captureList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rbracketLoc) {
    nospace();
    accept.writeToken(ast->rbracketLoc);
  }
  if (ast->lessLoc) {
    accept.writeToken(ast->lessLoc);
    nospace();
  }

  for (auto it = ast->templateParameterList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->greaterLoc) {
    nospace();
    accept.writeToken(ast->greaterLoc);
  }
  accept(ast->templateRequiresClause);
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->parameterDeclarationClause);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }

  for (auto it = ast->gnuAtributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->lambdaSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->exceptionSpecifier);

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->trailingReturnType);
  accept(ast->requiresClause);
  accept(ast->statement);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(FoldExpressionAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->leftExpression);
  if (ast->opLoc) {
    accept.write("{}", Token::spell(ast->op));
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->foldOpLoc) {
    accept.writeToken(ast->foldOpLoc);
  }
  accept(ast->rightExpression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    RightFoldExpressionAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->opLoc) {
    accept.write("{}", Token::spell(ast->op));
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    LeftFoldExpressionAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->opLoc) {
    accept.write("{}", Token::spell(ast->op));
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    RequiresExpressionAST* ast) {
  if (ast->requiresLoc) {
    accept.writeToken(ast->requiresLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->parameterDeclarationClause);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  if (ast->lbraceLoc) {
    space();
    accept.writeToken(ast->lbraceLoc);
    indent();
    newline();
  }

  for (auto it = ast->requirementList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->rbraceLoc) {
    unindent();
    newline();
    accept.writeToken(ast->rbraceLoc);
    newline();
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(VaArgExpressionAST* ast) {
  if (ast->vaArgLoc) {
    accept.writeToken(ast->vaArgLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->commaLoc) {
    nospace();
    accept.writeToken(ast->commaLoc);
  }
  accept(ast->typeId);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    SubscriptExpressionAST* ast) {
  accept(ast->baseExpression);
  if (ast->lbracketLoc) {
    nospace();
    accept.writeToken(ast->lbracketLoc);
    nospace();
  }
  accept(ast->indexExpression);
  if (ast->rbracketLoc) {
    nospace();
    accept.writeToken(ast->rbracketLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(CallExpressionAST* ast) {
  accept(ast->baseExpression);
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }

  for (auto it = ast->expressionList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(TypeConstructionAST* ast) {
  accept(ast->typeSpecifier);
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }

  for (auto it = ast->expressionList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    BracedTypeConstructionAST* ast) {
  accept(ast->typeSpecifier);
  accept(ast->bracedInitList);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    SpliceMemberExpressionAST* ast) {
  accept(ast->baseExpression);
  if (ast->accessLoc) {
    nospace();
    accept.writeToken(ast->accessLoc);
    nospace();
  }
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->splicer);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(MemberExpressionAST* ast) {
  accept(ast->baseExpression);
  if (ast->accessLoc) {
    nospace();
    accept.writeToken(ast->accessLoc);
    nospace();
  }
  accept(ast->nestedNameSpecifier);
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->unqualifiedId);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    PostIncrExpressionAST* ast) {
  accept(ast->baseExpression);
  if (ast->opLoc) {
    accept.write("{}", Token::spell(ast->op));
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    CppCastExpressionAST* ast) {
  if (ast->castLoc) {
    accept.writeToken(ast->castLoc);
  }
  if (ast->lessLoc) {
    accept.writeToken(ast->lessLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->greaterLoc) {
    nospace();
    accept.writeToken(ast->greaterLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    BuiltinBitCastExpressionAST* ast) {
  if (ast->castLoc) {
    accept.writeToken(ast->castLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->commaLoc) {
    nospace();
    accept.writeToken(ast->commaLoc);
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    BuiltinOffsetofExpressionAST* ast) {
  if (ast->offsetofLoc) {
    accept.writeToken(ast->offsetofLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->commaLoc) {
    nospace();
    accept.writeToken(ast->commaLoc);
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(TypeidExpressionAST* ast) {
  if (ast->typeidLoc) {
    accept.writeToken(ast->typeidLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    TypeidOfTypeExpressionAST* ast) {
  if (ast->typeidLoc) {
    accept.writeToken(ast->typeidLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(SpliceExpressionAST* ast) {
  accept(ast->splicer);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) {
  if (ast->caretLoc) {
    accept.writeToken(ast->caretLoc);
  }
  if (ast->scopeLoc) {
    nospace();
    accept.writeToken(ast->scopeLoc);
    nospace();
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    NamespaceReflectExpressionAST* ast) {
  if (ast->caretLoc) {
    accept.writeToken(ast->caretLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    TypeIdReflectExpressionAST* ast) {
  if (ast->caretLoc) {
    accept.writeToken(ast->caretLoc);
  }
  accept(ast->typeId);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    ReflectExpressionAST* ast) {
  if (ast->caretLoc) {
    accept.writeToken(ast->caretLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(UnaryExpressionAST* ast) {
  if (ast->opLoc) {
    accept.write("{}", Token::spell(ast->op));
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(AwaitExpressionAST* ast) {
  if (ast->awaitLoc) {
    accept.writeToken(ast->awaitLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(SizeofExpressionAST* ast) {
  if (ast->sizeofLoc) {
    accept.writeToken(ast->sizeofLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    SizeofTypeExpressionAST* ast) {
  if (ast->sizeofLoc) {
    accept.writeToken(ast->sizeofLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    SizeofPackExpressionAST* ast) {
  if (ast->sizeofLoc) {
    accept.writeToken(ast->sizeofLoc);
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    AlignofTypeExpressionAST* ast) {
  if (ast->alignofLoc) {
    accept.writeToken(ast->alignofLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    AlignofExpressionAST* ast) {
  if (ast->alignofLoc) {
    accept.writeToken(ast->alignofLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    NoexceptExpressionAST* ast) {
  if (ast->noexceptLoc) {
    accept.writeToken(ast->noexceptLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(NewExpressionAST* ast) {
  if (ast->scopeLoc) {
    nospace();
    accept.writeToken(ast->scopeLoc);
    nospace();
  }
  if (ast->newLoc) {
    accept.writeToken(ast->newLoc);
  }
  accept(ast->newPlacement);
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->declarator);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  accept(ast->newInitalizer);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(DeleteExpressionAST* ast) {
  if (ast->scopeLoc) {
    nospace();
    accept.writeToken(ast->scopeLoc);
    nospace();
  }
  if (ast->deleteLoc) {
    accept.writeToken(ast->deleteLoc);
  }
  if (ast->lbracketLoc) {
    nospace();
    accept.writeToken(ast->lbracketLoc);
    nospace();
  }
  if (ast->rbracketLoc) {
    nospace();
    accept.writeToken(ast->rbracketLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(CastExpressionAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    ImplicitCastExpressionAST* ast) {
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(BinaryExpressionAST* ast) {
  accept(ast->leftExpression);
  if (ast->opLoc) {
    accept.write("{}", Token::spell(ast->op));
  }
  accept(ast->rightExpression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    ConditionalExpressionAST* ast) {
  accept(ast->condition);
  if (ast->questionLoc) {
    accept.writeToken(ast->questionLoc);
  }
  accept(ast->iftrueExpression);
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
  }
  accept(ast->iffalseExpression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(YieldExpressionAST* ast) {
  if (ast->yieldLoc) {
    accept.writeToken(ast->yieldLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(ThrowExpressionAST* ast) {
  if (ast->throwLoc) {
    accept.writeToken(ast->throwLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    AssignmentExpressionAST* ast) {
  accept(ast->leftExpression);
  if (ast->opLoc) {
    accept.write("{}", Token::spell(ast->op));
  }
  accept(ast->rightExpression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    PackExpansionExpressionAST* ast) {
  accept(ast->expression);
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    DesignatedInitializerClauseAST* ast) {
  if (ast->dotLoc) {
    nospace();
    accept.writeToken(ast->dotLoc);
    nospace();
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  accept(ast->initializer);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    TypeTraitExpressionAST* ast) {
  if (ast->typeTraitLoc) {
    accept.writeToken(ast->typeTraitLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }

  for (auto it = ast->typeIdList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(
    ConditionExpressionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->declarator);
  accept(ast->initializer);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(EqualInitializerAST* ast) {
  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }
  accept(ast->expression);
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(BracedInitListAST* ast) {
  if (ast->lbraceLoc) {
    space();
    accept.writeToken(ast->lbraceLoc);
    indent();
  }

  for (auto it = ast->expressionList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->commaLoc) {
    nospace();
    accept.writeToken(ast->commaLoc);
  }
  if (ast->rbraceLoc) {
    unindent();
    accept.writeToken(ast->rbraceLoc);
  }
}

void ASTPrettyPrinter::ExpressionVisitor::operator()(ParenInitializerAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }

  for (auto it = ast->expressionList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) {
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  if (ast->lessLoc) {
    accept.writeToken(ast->lessLoc);
    nospace();
  }

  for (auto it = ast->templateParameterList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->greaterLoc) {
    nospace();
    accept.writeToken(ast->greaterLoc);
  }
  accept(ast->requiresClause);
  if (ast->classKeyLoc) {
    accept.writeToken(ast->classKeyLoc);
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }
  accept(ast->idExpression);
}

void ASTPrettyPrinter::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) {
  accept(ast->declaration);
}

void ASTPrettyPrinter::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) {
  if (ast->classKeyLoc) {
    accept.writeToken(ast->classKeyLoc);
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }
  accept(ast->typeId);
}

void ASTPrettyPrinter::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) {
  accept(ast->typeConstraint);
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }
  accept(ast->typeId);
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    GeneratedTypeSpecifierAST* ast) {
  if (ast->typeLoc) {
    accept.writeToken(ast->typeLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(TypedefSpecifierAST* ast) {
  if (ast->typedefLoc) {
    accept.writeToken(ast->typedefLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(FriendSpecifierAST* ast) {
  if (ast->friendLoc) {
    accept.writeToken(ast->friendLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    ConstevalSpecifierAST* ast) {
  if (ast->constevalLoc) {
    accept.writeToken(ast->constevalLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    ConstinitSpecifierAST* ast) {
  if (ast->constinitLoc) {
    accept.writeToken(ast->constinitLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    ConstexprSpecifierAST* ast) {
  if (ast->constexprLoc) {
    accept.writeToken(ast->constexprLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(InlineSpecifierAST* ast) {
  if (ast->inlineLoc) {
    accept.writeToken(ast->inlineLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(StaticSpecifierAST* ast) {
  if (ast->staticLoc) {
    accept.writeToken(ast->staticLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(ExternSpecifierAST* ast) {
  if (ast->externLoc) {
    accept.writeToken(ast->externLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    ThreadLocalSpecifierAST* ast) {
  if (ast->threadLocalLoc) {
    accept.writeToken(ast->threadLocalLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(ThreadSpecifierAST* ast) {
  if (ast->threadLoc) {
    accept.writeToken(ast->threadLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(MutableSpecifierAST* ast) {
  if (ast->mutableLoc) {
    accept.writeToken(ast->mutableLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(VirtualSpecifierAST* ast) {
  if (ast->virtualLoc) {
    accept.writeToken(ast->virtualLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(ExplicitSpecifierAST* ast) {
  if (ast->explicitLoc) {
    accept.writeToken(ast->explicitLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(AutoTypeSpecifierAST* ast) {
  if (ast->autoLoc) {
    accept.writeToken(ast->autoLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(VoidTypeSpecifierAST* ast) {
  if (ast->voidLoc) {
    accept.writeToken(ast->voidLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(SizeTypeSpecifierAST* ast) {
  if (ast->specifierLoc) {
    accept.writeToken(ast->specifierLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(SignTypeSpecifierAST* ast) {
  if (ast->specifierLoc) {
    accept.writeToken(ast->specifierLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    VaListTypeSpecifierAST* ast) {
  if (ast->specifierLoc) {
    accept.writeToken(ast->specifierLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    IntegralTypeSpecifierAST* ast) {
  if (ast->specifierLoc) {
    accept.writeToken(ast->specifierLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    FloatingPointTypeSpecifierAST* ast) {
  if (ast->specifierLoc) {
    accept.writeToken(ast->specifierLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    ComplexTypeSpecifierAST* ast) {
  if (ast->complexLoc) {
    accept.writeToken(ast->complexLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    NamedTypeSpecifierAST* ast) {
  accept(ast->nestedNameSpecifier);
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->unqualifiedId);
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    AtomicTypeSpecifierAST* ast) {
  if (ast->atomicLoc) {
    accept.writeToken(ast->atomicLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    UnderlyingTypeSpecifierAST* ast) {
  if (ast->underlyingTypeLoc) {
    accept.writeToken(ast->underlyingTypeLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    ElaboratedTypeSpecifierAST* ast) {
  if (ast->classLoc) {
    accept.writeToken(ast->classLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->nestedNameSpecifier);
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->unqualifiedId);
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    DecltypeAutoSpecifierAST* ast) {
  if (ast->decltypeLoc) {
    accept.writeToken(ast->decltypeLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  if (ast->autoLoc) {
    accept.writeToken(ast->autoLoc);
  }
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(DecltypeSpecifierAST* ast) {
  if (ast->decltypeLoc) {
    accept.writeToken(ast->decltypeLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    PlaceholderTypeSpecifierAST* ast) {
  accept(ast->typeConstraint);
  accept(ast->specifier);
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(ConstQualifierAST* ast) {
  if (ast->constLoc) {
    accept.writeToken(ast->constLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(VolatileQualifierAST* ast) {
  if (ast->volatileLoc) {
    accept.writeToken(ast->volatileLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(RestrictQualifierAST* ast) {
  if (ast->restrictLoc) {
    accept.writeToken(ast->restrictLoc);
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(EnumSpecifierAST* ast) {
  if (ast->enumLoc) {
    accept.writeToken(ast->enumLoc);
  }
  if (ast->classLoc) {
    accept.writeToken(ast->classLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
  }

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->lbraceLoc) {
    space();
    accept.writeToken(ast->lbraceLoc);
    indent();
    newline();
  }

  for (auto it = ast->enumeratorList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
      newline();
    }
  }

  if (ast->commaLoc) {
    nospace();
    accept.writeToken(ast->commaLoc);
  }
  if (ast->rbraceLoc) {
    unindent();
    newline();
    accept.writeToken(ast->rbraceLoc);
    newline();
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(ClassSpecifierAST* ast) {
  if (ast->classLoc) {
    accept.writeToken(ast->classLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  if (ast->finalLoc) {
    accept.writeToken(ast->finalLoc);
  }
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
  }

  for (auto it = ast->baseSpecifierList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->lbraceLoc) {
    space();
    accept.writeToken(ast->lbraceLoc);
    indent();
    newline();
  }

  for (auto it = ast->declarationList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->rbraceLoc) {
    unindent();
    newline();
    accept.writeToken(ast->rbraceLoc);
    newline();
  }
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(TypenameSpecifierAST* ast) {
  if (ast->typenameLoc) {
    accept.writeToken(ast->typenameLoc);
  }
  accept(ast->nestedNameSpecifier);
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->unqualifiedId);
}

void ASTPrettyPrinter::SpecifierVisitor::operator()(
    SplicerTypeSpecifierAST* ast) {
  if (ast->typenameLoc) {
    accept.writeToken(ast->typenameLoc);
  }
  accept(ast->splicer);
}

void ASTPrettyPrinter::PtrOperatorVisitor::operator()(PointerOperatorAST* ast) {
  if (ast->starLoc) {
    accept.writeToken(ast->starLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->cvQualifierList; it; it = it->next) {
    accept(it->value);
  }
}

void ASTPrettyPrinter::PtrOperatorVisitor::operator()(
    ReferenceOperatorAST* ast) {
  if (ast->refLoc) {
    accept.writeToken(ast->refLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }
}

void ASTPrettyPrinter::PtrOperatorVisitor::operator()(
    PtrToMemberOperatorAST* ast) {
  accept(ast->nestedNameSpecifier);
  if (ast->starLoc) {
    accept.writeToken(ast->starLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->cvQualifierList; it; it = it->next) {
    accept(it->value);
  }
}

void ASTPrettyPrinter::CoreDeclaratorVisitor::operator()(
    BitfieldDeclaratorAST* ast) {
  accept(ast->unqualifiedId);
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
  }
  accept(ast->sizeExpression);
}

void ASTPrettyPrinter::CoreDeclaratorVisitor::operator()(
    ParameterPackAST* ast) {
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  accept(ast->coreDeclarator);
}

void ASTPrettyPrinter::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast) {
  accept(ast->nestedNameSpecifier);
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->unqualifiedId);

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }
}

void ASTPrettyPrinter::CoreDeclaratorVisitor::operator()(
    NestedDeclaratorAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->declarator);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->parameterDeclarationClause);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }

  for (auto it = ast->cvQualifierList; it; it = it->next) {
    accept(it->value);
  }

  if (ast->refLoc) {
    accept.writeToken(ast->refLoc);
  }
  accept(ast->exceptionSpecifier);

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->trailingReturnType);
}

void ASTPrettyPrinter::DeclaratorChunkVisitor::operator()(
    ArrayDeclaratorChunkAST* ast) {
  if (ast->lbracketLoc) {
    nospace();
    accept.writeToken(ast->lbracketLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rbracketLoc) {
    nospace();
    accept.writeToken(ast->rbracketLoc);
  }

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }
}

void ASTPrettyPrinter::UnqualifiedIdVisitor::operator()(NameIdAST* ast) {
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
}

void ASTPrettyPrinter::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast) {
  if (ast->tildeLoc) {
    accept.writeToken(ast->tildeLoc);
  }
  accept(ast->id);
}

void ASTPrettyPrinter::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast) {
  accept(ast->decltypeSpecifier);
}

void ASTPrettyPrinter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionIdAST* ast) {
  if (ast->operatorLoc) {
    accept.writeToken(ast->operatorLoc);
  }
  if (ast->opLoc) {
    if (ast->op == TokenKind::T_NEW_ARRAY) {
      accept.write("new");
    } else if (ast->op == TokenKind::T_DELETE_ARRAY) {
      accept.write("delete");
    } else if (ast->op != TokenKind::T_LPAREN &&
               ast->op != TokenKind::T_LBRACKET) {
      accept.write("{}", Token::spell(ast->op));
    }
  }
  if (ast->openLoc) {
    nospace();
    accept.writeToken(ast->openLoc);
    nospace();
  }
  if (ast->closeLoc) {
    nospace();
    accept.writeToken(ast->closeLoc);
    nospace();
  }
}

void ASTPrettyPrinter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorIdAST* ast) {
  if (ast->operatorLoc) {
    accept.writeToken(ast->operatorLoc);
  }
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
}

void ASTPrettyPrinter::UnqualifiedIdVisitor::operator()(
    ConversionFunctionIdAST* ast) {
  if (ast->operatorLoc) {
    accept.writeToken(ast->operatorLoc);
  }
  accept(ast->typeId);
}

void ASTPrettyPrinter::UnqualifiedIdVisitor::operator()(
    SimpleTemplateIdAST* ast) {
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->lessLoc) {
    accept.writeToken(ast->lessLoc);
    nospace();
  }

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->greaterLoc) {
    nospace();
    accept.writeToken(ast->greaterLoc);
  }
}

void ASTPrettyPrinter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) {
  accept(ast->literalOperatorId);
  if (ast->lessLoc) {
    accept.writeToken(ast->lessLoc);
    nospace();
  }

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->greaterLoc) {
    nospace();
    accept.writeToken(ast->greaterLoc);
  }
}

void ASTPrettyPrinter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) {
  accept(ast->operatorFunctionId);
  if (ast->lessLoc) {
    accept.writeToken(ast->lessLoc);
    nospace();
  }

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->greaterLoc) {
    nospace();
    accept.writeToken(ast->greaterLoc);
  }
}

void ASTPrettyPrinter::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) {
  if (ast->scopeLoc) {
    nospace();
    accept.writeToken(ast->scopeLoc);
    nospace();
  }
}

void ASTPrettyPrinter::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) {
  accept(ast->nestedNameSpecifier);
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->scopeLoc) {
    nospace();
    accept.writeToken(ast->scopeLoc);
    nospace();
  }
}

void ASTPrettyPrinter::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) {
  accept(ast->decltypeSpecifier);
  if (ast->scopeLoc) {
    nospace();
    accept.writeToken(ast->scopeLoc);
    nospace();
  }
}

void ASTPrettyPrinter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) {
  accept(ast->nestedNameSpecifier);
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->templateId);
  if (ast->scopeLoc) {
    nospace();
    accept.writeToken(ast->scopeLoc);
    nospace();
  }
}

void ASTPrettyPrinter::FunctionBodyVisitor::operator()(
    DefaultFunctionBodyAST* ast) {
  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }
  if (ast->defaultLoc) {
    accept.writeToken(ast->defaultLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) {
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
  }

  for (auto it = ast->memInitializerList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  accept(ast->statement);
}

void ASTPrettyPrinter::FunctionBodyVisitor::operator()(
    TryStatementFunctionBodyAST* ast) {
  if (ast->tryLoc) {
    accept.writeToken(ast->tryLoc);
  }
  if (ast->colonLoc) {
    nospace();
    accept.writeToken(ast->colonLoc);
  }

  for (auto it = ast->memInitializerList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  accept(ast->statement);

  for (auto it = ast->handlerList; it; it = it->next) {
    accept(it->value);
  }
}

void ASTPrettyPrinter::FunctionBodyVisitor::operator()(
    DeleteFunctionBodyAST* ast) {
  if (ast->equalLoc) {
    space();
    accept.writeToken(ast->equalLoc);
  }
  if (ast->deleteLoc) {
    accept.writeToken(ast->deleteLoc);
  }
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::TemplateArgumentVisitor::operator()(
    TypeTemplateArgumentAST* ast) {
  accept(ast->typeId);
}

void ASTPrettyPrinter::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) {
  accept(ast->expression);
}

void ASTPrettyPrinter::ExceptionSpecifierVisitor::operator()(
    ThrowExceptionSpecifierAST* ast) {
  if (ast->throwLoc) {
    accept.writeToken(ast->throwLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::ExceptionSpecifierVisitor::operator()(
    NoexceptSpecifierAST* ast) {
  if (ast->noexceptLoc) {
    accept.writeToken(ast->noexceptLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::RequirementVisitor::operator()(
    SimpleRequirementAST* ast) {
  accept(ast->expression);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::RequirementVisitor::operator()(
    CompoundRequirementAST* ast) {
  if (ast->lbraceLoc) {
    space();
    accept.writeToken(ast->lbraceLoc);
    indent();
    newline();
  }
  accept(ast->expression);
  if (ast->rbraceLoc) {
    unindent();
    newline();
    accept.writeToken(ast->rbraceLoc);
    newline();
  }
  if (ast->noexceptLoc) {
    accept.writeToken(ast->noexceptLoc);
  }
  if (ast->minusGreaterLoc) {
    accept.writeToken(ast->minusGreaterLoc);
    space();
  }
  accept(ast->typeConstraint);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::RequirementVisitor::operator()(TypeRequirementAST* ast) {
  if (ast->typenameLoc) {
    accept.writeToken(ast->typenameLoc);
  }
  accept(ast->nestedNameSpecifier);
  if (ast->templateLoc) {
    accept.writeToken(ast->templateLoc);
  }
  accept(ast->unqualifiedId);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::RequirementVisitor::operator()(
    NestedRequirementAST* ast) {
  if (ast->requiresLoc) {
    accept.writeToken(ast->requiresLoc);
  }
  accept(ast->expression);
  if (ast->semicolonLoc) {
    nospace();
    nonewline();
    accept.writeToken(ast->semicolonLoc);
    newline();
  }
}

void ASTPrettyPrinter::NewInitializerVisitor::operator()(
    NewParenInitializerAST* ast) {
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }

  for (auto it = ast->expressionList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::NewInitializerVisitor::operator()(
    NewBracedInitializerAST* ast) {
  accept(ast->bracedInitList);
}

void ASTPrettyPrinter::MemInitializerVisitor::operator()(
    ParenMemInitializerAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }

  for (auto it = ast->expressionList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::MemInitializerVisitor::operator()(
    BracedMemInitializerAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  accept(ast->bracedInitList);
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::LambdaCaptureVisitor::operator()(
    ThisLambdaCaptureAST* ast) {
  if (ast->thisLoc) {
    accept.writeToken(ast->thisLoc);
  }
}

void ASTPrettyPrinter::LambdaCaptureVisitor::operator()(
    DerefThisLambdaCaptureAST* ast) {
  if (ast->starLoc) {
    accept.writeToken(ast->starLoc);
  }
  if (ast->thisLoc) {
    accept.writeToken(ast->thisLoc);
  }
}

void ASTPrettyPrinter::LambdaCaptureVisitor::operator()(
    SimpleLambdaCaptureAST* ast) {
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::LambdaCaptureVisitor::operator()(
    RefLambdaCaptureAST* ast) {
  if (ast->ampLoc) {
    accept.writeToken(ast->ampLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::LambdaCaptureVisitor::operator()(
    RefInitLambdaCaptureAST* ast) {
  if (ast->ampLoc) {
    accept.writeToken(ast->ampLoc);
  }
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  accept(ast->initializer);
}

void ASTPrettyPrinter::LambdaCaptureVisitor::operator()(
    InitLambdaCaptureAST* ast) {
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
  accept(ast->initializer);
}

void ASTPrettyPrinter::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) {
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
}

void ASTPrettyPrinter::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
  }

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    accept(it->value);
  }

  accept(ast->declarator);
}

void ASTPrettyPrinter::AttributeSpecifierVisitor::operator()(
    CxxAttributeAST* ast) {
  if (ast->lbracketLoc) {
    nospace();
    accept.writeToken(ast->lbracketLoc);
    nospace();
  }
  if (ast->lbracket2Loc) {
    nospace();
    accept.writeToken(ast->lbracket2Loc);
    nospace();
  }
  accept(ast->attributeUsingPrefix);

  for (auto it = ast->attributeList; it; it = it->next) {
    accept(it->value);
    if (it->next) {
      nospace();
      accept.write(",");
    }
  }

  if (ast->rbracketLoc) {
    nospace();
    accept.writeToken(ast->rbracketLoc);
  }
  if (ast->rbracket2Loc) {
    nospace();
    accept.writeToken(ast->rbracket2Loc);
  }
}

void ASTPrettyPrinter::AttributeSpecifierVisitor::operator()(
    GccAttributeAST* ast) {
  if (ast->attributeLoc) {
    newline();
    accept.writeToken(ast->attributeLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  if (ast->lparen2Loc) {
    nospace();

    nospace();
    for (auto loc = ast->lparen2Loc; loc; loc = loc.next()) {
      if (loc == ast->rparenLoc) break;
      accept.writeToken(loc);
    }
  }
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
  if (ast->rparen2Loc) {
    nospace();
    accept.writeToken(ast->rparen2Loc);
  }
}

void ASTPrettyPrinter::AttributeSpecifierVisitor::operator()(
    AlignasAttributeAST* ast) {
  if (ast->alignasLoc) {
    accept.writeToken(ast->alignasLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->expression);
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::AttributeSpecifierVisitor::operator()(
    AlignasTypeAttributeAST* ast) {
  if (ast->alignasLoc) {
    accept.writeToken(ast->alignasLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  accept(ast->typeId);
  if (ast->ellipsisLoc) {
    accept.writeToken(ast->ellipsisLoc);
  }
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::AttributeSpecifierVisitor::operator()(
    AsmAttributeAST* ast) {
  if (ast->asmLoc) {
    accept.writeToken(ast->asmLoc);
  }
  if (ast->lparenLoc) {
    nospace();
    accept.writeToken(ast->lparenLoc);
    nospace();
  }
  if (ast->literalLoc) {
    accept.writeToken(ast->literalLoc);
  }
  if (ast->rparenLoc) {
    nospace();
    accept.writeToken(ast->rparenLoc);
  }
}

void ASTPrettyPrinter::AttributeTokenVisitor::operator()(
    ScopedAttributeTokenAST* ast) {
  if (ast->attributeNamespaceLoc) {
    accept.writeToken(ast->attributeNamespaceLoc);
  }
  if (ast->scopeLoc) {
    nospace();
    accept.writeToken(ast->scopeLoc);
    nospace();
  }
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
}

void ASTPrettyPrinter::AttributeTokenVisitor::operator()(
    SimpleAttributeTokenAST* ast) {
  if (ast->identifierLoc) {
    accept.writeToken(ast->identifierLoc);
  }
}

ASTPrettyPrinter::ASTPrettyPrinter(TranslationUnit* unit, std::ostream& out)
    : unit_(unit), output_(out) {}

ASTPrettyPrinter::~ASTPrettyPrinter() {}

auto ASTPrettyPrinter::control() const -> Control* { return unit_->control(); }

void ASTPrettyPrinter::space() {
  if (newline_) return;
  space_ = true;
}

void ASTPrettyPrinter::nospace() { space_ = false; }

void ASTPrettyPrinter::newline() {
  space_ = false;
  newline_ = true;
}

void ASTPrettyPrinter::nonewline() { newline_ = false; }

void ASTPrettyPrinter::indent() { ++depth_; }

void ASTPrettyPrinter::unindent() { --depth_; }

void ASTPrettyPrinter::writeToken(SourceLocation loc) {
  if (!loc) return;
  const auto& tk = unit_->tokenAt(loc);
  write("{}", tk.spell());
  if (!space_) cxx_runtime_error("no space");
}

}  // namespace cxx
