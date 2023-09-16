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

import { cxx } from "./cxx.js";
import { SourceLocation } from "./SourceLocation.js";
import { ASTCursor } from "./ASTCursor.js";
import { ASTVisitor } from "./ASTVisitor.js";
import { ASTKind } from "./ASTKind.js";
import { Token } from "./Token.js";
import { TokenKind } from "./TokenKind.js";

interface TranslationUnitLike {
  getUnitHandle(): number;
}

export abstract class AST {
  constructor(
    private readonly handle: number,
    private readonly kind: ASTKind,
    protected readonly parser: TranslationUnitLike,
  ) {}

  walk(): ASTCursor {
    return new ASTCursor(this, this.parser);
  }

  getKind(): ASTKind {
    return this.kind;
  }

  is(kind: ASTKind): boolean {
    return this.kind === kind;
  }

  isNot(kind: ASTKind): boolean {
    return this.kind !== kind;
  }

  getHandle() {
    return this.handle;
  }

  getStartLocation(): SourceLocation {
    return cxx.getStartLocation(this.handle, this.parser.getUnitHandle());
  }

  getEndLocation(): SourceLocation {
    return cxx.getEndLocation(this.handle, this.parser.getUnitHandle());
  }

  abstract accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result;

  static from<T extends AST = AST>(
    handle: number,
    parser: TranslationUnitLike,
  ): T | undefined {
    if (handle) {
      const kind = cxx.getASTKind(handle) as ASTKind;
      const ast = new AST_CONSTRUCTORS[kind](handle, kind, parser) as T;
      return ast;
    }
    return;
  }
}

export abstract class AttributeSpecifierAST extends AST {}
export abstract class AttributeTokenAST extends AST {}
export abstract class CoreDeclaratorAST extends AST {}
export abstract class DeclarationAST extends AST {}
export abstract class DeclaratorModifierAST extends AST {}
export abstract class ExceptionDeclarationAST extends AST {}
export abstract class ExceptionSpecifierAST extends AST {}
export abstract class ExpressionAST extends AST {}
export abstract class FunctionBodyAST extends AST {}
export abstract class LambdaCaptureAST extends AST {}
export abstract class MemInitializerAST extends AST {}
export abstract class NestedNameSpecifierAST extends AST {}
export abstract class NewInitializerAST extends AST {}
export abstract class PtrOperatorAST extends AST {}
export abstract class RequirementAST extends AST {}
export abstract class SpecifierAST extends AST {}
export abstract class StatementAST extends AST {}
export abstract class TemplateArgumentAST extends AST {}
export abstract class UnitAST extends AST {}
export abstract class UnqualifiedIdAST extends AST {}

export class TypeIdAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeId(this, context);
  }
  *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class UsingDeclaratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingDeclarator(this, context);
  }
  getTypenameToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

export class HandlerAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitHandler(this, context);
  }
  getCatchToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getExceptionDeclaration(): ExceptionDeclarationAST | undefined {
    return AST.from<ExceptionDeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

export class EnumBaseAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEnumBase(this, context);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class EnumeratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEnumerator(this, context);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getIdentifierValue(slot);
  }
}

export class DeclaratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeclarator(this, context);
  }
  *getPtrOpList(): Generator<PtrOperatorAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<PtrOperatorAST>(cxx.getListValue(it), this.parser);
    }
  }
  getCoreDeclarator(): CoreDeclaratorAST | undefined {
    return AST.from<CoreDeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  *getModifiers(): Generator<DeclaratorModifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclaratorModifierAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class InitDeclaratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitInitDeclarator(this, context);
  }
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

export class BaseSpecifierAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBaseSpecifier(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 4) !== 0;
  }
  getIsVirtual(): boolean {
    return cxx.getASTSlot(this.getHandle(), 5) !== 0;
  }
  getAccessSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 6);
  }
}

export class BaseClauseAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBaseClause(this, context);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getBaseSpecifierList(): Generator<BaseSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<BaseSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class NewDeclaratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewDeclarator(this, context);
  }
  *getPtrOpList(): Generator<PtrOperatorAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<PtrOperatorAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getModifiers(): Generator<ArrayDeclaratorAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<ArrayDeclaratorAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class NewTypeIdAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewTypeId(this, context);
  }
  *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getNewDeclarator(): NewDeclaratorAST | undefined {
    return AST.from<NewDeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class RequiresClauseAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRequiresClause(this, context);
  }
  getRequiresToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class ParameterDeclarationClauseAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParameterDeclarationClause(this, context);
  }
  *getParameterDeclarationList(): Generator<
    ParameterDeclarationAST | undefined
  > {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<ParameterDeclarationAST>(
        cxx.getListValue(it),
        this.parser,
      );
    }
  }
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getIsVariadic(): boolean {
    return cxx.getASTSlot(this.getHandle(), 3) !== 0;
  }
}

export class ParametersAndQualifiersAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParametersAndQualifiers(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return AST.from<ParameterDeclarationClauseAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  *getCvQualifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRefToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getExceptionSpecifier(): ExceptionSpecifierAST | undefined {
    return AST.from<ExceptionSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 6);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class LambdaIntroducerAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLambdaIntroducer(this, context);
  }
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getCaptureDefaultToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getCaptureList(): Generator<LambdaCaptureAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<LambdaCaptureAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class LambdaDeclaratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLambdaDeclarator(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return AST.from<ParameterDeclarationClauseAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  *getDeclSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getExceptionSpecifier(): ExceptionSpecifierAST | undefined {
    return AST.from<ExceptionSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 5);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getTrailingReturnType(): TrailingReturnTypeAST | undefined {
    return AST.from<TrailingReturnTypeAST>(
      cxx.getASTSlot(this.getHandle(), 6),
      this.parser,
    );
  }
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 7),
      this.parser,
    );
  }
}

export class TrailingReturnTypeAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTrailingReturnType(this, context);
  }
  getMinusGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class CtorInitializerAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCtorInitializer(this, context);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getMemInitializerList(): Generator<MemInitializerAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<MemInitializerAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class RequirementBodyAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRequirementBody(this, context);
  }
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getRequirementList(): Generator<RequirementAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<RequirementAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class TypeConstraintAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeConstraint(this, context);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  *getTemplateArgumentList(): Generator<TemplateArgumentAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<TemplateArgumentAST>(cxx.getListValue(it), this.parser);
    }
  }
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getIdentifierValue(slot);
  }
}

export class GlobalModuleFragmentAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGlobalModuleFragment(this, context);
  }
  getModuleToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getDeclarationList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class PrivateModuleFragmentAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPrivateModuleFragment(this, context);
  }
  getModuleToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getPrivateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  *getDeclarationList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 4);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class ModuleQualifierAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleQualifier(this, context);
  }
  getModuleQualifier(): ModuleQualifierAST | undefined {
    return AST.from<ModuleQualifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getDotToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

export class ModuleNameAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleName(this, context);
  }
  getModuleQualifier(): ModuleQualifierAST | undefined {
    return AST.from<ModuleQualifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }
}

export class ModuleDeclarationAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleDeclaration(this, context);
  }
  getExportToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getModuleToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getModuleName(): ModuleNameAST | undefined {
    return AST.from<ModuleNameAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getModulePartition(): ModulePartitionAST | undefined {
    return AST.from<ModulePartitionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 4);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

export class ImportNameAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitImportName(this, context);
  }
  getHeaderToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getModulePartition(): ModulePartitionAST | undefined {
    return AST.from<ModulePartitionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getModuleName(): ModuleNameAST | undefined {
    return AST.from<ModuleNameAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

export class ModulePartitionAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModulePartition(this, context);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getModuleName(): ModuleNameAST | undefined {
    return AST.from<ModuleNameAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class AttributeArgumentClauseAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttributeArgumentClause(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

export class AttributeAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttribute(this, context);
  }
  getAttributeToken(): AttributeTokenAST | undefined {
    return AST.from<AttributeTokenAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getAttributeArgumentClause(): AttributeArgumentClauseAST | undefined {
    return AST.from<AttributeArgumentClauseAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class AttributeUsingPrefixAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttributeUsingPrefix(this, context);
  }
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getAttributeNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class DesignatorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDesignator(this, context);
  }
  getDotToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }
}

export class NewPlacementAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewPlacement(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getExpressionList(): Generator<ExpressionAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<ExpressionAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class GlobalNestedNameSpecifierAST extends NestedNameSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGlobalNestedNameSpecifier(this, context);
  }
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class SimpleNestedNameSpecifierAST extends NestedNameSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleNestedNameSpecifier(this, context);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class DecltypeNestedNameSpecifierAST extends NestedNameSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeNestedNameSpecifier(this, context);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getDecltypeSpecifier(): DecltypeSpecifierAST | undefined {
    return AST.from<DecltypeSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class TemplateNestedNameSpecifierAST extends NestedNameSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplateNestedNameSpecifier(this, context);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getTemplateId(): SimpleTemplateIdAST | undefined {
    return AST.from<SimpleTemplateIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 4) !== 0;
  }
}

export class ThrowExceptionSpecifierAST extends ExceptionSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThrowExceptionSpecifier(this, context);
  }
  getThrowToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class NoexceptSpecifierAST extends ExceptionSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNoexceptSpecifier(this, context);
  }
  getNoexceptToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class PackExpansionExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPackExpansionExpression(this, context);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

export class DesignatedInitializerClauseAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDesignatedInitializerClause(this, context);
  }
  getDesignator(): DesignatorAST | undefined {
    return AST.from<DesignatorAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class ThisExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThisExpression(this, context);
  }
  getThisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class CharLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCharLiteralExpression(this, context);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

export class BoolLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBoolLiteralExpression(this, context);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIsTrue(): boolean {
    return cxx.getASTSlot(this.getHandle(), 1) !== 0;
  }
}

export class IntLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIntLiteralExpression(this, context);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

export class FloatLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFloatLiteralExpression(this, context);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

export class NullptrLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNullptrLiteralExpression(this, context);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLiteral(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

export class StringLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStringLiteralExpression(this, context);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

export class UserDefinedStringLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUserDefinedStringLiteralExpression(this, context);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

export class IdExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIdExpression(this, context);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 3) !== 0;
  }
}

export class RequiresExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRequiresExpression(this, context);
  }
  getRequiresToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return AST.from<ParameterDeclarationClauseAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getRequirementBody(): RequirementBodyAST | undefined {
    return AST.from<RequirementBodyAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

export class NestedExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedExpression(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class RightFoldExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRightFoldExpression(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 5);
  }
}

export class LeftFoldExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLeftFoldExpression(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 5);
  }
}

export class FoldExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFoldExpression(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLeftExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getFoldOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getRightExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 7);
  }
  getFoldOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 8);
  }
}

export class LambdaExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLambdaExpression(this, context);
  }
  getLambdaIntroducer(): LambdaIntroducerAST | undefined {
    return AST.from<LambdaIntroducerAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getTemplateParameterList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getLambdaDeclarator(): LambdaDeclaratorAST | undefined {
    return AST.from<LambdaDeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 6),
      this.parser,
    );
  }
}

export class SizeofExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeofExpression(this, context);
  }
  getSizeofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class SizeofTypeExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeofTypeExpression(this, context);
  }
  getSizeofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class SizeofPackExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeofPackExpression(this, context);
  }
  getSizeofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getIdentifierValue(slot);
  }
}

export class TypeidExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeidExpression(this, context);
  }
  getTypeidToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class TypeidOfTypeExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeidOfTypeExpression(this, context);
  }
  getTypeidToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class AlignofTypeExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignofTypeExpression(this, context);
  }
  getAlignofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class AlignofExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignofExpression(this, context);
  }
  getAlignofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class TypeTraitsExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeTraitsExpression(this, context);
  }
  getTypeTraitsToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getTypeIdList(): Generator<TypeIdAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<TypeIdAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getTypeTraits(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 4);
  }
}

export class YieldExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitYieldExpression(this, context);
  }
  getYieldToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class AwaitExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAwaitExpression(this, context);
  }
  getAwaitToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class UnaryExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUnaryExpression(this, context);
  }
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 2);
  }
}

export class BinaryExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBinaryExpression(this, context);
  }
  getLeftExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getRightExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 3);
  }
}

export class AssignmentExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAssignmentExpression(this, context);
  }
  getLeftExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getRightExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 3);
  }
}

export class ConditionExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConditionExpression(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getDeclSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
}

export class BracedTypeConstructionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBracedTypeConstruction(this, context);
  }
  getTypeSpecifier(): SpecifierAST | undefined {
    return AST.from<SpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getBracedInitList(): BracedInitListAST | undefined {
    return AST.from<BracedInitListAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class TypeConstructionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeConstruction(this, context);
  }
  getTypeSpecifier(): SpecifierAST | undefined {
    return AST.from<SpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getExpressionList(): Generator<ExpressionAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<ExpressionAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class CallExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCallExpression(this, context);
  }
  getBaseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getExpressionList(): Generator<ExpressionAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<ExpressionAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class SubscriptExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSubscriptExpression(this, context);
  }
  getBaseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIndexExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class MemberExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitMemberExpression(this, context);
  }
  getBaseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getAccessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getMemberId(): IdExpressionAST | undefined {
    return AST.from<IdExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getAccessOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 4);
  }
}

export class PostIncrExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPostIncrExpression(this, context);
  }
  getBaseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 2);
  }
}

export class ConditionalExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConditionalExpression(this, context);
  }
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getQuestionToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIftrueExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getIffalseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

export class ImplicitCastExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitImplicitCastExpression(this, context);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

export class CastExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCastExpression(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
}

export class CppCastExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCppCastExpression(this, context);
  }
  getCastToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
}

export class NewExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewExpression(this, context);
  }
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getNewToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getNewPlacement(): NewPlacementAST | undefined {
    return AST.from<NewPlacementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getTypeId(): NewTypeIdAST | undefined {
    return AST.from<NewTypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getNewInitalizer(): NewInitializerAST | undefined {
    return AST.from<NewInitializerAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

export class DeleteExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeleteExpression(this, context);
  }
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getDeleteToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

export class ThrowExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThrowExpression(this, context);
  }
  getThrowToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class NoexceptExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNoexceptExpression(this, context);
  }
  getNoexceptToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class EqualInitializerAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEqualInitializer(this, context);
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class BracedInitListAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBracedInitList(this, context);
  }
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getExpressionList(): Generator<ExpressionAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<ExpressionAST>(cxx.getListValue(it), this.parser);
    }
  }
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class ParenInitializerAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParenInitializer(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getExpressionList(): Generator<ExpressionAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<ExpressionAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class SimpleRequirementAST extends RequirementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleRequirement(this, context);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

export class CompoundRequirementAST extends RequirementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundRequirement(this, context);
  }
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getNoexceptToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getMinusGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getTypeConstraint(): TypeConstraintAST | undefined {
    return AST.from<TypeConstraintAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
}

export class TypeRequirementAST extends RequirementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeRequirement(this, context);
  }
  getTypenameToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class NestedRequirementAST extends RequirementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedRequirement(this, context);
  }
  getRequiresToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class TypeTemplateArgumentAST extends TemplateArgumentAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeTemplateArgument(this, context);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

export class ExpressionTemplateArgumentAST extends TemplateArgumentAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExpressionTemplateArgument(this, context);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

export class ParenMemInitializerAST extends MemInitializerAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParenMemInitializer(this, context);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  *getExpressionList(): Generator<ExpressionAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<ExpressionAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

export class BracedMemInitializerAST extends MemInitializerAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBracedMemInitializer(this, context);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getBracedInitList(): BracedInitListAST | undefined {
    return AST.from<BracedInitListAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class ThisLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThisLambdaCapture(this, context);
  }
  getThisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class DerefThisLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDerefThisLambdaCapture(this, context);
  }
  getStarToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getThisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

export class SimpleLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleLambdaCapture(this, context);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }
}

export class RefLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRefLambdaCapture(this, context);
  }
  getAmpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

export class RefInitLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRefInitLambdaCapture(this, context);
  }
  getAmpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getIdentifierValue(slot);
  }
}

export class InitLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitInitLambdaCapture(this, context);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

export class NewParenInitializerAST extends NewInitializerAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewParenInitializer(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getExpressionList(): Generator<ExpressionAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<ExpressionAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class NewBracedInitializerAST extends NewInitializerAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewBracedInitializer(this, context);
  }
  getBracedInitList(): BracedInitListAST | undefined {
    return AST.from<BracedInitListAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

export class EllipsisExceptionDeclarationAST extends ExceptionDeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEllipsisExceptionDeclaration(this, context);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class TypeExceptionDeclarationAST extends ExceptionDeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeExceptionDeclaration(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

export class DefaultFunctionBodyAST extends FunctionBodyAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDefaultFunctionBody(this, context);
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getDefaultToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class CompoundStatementFunctionBodyAST extends FunctionBodyAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundStatementFunctionBody(this, context);
  }
  getCtorInitializer(): CtorInitializerAST | undefined {
    return AST.from<CtorInitializerAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class TryStatementFunctionBodyAST extends FunctionBodyAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTryStatementFunctionBody(this, context);
  }
  getTryToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getCtorInitializer(): CtorInitializerAST | undefined {
    return AST.from<CtorInitializerAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  *getHandlerList(): Generator<HandlerAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<HandlerAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class DeleteFunctionBodyAST extends FunctionBodyAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeleteFunctionBody(this, context);
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getDeleteToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class TranslationUnitAST extends UnitAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTranslationUnit(this, context);
  }
  *getDeclarationList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class ModuleUnitAST extends UnitAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleUnit(this, context);
  }
  getGlobalModuleFragment(): GlobalModuleFragmentAST | undefined {
    return AST.from<GlobalModuleFragmentAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getModuleDeclaration(): ModuleDeclarationAST | undefined {
    return AST.from<ModuleDeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  *getDeclarationList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
  getPrivateModuleFragment(): PrivateModuleFragmentAST | undefined {
    return AST.from<PrivateModuleFragmentAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
}

export class LabeledStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLabeledStatement(this, context);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

export class CaseStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCaseStatement(this, context);
  }
  getCaseToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
}

export class DefaultStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDefaultStatement(this, context);
  }
  getDefaultToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

export class ExpressionStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExpressionStatement(this, context);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

export class CompoundStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundStatement(this, context);
  }
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getStatementList(): Generator<StatementAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<StatementAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class IfStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIfStatement(this, context);
  }
  getIfToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getConstexprToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getInitializer(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 6),
      this.parser,
    );
  }
  getElseToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }
  getElseStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 8),
      this.parser,
    );
  }
}

export class SwitchStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSwitchStatement(this, context);
  }
  getSwitchToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getInitializer(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
}

export class WhileStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitWhileStatement(this, context);
  }
  getWhileToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

export class DoStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDoStatement(this, context);
  }
  getDoToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getWhileToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
}

export class ForRangeStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitForRangeStatement(this, context);
  }
  getForToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getInitializer(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRangeDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getRangeInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 7),
      this.parser,
    );
  }
}

export class ForStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitForStatement(this, context);
  }
  getForToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getInitializer(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 7),
      this.parser,
    );
  }
}

export class BreakStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBreakStatement(this, context);
  }
  getBreakToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

export class ContinueStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitContinueStatement(this, context);
  }
  getContinueToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

export class ReturnStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitReturnStatement(this, context);
  }
  getReturnToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class GotoStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGotoStatement(this, context);
  }
  getGotoToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

export class CoroutineReturnStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCoroutineReturnStatement(this, context);
  }
  getCoreturnToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class DeclarationStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeclarationStatement(this, context);
  }
  getDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

export class TryBlockStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTryBlockStatement(this, context);
  }
  getTryToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  *getHandlerList(): Generator<HandlerAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<HandlerAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class AccessDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAccessDeclaration(this, context);
  }
  getAccessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getAccessSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 2);
  }
}

export class FunctionDefinitionAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFunctionDefinition(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getDeclSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getFunctionBody(): FunctionBodyAST | undefined {
    return AST.from<FunctionBodyAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

export class ConceptDefinitionAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConceptDefinition(this, context);
  }
  getConceptToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getIdentifierValue(slot);
  }
}

export class ForRangeDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitForRangeDeclaration(this, context);
  }
}

export class AliasDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAliasDeclaration(this, context);
  }
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 6);
    return cxx.getIdentifierValue(slot);
  }
}

export class SimpleDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleDeclaration(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getDeclSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getInitDeclaratorList(): Generator<InitDeclaratorAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<InitDeclaratorAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
}

export class StructuredBindingDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStructuredBindingDeclaration(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getDeclSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRefQualifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  *getBindingList(): Generator<UnqualifiedIdAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 4);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<UnqualifiedIdAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 6),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }
}

export class StaticAssertDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStaticAssertDeclaration(this, context);
  }
  getStaticAssertToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getLiteralValue(slot);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }
}

export class EmptyDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEmptyDeclaration(this, context);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class AttributeDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttributeDeclaration(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

export class OpaqueEnumDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitOpaqueEnumDeclaration(this, context);
  }
  getEnumToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getClassToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getEnumBase(): EnumBaseAST | undefined {
    return AST.from<EnumBaseAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  getEmicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
}

export class NestedNamespaceSpecifierAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedNamespaceSpecifier(this, context);
  }
  getInlineToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
  getIsInline(): boolean {
    return cxx.getASTSlot(this.getHandle(), 4) !== 0;
  }
}

export class NamespaceDefinitionAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamespaceDefinition(this, context);
  }
  getInlineToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getNestedNamespaceSpecifierList(): Generator<
    NestedNamespaceSpecifierAST | undefined
  > {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<NestedNamespaceSpecifierAST>(
        cxx.getListValue(it),
        this.parser,
      );
    }
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  *getExtraAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 5);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
  *getDeclarationList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 7);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 8), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 9);
    return cxx.getIdentifierValue(slot);
  }
  getIsInline(): boolean {
    return cxx.getASTSlot(this.getHandle(), 10) !== 0;
  }
}

export class NamespaceAliasDefinitionAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamespaceAliasDefinition(this, context);
  }
  getNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 6);
    return cxx.getIdentifierValue(slot);
  }
}

export class UsingDirectiveAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingDirective(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

export class UsingDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingDeclaration(this, context);
  }
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getUsingDeclaratorList(): Generator<UsingDeclaratorAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<UsingDeclaratorAST>(cxx.getListValue(it), this.parser);
    }
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class UsingEnumDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingEnumDeclaration(this, context);
  }
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getEnumTypeSpecifier(): ElaboratedTypeSpecifierAST | undefined {
    return AST.from<ElaboratedTypeSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class AsmDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmDeclaration(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getAsmToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 6);
    return cxx.getLiteralValue(slot);
  }
}

export class ExportDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExportDeclaration(this, context);
  }
  getExportToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class ExportCompoundDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExportCompoundDeclaration(this, context);
  }
  getExportToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getDeclarationList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class ModuleImportDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleImportDeclaration(this, context);
  }
  getImportToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getImportName(): ImportNameAST | undefined {
    return AST.from<ImportNameAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class TemplateDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplateDeclaration(this, context);
  }
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getTemplateParameterList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
}

export class TypenameTypeParameterAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypenameTypeParameter(this, context);
  }
  getClassKeyToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getIdentifierValue(slot);
  }
}

export class TemplateTypeParameterAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplateTypeParameter(this, context);
  }
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getTemplateParameterList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getClassKeyToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }
  getIdExpression(): IdExpressionAST | undefined {
    return AST.from<IdExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 8),
      this.parser,
    );
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 9);
    return cxx.getIdentifierValue(slot);
  }
}

export class TemplatePackTypeParameterAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplatePackTypeParameter(this, context);
  }
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getTemplateParameterList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getClassKeyToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 7);
    return cxx.getIdentifierValue(slot);
  }
}

export class DeductionGuideAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeductionGuide(this, context);
  }
  getExplicitSpecifier(): SpecifierAST | undefined {
    return AST.from<SpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return AST.from<ParameterDeclarationClauseAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getArrowToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
  getTemplateId(): SimpleTemplateIdAST | undefined {
    return AST.from<SimpleTemplateIdAST>(
      cxx.getASTSlot(this.getHandle(), 6),
      this.parser,
    );
  }
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 8);
    return cxx.getIdentifierValue(slot);
  }
}

export class ExplicitInstantiationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExplicitInstantiation(this, context);
  }
  getExternToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

export class ParameterDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParameterDeclaration(this, context);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 0);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

export class LinkageSpecificationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLinkageSpecification(this, context);
  }
  getExternToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getStringliteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  *getDeclarationList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getStringLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getLiteralValue(slot);
  }
}

export class NameIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNameId(this, context);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getIdentifierValue(slot);
  }
}

export class DestructorIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDestructorId(this, context);
  }
  getTildeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class DecltypeIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeId(this, context);
  }
  getDecltypeSpecifier(): DecltypeSpecifierAST | undefined {
    return AST.from<DecltypeSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

export class OperatorFunctionIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitOperatorFunctionId(this, context);
  }
  getOperatorToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getOpenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getCloseToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 4);
  }
}

export class LiteralOperatorIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLiteralOperatorId(this, context);
  }
  getOperatorToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getLiteralValue(slot);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getIdentifierValue(slot);
  }
}

export class ConversionFunctionIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConversionFunctionId(this, context);
  }
  getOperatorToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class SimpleTemplateIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleTemplateId(this, context);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getTemplateArgumentList(): Generator<TemplateArgumentAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<TemplateArgumentAST>(cxx.getListValue(it), this.parser);
    }
  }
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getIdentifierValue(slot);
  }
}

export class LiteralOperatorTemplateIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLiteralOperatorTemplateId(this, context);
  }
  getLiteralOperatorId(): LiteralOperatorIdAST | undefined {
    return AST.from<LiteralOperatorIdAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getTemplateArgumentList(): Generator<TemplateArgumentAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<TemplateArgumentAST>(cxx.getListValue(it), this.parser);
    }
  }
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class OperatorFunctionTemplateIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitOperatorFunctionTemplateId(this, context);
  }
  getOperatorFunctionId(): OperatorFunctionIdAST | undefined {
    return AST.from<OperatorFunctionIdAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getTemplateArgumentList(): Generator<TemplateArgumentAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<TemplateArgumentAST>(cxx.getListValue(it), this.parser);
    }
  }
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class TypedefSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypedefSpecifier(this, context);
  }
  getTypedefToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class FriendSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFriendSpecifier(this, context);
  }
  getFriendToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class ConstevalSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstevalSpecifier(this, context);
  }
  getConstevalToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class ConstinitSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstinitSpecifier(this, context);
  }
  getConstinitToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class ConstexprSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstexprSpecifier(this, context);
  }
  getConstexprToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class InlineSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitInlineSpecifier(this, context);
  }
  getInlineToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class StaticSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStaticSpecifier(this, context);
  }
  getStaticToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class ExternSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExternSpecifier(this, context);
  }
  getExternToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class ThreadLocalSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThreadLocalSpecifier(this, context);
  }
  getThreadLocalToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class ThreadSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThreadSpecifier(this, context);
  }
  getThreadToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class MutableSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitMutableSpecifier(this, context);
  }
  getMutableToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class VirtualSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVirtualSpecifier(this, context);
  }
  getVirtualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class ExplicitSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExplicitSpecifier(this, context);
  }
  getExplicitToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class AutoTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAutoTypeSpecifier(this, context);
  }
  getAutoToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class VoidTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVoidTypeSpecifier(this, context);
  }
  getVoidToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class VaListTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVaListTypeSpecifier(this, context);
  }
  getSpecifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

export class IntegralTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIntegralTypeSpecifier(this, context);
  }
  getSpecifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

export class FloatingPointTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFloatingPointTypeSpecifier(this, context);
  }
  getSpecifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

export class ComplexTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitComplexTypeSpecifier(this, context);
  }
  getComplexToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class NamedTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamedTypeSpecifier(this, context);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 3) !== 0;
  }
}

export class AtomicTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAtomicTypeSpecifier(this, context);
  }
  getAtomicToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class UnderlyingTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUnderlyingTypeSpecifier(this, context);
  }
  getUnderlyingTypeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class ElaboratedTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitElaboratedTypeSpecifier(this, context);
  }
  getClassToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getClassKey(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 4);
  }
}

export class DecltypeAutoSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeAutoSpecifier(this, context);
  }
  getDecltypeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getAutoToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class DecltypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeSpecifier(this, context);
  }
  getDecltypeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

export class PlaceholderTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPlaceholderTypeSpecifier(this, context);
  }
  getTypeConstraint(): TypeConstraintAST | undefined {
    return AST.from<TypeConstraintAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getSpecifier(): SpecifierAST | undefined {
    return AST.from<SpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class ConstQualifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstQualifier(this, context);
  }
  getConstToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class VolatileQualifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVolatileQualifier(this, context);
  }
  getVolatileToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class RestrictQualifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRestrictQualifier(this, context);
  }
  getRestrictToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

export class EnumSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEnumSpecifier(this, context);
  }
  getEnumToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getClassToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
  getEnumBase(): EnumBaseAST | undefined {
    return AST.from<EnumBaseAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }
  *getEnumeratorList(): Generator<EnumeratorAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 8);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<EnumeratorAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 9), this.parser);
  }
}

export class ClassSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitClassSpecifier(this, context);
  }
  getClassToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
  getFinalToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getBaseClause(): BaseClauseAST | undefined {
    return AST.from<BaseClauseAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
  *getDeclarationList(): Generator<DeclarationAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 7);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<DeclarationAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 8), this.parser);
  }
  getClassKey(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 9);
  }
  getIsFinal(): boolean {
    return cxx.getASTSlot(this.getHandle(), 10) !== 0;
  }
}

export class TypenameSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypenameSpecifier(this, context);
  }
  getTypenameToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

export class BitfieldDeclaratorAST extends CoreDeclaratorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBitfieldDeclarator(this, context);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getSizeExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

export class ParameterPackAST extends CoreDeclaratorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParameterPack(this, context);
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getCoreDeclarator(): CoreDeclaratorAST | undefined {
    return AST.from<CoreDeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

export class IdDeclaratorAST extends CoreDeclaratorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIdDeclarator(this, context);
  }
  getDeclaratorId(): IdExpressionAST | undefined {
    return AST.from<IdExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class NestedDeclaratorAST extends CoreDeclaratorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedDeclarator(this, context);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class PointerOperatorAST extends PtrOperatorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPointerOperator(this, context);
  }
  getStarToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getCvQualifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class ReferenceOperatorAST extends PtrOperatorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitReferenceOperator(this, context);
  }
  getRefToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 1);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRefOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 2);
  }
}

export class PtrToMemberOperatorAST extends PtrOperatorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPtrToMemberOperator(this, context);
  }
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getStarToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 2);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
  *getCvQualifierList(): Generator<SpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<SpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class FunctionDeclaratorAST extends DeclaratorModifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFunctionDeclarator(this, context);
  }
  getParametersAndQualifiers(): ParametersAndQualifiersAST | undefined {
    return AST.from<ParametersAndQualifiersAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
  getTrailingReturnType(): TrailingReturnTypeAST | undefined {
    return AST.from<TrailingReturnTypeAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getIsFinal(): boolean {
    return cxx.getASTSlot(this.getHandle(), 2) !== 0;
  }
  getIsOverride(): boolean {
    return cxx.getASTSlot(this.getHandle(), 3) !== 0;
  }
  getIsPure(): boolean {
    return cxx.getASTSlot(this.getHandle(), 4) !== 0;
  }
}

export class ArrayDeclaratorAST extends DeclaratorModifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitArrayDeclarator(this, context);
  }
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  *getAttributeList(): Generator<AttributeSpecifierAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeSpecifierAST>(cxx.getListValue(it), this.parser);
    }
  }
}

export class CxxAttributeAST extends AttributeSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCxxAttribute(this, context);
  }
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLbracket2Token(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getAttributeUsingPrefix(): AttributeUsingPrefixAST | undefined {
    return AST.from<AttributeUsingPrefixAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  *getAttributeList(): Generator<AttributeAST | undefined> {
    for (
      let it = cxx.getASTSlot(this.getHandle(), 3);
      it;
      it = cxx.getListNext(it)
    ) {
      yield AST.from<AttributeAST>(cxx.getListValue(it), this.parser);
    }
  }
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
  getRbracket2Token(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

export class GccAttributeAST extends AttributeSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGccAttribute(this, context);
  }
  getAttributeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getLparen2Token(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getRparen2Token(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
}

export class AlignasAttributeAST extends AttributeSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignasAttribute(this, context);
  }
  getAlignasToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
}

export class AsmAttributeAST extends AttributeSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmAttribute(this, context);
  }
  getAsmToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getLiteralValue(slot);
  }
}

export class ScopedAttributeTokenAST extends AttributeTokenAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitScopedAttributeToken(this, context);
  }
  getAttributeNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

export class SimpleAttributeTokenAST extends AttributeTokenAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleAttributeToken(this, context);
  }
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

const AST_CONSTRUCTORS: Array<
  new (handle: number, kind: ASTKind, parser: TranslationUnitLike) => AST
> = [
  TypeIdAST,
  UsingDeclaratorAST,
  HandlerAST,
  EnumBaseAST,
  EnumeratorAST,
  DeclaratorAST,
  InitDeclaratorAST,
  BaseSpecifierAST,
  BaseClauseAST,
  NewDeclaratorAST,
  NewTypeIdAST,
  RequiresClauseAST,
  ParameterDeclarationClauseAST,
  ParametersAndQualifiersAST,
  LambdaIntroducerAST,
  LambdaDeclaratorAST,
  TrailingReturnTypeAST,
  CtorInitializerAST,
  RequirementBodyAST,
  TypeConstraintAST,
  GlobalModuleFragmentAST,
  PrivateModuleFragmentAST,
  ModuleQualifierAST,
  ModuleNameAST,
  ModuleDeclarationAST,
  ImportNameAST,
  ModulePartitionAST,
  AttributeArgumentClauseAST,
  AttributeAST,
  AttributeUsingPrefixAST,
  DesignatorAST,
  NewPlacementAST,
  GlobalNestedNameSpecifierAST,
  SimpleNestedNameSpecifierAST,
  DecltypeNestedNameSpecifierAST,
  TemplateNestedNameSpecifierAST,
  ThrowExceptionSpecifierAST,
  NoexceptSpecifierAST,
  PackExpansionExpressionAST,
  DesignatedInitializerClauseAST,
  ThisExpressionAST,
  CharLiteralExpressionAST,
  BoolLiteralExpressionAST,
  IntLiteralExpressionAST,
  FloatLiteralExpressionAST,
  NullptrLiteralExpressionAST,
  StringLiteralExpressionAST,
  UserDefinedStringLiteralExpressionAST,
  IdExpressionAST,
  RequiresExpressionAST,
  NestedExpressionAST,
  RightFoldExpressionAST,
  LeftFoldExpressionAST,
  FoldExpressionAST,
  LambdaExpressionAST,
  SizeofExpressionAST,
  SizeofTypeExpressionAST,
  SizeofPackExpressionAST,
  TypeidExpressionAST,
  TypeidOfTypeExpressionAST,
  AlignofTypeExpressionAST,
  AlignofExpressionAST,
  TypeTraitsExpressionAST,
  YieldExpressionAST,
  AwaitExpressionAST,
  UnaryExpressionAST,
  BinaryExpressionAST,
  AssignmentExpressionAST,
  ConditionExpressionAST,
  BracedTypeConstructionAST,
  TypeConstructionAST,
  CallExpressionAST,
  SubscriptExpressionAST,
  MemberExpressionAST,
  PostIncrExpressionAST,
  ConditionalExpressionAST,
  ImplicitCastExpressionAST,
  CastExpressionAST,
  CppCastExpressionAST,
  NewExpressionAST,
  DeleteExpressionAST,
  ThrowExpressionAST,
  NoexceptExpressionAST,
  EqualInitializerAST,
  BracedInitListAST,
  ParenInitializerAST,
  SimpleRequirementAST,
  CompoundRequirementAST,
  TypeRequirementAST,
  NestedRequirementAST,
  TypeTemplateArgumentAST,
  ExpressionTemplateArgumentAST,
  ParenMemInitializerAST,
  BracedMemInitializerAST,
  ThisLambdaCaptureAST,
  DerefThisLambdaCaptureAST,
  SimpleLambdaCaptureAST,
  RefLambdaCaptureAST,
  RefInitLambdaCaptureAST,
  InitLambdaCaptureAST,
  NewParenInitializerAST,
  NewBracedInitializerAST,
  EllipsisExceptionDeclarationAST,
  TypeExceptionDeclarationAST,
  DefaultFunctionBodyAST,
  CompoundStatementFunctionBodyAST,
  TryStatementFunctionBodyAST,
  DeleteFunctionBodyAST,
  TranslationUnitAST,
  ModuleUnitAST,
  LabeledStatementAST,
  CaseStatementAST,
  DefaultStatementAST,
  ExpressionStatementAST,
  CompoundStatementAST,
  IfStatementAST,
  SwitchStatementAST,
  WhileStatementAST,
  DoStatementAST,
  ForRangeStatementAST,
  ForStatementAST,
  BreakStatementAST,
  ContinueStatementAST,
  ReturnStatementAST,
  GotoStatementAST,
  CoroutineReturnStatementAST,
  DeclarationStatementAST,
  TryBlockStatementAST,
  AccessDeclarationAST,
  FunctionDefinitionAST,
  ConceptDefinitionAST,
  ForRangeDeclarationAST,
  AliasDeclarationAST,
  SimpleDeclarationAST,
  StructuredBindingDeclarationAST,
  StaticAssertDeclarationAST,
  EmptyDeclarationAST,
  AttributeDeclarationAST,
  OpaqueEnumDeclarationAST,
  NestedNamespaceSpecifierAST,
  NamespaceDefinitionAST,
  NamespaceAliasDefinitionAST,
  UsingDirectiveAST,
  UsingDeclarationAST,
  UsingEnumDeclarationAST,
  AsmDeclarationAST,
  ExportDeclarationAST,
  ExportCompoundDeclarationAST,
  ModuleImportDeclarationAST,
  TemplateDeclarationAST,
  TypenameTypeParameterAST,
  TemplateTypeParameterAST,
  TemplatePackTypeParameterAST,
  DeductionGuideAST,
  ExplicitInstantiationAST,
  ParameterDeclarationAST,
  LinkageSpecificationAST,
  NameIdAST,
  DestructorIdAST,
  DecltypeIdAST,
  OperatorFunctionIdAST,
  LiteralOperatorIdAST,
  ConversionFunctionIdAST,
  SimpleTemplateIdAST,
  LiteralOperatorTemplateIdAST,
  OperatorFunctionTemplateIdAST,
  TypedefSpecifierAST,
  FriendSpecifierAST,
  ConstevalSpecifierAST,
  ConstinitSpecifierAST,
  ConstexprSpecifierAST,
  InlineSpecifierAST,
  StaticSpecifierAST,
  ExternSpecifierAST,
  ThreadLocalSpecifierAST,
  ThreadSpecifierAST,
  MutableSpecifierAST,
  VirtualSpecifierAST,
  ExplicitSpecifierAST,
  AutoTypeSpecifierAST,
  VoidTypeSpecifierAST,
  VaListTypeSpecifierAST,
  IntegralTypeSpecifierAST,
  FloatingPointTypeSpecifierAST,
  ComplexTypeSpecifierAST,
  NamedTypeSpecifierAST,
  AtomicTypeSpecifierAST,
  UnderlyingTypeSpecifierAST,
  ElaboratedTypeSpecifierAST,
  DecltypeAutoSpecifierAST,
  DecltypeSpecifierAST,
  PlaceholderTypeSpecifierAST,
  ConstQualifierAST,
  VolatileQualifierAST,
  RestrictQualifierAST,
  EnumSpecifierAST,
  ClassSpecifierAST,
  TypenameSpecifierAST,
  BitfieldDeclaratorAST,
  ParameterPackAST,
  IdDeclaratorAST,
  NestedDeclaratorAST,
  PointerOperatorAST,
  ReferenceOperatorAST,
  PtrToMemberOperatorAST,
  FunctionDeclaratorAST,
  ArrayDeclaratorAST,
  CxxAttributeAST,
  GccAttributeAST,
  AlignasAttributeAST,
  AsmAttributeAST,
  ScopedAttributeTokenAST,
  SimpleAttributeTokenAST,
];
