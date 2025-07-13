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

import { cxx } from "./cxx";
import { SourceLocation } from "./SourceLocation";
import { ASTCursor } from "./ASTCursor";
import { ASTVisitor } from "./ASTVisitor";
import { ASTKind } from "./ASTKind";
import { Token } from "./Token";
import { TokenKind } from "./TokenKind";

/**
 * An interface that represents a translation unit.
 */
interface TranslationUnitLike {
  /**
   * Returns the handle of the translation unit.
   */
  getUnitHandle(): number;
}

/**
 * The base class of all the AST nodes.
 */
export abstract class AST {
  /**
   * Constructs an AST node.
   * @param handle the handle of the AST node.
   * @param kind the kind of the AST node.
   * @param parser the parser that owns the AST node.
   */
  constructor(
    private readonly handle: number,
    private readonly kind: ASTKind,
    protected readonly parser: TranslationUnitLike,
  ) {}

  /**
   * Returns the cursor of the AST node.
   *
   * The cursor is used to traverse the AST.
   *
   * @returns the cursor of the AST node.
   */
  walk(): ASTCursor {
    return new ASTCursor(this, this.parser);
  }

  /**
   * Returns the kind of the AST node.
   *
   * @returns the kind of the AST node.
   */
  getKind(): ASTKind {
    return this.kind;
  }

  /**
   * Returns true if the AST node is of the given kind.
   *
   * @param kind the kind to check.
   * @returns true if the AST node is of the given kind.
   */
  is(kind: ASTKind): boolean {
    return this.kind === kind;
  }

  /**
   * Returns true if the AST node is not of the given kind.
   *
   * @param kind the kind to check.
   * @returns true if the AST node is not of the given kind.
   */
  isNot(kind: ASTKind): boolean {
    return this.kind !== kind;
  }

  /**
   * Returns the handle of the AST node.
   *
   * @returns the handle of the AST node.
   */
  getHandle() {
    return this.handle;
  }

  /**
   * Returns the source location of the AST node.
   *
   * @returns the source location of the AST node.
   */
  getStartLocation(): SourceLocation | undefined {
    return cxx.getStartLocation(this.handle, this.parser.getUnitHandle());
  }

  /**
   * Returns the source location of the AST node.
   *
   * @returns the source location of the AST node.
   */
  getEndLocation(): SourceLocation | undefined {
    return cxx.getEndLocation(this.handle, this.parser.getUnitHandle());
  }

  /**
   * Accepts the given visitor.
   */
  abstract accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result;

  /**
   * Constructs an AST node from the given handle.
   *
   * @param handle the handle of the AST node.
   * @param parser the parser that owns the AST node.
   * @returns the AST node.
   */
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
export abstract class DeclaratorChunkAST extends AST {}
export abstract class DesignatorAST extends AST {}
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
export abstract class TemplateParameterAST extends AST {}
export abstract class UnitAST extends AST {}
export abstract class UnqualifiedIdAST extends AST {}

/**
 * TranslationUnitAST node.
 */
export class TranslationUnitAST extends UnitAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTranslationUnit(this, context);
  }

  /**
   * Returns the declarationList of this node
   */
  getDeclarationList(): Iterable<DeclarationAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DeclarationAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DeclarationAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }
}

/**
 * ModuleUnitAST node.
 */
export class ModuleUnitAST extends UnitAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleUnit(this, context);
  }

  /**
   * Returns the globalModuleFragment of this node
   */
  getGlobalModuleFragment(): GlobalModuleFragmentAST | undefined {
    return AST.from<GlobalModuleFragmentAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the moduleDeclaration of this node
   */
  getModuleDeclaration(): ModuleDeclarationAST | undefined {
    return AST.from<ModuleDeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the declarationList of this node
   */
  getDeclarationList(): Iterable<DeclarationAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DeclarationAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DeclarationAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the privateModuleFragment of this node
   */
  getPrivateModuleFragment(): PrivateModuleFragmentAST | undefined {
    return AST.from<PrivateModuleFragmentAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
}

/**
 * SimpleDeclarationAST node.
 */
export class SimpleDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleDeclaration(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declSpecifierList of this node
   */
  getDeclSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the initDeclaratorList of this node
   */
  getInitDeclaratorList(): Iterable<InitDeclaratorAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: InitDeclaratorAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<InitDeclaratorAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the requiresClause of this node
   */
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
}

/**
 * AsmDeclarationAST node.
 */
export class AsmDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmDeclaration(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the asmQualifierList of this node
   */
  getAsmQualifierList(): Iterable<AsmQualifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AsmQualifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AsmQualifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the asm token in this node
   */
  getAsmToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the outputOperandList of this node
   */
  getOutputOperandList(): Iterable<AsmOperandAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AsmOperandAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AsmOperandAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the inputOperandList of this node
   */
  getInputOperandList(): Iterable<AsmOperandAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AsmOperandAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AsmOperandAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the clobberList of this node
   */
  getClobberList(): Iterable<AsmClobberAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AsmClobberAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AsmClobberAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the gotoLabelList of this node
   */
  getGotoLabelList(): Iterable<AsmGotoLabelAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AsmGotoLabelAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AsmGotoLabelAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 9), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 10), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 11);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * NamespaceAliasDefinitionAST node.
 */
export class NamespaceAliasDefinitionAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamespaceAliasDefinition(this, context);
  }

  /**
   * Returns the location of the namespace token in this node
   */
  getNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): NameIdAST | undefined {
    return AST.from<NameIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 6);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * UsingDeclarationAST node.
 */
export class UsingDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingDeclaration(this, context);
  }

  /**
   * Returns the location of the using token in this node
   */
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the usingDeclaratorList of this node
   */
  getUsingDeclaratorList(): Iterable<UsingDeclaratorAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: UsingDeclaratorAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<UsingDeclaratorAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * UsingEnumDeclarationAST node.
 */
export class UsingEnumDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingEnumDeclaration(this, context);
  }

  /**
   * Returns the location of the using token in this node
   */
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the enumTypeSpecifier of this node
   */
  getEnumTypeSpecifier(): ElaboratedTypeSpecifierAST | undefined {
    return AST.from<ElaboratedTypeSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * UsingDirectiveAST node.
 */
export class UsingDirectiveAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingDirective(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the using token in this node
   */
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the namespace token in this node
   */
  getNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): NameIdAST | undefined {
    return AST.from<NameIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

/**
 * StaticAssertDeclarationAST node.
 */
export class StaticAssertDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStaticAssertDeclaration(this, context);
  }

  /**
   * Returns the location of the staticAssert token in this node
   */
  getStaticAssertToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the comma token in this node
   */
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getLiteralValue(slot);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }
}

/**
 * AliasDeclarationAST node.
 */
export class AliasDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAliasDeclaration(this, context);
  }

  /**
   * Returns the location of the using token in this node
   */
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the gnuAttributeList of this node
   */
  getGnuAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 7);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * OpaqueEnumDeclarationAST node.
 */
export class OpaqueEnumDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitOpaqueEnumDeclaration(this, context);
  }

  /**
   * Returns the location of the enum token in this node
   */
  getEnumToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the class token in this node
   */
  getClassToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): NameIdAST | undefined {
    return AST.from<NameIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }

  /**
   * Returns the typeSpecifierList of this node
   */
  getTypeSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the emicolon token in this node
   */
  getEmicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }
}

/**
 * FunctionDefinitionAST node.
 */
export class FunctionDefinitionAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFunctionDefinition(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declSpecifierList of this node
   */
  getDeclSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declarator of this node
   */
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the requiresClause of this node
   */
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the functionBody of this node
   */
  getFunctionBody(): FunctionBodyAST | undefined {
    return AST.from<FunctionBodyAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

/**
 * TemplateDeclarationAST node.
 */
export class TemplateDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplateDeclaration(this, context);
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the less token in this node
   */
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the templateParameterList of this node
   */
  getTemplateParameterList(): Iterable<TemplateParameterAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: TemplateParameterAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<TemplateParameterAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the greater token in this node
   */
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the requiresClause of this node
   */
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the declaration of this node
   */
  getDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
}

/**
 * ConceptDefinitionAST node.
 */
export class ConceptDefinitionAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConceptDefinition(this, context);
  }

  /**
   * Returns the location of the concept token in this node
   */
  getConceptToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * DeductionGuideAST node.
 */
export class DeductionGuideAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeductionGuide(this, context);
  }

  /**
   * Returns the explicitSpecifier of this node
   */
  getExplicitSpecifier(): SpecifierAST | undefined {
    return AST.from<SpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the parameterDeclarationClause of this node
   */
  getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return AST.from<ParameterDeclarationClauseAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the location of the arrow token in this node
   */
  getArrowToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }

  /**
   * Returns the templateId of this node
   */
  getTemplateId(): SimpleTemplateIdAST | undefined {
    return AST.from<SimpleTemplateIdAST>(
      cxx.getASTSlot(this.getHandle(), 6),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 8);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * ExplicitInstantiationAST node.
 */
export class ExplicitInstantiationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExplicitInstantiation(this, context);
  }

  /**
   * Returns the location of the extern token in this node
   */
  getExternToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the declaration of this node
   */
  getDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

/**
 * ExportDeclarationAST node.
 */
export class ExportDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExportDeclaration(this, context);
  }

  /**
   * Returns the location of the export token in this node
   */
  getExportToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the declaration of this node
   */
  getDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * ExportCompoundDeclarationAST node.
 */
export class ExportCompoundDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExportCompoundDeclaration(this, context);
  }

  /**
   * Returns the location of the export token in this node
   */
  getExportToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lbrace token in this node
   */
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the declarationList of this node
   */
  getDeclarationList(): Iterable<DeclarationAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DeclarationAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DeclarationAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rbrace token in this node
   */
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * LinkageSpecificationAST node.
 */
export class LinkageSpecificationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLinkageSpecification(this, context);
  }

  /**
   * Returns the location of the extern token in this node
   */
  getExternToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the stringliteral token in this node
   */
  getStringliteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the lbrace token in this node
   */
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the declarationList of this node
   */
  getDeclarationList(): Iterable<DeclarationAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DeclarationAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DeclarationAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rbrace token in this node
   */
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the stringLiteral attribute of this node
   */
  getStringLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * NamespaceDefinitionAST node.
 */
export class NamespaceDefinitionAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamespaceDefinition(this, context);
  }

  /**
   * Returns the location of the inline token in this node
   */
  getInlineToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the namespace token in this node
   */
  getNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the nestedNamespaceSpecifierList of this node
   */
  getNestedNamespaceSpecifierList(): Iterable<
    NestedNamespaceSpecifierAST | undefined
  > {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: NestedNamespaceSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<NestedNamespaceSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the extraAttributeList of this node
   */
  getExtraAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the lbrace token in this node
   */
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the declarationList of this node
   */
  getDeclarationList(): Iterable<DeclarationAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DeclarationAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DeclarationAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rbrace token in this node
   */
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 8), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 9);
    return cxx.getIdentifierValue(slot);
  }

  /**
   * Returns the isInline attribute of this node
   */
  getIsInline(): boolean {
    return cxx.getASTSlot(this.getHandle(), 10) !== 0;
  }
}

/**
 * EmptyDeclarationAST node.
 */
export class EmptyDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEmptyDeclaration(this, context);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * AttributeDeclarationAST node.
 */
export class AttributeDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttributeDeclaration(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * ModuleImportDeclarationAST node.
 */
export class ModuleImportDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleImportDeclaration(this, context);
  }

  /**
   * Returns the location of the import token in this node
   */
  getImportToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the importName of this node
   */
  getImportName(): ImportNameAST | undefined {
    return AST.from<ImportNameAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * ParameterDeclarationAST node.
 */
export class ParameterDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParameterDeclaration(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the this token in this node
   */
  getThisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeSpecifierList of this node
   */
  getTypeSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declarator of this node
   */
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 6);
    return cxx.getIdentifierValue(slot);
  }

  /**
   * Returns the isThisIntroduced attribute of this node
   */
  getIsThisIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 7) !== 0;
  }

  /**
   * Returns the isPack attribute of this node
   */
  getIsPack(): boolean {
    return cxx.getASTSlot(this.getHandle(), 8) !== 0;
  }
}

/**
 * AccessDeclarationAST node.
 */
export class AccessDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAccessDeclaration(this, context);
  }

  /**
   * Returns the location of the access token in this node
   */
  getAccessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the accessSpecifier attribute of this node
   */
  getAccessSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 2);
  }
}

/**
 * ForRangeDeclarationAST node.
 */
export class ForRangeDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitForRangeDeclaration(this, context);
  }
}

/**
 * StructuredBindingDeclarationAST node.
 */
export class StructuredBindingDeclarationAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStructuredBindingDeclaration(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declSpecifierList of this node
   */
  getDeclSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the refQualifier token in this node
   */
  getRefQualifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the lbracket token in this node
   */
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the bindingList of this node
   */
  getBindingList(): Iterable<NameIdAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: NameIdAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<NameIdAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rbracket token in this node
   */
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 6),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }
}

/**
 * AsmOperandAST node.
 */
export class AsmOperandAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmOperand(this, context);
  }

  /**
   * Returns the location of the lbracket token in this node
   */
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the symbolicName token in this node
   */
  getSymbolicNameToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the rbracket token in this node
   */
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the constraintLiteral token in this node
   */
  getConstraintLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the symbolicName attribute of this node
   */
  getSymbolicName(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 7);
    return cxx.getIdentifierValue(slot);
  }

  /**
   * Returns the constraintLiteral attribute of this node
   */
  getConstraintLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 8);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * AsmQualifierAST node.
 */
export class AsmQualifierAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmQualifier(this, context);
  }

  /**
   * Returns the location of the qualifier token in this node
   */
  getQualifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the qualifier attribute of this node
   */
  getQualifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

/**
 * AsmClobberAST node.
 */
export class AsmClobberAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmClobber(this, context);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * AsmGotoLabelAST node.
 */
export class AsmGotoLabelAST extends DeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmGotoLabel(this, context);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * LabeledStatementAST node.
 */
export class LabeledStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLabeledStatement(this, context);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * CaseStatementAST node.
 */
export class CaseStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCaseStatement(this, context);
  }

  /**
   * Returns the location of the case token in this node
   */
  getCaseToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * DefaultStatementAST node.
 */
export class DefaultStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDefaultStatement(this, context);
  }

  /**
   * Returns the location of the default token in this node
   */
  getDefaultToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * ExpressionStatementAST node.
 */
export class ExpressionStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExpressionStatement(this, context);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * CompoundStatementAST node.
 */
export class CompoundStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundStatement(this, context);
  }

  /**
   * Returns the location of the lbrace token in this node
   */
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the statementList of this node
   */
  getStatementList(): Iterable<StatementAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: StatementAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<StatementAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rbrace token in this node
   */
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * IfStatementAST node.
 */
export class IfStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIfStatement(this, context);
  }

  /**
   * Returns the location of the if token in this node
   */
  getIfToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the constexpr token in this node
   */
  getConstexprToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the condition of this node
   */
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 6),
      this.parser,
    );
  }

  /**
   * Returns the location of the else token in this node
   */
  getElseToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }

  /**
   * Returns the elseStatement of this node
   */
  getElseStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 8),
      this.parser,
    );
  }
}

/**
 * ConstevalIfStatementAST node.
 */
export class ConstevalIfStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstevalIfStatement(this, context);
  }

  /**
   * Returns the location of the if token in this node
   */
  getIfToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the exclaim token in this node
   */
  getExclaimToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the constval token in this node
   */
  getConstvalToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the else token in this node
   */
  getElseToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the elseStatement of this node
   */
  getElseStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the isNot attribute of this node
   */
  getIsNot(): boolean {
    return cxx.getASTSlot(this.getHandle(), 6) !== 0;
  }
}

/**
 * SwitchStatementAST node.
 */
export class SwitchStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSwitchStatement(this, context);
  }

  /**
   * Returns the location of the switch token in this node
   */
  getSwitchToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the condition of this node
   */
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }
}

/**
 * WhileStatementAST node.
 */
export class WhileStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitWhileStatement(this, context);
  }

  /**
   * Returns the location of the while token in this node
   */
  getWhileToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the condition of this node
   */
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

/**
 * DoStatementAST node.
 */
export class DoStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDoStatement(this, context);
  }

  /**
   * Returns the location of the do token in this node
   */
  getDoToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the while token in this node
   */
  getWhileToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
}

/**
 * ForRangeStatementAST node.
 */
export class ForRangeStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitForRangeStatement(this, context);
  }

  /**
   * Returns the location of the for token in this node
   */
  getForToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the rangeDeclaration of this node
   */
  getRangeDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the rangeInitializer of this node
   */
  getRangeInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 7),
      this.parser,
    );
  }
}

/**
 * ForStatementAST node.
 */
export class ForStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitForStatement(this, context);
  }

  /**
   * Returns the location of the for token in this node
   */
  getForToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the condition of this node
   */
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): StatementAST | undefined {
    return AST.from<StatementAST>(
      cxx.getASTSlot(this.getHandle(), 7),
      this.parser,
    );
  }
}

/**
 * BreakStatementAST node.
 */
export class BreakStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBreakStatement(this, context);
  }

  /**
   * Returns the location of the break token in this node
   */
  getBreakToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * ContinueStatementAST node.
 */
export class ContinueStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitContinueStatement(this, context);
  }

  /**
   * Returns the location of the continue token in this node
   */
  getContinueToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * ReturnStatementAST node.
 */
export class ReturnStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitReturnStatement(this, context);
  }

  /**
   * Returns the location of the return token in this node
   */
  getReturnToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * CoroutineReturnStatementAST node.
 */
export class CoroutineReturnStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCoroutineReturnStatement(this, context);
  }

  /**
   * Returns the location of the coreturn token in this node
   */
  getCoreturnToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * GotoStatementAST node.
 */
export class GotoStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGotoStatement(this, context);
  }

  /**
   * Returns the location of the goto token in this node
   */
  getGotoToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * DeclarationStatementAST node.
 */
export class DeclarationStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeclarationStatement(this, context);
  }

  /**
   * Returns the declaration of this node
   */
  getDeclaration(): DeclarationAST | undefined {
    return AST.from<DeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

/**
 * TryBlockStatementAST node.
 */
export class TryBlockStatementAST extends StatementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTryBlockStatement(this, context);
  }

  /**
   * Returns the location of the try token in this node
   */
  getTryToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the handlerList of this node
   */
  getHandlerList(): Iterable<HandlerAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: HandlerAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<HandlerAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }
}

/**
 * GeneratedLiteralExpressionAST node.
 */
export class GeneratedLiteralExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGeneratedLiteralExpression(this, context);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * CharLiteralExpressionAST node.
 */
export class CharLiteralExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCharLiteralExpression(this, context);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * BoolLiteralExpressionAST node.
 */
export class BoolLiteralExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBoolLiteralExpression(this, context);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the isTrue attribute of this node
   */
  getIsTrue(): boolean {
    return cxx.getASTSlot(this.getHandle(), 1) !== 0;
  }
}

/**
 * IntLiteralExpressionAST node.
 */
export class IntLiteralExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIntLiteralExpression(this, context);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * FloatLiteralExpressionAST node.
 */
export class FloatLiteralExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFloatLiteralExpression(this, context);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * NullptrLiteralExpressionAST node.
 */
export class NullptrLiteralExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNullptrLiteralExpression(this, context);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

/**
 * StringLiteralExpressionAST node.
 */
export class StringLiteralExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStringLiteralExpression(this, context);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * UserDefinedStringLiteralExpressionAST node.
 */
export class UserDefinedStringLiteralExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUserDefinedStringLiteralExpression(this, context);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * ObjectLiteralExpressionAST node.
 */
export class ObjectLiteralExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitObjectLiteralExpression(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the bracedInitList of this node
   */
  getBracedInitList(): BracedInitListAST | undefined {
    return AST.from<BracedInitListAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
}

/**
 * ThisExpressionAST node.
 */
export class ThisExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThisExpression(this, context);
  }

  /**
   * Returns the location of the this token in this node
   */
  getThisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * NestedStatementExpressionAST node.
 */
export class NestedStatementExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedStatementExpression(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * NestedExpressionAST node.
 */
export class NestedExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedExpression(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * IdExpressionAST node.
 */
export class IdExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIdExpression(this, context);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 3) !== 0;
  }
}

/**
 * LambdaExpressionAST node.
 */
export class LambdaExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLambdaExpression(this, context);
  }

  /**
   * Returns the location of the lbracket token in this node
   */
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the captureDefault token in this node
   */
  getCaptureDefaultToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the captureList of this node
   */
  getCaptureList(): Iterable<LambdaCaptureAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: LambdaCaptureAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<LambdaCaptureAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rbracket token in this node
   */
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the less token in this node
   */
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the templateParameterList of this node
   */
  getTemplateParameterList(): Iterable<TemplateParameterAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: TemplateParameterAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<TemplateParameterAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the greater token in this node
   */
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the templateRequiresClause of this node
   */
  getTemplateRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 7),
      this.parser,
    );
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 8), this.parser);
  }

  /**
   * Returns the parameterDeclarationClause of this node
   */
  getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return AST.from<ParameterDeclarationClauseAST>(
      cxx.getASTSlot(this.getHandle(), 9),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 10), this.parser);
  }

  /**
   * Returns the gnuAtributeList of this node
   */
  getGnuAtributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the lambdaSpecifierList of this node
   */
  getLambdaSpecifierList(): Iterable<LambdaSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: LambdaSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<LambdaSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the exceptionSpecifier of this node
   */
  getExceptionSpecifier(): ExceptionSpecifierAST | undefined {
    return AST.from<ExceptionSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 13),
      this.parser,
    );
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the trailingReturnType of this node
   */
  getTrailingReturnType(): TrailingReturnTypeAST | undefined {
    return AST.from<TrailingReturnTypeAST>(
      cxx.getASTSlot(this.getHandle(), 15),
      this.parser,
    );
  }

  /**
   * Returns the requiresClause of this node
   */
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 16),
      this.parser,
    );
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 17),
      this.parser,
    );
  }

  /**
   * Returns the captureDefault attribute of this node
   */
  getCaptureDefault(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 18);
  }
}

/**
 * FoldExpressionAST node.
 */
export class FoldExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFoldExpression(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the leftExpression of this node
   */
  getLeftExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the op token in this node
   */
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the foldOp token in this node
   */
  getFoldOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the rightExpression of this node
   */
  getRightExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the op attribute of this node
   */
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 7);
  }

  /**
   * Returns the foldOp attribute of this node
   */
  getFoldOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 8);
  }
}

/**
 * RightFoldExpressionAST node.
 */
export class RightFoldExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRightFoldExpression(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the op token in this node
   */
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the op attribute of this node
   */
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 5);
  }
}

/**
 * LeftFoldExpressionAST node.
 */
export class LeftFoldExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLeftFoldExpression(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the op token in this node
   */
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the op attribute of this node
   */
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 5);
  }
}

/**
 * RequiresExpressionAST node.
 */
export class RequiresExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRequiresExpression(this, context);
  }

  /**
   * Returns the location of the requires token in this node
   */
  getRequiresToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the parameterDeclarationClause of this node
   */
  getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return AST.from<ParameterDeclarationClauseAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the lbrace token in this node
   */
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the requirementList of this node
   */
  getRequirementList(): Iterable<RequirementAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: RequirementAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<RequirementAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rbrace token in this node
   */
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
}

/**
 * VaArgExpressionAST node.
 */
export class VaArgExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVaArgExpression(this, context);
  }

  /**
   * Returns the location of the vaArg token in this node
   */
  getVaArgToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the comma token in this node
   */
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

/**
 * SubscriptExpressionAST node.
 */
export class SubscriptExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSubscriptExpression(this, context);
  }

  /**
   * Returns the baseExpression of this node
   */
  getBaseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the lbracket token in this node
   */
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the indexExpression of this node
   */
  getIndexExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rbracket token in this node
   */
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * CallExpressionAST node.
 */
export class CallExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCallExpression(this, context);
  }

  /**
   * Returns the baseExpression of this node
   */
  getBaseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expressionList of this node
   */
  getExpressionList(): Iterable<ExpressionAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: ExpressionAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<ExpressionAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * TypeConstructionAST node.
 */
export class TypeConstructionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeConstruction(this, context);
  }

  /**
   * Returns the typeSpecifier of this node
   */
  getTypeSpecifier(): SpecifierAST | undefined {
    return AST.from<SpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expressionList of this node
   */
  getExpressionList(): Iterable<ExpressionAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: ExpressionAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<ExpressionAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * BracedTypeConstructionAST node.
 */
export class BracedTypeConstructionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBracedTypeConstruction(this, context);
  }

  /**
   * Returns the typeSpecifier of this node
   */
  getTypeSpecifier(): SpecifierAST | undefined {
    return AST.from<SpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the bracedInitList of this node
   */
  getBracedInitList(): BracedInitListAST | undefined {
    return AST.from<BracedInitListAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * SpliceMemberExpressionAST node.
 */
export class SpliceMemberExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSpliceMemberExpression(this, context);
  }

  /**
   * Returns the baseExpression of this node
   */
  getBaseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the access token in this node
   */
  getAccessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the splicer of this node
   */
  getSplicer(): SplicerAST | undefined {
    return AST.from<SplicerAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the accessOp attribute of this node
   */
  getAccessOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 4);
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 5) !== 0;
  }
}

/**
 * MemberExpressionAST node.
 */
export class MemberExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitMemberExpression(this, context);
  }

  /**
   * Returns the baseExpression of this node
   */
  getBaseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the access token in this node
   */
  getAccessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the accessOp attribute of this node
   */
  getAccessOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 5);
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 6) !== 0;
  }
}

/**
 * PostIncrExpressionAST node.
 */
export class PostIncrExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPostIncrExpression(this, context);
  }

  /**
   * Returns the baseExpression of this node
   */
  getBaseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the op token in this node
   */
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the op attribute of this node
   */
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 2);
  }
}

/**
 * CppCastExpressionAST node.
 */
export class CppCastExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCppCastExpression(this, context);
  }

  /**
   * Returns the location of the cast token in this node
   */
  getCastToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the less token in this node
   */
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the greater token in this node
   */
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
}

/**
 * BuiltinBitCastExpressionAST node.
 */
export class BuiltinBitCastExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBuiltinBitCastExpression(this, context);
  }

  /**
   * Returns the location of the cast token in this node
   */
  getCastToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the comma token in this node
   */
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

/**
 * BuiltinOffsetofExpressionAST node.
 */
export class BuiltinOffsetofExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBuiltinOffsetofExpression(this, context);
  }

  /**
   * Returns the location of the offsetof token in this node
   */
  getOffsetofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the comma token in this node
   */
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

/**
 * TypeidExpressionAST node.
 */
export class TypeidExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeidExpression(this, context);
  }

  /**
   * Returns the location of the typeid token in this node
   */
  getTypeidToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * TypeidOfTypeExpressionAST node.
 */
export class TypeidOfTypeExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeidOfTypeExpression(this, context);
  }

  /**
   * Returns the location of the typeid token in this node
   */
  getTypeidToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * SpliceExpressionAST node.
 */
export class SpliceExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSpliceExpression(this, context);
  }

  /**
   * Returns the splicer of this node
   */
  getSplicer(): SplicerAST | undefined {
    return AST.from<SplicerAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

/**
 * GlobalScopeReflectExpressionAST node.
 */
export class GlobalScopeReflectExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGlobalScopeReflectExpression(this, context);
  }

  /**
   * Returns the location of the caret token in this node
   */
  getCaretToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the scope token in this node
   */
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * NamespaceReflectExpressionAST node.
 */
export class NamespaceReflectExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamespaceReflectExpression(this, context);
  }

  /**
   * Returns the location of the caret token in this node
   */
  getCaretToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * TypeIdReflectExpressionAST node.
 */
export class TypeIdReflectExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeIdReflectExpression(this, context);
  }

  /**
   * Returns the location of the caret token in this node
   */
  getCaretToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * ReflectExpressionAST node.
 */
export class ReflectExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitReflectExpression(this, context);
  }

  /**
   * Returns the location of the caret token in this node
   */
  getCaretToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * UnaryExpressionAST node.
 */
export class UnaryExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUnaryExpression(this, context);
  }

  /**
   * Returns the location of the op token in this node
   */
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the op attribute of this node
   */
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 2);
  }
}

/**
 * AwaitExpressionAST node.
 */
export class AwaitExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAwaitExpression(this, context);
  }

  /**
   * Returns the location of the await token in this node
   */
  getAwaitToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * SizeofExpressionAST node.
 */
export class SizeofExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeofExpression(this, context);
  }

  /**
   * Returns the location of the sizeof token in this node
   */
  getSizeofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * SizeofTypeExpressionAST node.
 */
export class SizeofTypeExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeofTypeExpression(this, context);
  }

  /**
   * Returns the location of the sizeof token in this node
   */
  getSizeofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * SizeofPackExpressionAST node.
 */
export class SizeofPackExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeofPackExpression(this, context);
  }

  /**
   * Returns the location of the sizeof token in this node
   */
  getSizeofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * AlignofTypeExpressionAST node.
 */
export class AlignofTypeExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignofTypeExpression(this, context);
  }

  /**
   * Returns the location of the alignof token in this node
   */
  getAlignofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * AlignofExpressionAST node.
 */
export class AlignofExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignofExpression(this, context);
  }

  /**
   * Returns the location of the alignof token in this node
   */
  getAlignofToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * NoexceptExpressionAST node.
 */
export class NoexceptExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNoexceptExpression(this, context);
  }

  /**
   * Returns the location of the noexcept token in this node
   */
  getNoexceptToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * NewExpressionAST node.
 */
export class NewExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewExpression(this, context);
  }

  /**
   * Returns the location of the scope token in this node
   */
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the new token in this node
   */
  getNewToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the newPlacement of this node
   */
  getNewPlacement(): NewPlacementAST | undefined {
    return AST.from<NewPlacementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the typeSpecifierList of this node
   */
  getTypeSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declarator of this node
   */
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the newInitalizer of this node
   */
  getNewInitalizer(): NewInitializerAST | undefined {
    return AST.from<NewInitializerAST>(
      cxx.getASTSlot(this.getHandle(), 7),
      this.parser,
    );
  }
}

/**
 * DeleteExpressionAST node.
 */
export class DeleteExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeleteExpression(this, context);
  }

  /**
   * Returns the location of the scope token in this node
   */
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the delete token in this node
   */
  getDeleteToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the lbracket token in this node
   */
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the rbracket token in this node
   */
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

/**
 * CastExpressionAST node.
 */
export class CastExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCastExpression(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
}

/**
 * ImplicitCastExpressionAST node.
 */
export class ImplicitCastExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitImplicitCastExpression(this, context);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

/**
 * BinaryExpressionAST node.
 */
export class BinaryExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBinaryExpression(this, context);
  }

  /**
   * Returns the leftExpression of this node
   */
  getLeftExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the op token in this node
   */
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the rightExpression of this node
   */
  getRightExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the op attribute of this node
   */
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 3);
  }
}

/**
 * ConditionalExpressionAST node.
 */
export class ConditionalExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConditionalExpression(this, context);
  }

  /**
   * Returns the condition of this node
   */
  getCondition(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the question token in this node
   */
  getQuestionToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the iftrueExpression of this node
   */
  getIftrueExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the iffalseExpression of this node
   */
  getIffalseExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

/**
 * YieldExpressionAST node.
 */
export class YieldExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitYieldExpression(this, context);
  }

  /**
   * Returns the location of the yield token in this node
   */
  getYieldToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * ThrowExpressionAST node.
 */
export class ThrowExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThrowExpression(this, context);
  }

  /**
   * Returns the location of the throw token in this node
   */
  getThrowToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * AssignmentExpressionAST node.
 */
export class AssignmentExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAssignmentExpression(this, context);
  }

  /**
   * Returns the leftExpression of this node
   */
  getLeftExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the op token in this node
   */
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the rightExpression of this node
   */
  getRightExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the op attribute of this node
   */
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 3);
  }
}

/**
 * PackExpansionExpressionAST node.
 */
export class PackExpansionExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPackExpansionExpression(this, context);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * DesignatedInitializerClauseAST node.
 */
export class DesignatedInitializerClauseAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDesignatedInitializerClause(this, context);
  }

  /**
   * Returns the designatorList of this node
   */
  getDesignatorList(): Iterable<DesignatorAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DesignatorAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DesignatorAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * TypeTraitExpressionAST node.
 */
export class TypeTraitExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeTraitExpression(this, context);
  }

  /**
   * Returns the location of the typeTrait token in this node
   */
  getTypeTraitToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeIdList of this node
   */
  getTypeIdList(): Iterable<TypeIdAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: TypeIdAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<TypeIdAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * ConditionExpressionAST node.
 */
export class ConditionExpressionAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConditionExpression(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declSpecifierList of this node
   */
  getDeclSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declarator of this node
   */
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }
}

/**
 * EqualInitializerAST node.
 */
export class EqualInitializerAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEqualInitializer(this, context);
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * BracedInitListAST node.
 */
export class BracedInitListAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBracedInitList(this, context);
  }

  /**
   * Returns the location of the lbrace token in this node
   */
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expressionList of this node
   */
  getExpressionList(): Iterable<ExpressionAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: ExpressionAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<ExpressionAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the comma token in this node
   */
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the rbrace token in this node
   */
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * ParenInitializerAST node.
 */
export class ParenInitializerAST extends ExpressionAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParenInitializer(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expressionList of this node
   */
  getExpressionList(): Iterable<ExpressionAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: ExpressionAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<ExpressionAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * DotDesignatorAST node.
 */
export class DotDesignatorAST extends DesignatorAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDotDesignator(this, context);
  }

  /**
   * Returns the location of the dot token in this node
   */
  getDotToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * SubscriptDesignatorAST node.
 */
export class SubscriptDesignatorAST extends DesignatorAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSubscriptDesignator(this, context);
  }

  /**
   * Returns the location of the lbracket token in this node
   */
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the rbracket token in this node
   */
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * SplicerAST node.
 */
export class SplicerAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSplicer(this, context);
  }

  /**
   * Returns the location of the lbracket token in this node
   */
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the secondColon token in this node
   */
  getSecondColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the location of the rbracket token in this node
   */
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

/**
 * GlobalModuleFragmentAST node.
 */
export class GlobalModuleFragmentAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGlobalModuleFragment(this, context);
  }

  /**
   * Returns the location of the module token in this node
   */
  getModuleToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the declarationList of this node
   */
  getDeclarationList(): Iterable<DeclarationAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DeclarationAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DeclarationAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }
}

/**
 * PrivateModuleFragmentAST node.
 */
export class PrivateModuleFragmentAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPrivateModuleFragment(this, context);
  }

  /**
   * Returns the location of the module token in this node
   */
  getModuleToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the private token in this node
   */
  getPrivateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the declarationList of this node
   */
  getDeclarationList(): Iterable<DeclarationAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DeclarationAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DeclarationAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }
}

/**
 * ModuleDeclarationAST node.
 */
export class ModuleDeclarationAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleDeclaration(this, context);
  }

  /**
   * Returns the location of the export token in this node
   */
  getExportToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the module token in this node
   */
  getModuleToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the moduleName of this node
   */
  getModuleName(): ModuleNameAST | undefined {
    return AST.from<ModuleNameAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the modulePartition of this node
   */
  getModulePartition(): ModulePartitionAST | undefined {
    return AST.from<ModulePartitionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

/**
 * ModuleNameAST node.
 */
export class ModuleNameAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleName(this, context);
  }

  /**
   * Returns the moduleQualifier of this node
   */
  getModuleQualifier(): ModuleQualifierAST | undefined {
    return AST.from<ModuleQualifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * ModuleQualifierAST node.
 */
export class ModuleQualifierAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleQualifier(this, context);
  }

  /**
   * Returns the moduleQualifier of this node
   */
  getModuleQualifier(): ModuleQualifierAST | undefined {
    return AST.from<ModuleQualifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the dot token in this node
   */
  getDotToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * ModulePartitionAST node.
 */
export class ModulePartitionAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModulePartition(this, context);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the moduleName of this node
   */
  getModuleName(): ModuleNameAST | undefined {
    return AST.from<ModuleNameAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * ImportNameAST node.
 */
export class ImportNameAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitImportName(this, context);
  }

  /**
   * Returns the location of the header token in this node
   */
  getHeaderToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the modulePartition of this node
   */
  getModulePartition(): ModulePartitionAST | undefined {
    return AST.from<ModulePartitionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the moduleName of this node
   */
  getModuleName(): ModuleNameAST | undefined {
    return AST.from<ModuleNameAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

/**
 * InitDeclaratorAST node.
 */
export class InitDeclaratorAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitInitDeclarator(this, context);
  }

  /**
   * Returns the declarator of this node
   */
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the requiresClause of this node
   */
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

/**
 * DeclaratorAST node.
 */
export class DeclaratorAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeclarator(this, context);
  }

  /**
   * Returns the ptrOpList of this node
   */
  getPtrOpList(): Iterable<PtrOperatorAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: PtrOperatorAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<PtrOperatorAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the coreDeclarator of this node
   */
  getCoreDeclarator(): CoreDeclaratorAST | undefined {
    return AST.from<CoreDeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the declaratorChunkList of this node
   */
  getDeclaratorChunkList(): Iterable<DeclaratorChunkAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DeclaratorChunkAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DeclaratorChunkAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }
}

/**
 * UsingDeclaratorAST node.
 */
export class UsingDeclaratorAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingDeclarator(this, context);
  }

  /**
   * Returns the location of the typename token in this node
   */
  getTypenameToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the isPack attribute of this node
   */
  getIsPack(): boolean {
    return cxx.getASTSlot(this.getHandle(), 4) !== 0;
  }
}

/**
 * EnumeratorAST node.
 */
export class EnumeratorAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEnumerator(this, context);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * TypeIdAST node.
 */
export class TypeIdAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeId(this, context);
  }

  /**
   * Returns the typeSpecifierList of this node
   */
  getTypeSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declarator of this node
   */
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * HandlerAST node.
 */
export class HandlerAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitHandler(this, context);
  }

  /**
   * Returns the location of the catch token in this node
   */
  getCatchToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the exceptionDeclaration of this node
   */
  getExceptionDeclaration(): ExceptionDeclarationAST | undefined {
    return AST.from<ExceptionDeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }
}

/**
 * BaseSpecifierAST node.
 */
export class BaseSpecifierAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBaseSpecifier(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the virtualOrAccess token in this node
   */
  getVirtualOrAccessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the otherVirtualOrAccess token in this node
   */
  getOtherVirtualOrAccessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 7) !== 0;
  }

  /**
   * Returns the isVirtual attribute of this node
   */
  getIsVirtual(): boolean {
    return cxx.getASTSlot(this.getHandle(), 8) !== 0;
  }

  /**
   * Returns the isVariadic attribute of this node
   */
  getIsVariadic(): boolean {
    return cxx.getASTSlot(this.getHandle(), 9) !== 0;
  }

  /**
   * Returns the accessSpecifier attribute of this node
   */
  getAccessSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 10);
  }
}

/**
 * RequiresClauseAST node.
 */
export class RequiresClauseAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRequiresClause(this, context);
  }

  /**
   * Returns the location of the requires token in this node
   */
  getRequiresToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * ParameterDeclarationClauseAST node.
 */
export class ParameterDeclarationClauseAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParameterDeclarationClause(this, context);
  }

  /**
   * Returns the parameterDeclarationList of this node
   */
  getParameterDeclarationList(): Iterable<ParameterDeclarationAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: ParameterDeclarationAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<ParameterDeclarationAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the comma token in this node
   */
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the isVariadic attribute of this node
   */
  getIsVariadic(): boolean {
    return cxx.getASTSlot(this.getHandle(), 3) !== 0;
  }
}

/**
 * TrailingReturnTypeAST node.
 */
export class TrailingReturnTypeAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTrailingReturnType(this, context);
  }

  /**
   * Returns the location of the minusGreater token in this node
   */
  getMinusGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * LambdaSpecifierAST node.
 */
export class LambdaSpecifierAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLambdaSpecifier(this, context);
  }

  /**
   * Returns the location of the specifier token in this node
   */
  getSpecifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the specifier attribute of this node
   */
  getSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

/**
 * TypeConstraintAST node.
 */
export class TypeConstraintAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeConstraint(this, context);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the less token in this node
   */
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the templateArgumentList of this node
   */
  getTemplateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: TemplateArgumentAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<TemplateArgumentAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the greater token in this node
   */
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * AttributeArgumentClauseAST node.
 */
export class AttributeArgumentClauseAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttributeArgumentClause(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * AttributeAST node.
 */
export class AttributeAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttribute(this, context);
  }

  /**
   * Returns the attributeToken of this node
   */
  getAttributeToken(): AttributeTokenAST | undefined {
    return AST.from<AttributeTokenAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the attributeArgumentClause of this node
   */
  getAttributeArgumentClause(): AttributeArgumentClauseAST | undefined {
    return AST.from<AttributeArgumentClauseAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * AttributeUsingPrefixAST node.
 */
export class AttributeUsingPrefixAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttributeUsingPrefix(this, context);
  }

  /**
   * Returns the location of the using token in this node
   */
  getUsingToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the attributeNamespace token in this node
   */
  getAttributeNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * NewPlacementAST node.
 */
export class NewPlacementAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewPlacement(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expressionList of this node
   */
  getExpressionList(): Iterable<ExpressionAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: ExpressionAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<ExpressionAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * NestedNamespaceSpecifierAST node.
 */
export class NestedNamespaceSpecifierAST extends AST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedNamespaceSpecifier(this, context);
  }

  /**
   * Returns the location of the inline token in this node
   */
  getInlineToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the scope token in this node
   */
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }

  /**
   * Returns the isInline attribute of this node
   */
  getIsInline(): boolean {
    return cxx.getASTSlot(this.getHandle(), 4) !== 0;
  }
}

/**
 * TemplateTypeParameterAST node.
 */
export class TemplateTypeParameterAST extends TemplateParameterAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplateTypeParameter(this, context);
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the less token in this node
   */
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the templateParameterList of this node
   */
  getTemplateParameterList(): Iterable<TemplateParameterAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: TemplateParameterAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<TemplateParameterAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the greater token in this node
   */
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the requiresClause of this node
   */
  getRequiresClause(): RequiresClauseAST | undefined {
    return AST.from<RequiresClauseAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the classKey token in this node
   */
  getClassKeyToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 8), this.parser);
  }

  /**
   * Returns the idExpression of this node
   */
  getIdExpression(): IdExpressionAST | undefined {
    return AST.from<IdExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 9),
      this.parser,
    );
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 10);
    return cxx.getIdentifierValue(slot);
  }

  /**
   * Returns the isPack attribute of this node
   */
  getIsPack(): boolean {
    return cxx.getASTSlot(this.getHandle(), 11) !== 0;
  }
}

/**
 * NonTypeTemplateParameterAST node.
 */
export class NonTypeTemplateParameterAST extends TemplateParameterAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNonTypeTemplateParameter(this, context);
  }

  /**
   * Returns the declaration of this node
   */
  getDeclaration(): ParameterDeclarationAST | undefined {
    return AST.from<ParameterDeclarationAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

/**
 * TypenameTypeParameterAST node.
 */
export class TypenameTypeParameterAST extends TemplateParameterAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypenameTypeParameter(this, context);
  }

  /**
   * Returns the location of the classKey token in this node
   */
  getClassKeyToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getIdentifierValue(slot);
  }

  /**
   * Returns the isPack attribute of this node
   */
  getIsPack(): boolean {
    return cxx.getASTSlot(this.getHandle(), 6) !== 0;
  }
}

/**
 * ConstraintTypeParameterAST node.
 */
export class ConstraintTypeParameterAST extends TemplateParameterAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstraintTypeParameter(this, context);
  }

  /**
   * Returns the typeConstraint of this node
   */
  getTypeConstraint(): TypeConstraintAST | undefined {
    return AST.from<TypeConstraintAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 5);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * GeneratedTypeSpecifierAST node.
 */
export class GeneratedTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGeneratedTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the type token in this node
   */
  getTypeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * TypedefSpecifierAST node.
 */
export class TypedefSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypedefSpecifier(this, context);
  }

  /**
   * Returns the location of the typedef token in this node
   */
  getTypedefToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * FriendSpecifierAST node.
 */
export class FriendSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFriendSpecifier(this, context);
  }

  /**
   * Returns the location of the friend token in this node
   */
  getFriendToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * ConstevalSpecifierAST node.
 */
export class ConstevalSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstevalSpecifier(this, context);
  }

  /**
   * Returns the location of the consteval token in this node
   */
  getConstevalToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * ConstinitSpecifierAST node.
 */
export class ConstinitSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstinitSpecifier(this, context);
  }

  /**
   * Returns the location of the constinit token in this node
   */
  getConstinitToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * ConstexprSpecifierAST node.
 */
export class ConstexprSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstexprSpecifier(this, context);
  }

  /**
   * Returns the location of the constexpr token in this node
   */
  getConstexprToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * InlineSpecifierAST node.
 */
export class InlineSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitInlineSpecifier(this, context);
  }

  /**
   * Returns the location of the inline token in this node
   */
  getInlineToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * NoreturnSpecifierAST node.
 */
export class NoreturnSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNoreturnSpecifier(this, context);
  }

  /**
   * Returns the location of the noreturn token in this node
   */
  getNoreturnToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * StaticSpecifierAST node.
 */
export class StaticSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStaticSpecifier(this, context);
  }

  /**
   * Returns the location of the static token in this node
   */
  getStaticToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * ExternSpecifierAST node.
 */
export class ExternSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExternSpecifier(this, context);
  }

  /**
   * Returns the location of the extern token in this node
   */
  getExternToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * RegisterSpecifierAST node.
 */
export class RegisterSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRegisterSpecifier(this, context);
  }

  /**
   * Returns the location of the register token in this node
   */
  getRegisterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * ThreadLocalSpecifierAST node.
 */
export class ThreadLocalSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThreadLocalSpecifier(this, context);
  }

  /**
   * Returns the location of the threadLocal token in this node
   */
  getThreadLocalToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * ThreadSpecifierAST node.
 */
export class ThreadSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThreadSpecifier(this, context);
  }

  /**
   * Returns the location of the thread token in this node
   */
  getThreadToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * MutableSpecifierAST node.
 */
export class MutableSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitMutableSpecifier(this, context);
  }

  /**
   * Returns the location of the mutable token in this node
   */
  getMutableToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * VirtualSpecifierAST node.
 */
export class VirtualSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVirtualSpecifier(this, context);
  }

  /**
   * Returns the location of the virtual token in this node
   */
  getVirtualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * ExplicitSpecifierAST node.
 */
export class ExplicitSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExplicitSpecifier(this, context);
  }

  /**
   * Returns the location of the explicit token in this node
   */
  getExplicitToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * AutoTypeSpecifierAST node.
 */
export class AutoTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAutoTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the auto token in this node
   */
  getAutoToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * VoidTypeSpecifierAST node.
 */
export class VoidTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVoidTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the void token in this node
   */
  getVoidToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * SizeTypeSpecifierAST node.
 */
export class SizeTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the specifier token in this node
   */
  getSpecifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the specifier attribute of this node
   */
  getSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

/**
 * SignTypeSpecifierAST node.
 */
export class SignTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSignTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the specifier token in this node
   */
  getSpecifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the specifier attribute of this node
   */
  getSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

/**
 * VaListTypeSpecifierAST node.
 */
export class VaListTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVaListTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the specifier token in this node
   */
  getSpecifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the specifier attribute of this node
   */
  getSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

/**
 * IntegralTypeSpecifierAST node.
 */
export class IntegralTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIntegralTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the specifier token in this node
   */
  getSpecifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the specifier attribute of this node
   */
  getSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

/**
 * FloatingPointTypeSpecifierAST node.
 */
export class FloatingPointTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFloatingPointTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the specifier token in this node
   */
  getSpecifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the specifier attribute of this node
   */
  getSpecifier(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 1);
  }
}

/**
 * ComplexTypeSpecifierAST node.
 */
export class ComplexTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitComplexTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the complex token in this node
   */
  getComplexToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * NamedTypeSpecifierAST node.
 */
export class NamedTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamedTypeSpecifier(this, context);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 3) !== 0;
  }
}

/**
 * AtomicTypeSpecifierAST node.
 */
export class AtomicTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAtomicTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the atomic token in this node
   */
  getAtomicToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * UnderlyingTypeSpecifierAST node.
 */
export class UnderlyingTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUnderlyingTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the underlyingType token in this node
   */
  getUnderlyingTypeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * ElaboratedTypeSpecifierAST node.
 */
export class ElaboratedTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitElaboratedTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the class token in this node
   */
  getClassToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the classKey attribute of this node
   */
  getClassKey(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 5);
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 6) !== 0;
  }
}

/**
 * DecltypeAutoSpecifierAST node.
 */
export class DecltypeAutoSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeAutoSpecifier(this, context);
  }

  /**
   * Returns the location of the decltype token in this node
   */
  getDecltypeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the auto token in this node
   */
  getAutoToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * DecltypeSpecifierAST node.
 */
export class DecltypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeSpecifier(this, context);
  }

  /**
   * Returns the location of the decltype token in this node
   */
  getDecltypeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * PlaceholderTypeSpecifierAST node.
 */
export class PlaceholderTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPlaceholderTypeSpecifier(this, context);
  }

  /**
   * Returns the typeConstraint of this node
   */
  getTypeConstraint(): TypeConstraintAST | undefined {
    return AST.from<TypeConstraintAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the specifier of this node
   */
  getSpecifier(): SpecifierAST | undefined {
    return AST.from<SpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * ConstQualifierAST node.
 */
export class ConstQualifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstQualifier(this, context);
  }

  /**
   * Returns the location of the const token in this node
   */
  getConstToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * VolatileQualifierAST node.
 */
export class VolatileQualifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVolatileQualifier(this, context);
  }

  /**
   * Returns the location of the volatile token in this node
   */
  getVolatileToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * AtomicQualifierAST node.
 */
export class AtomicQualifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAtomicQualifier(this, context);
  }

  /**
   * Returns the location of the atomic token in this node
   */
  getAtomicToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * RestrictQualifierAST node.
 */
export class RestrictQualifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRestrictQualifier(this, context);
  }

  /**
   * Returns the location of the restrict token in this node
   */
  getRestrictToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * EnumSpecifierAST node.
 */
export class EnumSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEnumSpecifier(this, context);
  }

  /**
   * Returns the location of the enum token in this node
   */
  getEnumToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the class token in this node
   */
  getClassToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): NameIdAST | undefined {
    return AST.from<NameIdAST>(
      cxx.getASTSlot(this.getHandle(), 4),
      this.parser,
    );
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }

  /**
   * Returns the typeSpecifierList of this node
   */
  getTypeSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the lbrace token in this node
   */
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }

  /**
   * Returns the enumeratorList of this node
   */
  getEnumeratorList(): Iterable<EnumeratorAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: EnumeratorAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<EnumeratorAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the comma token in this node
   */
  getCommaToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 9), this.parser);
  }

  /**
   * Returns the location of the rbrace token in this node
   */
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 10), this.parser);
  }
}

/**
 * ClassSpecifierAST node.
 */
export class ClassSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitClassSpecifier(this, context);
  }

  /**
   * Returns the location of the class token in this node
   */
  getClassToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the final token in this node
   */
  getFinalToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }

  /**
   * Returns the baseSpecifierList of this node
   */
  getBaseSpecifierList(): Iterable<BaseSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: BaseSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<BaseSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the lbrace token in this node
   */
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 7), this.parser);
  }

  /**
   * Returns the declarationList of this node
   */
  getDeclarationList(): Iterable<DeclarationAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: DeclarationAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<DeclarationAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rbrace token in this node
   */
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 9), this.parser);
  }

  /**
   * Returns the classKey attribute of this node
   */
  getClassKey(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 10);
  }

  /**
   * Returns the isFinal attribute of this node
   */
  getIsFinal(): boolean {
    return cxx.getASTSlot(this.getHandle(), 11) !== 0;
  }
}

/**
 * TypenameSpecifierAST node.
 */
export class TypenameSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypenameSpecifier(this, context);
  }

  /**
   * Returns the location of the typename token in this node
   */
  getTypenameToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 4) !== 0;
  }
}

/**
 * SplicerTypeSpecifierAST node.
 */
export class SplicerTypeSpecifierAST extends SpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSplicerTypeSpecifier(this, context);
  }

  /**
   * Returns the location of the typename token in this node
   */
  getTypenameToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the splicer of this node
   */
  getSplicer(): SplicerAST | undefined {
    return AST.from<SplicerAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * PointerOperatorAST node.
 */
export class PointerOperatorAST extends PtrOperatorAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPointerOperator(this, context);
  }

  /**
   * Returns the location of the star token in this node
   */
  getStarToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the cvQualifierList of this node
   */
  getCvQualifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }
}

/**
 * ReferenceOperatorAST node.
 */
export class ReferenceOperatorAST extends PtrOperatorAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitReferenceOperator(this, context);
  }

  /**
   * Returns the location of the ref token in this node
   */
  getRefToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the refOp attribute of this node
   */
  getRefOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 2);
  }
}

/**
 * PtrToMemberOperatorAST node.
 */
export class PtrToMemberOperatorAST extends PtrOperatorAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPtrToMemberOperator(this, context);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the star token in this node
   */
  getStarToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the cvQualifierList of this node
   */
  getCvQualifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }
}

/**
 * BitfieldDeclaratorAST node.
 */
export class BitfieldDeclaratorAST extends CoreDeclaratorAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBitfieldDeclarator(this, context);
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): NameIdAST | undefined {
    return AST.from<NameIdAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the sizeExpression of this node
   */
  getSizeExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

/**
 * ParameterPackAST node.
 */
export class ParameterPackAST extends CoreDeclaratorAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParameterPack(this, context);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the coreDeclarator of this node
   */
  getCoreDeclarator(): CoreDeclaratorAST | undefined {
    return AST.from<CoreDeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * IdDeclaratorAST node.
 */
export class IdDeclaratorAST extends CoreDeclaratorAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIdDeclarator(this, context);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 4) !== 0;
  }
}

/**
 * NestedDeclaratorAST node.
 */
export class NestedDeclaratorAST extends CoreDeclaratorAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedDeclarator(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the declarator of this node
   */
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * FunctionDeclaratorChunkAST node.
 */
export class FunctionDeclaratorChunkAST extends DeclaratorChunkAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFunctionDeclaratorChunk(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the parameterDeclarationClause of this node
   */
  getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return AST.from<ParameterDeclarationClauseAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the cvQualifierList of this node
   */
  getCvQualifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the ref token in this node
   */
  getRefToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the exceptionSpecifier of this node
   */
  getExceptionSpecifier(): ExceptionSpecifierAST | undefined {
    return AST.from<ExceptionSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the trailingReturnType of this node
   */
  getTrailingReturnType(): TrailingReturnTypeAST | undefined {
    return AST.from<TrailingReturnTypeAST>(
      cxx.getASTSlot(this.getHandle(), 7),
      this.parser,
    );
  }

  /**
   * Returns the isFinal attribute of this node
   */
  getIsFinal(): boolean {
    return cxx.getASTSlot(this.getHandle(), 8) !== 0;
  }

  /**
   * Returns the isOverride attribute of this node
   */
  getIsOverride(): boolean {
    return cxx.getASTSlot(this.getHandle(), 9) !== 0;
  }

  /**
   * Returns the isPure attribute of this node
   */
  getIsPure(): boolean {
    return cxx.getASTSlot(this.getHandle(), 10) !== 0;
  }
}

/**
 * ArrayDeclaratorChunkAST node.
 */
export class ArrayDeclaratorChunkAST extends DeclaratorChunkAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitArrayDeclaratorChunk(this, context);
  }

  /**
   * Returns the location of the lbracket token in this node
   */
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the typeQualifierList of this node
   */
  getTypeQualifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rbracket token in this node
   */
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }
}

/**
 * NameIdAST node.
 */
export class NameIdAST extends UnqualifiedIdAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNameId(this, context);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * DestructorIdAST node.
 */
export class DestructorIdAST extends UnqualifiedIdAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDestructorId(this, context);
  }

  /**
   * Returns the location of the tilde token in this node
   */
  getTildeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the id of this node
   */
  getId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * DecltypeIdAST node.
 */
export class DecltypeIdAST extends UnqualifiedIdAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeId(this, context);
  }

  /**
   * Returns the decltypeSpecifier of this node
   */
  getDecltypeSpecifier(): DecltypeSpecifierAST | undefined {
    return AST.from<DecltypeSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

/**
 * OperatorFunctionIdAST node.
 */
export class OperatorFunctionIdAST extends UnqualifiedIdAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitOperatorFunctionId(this, context);
  }

  /**
   * Returns the location of the operator token in this node
   */
  getOperatorToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the op token in this node
   */
  getOpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the open token in this node
   */
  getOpenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the close token in this node
   */
  getCloseToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the op attribute of this node
   */
  getOp(): TokenKind {
    return cxx.getASTSlot(this.getHandle(), 4);
  }
}

/**
 * LiteralOperatorIdAST node.
 */
export class LiteralOperatorIdAST extends UnqualifiedIdAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLiteralOperatorId(this, context);
  }

  /**
   * Returns the location of the operator token in this node
   */
  getOperatorToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getLiteralValue(slot);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * ConversionFunctionIdAST node.
 */
export class ConversionFunctionIdAST extends UnqualifiedIdAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConversionFunctionId(this, context);
  }

  /**
   * Returns the location of the operator token in this node
   */
  getOperatorToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }
}

/**
 * SimpleTemplateIdAST node.
 */
export class SimpleTemplateIdAST extends UnqualifiedIdAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleTemplateId(this, context);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the less token in this node
   */
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the templateArgumentList of this node
   */
  getTemplateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: TemplateArgumentAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<TemplateArgumentAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the greater token in this node
   */
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * LiteralOperatorTemplateIdAST node.
 */
export class LiteralOperatorTemplateIdAST extends UnqualifiedIdAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLiteralOperatorTemplateId(this, context);
  }

  /**
   * Returns the literalOperatorId of this node
   */
  getLiteralOperatorId(): LiteralOperatorIdAST | undefined {
    return AST.from<LiteralOperatorIdAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the less token in this node
   */
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the templateArgumentList of this node
   */
  getTemplateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: TemplateArgumentAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<TemplateArgumentAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the greater token in this node
   */
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * OperatorFunctionTemplateIdAST node.
 */
export class OperatorFunctionTemplateIdAST extends UnqualifiedIdAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitOperatorFunctionTemplateId(this, context);
  }

  /**
   * Returns the operatorFunctionId of this node
   */
  getOperatorFunctionId(): OperatorFunctionIdAST | undefined {
    return AST.from<OperatorFunctionIdAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the less token in this node
   */
  getLessToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the templateArgumentList of this node
   */
  getTemplateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: TemplateArgumentAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<TemplateArgumentAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the greater token in this node
   */
  getGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * GlobalNestedNameSpecifierAST node.
 */
export class GlobalNestedNameSpecifierAST extends NestedNameSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGlobalNestedNameSpecifier(this, context);
  }

  /**
   * Returns the location of the scope token in this node
   */
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * SimpleNestedNameSpecifierAST node.
 */
export class SimpleNestedNameSpecifierAST extends NestedNameSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleNestedNameSpecifier(this, context);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }

  /**
   * Returns the location of the scope token in this node
   */
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * DecltypeNestedNameSpecifierAST node.
 */
export class DecltypeNestedNameSpecifierAST extends NestedNameSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeNestedNameSpecifier(this, context);
  }

  /**
   * Returns the decltypeSpecifier of this node
   */
  getDecltypeSpecifier(): DecltypeSpecifierAST | undefined {
    return AST.from<DecltypeSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the scope token in this node
   */
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * TemplateNestedNameSpecifierAST node.
 */
export class TemplateNestedNameSpecifierAST extends NestedNameSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplateNestedNameSpecifier(this, context);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the templateId of this node
   */
  getTemplateId(): SimpleTemplateIdAST | undefined {
    return AST.from<SimpleTemplateIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the scope token in this node
   */
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 4) !== 0;
  }
}

/**
 * DefaultFunctionBodyAST node.
 */
export class DefaultFunctionBodyAST extends FunctionBodyAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDefaultFunctionBody(this, context);
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the default token in this node
   */
  getDefaultToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * CompoundStatementFunctionBodyAST node.
 */
export class CompoundStatementFunctionBodyAST extends FunctionBodyAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundStatementFunctionBody(this, context);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the memInitializerList of this node
   */
  getMemInitializerList(): Iterable<MemInitializerAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: MemInitializerAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<MemInitializerAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

/**
 * TryStatementFunctionBodyAST node.
 */
export class TryStatementFunctionBodyAST extends FunctionBodyAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTryStatementFunctionBody(this, context);
  }

  /**
   * Returns the location of the try token in this node
   */
  getTryToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the colon token in this node
   */
  getColonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the memInitializerList of this node
   */
  getMemInitializerList(): Iterable<MemInitializerAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: MemInitializerAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<MemInitializerAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the statement of this node
   */
  getStatement(): CompoundStatementAST | undefined {
    return AST.from<CompoundStatementAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the handlerList of this node
   */
  getHandlerList(): Iterable<HandlerAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: HandlerAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<HandlerAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }
}

/**
 * DeleteFunctionBodyAST node.
 */
export class DeleteFunctionBodyAST extends FunctionBodyAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeleteFunctionBody(this, context);
  }

  /**
   * Returns the location of the equal token in this node
   */
  getEqualToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the delete token in this node
   */
  getDeleteToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * TypeTemplateArgumentAST node.
 */
export class TypeTemplateArgumentAST extends TemplateArgumentAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeTemplateArgument(this, context);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

/**
 * ExpressionTemplateArgumentAST node.
 */
export class ExpressionTemplateArgumentAST extends TemplateArgumentAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExpressionTemplateArgument(this, context);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

/**
 * ThrowExceptionSpecifierAST node.
 */
export class ThrowExceptionSpecifierAST extends ExceptionSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThrowExceptionSpecifier(this, context);
  }

  /**
   * Returns the location of the throw token in this node
   */
  getThrowToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * NoexceptSpecifierAST node.
 */
export class NoexceptSpecifierAST extends ExceptionSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNoexceptSpecifier(this, context);
  }

  /**
   * Returns the location of the noexcept token in this node
   */
  getNoexceptToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * SimpleRequirementAST node.
 */
export class SimpleRequirementAST extends RequirementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleRequirement(this, context);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * CompoundRequirementAST node.
 */
export class CompoundRequirementAST extends RequirementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundRequirement(this, context);
  }

  /**
   * Returns the location of the lbrace token in this node
   */
  getLbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the rbrace token in this node
   */
  getRbraceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the noexcept token in this node
   */
  getNoexceptToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the minusGreater token in this node
   */
  getMinusGreaterToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the typeConstraint of this node
   */
  getTypeConstraint(): TypeConstraintAST | undefined {
    return AST.from<TypeConstraintAST>(
      cxx.getASTSlot(this.getHandle(), 5),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 6), this.parser);
  }
}

/**
 * TypeRequirementAST node.
 */
export class TypeRequirementAST extends RequirementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeRequirement(this, context);
  }

  /**
   * Returns the location of the typename token in this node
   */
  getTypenameToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the template token in this node
   */
  getTemplateToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the isTemplateIntroduced attribute of this node
   */
  getIsTemplateIntroduced(): boolean {
    return cxx.getASTSlot(this.getHandle(), 5) !== 0;
  }
}

/**
 * NestedRequirementAST node.
 */
export class NestedRequirementAST extends RequirementAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedRequirement(this, context);
  }

  /**
   * Returns the location of the requires token in this node
   */
  getRequiresToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the semicolon token in this node
   */
  getSemicolonToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * NewParenInitializerAST node.
 */
export class NewParenInitializerAST extends NewInitializerAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewParenInitializer(this, context);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the expressionList of this node
   */
  getExpressionList(): Iterable<ExpressionAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: ExpressionAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<ExpressionAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }
}

/**
 * NewBracedInitializerAST node.
 */
export class NewBracedInitializerAST extends NewInitializerAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewBracedInitializer(this, context);
  }

  /**
   * Returns the bracedInitList of this node
   */
  getBracedInitList(): BracedInitListAST | undefined {
    return AST.from<BracedInitListAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }
}

/**
 * ParenMemInitializerAST node.
 */
export class ParenMemInitializerAST extends MemInitializerAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParenMemInitializer(this, context);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the expressionList of this node
   */
  getExpressionList(): Iterable<ExpressionAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: ExpressionAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<ExpressionAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

/**
 * BracedMemInitializerAST node.
 */
export class BracedMemInitializerAST extends MemInitializerAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBracedMemInitializer(this, context);
  }

  /**
   * Returns the nestedNameSpecifier of this node
   */
  getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return AST.from<NestedNameSpecifierAST>(
      cxx.getASTSlot(this.getHandle(), 0),
      this.parser,
    );
  }

  /**
   * Returns the unqualifiedId of this node
   */
  getUnqualifiedId(): UnqualifiedIdAST | undefined {
    return AST.from<UnqualifiedIdAST>(
      cxx.getASTSlot(this.getHandle(), 1),
      this.parser,
    );
  }

  /**
   * Returns the bracedInitList of this node
   */
  getBracedInitList(): BracedInitListAST | undefined {
    return AST.from<BracedInitListAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }
}

/**
 * ThisLambdaCaptureAST node.
 */
export class ThisLambdaCaptureAST extends LambdaCaptureAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThisLambdaCapture(this, context);
  }

  /**
   * Returns the location of the this token in this node
   */
  getThisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * DerefThisLambdaCaptureAST node.
 */
export class DerefThisLambdaCaptureAST extends LambdaCaptureAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDerefThisLambdaCapture(this, context);
  }

  /**
   * Returns the location of the star token in this node
   */
  getStarToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the this token in this node
   */
  getThisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }
}

/**
 * SimpleLambdaCaptureAST node.
 */
export class SimpleLambdaCaptureAST extends LambdaCaptureAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleLambdaCapture(this, context);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 2);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * RefLambdaCaptureAST node.
 */
export class RefLambdaCaptureAST extends LambdaCaptureAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRefLambdaCapture(this, context);
  }

  /**
   * Returns the location of the amp token in this node
   */
  getAmpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * RefInitLambdaCaptureAST node.
 */
export class RefInitLambdaCaptureAST extends LambdaCaptureAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRefInitLambdaCapture(this, context);
  }

  /**
   * Returns the location of the amp token in this node
   */
  getAmpToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 3),
      this.parser,
    );
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * InitLambdaCaptureAST node.
 */
export class InitLambdaCaptureAST extends LambdaCaptureAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitInitLambdaCapture(this, context);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the initializer of this node
   */
  getInitializer(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * EllipsisExceptionDeclarationAST node.
 */
export class EllipsisExceptionDeclarationAST extends ExceptionDeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEllipsisExceptionDeclaration(this, context);
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }
}

/**
 * TypeExceptionDeclarationAST node.
 */
export class TypeExceptionDeclarationAST extends ExceptionDeclarationAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeExceptionDeclaration(this, context);
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeSpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeSpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the typeSpecifierList of this node
   */
  getTypeSpecifierList(): Iterable<SpecifierAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: SpecifierAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<SpecifierAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the declarator of this node
   */
  getDeclarator(): DeclaratorAST | undefined {
    return AST.from<DeclaratorAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }
}

/**
 * CxxAttributeAST node.
 */
export class CxxAttributeAST extends AttributeSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCxxAttribute(this, context);
  }

  /**
   * Returns the location of the lbracket token in this node
   */
  getLbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lbracket2 token in this node
   */
  getLbracket2Token(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the attributeUsingPrefix of this node
   */
  getAttributeUsingPrefix(): AttributeUsingPrefixAST | undefined {
    return AST.from<AttributeUsingPrefixAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the attributeList of this node
   */
  getAttributeList(): Iterable<AttributeAST | undefined> {
    let it = cxx.getASTSlot(this.getHandle(), 0);
    let value: AttributeAST | undefined;
    let done = false;
    const p = this.parser;
    function advance() {
      done = it === 0;
      if (done) return;
      const ast = cxx.getListValue(it);
      value = AST.from<AttributeAST>(ast, p);
      it = cxx.getListNext(it);
    }
    function next() {
      advance();
      return { done, value };
    }
    return {
      [Symbol.iterator]() {
        return { next };
      },
    };
  }

  /**
   * Returns the location of the rbracket token in this node
   */
  getRbracketToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the location of the rbracket2 token in this node
   */
  getRbracket2Token(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 5), this.parser);
  }
}

/**
 * GccAttributeAST node.
 */
export class GccAttributeAST extends AttributeSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGccAttribute(this, context);
  }

  /**
   * Returns the location of the attribute token in this node
   */
  getAttributeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the lparen2 token in this node
   */
  getLparen2Token(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the rparen2 token in this node
   */
  getRparen2Token(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }
}

/**
 * AlignasAttributeAST node.
 */
export class AlignasAttributeAST extends AttributeSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignasAttribute(this, context);
  }

  /**
   * Returns the location of the alignas token in this node
   */
  getAlignasToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the expression of this node
   */
  getExpression(): ExpressionAST | undefined {
    return AST.from<ExpressionAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the isPack attribute of this node
   */
  getIsPack(): boolean {
    return cxx.getASTSlot(this.getHandle(), 5) !== 0;
  }
}

/**
 * AlignasTypeAttributeAST node.
 */
export class AlignasTypeAttributeAST extends AttributeSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignasTypeAttribute(this, context);
  }

  /**
   * Returns the location of the alignas token in this node
   */
  getAlignasToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the typeId of this node
   */
  getTypeId(): TypeIdAST | undefined {
    return AST.from<TypeIdAST>(
      cxx.getASTSlot(this.getHandle(), 2),
      this.parser,
    );
  }

  /**
   * Returns the location of the ellipsis token in this node
   */
  getEllipsisToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 4), this.parser);
  }

  /**
   * Returns the isPack attribute of this node
   */
  getIsPack(): boolean {
    return cxx.getASTSlot(this.getHandle(), 5) !== 0;
  }
}

/**
 * AsmAttributeAST node.
 */
export class AsmAttributeAST extends AttributeSpecifierAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmAttribute(this, context);
  }

  /**
   * Returns the location of the asm token in this node
   */
  getAsmToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the lparen token in this node
   */
  getLparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the literal token in this node
   */
  getLiteralToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the location of the rparen token in this node
   */
  getRparenToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 3), this.parser);
  }

  /**
   * Returns the literal attribute of this node
   */
  getLiteral(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getLiteralValue(slot);
  }
}

/**
 * ScopedAttributeTokenAST node.
 */
export class ScopedAttributeTokenAST extends AttributeTokenAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitScopedAttributeToken(this, context);
  }

  /**
   * Returns the location of the attributeNamespace token in this node
   */
  getAttributeNamespaceToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the location of the scope token in this node
   */
  getScopeToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 1), this.parser);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 2), this.parser);
  }

  /**
   * Returns the attributeNamespace attribute of this node
   */
  getAttributeNamespace(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 3);
    return cxx.getIdentifierValue(slot);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 4);
    return cxx.getIdentifierValue(slot);
  }
}

/**
 * SimpleAttributeTokenAST node.
 */
export class SimpleAttributeTokenAST extends AttributeTokenAST {
  /**
   * Traverse this node using the given visitor.
   * @param visitor the visitor.
   * @param context the context.
   * @returns the result of the visit.
   */
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleAttributeToken(this, context);
  }

  /**
   * Returns the location of the identifier token in this node
   */
  getIdentifierToken(): Token | undefined {
    return Token.from(cxx.getASTSlot(this.getHandle(), 0), this.parser);
  }

  /**
   * Returns the identifier attribute of this node
   */
  getIdentifier(): string | undefined {
    const slot = cxx.getASTSlot(this.getHandle(), 1);
    return cxx.getIdentifierValue(slot);
  }
}

const AST_CONSTRUCTORS: Array<
  new (handle: number, kind: ASTKind, parser: TranslationUnitLike) => AST
> = [
  TranslationUnitAST,
  ModuleUnitAST,
  SimpleDeclarationAST,
  AsmDeclarationAST,
  NamespaceAliasDefinitionAST,
  UsingDeclarationAST,
  UsingEnumDeclarationAST,
  UsingDirectiveAST,
  StaticAssertDeclarationAST,
  AliasDeclarationAST,
  OpaqueEnumDeclarationAST,
  FunctionDefinitionAST,
  TemplateDeclarationAST,
  ConceptDefinitionAST,
  DeductionGuideAST,
  ExplicitInstantiationAST,
  ExportDeclarationAST,
  ExportCompoundDeclarationAST,
  LinkageSpecificationAST,
  NamespaceDefinitionAST,
  EmptyDeclarationAST,
  AttributeDeclarationAST,
  ModuleImportDeclarationAST,
  ParameterDeclarationAST,
  AccessDeclarationAST,
  ForRangeDeclarationAST,
  StructuredBindingDeclarationAST,
  AsmOperandAST,
  AsmQualifierAST,
  AsmClobberAST,
  AsmGotoLabelAST,
  LabeledStatementAST,
  CaseStatementAST,
  DefaultStatementAST,
  ExpressionStatementAST,
  CompoundStatementAST,
  IfStatementAST,
  ConstevalIfStatementAST,
  SwitchStatementAST,
  WhileStatementAST,
  DoStatementAST,
  ForRangeStatementAST,
  ForStatementAST,
  BreakStatementAST,
  ContinueStatementAST,
  ReturnStatementAST,
  CoroutineReturnStatementAST,
  GotoStatementAST,
  DeclarationStatementAST,
  TryBlockStatementAST,
  GeneratedLiteralExpressionAST,
  CharLiteralExpressionAST,
  BoolLiteralExpressionAST,
  IntLiteralExpressionAST,
  FloatLiteralExpressionAST,
  NullptrLiteralExpressionAST,
  StringLiteralExpressionAST,
  UserDefinedStringLiteralExpressionAST,
  ObjectLiteralExpressionAST,
  ThisExpressionAST,
  NestedStatementExpressionAST,
  NestedExpressionAST,
  IdExpressionAST,
  LambdaExpressionAST,
  FoldExpressionAST,
  RightFoldExpressionAST,
  LeftFoldExpressionAST,
  RequiresExpressionAST,
  VaArgExpressionAST,
  SubscriptExpressionAST,
  CallExpressionAST,
  TypeConstructionAST,
  BracedTypeConstructionAST,
  SpliceMemberExpressionAST,
  MemberExpressionAST,
  PostIncrExpressionAST,
  CppCastExpressionAST,
  BuiltinBitCastExpressionAST,
  BuiltinOffsetofExpressionAST,
  TypeidExpressionAST,
  TypeidOfTypeExpressionAST,
  SpliceExpressionAST,
  GlobalScopeReflectExpressionAST,
  NamespaceReflectExpressionAST,
  TypeIdReflectExpressionAST,
  ReflectExpressionAST,
  UnaryExpressionAST,
  AwaitExpressionAST,
  SizeofExpressionAST,
  SizeofTypeExpressionAST,
  SizeofPackExpressionAST,
  AlignofTypeExpressionAST,
  AlignofExpressionAST,
  NoexceptExpressionAST,
  NewExpressionAST,
  DeleteExpressionAST,
  CastExpressionAST,
  ImplicitCastExpressionAST,
  BinaryExpressionAST,
  ConditionalExpressionAST,
  YieldExpressionAST,
  ThrowExpressionAST,
  AssignmentExpressionAST,
  PackExpansionExpressionAST,
  DesignatedInitializerClauseAST,
  TypeTraitExpressionAST,
  ConditionExpressionAST,
  EqualInitializerAST,
  BracedInitListAST,
  ParenInitializerAST,
  DotDesignatorAST,
  SubscriptDesignatorAST,
  SplicerAST,
  GlobalModuleFragmentAST,
  PrivateModuleFragmentAST,
  ModuleDeclarationAST,
  ModuleNameAST,
  ModuleQualifierAST,
  ModulePartitionAST,
  ImportNameAST,
  InitDeclaratorAST,
  DeclaratorAST,
  UsingDeclaratorAST,
  EnumeratorAST,
  TypeIdAST,
  HandlerAST,
  BaseSpecifierAST,
  RequiresClauseAST,
  ParameterDeclarationClauseAST,
  TrailingReturnTypeAST,
  LambdaSpecifierAST,
  TypeConstraintAST,
  AttributeArgumentClauseAST,
  AttributeAST,
  AttributeUsingPrefixAST,
  NewPlacementAST,
  NestedNamespaceSpecifierAST,
  TemplateTypeParameterAST,
  NonTypeTemplateParameterAST,
  TypenameTypeParameterAST,
  ConstraintTypeParameterAST,
  GeneratedTypeSpecifierAST,
  TypedefSpecifierAST,
  FriendSpecifierAST,
  ConstevalSpecifierAST,
  ConstinitSpecifierAST,
  ConstexprSpecifierAST,
  InlineSpecifierAST,
  NoreturnSpecifierAST,
  StaticSpecifierAST,
  ExternSpecifierAST,
  RegisterSpecifierAST,
  ThreadLocalSpecifierAST,
  ThreadSpecifierAST,
  MutableSpecifierAST,
  VirtualSpecifierAST,
  ExplicitSpecifierAST,
  AutoTypeSpecifierAST,
  VoidTypeSpecifierAST,
  SizeTypeSpecifierAST,
  SignTypeSpecifierAST,
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
  AtomicQualifierAST,
  RestrictQualifierAST,
  EnumSpecifierAST,
  ClassSpecifierAST,
  TypenameSpecifierAST,
  SplicerTypeSpecifierAST,
  PointerOperatorAST,
  ReferenceOperatorAST,
  PtrToMemberOperatorAST,
  BitfieldDeclaratorAST,
  ParameterPackAST,
  IdDeclaratorAST,
  NestedDeclaratorAST,
  FunctionDeclaratorChunkAST,
  ArrayDeclaratorChunkAST,
  NameIdAST,
  DestructorIdAST,
  DecltypeIdAST,
  OperatorFunctionIdAST,
  LiteralOperatorIdAST,
  ConversionFunctionIdAST,
  SimpleTemplateIdAST,
  LiteralOperatorTemplateIdAST,
  OperatorFunctionTemplateIdAST,
  GlobalNestedNameSpecifierAST,
  SimpleNestedNameSpecifierAST,
  DecltypeNestedNameSpecifierAST,
  TemplateNestedNameSpecifierAST,
  DefaultFunctionBodyAST,
  CompoundStatementFunctionBodyAST,
  TryStatementFunctionBodyAST,
  DeleteFunctionBodyAST,
  TypeTemplateArgumentAST,
  ExpressionTemplateArgumentAST,
  ThrowExceptionSpecifierAST,
  NoexceptSpecifierAST,
  SimpleRequirementAST,
  CompoundRequirementAST,
  TypeRequirementAST,
  NestedRequirementAST,
  NewParenInitializerAST,
  NewBracedInitializerAST,
  ParenMemInitializerAST,
  BracedMemInitializerAST,
  ThisLambdaCaptureAST,
  DerefThisLambdaCaptureAST,
  SimpleLambdaCaptureAST,
  RefLambdaCaptureAST,
  RefInitLambdaCaptureAST,
  InitLambdaCaptureAST,
  EllipsisExceptionDeclarationAST,
  TypeExceptionDeclarationAST,
  CxxAttributeAST,
  GccAttributeAST,
  AlignasAttributeAST,
  AlignasTypeAttributeAST,
  AsmAttributeAST,
  ScopedAttributeTokenAST,
  SimpleAttributeTokenAST,
];
