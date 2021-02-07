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

import { cxx } from "./cxx";
import { ASTVisitor } from "./ASTVisitor";

export abstract class AST {
    constructor(private handle: number) { }

    getHandle() {
        return this.handle;
    }

    abstract accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result;

    static from<T extends AST = AST>(handle: number): T | undefined {
        if (handle) {
            const kind = cxx.getASTKind(handle);
            const ast = new AST_CONSTRUCTORS[kind](handle) as T;
            return ast;
        }
        return;
    }
}

export abstract class AttributeAST extends AST { }
export abstract class CoreDeclaratorAST extends AST { }
export abstract class DeclarationAST extends AST { }
export abstract class DeclaratorModifierAST extends AST { }
export abstract class ExceptionDeclarationAST extends AST { }
export abstract class ExpressionAST extends AST { }
export abstract class InitializerAST extends AST { }
export abstract class NameAST extends AST { }
export abstract class NewInitializerAST extends AST { }
export abstract class PtrOperatorAST extends AST { }
export abstract class SpecifierAST extends AST { }
export abstract class StatementAST extends AST { }
export abstract class UnitAST extends AST { }

export class TypeIdAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitTypeId(this, context);
    }
    *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
    getDeclarator(): DeclaratorAST | undefined {
        return AST.from<DeclaratorAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
}

export class NestedNameSpecifierAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNestedNameSpecifier(this, context);
    }
    *getNameList(): Generator<NameAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<NameAST>(cxx.getListValue(it));
        }
    }
}

export class UsingDeclaratorAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitUsingDeclarator(this, context);
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return AST.from<NestedNameSpecifierAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class HandlerAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitHandler(this, context);
    }
    getExceptionDeclaration(): ExceptionDeclarationAST | undefined {
        return AST.from<ExceptionDeclarationAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
}

export class TemplateArgumentAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitTemplateArgument(this, context);
    }
}

export class EnumBaseAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitEnumBase(this, context);
    }
    *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
}

export class EnumeratorAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitEnumerator(this, context);
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
}

export class DeclaratorAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitDeclarator(this, context);
    }
    *getPtrOpList(): Generator<PtrOperatorAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<PtrOperatorAST>(cxx.getListValue(it));
        }
    }
    getCoreDeclarator(): CoreDeclaratorAST | undefined {
        return AST.from<CoreDeclaratorAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    *getModifiers(): Generator<DeclaratorModifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<DeclaratorModifierAST>(cxx.getListValue(it));
        }
    }
}

export class BaseSpecifierAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitBaseSpecifier(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
}

export class BaseClauseAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitBaseClause(this, context);
    }
    *getBaseSpecifierList(): Generator<BaseSpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<BaseSpecifierAST>(cxx.getListValue(it));
        }
    }
}

export class NewTypeIdAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNewTypeId(this, context);
    }
    *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
}

export class ParameterDeclarationClauseAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitParameterDeclarationClause(this, context);
    }
    *getTemplateParameterList(): Generator<ParameterDeclarationAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<ParameterDeclarationAST>(cxx.getListValue(it));
        }
    }
}

export class ParametersAndQualifiersAST extends AST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitParametersAndQualifiers(this, context);
    }
    getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
        return AST.from<ParameterDeclarationClauseAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    *getCvQualifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 3); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 5); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
}

export class EqualInitializerAST extends InitializerAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitEqualInitializer(this, context);
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
}

export class BracedInitListAST extends InitializerAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitBracedInitList(this, context);
    }
    *getExpressionList(): Generator<ExpressionAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<ExpressionAST>(cxx.getListValue(it));
        }
    }
}

export class ParenInitializerAST extends InitializerAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitParenInitializer(this, context);
    }
    *getExpressionList(): Generator<ExpressionAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<ExpressionAST>(cxx.getListValue(it));
        }
    }
}

export class NewParenInitializerAST extends NewInitializerAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNewParenInitializer(this, context);
    }
    *getExpressionList(): Generator<ExpressionAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<ExpressionAST>(cxx.getListValue(it));
        }
    }
}

export class NewBracedInitializerAST extends NewInitializerAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNewBracedInitializer(this, context);
    }
    getBracedInit(): BracedInitListAST | undefined {
        return AST.from<BracedInitListAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
}

export class EllipsisExceptionDeclarationAST extends ExceptionDeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitEllipsisExceptionDeclaration(this, context);
    }
}

export class TypeExceptionDeclarationAST extends ExceptionDeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitTypeExceptionDeclaration(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
    getDeclarator(): DeclaratorAST | undefined {
        return AST.from<DeclaratorAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class TranslationUnitAST extends UnitAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitTranslationUnit(this, context);
    }
    *getDeclarationList(): Generator<DeclarationAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<DeclarationAST>(cxx.getListValue(it));
        }
    }
}

export class ModuleUnitAST extends UnitAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitModuleUnit(this, context);
    }
}

export class ThisExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitThisExpression(this, context);
    }
}

export class CharLiteralExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitCharLiteralExpression(this, context);
    }
}

export class BoolLiteralExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitBoolLiteralExpression(this, context);
    }
}

export class IntLiteralExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitIntLiteralExpression(this, context);
    }
}

export class FloatLiteralExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitFloatLiteralExpression(this, context);
    }
}

export class NullptrLiteralExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNullptrLiteralExpression(this, context);
    }
}

export class StringLiteralExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitStringLiteralExpression(this, context);
    }
}

export class UserDefinedStringLiteralExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitUserDefinedStringLiteralExpression(this, context);
    }
}

export class IdExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitIdExpression(this, context);
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
}

export class NestedExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNestedExpression(this, context);
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
}

export class BinaryExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitBinaryExpression(this, context);
    }
    getLeftExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    getRightExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class AssignmentExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitAssignmentExpression(this, context);
    }
    getLeftExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    getRightExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class CallExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitCallExpression(this, context);
    }
    getBaseExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    *getExpressionList(): Generator<ExpressionAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<ExpressionAST>(cxx.getListValue(it));
        }
    }
}

export class SubscriptExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitSubscriptExpression(this, context);
    }
    getBaseExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    getIndexExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class MemberExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitMemberExpression(this, context);
    }
    getBaseExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
}

export class ConditionalExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitConditionalExpression(this, context);
    }
    getCondition(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    getIftrueExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getIffalseExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
}

export class CppCastExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitCppCastExpression(this, context);
    }
    getTypeId(): TypeIdAST | undefined {
        return AST.from<TypeIdAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 5));
    }
}

export class NewExpressionAST extends ExpressionAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNewExpression(this, context);
    }
    getTypeId(): NewTypeIdAST | undefined {
        return AST.from<NewTypeIdAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getNewInitalizer(): NewInitializerAST | undefined {
        return AST.from<NewInitializerAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
}

export class LabeledStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitLabeledStatement(this, context);
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class CaseStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitCaseStatement(this, context);
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
}

export class DefaultStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitDefaultStatement(this, context);
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class ExpressionStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitExpressionStatement(this, context);
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
}

export class CompoundStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitCompoundStatement(this, context);
    }
    *getStatementList(): Generator<StatementAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<StatementAST>(cxx.getListValue(it));
        }
    }
}

export class IfStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitIfStatement(this, context);
    }
    getInitializer(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
    getCondition(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 6));
    }
    getElseStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 7));
    }
}

export class SwitchStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitSwitchStatement(this, context);
    }
    getInitializer(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getCondition(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 5));
    }
}

export class WhileStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitWhileStatement(this, context);
    }
    getCondition(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
}

export class DoStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitDoStatement(this, context);
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
}

export class ForRangeStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitForRangeStatement(this, context);
    }
    getInitializer(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getRangeDeclaration(): DeclarationAST | undefined {
        return AST.from<DeclarationAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
    getRangeInitializer(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 5));
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 7));
    }
}

export class ForStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitForStatement(this, context);
    }
    getInitializer(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getCondition(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 5));
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 7));
    }
}

export class BreakStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitBreakStatement(this, context);
    }
}

export class ContinueStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitContinueStatement(this, context);
    }
}

export class ReturnStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitReturnStatement(this, context);
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
}

export class GotoStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitGotoStatement(this, context);
    }
}

export class CoroutineReturnStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitCoroutineReturnStatement(this, context);
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
}

export class DeclarationStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitDeclarationStatement(this, context);
    }
    getDeclaration(): DeclarationAST | undefined {
        return AST.from<DeclarationAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
}

export class TryBlockStatementAST extends StatementAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitTryBlockStatement(this, context);
    }
    getStatement(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    *getHandlerList(): Generator<HandlerAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<HandlerAST>(cxx.getListValue(it));
        }
    }
}

export class FunctionDefinitionAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitFunctionDefinition(this, context);
    }
    *getDeclSpecifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
    getDeclarator(): DeclaratorAST | undefined {
        return AST.from<DeclaratorAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    getFunctionBody(): StatementAST | undefined {
        return AST.from<StatementAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class ConceptDefinitionAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitConceptDefinition(this, context);
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
}

export class ForRangeDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitForRangeDeclaration(this, context);
    }
}

export class AliasDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitAliasDeclaration(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    getTypeId(): TypeIdAST | undefined {
        return AST.from<TypeIdAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
}

export class SimpleDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitSimpleDeclaration(this, context);
    }
    *getAttributes(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    *getDeclSpecifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
    *getDeclaratorList(): Generator<DeclaratorAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<DeclaratorAST>(cxx.getListValue(it));
        }
    }
}

export class StaticAssertDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitStaticAssertDeclaration(this, context);
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class EmptyDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitEmptyDeclaration(this, context);
    }
}

export class AttributeDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitAttributeDeclaration(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
}

export class OpaqueEnumDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitOpaqueEnumDeclaration(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return AST.from<NestedNameSpecifierAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
    getEnumBase(): EnumBaseAST | undefined {
        return AST.from<EnumBaseAST>(cxx.getASTSlot(this.getHandle(), 5));
    }
}

export class UsingEnumDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitUsingEnumDeclaration(this, context);
    }
}

export class NamespaceDefinitionAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNamespaceDefinition(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return AST.from<NestedNameSpecifierAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
    *getExtraAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 5); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    *getDeclarationList(): Generator<DeclarationAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 7); it; it = cxx.getListNext(it)) {
            yield AST.from<DeclarationAST>(cxx.getListValue(it));
        }
    }
}

export class NamespaceAliasDefinitionAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNamespaceAliasDefinition(this, context);
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return AST.from<NestedNameSpecifierAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
}

export class UsingDirectiveAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitUsingDirective(this, context);
    }
}

export class UsingDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitUsingDeclaration(this, context);
    }
    *getUsingDeclaratorList(): Generator<UsingDeclaratorAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<UsingDeclaratorAST>(cxx.getListValue(it));
        }
    }
}

export class AsmDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitAsmDeclaration(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
}

export class ExportDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitExportDeclaration(this, context);
    }
}

export class ModuleImportDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitModuleImportDeclaration(this, context);
    }
}

export class TemplateDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitTemplateDeclaration(this, context);
    }
    *getTemplateParameterList(): Generator<DeclarationAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<DeclarationAST>(cxx.getListValue(it));
        }
    }
    getDeclaration(): DeclarationAST | undefined {
        return AST.from<DeclarationAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
}

export class DeductionGuideAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitDeductionGuide(this, context);
    }
}

export class ExplicitInstantiationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitExplicitInstantiation(this, context);
    }
    getDeclaration(): DeclarationAST | undefined {
        return AST.from<DeclarationAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class ParameterDeclarationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitParameterDeclaration(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 0); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    *getTypeSpecifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
    getDeclarator(): DeclaratorAST | undefined {
        return AST.from<DeclaratorAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
}

export class LinkageSpecificationAST extends DeclarationAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitLinkageSpecification(this, context);
    }
    *getDeclarationList(): Generator<DeclarationAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 3); it; it = cxx.getListNext(it)) {
            yield AST.from<DeclarationAST>(cxx.getListValue(it));
        }
    }
}

export class SimpleNameAST extends NameAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitSimpleName(this, context);
    }
}

export class DestructorNameAST extends NameAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitDestructorName(this, context);
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
}

export class DecltypeNameAST extends NameAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitDecltypeName(this, context);
    }
    getDecltypeSpecifier(): SpecifierAST | undefined {
        return AST.from<SpecifierAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
}

export class OperatorNameAST extends NameAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitOperatorName(this, context);
    }
}

export class TemplateNameAST extends NameAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitTemplateName(this, context);
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    *getTemplateArgumentList(): Generator<TemplateArgumentAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<TemplateArgumentAST>(cxx.getListValue(it));
        }
    }
}

export class QualifiedNameAST extends NameAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitQualifiedName(this, context);
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return AST.from<NestedNameSpecifierAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
}

export class SimpleSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitSimpleSpecifier(this, context);
    }
}

export class ExplicitSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitExplicitSpecifier(this, context);
    }
}

export class NamedTypeSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNamedTypeSpecifier(this, context);
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
}

export class PlaceholderTypeSpecifierHelperAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitPlaceholderTypeSpecifierHelper(this, context);
    }
}

export class DecltypeSpecifierTypeSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitDecltypeSpecifierTypeSpecifier(this, context);
    }
}

export class UnderlyingTypeSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitUnderlyingTypeSpecifier(this, context);
    }
}

export class AtomicTypeSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitAtomicTypeSpecifier(this, context);
    }
}

export class ElaboratedTypeSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitElaboratedTypeSpecifier(this, context);
    }
}

export class DecltypeSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitDecltypeSpecifier(this, context);
    }
}

export class PlaceholderTypeSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitPlaceholderTypeSpecifier(this, context);
    }
}

export class CvQualifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitCvQualifier(this, context);
    }
}

export class EnumSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitEnumSpecifier(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return AST.from<NestedNameSpecifierAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 4));
    }
    getEnumBase(): EnumBaseAST | undefined {
        return AST.from<EnumBaseAST>(cxx.getASTSlot(this.getHandle(), 5));
    }
    *getEnumeratorList(): Generator<EnumeratorAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 8); it; it = cxx.getListNext(it)) {
            yield AST.from<EnumeratorAST>(cxx.getListValue(it));
        }
    }
}

export class ClassSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitClassSpecifier(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 2));
    }
    getBaseClause(): BaseClauseAST | undefined {
        return AST.from<BaseClauseAST>(cxx.getASTSlot(this.getHandle(), 3));
    }
    *getDeclarationList(): Generator<DeclarationAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 5); it; it = cxx.getListNext(it)) {
            yield AST.from<DeclarationAST>(cxx.getListValue(it));
        }
    }
}

export class TypenameSpecifierAST extends SpecifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitTypenameSpecifier(this, context);
    }
}

export class IdDeclaratorAST extends CoreDeclaratorAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitIdDeclarator(this, context);
    }
    getName(): NameAST | undefined {
        return AST.from<NameAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
}

export class NestedDeclaratorAST extends CoreDeclaratorAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitNestedDeclarator(this, context);
    }
    getDeclarator(): DeclaratorAST | undefined {
        return AST.from<DeclaratorAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
}

export class PointerOperatorAST extends PtrOperatorAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitPointerOperator(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    *getCvQualifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
}

export class ReferenceOperatorAST extends PtrOperatorAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitReferenceOperator(this, context);
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 1); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
}

export class PtrToMemberOperatorAST extends PtrOperatorAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitPtrToMemberOperator(this, context);
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return AST.from<NestedNameSpecifierAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 2); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
    *getCvQualifierList(): Generator<SpecifierAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 3); it; it = cxx.getListNext(it)) {
            yield AST.from<SpecifierAST>(cxx.getListValue(it));
        }
    }
}

export class FunctionDeclaratorAST extends DeclaratorModifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitFunctionDeclarator(this, context);
    }
    getParametersAndQualifiers(): ParametersAndQualifiersAST | undefined {
        return AST.from<ParametersAndQualifiersAST>(cxx.getASTSlot(this.getHandle(), 0));
    }
}

export class ArrayDeclaratorAST extends DeclaratorModifierAST {
    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {
        return visitor.visitArrayDeclarator(this, context);
    }
    getExpression(): ExpressionAST | undefined {
        return AST.from<ExpressionAST>(cxx.getASTSlot(this.getHandle(), 1));
    }
    *getAttributeList(): Generator<AttributeAST | undefined> {
        for (let it = cxx.getASTSlot(this.getHandle(), 3); it; it = cxx.getListNext(it)) {
            yield AST.from<AttributeAST>(cxx.getListValue(it));
        }
    }
}

const AST_CONSTRUCTORS: Array<new (handle: number) => AST> = [
    TypeIdAST,
    NestedNameSpecifierAST,
    UsingDeclaratorAST,
    HandlerAST,
    TemplateArgumentAST,
    EnumBaseAST,
    EnumeratorAST,
    DeclaratorAST,
    BaseSpecifierAST,
    BaseClauseAST,
    NewTypeIdAST,
    ParameterDeclarationClauseAST,
    ParametersAndQualifiersAST,
    EqualInitializerAST,
    BracedInitListAST,
    ParenInitializerAST,
    NewParenInitializerAST,
    NewBracedInitializerAST,
    EllipsisExceptionDeclarationAST,
    TypeExceptionDeclarationAST,
    TranslationUnitAST,
    ModuleUnitAST,
    ThisExpressionAST,
    CharLiteralExpressionAST,
    BoolLiteralExpressionAST,
    IntLiteralExpressionAST,
    FloatLiteralExpressionAST,
    NullptrLiteralExpressionAST,
    StringLiteralExpressionAST,
    UserDefinedStringLiteralExpressionAST,
    IdExpressionAST,
    NestedExpressionAST,
    BinaryExpressionAST,
    AssignmentExpressionAST,
    CallExpressionAST,
    SubscriptExpressionAST,
    MemberExpressionAST,
    ConditionalExpressionAST,
    CppCastExpressionAST,
    NewExpressionAST,
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
    FunctionDefinitionAST,
    ConceptDefinitionAST,
    ForRangeDeclarationAST,
    AliasDeclarationAST,
    SimpleDeclarationAST,
    StaticAssertDeclarationAST,
    EmptyDeclarationAST,
    AttributeDeclarationAST,
    OpaqueEnumDeclarationAST,
    UsingEnumDeclarationAST,
    NamespaceDefinitionAST,
    NamespaceAliasDefinitionAST,
    UsingDirectiveAST,
    UsingDeclarationAST,
    AsmDeclarationAST,
    ExportDeclarationAST,
    ModuleImportDeclarationAST,
    TemplateDeclarationAST,
    DeductionGuideAST,
    ExplicitInstantiationAST,
    ParameterDeclarationAST,
    LinkageSpecificationAST,
    SimpleNameAST,
    DestructorNameAST,
    DecltypeNameAST,
    OperatorNameAST,
    TemplateNameAST,
    QualifiedNameAST,
    SimpleSpecifierAST,
    ExplicitSpecifierAST,
    NamedTypeSpecifierAST,
    PlaceholderTypeSpecifierHelperAST,
    DecltypeSpecifierTypeSpecifierAST,
    UnderlyingTypeSpecifierAST,
    AtomicTypeSpecifierAST,
    ElaboratedTypeSpecifierAST,
    DecltypeSpecifierAST,
    PlaceholderTypeSpecifierAST,
    CvQualifierAST,
    EnumSpecifierAST,
    ClassSpecifierAST,
    TypenameSpecifierAST,
    IdDeclaratorAST,
    NestedDeclaratorAST,
    PointerOperatorAST,
    ReferenceOperatorAST,
    PtrToMemberOperatorAST,
    FunctionDeclaratorAST,
    ArrayDeclaratorAST,
];
