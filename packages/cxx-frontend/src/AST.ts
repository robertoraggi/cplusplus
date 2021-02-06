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

function __get_ast_kind(handle: number): any {
    throw new Error("Function not implemented.");
}

function __get_ast_slot(handle: number, slot: number): any {
    throw new Error("Function not implemented.");
}

function __get_list_value(handle: number): any {
    throw new Error("Function not implemented.");
}

function __get_list_next(handle: number): number {
    throw new Error("Function not implemented.");
}

function wrapNode<T extends AST>(handle: number): T | undefined {
    if (handle) {
        const kind = __get_ast_kind(handle);
        const ast = new AST_CONSTRUCTORS[kind](handle) as T;
        return ast;
    }
    return;
}

export abstract class AST {
    constructor(readonly handle: number) { }
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
    *getTypeSpecifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
    getDeclarator(): DeclaratorAST | undefined {
        return wrapNode<DeclaratorAST>(__get_ast_slot(this.handle, 1));
    }
}

export class NestedNameSpecifierAST extends AST {
    *getNameList(): Generator<NameAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<NameAST>(__get_list_value(it));
        }
    }
}

export class UsingDeclaratorAST extends AST {
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return wrapNode<NestedNameSpecifierAST>(__get_ast_slot(this.handle, 1));
    }
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 2));
    }
}

export class HandlerAST extends AST {
    getExceptionDeclaration(): ExceptionDeclarationAST | undefined {
        return wrapNode<ExceptionDeclarationAST>(__get_ast_slot(this.handle, 2));
    }
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 4));
    }
}

export class TemplateArgumentAST extends AST {
}

export class EnumBaseAST extends AST {
    *getTypeSpecifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
}

export class EnumeratorAST extends AST {
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 0));
    }
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 3));
    }
}

export class DeclaratorAST extends AST {
    *getPtrOpList(): Generator<PtrOperatorAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<PtrOperatorAST>(__get_list_value(it));
        }
    }
    getCoreDeclarator(): CoreDeclaratorAST | undefined {
        return wrapNode<CoreDeclaratorAST>(__get_ast_slot(this.handle, 1));
    }
    *getModifiers(): Generator<DeclaratorModifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<DeclaratorModifierAST>(__get_list_value(it));
        }
    }
}

export class BaseSpecifierAST extends AST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 1));
    }
}

export class BaseClauseAST extends AST {
    *getBaseSpecifierList(): Generator<BaseSpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<BaseSpecifierAST>(__get_list_value(it));
        }
    }
}

export class NewTypeIdAST extends AST {
    *getTypeSpecifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
}

export class ParameterDeclarationClauseAST extends AST {
    *getTemplateParameterList(): Generator<ParameterDeclarationAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<ParameterDeclarationAST>(__get_list_value(it));
        }
    }
}

export class ParametersAndQualifiersAST extends AST {
    getParameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
        return wrapNode<ParameterDeclarationClauseAST>(__get_ast_slot(this.handle, 1));
    }
    *getCvQualifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
}

export class EqualInitializerAST extends InitializerAST {
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 1));
    }
}

export class BracedInitListAST extends InitializerAST {
    *getExpressionList(): Generator<ExpressionAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<ExpressionAST>(__get_list_value(it));
        }
    }
}

export class ParenInitializerAST extends InitializerAST {
    *getExpressionList(): Generator<ExpressionAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<ExpressionAST>(__get_list_value(it));
        }
    }
}

export class NewParenInitializerAST extends NewInitializerAST {
    *getExpressionList(): Generator<ExpressionAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<ExpressionAST>(__get_list_value(it));
        }
    }
}

export class NewBracedInitializerAST extends NewInitializerAST {
    getBracedInit(): BracedInitListAST | undefined {
        return wrapNode<BracedInitListAST>(__get_ast_slot(this.handle, 0));
    }
}

export class EllipsisExceptionDeclarationAST extends ExceptionDeclarationAST {
}

export class TypeExceptionDeclarationAST extends ExceptionDeclarationAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    *getTypeSpecifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
    getDeclarator(): DeclaratorAST | undefined {
        return wrapNode<DeclaratorAST>(__get_ast_slot(this.handle, 2));
    }
}

export class TranslationUnitAST extends UnitAST {
    *getDeclarationList(): Generator<DeclarationAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<DeclarationAST>(__get_list_value(it));
        }
    }
}

export class ModuleUnitAST extends UnitAST {
}

export class ThisExpressionAST extends ExpressionAST {
}

export class CharLiteralExpressionAST extends ExpressionAST {
}

export class BoolLiteralExpressionAST extends ExpressionAST {
}

export class IntLiteralExpressionAST extends ExpressionAST {
}

export class FloatLiteralExpressionAST extends ExpressionAST {
}

export class NullptrLiteralExpressionAST extends ExpressionAST {
}

export class StringLiteralExpressionAST extends ExpressionAST {
}

export class UserDefinedStringLiteralExpressionAST extends ExpressionAST {
}

export class IdExpressionAST extends ExpressionAST {
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 0));
    }
}

export class NestedExpressionAST extends ExpressionAST {
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 1));
    }
}

export class BinaryExpressionAST extends ExpressionAST {
    getLeftExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 0));
    }
    getRightExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 2));
    }
}

export class AssignmentExpressionAST extends ExpressionAST {
    getLeftExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 0));
    }
    getRightExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 2));
    }
}

export class CallExpressionAST extends ExpressionAST {
    getBaseExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 0));
    }
    *getExpressionList(): Generator<ExpressionAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<ExpressionAST>(__get_list_value(it));
        }
    }
}

export class SubscriptExpressionAST extends ExpressionAST {
    getBaseExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 0));
    }
    getIndexExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 2));
    }
}

export class MemberExpressionAST extends ExpressionAST {
    getBaseExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 0));
    }
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 3));
    }
}

export class ConditionalExpressionAST extends ExpressionAST {
    getCondition(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 0));
    }
    getIftrueExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 2));
    }
    getIffalseExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 4));
    }
}

export class CppCastExpressionAST extends ExpressionAST {
    getTypeId(): TypeIdAST | undefined {
        return wrapNode<TypeIdAST>(__get_ast_slot(this.handle, 2));
    }
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 5));
    }
}

export class NewExpressionAST extends ExpressionAST {
    getTypeId(): NewTypeIdAST | undefined {
        return wrapNode<NewTypeIdAST>(__get_ast_slot(this.handle, 2));
    }
    getNewInitalizer(): NewInitializerAST | undefined {
        return wrapNode<NewInitializerAST>(__get_ast_slot(this.handle, 3));
    }
}

export class LabeledStatementAST extends StatementAST {
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 2));
    }
}

export class CaseStatementAST extends StatementAST {
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 1));
    }
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 3));
    }
}

export class DefaultStatementAST extends StatementAST {
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 2));
    }
}

export class ExpressionStatementAST extends StatementAST {
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 0));
    }
}

export class CompoundStatementAST extends StatementAST {
    *getStatementList(): Generator<StatementAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<StatementAST>(__get_list_value(it));
        }
    }
}

export class IfStatementAST extends StatementAST {
    getInitializer(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 3));
    }
    getCondition(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 4));
    }
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 6));
    }
    getElseStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 7));
    }
}

export class SwitchStatementAST extends StatementAST {
    getInitializer(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 2));
    }
    getCondition(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 3));
    }
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 5));
    }
}

export class WhileStatementAST extends StatementAST {
    getCondition(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 2));
    }
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 4));
    }
}

export class DoStatementAST extends StatementAST {
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 1));
    }
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 4));
    }
}

export class ForRangeStatementAST extends StatementAST {
    getInitializer(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 2));
    }
    getRangeDeclaration(): DeclarationAST | undefined {
        return wrapNode<DeclarationAST>(__get_ast_slot(this.handle, 3));
    }
    getRangeInitializer(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 5));
    }
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 7));
    }
}

export class ForStatementAST extends StatementAST {
    getInitializer(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 2));
    }
    getCondition(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 3));
    }
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 5));
    }
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 7));
    }
}

export class BreakStatementAST extends StatementAST {
}

export class ContinueStatementAST extends StatementAST {
}

export class ReturnStatementAST extends StatementAST {
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 1));
    }
}

export class GotoStatementAST extends StatementAST {
}

export class CoroutineReturnStatementAST extends StatementAST {
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 1));
    }
}

export class DeclarationStatementAST extends StatementAST {
    getDeclaration(): DeclarationAST | undefined {
        return wrapNode<DeclarationAST>(__get_ast_slot(this.handle, 0));
    }
}

export class TryBlockStatementAST extends StatementAST {
    getStatement(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 1));
    }
    *getHandlerList(): Generator<HandlerAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<HandlerAST>(__get_list_value(it));
        }
    }
}

export class FunctionDefinitionAST extends DeclarationAST {
    *getDeclSpecifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
    getDeclarator(): DeclaratorAST | undefined {
        return wrapNode<DeclaratorAST>(__get_ast_slot(this.handle, 1));
    }
    getFunctionBody(): StatementAST | undefined {
        return wrapNode<StatementAST>(__get_ast_slot(this.handle, 2));
    }
}

export class ConceptDefinitionAST extends DeclarationAST {
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 1));
    }
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 3));
    }
}

export class ForRangeDeclarationAST extends DeclarationAST {
}

export class AliasDeclarationAST extends DeclarationAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    getTypeId(): TypeIdAST | undefined {
        return wrapNode<TypeIdAST>(__get_ast_slot(this.handle, 4));
    }
}

export class SimpleDeclarationAST extends DeclarationAST {
    *getAttributes(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    *getDeclSpecifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
    *getDeclaratorList(): Generator<DeclaratorAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<DeclaratorAST>(__get_list_value(it));
        }
    }
}

export class StaticAssertDeclarationAST extends DeclarationAST {
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 2));
    }
}

export class EmptyDeclarationAST extends DeclarationAST {
}

export class AttributeDeclarationAST extends DeclarationAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
}

export class OpaqueEnumDeclarationAST extends DeclarationAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return wrapNode<NestedNameSpecifierAST>(__get_ast_slot(this.handle, 3));
    }
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 4));
    }
    getEnumBase(): EnumBaseAST | undefined {
        return wrapNode<EnumBaseAST>(__get_ast_slot(this.handle, 5));
    }
}

export class UsingEnumDeclarationAST extends DeclarationAST {
}

export class NamespaceDefinitionAST extends DeclarationAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return wrapNode<NestedNameSpecifierAST>(__get_ast_slot(this.handle, 3));
    }
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 4));
    }
    *getExtraAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    *getDeclarationList(): Generator<DeclarationAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<DeclarationAST>(__get_list_value(it));
        }
    }
}

export class NamespaceAliasDefinitionAST extends DeclarationAST {
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return wrapNode<NestedNameSpecifierAST>(__get_ast_slot(this.handle, 3));
    }
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 4));
    }
}

export class UsingDirectiveAST extends DeclarationAST {
}

export class UsingDeclarationAST extends DeclarationAST {
    *getUsingDeclaratorList(): Generator<UsingDeclaratorAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<UsingDeclaratorAST>(__get_list_value(it));
        }
    }
}

export class AsmDeclarationAST extends DeclarationAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
}

export class ExportDeclarationAST extends DeclarationAST {
}

export class ModuleImportDeclarationAST extends DeclarationAST {
}

export class TemplateDeclarationAST extends DeclarationAST {
    *getTemplateParameterList(): Generator<DeclarationAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<DeclarationAST>(__get_list_value(it));
        }
    }
    getDeclaration(): DeclarationAST | undefined {
        return wrapNode<DeclarationAST>(__get_ast_slot(this.handle, 4));
    }
}

export class DeductionGuideAST extends DeclarationAST {
}

export class ExplicitInstantiationAST extends DeclarationAST {
    getDeclaration(): DeclarationAST | undefined {
        return wrapNode<DeclarationAST>(__get_ast_slot(this.handle, 2));
    }
}

export class ParameterDeclarationAST extends DeclarationAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    *getTypeSpecifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
    getDeclarator(): DeclaratorAST | undefined {
        return wrapNode<DeclaratorAST>(__get_ast_slot(this.handle, 2));
    }
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 4));
    }
}

export class LinkageSpecificationAST extends DeclarationAST {
    *getDeclarationList(): Generator<DeclarationAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<DeclarationAST>(__get_list_value(it));
        }
    }
}

export class SimpleNameAST extends NameAST {
}

export class DestructorNameAST extends NameAST {
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 1));
    }
}

export class DecltypeNameAST extends NameAST {
    getDecltypeSpecifier(): SpecifierAST | undefined {
        return wrapNode<SpecifierAST>(__get_ast_slot(this.handle, 0));
    }
}

export class OperatorNameAST extends NameAST {
}

export class TemplateNameAST extends NameAST {
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 0));
    }
    *getTemplateArgumentList(): Generator<TemplateArgumentAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<TemplateArgumentAST>(__get_list_value(it));
        }
    }
}

export class QualifiedNameAST extends NameAST {
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return wrapNode<NestedNameSpecifierAST>(__get_ast_slot(this.handle, 0));
    }
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 2));
    }
}

export class SimpleSpecifierAST extends SpecifierAST {
}

export class ExplicitSpecifierAST extends SpecifierAST {
}

export class NamedTypeSpecifierAST extends SpecifierAST {
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 0));
    }
}

export class PlaceholderTypeSpecifierHelperAST extends SpecifierAST {
}

export class DecltypeSpecifierTypeSpecifierAST extends SpecifierAST {
}

export class UnderlyingTypeSpecifierAST extends SpecifierAST {
}

export class AtomicTypeSpecifierAST extends SpecifierAST {
}

export class ElaboratedTypeSpecifierAST extends SpecifierAST {
}

export class DecltypeSpecifierAST extends SpecifierAST {
}

export class PlaceholderTypeSpecifierAST extends SpecifierAST {
}

export class CvQualifierAST extends SpecifierAST {
}

export class EnumSpecifierAST extends SpecifierAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return wrapNode<NestedNameSpecifierAST>(__get_ast_slot(this.handle, 3));
    }
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 4));
    }
    getEnumBase(): EnumBaseAST | undefined {
        return wrapNode<EnumBaseAST>(__get_ast_slot(this.handle, 5));
    }
    *getEnumeratorList(): Generator<EnumeratorAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<EnumeratorAST>(__get_list_value(it));
        }
    }
}

export class ClassSpecifierAST extends SpecifierAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 2));
    }
    getBaseClause(): BaseClauseAST | undefined {
        return wrapNode<BaseClauseAST>(__get_ast_slot(this.handle, 3));
    }
    *getDeclarationList(): Generator<DeclarationAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<DeclarationAST>(__get_list_value(it));
        }
    }
}

export class TypenameSpecifierAST extends SpecifierAST {
}

export class IdDeclaratorAST extends CoreDeclaratorAST {
    getName(): NameAST | undefined {
        return wrapNode<NameAST>(__get_ast_slot(this.handle, 1));
    }
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
}

export class NestedDeclaratorAST extends CoreDeclaratorAST {
    getDeclarator(): DeclaratorAST | undefined {
        return wrapNode<DeclaratorAST>(__get_ast_slot(this.handle, 1));
    }
}

export class PointerOperatorAST extends PtrOperatorAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    *getCvQualifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
}

export class ReferenceOperatorAST extends PtrOperatorAST {
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
}

export class PtrToMemberOperatorAST extends PtrOperatorAST {
    getNestedNameSpecifier(): NestedNameSpecifierAST | undefined {
        return wrapNode<NestedNameSpecifierAST>(__get_ast_slot(this.handle, 0));
    }
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
        }
    }
    *getCvQualifierList(): Generator<SpecifierAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<SpecifierAST>(__get_list_value(it));
        }
    }
}

export class FunctionDeclaratorAST extends DeclaratorModifierAST {
    getParametersAndQualifiers(): ParametersAndQualifiersAST | undefined {
        return wrapNode<ParametersAndQualifiersAST>(__get_ast_slot(this.handle, 0));
    }
}

export class ArrayDeclaratorAST extends DeclaratorModifierAST {
    getExpression(): ExpressionAST | undefined {
        return wrapNode<ExpressionAST>(__get_ast_slot(this.handle, 1));
    }
    *getAttributeList(): Generator<AttributeAST> | undefined {
        for (let it = __get_ast_slot(this.handle, 0); it; it = __get_list_next(it)) {
            yield wrapNode<AttributeAST>(__get_list_value(it));
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
