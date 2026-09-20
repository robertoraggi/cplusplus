// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

import { Parser, Token } from "cxx-frontend";
import * as S from "cxx-frontend/model";
import { traverse } from "cxx-frontend/traverse";
import type {
  Model,
  ModelAccess,
  ModelAlias,
  ModelClass,
  ModelField,
  ModelMethod,
  ModelParameter,
  ModelType,
  ModelTemplateArgument,
  Span,
  Location,
} from "./parseModel.ts";

const MODELLED_DIRECTORIES = ["src/parser/cxx/", "src/codegen/cxx/"];

function isModelled(location: Location | undefined): boolean {
  if (!location) return false;
  return MODELLED_DIRECTORIES.some((directory) =>
    location.file.includes(directory),
  );
}

function accessOf(
  declaration: S.DeclarationAST | undefined,
  current: ModelAccess,
): ModelAccess {
  if (!(declaration instanceof S.AccessDeclarationAST)) return current;
  if (declaration.accessSpecifier === "public") return "public";
  if (declaration.accessSpecifier === "protected") return "protected";
  if (declaration.accessSpecifier === "private") return "private";
  return current;
}

function nameOf(symbol: S.Symbol | undefined): string {
  if (!symbol) return "";
  return symbol.text;
}

function qualifiedName(symbol: S.Symbol | undefined): string {
  if (!symbol) return "";
  const parts: string[] = [];
  let scope: S.Symbol | undefined = symbol.parent;
  while (scope) {
    if (!scope.parent) break;
    const skip = scope instanceof S.NamespaceSymbol && scope.isInline;
    if (!skip && !scope.isTemplateParameters && !scope.isFunctionParameters) {
      const name = nameOf(scope);
      if (name) parts.push(name);
    }
    scope = scope.parent;
  }
  parts.reverse();
  parts.push(nameOf(symbol));
  return "::" + parts.join("::");
}

function templateName(symbol: S.Symbol | undefined): string {
  if (symbol instanceof S.ClassSymbol)
    return qualifiedName(symbol.templatePattern ?? symbol);
  return qualifiedName(symbol);
}

function safeNumber(value: bigint): number {
  const number = Number(value);
  if (!Number.isSafeInteger(number))
    throw new Error(`model integer is outside the JSON safe range: ${value}`);
  return number;
}

type ElementOf<T> = T extends Iterable<infer U> ? U : never;
type Argument = ElementOf<S.ClassSymbol["expandedTemplateArguments"]>;
function argumentModel(argument: Argument): ModelTemplateArgument {
  if (argument instanceof S.Type)
    return { kind: "type", text: argument.text, type: typeModel(argument) };
  if (argument instanceof S.Symbol) {
    if (argument.isType)
      return {
        kind: "type",
        text: argument.text,
        type: typeModel(argument.type),
      };
    return {
      kind: "symbol",
      text: argument.text,
      name: qualifiedName(argument),
    };
  }
  if (argument instanceof S.AST) return { kind: "expression", text: "" };
  if (typeof argument === "bigint")
    return {
      kind: "value",
      text: String(argument),
      value: safeNumber(argument),
    };
  return { kind: "value", text: "" };
}

export function typeModel(type: S.Type | undefined): ModelType {
  if (!type) return { kind: "none" };
  if (type instanceof S.QualType)
    return {
      kind: "qual",
      isConst: type.isConst,
      isVolatile: type.isVolatile,
      element: typeModel(type.elementType),
    };
  if (type instanceof S.PointerType)
    return { kind: "pointer", element: typeModel(type.elementType) };
  if (type instanceof S.LvalueReferenceType)
    return { kind: "lvalue-reference", element: typeModel(type.elementType) };
  if (type instanceof S.RvalueReferenceType)
    return { kind: "rvalue-reference", element: typeModel(type.elementType) };
  if (type instanceof S.BoundedArrayType)
    return {
      kind: "array",
      size: type.size,
      element: typeModel(type.elementType),
    };
  if (type instanceof S.UnboundedArrayType)
    return { kind: "array", element: typeModel(type.elementType) };
  if (type instanceof S.ClassType) {
    const symbol = type.symbol!;
    return {
      kind: "class",
      name: templateName(symbol),
      isPolymorphic: symbol.isPolymorphic,
      arguments: (() => {
        const texts = [...symbol.expandedTemplateArgumentTexts];
        return [...symbol.expandedTemplateArguments].map((argument, index) => ({
          ...argumentModel(argument),
          text: texts[index]!,
        }));
      })(),
    };
  }
  if (type instanceof S.EnumType || type instanceof S.ScopedEnumType)
    return {
      kind: "enum",
      name: qualifiedName(type.symbol),
      isScoped: type instanceof S.ScopedEnumType,
    };
  if (
    type instanceof S.TypeParameterType ||
    type instanceof S.TemplateTypeParameterType
  )
    return {
      kind: "type-param",
      index: type.index,
      depth: type.depth,
      isPack: type.isParameterPack,
    };
  if (type instanceof S.FunctionType)
    return {
      kind: "function",
      returnType: typeModel(type.returnType),
      parameterTypes: [...type.parameterTypes].map(typeModel),
    };
  if (type instanceof S.MemberObjectPointerType)
    return {
      kind: "member-object-pointer",
      classType: typeModel(type.classType),
      element: typeModel(type.elementType),
    };
  if (type instanceof S.MemberFunctionPointerType)
    return {
      kind: "member-function-pointer",
      classType: typeModel(type.classType),
      functionType: typeModel(type.functionType),
    };
  return { kind: "builtin", name: type.text };
}

export function dumpModel(parser: Parser): Model {
  const model: Model = { enums: [], classes: [], aliases: [] };
  function location(loc: number): Location | undefined {
    if (!loc) return;
    const pos = new Token(loc, parser).getLocation();
    return { file: pos.fileName, line: pos.startLine, column: pos.startColumn };
  }
  function span(first: number, last: number): Span | undefined {
    if (!first || !last) return;
    const start = new Token(first, parser).getLocation();
    const end = new Token(last, parser).getLocation();
    return {
      file: start.fileName,
      startLine: start.startLine,
      startColumn: start.startColumn,
      endLine: end.endLine,
      endColumn: end.endColumn,
    };
  }
  function parameters(fn: S.FunctionSymbol): ModelParameter[] {
    return [...fn.parameters].map((parameter) => ({
      name: nameOf(parameter),
      typeName: parameter?.type?.text ?? "",
      type: typeModel(parameter?.type),
    }));
  }
  function methods(
    declarations: Iterable<S.DeclarationAST | undefined>,
  ): ModelMethod[] {
    const result: ModelMethod[] = [];
    let access: ModelAccess = "public";
    for (const declaration of declarations) {
      if (declaration instanceof S.AccessDeclarationAST) {
        access = accessOf(declaration, access);
        continue;
      }
      let fn: S.Symbol | undefined;
      if (declaration instanceof S.FunctionDefinitionAST)
        fn = declaration.symbol;
      if (declaration instanceof S.SimpleDeclarationAST)
        fn = [...declaration.initDeclaratorList].find(
          (d) => d?.symbol instanceof S.FunctionSymbol,
        )?.symbol;
      if (!(fn instanceof S.FunctionSymbol)) continue;
      if (fn.isConstructor || fn.isDestructor || fn.isDeleted) continue;
      if (!(fn.name instanceof S.Identifier)) continue;
      const type = fn.type;
      if (!(type instanceof S.FunctionType)) continue;
      result.push({
        name: fn.name.name,
        access,
        isStatic: fn.isStatic,
        isConst: ["Const", "ConstVolatile"].includes(type.cvQualifiers),
        isVirtual: fn.isVirtual,
        isPure: fn.isPure,
        returnTypeName: type.returnType?.text ?? "",
        returnType: typeModel(type.returnType),
        parameters: parameters(fn),
      });
    }
    return result;
  }
  function fields(
    declarations: Iterable<S.DeclarationAST | undefined>,
    anonymousPath: string[] = [],
  ): ModelField[] {
    const result: ModelField[] = [];
    let access: ModelAccess = "public";
    for (const declaration of declarations) {
      if (declaration instanceof S.AccessDeclarationAST) {
        access = accessOf(declaration, access);
        continue;
      }
      if (!(declaration instanceof S.SimpleDeclarationAST)) continue;
      const initDeclarators = [...declaration.initDeclaratorList];
      if (!initDeclarators.length) {
        const anonymous = [...declaration.declSpecifierList].find((s) => {
          if (!(s instanceof S.ClassSpecifierAST)) return false;
          if (!s.symbol) return false;
          const name = s.symbol.name;
          return !(name instanceof S.Identifier) || name.isAnonymous;
        });
        if (anonymous instanceof S.ClassSpecifierAST) {
          let kind = "struct";
          if (anonymous.symbol?.isUnion) kind = "union";
          result.push(
            ...fields(anonymous.declarationList, [...anonymousPath, kind]),
          );
          continue;
        }
      }
      for (const init of initDeclarators) {
        const field = init?.symbol;
        if (!(field instanceof S.FieldSymbol) || field.isStatic) continue;
        const entry: ModelField = {
          name: nameOf(field),
          access,
          isBitField: field.isBitField,
          isMutable: field.isMutable,
          anonymousPath,
          typeName: field.type?.text ?? "",
          type: typeModel(field.type),
        };
        const loc = location(field.location);
        if (loc) entry.location = loc;
        const decl = span(
          declaration.firstSourceLocation,
          declaration.lastSourceLocation,
        );
        if (decl) entry.declaration = decl;
        if (init?.initializer) {
          const value = span(
            init.initializer.firstSourceLocation,
            init.initializer.lastSourceLocation,
          );
          if (value) entry.initializer = value;
        }
        result.push(entry);
      }
    }
    return result;
  }
  traverse(parser.model.ast, {
    ClassSpecifier({ node }) {
      if (!node.symbol?.name) return;
      const loc = location(node.classLoc);
      if (isModelled(loc)) {
        const symbol = node.symbol;
        const entry: ModelClass = {
          name: qualifiedName(symbol),
          unqualifiedName: nameOf(symbol),
          isUnion: symbol.isUnion,
          isFinal: symbol.isFinal,
          isTemplate: symbol.isTemplatePattern,
          location: loc,
          bases: [...node.baseSpecifierList]
            .filter((b) => b !== undefined)
            .map((base) => {
              const symbol = base.symbol?.symbol;
              if (symbol)
                return {
                  type: templateName(symbol),
                  isVirtual: base.isVirtual,
                  typeNode: typeModel(symbol.type),
                };
              return {
                type: base.symbol?.name?.text ?? "",
                isVirtual: base.isVirtual,
              };
            }),
          fields: fields(node.declarationList),
          constructors: [...symbol.declaredConstructors]
            .filter(
              (fn): fn is S.FunctionSymbol => fn !== undefined && !fn.isDeleted,
            )
            .map((fn) => ({
              isExplicit: fn.isExplicit,
              parameters: parameters(fn),
            })),
          methods: methods(node.declarationList),
        };
        const body = span(node.lbraceLoc, node.rbraceLoc);
        if (body) entry.body = body;
        model.classes.push(entry);
      }
    },

    AliasDeclaration({ node }) {
      if (!node.symbol) return;
      const symbol = node.symbol;
      const loc = location(node.usingLoc);
      if (
        isModelled(loc) &&
        symbol.parent instanceof S.NamespaceSymbol &&
        !symbol.isTemplatePattern
      ) {
        const entry: ModelAlias = {
          name: qualifiedName(symbol),
          unqualifiedName: nameOf(symbol),
          typeName: symbol.type?.text ?? "",
          type: typeModel(symbol.type),
        };
        if (loc) entry.location = loc;
        model.aliases.push(entry);
      }
    },

    EnumSpecifier({ node }) {
      if (!node.symbol?.name) return;
      const loc = location(node.enumLoc);
      if (isModelled(loc)) {
        const symbol = node.symbol;
        let underlyingType = "int";
        if (
          symbol instanceof S.EnumSymbol ||
          symbol instanceof S.ScopedEnumSymbol
        )
          underlyingType = symbol.underlyingType?.text ?? "int";
        model.enums.push({
          name: qualifiedName(symbol),
          isScoped: symbol.isScopedEnum,
          location: loc,
          underlyingType,
          enumerators: [...node.enumeratorList]
            .filter((e) => e !== undefined)
            .map((e) => {
              const value = e.symbol?.value;
              if (typeof value === "bigint")
                return {
                  name: e.identifier?.name ?? "",
                  value: safeNumber(value),
                };
              return { name: e.identifier?.name ?? "" };
            }),
        });
      }
    },
  });

  return model;
}
