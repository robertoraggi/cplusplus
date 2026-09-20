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

import {
  type ModelClass,
  type ModelIndex,
  type ModelType,
  substituteType,
} from "./parseModel.ts";
import {
  type ExtraField,
  type FieldClass,
  bindingOf,
  extraFieldsOf,
} from "./semanticClassification.ts";
import { type Wire, WireMapper, normalizeClassName } from "./semanticWire.ts";

export interface FieldPlan {
  owner: string;
  name: string;
  cls: FieldClass;
  cppType: string;
  typeExpression?: string | undefined;
  wire?: Wire | undefined;
  read?: string | undefined;
  write?: string | undefined;
  writeElement?: string | undefined;
  why?: string | undefined;
}

export type EntityMode =
  "factory" | "allocate" | "construct" | "inline" | "base";

export interface EntityPlan {
  mode: EntityMode;
  /** `::cxx::ClassSymbol` */
  name: string;
  /** `ClassSymbol` */
  short: string;
  /** `cxx::ClassSymbol` */
  cpp: string;
  /** The enumerator that discriminates the record, when the entity has one. */
  tag?: string | undefined;
  /** The entity holding the leading fields of this record, when it has one. */
  base?: { name: string; short: string } | undefined;
  fields: FieldPlan[];
  /** Constructor parameters, for entities rebuilt through a factory. */
  factory?: FactoryPlan | undefined;
}

export interface FactoryPlan {
  call: string;
  parameters: { name: string; cppType: string; wire: Wire; read: string }[];
}

export interface CodecPlan {
  constValue: Wire;
  names: EntityPlan[];
  types: EntityPlan[];
  symbols: EntityPlan[];
  symbolBases: EntityPlan[];
  nodes: EntityPlan[];
  nodeBases: EntityPlan[];
  structs: EntityPlan[];
  diagnostics: string[];
  report: FieldPlan[];
}

function capitalize(text: string): string {
  return text.length === 0 ? text : text[0]!.toUpperCase() + text.slice(1);
}

function fieldBaseName(name: string): string {
  return name.endsWith("_") ? name.slice(0, -1) : name;
}

export class PlanBuilder {
  readonly mapper: WireMapper;
  readonly diagnostics: string[] = [];
  readonly report: FieldPlan[] = [];
  readonly index: ModelIndex;
  private readonly bases = new Map<string, EntityPlan>();

  constructor(index: ModelIndex) {
    this.index = index;
    this.mapper = new WireMapper(index);
  }

  build(): CodecPlan {
    for (const name of [
      "::cxx::Attribute",
      "::cxx::Meta",
      "::cxx::InitializerList",
      "::cxx::ConstComplex",
      "::cxx::ConstObject",
      "::cxx::ConstAddress",
      "::cxx::ConstLabelAddress",
    ]) {
      try {
        this.mapper.requireStruct(name);
      } catch (error) {
        this.diagnostics.push((error as Error).message);
      }
    }

    const names = this.domain(
      "::cxx::NameKind",
      (kind) => kind.slice(1),
      "factory",
    );
    const types = this.domain(
      "::cxx::TypeKind",
      (kind) => `${kind.slice(1)}Type`,
      "factory",
    );
    const symbolBases: EntityPlan[] = [];
    const symbols = this.domain(
      "::cxx::SymbolKind",
      (kind) => `${kind.slice(1)}Symbol`,
      "allocate",
      symbolBases,
    );
    const nodeBases: EntityPlan[] = [];
    const nodes = this.domain(
      "::cxx::ASTKind",
      (kind) => `${kind}AST`,
      "construct",
      nodeBases,
    );

    const structs: EntityPlan[] = [];
    const seen = new Set<string>();
    for (;;) {
      const pending = [...this.mapper.structs.keys()].filter(
        (name) => !seen.has(name),
      );
      if (pending.length === 0) break;
      for (const name of pending) {
        seen.add(name);
        const entry = this.index.classOf(name);
        if (!entry) {
          this.diagnostics.push(`struct '${name}' is not in the model`);
          continue;
        }
        structs.push(this.planEntity(entry, undefined, "inline", structs));
      }
    }

    const constValue = this.mapper.constValue;
    if (!constValue)
      this.diagnostics.push(
        "::cxx::ConstValue is not reachable from the model",
      );

    return {
      constValue: constValue ?? { k: "const-value", alternatives: [] },
      names,
      types,
      symbols,
      symbolBases,
      nodes,
      nodeBases,
      structs,
      diagnostics: this.diagnostics,
      report: this.report,
    };
  }

  private domain(
    enumName: string,
    classOfKind: (kind: string) => string,
    mode: EntityMode,
    bases: EntityPlan[] = [],
  ): EntityPlan[] {
    const kinds = this.index.enumOf(enumName);
    if (!kinds) {
      this.diagnostics.push(`enum '${enumName}' is not in the model`);
      return [];
    }

    const plans: EntityPlan[] = [];

    for (const enumerator of kinds.enumerators) {
      const short = classOfKind(enumerator.name);
      const entry = this.index.classOf(`::cxx::${short}`);
      if (!entry) {
        this.diagnostics.push(
          `${enumName}::${enumerator.name} has no class '::cxx::${short}'`,
        );
        continue;
      }
      const tag = `${normalizeClassName(enumName)}::${enumerator.name}`;
      plans.push(this.planEntity(entry, tag, mode, bases));
    }

    return plans;
  }

  private planEntity(
    entry: ModelClass,
    tag: string | undefined,
    mode: EntityMode,
    bases: EntityPlan[],
  ): EntityPlan {
    const plan: EntityPlan = {
      mode,
      name: entry.name,
      short: entry.unqualifiedName,
      cpp: normalizeClassName(entry.name),
      tag,
      fields: [],
    };

    if (mode === "factory") {
      plan.factory = this.planFactory(entry, true);
      for (const { owner } of this.index.layoutOf(entry)) {
        for (const field of owner.fields) {
          const binding = bindingOf(owner.name, field.name);
          this.report.push({
            owner: owner.name,
            name: field.name,
            cls: binding?.cls ?? "P",
            cppType: "",
            why: binding?.why ?? "carried by the interning factory arguments",
          });
        }
      }
      return plan;
    }

    const base = this.planBase(entry, bases);
    let layout = this.index.layoutOf(entry);

    if (base) {
      try {
        layout = this.index.ownLayoutOf(entry, this.index.classOf(base.name));
        plan.base = { name: base.name, short: base.short };
      } catch (error) {
        this.diagnostics.push((error as Error).message);
      }
    }

    for (const { owner, substitution } of layout) {
      for (const field of owner.fields) {
        const fieldPlan = this.planField(entry, owner, field, substitution);
        if (!fieldPlan) continue;
        this.report.push(fieldPlan);
        if (fieldPlan.cls === "P") plan.fields.push(fieldPlan);
      }
      for (const extra of extraFieldsOf(owner.name)) {
        const fieldPlan = this.planExtraField(
          entry,
          owner,
          extra,
          substitution,
        );
        if (!fieldPlan) continue;
        this.report.push(fieldPlan);
        if (fieldPlan.cls === "P") plan.fields.push(fieldPlan);
      }
    }

    if (mode === "allocate") plan.factory = this.planFactory(entry, false);

    return plan;
  }

  private planBase(
    entry: ModelClass,
    bases: EntityPlan[],
  ): EntityPlan | undefined {
    const base = this.index.primaryBaseOf(
      entry,
      (candidate) => !candidate.isTemplate,
    );
    if (!base) return undefined;

    const planned = this.bases.get(base.name);
    if (planned) return planned;

    const plan = this.planEntity(base, undefined, "base", bases);
    this.bases.set(base.name, plan);
    bases.push(plan);
    return plan;
  }

  /**
   * A field whose state is reachable only through the public API, declared in
   * the binding table and typed by the accessor it reads.
   */
  private planExtraField(
    entity: ModelClass,
    owner: ModelClass,
    extra: ExtraField,
    substitution: ModelType[],
  ): FieldPlan | undefined {
    const context = `${owner.name}::${extra.name}`;

    const accessor = this.accessorNamed(entity, extra.from);
    if (!accessor) {
      this.diagnostics.push(`${context}: no accessor '${extra.from}()'`);
      return undefined;
    }

    let wire: Wire;
    let cppType: string;
    try {
      const returnType = substituteType(accessor.returnType, substitution);
      wire = this.mapper.wireOf(returnType, context);
      cppType = this.mapper.cppTypeOf(returnType);
    } catch (error) {
      this.diagnostics.push((error as Error).message);
      return undefined;
    }

    if (wire.k === "vector" || wire.k === "deque") cppType = wire.cpp;

    return {
      owner: owner.name,
      name: extra.name,
      cls: extra.cls,
      cppType,
      wire,
      read: `$->${extra.from}()`,
      write: extra.write,
      writeElement: extra.writeElement,
    };
  }

  private accessorNamed(entity: ModelClass, name: string) {
    for (const { owner } of this.index.layoutOf(entity)) {
      for (const method of owner.methods) {
        if (method.name !== name) continue;
        if (method.parameters.length !== 0) continue;
        return method;
      }
    }
    return undefined;
  }

  private planField(
    entity: ModelClass,
    owner: ModelClass,
    field: {
      name: string;
      type: ModelType;
      isBitField: boolean;
      access: string;
    },
    substitution: ModelType[],
  ): FieldPlan | undefined {
    const context = `${owner.name}::${field.name}`;

    const binding =
      bindingOf(owner.name, field.name) ?? bindingOf(entity.name, field.name);

    const cls: FieldClass = binding?.cls ?? "P";

    if (cls !== "P") {
      return {
        owner: owner.name,
        name: field.name,
        cls,
        cppType: "",
        why: binding?.why,
      };
    }

    let type: ModelType;
    let typeExpression: string | undefined;

    if (binding?.from) {
      const accessor = this.accessorNamed(entity, binding.from);
      if (!accessor) {
        this.diagnostics.push(`${context}: no accessor '${binding.from}()'`);
        return undefined;
      }
      type = accessor.returnType;
      typeExpression = `$->${binding.from}()`;
    } else {
      try {
        type = substituteType(field.type, substitution);
      } catch (error) {
        this.diagnostics.push(`${context}: ${(error as Error).message}`);
        return undefined;
      }
    }

    let wire: Wire;
    let cppType: string;
    try {
      wire = this.mapper.wireOf(type, context);
      cppType = this.mapper.cppTypeOf(type);
    } catch (error) {
      this.diagnostics.push((error as Error).message);
      return undefined;
    }

    const isPublicData = field.access === "public" && !field.isBitField;

    if (!typeExpression && isPublicData) typeExpression = `$->${field.name}`;

    const read =
      binding?.read ??
      (isPublicData
        ? `$->${field.name}`
        : this.deriveRead(entity, field.name, context));
    if (!read) return undefined;

    const write = binding?.write;
    const writeElement = binding?.writeElement;

    let derivedWrite: string | undefined = write;
    if (!write && !writeElement) {
      derivedWrite = isPublicData
        ? `$->${field.name} = $value`
        : this.deriveWrite(entity, field.name, wire, context);
      if (!derivedWrite) return undefined;
    }

    return {
      owner: owner.name,
      name: field.name,
      cls,
      cppType,
      typeExpression,
      wire,
      read,
      write: derivedWrite,
      writeElement,
    };
  }

  private methodsOf(entity: ModelClass) {
    const result = new Map<
      string,
      { name: string; isConst: boolean; parameters: { name: string }[] }[]
    >();
    for (const { owner } of this.index.layoutOf(entity)) {
      for (const method of owner.methods) {
        const bucket = result.get(method.name) ?? [];
        bucket.push(method);
        result.set(method.name, bucket);
      }
    }
    return result;
  }

  private deriveRead(
    entity: ModelClass,
    fieldName: string,
    context: string,
  ): string | undefined {
    const base = fieldBaseName(fieldName);
    const methods = this.methodsOf(entity);
    const candidates = methods.get(base) ?? [];
    if (candidates.some((method) => method.parameters.length === 0))
      return `$->${base}()`;

    this.diagnostics.push(
      `${context}: no reader; add a '${base}()' accessor or a binding`,
    );
    return undefined;
  }

  private deriveWrite(
    entity: ModelClass,
    fieldName: string,
    wire: Wire,
    context: string,
  ): string | undefined {
    const base = fieldBaseName(fieldName);
    const methods = this.methodsOf(entity);

    const unary = (name: string) =>
      (methods.get(name) ?? []).some(
        (method) => method.parameters.length === 1 && !method.isConst,
      );

    const setters = [
      `set${capitalize(base)}`,
      `set${capitalize(base.replace(/^is/, ""))}`,
      `set${capitalize(base.replace(/^has/, ""))}`,
    ];

    for (const setter of setters)
      if (unary(setter)) return `$->${setter}($value)`;

    const isContainer =
      wire.k === "vector" || wire.k === "deque" || wire.k === "ast-list";

    this.diagnostics.push(
      isContainer
        ? `${context}: no writer; add a 'writeElement' binding or a '${setters[0]}' setter`
        : `${context}: no writer; add a '${setters[0]}' setter or a binding`,
    );
    return undefined;
  }

  /**
   * Names, types and symbols are rebuilt through `Control`, never constructed
   * directly, so their record carries the factory's arguments (7.6).
   */
  private planFactory(
    entry: ModelClass,
    argumentsAreIdentity: boolean,
  ): FactoryPlan | undefined {
    const control = this.index.classOf("::cxx::Control");
    if (!control) return undefined;

    const short = entry.unqualifiedName;

    const candidates = [`get${short}`, `new${short}`];
    const method = control.methods.find((candidate) =>
      candidates.includes(candidate.name),
    );
    if (!method) return undefined;

    const accessors = new Map<string, string[]>();
    for (const { owner } of this.index.layoutOf(entry)) {
      for (const accessor of owner.methods) {
        if (accessor.parameters.length !== 0) continue;
        if (!accessor.isConst) continue;
        const bucket = accessors.get(accessor.name) ?? [];
        bucket.push(accessor.returnTypeName);
        accessors.set(accessor.name, bucket);
      }
    }

    const claimed = new Set<string>();
    const parameters: FactoryPlan["parameters"] = [];

    for (const parameter of method.parameters) {
      if (parameter.name === "enclosingScope") {
        parameters.push({
          name: parameter.name,
          cppType: "cxx::ScopeSymbol*",
          wire: { k: "symbol", cpp: "cxx::ScopeSymbol" },
          read: "nullptr",
        });
        continue;
      }
      if (parameter.name === "sourceLocation") {
        parameters.push({
          name: parameter.name,
          cppType: "cxx::SourceLocation",
          wire: { k: "location" },
          read: "{}",
        });
        continue;
      }
      if (parameter.name === "unit") {
        parameters.push({
          name: parameter.name,
          cppType: "cxx::TranslationUnit*",
          wire: { k: "unit" },
          read: "$->translationUnit()",
        });
        continue;
      }

      const reader = this.matchAccessor(parameter, accessors, claimed);
      if (!reader) {
        if (!argumentsAreIdentity) {
          parameters.push({
            name: parameter.name,
            cppType: this.mapper.cppTypeOf(parameter.type),
            wire: { k: "unit" },
            read: "{}",
          });
          continue;
        }
        this.diagnostics.push(
          `${entry.name}: no accessor for factory argument '${parameter.name}'`,
        );
        return undefined;
      }

      let wire: Wire;
      try {
        wire = this.mapper.wireOf(
          parameter.type,
          `${entry.name}(${parameter.name})`,
        );
      } catch (error) {
        this.diagnostics.push((error as Error).message);
        return undefined;
      }

      parameters.push({
        name: parameter.name,
        cppType: this.mapper.cppTypeOf(parameter.type),
        wire,
        read: `$->${reader}()`,
      });
    }

    return { call: method.name, parameters };
  }

  /** Matches by name, then uniquely by type among the unclaimed accessors. */
  private matchAccessor(
    parameter: { name: string; typeName: string },
    accessors: Map<string, string[]>,
    claimed: Set<string>,
  ): string | undefined {
    if (accessors.has(parameter.name) && !claimed.has(parameter.name)) {
      claimed.add(parameter.name);
      return parameter.name;
    }

    const wanted = stripCvRef(parameter.typeName);
    const matches: string[] = [];

    for (const [name, returnTypeNames] of accessors) {
      if (claimed.has(name)) continue;
      if (returnTypeNames.some((text) => stripCvRef(text) === wanted))
        matches.push(name);
    }

    if (matches.length !== 1) return undefined;

    claimed.add(matches[0]!);
    return matches[0];
  }
}

function stripCvRef(text: string): string {
  return text
    .replace(/^const\s+/, "")
    .replace(/&+$/, "")
    .trim();
}
