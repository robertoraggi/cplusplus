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

/**
 * The persistence classification of section 9 of docs/design/symbol-storage.md.
 *
 * P — persisted in the artifact
 * R — rebuilt on load; never written
 * C — a compatibility key, validated rather than restored
 * E — required to be at its boundary value, verified at freeze time
 * D — derived from another persisted field of the same entity
 * F — flushed before freezing
 */
export type FieldClass = "P" | "R" | "C" | "E" | "D" | "F";

export interface FieldBinding {
  cls: FieldClass;
  /**
   * Types the field by this nullary accessor rather than by its declaration,
   * for a field the archive reaches through a different representation.
   */
  from?: string;
  /** Expression producing the field's value; `$` is the entity. */
  read?: string;
  /** Statement storing a decoded value; `$` is the entity, `$value` the value. */
  write?: string;
  /** Statement storing one element of a container; `$element` is the element. */
  writeElement?: string;
  /** Fields written together with this one by a single coupled writer. */
  writesWith?: string[];
  why?: string;
}

export type ClassBindings = Record<string, FieldBinding>;

/**
 * Only the entities whose wire form does not follow from the accessor
 * convention appear here. Everything else is persisted through
 * `foo()` / `setFoo(value)`, which the generator derives from the model, and a
 * field with neither a convention match nor an entry below fails the build.
 */
export const bindings: Record<string, ClassBindings> = {
  "::cxx::Symbol": {
    kind_: { cls: "D", why: "the symbol record's own kind tag" },
    internalId_: {
      cls: "E",
      why: "scratch space for a numbering walk; zero outside one",
    },
    abiTags_: {
      cls: "P",
      read: "$->abiTagList()",
      write: "$->setAbiTags($value)",
      why: "abiTags() answers a span over the interned list",
    },
    link_: {
      cls: "R",
      why: "the same-name chain is rebuilt by ScopeSymbol::rebuildLookupTable",
    },
  },

  "::cxx::NamespaceSymbol": {
    anonNamespaceIndex_: {
      cls: "P",
      read: "$->anonNamespaceIndex().value_or(-1)",
      write: "if ($value >= 0) $->setAnonNamespaceIndex($value)",
      why: "the accessor answers nullopt for a named namespace",
    },
  },

  "::cxx::ConstObject": {
    members_: {
      cls: "P",
      read: "$->members()",
      writeElement: "$->addMember($element.symbol, $element.value)",
    },
  },

  "::cxx::ConstAddress": {
    string_: {
      cls: "P",
      read: "$->stringLiteral()",
      write: "$->setStringLiteral($value)",
    },
  },

  "::cxx::ScopeSymbol": {
    members_: {
      cls: "P",
      read: "$->members()",
      writeElement: "$->addMember($element)",
    },
    buckets_: {
      cls: "R",
      why: "a derived lookup index, rebuilt from members_",
    },
    usingDirectives_: {
      cls: "P",
      read: "$->usingDirectives()",
      writeElement: "$->addUsingDirective($element)",
    },
  },

  "::cxx::ClassSymbol": {
    flags_: { cls: "D", why: "the individual bit-fields are persisted" },
    baseClasses_: {
      cls: "P",
      read: "$->baseClasses()",
      writeElement: "$->addBaseClass($element)",
    },
    befriendingClasses_: {
      cls: "P",
      read: "$->befriendingClasses()",
      writeElement: "$->addBefriendingClass($element)",
    },
    templateFriendships_: {
      cls: "P",
      read: "$->templateFriendships()",
      writeElement:
        "$->addBefriendingClass($element.befriendingClass, $element.arguments)",
    },
    deductionGuides_: {
      cls: "P",
      read: "$->deductionGuides()",
      writeElement: "$->addDeductionGuide($element)",
    },
    instantiationSubstitutionDepth_: {
      cls: "P",
      read: "$->instantiationSubstitutionDepth()",
      write:
        "$->setInstantiationSubstitution($value, $->instantiationSubstitutionArguments())",
    },
    instantiationSubstitutionArguments_: {
      cls: "P",
      read: "$->instantiationSubstitutionArguments()",
      write:
        "$->setInstantiationSubstitution($->instantiationSubstitutionDepth(), std::move($value))",
    },
    layout_: {
      cls: "P",
      read: "$->layout()",
      write: "$->setLayout(std::move($value))",
    },
    vtableLayout_: {
      cls: "P",
      read: "$->vtableLayout()",
      write: "$->setVTableLayout(std::move($value))",
    },
  },

  "::cxx::FunctionSymbol": {
    flags_: { cls: "D", why: "the individual bit-fields are persisted" },
    hasCLinkage_: {
      cls: "P",
      read: "$->hasCLinkage()",
      write:
        "$->setLanguageLinkage($value ? LanguageKind::kC : LanguageKind::kCXX)",
    },
    overriddenFunctions_: {
      cls: "P",
      read: "$->overriddenFunctions()",
      writeElement: "$->addOverriddenFunction($element)",
    },
    befriendingClasses_: {
      cls: "P",
      read: "$->befriendingClasses()",
      writeElement: "$->addBefriendingClass($element)",
    },
    templateFriendships_: {
      cls: "P",
      read: "$->templateFriendships()",
      writeElement:
        "$->addBefriendingClass($element.befriendingClass, $element.arguments)",
    },
    pendingBody_: {
      cls: "P",
      read: "$->pendingBody()",
      write: "$->setPendingBody(std::move($value))",
    },
    pendingExceptionSpecification_: {
      cls: "P",
      read: "$->pendingExceptionSpecification()",
      write: "$->setPendingExceptionSpecification(std::move($value))",
    },
  },

  "::cxx::OverloadSetSymbol": {
    declaredFunctions_: {
      cls: "P",
      read: "$->declaredFunctions()",
      writeElement: "$->addFunction($element)",
    },
    usingDeclarations_: {
      cls: "P",
      read: "$->usingDeclarations()",
      writeElement: "$->addUsingDeclaration($element)",
    },
  },

  "::cxx::LambdaSymbol": {
    flags_: { cls: "D", why: "the individual bit-fields are persisted" },
  },

  "::cxx::VariableSymbol": {
    flags_: { cls: "D", why: "the individual bit-fields are persisted" },
  },

  "::cxx::FieldSymbol": {
    flags_: { cls: "D", why: "the individual bit-fields are persisted" },
    pendingInitializer_: {
      cls: "P",
      read: "$->pendingInitializer()",
      write: "$->setPendingInitializer(std::move($value))",
    },
  },

  "::cxx::ParameterPackSymbol": {
    elements_: {
      cls: "P",
      read: "$->elements()",
      writeElement: "$->addElement($element)",
    },
  },

  "::cxx::EnumeratorSymbol": {
    value_: { cls: "P", read: "$->value()", write: "$->setValue($value)" },
  },

  "::cxx::MaybeRedecl": {
    canonical_: {
      cls: "P",
      read: "$->canonicalOrNull()",
      write: "$->setCanonical($value)",
      why: "canonical() answers `this` when the field is null",
    },
    redeclarations_: {
      cls: "P",
      read: "$->redeclarations()",
      writeElement: "$->addRedeclaration($element)",
    },
  },

  "::cxx::MaybeTemplate": {
    template_: {
      cls: "D",
      why: "persisted through the template payload's public accessors",
    },
    declaration_: {
      cls: "P",
      read: "$->declaration()",
      write: "$->setDeclaration($value)",
    },
  },

  "::cxx::Name": {
    kind_: { cls: "D", why: "the name record's own kind tag" },
    hashValue_: { cls: "D", why: "recomputed when the name is interned" },
  },

  "::cxx::Identifier": {
    info_: {
      cls: "R",
      why: "a builtin-identifier cache, re-resolved from the spelling",
    },
  },

  "::cxx::Type": {
    kind_: { cls: "D", why: "the type record's own kind tag" },
  },

  "::cxx::AST": {
    kind_: { cls: "D", why: "the AST record's own kind tag" },
    internalId_: {
      cls: "E",
      why: "scratch space for a numbering walk; zero outside one",
    },
  },

  "::cxx::ClassLayout": {
    fields_: {
      cls: "P",
      from: "sortedFieldInfos",
      read: "$->sortedFieldInfos()",
      writeElement: "$->setFieldInfo($element.first, $element.second)",
    },
    bases_: {
      cls: "P",
      from: "sortedBaseInfos",
      read: "$->sortedBaseInfos()",
      writeElement: "$->setBaseInfo($element.first, $element.second)",
    },
    virtualBases_: {
      cls: "P",
      read: "$->virtualBases()",
      writeElement: "$->addVirtualBase($element)",
    },
    padding_: {
      cls: "P",
      read: "$->padding()",
      writeElement:
        "$->addPadding($element.index, $element.offset, $element.sizeInBytes)",
    },
    primaryBase_: {
      cls: "P",
      read: "$->primaryBase()",
      write: "$->setPrimaryBase($value, $->primaryBaseIsVirtual())",
    },
    primaryBaseIsVirtual_: {
      cls: "P",
      read: "$->primaryBaseIsVirtual()",
      write: "$->setPrimaryBase($->primaryBase(), $value)",
    },
    abiEmpty_: {
      cls: "P",
      read: "$->isAbiEmpty()",
      write: "$->setAbiEmpty($value)",
    },
    hasVtable_: {
      cls: "P",
      read: "$->hasVtable()",
      write: "$->setHasVtable($value)",
    },
    hasDirectVtable_: {
      cls: "P",
      read: "$->hasDirectVtable()",
      write: "$->setHasDirectVtable($value)",
    },
  },
};

export function bindingOf(
  className: string,
  fieldName: string,
): FieldBinding | undefined {
  return bindings[className]?.[fieldName];
}

export interface ExtraField {
  /** The name the field carries in the archive and in the report. */
  name: string;
  /** The nullary accessor the value is read from; it also types the field. */
  from: string;
  cls: FieldClass;
  write?: string;
  writeElement?: string;
  why?: string;
}

/**
 * `MaybeTemplate` keeps its payload in a private `TemplateData` that is a
 * nested type of a template, so it is persisted through the public API rather
 * than field by field.
 */
export const extraFields: Record<string, ExtraField[]> = {
  "::cxx::MaybeTemplate": [
    {
      name: "templateDeclaration",
      from: "templateDeclaration",
      cls: "P",
      write: "$->setTemplateDeclaration($value)",
    },
    {
      name: "templateParameters",
      from: "templateParameters",
      cls: "P",
      write: "$->setTemplateParameters($value)",
    },
    {
      name: "specializations",
      from: "specializations",
      cls: "P",
      writeElement: "$->restoreSpecialization(std::move($element))",
    },
    {
      name: "primaryTemplateSymbol",
      from: "primaryTemplateSymbol",
      cls: "P",
      write:
        "$->restoreSpecializationInfo($value, $->templateSpecializationIndex())",
    },
    {
      name: "templateSpecializationIndex",
      from: "templateSpecializationIndex",
      cls: "P",
      write: "$->restoreSpecializationInfo($->primaryTemplateSymbol(), $value)",
    },
    {
      name: "externInstantiationDeclarations",
      from: "externInstantiationDeclarations",
      cls: "P",
      writeElement: "$->addExternInstantiationDeclaration($element)",
    },
  ],
};

export function extraFieldsOf(className: string): ExtraField[] {
  return extraFields[className] ?? [];
}
