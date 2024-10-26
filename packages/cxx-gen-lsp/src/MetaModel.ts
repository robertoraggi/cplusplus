// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

export type Type =
  | BaseType
  | TupleType
  | OrType
  | AndType
  | ReferenceType
  | ArrayType
  | MapType
  | StringLiteralType
  | LiteralType;

export type BaseTypeName = "null" | "string" | "integer" | "uinteger" | "decimal" | "boolean" | "DocumentUri" | "URI";

export type BaseType = {
  kind: "base";
  name: BaseTypeName;
};

export type TupleType = {
  kind: "tuple";
  items: Type[];
};

export type OrType = {
  kind: "or";
  items: Type[];
};

export type AndType = {
  kind: "and";
  items: Type[];
};

export type ReferenceType = {
  kind: "reference";
  name: string;
};

export type ArrayType = {
  kind: "array";
  element: Type;
};

export type MapType = {
  kind: "map";
  key: Type;
  value: Type;
};

export type StringLiteralType = {
  kind: "stringLiteral";
  value: string;
};

export type LiteralType = {
  kind: "literal";
  value: {
    properties: unknown[];
  };
};

export type EnumerationValue = {
  name: string;
  value: string;
};

export type MetaData = {};

export type Enumeration = {
  documentation?: string;
  name: string;
  since?: string;
  supportsCustomValues?: boolean;
  type: BaseType;
  values: EnumerationValue[];
};

export type Notification = Omit<Request, "result" | "partialResult" | "registrationOptions">;

type MessageDirection = "clientToServer" | "serverToClient" | "both";

export type Request = {
  documentation?: string;
  messageDirection: MessageDirection;
  method: string;
  params: ReferenceType;
  partialResult?: ReferenceType | ArrayType | OrType;
  registrationOptions?: ReferenceType | AndType;
  result?: BaseType | ReferenceType | ArrayType | OrType;
  typeName: string;
};

export type Property = {
  documentation?: string;
  name: string;
  type: Type;
  optional?: boolean;
};

export type Structure = {
  extends?: ReferenceType[];
  mixins?: ReferenceType[];
  name: string;
  properties: Property[];
};

export type TypeAlias = {
  documentation?: string;
  name: string;
  type: Type;
};

export type MetaModel = {
  metaData: MetaData;
  enumerations: Enumeration[];
  notifications: Notification[];
  requests: Request[];
  structures: Structure[];
  typeAliases: TypeAlias[];
};

export function isRequest(request: Request | Notification): request is Request {
  return "result" in request;
}

export function getEnumBaseType(enumeration: Enumeration) {
  switch (enumeration.type.name) {
    case "integer":
      return " : int";
    case "uinteger":
      return " : long";
    default:
      return "";
  }
}

export function toUpperCamelCase(name: string) {
  return name[0].toUpperCase() + name.slice(1);
}

export function getEnumeratorName(enumerator: EnumerationValue) {
  const name = toUpperCamelCase(enumerator.name);
  return `k${name}`;
}

export function getEnumeratorInitializer(enumeration: Enumeration, enumerator: EnumerationValue) {
  if (enumeration.type.name === "string") {
    return "";
  }
  return ` = ${enumerator.value}`;
}

export function toCppType(type: Type): string {
  switch (type.kind) {
    case "base": {
      switch (type.name) {
        case "null":
          return "std::nullptr_t";
        case "string":
          return "std::string";
        case "integer":
          return "int";
        case "uinteger":
          return "long";
        case "decimal":
          return "double";
        case "boolean":
          return "bool";
        case "DocumentUri":
          return "std::string";
        case "URI":
          return "std::string";
        default:
          throw new Error(`Unknown base type: ${JSON.stringify(type)}`);
      } // switch type.name
    } // case "base"

    case "stringLiteral":
      return "std::string";

    case "literal":
      return "json";

    case "reference":
      return type.name;

    case "array":
      return `Vector<${toCppType(type.element)}>`;

    case "map":
      return `Map<${toCppType(type.key)}, ${toCppType(type.value)}>`;

    case "tuple":
      return `std::tuple<${type.items.map(toCppType).join(", ")}>`;

    case "or":
      return `std::variant<${type.items.map(toCppType).join(", ")}>`;

    case "and":
      return `std::tuple<${type.items.map(toCppType).join(", ")}>`;

    default:
      throw new Error(`Unknown type kind: ${JSON.stringify(type)}`);
  } // switch
}

export function getStructureProperties(model: MetaModel, structure: Structure): Property[] {
  const structByName = new Map(model.structures.map((s) => [s.name, s]));
  const added = new Set<string>();
  return getStructurePropertiesHelper({ structure, added, structByName });
}

function getStructurePropertiesHelper({
  structure,
  added,
  structByName,
}: {
  structure: Structure;
  added: Set<string>;
  structByName: Map<string, Structure>;
}): Property[] {
  const properties: Property[] = [];

  for (const property of structure.properties) {
    if (added.has(property.name)) {
      continue;
    }
    added.add(property.name);
    properties.push(property);
  }

  structure.extends?.forEach((ref) => {
    const extend = structByName.get(ref.name);

    if (!extend) {
      throw new Error(`Unknown extends ${ref.name}`);
    }

    properties.push(
      ...getStructurePropertiesHelper({
        structure: extend,
        added,
        structByName,
      }),
    );
  });

  structure.mixins?.forEach((ref) => {
    const mixin = structByName.get(ref.name);

    if (!mixin) {
      throw new Error(`Unknown mixin ${ref.name}`);
    }

    properties.push(...getStructurePropertiesHelper({ structure: mixin, added, structByName }));
  });

  return properties;
}
