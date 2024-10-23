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

export type Type = BaseType;

export type BaseTypeName = "string" | "integer" | "uinteger";

export type BaseType = {
  kind: "base";
  name: BaseTypeName;
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

export type Notification = {};

export type Request = {};

export type Structure = {};

export type TypeAlias = {};

export type MetaModel = {
  metaData: MetaData;
  enumerations: Enumeration[];
  notifications: Notification[];
  requests: Request[];
  structures: Structure[];
  typeAliases: TypeAlias[];
};

export function getEnumBaseType(enumeration: Enumeration) {
  switch (enumeration.type.name) {
    case "integer":
      return " : int";
    case "uinteger":
      return " : unsigned int";
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

export function getEnumeratorInitializer(
  enumeration: Enumeration,
  enumerator: EnumerationValue
) {
  if (enumeration.type.name === "string") {
    return "";
  }
  return ` = ${enumerator.value}`;
}
