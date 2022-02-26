// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

//@ts-ignore
import makeCxx from "./cxx-js.js";
import { SourceLocation } from "./SourceLocation.js";
import { TokenKind } from "./TokenKind.js";
import { Unit } from "./Unit.js";

export enum ASTSlotKind {
    Invalid,
    Token,
    Node,
    TokenList,
    NodeList,
};

export interface CXX {
    createUnit(source: string, path: string): Unit;
    getASTKind(handle: number): number;
    getASTSlot(handle: number, slot: number): number;
    getASTSlotKind(handle: number, slot: number): ASTSlotKind;
    getASTSlotCount(handle: number, slot: number): number;
    getListValue(handle: number): number;
    getListNext(handle: number): number;
    getTokenText(handle: number, unitHandle: number): string;
    getTokenKind(handle: number, unitHandle: number): TokenKind;
    getTokenLocation(handle: number, unitHandle: number): SourceLocation;
    getStartLocation(handle: number, unitHandle: number): SourceLocation;
    getEndLocation(handle: number, unitHandle: number): SourceLocation;
}

export let cxx!: CXX

export default async ({ wasmBinary }: { wasmBinary: Uint8Array }) => {
    cxx = await makeCxx({ wasmBinary });
    return cxx;
}
