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

import { cxx, Unit } from "./cxx";
import { AST } from "./AST";

interface ParseParams {
    /**
     * Path to the file to parse.
     */
    path: string;

    /**
     * Source code to parse.
     */
    source: string;
}

export class Parser {
    private unit: Unit | undefined;
    private m_ast: AST | undefined;

    constructor(private options: ParseParams) {
    }

    parse() {
        const { path, source } = this.options;

        if (typeof path !== "string") {
            throw new TypeError("expected parameter 'path' of type 'string'");
        }

        if (typeof source !== "string") {
            throw new TypeError("expected parameter 'source' of type 'string'");
        }

        const unit = cxx.parse(source, path);

        this.unit = unit;

        this.m_ast = AST.from(this.unit.getHandle());
    }

    dispose() {
        this.unit?.delete();
        this.unit = undefined;
        this.m_ast = undefined;
    }

    getAST(): AST | undefined {
        return this.m_ast;
    }

    getDiagnostics() {
        return this.unit?.getDiagnostics() ?? [];
    }
}
