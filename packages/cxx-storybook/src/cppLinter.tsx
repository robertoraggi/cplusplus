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

import { EditorView } from "@codemirror/view";
import { linter, Diagnostic } from "@codemirror/lint";
import { Parser, Diagnostic as ParserDiagnostic } from "cxx-frontend";

export function cppLinter({
  onDocumentChanged,
  delay,
}: {
  onDocumentChanged: (ast: Parser | null) => void;
  delay?: number;
}) {
  const checkSyntax = (view: EditorView) => {
    function convertDiagnostic(diagnostic: ParserDiagnostic): Diagnostic {
      const doc = view.state.doc;

      const { startLine, startColumn, endLine, endColumn, message } =
        diagnostic;

      const from = doc.line(startLine).from + Math.max(startColumn - 1, 0);
      const to = doc.line(endLine).from + Math.max(endColumn - 1, 0);

      return {
        severity: "error",
        from,
        to,
        message,
      };
    }

    const source = view.state.doc.toString();

    const parser = new Parser({ source, path: "-" });

    parser.parse();

    onDocumentChanged(parser);

    const diagnostics = parser
      .getDiagnostics()
      .map((diagnostic) => convertDiagnostic(diagnostic));

    return diagnostics;
  };

  return linter(checkSyntax, { delay });
}
