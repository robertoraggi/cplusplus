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

import { useContext, useEffect, useState } from "react";
import ASTContext from "./ast-context";
import EditorContext from "./editor-context";
import { SyntaxTree } from "./syntax-tree";
import useDebouncedOnDidChangeCursorPosition from "./hooks/use-debounced-on-did-change-cursor-position";
import * as monaco from "monaco-editor";
import ModelContext from "./editor-model-context";

export default function AbstractSyntaxTree() {
  const { parser, fileName } = useContext(ASTContext);
  const { editor } = useContext(EditorContext);
  const { model } = useContext(ModelContext);

  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });

  useEffect(() => {
    const markers: monaco.editor.IMarkerData[] = [];

    const diagnostics = parser?.getDiagnostics() ?? [];

    diagnostics.forEach(
      ({ startLine, startColumn, endLine, endColumn, message }) => {
        markers.push({
          severity: monaco.MarkerSeverity.Error,
          startLineNumber: startLine,
          startColumn: startColumn,
          endLineNumber: endLine,
          endColumn: endColumn,
          message: message,
          source: "C++",
        });
      },
    );

    monaco.editor.setModelMarkers(model, "cxx", markers);
  }, [model, parser]);

  useDebouncedOnDidChangeCursorPosition({
    editor,
    onDidChangeCursorPosition: (editor, position) => {
      const model = editor.getModel();
      if (!model) return;

      const line = position.lineNumber;
      const column = Math.max(0, position.column - 1);
      setCursorPosition({ line, column });
    },
  });

  return (
    <SyntaxTree
      parser={parser}
      mainFileName={fileName}
      cursorPosition={cursorPosition}
    />
  );
}
