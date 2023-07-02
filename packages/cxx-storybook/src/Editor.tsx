// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

import { FC, useEffect, useRef, useState } from "react";
import { basicSetup } from "codemirror";
import { cpp } from "@codemirror/lang-cpp";
import { EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
import { linter, Diagnostic } from "@codemirror/lint";
import * as cxx from "cxx-frontend";
import wasmBinaryUrl from "cxx-frontend/dist/cxx-js.wasm?url";
import "./Editor.css";

const setupCxxFrontend = async () => {
  console.log("Setting up cxx-frontend");
  const response = await fetch(wasmBinaryUrl);
  const data = await response.arrayBuffer();
  const wasmBinary = new Uint8Array(data);
  return await cxx.Parser.init({ wasmBinary });
};

await setupCxxFrontend();

export interface EditorProps {
  /**
   * The initial text of the editor.
   */
  initialValue?: string;

  /**
   * Whether the Editor owns the syntax tree.
   *
   * @default true
   */
  editorWillDisposeSyntaxTree?: boolean;

  /**
   * Called when the cursor position changes.
   *
   * @param lineNumber 1-based line number
   * @param column 1-based column number
   */
  onCursorPositionChanged?: (lineNumber: number, column: number) => void;

  /**
   * Called when the syntax is parsed.
   *
   * @param parser the Parser
   */
  onSyntaxChanged?: (parser: cxx.Parser) => void;
}

export const Editor: FC<EditorProps> = ({
  initialValue,
  editorWillDisposeSyntaxTree = true,
  onCursorPositionChanged,
  onSyntaxChanged,
}) => {
  const editorRef = useRef<HTMLDivElement>(null);

  const [editor, setEditor] = useState<EditorView | null>(null);

  useEffect(() => {
    if (!editor) return;

    editor.dispatch({
      changes: { from: 0, to: editor.state.doc.length, insert: initialValue },
    });
  }, [editor, initialValue]);

  useEffect(() => {
    const domElement = editorRef.current;

    if (!domElement) {
      return;
    }

    const syntaxChecker = (view: EditorView) => {
      if (!cxx.Parser.isInitialized()) {
        // The parser is not ready yet, we can't do anything
        return [];
      }

      const source = view.state.doc.toString();

      const parser = new cxx.Parser({
        path: "main.cc",
        source,
      });

      parser.parse();

      const diagnostics: Diagnostic[] = [];

      for (const diagnostic of parser.getDiagnostics()) {
        const { startLine, startColumn, endLine, endColumn, message } =
          diagnostic;

        const from =
          view.state.doc.line(startLine).from + Math.max(startColumn - 1, 0);

        const to =
          view.state.doc.line(endLine).from + Math.max(endColumn - 1, 0);

        diagnostics.push({
          severity: "error",
          from,
          to,
          message,
        });
      }

      onSyntaxChanged?.(parser);

      if (editorWillDisposeSyntaxTree || !onSyntaxChanged) {
        parser.dispose();
      }

      return diagnostics;
    };

    const needsRefresh = () => {
      return !cxx.Parser.isInitialized();
    };

    const cppLinter = linter(syntaxChecker, { needsRefresh });

    const updateListener = EditorView.updateListener.of((update) => {
      if (update.selectionSet && onCursorPositionChanged) {
        const sel = update.state.selection.main;
        const line = update.state.doc.lineAt(sel.to);
        const column = sel.from - line.from;
        onCursorPositionChanged(line.number, column);
      }
    });

    const startState = EditorState.create({
      doc: "",
      extensions: [basicSetup, cpp(), updateListener, cppLinter],
    });

    const editor = new EditorView({
      state: startState,
      parent: domElement,
    });

    setEditor(editor);

    return () => {
      editor.destroy();
    };
  }, [editorRef]);

  return <div ref={editorRef} className="Editor" />;
};
