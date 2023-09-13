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

import { useCallback, useEffect, useState } from "react";
import { basicSetup } from "codemirror";
import { EditorView } from "@codemirror/view";
import { Compartment, EditorState, Extension } from "@codemirror/state";
import { lintGutter } from "@codemirror/lint";
import { cpp } from "@codemirror/lang-cpp";
import { Parser } from "cxx-frontend";
import { cppLinter } from "./cppLinter";

interface Position {
  line: number;
  column: number;
}

/**
 * Configure the editor.
 */
export function useConfigureEditor({
  initialValue,
  editable,
  checkSyntax,
  delay,
  onParserChanged,
  onCursorPositionChanged,
}: {
  initialValue: string;
  editable: boolean;
  checkSyntax: boolean;
  delay: number;
  onParserChanged?: (parser: Parser | null) => void;
  onCursorPositionChanged?: (cursor: Position) => void;
}) {
  const updateParser = useCallback((parser: Parser | null) => {
    setParser((previousParser) => {
      previousParser?.dispose();
      return parser;
    });
  }, []);

  const [editor, setEditor] = useState<EditorView | null>(null);
  const [parser, setParser] = useState<Parser | null>(null);
  const [cursorPosition, setCursorPosition] = useState<Position>({
    line: 1,
    column: 0,
  });
  const editableCompartment = useConfigureEditable({ editor, editable });
  const lintCompartment = useConfigureEditorLinter({
    editor,
    checkSyntax,
    delay,
    updateParser,
  });

  // sync parser updates
  useEffect(() => onParserChanged?.(parser), [parser, onParserChanged]);

  // sync cursor position updates
  useEffect(
    () => onCursorPositionChanged?.(cursorPosition),
    [cursorPosition, onCursorPositionChanged]
  );

  useSetEditorValue({ editor, initialValue });

  return useCallback(
    (parent: HTMLDivElement | null) => {
      if (parent === null) {
        // unmount
        updateParser(null);

        setEditor((previousEditor) => {
          previousEditor?.destroy();
          return null;
        });

        return;
      }

      const updateListener = EditorView.updateListener.of((update) => {
        if (update.selectionSet) {
          const selection = update.state.selection.main;
          const editorLine = update.state.doc.lineAt(selection.head);
          const line = editorLine.number;
          const column = selection.from - editorLine.from;
          setCursorPosition({ line, column });
        }
      });

      // mount
      const state = EditorState.create({
        extensions: [
          basicSetup,
          updateListener,
          editableCompartment.of([]),
          lintCompartment.of([]),
          cpp(),
        ],
      });

      const view = new EditorView({
        state,
        parent,
      });

      setEditor((previousEditor) => {
        previousEditor?.destroy();
        return view;
      });
    },
    [editableCompartment, lintCompartment, updateParser]
  );
}

/**
 * Configure the linter.
 * @param editor The editor.
 * @param checkSyntax Whether to check the syntax.
 * @param delay The delay in milliseconds to wait before checking the syntax.
 * @param updateParser The function to call when the parser changes.
 * @returns The linter compartment.
 */
function useConfigureEditorLinter({
  editor,
  checkSyntax,
  delay,
  updateParser,
}: {
  editor: EditorView | null;
  checkSyntax: boolean;
  delay: number;
  updateParser: (parser: Parser | null) => void;
}) {
  const [compartment] = useState(() => new Compartment());

  useEffect(() => {
    if (!editor) return;
    if (!compartment) return;

    updateParser(null);

    const extensions: Extension[] = [];

    if (checkSyntax) {
      const linter = cppLinter({
        onDocumentChanged: updateParser,
        delay,
      });
      extensions.push(linter, lintGutter());
    }

    const effects = compartment.reconfigure(extensions);
    editor.dispatch({ effects });
  }, [editor, checkSyntax, delay, compartment, updateParser]);

  return compartment;
}

/**
 * Configure whether the editor is editable.
 * @param editor The editor.
 * @param editable Whether the editor is editable.
 * @returns The editable compartment.
 */
function useConfigureEditable({
  editor,
  editable,
}: {
  editor: EditorView | null;
  editable: boolean;
}) {
  const [compartment] = useState(() => new Compartment());

  useEffect(() => {
    if (!editor) return;
    if (!compartment) return;
    const effects = compartment.reconfigure(EditorView.editable.of(editable));
    editor.dispatch({ effects });
  }, [editor, editable, compartment]);

  return compartment;
}

/**
 * Set the value of the editor.
 * @param editor The editor.
 * @param initialValue The initial value.
 */
function useSetEditorValue({
  editor,
  initialValue,
}: {
  editor: EditorView | null;
  initialValue: string;
}) {
  useEffect(() => {
    if (editor === null) return;

    const changes = editor.state.changes({
      from: 0,
      to: editor.state.doc.length,
      insert: initialValue,
    });

    editor.dispatch({ changes });
  }, [editor, initialValue]);
}
