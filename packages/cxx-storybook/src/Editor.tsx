import { FC, useEffect, useRef, useState } from "react";
import { basicSetup } from "codemirror";
import { cpp } from "@codemirror/lang-cpp";
import { EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
import { linter, Diagnostic } from "@codemirror/lint";
import * as cxx from "cxx-frontend";

interface EditorProps {
  initialValue?: string;
  didChangeCursorPosition?: (lineNumber: number, column: number) => void;
  didParse?: (parser: cxx.Parser) => void;
}

export const Editor: FC<EditorProps> = ({
  initialValue: value,
  didChangeCursorPosition,
  didParse,
}) => {
  const editorRef = useRef<HTMLDivElement>(null);
  const didChangeCursorPositionRef = useRef(didChangeCursorPosition);
  const didParseRef = useRef(didParse);

  didChangeCursorPositionRef.current = didChangeCursorPosition;
  didParseRef.current = didParse;

  const [editor, setEditor] = useState<EditorView | null>(null);

  const parserIsReady = useRef(false);

  const [cxxPromise] = useState(() => {
    const setup = async () => {
      const response = await fetch(cxx.Parser.DEFAULT_WASM_BINARY_URL);
      const data = await response.arrayBuffer();
      const wasmBinary = new Uint8Array(data);
      await cxx.Parser.init({ wasmBinary });
      parserIsReady.current = true;
    };
    return setup();
  });

  useEffect(() => {
    if (!editor) return;

    editor.dispatch({
      changes: { from: 0, to: editor.state.doc.length, insert: value },
    });
  }, [editor, value]);

  useEffect(() => {
    const domElement = editorRef.current;

    if (!domElement) {
      return;
    }

    const syntaxChecker = linter((view) => {
      if (!parserIsReady.current) {
        // if cxxPromise is not resolved yet, we can't do anything
        return [];
      }

      const source = view.state.doc.toString();

      const parser = new cxx.Parser({
        path: "main.cc",
        source,
      });

      parser.parse();

      didParseRef.current?.(parser);

      const diagnostics: Diagnostic[] = [];

      for (const diagnostic of parser.getDiagnostics()) {
        const { startLine, startColumn, endLine, endColumn, message } =
          diagnostic;
        diagnostics.push({
          severity: "error",
          from: view.state.doc.line(startLine).from + startColumn - 1,
          to: view.state.doc.line(endLine).from + endColumn - 1,
          message,
        });
      }

      parser.dispose();

      return diagnostics;
    });

    const updateListener = EditorView.updateListener.of((update) => {
      if (update.selectionSet && didChangeCursorPositionRef.current) {
        const sel = update.state.selection.main;
        const line = update.state.doc.lineAt(sel.to);
        const column = sel.from - line.from;
        didChangeCursorPositionRef.current?.(line.number, column);
      }
    });

    const startState = EditorState.create({
      doc: "",
      extensions: [basicSetup, cpp(), updateListener, syntaxChecker],
    });

    const editor = new EditorView({
      state: startState,
      parent: domElement,
    });

    setEditor(editor);

    return () => {
      editor.destroy();
    };
  }, [editorRef, cxxPromise]);

  return <div ref={editorRef} />;
};
