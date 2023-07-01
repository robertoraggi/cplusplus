import { FC, useEffect, useRef, useState } from "react";
import { basicSetup } from "codemirror";
import { cpp } from "@codemirror/lang-cpp";
import { EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
import * as cxx from "cxx-frontend";

interface EditorProps {
  initialValue?: string;
  didChangeCursorPosition?: (lineNumber: number, column: number) => void;
  didChangeValue?: (value: string) => void;
  didParse?: (parser: cxx.Parser) => void;
}

export const Editor: FC<EditorProps> = ({
  initialValue: value,
  didChangeCursorPosition,
  didChangeValue,
  didParse,
}) => {
  const editorRef = useRef<HTMLDivElement>(null);
  const didChangeCursorPositionRef = useRef(didChangeCursorPosition);
  const didChangeValueRef = useRef(didChangeValue);
  const didParseRef = useRef(didParse);

  didChangeCursorPositionRef.current = didChangeCursorPosition;
  didChangeValueRef.current = didChangeValue;
  didParseRef.current = didParse;

  const [editor, setEditor] = useState<EditorView | null>(null);

  const [cxxPromise] = useState(() => {
    const setup = async () => {
      const response = await fetch(cxx.Parser.DEFAULT_WASM_BINARY_URL);
      const data = await response.arrayBuffer();
      const wasmBinary = new Uint8Array(data);
      await cxx.Parser.init({ wasmBinary });
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

    const updateListener = EditorView.updateListener.of((update) => {
      if (update.selectionSet && didChangeCursorPositionRef.current) {
        const sel = update.state.selection.main;
        const line = update.state.doc.lineAt(sel.to);
        const column = sel.from - line.from;
        didChangeCursorPositionRef.current?.(line.number, column);
      }

      if (
        update.docChanged &&
        (didChangeValueRef.current || didParseRef.current)
      ) {
        const value = update.state.doc.toString();

        didChangeValueRef.current?.(value);

        // ### TODO: delay
        const checkSyntax = async () => {
          await cxxPromise;

          const parser = new cxx.Parser({
            source: value,
            path: "main.cc",
          });

          parser.parse();

          didParseRef.current?.(parser);

          parser.dispose();
        };

        checkSyntax();
      }
    });

    const startState = EditorState.create({
      doc: "",
      extensions: [basicSetup, cpp(), updateListener],
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
