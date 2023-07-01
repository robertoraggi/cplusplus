import { FC, useEffect, useRef, useState } from "react";
import { basicSetup } from "codemirror";
import { cpp } from "@codemirror/lang-cpp";
import { EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";

interface EditorProps {
  initialValue?: string;
  didChangeCursorPosition?: (lineNumber: number, column: number) => void;
  didChangeValue?: (value: string) => void;
}

export const Editor: FC<EditorProps> = ({
  initialValue: value,
  didChangeCursorPosition,
  didChangeValue,
}) => {
  const editorRef = useRef<HTMLDivElement>(null);
  const didChangeCursorPositionRef = useRef(didChangeCursorPosition);
  const didChangeValueRef = useRef(didChangeValue);

  didChangeCursorPositionRef.current = didChangeCursorPosition;
  didChangeValueRef.current = didChangeValue;

  const [editor, setEditor] = useState<EditorView | null>(null);

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
        console.log(`selection set: ${line.number}, ${column}`);
        didChangeCursorPositionRef.current?.(line.number, column);
      }

      if (update.docChanged && didChangeValueRef.current) {
        const value = update.state.doc.toString();
        didChangeValueRef.current(value);
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
  }, [editorRef]);

  return <div ref={editorRef} />;
};
