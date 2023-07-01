import { FC, useEffect, useRef, useState } from "react";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { defaultKeymap } from "@codemirror/commands";
import { cpp } from "@codemirror/lang-cpp";
import { basicSetup } from "codemirror";

interface EditorProps {
  value?: string;
}

export const Editor: FC<EditorProps> = ({ value }) => {
  const editorRef = useRef<HTMLDivElement>(null);

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

    const startState = EditorState.create({
      doc: "Hello World",
      extensions: [basicSetup, cpp()],
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
