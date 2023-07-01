import { FC, useEffect, useRef, useState } from "react";
import { basicSetup } from "codemirror";
import { cpp } from "@codemirror/lang-cpp";
import { EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";

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
      doc: "",
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
