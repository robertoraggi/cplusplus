import { FC, useEffect, useRef, useState } from "react";
import * as monaco from "monaco-editor";

interface EditorProps {
  value?: string;
}

export const Editor: FC<EditorProps> = ({ value }) => {
  const editorRef = useRef<HTMLDivElement>(null);

  const [editor, setEditor] =
    useState<monaco.editor.IStandaloneCodeEditor | null>(null);

  useEffect(() => {
    editor?.setValue(value ?? "");
  }, [editor, value]);

  useEffect(() => {
    const domElement = editorRef.current;

    if (!domElement) {
      return;
    }

    const editor = monaco.editor.create(domElement, {
      automaticLayout: true,
      minimap: {
        enabled: false,
      },
      language: "cpp",
      value: "",
    });

    setEditor(editor);

    return () => {
      editor.dispose();
    };
  }, [editorRef]);

  return <div ref={editorRef} style={{ height: "100%", minHeight: "200px" }} />;
};
