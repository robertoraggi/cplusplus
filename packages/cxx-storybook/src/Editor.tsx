import { FC, useState } from "react";
import * as monaco from "monaco-editor";

export const Editor: FC = ({}) => {
  const [editor, setEditor] =
    useState<monaco.editor.IStandaloneCodeEditor | null>(null);

  const setup = (domElement: HTMLDivElement) => {
    if (!domElement) {
      editor?.dispose();
      return;
    }

    setEditor((editor) => {
      if (!editor) {
        editor = monaco.editor.create(domElement, {
          automaticLayout: true,
          minimap: {
            enabled: false,
          },
          language: "cpp",
          value: ["int main() {", "\treturn 0;", "}"].join("\n"),
        });
      }

      return editor;
    });
  };

  return <div ref={setup} style={{ height: "100%" }} />;
};
