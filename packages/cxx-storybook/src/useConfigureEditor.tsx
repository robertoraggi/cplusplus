import { useCallback, useEffect, useState } from "react";
import { basicSetup } from "codemirror";
import { EditorView } from "@codemirror/view";
import { Compartment, EditorState, Extension } from "@codemirror/state";
import { linter, Diagnostic, lintGutter } from "@codemirror/lint";
import { cpp } from "@codemirror/lang-cpp";
import { Parser } from "cxx-frontend";

function cppLinter({
  onDocumentChanged,
}: {
  onDocumentChanged: (ast: Parser | null) => void;
}) {
  const checkSyntax = (view: EditorView) => {
    const state = view.state;
    const doc = state.doc;

    const source = view.state.doc.toString();

    const parser = new Parser({ source, path: "-" });

    parser.parse();

    onDocumentChanged(parser);

    const diags: Diagnostic[] = parser.getDiagnostics().map((diagnostic) => {
      const { startLine, startColumn, endLine, endColumn, message } =
        diagnostic;

      const from = doc.line(startLine).from + Math.max(startColumn - 1, 0);

      const to = doc.line(endLine).from + Math.max(endColumn - 1, 0);

      return {
        severity: "error",
        from,
        to,
        message,
      };
    });

    return diags;
  };

  return linter(checkSyntax, {
    delay: 500,
  });
}

export function useConfigureEditor({
  initialValue = "",
  editable = true,
  checkSyntax = true,
  onParserChanged,
}: {
  initialValue?: string;
  editable?: boolean;
  checkSyntax?: boolean;
  onParserChanged?: (parser: Parser | null) => void;
} = {}) {
  const [editor, setEditor] = useState<EditorView | null>(null);
  const [editableCompartment, setEditableCompartment] = useState<Compartment>();
  const [lintCompartment, setLintCompartment] = useState<Compartment>();
  const [parser, setParser] = useState<Parser | null>(null);

  useEffect(() => {
    onParserChanged?.(parser);
  }, [parser, onParserChanged]);

  useEffect(() => {
    if (editor === null) return;

    const changes = editor.state.changes({
      from: 0,
      to: editor.state.doc.length,
      insert: initialValue,
    });

    editor.dispatch({ changes });
  }, [editor, initialValue]);

  useEffect(() => {
    if (!editor) return;
    if (!editableCompartment) return;
    const effects = editableCompartment.reconfigure(
      EditorView.editable.of(editable)
    );
    editor.dispatch({ effects });
  }, [editor, editable, editableCompartment]);

  useEffect(() => {
    if (!editor) return;
    if (!lintCompartment) return;

    setParser((previousParser) => {
      previousParser?.dispose();
      return null;
    });

    const extensions: Extension[] = [];

    if (checkSyntax) {
      const linter = cppLinter({
        onDocumentChanged: (parser) => {
          setParser((previousParser) => {
            previousParser?.dispose();
            return parser;
          });
        },
      });

      extensions.push(linter, lintGutter());
    }

    const effects = lintCompartment.reconfigure(extensions);
    editor.dispatch({ effects });
  }, [editor, checkSyntax, lintCompartment]);

  return useCallback((parent: HTMLDivElement | null) => {
    if (parent === null) {
      // unmount
      setParser((previousParser) => {
        previousParser?.dispose();
        return null;
      });

      setEditor((previousEditor) => {
        previousEditor?.destroy();
        return null;
      });

      return;
    }

    const editableCompartment = new Compartment();
    setEditableCompartment(editableCompartment);

    const lintCompartment = new Compartment();
    setLintCompartment(lintCompartment);

    // mount
    const state = EditorState.create({
      extensions: [
        basicSetup,
        cpp(),
        editableCompartment.of([]),
        lintCompartment.of([]),
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
  }, []);
}
