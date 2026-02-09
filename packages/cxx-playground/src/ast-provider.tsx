// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

import ASTContext from "./ast-context";
import { Parser } from "cxx-frontend";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import * as monaco from "monaco-editor";
import useDebouncedOnDidChangeContent from "./hooks/use-debounced-on-did-change-content";

export default function ASTProvider({
  interval = 250,
  model,
  children,
}: {
  interval?: number;
  model: monaco.editor.ITextModel;
  children: ReactNode;
}) {
  const [parser, setParser] = useState<Parser | null>(null);
  const fileName = "main.cc";

  const onDidChangeContent = useCallback((model: monaco.editor.ITextModel) => {
    const source = model.getValue();

    let parser: Parser | null = new Parser({
      path: fileName,
      source,
    });

    parser
      .parse()
      .catch((error) => {
        console.error("Error parsing source code:", error);
        parser?.dispose();
        parser = null;
      })
      .then(() => {
        setParser((previous) => {
          previous?.dispose();
          return parser;
        });
      });

    return () => {
      parser?.dispose();
    };
  }, []);

  useEffect(() => {
    onDidChangeContent(model);
  }, [model, onDidChangeContent]);

  useDebouncedOnDidChangeContent({ model, onDidChangeContent, interval });

  return (
    <ASTContext.Provider value={{ parser, fileName }}>
      {children}
    </ASTContext.Provider>
  );
}
