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

import { Editor } from "../Editor";
import { Parser } from "cxx-frontend";
import { useState } from "react";
import { SyntaxTree } from "../SyntaxTree";

interface EditorWithSyntaxTreeProps {
  /**
   * The initial value of the editor.
   */
  initialValue: string;

  /**
   * The delay in milliseconds before parsing the document.
   * @default 250
   */
  delay: number;
}

export function EditorWithSyntaxTree({
  initialValue,
  delay = 250,
}: EditorWithSyntaxTreeProps) {
  const [parser, setParser] = useState<Parser | null>(null);
  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 0 });

  return (
    <div style={{ display: "flex", height: "100svh", overflow: "hidden" }}>
      <Editor
        style={{ flex: 1, overflow: "auto" }}
        initialValue={initialValue}
        delay={delay}
        onParserChanged={setParser}
        onCursorPositionChanged={setCursorPosition}
      />
      <SyntaxTree parser={parser} cursorPosition={cursorPosition} />
    </div>
  );
}
