// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

import { ComponentProps } from "react";
import { useConfigureEditor } from "./useConfigureEditor";
import { Parser } from "cxx-frontend";

export interface EditorProps
  extends Pick<ComponentProps<"div">, "className" | "style"> {
  /**
   * The initial value of the editor.
   * @default ""
   */
  initialValue?: string;

  /**
   * Whether the editor is editable.
   * @default true
   */
  editable?: boolean;

  /**
   * The delay in milliseconds before parsing the document.
   * @default 250
   */
  delay?: number;

  /**
   * Whether to check the syntax of the document.
   * @default true
   */
  checkSyntax?: boolean;

  /**
   * Called when the parser changes.
   * @param parser The new parser or null if the parser was unset.
   */
  onParserChanged?: (parser: Parser | null) => void;

  /**
   * Called when the cursor position changes.
   * @param line The new line.
   * @param column The new column.
   */
  onCursorPositionChanged?: ({
    line,
    column,
  }: {
    line: number;
    column: number;
  }) => void;
}

export function Editor({
  initialValue = "",
  delay = 250,
  checkSyntax = true,
  editable = true,
  onParserChanged,
  onCursorPositionChanged,
  ...props
}: EditorProps) {
  const configureEditor = useConfigureEditor({
    initialValue,
    checkSyntax,
    editable,
    delay,
    onParserChanged,
    onCursorPositionChanged,
  });
  return <div ref={configureEditor} {...props} />;
}
