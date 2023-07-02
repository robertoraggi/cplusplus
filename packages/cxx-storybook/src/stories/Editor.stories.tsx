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

import type { Meta, StoryObj } from "@storybook/react";

import { Editor, EditorProps } from "../Editor";
import { Parser } from "cxx-frontend";

const meta = {
  title: "Example/Editor",
  component: Editor,
  tags: ["autodocs"],
  argTypes: {},
} satisfies Meta<typeof Editor>;

export default meta;
type Story = StoryObj<typeof meta>;

class SyntaxCheckerArgs implements EditorProps {
  #lastParser?: Parser;

  editorWillDisposeSyntaxTree = false;

  initialValue = [
    "#include <iostream>",
    "",
    "int main() {",
    '  std::cout << "Hello, world!" << std::endl;',
    "  return 0;",
    "}",
  ].join("\n");

  onCursorPositionChanged(_lineNumber: number, _column: number) {}

  onSyntaxChanged(parser: Parser) {
    this.#lastParser?.dispose();
    this.#lastParser = parser;
  }
}

export const SyntaxChecker: Story = {
  args: new SyntaxCheckerArgs(),
};
