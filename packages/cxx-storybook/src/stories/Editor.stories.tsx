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

import { Editor } from "../Editor";
import { CxxFrontendProvider } from "../CxxFrontendProvider";
import { CxxFrontendClient } from "../CxxFrontendClient";
import { Parser } from "cxx-frontend";
import { useState } from "react";
import { SyntaxTree } from "./SyntaxTree";

const client = new CxxFrontendClient();

const meta: Meta<typeof Editor> = {
  title: "CxxFrontend/Editor",
  component: Editor,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <CxxFrontendProvider client={client}>
        <Story />
      </CxxFrontendProvider>
    ),
  ],
};

export default meta;

type EditorStory = StoryObj<typeof Editor>;

export const EditableEditorWithSyntaxChecker: EditorStory = {
  args: {
    initialValue: `auto main() -> int {\n  return 0\n}`,
    editable: true,
    checkSyntax: true,
  },
};

export const EditableEditor: EditorStory = {
  args: {
    initialValue: `auto main() -> int {\n  return 0;\n}`,
    editable: true,
    checkSyntax: false,
  },
};

export const ReadonlyEditor: EditorStory = {
  args: {
    initialValue: `auto main() -> int {\n  return 0;\n}`,
    editable: false,
    checkSyntax: false,
  },
};

export function EditorWithSyntaxTree() {
  const [parser, setParser] = useState<Parser | null>(null);

  const value = `
template <typename T>
concept CanAdd = requires(T n) {
  n + n;
};

auto twice(CanAdd auto n) {
  return n + n;
}

int main() {
  return twice(2);
}
`;

  return (
    <div style={{ display: "flex", height: "90svh" }}>
      <Editor
        initialValue={value}
        style={{ flex: 1, overflow: "auto" }}
        onParserChanged={setParser}
      />
      <SyntaxTree parser={parser} />
    </div>
  );
}
