// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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
import { EditorWithSyntaxTree } from "./EditorWithSyntaxTree";
import { CxxFrontendProvider } from "../CxxFrontendProvider";
import { CxxFrontendClient } from "../CxxFrontendClient";

const client = new CxxFrontendClient();

const meta: Meta<typeof EditorWithSyntaxTree> = {
  title: "CxxFrontend/SyntaxTree",
  component: EditorWithSyntaxTree,
  tags: ["autodocs"],
  parameters: {
    layout: "fullscreen",
  },
  decorators: [
    (Story) => (
      <CxxFrontendProvider client={client}>
        <Story />
      </CxxFrontendProvider>
    ),
  ],
};

export default meta;

type EditorWithSyntaxTreeStory = StoryObj<typeof EditorWithSyntaxTree>;

export const Basic: EditorWithSyntaxTreeStory = {
  loaders: [
    async () => {
      const response = await fetch(
        "https://raw.githubusercontent.com/robertoraggi/cplusplus/main/src/parser/cxx/parser.cc",
      );

      if (!response.ok) {
        return {
          source: [
            "auto main() -> int",
            "{",
            "  int exitValue = 0;",
            "  return exitValue;",
            "}",
          ].join("\n"),
        };
      }

      const source = await response.text();

      return { source };
    },
  ],
  render: (args, { loaded: { source } }) => {
    return <EditorWithSyntaxTree {...args} initialValue={source} />;
  },
};
