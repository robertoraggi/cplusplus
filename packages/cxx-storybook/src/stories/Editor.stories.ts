import type { Meta, StoryObj } from "@storybook/react";

import { Editor } from "../Editor";

// More on how to set up stories at: https://storybook.js.org/docs/react/writing-stories/introduction
const meta = {
  title: "Example/Editor",
  component: Editor,
  tags: ["autodocs"],
  argTypes: {},
} satisfies Meta<typeof Editor>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    value: [
      "#include <iostream>",
      "",
      "int main() {",
      '  std::cout << "Hello, world!" << std::endl;',
      "  return 0;",
      "}",
    ].join("\n"),
  },
};
