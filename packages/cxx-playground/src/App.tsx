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

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ASTProvider from "./ast-provider";
import CxxProvider from "./cxx-provider";
import Editor from "./editor";
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import ModelProvider from "./editor-model-provider";
import EditorProvider from "./editor-provider";
import AbstractSyntaxTree from "./abstract-syntax-tree";

import * as monaco from "monaco-editor";

window.MonacoEnvironment = {
  getWorker() {
    return new EditorWorker();
  },
};

const queryClient = new QueryClient();

const DEFAULT_VALUE = `#include <iostream>

auto main() -> int {
  std::cout << "Hello, World!" << std::endl;
  return 0;
}`;

function App() {
  const interval = 250;

  const model = monaco.editor.createModel(DEFAULT_VALUE, "cpp");

  return (
    <div className="w-svw h-svh flex">
      <QueryClientProvider client={queryClient}>
        <CxxProvider fallback={<div />}>
          <ModelProvider model={model}>
            <EditorProvider model={model}>
              <ASTProvider model={model} interval={interval}>
                <Editor />
                <AbstractSyntaxTree />
              </ASTProvider>
            </EditorProvider>
          </ModelProvider>
        </CxxProvider>
      </QueryClientProvider>
    </div>
  );
}

export default App;
