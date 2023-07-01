import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';

// @ts-ignore
self.MonacoEnvironment = {
    getWorker(_workerId: string, _label: string) {
        return new editorWorker();
    },
};
