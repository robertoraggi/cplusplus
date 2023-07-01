import { FC } from "react";
import { Editor } from "./Editor";
import "./userWorker";
import "./App.css";

const App: FC = () => {
  return (
    <div className="App">
      <Editor />
    </div>
  );
};

export default App;
