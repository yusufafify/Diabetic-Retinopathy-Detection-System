import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import ScanPage from "./pages/ScanPage";
import ResultsPage from "./pages/ResultPage";

function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/scan" element={<ScanPage />} />
          <Route path="/scan/results" element={<ResultsPage />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
