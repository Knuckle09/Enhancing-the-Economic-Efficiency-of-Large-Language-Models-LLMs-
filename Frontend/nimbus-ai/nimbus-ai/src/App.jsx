// App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Pages/Home.jsx';
import Aboutme from './Pages/Aboutme.jsx';

function App() {
  return (
    <Router basename={import.meta.env.BASE_URL}>
      <div className="w-full min-h-screen">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<Aboutme />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
