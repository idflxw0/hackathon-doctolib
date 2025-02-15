import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { Chat } from "./components/Chat/Chat";
import PatientsList from "./components/List/patientsList"; // Assure-toi que ce fichier existe

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/chat" element={<Chat />} />
        <Route path="/list" element={<PatientsList />} />
        {/* Redirection automatique de la racine vers /chat */}
        <Route path="*" element={<Navigate to="/chat" />} />
      </Routes>
    </Router>
  );
}

export default App;
