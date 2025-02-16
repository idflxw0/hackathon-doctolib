import React from "react";
import { listePatients, Patient } from "../../types/list";

const PatientsList: React.FC = () => {
  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-200">
      <div className="w-full max-w-4xl p-6 bg-white shadow-xl rounded-lg">
        <h1 className="text-3xl font-bold text-gray-900 text-center mb-6">Liste des Patients</h1>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 bg-white">
            <thead>
              <tr className="bg-blue-600 text-white">
                <th className="py-3 px-4 text-left">ID</th>
                <th className="py-3 px-4 text-left">Rang</th>
                <th className="py-3 px-4 text-left">Nom</th>
                <th className="py-3 px-4 text-left">Prénom</th>
              </tr>
            </thead>
            <tbody>
              {listePatients.map((patient: Patient, index) => (
                <tr
                  key={patient.id}
                  className={index % 2 === 0 ? "bg-gray-100" : "bg-gray-300"}
                >
                  <td className="py-3 px-4 text-gray-900">{patient.id}</td>
                  <td className="py-3 px-4 text-gray-900">{patient.rank}</td>
                  <td className="py-3 px-4 text-gray-900">{patient.nom}</td>
                  <td className="py-3 px-4 text-gray-900">{patient.prenom}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default PatientsList;
