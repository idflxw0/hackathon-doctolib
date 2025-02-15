export interface Patient {
  id: number;
  rank: number;
  nom: string;
  prenom: string;
}

export const listePatients: Patient[] = [
  { id: 1, rank: 1, nom: "Dupont", prenom: "Jean" },
  { id: 2, rank: 2, nom: "Martin", prenom: "Sophie" },
  { id: 3, rank: 3, nom: "Durand", prenom: "Paul" },
  { id: 4, rank: 4, nom: "Lefebvre", prenom: "Marie" },
  { id: 5, rank: 5, nom: "Morel", prenom: "Thomas" }
];
