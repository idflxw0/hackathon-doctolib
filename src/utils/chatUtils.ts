import { BotResponse, ConversationContext } from '@/types/chat';

export const generateResponse = (userMessage: string, context?: ConversationContext): BotResponse => {
    const message = userMessage.toLowerCase();

    // Headache-specific flow
    if (message.includes('mal à la tête') || message.includes('migraine') || context?.currentSymptom === 'headache') {
        return {
            response: context?.initialMessage ?
                "D'après ce que vous me dites sur votre mal de tête, j'aimerais en savoir plus :" :
                "Je comprends que vous avez mal à la tête. Je vais vous poser quelques questions pour mieux comprendre votre situation.",
            context: {
                currentSymptom: 'headache',
                initialMessage: false
            },
            suggestions: [
                {
                    question: "Où est localisée la douleur ?",
                    answer: "La localisation de la douleur peut nous aider à identifier la cause :\n- Front : possible tension ou sinusite\n- Un côté : possible migraine\n- Arrière de la tête : possible tension cervicale\n\nEn attendant de consulter :\n- Reposez-vous dans un endroit calme\n- Évitez les écrans et la lumière vive\n- Restez hydraté\n\nConsultez si la douleur est inhabituelle ou persiste."
                },
                {
                    question: "La douleur est-elle pulsatile ?",
                    answer: "Une douleur pulsatile suggère une possible migraine. Voici ce que vous pouvez faire :\n\n- Isolez-vous dans une pièce sombre et calme\n- Appliquez une compresse froide\n- Prenez un antalgique si vous en avez\n- Évitez tout effort physique\n\nSi c'est votre première migraine ou si elle est particulièrement intense, consultez un médecin."
                },
                {
                    question: "Y a-t-il des facteurs qui aggravent la douleur ?",
                    answer: "Certains facteurs peuvent aggraver les maux de tête :\n\n- Lumière/bruit : typique des migraines\n- Mouvements : possible problème cervical\n- Effort physique : attention particulière nécessaire\n\nNotez ces facteurs pour en parler à votre médecin. En attendant :\n- Évitez les facteurs déclenchants identifiés\n- Maintenez un rythme de sommeil régulier\n- Gérez votre stress"
                },
                {
                    question: "Avez-vous d'autres symptômes associés ?",
                    answer: "Les symptômes associés sont importants :\n\n- Nausées/vomissements : possible migraine\n- Troubles visuels : consultation rapide nécessaire\n- Fièvre : possible infection\n\nEn cas de :\n- Perte d'équilibre\n- Vision double\n- Confusion\n→ Consultez immédiatement les urgences"
                }
            ]
        };
    }

    // Fever-specific flow
    if (message.includes('fièvre') || message.includes('température') || context?.currentSymptom === 'fever') {
        // ... similar structure for fever
    }

    // Default fallback if no context or unrecognized symptom
    return {
        response: "Je vois. Pour mieux vous aider, pourriez-vous me donner plus de détails sur vos symptômes ?",
        context: {
            currentSymptom: 'unknown',
            initialMessage: true
        },
        suggestions: [
            {
                question: "Depuis quand avez-vous ces symptômes ?",
                answer: "La durée est un facteur important. Consultez si :\n- Les symptômes s'aggravent\n- Durent plus d'une semaine\n- Sont accompagnés de fièvre"
            },
            {
                question: "Avez-vous pris des médicaments ?",
                answer: "Avant toute automédication :\n- Consultez un professionnel\n- Lisez les notices\n- Respectez les doses"
            }
        ]
    };
};