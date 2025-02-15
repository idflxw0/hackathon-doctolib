/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                blue: {
                    50: '#f0f7ff',
                    100: '#e0efff',
                    200: '#b9dfff',
                    300: '#7cc4ff',
                    400: '#36a9ff',
                    500: '#0090ff',
                    600: '#006fee',
                    700: '#0057cc',
                    800: '#0047a6',
                    900: '#003d87',
                },
            },
            animation: {
                'fade-in': 'fadeIn 0.3s ease-out forwards',
                'slide-in': 'slideIn 0.3s ease-out forwards',
                'slide-up': 'slideUp 0.3s ease-out forwards',
            },
            keyframes: {
                fadeIn: {
                    '0%': {
                        opacity: '0',
                        transform: 'translateY(10px)',
                    },
                    '100%': {
                        opacity: '1',
                        transform: 'translateY(0)',
                    },
                },
                slideIn: {
                    '0%': {
                        transform: 'translateX(-100%)',
                        opacity: '0',
                    },
                    '100%': {
                        transform: 'translateX(0)',
                        opacity: '1',
                    },
                },
                slideUp: {
                    '0%': {
                        transform: 'translateY(20px)',
                        opacity: '0',
                    },
                    '100%': {
                        transform: 'translateY(0)',
                        opacity: '1',
                    },
                },
            },
            boxShadow: {
                'message': '0 2px 8px rgba(0, 0, 0, 0.06)',
            },
            borderRadius: {
                'message': '1.5rem',
            },
            maxHeight: {
                'chat': '32rem',
            },
        },
    },
    plugins: [],
}