import { useState } from 'react';
import { Chat } from '@/components/Chat/Chat';
import {
    Search,
    MapPin,
    Calendar,
    Heart,
    Star,
    ArrowRight,
    CheckCircle,
    AlertCircle,
} from 'lucide-react';

const LandingPage = () => {
    const [isChatOpen, setIsChatOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [location, setLocation] = useState('');

    const features = [
        {
            icon: <Calendar className="w-6 h-6" />,
            title: "Prise de rendez-vous simplifiée",
            description: "Planifiez vos consultations en quelques clics, 24/7"
        },
        {
            icon: <Heart className="w-6 h-6" />,
            title: "Suivi personnalisé",
            description: "Une prise en charge adaptée à vos besoins spécifiques"
        },
        {
            icon: <Star className="w-6 h-6" />,
            title: "Professionnels qualifiés",
            description: "Un réseau de spécialistes certifiés à votre service"
        }
    ];

    const stats = [
        { number: "98%", label: "Satisfaction client" },
        { number: "24/7", label: "Support disponible" },
        { number: "1000+", label: "Professionnels" },
    ];



    return (
        <div className="min-h-screen bg-gradient-to-b from-white to-blue-50">
            {/* Navbar */}
            <nav className="fixed top-0 w-full z-40 bg-white/80 backdrop-blur-md border-b border-gray-100">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="flex items-center justify-between h-20">
                        <img src="/logo-blue.png" alt="Logo" className="h-8" />
                        <div className="hidden md:flex items-center gap-8">
                            <a href="#" className="text-gray-600 hover:text-blue-600 transition-colors">Accueil</a>
                            <a href="#" className="text-gray-600 hover:text-blue-600 transition-colors">Services</a>
                            <a href="#" className="text-gray-600 hover:text-blue-600 transition-colors">Professionnels</a>
                            <button
                                onClick={() => setIsChatOpen(true)}
                                className="flex items-center gap-2 text-red-600 hover:text-red-700 transition-colors"
                            >
                                <AlertCircle className="w-5 h-5" />
                                <span>Urgence</span>
                            </button>
                        </div>
                        <button className="bg-gray-200 text-gray-600 px-4 py-2 rounded-full hover:bg-gray-300 transition-colors flex items-center shadow-md">
                            <img src="/profile.webp" alt="Profile" className="w-8 h-8 rounded-full border-2 border-gray-300" />
                            <span className="ml-2 text-lg font-semibold">Profile</span>
                        </button>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="pt-32 pb-24">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="grid lg:grid-cols-2 gap-16 items-center">
                        <div className="space-y-8">
                            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-full">
                                <CheckCircle className="w-4 h-4 text-blue-600" />
                                <span className="text-sm text-blue-600 font-medium">Plateforme certifiée santé</span>
                            </div>

                            <div className="space-y-6">
                                <h1 className="text-5xl lg:text-6xl font-bold text-gray-900">
                                    Votre santé, notre priorité
                                </h1>
                                <p className="text-xl text-gray-600 leading-relaxed">
                                    Accédez à des soins de qualité et à un suivi personnalisé grâce à notre réseau de professionnels de santé qualifiés.
                                </p>
                            </div>

                            <div className="bg-white rounded-2xl shadow-xl p-6 space-y-4">
                                <div className="flex items-center gap-3 bg-gray-50 rounded-xl p-4">
                                    <Search className="w-5 h-5 text-gray-400" />
                                    <input
                                        type="text"
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        placeholder="Rechercher un spécialiste..."
                                        className="w-full bg-transparent focus:outline-none text-gray-700"
                                    />
                                </div>

                                <div className="flex items-center gap-3 bg-gray-50 rounded-xl p-4">
                                    <MapPin className="w-5 h-5 text-gray-400" />
                                    <input
                                        type="text"
                                        value={location}
                                        onChange={(e) => setLocation(e.target.value)}
                                        placeholder="Localisation"
                                        className="w-full bg-transparent focus:outline-none text-gray-700"
                                    />
                                </div>

                                <button className="w-full bg-blue-600 text-white p-4 rounded-xl font-medium hover:bg-blue-700 transition-colors flex items-center justify-center gap-2 group">
                                    <span>Trouver un professionnel</span>
                                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                                </button>
                            </div>

                            <div className="flex items-center gap-8">
                                {stats.map((stat, index) => (
                                    <div key={index} className="space-y-1">
                                        <div className="text-2xl font-bold text-blue-600">{stat.number}</div>
                                        <div className="text-sm text-gray-500">{stat.label}</div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="hidden lg:block relative">
                            <div className="absolute inset-0 bg-gradient-to-br from-blue-600 to-blue-400 rounded-3xl transform rotate-3"></div>
                            <img
                                src="/happyDoctor.jpg"
                                alt="Healthcare Professional"
                                className="relative rounded-3xl shadow-xl object-cover w-full h-full transform -rotate-3 transition-transform hover:rotate-0 duration-500"
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-24 bg-white">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-gray-900 mb-4">Nos services</h2>
                        <p className="text-gray-600 max-w-2xl mx-auto">
                            Découvrez comment notre plateforme transforme l'accès aux soins de santé
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {features.map((feature, index) => (
                            <div
                                key={index}
                                className="group p-8 rounded-2xl bg-white border border-gray-100 hover:border-blue-100 hover:shadow-lg transition-all duration-300"
                            >
                                <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center text-blue-600 mb-6 group-hover:scale-110 transition-transform">
                                    {feature.icon}
                                </div>
                                <h3 className="text-xl font-semibold text-gray-900 mb-4">{feature.title}</h3>
                                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>



            {isChatOpen && (
                <Chat onClose={() => setIsChatOpen(false)} />
            )}

        </div>
    );
};

export default LandingPage;