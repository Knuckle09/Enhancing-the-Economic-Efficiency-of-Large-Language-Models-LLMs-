import React, { useState } from 'react';
import { 
  ArrowLeft, 
  Linkedin, 
  Github, 
  Mail, 
  MapPin, 
  Calendar,
  Award,
  Users,
  Coffee,
  Code,
  Briefcase,
  Star
} from 'lucide-react';
import FarzanaImg from '../assets/Farzana.jpg';  
import SamarthImg from '../assets/Samarth.jpg';
import Student1Img from '../assets/Student1.jpg';
import Student2Img from '../assets/Student2.jpg';
import Student4Img from '../assets/Student4.jpg';


// Real Team Data
const teamData = {
  guide: {
    id: 'guide-1',
    name: 'Professor Farzana Nadaf',
    role: 'Project Guide & Mentor',
    bio: 'This project was completed under the mentorship of Professor Farzana Nadaf, whose expertise in artificial intelligence, natural language processing, and optimization played a pivotal role in shaping its success. Her guidance ensured a strong balance between theoretical depth and practical implementation, while her dedication to innovation and research excellence inspired the team to deliver meaningful and impactful work.',
    image: FarzanaImg,
    location: 'Hubballi, Karnataka',
    joinDate: 'Project Inception',
    specialties: ['Artificial Intelligence', 'Natural Language Processing', 'Optimization'],
    achievements: [
      'Expert in AI and NLP research',
      'Mentored numerous successful projects',
      'Dedicated to innovation and research excellence'
    ],
    socials: {
      linkedin: 'https://www.linkedin.com/in/farzana-nadaf-a1794918',
      email: 'farzana@university.edu'
    },
    stats: {
      'Research Area': 'AI/NLP',
      'Experience': '10+ Years',
      'Projects Guided': '50+'
    }
  },
  students: [
    {
      id: 'student-1',
      name: 'Sai Samarth S Budihal',
      role: 'Team Lead & Full-Stack Developer',
      bio: 'Sai Samarth leads the team and owns the entire frontend development. He implemented the LLMEfficiencyTest class, integrated LLMs (local or API), semantic similarity models, and token counting. He developed prompt optimization, similarity calculation, and efficiency metrics while building evaluation pipelines and visualizing test results.',
      image: SamarthImg,
      location: 'Hubballi, Karnataka',
      joinDate: 'Project Start',
      specialties: ['Frontend Development', 'LLM Integration', 'Efficiency Testing'],
      socials: {
        linkedin: 'https://www.linkedin.com/in/sai-samarth-budihal-a0466b258',
        github: 'https://github.com/Knuckle09',
        email: 'saisamarth@university.edu'
      },
      stats: {
        'Role': 'Team Lead',
        'Focus': 'Full-Stack',
        'Integration': 'LLM APIs'
      }
    },
    {
      id: 'student-2',
      name: 'Sughnva S Chappar',
      role: 'RL Pipeline & Environment Developer',
      bio: 'Sughnva built the RL environment (TextOptimizationEnv) for prompt optimization and implemented RL agent training (RLOptimizer) using PPO and Stable-Baselines3. He integrated the environment with LLM efficiency tester, developed reward functions and state/action representations, and documented the RL pipeline with beginner-friendly explanations.',
      image: Student2Img,
      location: 'Hubballi, Karnataka',
      joinDate: 'Project Start',
      specialties: ['Reinforcement Learning', 'PPO', 'Environment Design'],
      socials: {
        linkedin: 'https://www.linkedin.com/in/sughnva-chappar-67a54034a',
        github: 'https://github.com/Sughnva4523',
        email: 'sughnva@university.edu'
      },
      stats: {
        'Focus': 'RL Pipeline',
        'Framework': 'PPO/SB3',
        'Contribution': 'Core RL'
      }
    },
    {
      id: 'student-3',
      name: 'Suprit R Mundagod',
      role: 'Data & Prompt Management / Testing Lead',
      bio: 'Suprit manages prompt datasets (CSV loading, sampling, and organization) and implements prompt diversity and RL readiness tests. He generates and curates training data for RL, analyzes and reports on test results, and maintains scripts for dataset conversion. He also documents the data/test pipeline comprehensively.',
      image: Student1Img,
      location: 'Hubballi, Karnataka',
      joinDate: 'Project Start',
      specialties: ['Data Management', 'Testing', 'Dataset Curation'],
      socials: {
        linkedin: 'https://www.linkedin.com/in/suprit-mundagod-2a4939323',
        github: 'https://github.com/supritmundagod',
        email: 'suprit@university.edu'
      },
      stats: {
        'Role': 'Testing Lead',
        'Focus': 'Data Pipeline',
        'Datasets': 'Multiple'
      }
    },
    {
      id: 'student-4',
      name: 'Vishwanath B Kotyal',
      role: 'Text Preprocessing Specialist',
      bio: 'Vishwanath designed and implemented the TextPreprocessor class and developed domain-specific optimizers (math, coding, business, generic). He writes and maintains preprocessing logic including tokenization, stopwords, lemmatization, and synonym finding, while ensuring semantic preservation and providing clear documentation.',
      image: Student4Img,
      location: 'Hubballi, Karnataka',
      joinDate: 'Project Start',
      specialties: ['Text Processing', 'NLP', 'Domain Optimization'],
      socials: {
        linkedin: 'https://www.linkedin.com/in/vishwanath-kotyal-1b67b6385',
        github: 'https://github.com/vishwanathkotyal674-cpu',
        email: 'vishwanath@university.edu'
      },
      stats: {
        'Focus': 'Preprocessing',
        'Optimizers': '4 Domains',
        'NLP': 'Expert'
      }
    }
  ]
};

export default function AboutUsPage({ onBackToChat }) {
  const [selectedMember, setSelectedMember] = useState(null);

  const getSocialIcon = (platform) => {
    switch (platform) {
      case 'linkedin': return <Linkedin className="w-5 h-5" />;
      case 'github': return <Github className="w-5 h-5" />;
      case 'email': return <Mail className="w-5 h-5" />;
      default: return null;
    }
  };

  const renderMemberCard = (member, isGuide = false) => (
    <div 
      key={member.id}
      className={`
        relative group cursor-pointer transition-all duration-300 hover:scale-105
        ${isGuide ? 'lg:col-span-2' : ''}
      `}
      onClick={() => setSelectedMember(member)}
    >
      <div className={`
        bg-gray-800 rounded-2xl overflow-hidden shadow-2xl border border-gray-700
        hover:border-purple-500 transition-all duration-300
        ${isGuide ? 'p-8' : 'p-6'}
      `}>
        {isGuide && (
          <div className="absolute top-4 right-4 bg-gradient-to-r from-yellow-400 to-orange-500 text-gray-900 text-xs font-bold px-3 py-1 rounded-full flex items-center space-x-1">
            <Star className="w-3 h-3" />
            <span>GUIDE</span>
          </div>
        )}

        <div className={`flex ${isGuide ? 'flex-col lg:flex-row lg:items-center lg:space-x-6' : 'flex-col items-center'} space-y-4 ${isGuide ? 'lg:space-y-0' : ''}`}>
          <div className={`relative ${isGuide ? 'w-32 h-32 lg:w-40 lg:h-40' : 'w-24 h-24'} flex-shrink-0`}>
            <img
              src={member.image}
              alt={member.name}
              className="w-full h-full rounded-full object-cover border-4 border-purple-500"
            />
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-500/20 to-purple-500/20 group-hover:from-blue-500/30 group-hover:to-purple-500/30 transition-all duration-300" />
          </div>

          <div className={`flex-1 ${isGuide ? 'text-left lg:text-left' : 'text-center'}`}>
            <h3 className={`font-bold text-gray-100 mb-2 ${isGuide ? 'text-2xl lg:text-3xl' : 'text-xl'}`}>
              {member.name}
            </h3>
            <p className={`text-transparent bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text font-semibold mb-3 ${isGuide ? 'text-lg' : 'text-base'}`}>
              {member.role}
            </p>
            <p className={`text-gray-400 leading-relaxed mb-4 ${isGuide ? 'text-base' : 'text-sm'}`}>
              {member.bio.length > 120 && !isGuide ? `${member.bio.substring(0, 120)}...` : member.bio}
            </p>

            <div className={`grid grid-cols-3 gap-2 mb-4 ${isGuide ? 'text-base' : 'text-sm'}`}>
              {Object.entries(member.stats).map(([key, value]) => (
                <div key={key} className="bg-gray-700 rounded-lg p-2 text-center">
                  <div className="text-blue-400 font-bold">{value}</div>
                  <div className="text-gray-400 text-xs">{key}</div>
                </div>
              ))}
            </div>

            <div className="flex justify-center space-x-3">
              {Object.entries(member.socials).map(([platform, url]) => (
                <a
                  key={platform}
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 bg-gray-700 hover:bg-gray-600 rounded-full transition-colors duration-200 group"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="text-gray-400 group-hover:text-blue-400 transition-colors">
                    {getSocialIcon(platform)}
                  </div>
                </a>
              ))}
            </div>
          </div>
        </div>

        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/5 to-purple-600/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl pointer-events-none" />
      </div>
    </div>
  );

  const renderDetailModal = () => {
    if (!selectedMember) return null;

    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-gray-800 rounded-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto border border-gray-700">
          <div className="p-6">
            <button
              onClick={() => setSelectedMember(null)}
              className="float-right p-2 hover:bg-gray-700 rounded-full transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-400" />
            </button>

            <div className="text-center mb-6">
              <img
                src={selectedMember.image}
                alt={selectedMember.name}
                className="w-32 h-32 rounded-full object-cover border-4 border-purple-500 mx-auto mb-4"
              />
              <h2 className="text-2xl font-bold text-gray-100 mb-2">{selectedMember.name}</h2>
              <p className="text-transparent bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text font-semibold mb-4">
                {selectedMember.role}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className="space-y-4">
                <div className="flex items-center space-x-3 text-gray-300">
                  <MapPin className="w-5 h-5 text-blue-400" />
                  <span>{selectedMember.location}</span>
                </div>
                <div className="flex items-center space-x-3 text-gray-300">
                  <Calendar className="w-5 h-5 text-green-400" />
                  <span>Joined {selectedMember.joinDate}</span>
                </div>
              </div>
              <div>
                <h4 className="font-semibold text-gray-200 mb-2">Specialties</h4>
                <div className="flex flex-wrap gap-2">
                  {selectedMember.specialties.map((specialty) => (
                    <span key={specialty} className="bg-purple-600/20 text-purple-300 px-3 py-1 rounded-full text-sm">
                      {specialty}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="mb-6">
              <h4 className="font-semibold text-gray-200 mb-3">About</h4>
              <p className="text-gray-400 leading-relaxed">{selectedMember.bio}</p>
            </div>

            {selectedMember.achievements && (
              <div className="mb-6">
                <h4 className="font-semibold text-gray-200 mb-3 flex items-center space-x-2">
                  <Award className="w-5 h-5 text-yellow-400" />
                  <span>Key Achievements</span>
                </h4>
                <ul className="space-y-2">
                  {selectedMember.achievements.map((achievement, index) => (
                    <li key={index} className="text-gray-400 flex items-start space-x-2">
                      <Star className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                      <span>{achievement}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex justify-center space-x-4">
              {Object.entries(selectedMember.socials).map(([platform, url]) => (
                <a
                  key={platform}
                  href={url}
                  className="p-3 bg-gray-700 hover:bg-gray-600 rounded-full transition-colors duration-200 group"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <div className="text-gray-400 group-hover:text-blue-400 transition-colors">
                    {getSocialIcon(platform)}
                  </div>
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <div className="bg-gray-800 border-b border-gray-700 p-4 lg:p-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-3xl lg:text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              Meet Our Team
            </h1>
            <p className="text-gray-400">
              The brilliant minds behind LLM Prompt Optimization
            </p>
          </div>
          <button
            onClick={onBackToChat}
            className="bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back</span>
          </button>
        </div>
      </div>

      <div className="bg-gradient-to-r from-blue-600/10 to-purple-600/10 border-b border-gray-700 p-6">
        <div className="max-w-7xl mx-auto grid grid-cols-2 lg:grid-cols-4 gap-6 text-center">
          <div>
            <Users className="w-8 h-8 text-blue-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-100">5</div>
            <div className="text-gray-400 text-sm">Team Members</div>
          </div>
          <div>
            <Code className="w-8 h-8 text-green-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-100">4</div>
            <div className="text-gray-400 text-sm">Core Components</div>
          </div>
          <div>
            <Briefcase className="w-8 h-8 text-purple-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-100">1</div>
            <div className="text-gray-400 text-sm">Major Project</div>
          </div>
          <div>
            <Coffee className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-100">∞</div>
            <div className="text-gray-400 text-sm">Coffee Consumed</div>
          </div>
        </div>
      </div>

      <div className="p-6 lg:p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-12">
            <h2 className="text-2xl font-bold text-gray-200 mb-6 text-center lg:text-left">Project Guide</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {renderMemberCard(teamData.guide, true)}
            </div>
          </div>

          <div>
            <h2 className="text-2xl font-bold text-gray-200 mb-6 text-center lg:text-left">Student Team</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {teamData.students.map(student => renderMemberCard(student))}
            </div>
          </div>
        </div>
      </div>

      {renderDetailModal()}
    </div>
  );
}