import React from 'react';
import { useNavigate } from 'react-router-dom';

function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-blue-100 flex items-center justify-center">
      <div className="bg-white w-full max-w-xl p-16 rounded-lg shadow-lg text-center">
        <h1 className="text-4xl font-bold text-gray-800 mb-6">Welcome to the Course Generator</h1>
        <p className="text-gray-600 mb-8">Please follow the steps to generate your individual course.</p>
        <button
          onClick={() => navigate('/course-info')}
          className="bg-blue-500 text-white py-3 px-8 rounded-full hover:bg-blue-600 transition duration-200"
        >
          Get Started
        </button>
      </div>
    </div>
  );
}

export default HomePage;
