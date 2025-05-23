import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './CourseInfo.css';

function CourseInfo() {
  const [courseName, setCourseName] = useState('Artificial Intelligence');
  const [description, setDescription] = useState('This course provides an in-depth introduction to the field of Artificial Intelligence (AI). It covers the foundational concepts, theories, and applications of AI, equipping students with the knowledge and skills needed to understand and develop AI systems');
  const [prerequisites, setPrerequisites] = useState('The course is designed for beginners with a basic understanding of computer science principles.');
  const [numLessons, setNumLessons] = useState('1');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const data = {
      course_name: courseName.trim(),
      course_description: description.trim(),
      prerequisites: prerequisites.trim(),
      number_of_lessons: numLessons.trim(),
    };

    try {
      const response = await axios.post('http://127.0.0.1:5000/generate_course_plan', data);
      console.log('Response:', response.data);

      navigate('/course-content', { state: { course: response.data.course } });
    } catch (err) {
      setError('Failed to submit the form. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-blue-100 flex items-center justify-center">
      <div className="bg-white w-full max-w-3xl p-16 rounded-lg shadow-lg">
        <h1 className="text-4xl font-bold text-gray-800 mb-8 text-center">Course Information</h1>
        <form onSubmit={handleSubmit}>
          <div className="relative mb-6">
            <input
              type="text"
              id="courseName"
              value={courseName}
              onChange={(e) => setCourseName(e.target.value)}
              className="input-field peer"
              required
            />
            <label htmlFor="courseName" className="input-label">
              Course Name
            </label>
          </div>
          <div className="relative mb-6">
            <textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="input-field peer"
              required
              rows="4"
            ></textarea>
            <label htmlFor="description" className="input-label">
              Description
            </label>
          </div>
          <div className="relative mb-6">
            <input
              type="text"
              id="prerequisites"
              value={prerequisites}
              onChange={(e) => setPrerequisites(e.target.value)}
              className="input-field peer"
            />
            <label htmlFor="prerequisites" className="input-label">
              Prerequisites (Optional)
            </label>
          </div>
          <div className="relative mb-8">
            <input
              type="number"
              id="numLessons"
              value={numLessons}
              onChange={(e) => setNumLessons(e.target.value)}
              className="input-field peer"
              required
            />
            <label htmlFor="numLessons" className="input-label">
              Number of Lessons
            </label>
          </div>
          <div className="flex items-center justify-center">
            <button
              type="submit"
              className="bg-blue-500 text-white py-3 px-8 rounded-lg hover:bg-blue-600 transition duration-200"
              disabled={loading}
            >
              {loading ? 'Submitting...' : 'Submit'}
            </button>
          </div>
          {error && <p className="text-red-500 text-center mt-4">{error}</p>}
        </form>
      </div>
    </div>
  );
}

export default CourseInfo;
