import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './components/HomePage';
import CourseInfo from './components/CourseInfo';
import CourseContent from './components/CourseContent';
import LessonList from './components/LessonList';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/course-info" element={<CourseInfo />} />
        <Route path="/course-content" element={<CourseContent />} />
        <Route path="/lesson-content" element={<LessonList />} />
      </Routes>
    </Router>
  );
}

export default App;
