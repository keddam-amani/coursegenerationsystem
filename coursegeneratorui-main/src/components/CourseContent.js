import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { useNavigate } from 'react-router-dom';

const ItemType = {
  LESSON: 'lesson',
  TOPIC: 'topic',
  SUBTOPIC: 'subtopic',
};

function Lesson({ lesson, index, moveLesson, setSelectedLesson, selectedLessonId }) {
  const ref = React.useRef(null);

  const [, drop] = useDrop({
    accept: ItemType.LESSON,
    hover(item) {
      if (item.index === index) return;

      moveLesson(item.index, index);
      item.index = index;
    },
  });

  const [{ isDragging }, drag] = useDrag({
    type: ItemType.LESSON,
    item: { index },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  drag(drop(ref));

  return (
    <li
      ref={ref}
      style={{
        opacity: isDragging ? 0.5 : 1,
        cursor: 'move',
        backgroundColor: selectedLessonId === lesson.id ? '#e3f2fd' : 'transparent',
      }}
      className="mb-2 text-gray-700 hover:bg-gray-100 p-2 rounded"
    >
      <button
        className="text-left w-full"
        onClick={() => setSelectedLesson(lesson)}
      >
        {index + 1}. {lesson.lesson_title}
      </button>
    </li>
  );
}

function Subtopic({ subtopic, subIndex, topicIndex, moveSubtopic }) {
  const ref = React.useRef(null);

  const [, drop] = useDrop({
    accept: ItemType.SUBTOPIC,
    hover(item) {
      if (item.subIndex === subIndex && item.topicIndex === topicIndex) return;

      moveSubtopic(item.topicIndex, topicIndex, item.subIndex, subIndex);
      item.subIndex = subIndex;
      item.topicIndex = topicIndex;
    },
  });

  const [{ isDragging }, drag] = useDrag({
    type: ItemType.SUBTOPIC,
    item: { subIndex, topicIndex },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  drag(drop(ref));

  return (
    <li
      ref={ref}
      style={{
        opacity: isDragging ? 0.5 : 1,
        cursor: 'move',
        backgroundColor: isDragging ? '#e0f7fa' : 'transparent',
      }}
      className="relative before:content-['▹'] before:absolute before:left-0 before:text-blue-500 pl-5 text-gray-700 rounded-md p-2"
    >
      {subtopic}
    </li>
  );
}

function Topic({ topic, index, moveTopic, moveSubtopic }) {
  const ref = React.useRef(null);

  const [, drop] = useDrop({
    accept: ItemType.TOPIC,
    hover(item) {
      if (item.index === index) return;

      moveTopic(item.index, index);
      item.index = index;
    },
  });

  const [{ isDragging }, drag] = useDrag({
    type: ItemType.TOPIC,
    item: { index },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  drag(drop(ref));

  return (
    <div
      ref={ref}
      style={{
        opacity: isDragging ? 0.5 : 1,
        cursor: 'move',
        backgroundColor: isDragging ? '#e3f2fd' : '#ffffff',
      }}
      className="bg-blue-50 p-4 rounded-lg mb-4 shadow"
    >
      <h3 className="text-xl font-semibold text-blue-800 mb-2">
        {index + 1}. {topic.title}
      </h3>
      <p className="text-gray-700 mb-4">{topic.description}</p>

      <h4 className="text-lg font-semibold text-blue-700 mb-2">Subtopics</h4>
      <ul className="list-inside pl-5 space-y-2">
        {topic.subtopics.map((subtopic, subIndex) => (
          <Subtopic
            key={uuidv4()}
            subtopic={subtopic}
            subIndex={subIndex}
            topicIndex={index}
            moveSubtopic={moveSubtopic}
          />
        ))}
      </ul>
    </div>
  );
}

function CourseContent() {
  const location = useLocation();
  const { course } = location.state || {};
  const [loading, setLoading] = useState(false);
  const [selectedLesson, setSelectedLesson] = useState(course ? course[0] : null);
  const navigate = useNavigate();

  const moveLesson = (from, to) => {
    const updatedLessons = [...course];
    const [movedLesson] = updatedLessons.splice(from, 1);
    updatedLessons.splice(to, 0, movedLesson);
    setSelectedLesson(updatedLessons[to]);
    location.state.course = updatedLessons;
  };

  const moveTopic = (from, to) => {
    const updatedTopics = [...selectedLesson.topics];
    const [movedTopic] = updatedTopics.splice(from, 1);
    updatedTopics.splice(to, 0, movedTopic);
    setSelectedLesson({ ...selectedLesson, topics: updatedTopics });
  };

  const moveSubtopic = (fromTopicIndex, toTopicIndex, fromSubIndex, toSubIndex) => {
    const updatedTopics = [...selectedLesson.topics];
    const sourceSubtopics = [...updatedTopics[fromTopicIndex].subtopics];
    const [movedSubtopic] = sourceSubtopics.splice(fromSubIndex, 1);

    if (fromTopicIndex === toTopicIndex) {
      sourceSubtopics.splice(toSubIndex, 0, movedSubtopic);
      updatedTopics[fromTopicIndex].subtopics = sourceSubtopics;
    } else {
      const destinationSubtopics = [...updatedTopics[toTopicIndex].subtopics];
      destinationSubtopics.splice(toSubIndex, 0, movedSubtopic);
      updatedTopics[fromTopicIndex].subtopics = sourceSubtopics;
      updatedTopics[toTopicIndex].subtopics = destinationSubtopics;
    }

    setSelectedLesson({ ...selectedLesson, topics: updatedTopics });
  };
  const handleGenerateContent = async () => {
    setLoading(true);

    const coursePlan = course.map((lesson) => ({
      id: lesson.id,
      lesson_title: lesson.lesson_title,
      description: lesson.description,
      learningObjectives: lesson.learningObjectives,
      topics: lesson.topics,
    }));

    const data = {
        course_plan: {
            course: coursePlan
        }
    };

    try {
      const response = await axios.post('http://127.0.0.1:5000/generate_lessons', data);
      console.log('Response:', response.data);
      navigate('/lesson-content', { state: { lessons: response.data } });
    } catch (error) {
      console.error('Error generating content:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!course) {
    return <div>No course data available.</div>;
  }

  return (
    <div className="min-h-screen bg-gray-100 flex">
      {/* Sidebar */}
      <aside className="w-1/4 bg-white p-4 border-r border-gray-200">
        <h2 className="text-xl font-bold mb-4">Table of Contents</h2>
        <DndProvider backend={HTML5Backend}>
          <ul>
            {course.map((lesson, index) => (
              <Lesson
                key={uuidv4()}
                index={index}
                lesson={lesson}
                moveLesson={moveLesson}
                setSelectedLesson={setSelectedLesson}
                selectedLessonId={selectedLesson?.id}
              />
            ))}
          </ul>
        </DndProvider>
        <div className="mt-4">
          <button
            className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
            onClick={handleGenerateContent}
            disabled={loading} // Disable the button while loading
          >
            {loading ? 'Generating...' : 'Generate Content'} {/* Show loading text */}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="w-3/4 p-8">
        {selectedLesson ? (
          <>
            <h1 className="text-3xl font-bold text-gray-900 mb-4">{selectedLesson.lesson_title}</h1>
            <p className="text-gray-700 mb-6">{selectedLesson.description}</p>

            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Learning Objectives</h2>
            <ul className="list-disc list-inside mb-6">
              {selectedLesson.learningObjectives.map((objective, index) => (
                <li key={index} className="text-gray-700">{objective}</li>
              ))}
            </ul>

            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Topics</h2>
            <DndProvider backend={HTML5Backend}>
              {selectedLesson.topics.map((topic, index) => (
                <Topic
                  key={uuidv4()}
                  index={index}
                  topic={topic}
                  moveTopic={moveTopic}
                  moveSubtopic={moveSubtopic}
                />
              ))}
            </DndProvider>
          </>
        ) : (
          <p>Select a lesson from the sidebar to view details.</p>
        )}
      </main>
    </div>
  );
}

export default CourseContent;
