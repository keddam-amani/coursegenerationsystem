import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { FaSyncAlt, FaExpandAlt, FaCompressAlt, FaSpinner, FaTimes } from 'react-icons/fa';

const convertLessonToMarkdown = (lesson) => {
    let markdownContent = `# ${lesson.lesson_title}\n\n`;
    markdownContent += `${lesson.lesson_description}\n\n`;
    markdownContent += `## Learning Objectives\n`;
    lesson.learning_objectives.forEach((objective) => {
        markdownContent += `- ${objective}\n`;
    });
    markdownContent += `\n## Topics\n`;
    lesson.topics.forEach((topic, index) => {
        markdownContent += `### ${index + 1}. ${topic.title}\n`;
        markdownContent += `${topic.content}\n\n`;
        if (topic.subtopics) {
            topic.subtopics.forEach((subtopic, subIndex) => {
                markdownContent += `#### ${index + 1}.${subIndex + 1} ${subtopic.title}\n`;
                markdownContent += `${subtopic.content}\n\n`;
            });
        }
    });
    return markdownContent;
};

const downloadMarkdown = (lesson) => {
    const markdownContent = convertLessonToMarkdown(lesson);
    const blob = new Blob([markdownContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${lesson.lesson_title}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

const Sidebar = ({ lessons, selectedLesson, setSelectedLesson }) => (
    <aside className="w-1/4 bg-white p-6 border-r border-gray-200 overflow-y-auto h-screen">
        <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-gray-800">Lessons</h2>
            {selectedLesson && (
                <button
                    onClick={() => downloadMarkdown(selectedLesson)}
                    className="bg-black text-white px-4 py-2 rounded hover:bg-gray-800 transition duration-200"
                >
                    Download as MD
                </button>
            )}
        </div>
        <ul className="space-y-4">
            {lessons.map((lesson, index) => (
                <li key={lesson.id}>
                    <div
                        className={`p-3 rounded-lg cursor-pointer transition-all duration-200 ${selectedLesson?.id === lesson.id
                            ? 'bg-blue-100 text-blue-700'
                            : 'text-gray-700 hover:bg-gray-100'
                            }`}
                        onClick={() => setSelectedLesson(lesson)}
                    >
                        <span className="font-semibold">{index + 1}. {lesson.lesson_title}</span>
                    </div>
                    <ul className="ml-4 space-y-2">
                        {lesson.topics.map((topic, topicIndex) => (
                            <li key={topicIndex} className="text-sm text-gray-600">
                                <a
                                    href={`#topic-${topicIndex}`}
                                    onClick={() => setSelectedLesson(lesson)}
                                    className="font-semibold hover:text-blue-500"
                                >
                                    • {topic.title}
                                </a>
                                {topic.subtopics && topic.subtopics.length > 0 && (
                                    <ul className="ml-4 space-y-1">
                                        {topic.subtopics.map((subtopic, subtopicIndex) => (
                                            <li key={subtopicIndex} className="text-xs text-gray-500">
                                                <a
                                                    href={`#subtopic-${topicIndex}-${subtopicIndex}`}
                                                    onClick={() => setSelectedLesson(lesson)}
                                                    className="hover:text-blue-400"
                                                >
                                                    {subtopic.title}
                                                </a>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </li>
                        ))}
                    </ul>
                </li>
            ))}
        </ul>
    </aside>
);

const LessonContent = ({ lesson, updateContent }) => {
    const [loadingState, setLoadingState] = useState({});
    const [factCheckResults, setFactCheckResults] = useState({});
    const [isPopupOpen, setIsPopupOpen] = useState({});

    const handleAction = (actionType, topicIndex, subtopicIndex = null) => {
        const key = subtopicIndex !== null ? `${topicIndex}-${subtopicIndex}-${actionType}` : `${topicIndex}-${actionType}`;
        setLoadingState(prev => ({ ...prev, [key]: true }));

        const contentToSend = subtopicIndex !== null ? lesson.topics[topicIndex].subtopics[subtopicIndex].content : lesson.topics[topicIndex].content;

        const endpointMap = {
            regenerate: 'http://127.0.0.1:5000/regenerate_topic',
            expand: 'http://127.0.0.1:5000/expand_topic',
            shorten: 'http://127.0.0.1:5000/shorten_topic',
        };

        fetch(endpointMap[actionType], {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ content: contentToSend }),
        })
            .then(response => response.json())
            .then(data => {
                console.log(`${actionType} action successful:`, data);
                updateContent(lesson.id, topicIndex, subtopicIndex, data.content);
            })
            .catch(error => {
                console.error(`${actionType} action failed:`, error);
            })
            .finally(() => {
                setLoadingState(prev => ({ ...prev, [key]: false }));
            });
    };

    const handleCheckFacts = (topicIndex) => {
        const key = `${topicIndex}-fact_check`;
        setLoadingState(prev => ({ ...prev, [key]: true }));

        const topicData = lesson.topics[topicIndex];

        fetch('http://127.0.0.1:5000/fact_checking', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(topicData),
        })
            .then(response => response.json())
            .then(data => {
                console.log(`Fact check for topic ${topicIndex} successful:`, data);
                setFactCheckResults(prev => ({ ...prev, [topicIndex]: data }));
                setIsPopupOpen(prev => ({ ...prev, [topicIndex]: true }));
            })
            .catch(error => {
                console.error(`Fact check for topic ${topicIndex} failed:`, error);
            })
            .finally(() => {
                setLoadingState(prev => ({ ...prev, [key]: false }));
            });
    };

    const togglePopup = (topicIndex) => {
        setIsPopupOpen(prev => ({ ...prev, [topicIndex]: !prev[topicIndex] }));
    };

    return (
        <div className="space-y-8">
            <h1 className="text-4xl font-bold text-gray-900">{lesson.lesson_title}</h1>
            <ReactMarkdown
                className="text-gray-700 leading-relaxed"
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeKatex]}
            >
                {lesson.lesson_description}
            </ReactMarkdown>

            <section>
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">Learning Objectives</h2>
                <ul className="list-none space-y-2">
                    {lesson.learning_objectives.map((objective, index) => (
                        <li key={index} className="flex items-start text-gray-700">
                            <span className="text-blue-600 mr-2 mt-1">✔</span>
                            <span>
                                <ReactMarkdown
                                    remarkPlugins={[remarkGfm]}
                                    rehypePlugins={[rehypeKatex]}
                                    className="inline"
                                >
                                    {objective}
                                </ReactMarkdown>
                            </span>
                        </li>
                    ))}
                </ul>
            </section>

            <section>
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">Topics</h2>
                {lesson.topics.map((topic, index) => (
                    <div key={index} className="mb-8 relative" id={`topic-${index}`}>
                        <div className="bg-blue-50 p-6 rounded-lg shadow-md relative">
                            <div className="flex items-center mb-4 justify-between">
                                <h3 className="text-xl font-semibold text-blue-700">
                                    {index + 1}. {topic.title}
                                </h3>
                                <CheckFactsButton
                                    onClick={() => handleCheckFacts(index)}
                                    isLoading={loadingState[`${index}-fact_check`]}
                                />
                            </div>
                            <div className="flex space-x-4">
                                <ActionButton
                                    icon={loadingState[`${index}-regenerate`] ? FaSpinner : FaSyncAlt}
                                    label="Regenerate Text"
                                    onClick={() => handleAction('regenerate', index)}
                                    isLoading={loadingState[`${index}-regenerate`]}
                                />
                                <ActionButton
                                    icon={loadingState[`${index}-expand`] ? FaSpinner : FaExpandAlt}
                                    label="Expand Text"
                                    onClick={() => handleAction('expand', index)}
                                    isLoading={loadingState[`${index}-expand`]}
                                />
                                <ActionButton
                                    icon={loadingState[`${index}-shorten`] ? FaSpinner : FaCompressAlt}
                                    label="Shorten Text"
                                    onClick={() => handleAction('shorten', index)}
                                    isLoading={loadingState[`${index}-shorten`]}
                                />
                                <button
                                    onClick={() => togglePopup(index)}
                                    className="ml-auto bg-green-600 text-white text-xs font-semibold px-4 py-2 rounded-md hover:bg-green-700 transition-colors"
                                >
                                    {isPopupOpen[index] ? "Hide Facts" : "Show Facts"}
                                </button>
                            </div>
                            <ReactMarkdown
                                className="text-gray-700 mb-6 leading-relaxed"
                                remarkPlugins={[remarkGfm]}
                                rehypePlugins={[rehypeKatex]}
                            >
                                {topic.content}
                            </ReactMarkdown>

                            {topic.subtopics && topic.subtopics.length > 0 && (
                                <div>
                                    <h4 className="text-lg font-semibold text-blue-600 mb-3">Subtopics</h4>
                                    <ul className="space-y-4">
                                        {topic.subtopics.map((subtopic, subIndex) => (
                                            <li key={subIndex} className="relative" id={`subtopic-${index}-${subIndex}`}>
                                                <div className="flex items-center bg-blue-100 p-2 rounded-md">
                                                    <h5 className="font-semibold text-gray-800">
                                                        {subtopic.title}
                                                    </h5>
                                                    <div className="flex space-x-4 ml-4">
                                                        <ActionButton
                                                            icon={loadingState[`${index}-${subIndex}-regenerate`] ? FaSpinner : FaSyncAlt}
                                                            label="Regenerate Text"
                                                            onClick={() => handleAction('regenerate', index, subIndex)}
                                                            isLoading={loadingState[`${index}-${subIndex}-regenerate`]}
                                                        />
                                                        <ActionButton
                                                            icon={loadingState[`${index}-${subIndex}-expand`] ? FaSpinner : FaExpandAlt}
                                                            label="Expand Text"
                                                            onClick={() => handleAction('expand', index, subIndex)}
                                                            isLoading={loadingState[`${index}-${subIndex}-expand`]}
                                                        />
                                                        <ActionButton
                                                            icon={loadingState[`${index}-${subIndex}-shorten`] ? FaSpinner : FaCompressAlt}
                                                            label="Shorten Text"
                                                            onClick={() => handleAction('shorten', index, subIndex)}
                                                            isLoading={loadingState[`${index}-${subIndex}-shorten`]}
                                                        />
                                                    </div>
                                                </div>
                                                <ReactMarkdown
                                                    className="text-gray-600 leading-relaxed mt-2 ml-4"
                                                    remarkPlugins={[remarkGfm]}
                                                    rehypePlugins={[rehypeKatex]}
                                                >
                                                    {subtopic.content}
                                                </ReactMarkdown>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {isPopupOpen[index] && factCheckResults[index] && (
                                <FactCheckPopup
                                    facts={factCheckResults[index]}
                                    onClose={() => togglePopup(index)}
                                />
                            )}
                        </div>
                    </div>
                ))}
            </section>
        </div>
    );
};

const FactCheckPopup = ({ facts, onClose }) => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white p-6 rounded-lg shadow-lg max-w-lg w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-gray-800">Fact Check Results</h2>
                <FaTimes className="cursor-pointer text-gray-600" onClick={onClose} />
            </div>
            <ul className="space-y-4 overflow-y-auto max-h-[60vh] pr-2">
                {facts.map((fact, index) => {
                    let statusColor;
                    switch (fact.status.toLowerCase()) {
                        case 'verified':
                            statusColor = 'bg-green-100 text-green-700';
                            break;
                        case 'moderate':
                            statusColor = 'bg-orange-100 text-orange-700';
                            break;
                        case 'unverified':
                            statusColor = 'bg-red-100 text-red-700';
                            break;
                        default:
                            statusColor = 'bg-gray-100 text-gray-700';
                            break;
                    }

                    return (
                        <li key={index} className="p-4 bg-gray-100 rounded-md">
                            <p className="text-sm font-medium">
                                <strong>Fact:</strong> {fact.fact}
                            </p>
                            <p className={`text-sm font-semibold p-2 rounded ${statusColor}`}>
                                <strong>Status:</strong> {fact.status}
                            </p>
                            <p className="text-sm">
                                <strong>Best Similarity:</strong> {fact.best_similarity.toFixed(2)}
                            </p>
                            <p className="text-sm">
                                <strong>Best Source:</strong> {fact.best_source}
                            </p>
                            <p className="text-sm">
                                <strong>Text:</strong> {fact.text}
                            </p>
                        </li>
                    );
                })}
            </ul>
            <button
                onClick={onClose}
                className="mt-6 w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 transition-colors"
            >
                Close
            </button>
        </div>
    </div>
);

const CheckFactsButton = ({ onClick, isLoading }) => (
    <button
        onClick={onClick}
        className={`bg-blue-600 text-white text-xs font-semibold px-4 py-2 rounded-md hover:bg-blue-700 transition-colors ${isLoading ? 'cursor-wait' : 'cursor-pointer'}`}
        disabled={isLoading}
    >
        {isLoading ? (
            <FaSpinner className="animate-spin" />
        ) : (
            "Check Facts"
        )}
    </button>
);

const ActionButton = ({ icon: Icon, label, onClick, isLoading }) => (
    <div className="relative group">
        <Icon className={`text-gray-500 cursor-pointer ${isLoading ? 'animate-spin' : ''}`} onClick={onClick} />
        <span className="absolute left-1/2 transform -translate-x-1/2 translate-y-full mt-1 bg-gray-800 text-white text-xs rounded-md px-2 py-1 opacity-0 group-hover:opacity-100 transition-opacity">
            {label}
        </span>
    </div>
);


function LessonList() {
  
const location = useLocation();
const lessonsData = location.state?.lessons || [];
const [lessons, setLessons] = useState(lessonsData);
const [selectedLesson, setSelectedLesson] = useState(lessons[0] || null);

if (!lessons || lessons.length === 0) {
    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100">
            <p className="text-xl text-gray-600">No lessons available.</p>
        </div>
    );
}

const updateContent = (lessonId, topicIndex, subtopicIndex, newContent) => {
    const updatedLessons = lessons.map(lesson => {
        if (lesson.id === lessonId) {
            const updatedTopics = lesson.topics.map((topic, tIndex) => {
                if (tIndex === topicIndex) {
                    if (subtopicIndex !== null) {
                        const updatedSubtopics = topic.subtopics.map((subtopic, sIndex) => {
                            if (sIndex === subtopicIndex) {
                                return { ...subtopic, content: newContent };
                            }
                            return subtopic;
                        });
                        return { ...topic, subtopics: updatedSubtopics };
                    } else {
                        return { ...topic, content: newContent };
                    }
                }
                return topic;
            });
            return { ...lesson, topics: updatedTopics };
        }
        return lesson;
    });

    setLessons(updatedLessons);

    const updatedSelectedLesson = updatedLessons.find(lesson => lesson.id === lessonId);
    setSelectedLesson(updatedSelectedLesson);
};

return (
    <div className="min-h-screen bg-gray-100 flex">
        <Sidebar
            lessons={lessons}
            selectedLesson={selectedLesson}
            setSelectedLesson={setSelectedLesson}
        />
        <main className="w-3/4 p-8 overflow-y-auto h-screen">
            {selectedLesson ? (
                <LessonContent lesson={selectedLesson} updateContent={updateContent} />
            ) : (
                <p className="text-xl text-gray-600">Select a lesson from the sidebar to view details.</p>
            )}
        </main>
    </div>
);
}

export default LessonList;
