from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging
import time
from fact_verification import verify_fact

load_dotenv()

api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

gpt_model = "gpt-4o-2024-08-06"

with open("courseplan_format.json", "r") as json_file:
    example_format = json_file.read()
with open("lessoncontent_format.json", "r") as json_file:
    lesson_format = json_file.read()

app = Flask(__name__)
CORS(app)

def generate_topic_content(prompt):
    response = client.chat.completions.create(
        response_format={"type": "json_object"},
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are a content generation system specialized in creating comprehensive and well-structured educational materials for higher education. Your task is to generate detailed and organized content for the given topic and its subtopics, who provides responses in JSON format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4096,
        top_p=1
    )
    return response.choices[0].message.content.strip()

def summarize_content(content):
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are a highly experienced academic course designer who summarizes content."},
            {"role": "user", "content": f"Summarize the following content:\n\n{content}"}
        ],
        temperature=0.7,
        max_tokens=500,
        top_p=1
    )
    return response.choices[0].message.content.strip()

def generate_detailed_lesson_content(lesson, previous_lessons_summary):
    detailed_lesson_content = {
        "id": lesson['id'],
        "lesson_title": lesson['lesson_title'],
        "lesson_description": lesson['description'],
        "learning_objectives": lesson['learningObjectives'],
        "topics": []
    }

    previous_sections_summary = ""
    for topic in lesson['topics']:
        topic_prompt = f"""
        # Task: 
        Generate detailed and well-organized content for the given topic and its respective subtopics.

        # Requirements:
        The following list contains all information you need to consider when writing to help build the lesson:  
        * Topic: {topic['title']}.
        * Description: {topic['description']}
        * Subtopics: {topic['subtopics']}
        * Previous_topics_summary: {previous_sections_summary}
        * Previous_lessons_summary: {previous_lessons_summary}

        # Instructions:
        1. Ensure each section includes sufficient detail and depth. When applicable include examples to provide more explanation.
        2. Use clear and concise language to ensure understanding.
        3. Use an academic and formal tone.
        4. Organize the content logically, with smooth transitions between sections.
        5. Refer to the Previous_topics_summary {previous_sections_summary} to build continuity within the lesson and the Previous_lessons_summary {previous_lessons_summary} where relevant to build continuity among all of the lessons and reinforce learning.
        6. Please provide the output in the following JSON format:
        {{
            "title": "{topic['title']}",
            "content": "Provide a thorough and detailed explanation of the main topic. Depending on the topic, include definitions, key concepts, the importance of the topic, and relevant examples. Ensure the content is clear, logical, and easy to understand.",
            "subtopics": [
                {{
                    "title": "Subtopic Title 1",
                    "content": "Offer an in-depth discussion of Subtopic Title 1. Provide additional details, examples, or explanations that elaborate on how this subtopic relates to the main topic. Make sure the content is well-organized and informative."
                }},
                {{
                    "title": "Subtopic Title 2",
                    "content": "Offer an in-depth discussion of Subtopic Title 2. Provide additional details, examples, or explanations that elaborate on how this subtopic relates to the main topic. Make sure the content is well-organized and informative."
                }}
            ]
        }}
        7. The content within topics and subtopics must be formatted in Markdown. Depending on the course the following elements might be needed:
            * Code snippets: Use Markdown triple backticks for the code block.
            * Math formulas: 
            * Bulleted or Numbered Lists:
            * Tables: """

        detailed_topic_content = generate_topic_content(topic_prompt)
        previous_sections_summary += summarize_content(detailed_topic_content) + " "
        
        # Parse the generated content to ensure it matches the format
        try:
            detailed_topic = json.loads(detailed_topic_content)
        except json.JSONDecodeError as e:
            print(f"JSON decode error for topic {topic['title']}: {str(e)}")
            continue
        
        detailed_lesson_content["topics"].append(detailed_topic)
    
    return detailed_lesson_content

def get_course_details(course_plan):
    course_details = []

    for course in course_plan['course']:
        detail = {
            'course_id': course['id'],
            'lesson_title': course['lesson_title'],
            'lesson_description': course['description'],
            'learning_objectives': course['learningObjectives'],
            'topics': course['topics']
        }
        course_details.append(detail)
    
    return course_details

def regenerate_content(content):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
        # Role
        You are are a skilled academic writer specializing in paraphrasing scholarly content. 

        # Objective
        Paraphrase the following text to convey the same meaning using different words and sentence structures. Ensure that the paraphrased content is clear, coherent, and suitable for an academic audience.

        # Imprtant
        Avoid adding new information, the paraphrased text must only contain information from the original text.

        # Paraphrasing Guidelines
            * Carefully read the original text to fully understand the key concepts and main points.
            * Break down the text into its main ideas and supporting details.
            * Use varied vocabulary to replace words with suitable synonyms or phrases when it enhances readabiliy. 
            * Avoid using overly complex words that might confuse the audience, in such cases it's better to keep the original word. 
            * Change the structure of sentences where it enhances clarity. You can split complex sentences into simpler ones or combine simple sentences to add depth.
            * Retain Original Meaning: Ensure that the paraphrased text reflects the original meaning and intent without omitting or distorting key information.
            * Reorganize Ideas if Necessary: Rearrange the sequence of ideas if it enhances the flow and clarity of the text, but maintain the logical progression of arguments or concepts.
            * Revise and Edit: Review the paraphrased text for coherence, clarity, and grammatical accuracy.

        # Examples
            Original text: "A thread of execution is the smallest sequence of instructions that can be independently managed by a scheduler. It is an essential concept in concurrent programming and operating systems. Threads are small components of a process, and multiple threads can run concurrently, effectively sharing the same code, memory, variables, etc. Each thread shares the same code, data, and heap blocks but will have its own stack. This ensures that local variables and function-related data for one thread do not conflict with those of another. Threads are often called lightweight processes because they have their own stack memory and minimal overhead compared to full processes."
            Paraphrased Text: "An execution thread represents the smallest unit of instruction that a scheduler can manage independently. This concept is crucial in concurrent programming and operating systems. Threads are subsets of a process, allowing multiple threads to execute simultaneously while sharing the same code, memory, variables, and more. All threads operate within the same code, data, and heap regions, but each possesses its own stack. This design prevents conflicts among local variables and function-specific data across different threads. Threads are frequently referred to as lightweight processes due to their separate stack memory and significantly lower overhead compared to complete processes."

        # Output format
        The response format must be only a string containing the new text.
                
        """},
                    {"role": "user", "content": f"""
                    The original text: {content}        
          """}
        ]
    ) 
    return response.choices[0].message.content.strip()


def extend_content(content):
    lower_bound, upper_bound = calculate_bounds_shorten(count_words(content))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
        # Role
        You are a skilled academic writer specializing in expanding and adding depth to academic content. 

        # Task
        Your task is to expand the given text by developing further the existing concepts and explanations, ensuring that your expansions add depth and detail, while maintaining the original structure and flow of the text.

        # Objective
        Aim to expand the text to a total of {lower_bound} to {upper_bound} words.

        # Expansion Guidelines:

        1. Detailed Explanations:
            * Break down each concept clearly and thoroughly, aiming for depth and clarity.
            * Ensure students can grasp complex ideas more effectively through comprehensive coverage without unnecessary extension.
        2. Selective Examples:
            * Use relevant examples and case studies where applicable to clarify complex concepts.
            * Ensure examples are directly connected to the topic and enhance understanding.
        3. Preserve Core Ideas:
            * Focus on expanding the existing material to deepen understanding without deviating from the main points.
        4. Maintain Original Structure:
            * Keep the original organization of the content intact.
            * Ensure the expansion flows naturally within the established framework.
        5. Avoid Unnecessary Summaries:
            * Do not add summaries at the end unless they are part of the original text.
        6. Use a Formal and Academic Tone:
            * Maintain a formal and academic tone throughout the expanded content.
        7. Ensure Accuracy of the expansions

        # Output format
        The response format must be only a string containing the new text

        # Important
        Ensure the text is expanded only to {lower_bound} till {upper_bound} words in total.           

        """},
                    {"role": "user", "content": f"""
                    The original text: {content}
                    The lower_bound: {lower_bound}: The minimum target for increasing the word count.
                    The upper_bound: {upper_bound}: The maximum target for increasing the word count
                
                """}
        ]
    ) 
    return response.choices[0].message.content.strip()

def shorten_content(content):
    lower_bound, upper_bound = calculate_bounds_shorten(count_words(content))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
        # Role
        You are a skilled academic writer specializing in writing concise and clear academic content. 

        # Task
        Your task is to shorten the given text while preserving its original meaning and essential information, ensuring that no key details are lost or altered.    

        # Objective
        Aim to condense the text to a total of {lower_bound} to {upper_bound} words.

        # Instructions
        Follow these steps to achieve the targeted length while maintaining the main goals of the text:

        1. Identify key Concepts
        * Highlight all essential concepts, information, and explanations that must be retained.
        * Preserve the structure and sequence of key concepts, information, and important explanations or examples.

        2. Remove Redundancies:
        * Eliminate repetitive information and redundant phrases.
        * Condense lengthy explanations without losing important details.

        3. Streamline Content:
        * Remove non-essential examples or anecdotes while retaining essential examples and data findings.
        * Focus on retaining the primary message and educational purpose

        4. Ensure Coherence:
        * Ensure the modified text is coherent and easy to understand.
        * Keep the logical flow and structure of the original text.

        5. Review and Revise:
        * Read through the shortened text to ensure it accurately conveys the original meaning.
        * Verify that all key information and concepts are present.
        * Make adjustments as needed to improve clarity and flow.  
        
        # Examples

        # Output format
        Return only the shortened text as a string.

    """},
                {"role": "user", "content": f"""
                The original text: {content}
                The lower_bound: {lower_bound}
                The upper_bound: {upper_bound}                
                """}
            ]
        )
    return response.choices[0].message.content.strip()

@app.route('/generate_course_plan', methods=['POST'])
def generate_course_plan():
    data = request.json
    course_name = data.get('course_name', '').strip()
    course_description = data.get('course_description', '').strip()
    prerequisites = data.get('prerequisites', '').strip()
    number_of_lessons = data.get('number_of_lessons', '').strip()
    try:
        response = client.chat.completions.create(
            response_format={"type": "json_object"},
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a highly experienced academic course designer with expertise in creating comprehensive and detailed course plans for various subjects."},
                {"role": "user", "content": f"""
                I need you to create a detailed course plan for the following course:
                - Course Name: {course_name}
                - Course Description: {course_description}
                - Prerequisites: {prerequisites}
                - Number of Lessons: {number_of_lessons}
                """},
                {"role": "assistant", "content": f"""
                Your task is to generate a structured and detailed course plan in JSON format. The course plan should include a coherent sequence of lessons, each containing a high-level outline with relevant topics, subtopics, and learning objectives. Each lesson should build upon the previous one to ensure a logical progression of knowledge.

                Instructions:
                1. **Initialization**:
                   - Define the course structure with the given number of lessons.
                   - Each lesson should have a unique ID, title, description, topics, subtopics, and learning objectives.

                2. **Lesson Structure**:
                   - For each lesson, generate a descriptive and engaging title.
                   - Write a detailed description summarizing the lesson content.
                   - Identify and list the main topics covered in the lesson.
                   - For each topic, provide relevant subtopics.
                   - Provide a detailed description for each topic, including these points if applicable:
                       - **Definition**: Clearly define the topic.
                       - **Key Concepts**: Explain the key concepts related to the topic.
                       - **Historical Context**: Provide historical background if applicable.
                       - **Importance**: Discuss the importance and relevance of the topic.
                       - **Real-World Applications**: Describe how the topic is applied in real-world scenarios.
                       - **Examples**: Provide examples to illustrate the topic.
                       - **Case Studies**: Include relevant case studies to deepen understanding.
                   - For each topic, provide relevant subtopics that align with the description of the topic.
                   - Define clear and measurable learning objectives for each lesson.
                   - Reference the prerequisites where necessary to ensure each lesson builds upon the assumed prior knowledge.
                   - Adjust the content length to match the lesson duration provided.

                3. **Content Variability**:
                   - Ensure that the number of topics and subtopics varies based on the complexity and content of each lesson.
                   - Avoid having a fixed number of topics or subtopics.

                4. **Coherence and Order**:
                   - Ensure that each lesson logically follows from the previous one.
                   - Maintain coherence in the progression of topics.

                5. **Recursive Criticism and Improvement**:
                   - Generate an initial version of the course plan.
                   - Critically evaluate the initial version, identifying areas for improvement.
                   - Refine the course plan iteratively based on the feedback.
                   - Repeat the evaluation and refinement process until the course plan is comprehensive and well-structured.

                6. **Final Output**:
                   - Format the entire course plan in JSON as specified in this format: {example_format}
                
                Begin generating the initial course plan based on these guidelines, then provide a critical review and subsequent improvements.
                """}
            ],
            temperature=0.7,
            max_tokens=4096,
            top_p=1
        )
        result = response.choices[0].message.content
        
        # Parse JSON response
        course_plan = json.loads(result)
        print(course_plan)  # Log the parsed lesson titles
        return jsonify(course_plan)
    except json.JSONDecodeError as json_err:
        print('JSON decode error:', str(json_err))  # Log the JSON decode error
        return jsonify({"error": "Failed to parse JSON response from OpenAI API"}), 500
    except Exception as e:
        print('Error:', str(e))  # Log the error
        return jsonify({"error": str(e)}), 500

@app.route('/generate_lessons', methods=['POST'])
def generate_lessons_content():
    data = request.json
    print(data)
    course_plan = data.get('course_plan', {})
    detailed_course_plan = []

    try:
        # Generate detailed content for each lesson
        previous_lessons_summary = ""
        for lesson in course_plan['course']:
            print(lesson)
            lesson_details = get_course_details(course_plan)
            lesson_content = generate_detailed_lesson_content(lesson, previous_lessons_summary)
            previous_lessons_summary += summarize_content(json.dumps(lesson_content)) + " "
            lesson['detailed_content'] = lesson_content

            # Print the detailed content for debugging
            print(f"Lesson ID: {lesson['id']}")
            print(json.dumps(lesson_content, indent=2))

            detailed_course_plan.append(lesson_content)

        return jsonify(detailed_course_plan)
    except json.JSONDecodeError as json_err:
        print('JSON decode error:', str(json_err))  # Log the JSON decode error
        return jsonify({"error": "Failed to parse JSON response from OpenAI API"}), 500
    except Exception as e:
        print('Error:', str(e))  # Log the error
        return jsonify({"error": str(e)}), 500

# count words
def count_words(string):
    words = string.split()
    return len(words)

def calculate_bounds_shorten(word_count):
    """Calculates the lower and upper bounds of word reduction based on word count."""
    if word_count < 200:
        # Short sections: 10-20% reduction
        upper_bound = word_count * 0.9  # 10% reduction
        lower_bound = word_count * 0.8  # 20% reduction
    elif 200 <= word_count <= 450:
        # Medium sections: 15-30% reduction
        upper_bound = word_count * 0.85  # 15% reduction
        lower_bound = word_count * 0.7   # 30% reduction
    else:
        # Long sections: 20-40% reduction
        upper_bound = word_count * 0.75   # 25% reduction
        lower_bound = word_count * 0.55   # 45% reduction
    
    return int(lower_bound), int(upper_bound)

def calculate_bounds_lengthen(word_count):
    """Calculates the lower and upper bounds of word count increase based on word count."""
    if word_count < 200:
        # Short sections: 20-40% increase
        upper_bound = word_count * 1.4  # 40% increase
        lower_bound = word_count * 1.2  # 20% increase
    elif 200 <= word_count <= 450:
        # Medium sections: 15-30% increase
        upper_bound = word_count * 1.3  # 30% increase
        lower_bound = word_count * 1.15 # 15% increase
    else:
        # Long sections: 10-20% increase
        upper_bound = word_count * 1.2  # 20% increase
        lower_bound = word_count * 1.1  # 10% increase
    return int(lower_bound), int(upper_bound)

@app.route('/regenerate_topic', methods=['POST'])
def regenerate_topic():
    data = request.json
    content = data.get('content', '').strip()
    
    try:
        regenerated_content = regenerate_content(content)
        return jsonify({"content": regenerated_content})
    except Exception as e:
        print('Error:', str(e))  # Log the error
        return jsonify({"error": str(e)}), 500

@app.route('/shorten_topic', methods=['POST'])
def shorten_topic():
    data = request.json
    content = data.get('content', '').strip()
    
    try:
        shorter_content = shorten_content(content)
        return jsonify({"content": shorter_content})
    except Exception as e:
        print('Error:', str(e))  # Log the error
        return jsonify({"error": str(e)}), 500

@app.route('/expand_topic', methods=['POST'])
def expand_topic():
    data = request.json
    content = data.get('content', '').strip()
    
    try:
        extended_content = extend_content(content)
        return jsonify({"content": extended_content})
    except Exception as e:
        print('Error:', str(e))  # Log the error
        return jsonify({"error": str(e)}), 500

@app.route('/fact_checking', methods=['POST'])
def fact_checking():
    detailed_lesson_content = request.get_json()

    response = client.chat.completions.create(
            response_format={"type": "json_object"},
            model=gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an experienced fact checker. Please perform the following tasks:

                    * Extract only the key facts from the provided lesson data that you believe may require verification. Focus on those facts that are critical to the lesson's accuracy or that could be subject to debate or misinterpretation.
                    * If a fact is generally accepted, uncontroversial, and highly likely to be correct, do not include it for verification.
                    * Each fact should be a clear, concise statement that can be independently verified.
                    * Ensure that the selected facts are directly relevant to the core content and claims made in the text.
                    * Strictly adhere to the response format.
                    JSON Response format: {"facts": ["fact1", "fact2", "..."]}
                """
                },
                {"role": "user", "content": f"""{detailed_lesson_content}
                """},
            ],
            temperature=0.5,
            max_tokens=4096,
        )
    facts_json = json.loads(response.choices[0].message.content)

    checked_facts = []
    for fact in facts_json['facts']:
        result = verify_fact(fact)
        checked_facts.append(result)
    
    return jsonify(checked_facts)
    




if __name__ == '__main__':
    app.run(debug=True)
