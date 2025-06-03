#  An end to End AI Course Generation System

This project presents an AI-powered system designed to automate the generation of higher education academic courses using OpenAI's language models. It integrates advanced prompt engineering techniques, natural language processing (NLP), and automated fact verification to simplify, accelerate, and enhance the course development process for educators.

**Read the full Master’s Thesis:**  
[*Designing an End-to-End AI Course Generation System (PDF)*](https://drive.google.com/file/d/1ux6LOoFCKjYs1oqOs-W4lIZ5BBG2fd_-/view?usp=sharing)

---

##  Objectives

- Develop an academic course generation pipeline tailored for higher education.
- Leverage prompt engineering to maximize the effectiveness of the ChatGPT model.
- Reduce the manual workload and time investment required for course design.
- Ensure academic accuracy and integrity through automated fact-checking.

---

##  System Overview

This system guides educators through a comprehensive pipeline:

1. **Course Outline Generation** – Automatically generates structured outlines for academic courses.
2. **Content Development** – Creates and refines lecture content (shorten, expand, paraphrase).
3. **Fact Verification** – Validates claims using:
   - Named Entity Recognition (NER)
   - Wikipedia API for source retrieval
   - SentenceTransformers for semantic similarity analysis
   - A classification system: `Verified`, `Moderate`, or `Unverified`
     
<img width="402" alt="Screenshot 2025-05-24 at 00 20 17" src="https://github.com/user-attachments/assets/dfc59a43-cb45-4869-a41e-3798cbdbd867" />

---

##  Prompt Engineering Framework

To optimize AI interactions, the system incorporates a custom prompt framework:

**Structure:**
- **Role**
- **Task**
- **Instructions**
- **Examples**
- **Output Format**

### Prompting Techniques Used

- **Prompt Patterns:** Persona Pattern, Fact Check List Pattern
- **Prompting Techniques:** 
  - Generated Knowledge Prompting
  - Recursive Criticism and Improvement (RCI)
  - Few-Shot Prompting

---

##  Evaluation Metrics

The system’s output was evaluated using a set of NLP and LLM-informed metrics, for instance:

- **Contextual Relevance** – Evaluates how well content fits within the course scope.
- **Word Count Analysis** – Measures consistency and conciseness across content blocks.
- **Factual Accuracy** – Using the verification system to label information quality.

---

##  Skills & Technologies Used

- OpenAI GPT-4 API
- Prompt Engineering
- Named Entity Recognition (NER)
- Wikipedia API
- SentenceTransformers (Semantic Similarity)
- Natural Language Processing (NLP)
- Python

---

##  Conclusion

This project demonstrates the potential of generative AI in education by streamlining the course creation process. While challenges remain, such as maintaining academic integrity and ensuring content relevance, the system provides a scalable starting point for AI-assisted curriculum design.

---
##  License

This project is licensed under the [MIT License](LICENSE).
