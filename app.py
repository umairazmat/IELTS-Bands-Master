import streamlit as st
import requests
import os
import spacy
import subprocess  # Add this import
import language_tool_python
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Retrieve the Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # Accessing the key securely

# Set the API key in the environment (optional, depending on your usage)
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

# URL for the Groq API endpoint
url = "https://api.groq.com/openai/v1/chat/completions"  # Use the chat completions endpoint

# Headers for the API request
headers = {
    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
    "Content-Type": "application/json"
}

# Function to generate the prompt based on the selected question type
def generate_ielts_prompt(question_type):
    if question_type == "Opinion":
        return (
            "Generate a strict IELTS Writing Task 2 question for the type: 'Opinion'. "
            "The format should include: "
            "1. A clear task instruction, e.g., 'Write about the following topic.' "
            "2. No extra text or explanations—only the task question."
        )
    elif question_type == "Discussion":
        return (
            "Generate a strict IELTS Writing Task 2 question for the type: 'Discussion'. "
            "The format should include: "
            "1. A clear task instruction, e.g., 'Write about the following topic.' "
            "2. If needed, include one or more sub-questions in a numbered or bulleted list. "
            "3. No extra text or explanations—only the task question."
        )
    elif question_type == "Advantage/Disadvantage":
        return (
            "Generate a strict IELTS Writing Task 2 question for the type: 'Advantage/Disadvantage'. "
            "The format should include: "
            "1. A clear task instruction, e.g., 'Write about the following topic.' "
            "2. If needed, include one or more sub-questions in a numbered or bulleted list. "
            "3. No extra text or explanations—only the task question."
        )
    elif question_type == "Problem/Solution":
        return (
            "Generate a strict IELTS Writing Task 2 question for the type: 'Problem/Solution'. "
            "The format should include: "
            "1. A clear task instruction, e.g., 'Write about the following topic.' "
            "2. If needed, include one or more sub-questions in a numbered or bulleted list. "
            "3. No extra text or explanations—only the task question."
        )
    else:
        return "Invalid question type selected. Please choose from: Opinion, Discussion, Advantage/Disadvantage, Problem/Solution."

# Load spaCy model for grammar checking
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # This might not work in cloud environments, so ideally this should be handled via requirements.txt or Dockerfile
    st.error("spaCy model not found. Attempting to install...")
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

# Load the BERT tokenizer and model for coherence evaluation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
coherence_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Initialize LanguageTool for grammar checking (using remote server)
tool = language_tool_python.LanguageTool('en-US', remote_server='https://api.languagetool.org')

# Function to evaluate grammar using LanguageTool
def evaluate_grammar(text):
    matches = tool.check(text)
    return len(matches), matches

# Function to calculate vocabulary richness using unique words
def evaluate_vocabulary(text):
    words = text.split()
    unique_words = set(words)
    vocabulary_richness = len(unique_words) / len(words) if len(words) > 0 else 0
    return vocabulary_richness

# Function to evaluate coherence using BERT model
def evaluate_coherence(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = coherence_model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    coherence_score = probabilities[0][1].item()
    return coherence_score

# Function to evaluate structure (average sentence length)
def evaluate_structure(text):
    sentences = [sent.text for sent in nlp(text).sents]
    avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])
    return avg_sentence_length

# Function to assess idea formation (based on content overlap with predefined ideas)
def evaluate_idea_formation(text):
    predefined_ideas = ["global warming is a serious issue", "education should be free", "technology impacts society positively"]
    text_vectors = [nlp(word).vector for word in text.split() if word in nlp.vocab]
    if not text_vectors:
        return 0
    text_vector = np.mean(text_vectors, axis=0)
    similarities = []
    for idea in predefined_ideas:
        idea_vectors = [nlp(word).vector for word in idea.split() if word in nlp.vocab]
        if not idea_vectors:
            continue
        idea_vector = np.mean(idea_vectors, axis=0)
        similarity = cosine_similarity([text_vector], [idea_vector])
        similarities.append(similarity[0][0])
    return max(similarities) if similarities else 0

# Function to give status feedback based on score thresholds
def get_feedback(grammar_errors, vocabulary_richness, coherence, sentence_length, idea_formation):
    feedback = []

    # Grammar feedback
    if grammar_errors == 0:
        feedback.append("No grammar mistakes detected. Great job!")
    else:
        feedback.append("There are some grammar mistakes. Pay attention to subject-verb agreement and punctuation.")

    # Vocabulary feedback
    if vocabulary_richness >= 0.75:
        feedback.append("Your vocabulary usage is rich. Well done!")
    else:
        feedback.append("Try to use a wider range of vocabulary to make your writing more engaging.")

    # Coherence feedback
    if coherence >= 0.7:
        feedback.append("The essay flows well with a clear logical structure.")
    else:
        feedback.append("Your essay needs better coherence. Consider improving transitions and the logical flow between ideas.")

    # Structure feedback
    if sentence_length <= 20:
        feedback.append("Your sentence length is ideal. It's not too long or too short.")
    else:
        feedback.append("Consider breaking up long sentences to improve readability.")

    # Idea formation feedback
    if idea_formation >= 0.7:
        feedback.append("Your ideas are well-formed and supported.")
    else:
        feedback.append("Focus on strengthening your ideas. Provide more concrete examples or stronger arguments.")

    return "\n".join(feedback)

# Streamlit UI
def main():
    st.title("IELTS Writing Task 2 Evaluation")

    # Step 1: Select a Question Type
    question_type = st.selectbox(
        "Select a Question Type",
        options=["Opinion", "Discussion", "Advantage/Disadvantage", "Problem/Solution"]
    )
    
    # Step 2: Generate Prompt from Groq API
    if st.button("Generate IELTS Prompt"):
        prompt = generate_ielts_prompt(question_type)
        
        # If the prompt is valid, send the request to the Groq API
        if "Invalid question type" not in prompt:
            data = {
                "model": "llama3-groq-70b-8192-tool-use-preview",  # Model ID from GroqCloud
                "messages": [  # Chat messages format for the completion
                    {
                        "role": "user",  # The role of the sender (user, assistant)
                        "content": prompt  # The message content
                    }
                ],
                "max_tokens": 150  # Set an appropriate limit for the output length
            }

            # Send the request to the API
            response = requests.post(url, headers=headers, json=data)

            # Handle the API response
            if response.status_code == 200:
                generated_prompt = response.json()['choices'][0]['message']['content'].strip()
                st.subheader("Generated Prompt")
                st.write(generated_prompt)
                st.session_state.generated_prompt = generated_prompt  # Store generated prompt
            else:
                st.error(f"Error: {response.status_code}, {response.text}")
        else:
            st.error(prompt)

    # Step 3: Write an Essay
    essay = st.text_area("Write your essay here:", height=200)
    st.session_state.essay = essay  # Persist the essay text

    # Evaluate different aspects
    if st.button("Evaluate Grammar"):
        if essay:
            grammar_errors, _ = evaluate_grammar(essay)
            st.session_state.grammar_errors = grammar_errors
            st.write(f"Grammar Errors: {grammar_errors}")
        else:
            st.warning("Please write an essay before evaluating.")
    
    if st.button("Evaluate Vocabulary"):
        if essay:
            vocabulary_richness = evaluate_vocabulary(essay)
            st.session_state.vocabulary_richness = vocabulary_richness
            st.write(f"Vocabulary Richness: {vocabulary_richness:.2f}")
        else:
            st.warning("Please write an essay before evaluating.")

    if st.button("Evaluate Coherence"):
        if essay:
            coherence = evaluate_coherence(essay)
            st.session_state.coherence = coherence
            st.write(f"Coherence: {coherence:.2f}")
        else:
            st.warning("Please write an essay before evaluating.")

    if st.button("Evaluate Structure"):
        if essay:
            avg_sentence_length = evaluate_structure(essay)
            st.session_state.avg_sentence_length = avg_sentence_length
            st.write(f"Average Sentence Length: {avg_sentence_length:.2f}")
        else:
            st.warning("Please write an essay before evaluating.")

    if st.button("Evaluate Idea Formation"):
        if essay:
            idea_formation = evaluate_idea_formation(essay)
            st.session_state.idea_formation = idea_formation
            st.write(f"Idea Formation: {idea_formation:.2f}")
        else:
            st.warning("Please write an essay before evaluating.")

    # Show detailed feedback if available
    if st.button("Show Feedback"):
        if 'grammar_errors' in st.session_state:
            feedback = get_feedback(
                st.session_state.grammar_errors,
                st.session_state.vocabulary_richness,
                st.session_state.coherence,
                st.session_state.avg_sentence_length,
                st.session_state.idea_formation
            )
            st.subheader("Detailed Feedback:")
            st.write(feedback)
        else:
            st.warning("Please evaluate all aspects first.")

if __name__ == "__main__":
    main()
    
