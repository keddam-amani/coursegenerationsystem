import json
import requests
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import spacy
import nltk

# Download the NLTK punkt tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# Load the sentence transformer model and spaCy NER model
model = SentenceTransformer('all-mpnet-base-v2')
# Faster model
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
nlp = spacy.load("en_core_web_sm")

# Function to get full text from Wikipedia
def get_full_text_wikipedia(query):
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['query']['search']:
            pageid = data['query']['search'][0]['pageid']
            return get_wikipedia_content(pageid)
    return None

def get_wikipedia_content(pageid):
    url = f"https://en.wikipedia.org/w/api.php?action=parse&pageid={pageid}&prop=text&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return BeautifulSoup(data['parse']['text']['*'], 'html.parser').text
    return None

# Function to get full text from Wikidata
def get_full_text_wikidata(query):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language=en&format=json"
    response = requests.get(url)
    if (response.status_code == 200):
        data = response.json()
        if data['search']:
            entity_id = data['search'][0]['id']
            return get_wikidata_content(entity_id)
    return None

def get_wikidata_content(entity_id):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&languages=en&format=json"
    response = requests.get(url)
    if (response.status_code == 200):
        data = response.json()
        return json.dumps(data['entities'][entity_id], indent=2)
    return None

def named_entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def verify_entities(entities):
    entity_verification_results = {}
    for entity, label in entities:
        entity_verification_results[entity] = {
            'wikipedia': get_full_text_wikipedia(entity),
            'wikidata': get_full_text_wikidata(entity),
        }
    return entity_verification_results

def verify_fact(fact, max_text_length=300):
    verification_results = {}

    # Run NER and entity verification
    entities = named_entity_recognition(fact)
    entity_verifications = verify_entities(entities)

    verification_results[fact] = {
        'entities': entities,
        'entity_verifications': entity_verifications,
        'wikipedia': get_full_text_wikipedia(fact),
        'wikidata': get_full_text_wikidata(fact),
    }

    # Analyze the results
    best_similarity, best_source, best_text = analyze_fact_results(verification_results, max_text_length)

    # Determine the status based on similarity
    status = 'verified' if best_similarity > 0.8 else 'moderate' if best_similarity > 0.5 else 'unverified'

    return {
        'fact': fact,
        'status': status,
        'best_similarity': best_similarity,
        'best_source': best_source,
        'text': best_text
    }

def analyze_fact_results(verification_results, max_text_length):
    print("it's analyzing")
    analysis = []
    for fact, sources in verification_results.items():
        fact_embedding = model.encode(fact, convert_to_tensor=True)
        for source, text in sources.items():
            if source in ['entities', 'entity_verifications']:
                continue
            if text:
                sentences = sent_tokenize(text)
                best_similarity = 0
                best_sentence = ""
                for sentence in sentences:
                    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(fact_embedding, sentence_embedding).item()
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_sentence = sentence
                if best_sentence:
                    if len(best_sentence) > max_text_length:
                        best_sentence = best_sentence[:max_text_length] + '...'
                    analysis.append({
                        'fact': fact,
                        'source': source,
                        'similarity': best_similarity,
                        'text': best_sentence
                    })
    print("done analyzing all sources")
    # Determine the best result from analysis
    if analysis:
        best_result = max(analysis, key=lambda x: x['similarity'])
        print (best_result['similarity'])
        print(best_result['source'])
        print(best_result['text'])
        return best_result['similarity'], best_result['source'], best_result['text']
    else:
        return 0, None, None  # Default values if no suitable analysis found

