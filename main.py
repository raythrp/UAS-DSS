import streamlit as st
import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2E4053;
        font-size: 3rem !important;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .emotion-joy {
        background-color: #D4EFDF;
        border: 2px solid #82E0AA;
    }
    .emotion-fear {
        background-color: #FAD7A0;
        border: 2px solid #F8C471;
    }
    .emotion-anger {
        background-color: #F5B7B1;
        border: 2px solid #E74C3C;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load spaCy model
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

def contains_indonesian_words(text):
    # Common Indonesian words/patterns to check
    indonesian_words = {
        'yang', 'dengan', 'dalam', 'untuk', 'dari', 'pada', 'kepada', 'akan', 'oleh',
        'saya', 'aku', 'kamu', 'dia', 'kami', 'kita', 'mereka',
        'ini', 'itu', 'disini', 'disana',
        'tidak', 'bukan', 'belum', 'sudah', 'telah',
        'bisa', 'dapat', 'harus', 'boleh',
        'dan', 'atau', 'tetapi', 'namun', 'karena', 'sehingga',
        'sangat', 'sekali', 'terlalu',
        'sedang', 'adalah', 'tentang', 'seperti', 'juga',
        'ada', 'bagi', 'lagi', 'setelah', 'apabila',
        'gimana', 'kenapa', 'bagaimana', 'mengapa', 'dimana'
    }
    
    # Convert text to lowercase and split into words
    words = text.lower().split()
    
    # Check if any Indonesian words are present
    return any(word in indonesian_words for word in words)

def is_likely_english(text):
    # First check for Indonesian words
    if contains_indonesian_words(text):
        return False
    
    # Then proceed with English checking
    doc = nlp(text)
    english_tokens = sum(1 for token in doc if token.is_alpha and not token.is_punct and not token.is_space)
    total_tokens = sum(1 for token in doc if token.is_alpha)
    return total_tokens == 0 or (english_tokens / total_tokens) > 0.6

def validate_input(text):
    if not text.strip():
        return False, "Please enter some text to analyze!"
    
    if contains_indonesian_words(text):
        return False, """
        ⚠️ Error: Indonesian text detected!
        
        Mohon maaf, aplikasi ini hanya mendukung teks berbahasa Inggris.
        Please enter your text in English.
        
        Example: "I am happy" (bukan "Saya senang")
        """
    
    if not is_likely_english(text):
        return False, """
        ⚠️ Error: Non-English text detected!
        Please enter your text in English.
        """
    return True, ""

def preprocess(text):
    document = nlp(text)
    done_tokens = []
    for token in document:
        if token.is_stop or token.is_punct:
            continue
        done_tokens.append(token.lemma_)
    return " ".join(done_tokens)

@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv("./Emotion_classify_Data.csv")
    data['Preprocessed Comment'] = data['Comment'].apply(preprocess)
    data['Emotion Number'] = data['Emotion'].map({'joy': 0, 'fear': 1, 'anger': 2})
    return data

@st.cache_resource
def train_model(data):
    x_train, x_test, y_train, y_test = train_test_split(
        data['Preprocessed Comment'], 
        data['Emotion Number'], 
        test_size=0.2, 
        random_state=42, 
        stratify=data['Emotion Number']
    )
    
    vectorizer = TfidfVectorizer()
    x_train_cv = vectorizer.fit_transform(x_train)
    
    model = MultinomialNB()
    model.fit(x_train_cv, y_train)
    
    return model, vectorizer

def predict_emotion_with_proba(text, model, vectorizer):
    preprocessed_text = preprocess(text)
    text_vectorized = vectorizer.transform([preprocessed_text])
    
    # Get prediction probabilities
    prediction_proba = model.predict_proba(text_vectorized)[0]
    prediction = model.predict(text_vectorized)[0]
    
    emotion_map = {0: 'Joy', 1: 'Fear', 2: 'Anger'}
    probabilities = {emotion_map[i]: prob for i, prob in enumerate(prediction_proba)}
    
    return emotion_map[prediction], probabilities

def create_emotion_confidence_plot(probabilities):
    emotions = list(probabilities.keys())
    confidence_scores = list(probabilities.values())
    
    colors = {'Joy': '#82E0AA', 'Fear': '#F8C471', 'Anger': '#E74C3C'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=[score * 100 for score in confidence_scores],  # Convert to percentage
            marker_color=[colors[emotion] for emotion in emotions],
            text=[f"{score:.1f}%" for score in [score * 100 for score in confidence_scores]],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Emotion Confidence Scores",
        yaxis_title="Confidence (%)",
        xaxis_title="Emotion",
        yaxis_range=[0, 100],
        showlegend=False,
        height=400
    )
    
    return fig

def main():
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.info("""
        This app uses Natural Language Processing to detect emotions in text.
        
        Currently supports:
        - Joy 😊
        - Fear 😨
        - Anger 😠
        
        Note: This app only supports English text input.
        Catatan: Aplikasi ini hanya mendukung teks berbahasa Inggris.
        """)
        
        st.title("Instructions")
        st.write("""
        1. Enter your text in English
        2. Click 'Analyze Emotion'
        3. View the predicted emotion and confidence scores
        """)
    
    # Main content
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        st.title("🎭 Emotion Detection AI")
    
    st.write("---")
    
    # Load data and train model
    with st.spinner('Initializing the AI...'):
        data = load_and_prepare_data()
        model, vectorizer = train_model(data)
    
    st.subheader("Enter Your Text")
    user_input = st.text_area(
        "Type or paste your text here (English only):",
        height=150,
        placeholder="Example: I'm so excited about the upcoming vacation!"
    )
    
    analyze_button = st.button('Analyze Emotion 🔍')
    
    if analyze_button:
        is_valid, error_message = validate_input(user_input)
        
        if not is_valid:
            st.error(error_message)
        else:
            with st.spinner('Analyzing emotions in your text...'):
                # Add a small delay for effect
                time.sleep(1)
                prediction, probabilities = predict_emotion_with_proba(user_input, model, vectorizer)
                
                # Create two columns for results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display prediction result
                    emotion_class = f"prediction-box emotion-{prediction.lower()}"
                    st.markdown(f"""
                        <div class="{emotion_class}">
                            <h2>Primary Emotion</h2>
                            <h1>{prediction} {
                                '😊' if prediction == 'Joy' else
                                '😨' if prediction == 'Fear' else
                                '😠'
                            }</h1>
                            <p>Confidence: {probabilities[prediction]*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Display confidence plot
                    fig = create_emotion_confidence_plot(probabilities)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show preprocessing details in expander
                with st.expander("See technical details"):
                    st.write("**Original Text:**")
                    st.write(user_input)
                    st.write("**Preprocessed Text:**")
                    st.write(preprocess(user_input))
    
    # Example section
    st.write("---")
    with st.expander("💡 Example inputs to try"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Joy Examples")
            st.write("""
            - "I'm so happy to see you!"
            - "We won the championship!"
            - "Just got promoted at work!"
            """)
            
        with col2:
            st.markdown("### Fear Examples")
            st.write("""
            - "I'm worried about the exam"
            - "The dark shadows scare me"
            - "What if everything goes wrong?"
            """)
            
        with col3:
            st.markdown("### Anger Examples")
            st.write("""
            - "I can't believe they cancelled!"
            - "The service here is terrible"
            - "Why would they do this to me?"
            """)

if __name__ == "__main__":
    main()