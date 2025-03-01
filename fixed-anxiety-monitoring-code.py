# **Function: Train Anxiety Prediction Model**
def train_anxiety_prediction_model(df):
    """
    Train a linear regression model to predict anxiety levels
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing historical anxiety data
        
    Returns:
        tuple: (model, scaler) - Trained model and feature scaler
    """
    try:
        # Check if we have enough data
        if len(df) < 3:
            logger.warning("Not enough data to train prediction model (need at least 3 records)")
            return None, None
            
        # Prepare the data
        df = df.copy()
        
        # Ensure date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Create time-based features
        df['Hour'] = df['Date'].dt.hour
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        
        # Create lag features (previous anxiety scores)
        df['PrevAnxiety'] = df['Anxiety Index'].shift(1)
        df['Prev2Anxiety'] = df['Anxiety Index'].shift(2)
        
        # Drop rows with NaN values (first 2 rows will have NaN for lag features)
        df = df.dropna(subset=['PrevAnxiety', 'Prev2Anxiety'])
        
        # Select features and target
        features = ['Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'PrevAnxiety', 'Prev2Anxiety']
        
        # Add emotion features if available
        emotion_features = [e for e in ANXIETY_RELATED_EMOTIONS if e in df.columns]
        features.extend(emotion_features)
        
        X = df[features]
        y = df['Anxiety Index']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Save the model and scaler
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
        logger.info(f"Anxiety prediction model trained with {len(df)} records and {len(features)} features")
        return model, scaler
        
    except Exception as e:
        logger.error(f"Failed to train prediction model: {e}")
        return None, None

# **Function: Predict Anxiety Score**
def predict_anxiety_score(df):
    """
    Predict the next anxiety score based on historical data
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing historical anxiety data
        
    Returns:
        float: Predicted anxiety score
    """
    try:
        # First, try to load the model and scaler
        model = None
        scaler = None
        
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        
        # If model doesn't exist, train it
        if model is None or scaler is None:
            model, scaler = train_anxiety_prediction_model(df)
            
        # If we still don't have a model, we can't make predictions
        if model is None or scaler is None:
            logger.warning("Could not create or load prediction model")
            return None
            
        # Prepare data for prediction
        if len(df) < 2:
            logger.warning("Not enough historical data for prediction")
            return None
            
        # Get the most recent data
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Current time for prediction
        now = pd.Timestamp.now()
        
        # Create features for prediction
        features = {
            'Hour': now.hour,
            'DayOfWeek': now.dayofweek,
            'DayOfMonth': now.day,
            'Month': now.month,
            'PrevAnxiety': df['Anxiety Index'].iloc[-1],
            'Prev2Anxiety': df['Anxiety Index'].iloc[-2] if len(df) > 1 else df['Anxiety Index'].iloc[-1]
        }
        
        # Add emotion features if available
        for emotion in ANXIETY_RELATED_EMOTIONS:
            if emotion in df.columns:
                features[emotion] = df[emotion].iloc[-1]
        
        # Convert to DataFrame for prediction
        X_pred = pd.DataFrame([features])
        
        # Make sure we have all the features the model was trained on
        missing_cols = set(scaler.feature_names_in_) - set(X_pred.columns)
        for col in missing_cols:
            X_pred[col] = 0
            
        # Reorder columns to match training order
        X_pred = X_pred[scaler.feature_names_in_]
        
        # Scale features
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        prediction = model.predict(X_pred_scaled)[0]
        
        # Ensure prediction is reasonable (not negative)
        prediction = max(0, prediction)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Failed to predict anxiety score: {e}")
        return None

# **Function: Display Anxiety Prediction**
def display_anxiety_prediction(df):
    """Display the predicted anxiety score and visualize it"""
    try:
        # Get prediction
        predicted_score = predict_anxiety_score(df)
        
        if predicted_score is None:
            logger.info("No anxiety prediction available")
            return
            
        # Display the prediction
        print("\n" + "="*50)
        print("ANXIETY PREDICTION")
        print("="*50)
        print(f"Predicted anxiety score for current time: {predicted_score:.2f}")
        
        # Compare with recent scores
        if len(df) > 0:
            latest_score = df['Anxiety Index'].iloc[-1]
            avg_score = df['Anxiety Index'].mean()
            
            print(f"Latest recorded score: {latest_score:.2f}")
            print(f"Average historical score: {avg_score:.2f}")
            
            # Calculate percent change
            percent_change = ((predicted_score - latest_score) / latest_score) * 100 if latest_score > 0 else 0
            
            if abs(percent_change) < 5:
                trend = "STABLE"
            elif percent_change > 0:
                trend = "INCREASING"
            else:
                trend = "DECREASING"
                
            print(f"Predicted trend: {trend} ({percent_change:.1f}%)")
        
        # Create visualization
        if len(df) > 0:
            try:
                plt.figure(figsize=(12, 6))
                
                # Handle data issues
                df_sorted = df.copy()
                df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
                df_sorted = df_sorted.sort_values('Date')
                
                # Remove any NaN or negative anxiety values
                df_sorted = df_sorted.dropna(subset=['Anxiety Index'])
                df_sorted['Anxiety Index'] = df_sorted['Anxiety Index'].clip(lower=0)
                
                dates = df_sorted['Date']
                scores = df_sorted['Anxiety Index']
                
                # Plot historical data with improved formatting
                plt.plot(dates, scores, marker='o', linestyle='-', linewidth=2, color='blue', 
                         label='Historical Scores', markersize=8)
                
                # Plot prediction
                future_date = pd.Timestamp.now()
                
                # Ensure prediction is reasonable
                min_score = max(0, scores.min() * 0.5)
                max_score = scores.max() * 1.5 if scores.max() > 0 else 100
                predicted_score = max(min_score, min(max_score, predicted_score))
                
                # Add prediction point with clear formatting
                plt.plot([future_date], [predicted_score], marker='X', markersize=12, 
                         color='red', label='Current Prediction')
                
                # Add a more visible highlight for the predicted point
                plt.scatter([future_date], [predicted_score], s=150, color='red', alpha=0.5)
                
                # Add horizontal reference line
                plt.axhline(y=predicted_score, color='r', linestyle='--', alpha=0.3)
                
                # Annotate the predicted value
                plt.annotate(f'Predicted: {predicted_score:.2f}', 
                            xy=(future_date, predicted_score),
                            xytext=(10, 0), 
                            textcoords='offset points',
                            ha='left',
                            va='center',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            fontsize=10,
                            fontweight='bold')
                
                # Add a vertical line for current time
                plt.axvline(x=future_date, color='gray', linestyle='--', alpha=0.3)
                
                # Try to add a trend line if enough points
                if len(dates) >= 3:
                    try:
                        # Get recent dates for trend (last 7 days max)
                        week_ago = future_date - pd.Timedelta(days=7)
                        recent_data = df_sorted[df_sorted['Date'] >= week_ago]
                        
                        if len(recent_data) >= 2:
                            recent_dates = recent_data['Date']
                            recent_scores = recent_data['Anxiety Index']
                            
                            # Create numerical x values for fitting (seconds since first date)
                            x_seconds = [(d - recent_dates.iloc[0]).total_seconds() for d in recent_dates]
                            future_seconds = (future_date - recent_dates.iloc[0]).total_seconds()
                            
                            # Fit trend line
                            z = np.polyfit(x_seconds, recent_scores, 1)
                            p = np.poly1d(z)
                            
                            # Plot trend line
                            x_range = np.array([min(x_seconds), future_seconds])
                            date_range = [recent_dates.iloc[0], future_date]
                            plt.plot(date_range, p(x_range), "r--", alpha=0.7, 
                                     label=f'Trend (m={z[0]:.4f})', linewidth=2)
                    except Exception as e:
                        logger.debug(f"Could not plot trend line: {e}")
                
                # Better formatting
                plt.title("Anxiety Score Prediction", fontsize=14, fontweight='bold')
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Anxiety Index", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                
                # Improve date formatting on x-axis
                plt.gcf().autofmt_xdate()
                
                # Add some padding to y-axis
                y_min, y_max = plt.ylim()
                y_range = y_max - y_min
                plt.ylim(max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)
                
                plt.tight_layout()
                
                # Save figure
                prediction_path = FIGURES_DIR / "anxiety_prediction.png"
                plt.savefig(prediction_path, format='png', dpi=300, bbox_inches='tight')
                
                # Generate a simpler linear-focused visualization
                plt.figure(figsize=(12, 6))
                
                if len(df_sorted) >= 2:
                    # Plot simple line with large data points
                    plt.plot(range(len(df_sorted)), df_sorted['Anxiety Index'], 
                            '-o', linewidth=2, markersize=8, color='blue', label='Historical Data')
                    
                    # Add prediction point at end
                    prediction_x = len(df_sorted)
                    plt.plot([prediction_x], [predicted_score], 'rX', markersize=12, label='Prediction')
                    
                    # Add connecting line from last point to prediction
                    plt.plot([prediction_x-1, prediction_x], 
                            [df_sorted['Anxiety Index'].iloc[-1], predicted_score], 
                            'r--', alpha=0.7)
                    
                    # Simple formatting
                    plt.title('Anxiety Prediction (Linear View)', fontsize=14, fontweight='bold')
                    plt.xlabel('Measurements Over Time', fontsize=12)
                    plt.ylabel('Anxiety Index', fontsize=12)
                    plt.grid(True)
                    plt.legend()
                    
                    # Set x-axis labels to data collection points
                    x_ticks = range(0, len(df_sorted)+1)
                    plt.xticks(x_ticks)
                    
                    # Set y-axis to include zero
                    plt.ylim(bottom=0)
                    
                    # Save linear visualization
                    linear_prediction_path = FIGURES_DIR / "anxiety_linear_prediction.png"
                    plt.savefig(linear_prediction_path, format='png', dpi=300, bbox_inches='tight')
                
                # Try to display
                try:
                    plt.show()
                except Exception as e:
                    logger.debug(f"Could not display prediction plot: {e}")
                finally:
                    plt.close('all')
                    
                logger.info(f"Prediction visualizations saved to {FIGURES_DIR}")
                
            except Exception as e:
                logger.error(f"Failed to generate prediction visualization: {e}")
                plt.close('all')  # Close any open figures in case of error
        
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to display anxiety prediction: {e}")
import firebase_admin
from firebase_admin import credentials, storage
import os
import time
import numpy as np
import pandas as pd
import librosa
import speech_recognition as sr
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pydub import AudioSegment
from transformers import pipeline
import pickle
import logging
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("anxiety_monitor.log")
    ]
)
logger = logging.getLogger("anxiety_monitor")

# Suppress HuggingFace parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local directory to store downloaded files
DOWNLOAD_DIR = Path("downloaded_files")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Data paths
FIREBASE_CREDENTIALS_PATH = Path("serviceAccountKey.json")
DOWNLOADED_FILES_LIST = DOWNLOAD_DIR / "downloaded_files.txt"
ANALYSIS_DATA_FILE = DOWNLOAD_DIR / "anxiety_data.pkl"
FIGURES_DIR = DOWNLOAD_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
MODEL_PATH = DOWNLOAD_DIR / "anxiety_prediction_model.joblib"
SCALER_PATH = DOWNLOAD_DIR / "anxiety_scaler.joblib"

# Keep track of already downloaded files
downloaded_files = set()

# Emotions to track from the original model output
EMOTIONS_TO_TRACK = ["Nervousness", "Fear", "Sadness", "Disappointment", "Confusion", "Remorse", 
                    "Anger", "Annoyance", "Disgust", "Surprise", "Joy", "Admiration", 
                    "Approval", "Gratitude", "Love", "Relief", "Pride", "Excitement", "Desire"]

# Anxiety-Related Emotions (subset used for anxiety index calculation)
ANXIETY_RELATED_EMOTIONS = ["Nervousness", "Fear", "Sadness", "Disappointment", "Confusion", "Remorse"]
SPEECH_FEATURES = ["Speech Rate", "Pitch", "Jitter", "Shimmer", "Pauses", "Sighs"]

# Initialize Firebase
def initialize_firebase():
    """Initialize Firebase with credentials and return the storage bucket"""
    try:
        if not FIREBASE_CREDENTIALS_PATH.exists():
            logger.error(f"Firebase credentials file not found at {FIREBASE_CREDENTIALS_PATH}")
            return None
            
        cred = credentials.Certificate(str(FIREBASE_CREDENTIALS_PATH))
        if not firebase_admin._apps:  # Only initialize if not already initialized
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'dlwimsodead.firebasestorage.app'
            })
        return storage.bucket()
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return None

# Data Storage for Analysis
def initialize_dataframe():
    """Initialize or load existing dataframe"""
    if ANALYSIS_DATA_FILE.exists():
        try:
            return pd.read_pickle(ANALYSIS_DATA_FILE)
        except Exception as e:
            logger.error(f"Failed to load analysis data: {e}")
    
    # Create new dataframe if loading fails
    return pd.DataFrame(columns=["Date", "Anxiety Index", "SpeechText"] + EMOTIONS_TO_TRACK + SPEECH_FEATURES)

anxiety_data = initialize_dataframe()
bucket = initialize_firebase()

# Load GoEmotions Pretrained Model for Emotion Detection
def initialize_emotion_classifier():
    """Initialize the emotion classifier model"""
    try:
        model_name = "monologg/bert-base-cased-goemotions-original"
        return pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=1)
    except Exception as e:
        logger.error(f"Failed to initialize emotion classifier: {e}")
        return None

classifier = initialize_emotion_classifier()
BATCH_SIZE = 5

# **Function: Analyze Common Phrases by Day**
def analyze_daily_phrases(df, date=None):
    """
    Analyze common phrases for a specific day (defaults to today)
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the anxiety data
        date (str or datetime, optional): The date to analyze. Defaults to today.
        
    Returns:
        dict: Dictionary of common phrases and their frequencies
    """
    try:
        # Default to today if no date provided
        if date is None:
            date = pd.Timestamp.now().floor('D')  # Floor to start of day
        else:
            date = pd.Timestamp(date).floor('D')
            
        # Get end of the day
        next_day = date + pd.Timedelta(days=1)
        
        # Filter data for the requested day
        if 'SpeechText' in df.columns:
            day_data = df[(df['Date'] >= date) & (df['Date'] < next_day)]
            
            # Extract speech text from the day
            text_entries = day_data['SpeechText'].dropna().tolist()
            text_entries = [t for t in text_entries if isinstance(t, str) and t.strip()]
            
            if text_entries:
                # Analyze common phrases for this day
                return analyze_common_phrases(text_entries, top_n=20)
            else:
                logger.info(f"No speech text found for {date.strftime('%Y-%m-%d')}")
                return {}
        else:
            logger.info("No speech text data available")
            return {}
    except Exception as e:
        logger.error(f"Daily phrase analysis failed: {e}")
        return {}

# **Function: List Files in Firebase Storage**
def list_files():
    """List all files in the Firebase Storage bucket"""
    if not bucket:
        logger.error("Firebase bucket not initialized")
        return []
        
    try:
        blobs = bucket.list_blobs()
        return [blob.name for blob in blobs]
    except Exception as e:
        logger.error(f"Failed to list files from Firebase: {e}")
        return []

# **Function: Save Downloaded Files List**
def save_downloaded_files():
    """Save the set of downloaded files to disk"""
    try:
        with open(DOWNLOADED_FILES_LIST, "w") as f:
            for file in downloaded_files:
                f.write(f"{file}\n")
        logger.info(f"Saved list of {len(downloaded_files)} downloaded files")
    except Exception as e:
        logger.error(f"Failed to save downloaded files list: {e}")

# **Function: Load Downloaded Files List**
def load_downloaded_files():
    """Load the set of downloaded files from disk"""
    global downloaded_files
    try:
        if DOWNLOADED_FILES_LIST.exists():
            with open(DOWNLOADED_FILES_LIST, "r") as f:
                downloaded_files = set(line.strip() for line in f)
            logger.info(f"Loaded list of {len(downloaded_files)} previously downloaded files")
        else:
            downloaded_files = set()
            logger.info("No previous downloaded files list found")
    except Exception as e:
        logger.error(f"Failed to load downloaded files list: {e}")
        downloaded_files = set()

# **Function: Download File from Firebase Storage**
def download_file(remote_path):
    """
    Download a file from Firebase Storage if it's not already downloaded
    
    Parameters:
        remote_path (str): The path to the file in Firebase Storage
        
    Returns:
        Path or None: The local path to the downloaded file, or None if download failed
    """
    if not bucket:
        logger.error("Firebase bucket not initialized")
        return None
        
    if remote_path in downloaded_files:
        return None  

    local_path = DOWNLOAD_DIR / os.path.basename(remote_path)
    
    # Check if the file already exists locally
    if local_path.exists():
        downloaded_files.add(remote_path)
        return local_path
        
    blob = bucket.blob(remote_path)

    try:
        blob.download_to_filename(str(local_path))
        logger.info(f"New File Downloaded: {remote_path} -> {local_path}")
        downloaded_files.add(remote_path)
        return local_path  
    except Exception as e:
        logger.error(f"Could not download {remote_path}: {e}")
        return None

# **Function: Convert M4A/MP4 to WAV**
def convert_to_wav(input_file):
    """
    Convert audio file to WAV format
    
    Parameters:
        input_file (str or Path): Path to the audio file
        
    Returns:
        Path or None: Path to the converted WAV file, or None if conversion failed
    """
    input_file = Path(input_file)
    output_file = input_file.with_suffix('.wav')
    
    try:
        audio = AudioSegment.from_file(str(input_file))
        audio.export(str(output_file), format="wav")
        return output_file
    except Exception as e:
        logger.error(f"Could not convert {input_file} to WAV: {e}")
        return None

# **Function: Convert Speech to Text**
def speech_to_text(audio_file):
    """
    Convert speech in audio file to text
    
    Parameters:
        audio_file (Path or str): Path to the audio file
        
    Returns:
        str: Transcribed text, or empty string if transcription failed
    """
    if not audio_file:
        return ""
        
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(str(audio_file)) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        logger.warning(f"Speech recognition could not understand audio in {audio_file}")
        return ""
    except Exception as e:
        logger.error(f"Speech recognition failed: {e}")
        return ""

# **Function: Analyze Common Phrases**
def analyze_common_phrases(text_data, top_n=10):
    """
    Analyze and extract most common phrases from speech text data
    
    Parameters:
        text_data (list): List of text strings to analyze
        top_n (int): Number of top phrases to return
        
    Returns:
        dict: Dictionary of phrases and their frequencies
    """
    try:
        if not text_data or len(text_data) == 0:
            return {}
            
        # Tokenize text into words
        all_words = " ".join(text_data).lower()
        
        # Remove punctuation and split into words
        import re
        words = re.findall(r'\b\w+\b', all_words)
        
        # Remove common stopwords
        stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
                    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", 
                    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
                    "their", "theirs", "themselves", "what", "which", "who", "whom", "this", 
                    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", 
                    "being", "have", "has", "had", "having", "do", "does", "did", "doing", 
                    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
                    "while", "of", "at", "by", "for", "with", "about", "against", "between", 
                    "into", "through", "during", "before", "after", "above", "below", "to", 
                    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
                    "further", "then", "once", "here", "there", "when", "where", "why", "how", 
                    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
                    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
                    "t", "can", "will", "just", "don", "don't", "should", "now"]
                    
        filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Count frequency
        word_counts = Counter(filtered_words)
        
        # Get top phrases
        return dict(word_counts.most_common(top_n))
    except Exception as e:
        logger.error(f"Phrase analysis failed: {e}")
        return {}

# **Function: Analyze Emotions**
def analyze_emotions(text):
    """
    Analyze emotions in text using the pretrained classifier
    
    Parameters:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary of emotions and their scores
    """
    if not text or text.strip() == "" or not classifier:
        return {emotion: 0.0 for emotion in EMOTIONS_TO_TRACK}
        
    try:
        results = classifier(text)
        
        # Extract all emotions from the classifier results
        all_detected_emotions = {}
        for item in results[0]:
            emotion_name = item['label'].capitalize()
            emotion_score = item['score'] * 100  # Convert to percentage
            all_detected_emotions[emotion_name] = emotion_score
            
        # Make sure we have some non-zero values
        has_meaningful_values = any(v > 1.0 for v in all_detected_emotions.values())
        
        if not has_meaningful_values and all_detected_emotions:
            # Scale up the highest values to make them more meaningful
            max_val = max(all_detected_emotions.values())
            if max_val > 0:
                scaling_factor = 100.0 / max_val  # Scale highest to 100%
                all_detected_emotions = {k: min(v * scaling_factor * 0.3, 100.0) for k, v in all_detected_emotions.items()}
                logger.info(f"Applied emotion scaling factor: {scaling_factor}")
            
        # Return only the emotions we're tracking
        return {emotion: all_detected_emotions.get(emotion, 0.0) for emotion in EMOTIONS_TO_TRACK}
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        return {emotion: 0.0 for emotion in EMOTIONS_TO_TRACK}

# **Function: Extract Speech Features**
def extract_speech_features(audio_file):
    """
    Extract speech features from audio file
    
    Parameters:
        audio_file (Path or str): Path to the audio file
        
    Returns:
        dict: Dictionary of speech features and their values
    """
    if not audio_file:
        return {feature: 0.0 for feature in SPEECH_FEATURES}
        
    try:
        y, sr = librosa.load(str(audio_file), sr=22050)
        
        # Handle potential errors in pitch estimation
        pitch_values = librosa.yin(y, fmin=50, fmax=300)
        valid_pitch = pitch_values[~np.isnan(pitch_values) & (pitch_values > 0)]
        
        pitch = np.mean(valid_pitch) if len(valid_pitch) > 0 else 0
        jitter = np.std(np.diff(valid_pitch)) if len(valid_pitch) > 1 else 0
        
        # Calculate speech rate safely
        duration = librosa.get_duration(y=y, sr=sr)
        speech_rate = len(librosa.effects.split(y)) / max(duration, 0.1)
        
        # Calculate other features safely
        rms = librosa.feature.rms(y=y)[0]
        shimmer = np.std(rms) if len(rms) > 0 else 0
        
        return {
            "Speech Rate": speech_rate,
            "Pitch": pitch,
            "Jitter": jitter,
            "Shimmer": shimmer,
            "Pauses": len(librosa.effects.split(y, top_db=20)),
            "Sighs": np.mean(librosa.feature.zero_crossing_rate(y=y)[0]) if len(y) > 0 else 0
        }
    except Exception as e:
        logger.error(f"Speech features extraction failed: {e}")
        return {feature: 0.0 for feature in SPEECH_FEATURES}

# **Function: Compute Anxiety Index**
def compute_anxiety_index(emotion_scores, speech_scores):
    """
    Compute anxiety index from emotion and speech scores
    
    Parameters:
        emotion_scores (dict): Dictionary of emotion scores
        speech_scores (dict): Dictionary of speech feature scores
        
    Returns:
        float: Calculated anxiety index
    """
    try:
        # Extract only anxiety-related emotions for the calculation
        anxiety_emotions = {emotion: emotion_scores.get(emotion, 0.0) for emotion in ANXIETY_RELATED_EMOTIONS}
        
        # Define weights for emotions and speech features
        emotion_weights = np.array([1.2, 1.5, 1.3, 1.1, 1.0, 1.0])
        feature_weights = np.array([0.8, 0.6, 1.2, 1.1, 1.0, 0.9])
        
        # Convert dictionaries to arrays in the correct order
        emotion_values = np.array([anxiety_emotions.get(emotion, 0.0) for emotion in ANXIETY_RELATED_EMOTIONS])
        speech_values = np.array([speech_scores.get(feature, 0.0) for feature in SPEECH_FEATURES])
        
        # Ensure arrays have the correct shape
        if len(emotion_values) != len(emotion_weights) or len(speech_values) != len(feature_weights):
            logger.error(f"Dimension mismatch in anxiety index calculation: emotions={len(emotion_values)}, features={len(speech_values)}")
            return 0.0
            
        # Normalize values to prevent extreme scores
        emotion_contribution = np.dot(emotion_values, emotion_weights)
        speech_contribution = np.dot(speech_values, feature_weights)
        
        return emotion_contribution + speech_contribution
    except Exception as e:
        logger.error(f"Anxiety index calculation failed: {e}")
        return 0.0

# **Function: Save Analysis Data**
def save_analysis_data(df):
    """Save the analysis dataframe to disk"""
    try:
        df.to_pickle(ANALYSIS_DATA_FILE)
        logger.info(f"Analysis data saved to {ANALYSIS_DATA_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save analysis data: {e}")
        return False

# **Function: Store Speech Text**
def store_speech_text(df, file_path, text):
    """
    Store speech text in the dataframe
    
    Parameters:
        df (pandas.DataFrame): DataFrame to update
        file_path (str): Path to the audio file
        text (str): Transcribed text
        
    Returns:
        pandas.DataFrame: Updated DataFrame
    """
    if not text.strip():
        return df
        
    try:
        # If the SpeechText column doesn't exist, add it
        if 'SpeechText' not in df.columns:
            df['SpeechText'] = ""
            
        # Find the latest record index
        latest_idx = df.index[-1] if len(df) > 0 else 0
        
        # Store the text
        df.at[latest_idx, 'SpeechText'] = text
        
        return df
    except Exception as e:
        logger.error(f"Failed to store speech text: {e}")
        return df

# **Function: Process and Analyze New Audio File**
def process_audio_file(df, file_path):
    """
    Process and analyze a new audio file
    
    Parameters:
        df (pandas.DataFrame): DataFrame to update
        file_path (Path): Path to the audio file
        
    Returns:
        pandas.DataFrame: Updated DataFrame
        bool: True if processing was successful, False otherwise
    """
    try:
        # Convert audio format if needed
        wav_file = None
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.m4a', '.mp4']:
            wav_file = convert_to_wav(file_path)
        else:
            wav_file = file_path
        
        if not wav_file:
            logger.error(f"Could not process {file_path} - conversion failed")
            return df, False
            
        # Extract text from speech
        text = speech_to_text(wav_file)
        if not text:
            logger.warning(f"No speech detected in {file_path}")
            return df, False

        # Analyze emotions and speech features
        emotion_scores = analyze_emotions(text)
        speech_scores = extract_speech_features(wav_file)
        
        # Calculate anxiety index
        anxiety_index = compute_anxiety_index(emotion_scores, speech_scores)

        # Add to dataframe
        new_data = {"Date": pd.Timestamp.now(), "Anxiety Index": anxiety_index, 
                   "SpeechText": text, **emotion_scores, **speech_scores}
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        logger.info(f"Processed {file_path} -> Anxiety Index: {anxiety_index:.2f}")
        return df, True
    except Exception as e:
        logger.error(f"Failed processing {file_path}: {e}")
        return df, False

# **Function: Print Current Analysis Data**
def print_current_data(df):
    """Print a summary of the current analysis data"""
    if len(df) == 0:
        logger.info("No anxiety data available yet.")
        return
        
    print("\n" + "-"*50)
    print("CURRENT ANXIETY DATA (Last 5 records)")
    print("-"*50)
    
    # Display the last 5 records in a readable format
    display_df = df.sort_values("Date", ascending=False).head(5).copy()
    
    # Format the date column
    display_df["Date"] = display_df["Date"].dt.strftime('%Y-%m-%d %H:%M')
    
    # Select only the most important columns for display
    display_cols = ["Date", "Anxiety Index"] + ANXIETY_RELATED_EMOTIONS[:3]
    
    # Print as a simple table
    print(display_df[display_cols].to_string(index=False))
    print("-"*50)

# **Function: Visualize Data**
def visualize_data(df, only_new=False):
    """
    Visualize all the anxiety data
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the anxiety data
        only_new (bool): If True, only visualize new data; if False, visualize all data
    """
    try:
        # Clean and prepare data
        df = df.copy()
        df.fillna(0, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date", "Anxiety Index"], inplace=True)
        
        if len(df) < 1:
            logger.info("No data points for analysis.")
            return

        # Sort by date for proper time series analysis
        df = df.sort_values(by="Date")
        
        print("\n" + "="*80)
        print("ANXIETY ANALYSIS SUMMARY")
        print("="*80)
        
        recent_data = df.tail(5)
        print(f"\nMost recent anxiety measurements ({len(recent_data)} records):")
        for _, row in recent_data.iterrows():
            print(f"  {row['Date'].strftime('%Y-%m-%d %H:%M')} - Anxiety Index: {row['Anxiety Index']:.2f}")
        
        if len(df) >= 2:
            print(f"\nTrend analysis:")
            first_date = df['Date'].min().strftime('%Y-%m-%d')
            last_date = df['Date'].max().strftime('%Y-%m-%d')
            avg_anxiety = df['Anxiety Index'].mean()
            max_anxiety = df['Anxiety Index'].max()
            max_date = df.loc[df['Anxiety Index'].idxmax(), 'Date'].strftime('%Y-%m-%d %H:%M')
            print(f"  Period: {first_date} to {last_date}")
            print(f"  Average Anxiety Index: {avg_anxiety:.2f}")
            print(f"  Peak Anxiety Index: {max_anxiety:.2f} on {max_date}")
            
            # Calculate trend
            if len(df) >= 3:
                recent_trend = df['Anxiety Index'].tail(3).pct_change().mean() * 100
                if recent_trend > 5:
                    trend_message = "INCREASING (> 5%)"
                elif recent_trend < -5:
                    trend_message = "DECREASING (> 5%)"
                else:
                    trend_message = "STABLE (< 5% change)"
                print(f"  Recent trend: {trend_message}")
        
        # Figure 1: Anxiety Index Trend (ONLY focus on anxiety index)
        plt.figure(figsize=(12, 5))
        sns.lineplot(x=df["Date"], y=df["Anxiety Index"], label="Anxiety Index", marker="o", color="red")
        
        # Calculate rolling average only if enough data points
        if len(df) >= 2:
            window_size = min(7, len(df))
            df["7-day Avg"] = df["Anxiety Index"].rolling(window=window_size, min_periods=1).mean()
            sns.lineplot(x=df["Date"], y=df["7-day Avg"], label=f"{window_size}-day Moving Avg", linestyle="--", color="blue")
        
        plt.xlabel("Date")
        plt.ylabel("Anxiety Index")
        plt.title("Anxiety Index Trend Over Time")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        trend_chart_path = FIGURES_DIR / "anxiety_trend.png"
        plt.savefig(trend_chart_path)
        
        # Display the plot in the terminal output if running in a compatible environment
        try:
            plt.show()
        except Exception as e:
            logger.debug(f"Could not display plot in terminal: {e}")
        finally:
            plt.close()
        
        # Daily Emotion Pie Charts (using ALL emotions, not just anxiety-related ones)
        if len(df) > 0:
            try:
                # Create pie chart for the most recent record
                plt.figure(figsize=(12, 10))
                
                # Get the most recent entry's emotions
                latest_emotions = df[EMOTIONS_TO_TRACK].iloc[-1]
                
                # Filter out emotions with very low values
                significant_emotions = {k: v for k, v in latest_emotions.items() if v > 0.5}
                
                # If all values are too small, force include at least the top 5
                if not significant_emotions or sum(significant_emotions.values()) < 2:
                    # Get top 5 emotions regardless of value
                    top_emotions = latest_emotions.nlargest(5)
                    significant_emotions = {k: max(v, 1.0) for k, v in top_emotions.items()}
                
                # Ensure we have at least some data to display
                if not significant_emotions:
                    # Create dummy data if no emotions detected
                    significant_emotions = {"Neutral": 100.0}
                
                # Create labels with percentages
                labels = [f"{emotion} ({value:.1f}%)" for emotion, value in significant_emotions.items()]
                sizes = list(significant_emotions.values())
                
                # Create pie chart with explicit color map
                colors = plt.cm.tab20.colors[:len(sizes)] if len(sizes) <= 20 else plt.cm.viridis(np.linspace(0, 1, len(sizes)))
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                plt.axis('equal')
                
                # Add title with date information
                latest_date = df['Date'].iloc[-1].strftime('%Y-%m-%d %H:%M')
                plt.title(f"Emotion Distribution - Latest Recording ({latest_date})")
                
                # Add annotation about anxiety-related emotions
                anxiety_related = [emotion for emotion in significant_emotions.keys() 
                                if emotion in ANXIETY_RELATED_EMOTIONS]
                anxiety_note = "Anxiety-related emotions: " + (", ".join(anxiety_related) if anxiety_related else "None detected")
                plt.annotate(anxiety_note, xy=(0, -0.1), xycoords='axes fraction', 
                            ha='left', va='center', fontsize=10)
                
                # Save pie chart with explicit format
                pie_chart_path = FIGURES_DIR / "latest_emotion_pie_chart.png"
                plt.savefig(str(pie_chart_path), format='png', dpi=300)
                logger.info(f"Emotion pie chart saved to {pie_chart_path}")
                
                # Display the plot
                try:
                    plt.show()
                except Exception as e:
                    logger.debug(f"Could not display pie chart: {e}")
                finally:
                    plt.close()
            except Exception as e:
                logger.error(f"Failed to generate emotion pie chart: {e}")
                plt.close('all')  # Close any open figures in case of error
            
        # Figure 3: Speech Feature Analysis
        if len(df) > 0:
            plt.figure(figsize=(12, 6))
            speech_data = df[SPEECH_FEATURES].iloc[-min(10, len(df)):]
            
            # Scale the features for better visualization
            for feature in SPEECH_FEATURES:
                if speech_data[feature].max() > 0:
                    speech_data[feature] = speech_data[feature] / speech_data[feature].max() * 100
                    
            speech_data.index = df["Date"].iloc[-min(10, len(df)):].dt.strftime('%Y-%m-%d %H:%M')
            sns.heatmap(speech_data.T, cmap="Blues", annot=True, fmt=".1f", linewidths=.5)
            plt.title("Normalized Speech Features (Most Recent Records)")
            plt.ylabel("Feature")
            plt.xlabel("Date/Time")
            plt.tight_layout()
            speech_chart_path = FIGURES_DIR / "speech_features.png"
            plt.savefig(speech_chart_path)
            
            # Print speech features summary
            print("\nSpeech Features (most recent record):")
            latest_features = df[SPEECH_FEATURES].iloc[-1]
            for feature, value in latest_features.items():
                print(f"  {feature}: {value:.2f}")
            
            # Display the plot
            try:
                plt.show()
            except Exception:
                pass
            finally:
                plt.close()
        
        # Emotion summary in text
        if len(df) > 0:
            latest_emotions = df[EMOTIONS_TO_TRACK].iloc[-1]
            significant_emotions = {k: v for k, v in latest_emotions.items() if v > 1.0}
            
            print("\nEmotion Analysis (most recent record):")
            for emotion, value in sorted(significant_emotions.items(), key=lambda x: x[1], reverse=True):
                emotion_type = "(anxiety-related)" if emotion in ANXIETY_RELATED_EMOTIONS else ""
                print(f"  {emotion}: {value:.2f}% {emotion_type}")
                
            # Highlight anxiety-related emotions
            anxiety_emotions = {emotion: value for emotion, value in latest_emotions.items() 
                              if emotion in ANXIETY_RELATED_EMOTIONS and value > 0.5}
            if anxiety_emotions:
                print("\nAnxiety-related emotions detected:")
                for emotion, value in sorted(anxiety_emotions.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {emotion}: {value:.2f}%")
        
        # Day-specific Word Frequency Histogram
        try:
            if 'SpeechText' in df.columns and len(df) > 0:
                # Try both today and the last 7 days to ensure we have data
                today = pd.Timestamp.now().floor('D')  # Floor to start of day
                
                # First try analyzing just today's data
                daily_phrases = analyze_daily_phrases(df, today)
                
                # If no phrases found for today, try the last record's date
                if not daily_phrases and len(df) > 0:
                    last_date = df['Date'].iloc[-1].floor('D')
                    daily_phrases = analyze_daily_phrases(df, last_date)
                    logger.info(f"Using last record date ({last_date.strftime('%Y-%m-%d')}) for word analysis")
                
                # If still no data, try the last 7 days
                if not daily_phrases:
                    # Get data from the last 7 days
                    week_ago = today - pd.Timedelta(days=7)
                    week_data = df[(df['Date'] >= week_ago) & (df['Date'] <= today)]
                    
                    if len(week_data) > 0 and 'SpeechText' in week_data.columns:
                        # Collect all text from the last 7 days
                        all_text = week_data['SpeechText'].dropna().tolist()
                        all_text = [t for t in all_text if isinstance(t, str) and t.strip()]
                        
                        if all_text:
                            daily_phrases = analyze_common_phrases(all_text, top_n=20)
                            logger.info("Using last 7 days data for word analysis")
                
                if daily_phrases:
                    plt.figure(figsize=(12, 8))
                    
                    # Sort by frequency
                    words = list(daily_phrases.keys())
                    frequencies = list(daily_phrases.values())
                    
                    # Sort from highest to lowest
                    sorted_indices = sorted(range(len(frequencies)), key=lambda i: frequencies[i], reverse=True)
                    words = [words[i] for i in sorted_indices]
                    frequencies = [frequencies[i] for i in sorted_indices]
                    
                    # Cap at 15 words for better visualization
                    if len(words) > 15:
                        words = words[:15]
                        frequencies = frequencies[:15]
                    
                    # Create histogram with a colormap
                    colors = plt.cm.Blues(np.linspace(0.6, 1.0, len(words)))
                    plt.barh(words, frequencies, color=colors)
                    plt.xlabel('Frequency')
                    plt.ylabel('Word')
                    
                    # Set appropriate title based on what data we used
                    if daily_phrases:
                        plt.title(f'Most Frequent Words for {today.strftime("%Y-%m-%d")}')
                    
                    plt.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    
                    # Save histogram with explicit format
                    word_hist_path = FIGURES_DIR / "daily_word_frequency.png"
                    plt.savefig(str(word_hist_path), format='png', dpi=300)
                    logger.info(f"Word frequency histogram saved to {word_hist_path}")
                    
                    # Print common words
                    print(f"\nMost common words found:")
                    for word, freq in zip(words[:10], frequencies[:10]):
                        print(f"  {word}: {freq} occurrences")
                    
                    # Display the plot
                    try:
                        plt.show()
                    except Exception as e:
                        logger.debug(f"Could not display histogram: {e}")
                    finally:
                        plt.close()
                else:
                    logger.info(f"No significant phrases found in the data")
            else:
                logger.info("No speech text available for word analysis")
        except Exception as e:
            logger.error(f"Failed to generate word frequency histogram: {e}")
            plt.close('all')  # Close any open figures in case of error
        
        logger.info(f"Charts saved to {FIGURES_DIR}")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to visualize data: {e}")

def main():
    """Main function to run the anxiety monitoring system"""
    global anxiety_data
    
    logger.info("Firebase Storage monitoring started...")

    # Load the list of already downloaded files
    load_downloaded_files()

    # Display current data on startup if available
    print_current_data(anxiety_data)
    
    # Display anxiety prediction
    if len(anxiety_data) >= 3:
        logger.info("Generating anxiety prediction...")
        display_anxiety_prediction(anxiety_data)

    # Visualize on startup if data exists - using ALL historical data
    if len(anxiety_data) > 0:
        logger.info("Generating initial visualizations using all historical data...")
        visualize_data(anxiety_data)

    logger.info("Entering continuous monitoring loop...")
    logger.info("Will check for new files every 30 seconds")

    try:
        while True:
            downloaded_batch = []

            try:
                # List all files in Firebase storage
                all_files = list_files()
                
                # Find files that haven't been processed yet
                new_files = [file for file in all_files if file not in downloaded_files]
                
                if new_files:
                    logger.info(f"Found {len(new_files)} new files to process")
                    
                    # Download and process each new file
                    for file in new_files:
                        local_file = download_file(file)
                        if local_file:
                            downloaded_batch.append(local_file)
                
                    if downloaded_batch:
                        logger.info(f"Processing {len(downloaded_batch)} files...")
                        processed_count = 0
                        current_data = anxiety_data.copy()
                        
                        for file in downloaded_batch:
                            current_data, success = process_audio_file(current_data, file)
                            if success:
                                processed_count += 1
                        
                        if processed_count > 0:
                            # Update the dataframe
                            anxiety_data = current_data
                            
                            # Save data
                            save_analysis_data(anxiety_data)
                            save_downloaded_files()
                            
                            # Display updated data - INCLUDING all historical data
                            print_current_data(anxiety_data)
                            
                            # Visualize ALL data (both historical and new)
                            logger.info("Generating visualizations with all historical + new data...")
                            visualize_data(anxiety_data)
                            
                            # Generate new prediction after processing new data
                            if len(anxiety_data) >= 3:
                                logger.info("Updating anxiety prediction with new data...")
                                display_anxiety_prediction(anxiety_data)
                            
                            logger.info(f"Successfully processed {processed_count} of {len(downloaded_batch)} files")
                        else:
                            logger.warning("No files were successfully processed in this batch")
                    else:
                        logger.info("No new files were successfully downloaded")
                else:
                    logger.info("No new files to process")
                
            except Exception as e:
                logger.error(f"Failed in main loop: {e}")
            
            # Clear message about waiting for next check
            print(f"\nINFO: Waiting for 30 seconds before next check... (Press Ctrl+C to exit)")
            print("-" * 50)
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Exiting monitoring loop due to user interrupt")
        save_analysis_data(anxiety_data)
        save_downloaded_files()
        logger.info("Data saved. Exiting.")

if __name__ == "__main__":
    main()
