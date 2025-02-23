import spacy
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
from datetime import datetime
from src.get_conversation import get_all_conversations, get_output_as_string, merge_same_speaker_sections
import json

class LexicalDensityAgent:
    def __init__(self):
        # Load English language model
        self.nlp = spacy.load("en_core_web_sm")
        
    def calculate_lexical_density(self, text: str) -> float:
        """
        Calculate lexical density of given text.
        Lexical density = (number of content words / total number of words) * 100
        """
        doc = self.nlp(text)
        
        # Count content words (nouns, verbs, adjectives, adverbs)
        content_words = sum(1 for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'])
        total_words = len([token for token in doc if not token.is_punct and not token.is_space])
        
        if total_words == 0:
            return 0.0
            
        lexical_density = (content_words / total_words) * 100
        return round(lexical_density, 2)

    def analyze_conversation(self, messages: List[Dict[Any, Any]]) -> Dict:
        """
        Analyze lexical density trends in a conversation over time.
        """
        results = []
        
        for message in messages:
            if 'content' not in message or not message['content']:
                continue
                
            timestamp = datetime.fromtimestamp(message['timestamp'])
            density = self.calculate_lexical_density(message['content'])
            
            results.append({
                'timestamp': timestamp,
                'lexical_density': density,
                'speaker': message['role'],
                'message': message['content'][:100] + '...' if len(message['content']) > 100 else message['content']
            })
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Create timeline plot
        fig = px.line(df, x='timestamp', y='lexical_density', color='speaker',
                     title='Lexical Density Over Time',
                     labels={'timestamp': 'Time', 'lexical_density': 'Lexical Density (%)', 'speaker': 'Speaker'},
                     hover_data=['message'])
        
        # Calculate average density per speaker
        avg_density = df.groupby('speaker')['lexical_density'].agg(['mean', 'std']).round(2)
        
        return {
            'plot': fig,
            'average_density': avg_density.to_dict('index'),
            'raw_data': df.to_dict('records')
        }

    def analyze_conversation_by_id(self, conversation_id: str) -> Dict:
        """
        Analyze lexical density for a specific conversation using its ID.
        """
        # Comment out or remove this method since we don't have get_conversation_messages
        raise NotImplementedError("This method is not currently supported")

    def analyze_all_conversations(self) -> Dict:
        """
        Analyze lexical density across all conversations in the database.
        """
        all_results = []
        data = get_all_conversations()
        conversations = merge_same_speaker_sections(data)
        
        # Add utterance number to track sequential order
        for idx, conv in enumerate(conversations):
            conv_dict = conv.dict()
            density = self.calculate_lexical_density(conv_dict['transcript'])
            
            all_results.append({
                'utterance_number': idx + 1,  # 1-based indexing
                'timestamp': conv_dict['timestamp'],  # keep timestamp for reference
                'lexical_density': density,
                'speaker': conv_dict['speaker'],
                'message': conv_dict['transcript'][:100] + '...' if len(conv_dict['transcript']) > 100 else conv_dict['transcript']
            })
        
        # Convert to DataFrame - no need to sort since we're using utterance number
        df = pd.DataFrame(all_results)
        
        # Create timeline plot for all conversations
        fig = px.line(df, x='utterance_number', y='lexical_density', color='speaker',
                     title='Lexical Density Across All Conversations',
                     labels={
                         'utterance_number': 'Utterance Number', 
                         'lexical_density': 'Lexical Density (%)', 
                         'speaker': 'Speaker'
                     },
                     hover_data=['message', 'timestamp'])
        
        # Calculate overall statistics per speaker
        avg_density = df.groupby('speaker')['lexical_density'].agg(['mean', 'std']).round(2)
        
        return {
            'plot': fig,
            'average_density': avg_density.to_dict('index'),
            'raw_data': df.to_dict('records')
        }

    def _analyze_conversation_list(self, conversations: List[Any], start_id: int, end_id: int) -> Dict:
        """
        Internal method to analyze a list of conversations from any source.
        """
        # Filter conversations within the range
        filtered_conversations = conversations[start_id:end_id+1]
        
        if not filtered_conversations:
            return {
                'error': f'No valid conversations found in range {start_id}-{end_id}'
            }
        
        all_results = []
        # Add utterance number to track sequential order
        for idx, conv in enumerate(filtered_conversations):
            # Handle both dict and Pydantic model formats
            if isinstance(conv, dict):
                transcript = conv['transcript']
                speaker = conv['speaker']
                timestamp = datetime.fromisoformat(conv['timestamp'].replace('Z', '+00:00'))
            else:
                conv_dict = conv.dict()
                transcript = conv_dict['transcript']
                speaker = conv_dict['speaker']
                timestamp = conv_dict['timestamp']
            
            density = self.calculate_lexical_density(transcript)
            
            all_results.append({
                'utterance_number': idx + 1,  # 1-based indexing
                'timestamp': timestamp,
                'lexical_density': density,
                'speaker': speaker,
                'message': transcript[:100] + '...' if len(transcript) > 100 else transcript
            })
            
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Create timeline plot
        fig = px.line(df, x='utterance_number', y='lexical_density', color='speaker',
                     title=f'Lexical Density for Conversations {start_id}-{end_id}',
                     labels={
                         'utterance_number': 'Utterance Number', 
                         'lexical_density': 'Lexical Density (%)', 
                         'speaker': 'Speaker'
                     },
                     hover_data=['message', 'timestamp'])
        
        # Calculate overall statistics per speaker
        avg_density = df.groupby('speaker')['lexical_density'].agg(['mean', 'std']).round(2)
        
        return {
            'plot': fig,
            'average_density': avg_density.to_dict('index'),
            'raw_data': df.to_dict('records')
        }

    def analyze_conversation_range(self, start_id: int, end_id: int) -> Dict:
        """
        Analyze lexical density for a range of conversations from the database.
        """
        data = get_all_conversations()
        conversations = merge_same_speaker_sections(data)
        return self._analyze_conversation_list(conversations, start_id, end_id)

    def analyze_json_file(self, json_file: str, start_id: int = 0, end_id: int = None) -> Dict:
        """
        Analyze lexical density for conversations from a JSON file.
        
        Args:
            json_file (str): Path to the JSON file containing conversations
            start_id (int): Start conversation ID (default: 0)
            end_id (int): End conversation ID (default: None, meaning all conversations)
        """
        try:
            with open(json_file, 'r') as f:
                conversations = json.load(f)
            
            if end_id is None:
                end_id = len(conversations) - 1
                
            return self._analyze_conversation_list(conversations, start_id, end_id)
        except FileNotFoundError:
            return {'error': f'File not found: {json_file}'}
        except json.JSONDecodeError:
            return {'error': f'Invalid JSON file: {json_file}'}
        except Exception as e:
            return {'error': f'Error processing file {json_file}: {str(e)}'}
