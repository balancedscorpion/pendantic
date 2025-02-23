import spacy
import numpy as np
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
from datetime import datetime
from scipy.stats import entropy
from sklearn.preprocessing import normalize
from src.get_conversation import get_all_conversations, merge_same_speaker_sections
import json

class SemanticEntropyAgent:
    def __init__(self):
        # Load English language model with word vectors
        self.nlp = spacy.load("en_core_web_lg")
        
    def compute_local_entropy(self, text: str) -> float:
        """
        Compute semantic entropy within the given text using word embeddings.
        Higher entropy indicates more diverse/unpredictable semantic content.
        """
        doc = self.nlp(text)
        
        # Get word vectors for content words only
        vectors = [token.vector for token in doc if token.has_vector and not token.is_stop and not token.is_punct]
        
        if not vectors or len(vectors) < 2:  # Need at least 2 words for meaningful comparison
            return 0.0
            
        # Convert vectors to probability distribution using cosine similarity
        vectors = np.array(vectors)
        # Normalize vectors first for better cosine similarity
        vectors = normalize(vectors, norm='l2', axis=1)
        similarity_matrix = np.dot(vectors, vectors.T)
        
        # Ensure non-negative values and proper probability distribution
        similarity_matrix = (similarity_matrix + 1) / 2  # Scale from [-1,1] to [0,1]
        # Normalize rows to sum to 1, avoiding division by zero
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        prob_dist = np.where(row_sums > 0, similarity_matrix / row_sums, 0)
        
        # Calculate entropy for each word and take average
        word_entropies = [entropy(dist) for dist in prob_dist]
        return round(np.mean(word_entropies), 4)
        
    def compute_global_entropy(self, text: str) -> float:
        """
        Compute semantic entropy relative to general language distribution.
        Uses pre-trained word vectors as proxy for global semantics.
        """
        doc = self.nlp(text)
        
        # Get vectors for content words
        vectors = [token.vector for token in doc if token.has_vector and not token.is_stop and not token.is_punct]
        
        if not vectors:
            return 0.0
            
        # Compare against "average" word vector as baseline
        vectors = np.array(vectors)
        mean_vector = vectors.mean(axis=0)
        
        # Calculate divergence from mean
        distances = np.linalg.norm(vectors - mean_vector, axis=1)
        # Convert to probability distribution
        prob_dist = distances / np.sum(distances)
        
        return round(entropy(prob_dist), 4)

    def analyze_conversation(self, messages: List[Dict[Any, Any]]) -> Dict:
        """
        Analyze semantic entropy trends in a conversation over time.
        """
        results = []
        local_entropies = []
        global_entropies = []
        
        for message in messages:
            if 'content' not in message or not message['content']:
                continue
                
            timestamp = datetime.fromtimestamp(message['timestamp'])
            # Compute entropy for current message
            local_ent = self.compute_local_entropy(message['content'])
            global_ent = self.compute_global_entropy(message['content'])
            
            # Add to running lists for average calculation
            local_entropies.append(local_ent)
            global_entropies.append(global_ent)
            
            # Calculate true running averages
            avg_local_ent = np.mean(local_entropies)
            avg_global_ent = np.mean(global_entropies)
            
            results.append({
                'timestamp': timestamp,
                'local_entropy': local_ent,
                'global_entropy': global_ent,
                'avg_local_entropy': avg_local_ent,
                'avg_global_entropy': avg_global_ent,
                'speaker': message['role'],
                'message': message['content'][:100] + '...' if len(message['content']) > 100 else message['content']
            })
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Create separate traces for individual contributions and overall average
        local_fig = px.scatter(df, x='timestamp', y='local_entropy', color='speaker',
                             title='Local Semantic Entropy Over Time',
                             labels={
                                 'timestamp': 'Time',
                                 'local_entropy': 'Local Entropy',
                                 'speaker': 'Speaker'
                             })
        
        # Add overall average line
        local_fig.add_scatter(x=df['timestamp'], y=df['avg_local_entropy'],
                            name='Running Average',
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=True)
                           
        global_fig = px.scatter(df, x='timestamp', y='global_entropy', color='speaker',
                              title='Global Semantic Entropy Over Time',
                              labels={
                                  'timestamp': 'Time',
                                  'global_entropy': 'Global Entropy',
                                  'speaker': 'Speaker'
                              })
                              
        # Add overall average line
        global_fig.add_scatter(x=df['timestamp'], y=df['avg_global_entropy'],
                             name='Running Average',
                             line=dict(color='black', width=2, dash='dash'),
                             showlegend=True)
        
        # Update hover data
        local_fig.update_traces(hovertemplate='%{x}<br>Entropy: %{y:.4f}<br>%{customdata[0]}<extra></extra>',
                              customdata=df[['message']].values)
        global_fig.update_traces(hovertemplate='%{x}<br>Entropy: %{y:.4f}<br>%{customdata[0]}<extra></extra>',
                               customdata=df[['message']].values)
        
        # Calculate average entropy per speaker
        avg_local = df.groupby('speaker')['local_entropy'].agg(['mean', 'std']).round(4)
        avg_global = df.groupby('speaker')['global_entropy'].agg(['mean', 'std']).round(4)
        
        return {
            'local_plot': local_fig,
            'global_plot': global_fig,
            'average_local': avg_local.to_dict('index'),
            'average_global': avg_global.to_dict('index'),
            'raw_data': df.to_dict('records')
        }

    def analyze_all_conversations(self) -> Dict:
        """
        Analyze semantic entropy across all conversations in the database.
        """
        all_results = []
        cumulative_text = ""
        data = get_all_conversations()
        conversations = merge_same_speaker_sections(data)
        
        for idx, conv in enumerate(conversations):
            conv_dict = conv.dict()
            transcript = conv_dict['transcript']
            
            # Compute entropy for current message
            local_ent = self.compute_local_entropy(transcript)
            global_ent = self.compute_global_entropy(transcript)
            
            # Update cumulative text and compute running average entropy
            cumulative_text += " " + transcript
            avg_local_ent = self.compute_local_entropy(cumulative_text)
            avg_global_ent = self.compute_global_entropy(cumulative_text)
            
            all_results.append({
                'utterance_number': idx + 1,
                'timestamp': conv_dict['timestamp'],
                'local_entropy': local_ent,
                'global_entropy': global_ent,
                'avg_local_entropy': avg_local_ent,
                'avg_global_entropy': avg_global_ent,
                'speaker': conv_dict['speaker'],
                'message': transcript[:100] + '...' if len(transcript) > 100 else transcript
            })
        
        df = pd.DataFrame(all_results)
        
        # Create separate traces for individual contributions and overall average
        local_fig = px.scatter(df, x='utterance_number', y='local_entropy', color='speaker',
                             title='Local Semantic Entropy Across All Conversations',
                             labels={
                                 'utterance_number': 'Utterance Number',
                                 'local_entropy': 'Local Entropy',
                                 'speaker': 'Speaker'
                             })
        
        # Add overall average line
        local_fig.add_scatter(x=df['utterance_number'], y=df['avg_local_entropy'],
                            name='Conversation Average',
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=True)
                           
        global_fig = px.scatter(df, x='utterance_number', y='global_entropy', color='speaker',
                              title='Global Semantic Entropy Across All Conversations',
                              labels={
                                  'utterance_number': 'Utterance Number',
                                  'global_entropy': 'Global Entropy',
                                  'speaker': 'Speaker'
                              })
                              
        # Add overall average line
        global_fig.add_scatter(x=df['utterance_number'], y=df['avg_global_entropy'],
                             name='Conversation Average',
                             line=dict(color='black', width=2, dash='dash'),
                             showlegend=True)
        
        # Update hover data
        local_fig.update_traces(hovertemplate='%{x}<br>Entropy: %{y:.4f}<br>%{customdata[0]}<extra></extra>',
                              customdata=df[['message']].values)
        global_fig.update_traces(hovertemplate='%{x}<br>Entropy: %{y:.4f}<br>%{customdata[0]}<extra></extra>',
                               customdata=df[['message']].values)
        
        avg_local = df.groupby('speaker')['local_entropy'].agg(['mean', 'std']).round(4)
        avg_global = df.groupby('speaker')['global_entropy'].agg(['mean', 'std']).round(4)
        
        return {
            'local_plot': local_fig,
            'global_plot': global_fig,
            'average_local': avg_local.to_dict('index'),
            'average_global': avg_global.to_dict('index'),
            'raw_data': df.to_dict('records')
        }

    def analyze_conversation_range(self, start_id: int, end_id: int) -> Dict:
        """
        Analyze semantic entropy for a range of conversations from the database.
        """
        data = get_all_conversations()
        conversations = merge_same_speaker_sections(data)
        return self._analyze_conversation_list(conversations, start_id, end_id)

    def _analyze_conversation_list(self, conversations: List[Any], start_id: int, end_id: int) -> Dict:
        """
        Internal method to analyze a list of conversations from any source.
        """
        filtered_conversations = conversations[start_id:end_id+1]
        
        if not filtered_conversations:
            return {
                'error': f'No valid conversations found in range {start_id}-{end_id}'
            }
        
        all_results = []
        local_entropies = []
        global_entropies = []
        
        for idx, conv in enumerate(filtered_conversations):
            if isinstance(conv, dict):
                transcript = conv['transcript']
                speaker = conv['speaker']
                timestamp = datetime.fromisoformat(conv['timestamp'].replace('Z', '+00:00'))
            else:
                conv_dict = conv.dict()
                transcript = conv_dict['transcript']
                speaker = conv_dict['speaker']
                timestamp = conv_dict['timestamp']
            
            # Compute entropy for current message
            local_ent = self.compute_local_entropy(transcript)
            global_ent = self.compute_global_entropy(transcript)
            
            # Add to running lists for average calculation
            local_entropies.append(local_ent)
            global_entropies.append(global_ent)
            
            # Calculate true running averages
            avg_local_ent = np.mean(local_entropies)
            avg_global_ent = np.mean(global_entropies)
            
            all_results.append({
                'utterance_number': idx + 1,
                'timestamp': timestamp,
                'local_entropy': local_ent,
                'global_entropy': global_ent,
                'avg_local_entropy': avg_local_ent,
                'avg_global_entropy': avg_global_ent,
                'speaker': speaker,
                'message': transcript[:100] + '...' if len(transcript) > 100 else transcript
            })
            
        df = pd.DataFrame(all_results)
        
        # Create separate traces for individual contributions and overall average
        local_fig = px.scatter(df, x='utterance_number', y='local_entropy', color='speaker',
                             title=f'Local Semantic Entropy for Conversations {start_id}-{end_id}',
                             labels={
                                 'utterance_number': 'Utterance Number',
                                 'local_entropy': 'Local Entropy',
                                 'speaker': 'Speaker'
                             })
        
        # Add overall average line
        local_fig.add_scatter(x=df['utterance_number'], y=df['avg_local_entropy'],
                            name='Running Average',
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=True)
                           
        global_fig = px.scatter(df, x='utterance_number', y='global_entropy', color='speaker',
                              title=f'Global Semantic Entropy for Conversations {start_id}-{end_id}',
                              labels={
                                  'utterance_number': 'Utterance Number',
                                  'global_entropy': 'Global Entropy',
                                  'speaker': 'Speaker'
                              })
                              
        # Add overall average line
        global_fig.add_scatter(x=df['utterance_number'], y=df['avg_global_entropy'],
                             name='Running Average',
                             line=dict(color='black', width=2, dash='dash'),
                             showlegend=True)
        
        # Update hover data
        local_fig.update_traces(hovertemplate='%{x}<br>Entropy: %{y:.4f}<br>%{customdata[0]}<extra></extra>',
                              customdata=df[['message']].values)
        global_fig.update_traces(hovertemplate='%{x}<br>Entropy: %{y:.4f}<br>%{customdata[0]}<extra></extra>',
                               customdata=df[['message']].values)
        
        avg_local = df.groupby('speaker')['local_entropy'].agg(['mean', 'std']).round(4)
        avg_global = df.groupby('speaker')['global_entropy'].agg(['mean', 'std']).round(4)
        
        return {
            'local_plot': local_fig,
            'global_plot': global_fig,
            'average_local': avg_local.to_dict('index'),
            'average_global': avg_global.to_dict('index'),
            'raw_data': df.to_dict('records')
        }

    def analyze_json_file(self, json_file: str, start_id: int = 0, end_id: int = None) -> Dict:
        """
        Analyze semantic entropy for conversations from a JSON file.
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
