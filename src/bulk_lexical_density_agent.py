import spacy
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
from datetime import datetime
from src.lexical_density_agent import LexicalDensityAgent
from src.get_conversation import get_all_conversations, merge_same_speaker_sections
import numpy as np
import json
from pydantic import BaseModel

class BulkLexicalDensityAgent(LexicalDensityAgent):
    def _analyze_conversations(self, conversations: List[Any], start_id: int, end_id: int, bin_size: int = 5) -> Dict:
        """
        Internal method to analyze conversations from any source.
        """
        # Filter conversations within the range
        filtered_conversations = conversations[start_id:end_id+1]
        
        if not filtered_conversations:
            return {
                'error': f'No valid conversations found in range {start_id}-{end_id}'
            }
        
        # Calculate number of bins
        num_conversations = len(filtered_conversations)
        num_bins = (num_conversations + bin_size - 1) // bin_size  # Ceiling division
        
        results = []
        for bin_idx in range(num_bins):
            start_idx = bin_idx * bin_size
            end_idx = min((bin_idx + 1) * bin_size, num_conversations)
            
            # Combine transcripts in this bin
            if isinstance(filtered_conversations[0], dict):
                # Handle JSON format
                bin_text = " ".join(conv['transcript'] for conv in filtered_conversations[start_idx:end_idx])
            else:
                # Handle Pydantic model format
                bin_text = " ".join(conv.dict()['transcript'] for conv in filtered_conversations[start_idx:end_idx])
            
            bin_density = self.calculate_lexical_density(bin_text)
            
            results.append({
                'bin_number': bin_idx + 1,
                'start_utterance': start_idx + start_id,
                'end_utterance': end_idx + start_id - 1,
                'lexical_density': bin_density,
                'num_utterances': end_idx - start_idx
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate overall statistics
        mean_density = df['lexical_density'].mean()
        std_density = df['lexical_density'].std()
        
        # Create a line plot showing density over bins
        fig = px.line(df, x='bin_number', y='lexical_density',
                     title=f'Binned Lexical Density (bin size={bin_size}) for Conversations {start_id}-{end_id}',
                     labels={
                         'bin_number': 'Bin Number',
                         'lexical_density': 'Lexical Density (%)'
                     })
        
        # Add markers to the line plot
        fig.update_traces(mode='lines+markers', marker=dict(size=8))
        
        # Add hover data showing utterance ranges
        fig.update_traces(
            hovertemplate="Bin %{x}<br>" +
            "Density: %{y:.2f}%<br>" +
            "Utterances: %{customdata[0]}-%{customdata[1]}<extra></extra>",
            customdata=df[['start_utterance', 'end_utterance']].values
        )
        
        # Add a horizontal line for the mean
        fig.add_hline(y=mean_density, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_density:.2f}%",
                     annotation_position="bottom right")
        
        return {
            'plot': fig,
            'average_density': {
                'Binned Analysis': {
                    'mean': round(mean_density, 2),
                    'std': round(std_density, 2)
                }
            },
            'raw_data': df.to_dict('records')
        }

    def analyze_conversation_range_bulk(self, start_id: int, end_id: int, bin_size: int = 5) -> Dict:
        """
        Analyze lexical density for a range of conversations from the database.
        """
        data = get_all_conversations()
        conversations = merge_same_speaker_sections(data)
        return self._analyze_conversations(conversations, start_id, end_id, bin_size)

    def analyze_json_file_bulk(self, json_file: str, start_id: int = 0, end_id: int = None, bin_size: int = 5) -> Dict:
        """
        Analyze lexical density for conversations from a JSON file.
        
        Args:
            json_file (str): Path to the JSON file containing conversations
            start_id (int): Start conversation ID (default: 0)
            end_id (int): End conversation ID (default: None, meaning all conversations)
            bin_size (int): Number of utterances to combine in each bin
        """
        try:
            with open(json_file, 'r') as f:
                conversations = json.load(f)
            
            if end_id is None:
                end_id = len(conversations) - 1
                
            return self._analyze_conversations(conversations, start_id, end_id, bin_size)
        except FileNotFoundError:
            return {'error': f'File not found: {json_file}'}
        except json.JSONDecodeError:
            return {'error': f'Invalid JSON file: {json_file}'}
        except Exception as e:
            return {'error': f'Error processing file {json_file}: {str(e)}'}
