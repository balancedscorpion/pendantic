from typing import List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
import os
import requests

SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
if not SUPABASE_API_KEY:
    raise EnvironmentError("Missing environment variable 'SUPABASE_API_KEY'.")

PROJECT_ID = "mqaiuwpvphctupwtvidm"
# Construct the Supabase REST URL using the PROJECT_ID
SUPABASE_URL = f"https://{PROJECT_ID}.supabase.co/rest/v1/conversations"

def get_all_conversations():
    """
    Retrieves all rows from the 'conversations' table via GET.
    """
    url = f"{SUPABASE_URL}?select=*"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to retrieve conversations from Supabase: {e}")
        return []



class TranscriptSection(BaseModel):
    id: UUID
    speaker: str
    transcript: str
    timestamp: datetime
    created_at: datetime
    updated_at: datetime


def merge_same_speaker_sections(data: List[dict]) -> List[TranscriptSection]:
    """
    Ingests a list of transcript dictionaries, validates them with Pydantic,
    and merges consecutive items that share the same speaker.

    :param data: A list of dicts, each representing transcript data.
    :return: A list of TranscriptSection objects with merged transcripts
             for consecutive sections of the same speaker.
    """
    # Validate each dict with the TranscriptSection model
    validated_sections = [TranscriptSection(**item) for item in data]

    merged_sections: List[TranscriptSection] = []
    for section in validated_sections:
        if not merged_sections:
            # First entry, just add it
            merged_sections.append(section)
        else:
            # Compare speaker with the last merged section
            last_section = merged_sections[-1]
            if last_section.speaker == section.speaker:
                # Merge transcript text (append with a space or any desired delimiter)
                merged_text = f"{last_section.transcript} {section.transcript}"
                merged_sections[-1] = last_section.copy(update={"transcript": merged_text})
            else:
                # Different speaker, just append
                merged_sections.append(section)

    return merged_sections

def get_output_as_string():
    data = get_all_conversations()
    merged = merge_same_speaker_sections(data)
    
    lines = []
    for m in merged:
        d = m.dict()
        line = f"{d['speaker']} {d['transcript']}"
        lines.append(line)
        
    return "\n".join(lines)



if __name__ == "__main__":
    # Example usage
   print(get_output_as_string())