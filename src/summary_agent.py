from get_conversation import get_output_as_string, merge_same_speaker_sections, get_all_conversations
from litellm import completion


conversation_history=get_output_as_string()

def get_conversation_history(n=None):
    data=get_all_conversations()
    all_conversations = merge_same_speaker_sections(data)

    if n is None:
        return [(i.dict()['speaker'],i.dict()['transcript']) for i in all_conversations]
    else:
        return [(i.dict()['speaker'],i.dict()['transcript']) for i in all_conversations[:n]]


conversation_history=get_conversation_history(10)

system_prompt = """You are reviewing a conversation among multiple participants.

    Your goal is to produce a JSON array containing the objective of the conversation.

    The output:
    - MUST be valid JSON conforming to the schema below:
      [
        {
          "Objective": "some string"
        }
      ]
    - MUST NOT include additional commentary or formatting."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": conversation_history}
]

import os
import openai

client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

response = client.chat.completions.create(
    model="Meta-Llama-3.3-70B-Instruct",
    messages=messages,
    response_format={"type": "json_object"},
    temperature=0.1,
    top_p=0.1
)

print(response.choices[0].message.content)