from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# Preprocess text (username and link placeholders)

from get_conversation import get_output_as_string, merge_same_speaker_sections, get_all_conversations
from litellm import completion


conversation_history=get_output_as_string()

def get_conversation_history(n=None, o=None):
    data=get_all_conversations()
    all_conversations = merge_same_speaker_sections(data)

    if n is None:
        if o is None:
            return [(i.dict()['speaker'],i.dict()['transcript']) for i in all_conversations]
        else:
            return [(i.dict()['speaker'],i.dict()['transcript']) for i in all_conversations[:o]]
    else:
        if o is None:
            return [(i.dict()['speaker'],i.dict()['transcript']) for i in all_conversations[:n]]
        else:
            return [(i.dict()['speaker'],i.dict()['transcript']) for i in all_conversations[:n][:o]]


conversation_history=get_conversation_history(10, 50)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#model.save_pretrained(MODEL)
text = str(conversation_history)
text = preprocess(text)
print(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
ranking = np.argsort(scores)
ranking = ranking[::-1]

for i in range(scores.shape[0]):
    l = config.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")