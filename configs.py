import os, re, json, torch
import numpy as np

def process_text(text):
    # text = re.sub(r"\n\t\s\r\f\'\"\[]\{}", "", text)
    text = re.sub(r"[(\{)(\})(\')(\")(\n)(\t)(\r)]", "", text)
    # text = text.replace('\'','').replace('\"','').replace('[','').replace(']','').replace('{','').replace('}','').replace('\n','').replace('\t','')
    return text

text = "\n\t\r\f{}{}\': : :\""
print(text)
text = process_text(text)
print(text)
label = "discovery"
datum = {"sentence1": text, "label": label}
with open('training_data.json', 'w') as f:
    json.dump(datum, f)
    
with open('training_data.json') as f:
    text=json.load(f)
    print(text)

a = '22222222'
print(a[:100])
