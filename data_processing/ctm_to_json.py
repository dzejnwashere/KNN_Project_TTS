import os
import json

def write_json(new_entry, filepath='data.json'):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
            file_data = json.loads(content) if content.strip() else {"path": "", "data": []}
    else:
        file_data = {"path": data_dir, "data": []}
    file_data["data"].append(new_entry)
    with open(filepath, 'w') as f:
        json.dump(file_data, f, indent=4)

data_dir = "/home/alex/Documents/KNN/nfa-out/ctm/words"

for filename in os.listdir(data_dir):
    if filename.endswith('.ctm'):
        parts = filename.replace('.ctm', '').split('_')
        emotion = parts[0]
        words = []
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            x = line.split()
            if len(x) < 9:
                continue
            words.append({
                "word": x[4],
                "start_time": round(float(x[2]), 2),
                "end_time": round(float(x[2]) + float(x[3]), 2),
                "confidence": round(float(x[8]), 2)
            })
        text = ' '.join(w['word'] for w in words)
        data = {
            "name": filename.replace('.ctm', '.wav'),
            "text": text,
            "emotion": emotion,
            "emo-text": f"<{emotion}> {text} </{emotion}>",
            "words": words
        }
        write_json(data)