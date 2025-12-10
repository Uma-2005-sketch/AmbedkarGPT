import json 
import re 
 
with open('chunks.json', 'r', encoding='utf-8') as f: 
    data = json.load(f) 
chunks = data['chunks'] 
print(f'Total chunks: {len(chunks)}') 
print('Searching for Jat-Pat-Todak Mandal references...') 
print('='*80) 
found = 0 
for i, chunk in enumerate(chunks): 
    text = chunk.get('text', '').lower() 
    if 'jat' in text or 'pat' in text or 'todak' in text or 'mandal' in text or 'reject' in text: 
        found += 1 
        print(f'Chunk {i}:') 
        # Show context around the keyword 
        if 'jat' in text: 
            idx = text.find('jat') 
            print(f'  ...{text[max(0, idx-50):min(len(text), idx+150)]}...') 
        elif 'mandal' in text: 
            idx = text.find('mandal') 
            print(f'  ...{text[max(0, idx-50):min(len(text), idx+150)]}...') 
        print() 
print(f'Found {found} potentially relevant chunks') 
