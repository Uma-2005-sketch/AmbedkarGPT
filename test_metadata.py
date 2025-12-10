import json

with open('corpus_metadata.json', 'r') as f:
    metadata = json.load(f)

print("Loaded metadata for speeches:")
for speech, data in metadata.items():
    print(f"\n{speech}:")
    print(f"  Title: {data['title']}")
    print(f"  Year: {data['year']}")
    print(f"  Cultural markers: {', '.join(data['cultural_markers'][:3])}")