import json

content = open('scraping/id.json', 'r', encoding='utf-8').read().strip()
content = content.rstrip("\\' \n")
content = content.replace('\\"', '"')

# Add missing closing brace at the end if needed
if not content.endswith('}'):
    content += '}'

# The content already has the objects with commas, just wrap it
content = '[' + content + ']'

try:
    data = json.loads(content)
    with open('scraping/id.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'Success! Formatted {len(data)} items')
except json.JSONDecodeError as e:
    print(f'Error at position {e.pos}: {e.msg}')
    start = max(0, e.pos - 100)
    end = min(len(content), e.pos + 100)
    print(f'Context: {content[start:end]}')
