import json
import re

with open('scraping/id.json', 'r', encoding='utf-8') as f:
    content = f.read().strip()

# Clean up
content = content.rstrip("\\' \n")
content = content.replace('\\"', '"')

# Fix object separators more aggressively
# Look for patterns like }"{ and replace with },{ 
content = re.sub(r'"\}\{', '},{"', content)  # "}{  becomes },{
content = re.sub(r'}\{', '},{', content)

# Find closing brace of KbliVersion and add comma before next object
content = re.sub(r'(\}\})(,)?\{', r'}\},{', content)

# Wrap in array
content = '[' + content + ']'

try:
    data = json.loads(content)
    with open('scraping/id.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print('OK')
except Exception as e:
    print(f'Error: {str(e)[:100]}')
