import json

content = open('scraping/id.json', 'r', encoding='utf-8').read().strip()
content = content.rstrip("\\' \n")
content = content.replace('\\"', '"')

# Count braces
open_count = content.count('{')
close_count = content.count('}')
print(f'Open braces: {open_count}')
print(f'Close braces: {close_count}')
print(f'Difference (should be 1 for array wrapping): {open_count - close_count}')

# Show last 200 chars
print(f'\nLast 200 chars:')
print(repr(content[-200:]))

# Add missing braces if needed
while open_count > close_count:
    content += '}'
    close_count += 1
    print(f'Added closing brace. Now: open={open_count}, close={close_count}')

# Wrap in array
content = '[' + content + ']'

try:
    data = json.loads(content)
    with open('scraping/id.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'\nSuccess! Formatted {len(data)} items')
except json.JSONDecodeError as e:
    print(f'\nError at position {e.pos}: {e.msg}')
    start = max(0, e.pos - 100)
    end = min(len(content), e.pos + 100)
    print(f'Context: {content[start:end]}')
