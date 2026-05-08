import ast
import re

with open(r'c:\Users\ELITEBOOK\Downloads\MIRA\services\llm-service\tests\test_llm_service.py', 'r') as f:
    content = f.read()

lines = content.split('\n')
new_lines = []
for i, line in enumerate(lines):
    if line.strip().startswith('def test_'):
        # Check if it lacks return type
        if '->' not in line and line.endswith(':'):
            line = line[:-1] + ' -> None:'
        new_lines.append(line)
        # Check next line for docstring
        next_line = lines[i+1] if i+1 < len(lines) else ''
        if not next_line.strip().startswith('"""'):
            indent = line[:len(line) - len(line.lstrip())]
            func_name = line.split('def ')[1].split('(')[0]
            doc = func_name.replace('test_', '').replace('_', ' ').capitalize() + ' test.'
            new_lines.append(indent + '    """' + doc + '"""')
    else:
        new_lines.append(line)

with open(r'c:\Users\ELITEBOOK\Downloads\MIRA\services\llm-service\tests\test_llm_service.py', 'w') as f:
    f.write('\n'.join(new_lines))
