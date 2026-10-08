import re, sys
from pathlib import Path
root=Path(__file__).parent/'reading_text'
for spec in sys.argv[1:]:
    method, pages=spec.split(':')
    for item in pages.split(','):
        a,b=(map(int,item.split('-')) if '-' in item else (int(item),int(item)))
        for n in range(a,b+1):
            s=(root/method/f'p{n:02d}.txt').read_text(encoding='utf-8')
            s=re.sub(r'<latexit.*?</latexit>','',s,flags=re.S)
            s=re.sub(r'/uni[0-9A-Fa-f]{8}','',s)
            s=s.replace('\x00','')
            s=re.sub(r'\n(?:[ \t]*\n){2,}','\n\n',s)
            print(f'\n===== {method} PDF PAGE {n} =====\n{s}')
