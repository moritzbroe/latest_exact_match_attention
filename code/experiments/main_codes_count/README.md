# Dictionary size against context length

Tables 11 and 12: entries per head of a trained LEMA model after t tokens of held-out text.

1. Count, per run (lema{256,512,768,1024,1280,1536}_h64 and lema1024_h{8,16,32}):

       python count_codes.py <run> --count 32         # -> out/<run>.json

2. Tables:

       python table.py --latex
