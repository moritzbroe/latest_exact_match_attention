# Heads of the 834M LEMA model

Figure 18: hit rate and attention distance of every head on held-out text.

1. Record hit rates and distances:

       python analyze_heads.py lema1536_h64                # -> out/lema1536_h64.json

2. Figure:

       python plot_distances.py out/lema1536_h64.json      # -> out/distances_lema1536_h64.pdf
