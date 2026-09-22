# Recall without a curriculum

Figure 9: softmax transformers trained from scratch at a fixed number of pairs.

1. Train n = 2, 3, 4 with seeds 0 to 2:

       ./sweep.sh rope 2 3 4

2. Figure:

       python plot_discovery.py                       # -> out/discovery.pdf
