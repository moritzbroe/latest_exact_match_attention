# Learning-rate ablation

Table 9: the LM recipe at half and double the peak learning rate, for the three small sizes and
all three architectures.

1. Train (dim 256 512 768; `--half` and `--double`; `lema <dim> --head 64`, `rope <dim>`,
   `gdn <dim>`):

       python train_lr_ablation.py lema <dim> --head 64 --half

2. Evaluate with the main protocol, then print the table:

       python ../main_lm/eval_lm.py --out out
       python collect.py --latex
