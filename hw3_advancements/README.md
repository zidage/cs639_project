# CS639 NLP Project

For this premilinary work, I focused on evaluation the baseline (with SFT) model trained by Yurun.

I ran the same python files with the default parameters that were found in the script sft.sh found at : Antidote/script/alignment
/sft.sh. However I re-used the already SFT aligned model and LoRA weights computed by Yurun. I pushed the used python files and script file to this branch.

The python files run are : 

- `Antidote/poison/evaluation/pred.py` : asks the model to answer unsafe prompts from BeaverTails. The outputs are in `eval_poison/Llama-2-7b-hf_sft.json`.

-  `Antidote/poison/evaluation/eval_sentiment.py` : This python file reads the json file outputted by pred.py, and evaluates if the answer is harmful or not. The model used to evaluate the harmfulness of the model is beaver-dam-7b. The results are then saved to `eval_poison/Llama-2-7b-hf_sft_sentiment_eval.json`. The harmful ratio (# of harmful answers / # of prompts answered) is of 54.40 %. This can be seen at the very end of the file.

- `Antidote/gsm8k/pred_eval.py` : This file will ask the model to answer prompts from the GSM8K dataset, and evaluate its answers. The dataset contains 8th grade level math questions. Once answered, the output is compared with the ground truth. The answer of the prompts is found in `eval_gsm8k/Llama-2-7b-hf_sft_eval.json`. The accrucacy is 4.90. This low sdcore can be expected by the fact that the model was not fine-tuned at all for this task.

Overall, the process took 15 minutes for each evaluation, but the GPU used was the free GPU on Google Colab : T4 GPU. When the whole process will be done on a NVIDIA H100, like it was done for the safety aligned model, the evaluation should be much quicker.


## Current Status

- The safety aligned model is tested on its harmfulness, which will constitute a useful baseline for the rest of the project.
- The setup for evaluating models is functional.

