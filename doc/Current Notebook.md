current notebook we have  : 

based on Ho et al (2023) : Large Language Models Are Reasoning Teachers 

\=\> we fine tune Flan T5 base on generated COT By Instruct GPT 175B ( text-davenci-002)

Dataset : GSM8K (7.473 train/1.319 test data)  
GPT ready zero-shot COT : 7.473 COT

we construct 2 training sets:

A : No-Filter set : All 7.473 COT  
B \= Filtered COT ( Final answer \= Target answer) \=\> resulted in 3,389 COT (45% of original training data)

Fine tuning:  
 we finetune the student model Flan T5 base on both sets A and B 

Inference :   
we use Greedy decoding ( 3 epochs)

we run test data 1.319 examples for 3 conditions  : 

* Baseline : Flan T5 base zero shot (no fine tuning)  
* Student Set A  
* Student Set B

Results : Accuracy achieved : 

Baseline : 4.78%  
Student Set A : 2.65%  
Student Set B : 2.35%

Why do we have these results ?

Set B achieved only 2.35% ACC worse than baseline and worse than Set A \=\> means Filtered data is Noisy

Possible reasons : 

* Max Token Cap/ Max sequence length \= 128 is problem here : The training was stopped before it finishes because the cot is too long  
* Learning rate is too high : here we are using 3e-4 we should use 1e-5/3e-5 to 5e-5 ( in the paper they used 1e-4)  
* there is no Regularization : we should use weight Decay  
* Greedy decoding : the model is resting the same sequence with highest probability until it reaches the max sequence cap : the model has no repetition penalty / even with repetition penalty ( no rep decoding didn’t achieve high accuracy either possibly because of the learning rate) 

 

