Proposed Plan : 

              

1. use teacher: Instruct GPT 175B generated COT ( Text \-davenci \-002) : Zero-shot CoT.  
2. Fine Tune the student model on teacher Cot : primary model : Flan T5 base (220M) 

   Possible ablation : use other student models Flan T5 Small ( 60M) and Flan T5 Large (700B)

   

   Possible additions :  

* Filter COt Based on Final answer  
* Use calculator to handle calculation errors 

  \=\> Evaluate performance with answer only filter and with calculator)

* use Receval : (Prasad et al (2023) : RECEVAL: Evaluating Reasoning Chains via Correctness and Informativeness ) As a mùetric to evaluate COT Correctness  
* use different benchmarks : primarily GSM8K ( Arithmetic reasoning) \+ possible ablation : Strategy QA ( Common sense reasoning)   
* Test Inference Cost : Use metrics ( Flops, model size, dataset size)

Baseline : Flan T5 Base zero shot  
Proposed method : COT fine tuning \+ key additions

example of results of Ho et al (2023) Accuracy achieved on GSM8k for specific setting:

For Flan T5 Base (220 M):  
 Zero-shot : achieved 2.5% accuracy  
Cot fine tuning achieved 2.96% acc  
Fine tuning : 4.93% ??(what is fine tuning here how does it work )

 

