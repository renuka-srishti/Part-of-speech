#### This repository contains assignment given by Prof. David Crandall during the Elements of Artificial Intelligence Spring 2021 class.
# Part-of-speech tagging

## Introduction
 The goal of the project is to tag part of speech of given sentence
 
## Approaches
Emission probability :
Probability what part of speech will be tagged to given word.

Transition probability:
Probability what part of speech will be tagged given another part of speech.

1. Simple : Naive Bayes
    Here we are calcualating the final part of speech by calculating the max emission probability for give words with all part of speech.
2. HMM : Viterbi
   In this mehtod, first we are taking care of transition probability as well with emission probability.
   Here we are preparing one table to maintaining the words and its possible part of speech and after completion of the sentence, we are backtracking from end        sentence to intital and finding the final part of speech.
3. Complex : Gibbs sampling
    1. Initially we assign all part of speech a noun of a sentence.
    2. Then, we are genrating samples by assigning all part of speech.
    3. We are assigning final part of speech by using randomization of choices in those sample after 1000 iterations
    4. Here we are considering all connections while calculating the probability

## Points we discussed before impelmentation
1. Disucss with team abot what should be the emission probability, intitial probability and transiiton probability.
2. How it should be calculated

## Output
|  | Words Correct | Sentence Correct |
| ------------- | ------------- | ------------- |
| Ground truth   | 100.00%  | 100.00% |
| Simple  | 90.71% | 34.95% |
| HMM  | 93.46% | 46.20% |
| Complex  | 78.05%  | 20.25% |