# Part1 : Part-of-speech tagging

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
                        Words Correct Sentence Correct
   0. Ground truth:      100.00%              100.00%
        1. Simple:       90.71%               34.95%
        2. HMM:          93.46%               46.20%
        3. Complex:      78.73.05%            13.25%

# Part 2: Moutain finding

## Introduction 

A interesting problem to find the ridgeline of a mountain in a given photo. 
For each column of the image we needed to estimate the row of the image corresponding to the boundary position using:- 
- A simple baye's net
- With the use of Viterbi algorithm along with the HMM
- Improving the result from the Viterbi algorithm using some human feedback

This was then to superimposed on the image in three different colors - red, blue and green which can help us visualize the difference in each of the above mentioned approaches.

## Approach

Initial formulation of the problem was a little difficult to us. After two hours of reading and one office hours later, we were able to decide the parameters (transition and emission probabilities) and for the Bayes net. At this point it was clear to us that the emission probabilities would be a funciton of the edge strength values provided and transitiion probabilites would be a function of the distance/difference between the row numbers (states).


Since, this was a good starting point, and as our first experiment we decided to use a gaussian distribution for each state's transition probabilities with mean being equal to the row_index of the state and standard deviation being equal to number of rows divided by 25. This gave a lot of leeway for discontinuties in the ridgeline. So we thought we could use some other arbitray function that is decaying exponentially. So the function we came up with works this way:
 
 ```
 k = (arange(num_rows) - full(num_rows, i))/num_rows 
 k[i+1:] = negative(k[i+1:])
 k = (k + ones(num_rows)) ** 10 
 ```
 
 It takes the list of states \[0...num_rows) subtracts with row_index (current state) and divides it by the total number of rows. Then the half of the array is made symmetric with the other half. Row of ones is added to get the probabilies to positive and this is then raised to the power of 10 (we found that it worked better emperically). Example plot shown is row_index 50, the transition probabilities from the above mentioned function. 
 
![transition probabilities for row_index 50](https://imgur.com/H2ci8dG.png)

We also experimented with one other transition probability function which was a step function, which hard assigned values to the particular differences in the rows. (10 pixels up and 10 pixels down)

```
k = zeros(num_rows)
weight_array = array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
if i + 10 < num_rows:
    k[i:i+10] = weight_array
else:
    idx = i
    t = -1
    while idx < num_rows and idx < i + 10:
        t = t + 1
        k[idx] = weight_array[t]
        idx = idx + 1

if i - 10 > 0:
    k[i-10:i] = flip(weight_array)
else:
    idx = i
    t = 10
    while idx > 0 and idx < i - 10:
        t = t - 1
        k[idx] = weight_array[t]
        idx = idx - 1
```

This function gave us satisfactory results as well but we just decided to go with the first function as it had non 0 probabilities for all the states making it a intuitively better choice. 

For emission probabilities, we calculated it in a pretty straight forward manner. We took it as the edge strength 2D array normalized with the max value found in each column + 1 (to avoid having 1 probability in the problem)

```
for i in range(edge_strength.shape[1]):
    edge_strength[:,i] = divide(edge_strength[:,i],(max(edge_strength[:,i])+1))
```

For the Viterbi algorithm we re-used the same code from Task 1.

To accomate the human guided pixel, we just modified the emission probabilities for that particular column, by setting all of the other values in that column to zero other than the row index given to us in the command line.

```
    #checking if the human entered pixel value exist 
    if start_col != -1 or start_row != -1:
        #then set all the emission probabilities in that particular step to 0 
        edge_strength[:,start_col] = zeros(edge_strength.shape[0])
        
        #set only the state which is marked to be oberved as 1
        edge_strength[start_row, start_col] = 1
```

### Results
Even with all of this we were not able to get the best result on few of the images, as the edge values were heavily concentrated at or near the grass/treeline. But for the mountains that had a distinguishable ridge, we were able to smoothen the ridgeline accurately with Viterbi and human feedback.

To reproduce the results commited, please use the below human feedback values mentioned below:

| Filename | Row | Column |
|:-:|:-:|:-:|
| mountain.jpg | 68 | 99 |
| mountain2.jpg | 68 | 233 |
| mountain3.jpg | 61 | 66 |
| mountain4.jpg | 67 | 198 |
| mountain5.jpg | 83 | 194 |
| mountain6.jpg | 68 | 99 |
| mountain7.jpg | 62 | 57 |
| mountain8.jpg | nil | nil |
| mountain9.jpg | 81 | 105 |


A good result:

![mountain](https://raw.github.iu.edu/cs-b551-sp2021/chshan-rsrishti-tsawaji-a3/master/part2/output_images/output_human.jpg?token=AAAD5OJLXIGFEMNGFEDKSGLASYCYY)

A bad result:

![mountain2](https://raw.github.iu.edu/cs-b551-sp2021/chshan-rsrishti-tsawaji-a3/master/part2/output_images/output2_human.jpg?token=AAAD5OONYHMWGCVK53B36OTASYC2Y)

# Part 3: Reading Text

## Introduction

The goal of this project is to recognise text in an image. But these images are noisy, so any particular letter may be difficult to recognize. However, if we make the assumption that these imageshave English words and sentences, we can use statistical properties of the language to resolve ambiguities. The program should detect the text in the test image, using (1) the simple Bayes net and (2) the HMM with MAPinference (Viterbi). 

Assumptions in this problem are:
  1. All the text in our images has the same fixed-width font of the same size.  In particular,each letter fits in a box that’s 16 pixels wide and 25 pixels tall.
  2. Our documents onlyhave the 26 uppercase latin characters, the 26 lowercase characters, the 10 digits, spaces, and 7 punctuation symbols ,(),.-!?’"

## Approach

  1. The first step was to find a text file that contained a lot of English words. I searched for a novel online and got the text of the famous book 'Pride and Prejudice' by Jane Austen. This provided the relationship of each character with other characters in the English language.
     1. Initial probability is calculated by counting the number of times a specific character starts a word and dividing this number by total number of words in the training file. This is maintained in a dictionary data structure and it is initialized to 0 for all valid characters.
     ````
        initial_probability = {x : 0 for x in valid_characters}
     ````

     2. Transition probability is calculated by counting the number of times a character x comes after a chaaracter y and dividing this by the number of occurrences of character x. This is maintained in a dictionary data structure and it is initialized to 0 for all valid characters.
     ````
        transition_probability = {x : {y : 0 for y in valid_characters} for x in valid_characters}
     ````
  2. Emission probability - For the emission probability we consider that m% of the pixels are noisy, then a naive Bayes classifier could assume that each pixel of a given noisy image of a letter will match the corresponding pixel in the referenceletter with probability (100−m)%. The emission probability is maintained in a dictionary data structure and it is initialized to 0 for all valid characters.
  ````
      emission_probability = {x : {y : 0 for y in range(len(test_letters))} for x in valid_characters}
  ````    
  
  The emission probability for each character for a specific test case, after considering all pixels are conditionally independent of each other, is given by
  ````
      (1 - noise)^matching_pixels * noise^non_matching_pixels
  ````
  3. Bayes Network - The Bayes Net for predicting text in an image only takes into account the emission probabilities, and returns a sequence of characters for which the emission probability was highest against a test sample.
   
  4. Viterbi algorithm - Viterbi algorithm was applied using a table to store maximum probability values upto the specific sequence and the path with the highest probability values was chose to predict the text in the image. All calculations were done in the logarithmic scales to make multiplication and division of very small values easier to maintain.
 
## Problems we faced
1. Calculation of emission probabilities was difficult. It took some time to understand the calculations that are required to form the emission probabilities.
   
## Usage:
    ````
        python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
    ````

## Output:
The output of the program for one test image is:
````
    Simple: 1t 1s so orcerec.
       HMM: It is so ordered.
````
