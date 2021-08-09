###################################
# CS B551 Spring 2021, Assignment #3
#
# RENUKA SRISHTI and rsrishti:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
from collections import defaultdict
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            prob = 0
            for i, word in enumerate(sentence):
                if word in self.emission_probabilities and label[i] in self.emission_probabilities[word]:
                    prob = prob + math.log(self.emission_probabilities[word][label[i]])

            return prob
        elif model == "HMM":
            return self.hmm_posterior
        elif model == "Complex":
            prob = 0.0
            prob = prob + math.log(self.initial_probabilities.get(sentence[0], 1e-5)) + math.log(self.emission_probabilities[sentence[0]].get(label[0], 1e-5))

            for i, word in enumerate(sentence):
                if i == 0:
                    continue
                else:
                    prob = prob + math.log(self.emission_probabilities[sentence[i]].get(label[i], 1e-5)) + \
                    math.log(self.emission_probabilities[sentence[i - 1]].get(label[i], 1e-5)) + \
                        math.log(self.transition_probabilities[label[i - 1]].get(label[i], 1e-5))
            return prob
        else:
            print("Unknown algo!")

    def create_data_dict(self, data):
        word_with_speech = defaultdict(lambda: defaultdict(int))
        pos_count = defaultdict(int)
        word_count = defaultdict(int)
        pos_seq_count = defaultdict(lambda: defaultdict(int))
        for word_speech in data:
            for word, pos in zip(word_speech[0], word_speech[1]):
                word_with_speech[word][pos] = word_with_speech[word][pos] + 1
                pos_count[pos] = pos_count[pos] + 1
                word_count[word] = word_count[word] + 1
        for word_speech in data:
            for pos_1, pos_2 in zip(word_speech[1], word_speech[1][1:]):
                pos_seq_count[pos_1][pos_2] = pos_seq_count[pos_1][pos_2] + 1
        return word_with_speech, pos_count, pos_seq_count

    # Do the training!
    #
    def train(self, data):
        self.word_with_speech, self.pos_count, self.pos_seq_count = self.create_data_dict(data)
        self.vocab = set(self.word_with_speech.keys())
        self.emission_probabilities = self.emission_prob(self.word_with_speech, self.pos_count)
        self.transition_probabilities = self.transition_prob(self.pos_seq_count, self.pos_count)
        self.initial_probabilities = self.intitial_prob(data, self.vocab)

        # self.

    def emission_prob(self, word_with_speech, pos_count):
        emission_prob_dict = defaultdict(lambda: defaultdict(float))
        for word in word_with_speech:
            for pos in word_with_speech[word]:
                emission_prob_dict[word][pos] = word_with_speech[word][pos] / pos_count[pos]
        # if word not in word_with_speech:
        #     emission_prob_dict[word] = 1e-5
        # for w in word_with_speech.keys() - set(emission_prob_dict.keys()):
        #
        return emission_prob_dict

    def transition_prob(self, pos_seq_count, pos_count):
        transition_prob_dict = defaultdict(lambda: defaultdict(float))
        for pos_1 in pos_seq_count:
            for pos_2 in pos_seq_count[pos_1]:
                transition_prob_dict[pos_1][pos_2] = pos_seq_count[pos_1][pos_2] / pos_count[pos_2]
        for pos_1 in transition_prob_dict:
            for pos_2 in transition_prob_dict:
                if pos_2 not in transition_prob_dict[pos_1]:
                    transition_prob_dict[pos_1][pos_2] = 1e-5
        # for t in pos_count.keys() - set(transition_prob_dict.keys()):
        #     transition_prob_dict[t] = 1e-5
        return transition_prob_dict

    def intitial_prob(self, data, vocab):
        initial_prob_dict = defaultdict(float)
        for word, _ in data:
            initial_prob_dict[word[0]] = initial_prob_dict[word[0]] + 1
        for k in initial_prob_dict:
            initial_prob_dict[k] = initial_prob_dict[k] / len(data)
        # for k in vocab - set(initial_prob_dict.keys()):
        #      initial_prob_dict[k] = 1e-5
        return initial_prob_dict

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        pos = []
        for word in sentence:
            if word not in self.vocab:
                pos.append('Noun')
            else:
                pos.append(max(self.emission_probabilities[word], key=self.emission_probabilities[word].get))
        return pos

    def hmm_viterbi(self, sentence):
        states = list(self.pos_count.keys())
        number_of_states = len(states)
        number_of_outcomes = len(sentence)
        predicted_sequence = []
        viterbi_dynamic_table = np.zeros((number_of_states, number_of_outcomes))
        viterbi_dynamic_table_backtrack = np.zeros((number_of_states, number_of_outcomes), np.int)
        for index, word in enumerate(sentence):
            for state_id, state in enumerate(states):
                if index == 0:
                    viterbi_dynamic_table[state_id][index] = self.initial_probabilities.get(word, 1e-5) * \
                                                             self.emission_probabilities[word].get(state, 1e-5)
                    viterbi_dynamic_table_backtrack[state_id][index] = 0
                else:
                    temp_prob_list = [viterbi_dynamic_table[x][index - 1] *
                                      self.emission_probabilities[word].get(state, 1e-5) *
                                      self.transition_probabilities[states[x]][state] for x, _ in
                                      enumerate(states)]
                    viterbi_dynamic_table[state_id][index] = np.max(temp_prob_list)
                    viterbi_dynamic_table_backtrack[state_id][index] = np.argmax(temp_prob_list)
        # print(viterbi_dynamic_table)
        self.hmm_posterior = 0.0
        best_last_index = np.argmax(viterbi_dynamic_table[:, -1])
        self.hmm_posterior = self.hmm_posterior + math.log(np.max(viterbi_dynamic_table[:, -1]))
        predicted_sequence.append(states[best_last_index])

        for i in range(len(sentence) - 1, 0, -1):
            best_last_index = viterbi_dynamic_table_backtrack[best_last_index][i]
            predicted_sequence.append(states[best_last_index])
            self.hmm_posterior = self.hmm_posterior + math.log(viterbi_dynamic_table[best_last_index][i])

        return predicted_sequence[::-1]

    def complex_mcmc(self, sentence):
        pos = ["noun"] * len(sentence)
        res_pos = []
        states = list(self.pos_count.keys())
        for i in range(1000):
            sample_pos = []
            for index in range(len(sentence)):
                probs = np.zeros(len(states))
                for state_id, state in enumerate(list(self.pos_count.keys())):
                    if index == 0:
                        probs[state_id] = self.initial_probabilities.get(sentence[index], 1e-5) * \
                                          self.emission_probabilities[sentence[index]].get(state, 1e-5)
                    else:
                        probs[state_id] = self.emission_probabilities[sentence[index]].get(state, 1e-5) * \
                                          self.transition_probabilities[pos[index - 1]][state] * \
                                          self.emission_probabilities[sentence[index - 1]].get(state, 1e-5)

                probs = probs / np.sum(probs)

                sample_pos.append(np.random.choice(states, 1, p=probs)[0])
            pos = sample_pos
            res_pos.append(sample_pos)

        return res_pos[-1]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")