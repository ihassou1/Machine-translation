"""
This module implements the IBM Model for machine translation. 
It calculates all the taus and serializes them into a file using Python's builtin 
utility Pickle
"""
import pickle
import unittest

BASE_DIR = "/data/cs65-S22/mt/"


def segmentText(path):
    """
    This function takes the text contained in the path
    and creates a matrix in which each row contains the words of each sentence as an array
    """
    with open(path, "r", encoding="latin1") as f:
        lines = f.read().splitlines()
        return [line.split(" ") for line in lines]


def preprocess(reverse):
    """
    Pairs the foreign and English sentences.
    @Param reverse is True if P(e|f), otherwise P(f|e)
    """
    eng_file = "english-senate-0.txt"
    foreign_file = "french-senate-0.txt"
    english_sentences = []
    foreign_sentences = []
    foreign_sentences = segmentText(BASE_DIR + foreign_file)
    english_sentences = segmentText(BASE_DIR + eng_file)
    if reverse:
        return [(e, f) for f, e in zip(foreign_sentences, english_sentences)]
    return [(f, e) for f, e in zip(foreign_sentences, english_sentences)]


class IBMModel1:
    def __init__(self, reverse):
        self.pairs = preprocess(reverse)
        self.tb = {}
        self.count = {}

    def initialize_tb(self, val):
        """
        A function to initialize all the possible pairs with val as the tau value
        """
        for pair in self.pairs:
            for e in pair[1]:
                if e not in self.tb:
                    self.tb[e] = {}
                for f in pair[0]:
                    if f == "elle" and e == "eats":
                        print("elle")
                    self.tb[e][f] = val

    def initialize_count(self, val):
        """
        a function to initialize all the possible pairs with val as the count value
        """
        for pair in self.pairs:
            for e in pair[1]:
                if e not in self.count:
                    self.count[e] = {}
                for f in pair[0]:
                    self.count[e][f] = val

    def make_taus(self):
        """
        Initializes the taus and count and then generates the subsequent new taus and counts
        by applying the IBM Model 1 which implements the EM algorithm
        """
        self.initialize_tb(1)
        self.initialize_count(0)

        THRESHOLD = 0.5
        # We chose 0.5% instead of 1% because you mentionted that 10 is a nice number for iteration
        # and we found that with 0.5% it's just slightly above 10 interation inside the while loop
        current_pk = 1
        change_pct = 1
        while change_pct >= THRESHOLD:
            next_pk = 0
            self.initialize_count(0)
            #    E Step:
            for pair in self.pairs:
                englishSentence = pair[1]
                foreignSentence = pair[0]
                p = {}
                for k in range(len(foreignSentence)):
                    p[k] = 0
                    foreignWord = foreignSentence[k]
                    for englishWord in englishSentence:
                        p[k] += self.tb[englishWord][foreignWord]
                    for englishWord in englishSentence:
                        self.count[englishWord][foreignWord] += (
                            self.tb[englishWord][foreignWord] / p[k]
                        )
                    # We decided to aggregate instead of multiplying, just to avoid underflow
                    # and not have to deal with log. It should not matter after all, since this
                    # is for comparison.
                    next_pk += p[k]
                #    M Step:
            for e in self.tb:
                n_e_o = sum(list(self.count[e].values()))
                for f in self.tb[e]:
                    # get the sum of count for englishWord to any frenchWord
                    self.tb[e][f] = self.count[e][f] / n_e_o
            change_pct = abs(((next_pk - current_pk) / current_pk) * 100)
            current_pk = next_pk


class TEST_IBMModel1(unittest.TestCase):
    def test_make_ts(self):
        ibm = IBMModel1(False)
        ibm.make_taus()
        # With the understanding of the model, we should expect an obvious translation
        # like "Honourable" in English to "honorable" in French to have a greater tau
        # than an incorrect translation ("le" means "the")
        self.assertGreater(
            ibm.tb["Honourable"]["honorable"], ibm.tb["Honourable"]["le"]
        )


def writeToFile():
    """Generates the taus by running the make_taus and then serializes them with
    the pickle
    """
    ibm = IBMModel1(False)
    ibm2 = IBMModel1(True)
    ibm.make_taus()
    ibm2.make_taus()

    dbfile = open("from_e_to_f_taus.pkl", "ab")
    dbfile2 = open("from_f_to_e_taus.pkl", "ab")
    pickle.dump(ibm.tb, dbfile)
    dbfile.close()
    pickle.dump(ibm2.tb, dbfile2)
    dbfile2.close()


def main():
    unittest.main()
    writeToFile()


if __name__ == "__main__":
    main()
