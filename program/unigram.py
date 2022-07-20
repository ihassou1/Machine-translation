import math


def getText(file):
    BASE_DIR = "/data/cs65-S22/langmod/hansard/"
    with open(f"{BASE_DIR}{file}", "r", encoding="latin1") as f:
        return f.read()


def getWords(file):
    return getText(file).split()


def getFrequencies(words):
    """Takes a list of words and returns a a dictionary with
    key being a particular word and the value is the frequency of
    of that word in the list of words

    Arguments:
        words: the list of words
    Returns:
        A dictionary with each word and its frequency
    """

    frequencies = {}
    for w in words:
        if w in frequencies:
            frequencies[w] += 1
        else:
            frequencies[w] = 1
    return frequencies


class Unigram:
    def __init__(self, trainSet, tokens):
        self.trainSet = trainSet
        self.tokens = tokens
        self.staticProbs = {}

    def buildProbs(self, alpha):
        probs = {}
        for w, c in self.trainSet.items():
            # alpha = 1
            theta = (c + alpha) / (
                self.tokens + alpha * len(self.trainSet.keys())
            )
            probs[w] = theta

        # Making the words set finite
        theta = (alpha) / (self.tokens + alpha * len(self.trainSet.keys()))
        probs["U"] = theta
        return probs

    def buildStaticProbs(self, alpha):
        """creates the probability distribution with assumption
        that the alpha = optimized value for alpha
        """
        self.staticProbs = self.buildProbs(alpha)

    def lh(self, alpha, testSet):
        probs = self.buildProbs(alpha)
        liklihood = 0
        for word, count in testSet.items():
            if word not in probs:
                word = "U"
            testTheta = math.log(probs[word]) * count
            # log property : log(a^b) = b*log(a)
            liklihood += testTheta
        return liklihood

    def static_lh(self, testSet):
        liklihood = 0
        for word, count in testSet.items():
            if word not in self.staticProbs:
                word = "U"
            testTheta = math.log(self.staticProbs[word]) * count
            # log property : log(a^b) = b*log(a)
            liklihood += testTheta
        return liklihood

    def differentiate(self, text):
        sentences = list(filter(lambda s: len(s) != 0, text.split("\n")))
        counter = 0
        i = 0
        self.buildStaticProbs(1.6)
        while i < len(sentences):
            freq1 = getFrequencies(sentences[i].split())
            freq2 = getFrequencies(sentences[i + 1].split())
            lh1 = self.static_lh(freq1)
            lh2 = self.static_lh(freq2)
            if lh1 > lh2:
                counter += 1
            i += 2
        return (counter / (len(sentences) / 2)) * 100

    def optimizer(self, heldout):
        alpha = 1.0
        current_lh = self.lh(alpha, heldout)
        i = 1
        while i < 10:
            new_lh = self.lh(i, heldout)
            if new_lh > current_lh:
                alpha = i
                current_lh = new_lh
            i += 0.1
        return alpha


def main():
    train_words = getWords("english-senate-0.txt")
    train_set = getFrequencies(train_words)

    unigram = Unigram(train_set, len(train_words))
    unigram.buildStaticProbs(1)

    test_words = getWords("english-senate-2.txt")
    test_set = getFrequencies(test_words)

    dev_words = getWords("english-senate-2.txt")
    dev_set = getFrequencies(dev_words)

    lh1 = unigram.static_lh(test_set)
    unigram.buildStaticProbs(1.6)
    lh2 = unigram.static_lh(test_set)
    optimized_alpha = unigram.optimizer(dev_set)

    good_bad = getText("good-bad-split.txt")
    accuracy = unigram.differentiate(good_bad)

    print("===============OUTPUT===============")
    print("likelihood with alpha=1: ", lh1)
    print("optimized alpha: ", optimized_alpha)
    print(f"likelihood with alpha={optimized_alpha}: ", lh2)
    print("Good-Bad differentiation accuracy (%):", accuracy)


if __name__ == "__main__":
    main()
