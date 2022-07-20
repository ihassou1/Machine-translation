import pickle

taus_file1 = open("from_e_to_f_taus.pkl", "rb")
taus_file2 = open("from_f_to_e_taus.pkl", "rb")
taus1 = pickle.load(taus_file1)
taus2 = pickle.load(taus_file2)


def invertAndGetMaxTaus(taus):
    """
    Given a dictionary of taus, this function inverts this dictionary.
    Additionally, it only takes the max of all the taus for each word.
    """
    inverted = {}
    for e in taus:
        for f in taus[e]:
            if f not in inverted:
                inverted[f] = (e, taus[e][f])
            else:
                if taus[e][f] > inverted[f][1]:
                    inverted[f] = (e, taus[e][f])
    return inverted


inverted1 = invertAndGetMaxTaus(taus2)
taus_file_inverted1 = open("from_f_to_e_taus_inverted.pkl", "ab")
pickle.dump(inverted1, taus_file_inverted1)
taus_file_inverted1.close()
