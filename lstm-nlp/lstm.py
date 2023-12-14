import numpy as np

# split a uni-variate sequence into samples


def split_sequence(sequence, lookback):
    X, y = list(), list()
    for i in range(len(sequence)):
        print(len(sequence))
        # find the end of this pattern
        end_ix = i + lookback
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        return np.array(X), np.array(y)


# define input sequence
raw_seq = [10, 20, 30, 
           40, 50, 60,
           70, 80, 90]
# choose a number of time steps
lookback = 5
# split into samples
X, y = split_sequence(raw_seq, lookback)
# summarize the data
for i in range(len(X)):
    print(f"Sequence: {X[i]}")
    print(f"Prediction: {y[i]}")