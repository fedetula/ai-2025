import numpy as np
import matplotlib.pyplot as plt

DATASIZE_LEN = 6

x = np.random.rand(DATASIZE_LEN)
noise1 = np.random.normal(0, 0.1, DATASIZE_LEN)

y = 3.2 * x + noise1

plt.scatter(x, y, label="Data points")
plt.title("Linear Regression")
#plt.show()


def split(*iterables, percentage=0.8, shuffle=False):
    if not iterables:
        raise ValueError("At least one iterable is required")
    
    length = len(iterables[0])
    if any(len(it) != length for it in iterables):
        raise ValueError("All iterables must have the same length")
    
    if shuffle:
        indices = np.random.permutation(length)
        iterables = [np.array(it)[indices] for it in iterables]
    
    mid = int(length * percentage)
    split_iterables = [(it[:mid], it[mid:]) for it in iterables]
    
    return tuple(item for pair in zip(*split_iterables) for item in pair)


x_train, x_test, y_train, y_test = split(x, y)

print(x_train, x_test)
print(y_train, y_test)
