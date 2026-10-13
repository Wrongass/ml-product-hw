def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
    return correct / len(y_true)

