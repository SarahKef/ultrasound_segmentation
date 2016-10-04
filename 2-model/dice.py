import numpy as np
def diceCoefficient(predY, Y):
    """
    This works for one image
    http://stackoverflow.com/a/31275008/116067
    """
    denom = (np.sum(predY == 1) + np.sum(Y == 1))
    if denom == 0:
        # By definition, see https://www.kaggle.com/c/ultrasound-nerve-segmentation/details/evaluation
        return 1
    else:
        return 2 * np.sum(Y[predY == 1]) / float(denom)

def averageDiceCoefficient(predY, Y):
    diceCoefficients = []
    for i in range(predY.shape[0]):
        diceCoefficients.append(diceCoefficient(predY[i], Y[i]))
    return np.mean(diceCoefficients)