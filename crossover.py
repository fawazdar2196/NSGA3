import numpy as np

#uniform_crossover is for continuous decision variables, while
#SinglePointCrossoverb and DoublePointCrossoverb are more suitable for binary decision variables

def uniform_crossover(x1, x2):
# #
          alpha = np.random.rand(len(x1))
          y1 = alpha*x1 + (1-alpha)*x2
          y2 = alpha*x2 + (1-alpha)*x1
          return y1, y2


def SinglePointCrossoverb(x1, x2):
           nVar = len(x1)
           c = np.random.randint(1, nVar)
           y1 = np.concatenate((x1[:c], x2[c:]))
           y2 = np.concatenate((x2[:c], x1[c:]))
           return y1, y2


def DoublePointCrossoverb(x1, x2):
            nVar = len(x1)
            cc = np.random.choice(range(nVar - 1), 2, replace=False)
            c1 = min(cc)
            c2 = max(cc)
            y1 = np.concatenate((x1[:c1], x2[c1:c2], x1[c2:]))
            y2 = np.concatenate((x2[:c1], x1[c1:c2], x2[c2:]))
            return y1, y2
# # #

