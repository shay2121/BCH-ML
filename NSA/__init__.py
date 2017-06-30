from pulp import *
import settings

'''
globalMode=settings.globalMode
globalK=settings.globalK
globalN=settings.globalN
globalM=settings.globalM
globalSnr=settings.globalSnr
globalInd=settings.globalInd
globalnumOfWords=settings.globalnumOfWords
globallogFolder=settings.globallogFolder
'''



class Solver:
    def __init__(self):
        print(4)
        print(globalK)
        self.n=5
    def remConstraints(self):
        return
    def matrixAdaptation(self):
        return
    def buildProblem(self):
        return
    def NSA(self):
        #return -1 or 0= > fail, 1 = > success, 2 = > anomali, 3 = > no feasible solution(constraints collide)
        mode=0
        def __init__(self, mode):
            self.mode=mode
        while (1):
            if self.first():
                self.remConstraints()
                self.matrixAdaptation()
            self.buildProblem()
            solveAns=self.solve() #add if not integral....
            if self.integral(): return 1
            if self.integralX():
                self.CGA1()
            else:
                self.constructH()
                self.CGA2()

    def decode(self):
        # declare your variables
        x1 = LpVariable("x1", 0, 40)  # 0<= x1 <= 40
        x2 = LpVariable("x2", 0, 1000)  # 0<= x2 <= 1000

        # defines the problem
        prob = LpProblem("problem", LpMaximize)

        # defines the constraints
        prob += 2 * x1 + x2 <= 100
        prob += x1 + x2 <= 80
        prob += x1 <= 40
        prob += x1 >= 0
        prob += x2 >= 0

        # defines the objective function to maximize
        prob += 3 * x1 + 2 * x2

        # solve the problem
        status = prob.solve(GLPK(msg=0))
        LpStatus[status]

        # print the results x1 = 20, x2 = 60
        value(x1)
        value(x2)
        print("1")
