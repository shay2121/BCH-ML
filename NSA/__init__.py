from pulp import *

class Solver:
    """A simple example class"""
    i = 12345

    def f(self):
        return 'hello world'

def NSA():
    #return -1 or 0= > fail, 1 = > success, 2 = > anomali, 3 = > no feasible solution(constraints collide)
    x, i, solveAns;
    int
    tries = 0;
    if (CODE_DIM < 21) IPD.ML();
    if (mode == 2) resetCounters();
    globalNSAfirst=true;
    while (1)
        {
        if (mode == 1 | | mode == 3 | | (mode == 2 | | (
        code == 7 | | code == 8 | | code == 9 | | code == 19 | | code == 20 | | code == 18 | | code == 6)))
        {
            IPD.remConstraints(globalNSAfirst);
        globalRemCon = 1;
        }
        // if (mode == 3 | | (
            mode == 2 & & (code != 7 & & code != 7 & & code != 9 & & code != 19 & & code != 20 & & code != 18)))
        if ((mode == 2 & & (
                                    code != 7 & & code != 7 & & code != 9 & & code != 19 & & code != 20 & & code != 18 & & code != 6)))
        {
        IPD.matrixAdaptation();
        globalMA=1;
        }

        IPD.Build();
        complexity += 1;
        / *
        if (!globalFirst & & first)
        {
        IPD.updatePerms();
        first=false;
        } * /
        tries++;
        if (tries > 100)
        {
        anomalies++;
        complexity -= 100;
        complexity += complex;
    return 4;
    }
    solveAns = IPD.Solve();
    if (solveAns == 2)
    {
    return 3;

}
globalNSAfirst = false;
if (solveAns == 1)
{
return 1;
if (inFesibilityMode == 1) return 1;
return 1;
}
if (mode == 2) updateFractionalCounters();
if (IPD.sameSolution())
{
// cout << "[1] Can not find optimal Solution\n";
return 0;
mistakesNSA + +;
IPD.printML();
return 0;
}
if (IPD.integralX())
{
IPD.CGA1();
continue;
}
else
{
IPD.ConstructH();
if (IPD.CGA2())
    {
    // cout << endl << "**" << endl;
continue;
}
else
{
// cout << "[2] Can not find optimal Solution" << "endl";
mistakesNSA + +;
return 0;
}
}
return -1;
}
}

def decode():
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
