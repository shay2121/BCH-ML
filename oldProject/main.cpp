  #define GENERAL_MODE 0

#include <windows.h>
#include <process.h>
#include <iostream>
#include "NSA.hpp"
//#include "LP_ADMM_Decoder.h"
#include <time.h>
#include <cstdlib>

#include <math.h>


using namespace std;
int glpkLoaded=1;
int globalSolveADMM=0;
int globalDecoder=1;
int globalUseReduction=0;
int globalRoundCheck=1;
int globalWeakerDepth=5;
int globalWeakerRange=5;
int globalAvgIterations=1;
int primeIterations=5;
int globalWeakerStep=0;
int globalTotTries=3;
int globalSH;
int modePT=0;
int mode=7;
int code=-2;
int MaxDepth=4;
#define RA 9
string basicPwd="C:\\MyProj\\";
string pwd=basicPwd+"CFG1\\";
int globalReBorn=12;
double FACTOR=1.0;

int globalLim=0;
int EnoughErrors=10;
double globalSafeZone=0.1;
double GSNStep=globalSafeZone;
unsigned int globalLPI=0;
#define NumOfCores 8
int MaxDecodingAttempts=5;
int globalMaxCorrected=1;
int globalALG1RowPermutations=10;
int GAP=7;
bool AntiRemInactive=false;
int globalSuccess7;
bool globalBL[bigN];
int MLError;
HANDLE ghMutex;
glp_prob *P1;
ofstream logFile;
int History[N];
int globalML4;
bool globalConverged=false;
bool globalRecycle=false;
int global1bitCorrection=0;
double globalSolutions[N][N];
double LPIR[RA];
int globalALG1Iteration;
int globalLastRows;
double globalDW[N];
double originalCandidate;
int localWeaks[N];
int globalR=27;
double thres=0;
int globalSolve=0;
bool globalFreeze=false;
bool global1BitUse=false;
ofstream globalf1,globalh1,globalf6,globalh6;
float globalGR;
int localErrs;
ostringstream convert;
string globalCMD;
int globalThreadNum;
int global=3;
bool noPerms=false;
bool useExternal=0;
int globalDmin;
int globalN,globalK,globalM,permMod;
int globalSGD4=0;int globalEVEN=0;
double globalSGD4R[RA];double globalEVENR[RA];double globalML4R[RA];double B1R[RA];
int globalSerial=0;
double globalSNR;
int polarIterations;
int globalZs;
int globalZ=0;
int backupGlobalN;
double globalDistWinner;
int globalErrs=0;
bool globalRefined=false;
int start;
int cycles=0;
int Ones=0;
int MaxGlobalTot=100000;
int inFesibilityMode=0;
int STOP=100;
int G[N][M];
int SHIFT;
int globalVars;
double globalSpecialBackup[N];
int globalTimes=1;
int globalMA=0;
int globalRemCon=0;
double globalStep;
int globalFrozen[N];
double OX4print;
int leafTranslation[bigN];
CP globalCP[N/2][medN];
int Degrees[M][N];
int globalOcc[medN];
int globalLeavesChildren;
int globalDynamicRows=0;
CP *globalPT[bigN];

#define CODE_DIM 63
#define MAX_Errors 1000

clock_t startClk,sotpClk;
double globalRD;
double globalGT;
double avgIterationTime;
int useBackup=0;

bool WeakNodes[N];
extern double solution[bigN];
bool putZ[bigN];
extern double original[N];
extern double test[N];
double originalX[bigN];
int FP[N];
int globalH[N][N];
bool pB[N][N];
int W[N];
double TimeComplexity[RA];
double avgCGA2Time;
int globalCGA2Times=0;
double avgCGA1Time;
int globalCGA1Times=0;
double avgNSATime;
double globalTime;
int globalNSAfirst=false;
bool countErors=false;
int globalNSAerrors=0;
double GSReal=0;
bool globalFirst=true;
bool alg3Action=false;
bool countNSA=false;
int globalTot=0;
int globalX=0;
double GR;
double GRC;
double GRS;
int GC=0;
int HBackup[N][N];
int NSAright=0;
double complex=0;
int anomalies=0;
extern int XML[N];
int iteration=0,len,sol[N],solMLD[N];
int globalB1[N][N];
int globalB2[N][N];
int correctedAnomalies=0;
int FractionalCounters[N];
int depthConstraints[MaximalDepthPossible][N];
int depthB[MaximalDepthPossible];

string pcMatrixFile;
string gMatrixFile;
string politopeFile;
string frozenFile;
double globalSnrL=0,globalSnrH=0;
string permFile;
int constraintsLocations[MaximalDepthPossible][2];
int checkRightSolution();
double S[N]={0.0};
double costVector[N];
void printSource();
int mistakesML=0,mistakesNSA=0,mistakesRML=0,corrected1=0,corrected2=0,corrected3=0,corrected4=0;
double complexity=0; //complexity with errors. counting also the times that the decipher failed.
double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks*10)/(CLOCKS_PER_SEC*3);
	return diffms;
}
ifstream myfile3 (permFile.c_str());
void printOriginalX()
{
	int i;
	for(i=0;i<globalN;i++)
		cout<<originalX[i]<<" ";
	cout<<endl;
}

int getMistakes()
{
	bool flag1=true,flag2=true;
	int ans=0; //0=>s=sol=MLD, 1=>s=MLD!=sol, 2=>s!=MLD,sol, 3=>s=sol!=MLD
	int j;
	for(j=0;j<CODE_DIM;j++)
		if(solution[j]!=S[j])
		{
			flag1=false;
			break;
		}
	if(CODE_DIM<21)
	{
		for(j=0;j<CODE_DIM;j++)
			if(XML[j]!=S[j])
			{
				flag2=false;
				break;
		}
	}
	if(flag1 && flag2) return 0;
	if(!flag1 &&flag2) return 1;
	if(!flag1 && !flag2) return 2;
	if(flag1 && !flag2) return 3;
}

bool Perm(int k,double *A)
	{
		int i,source,dest;
		if (k==0) return false;
		double Temp[N],T2[N];
		for(i=0;i<globalN;i++)
		{
			Temp[i]=A[i];
			T2[i] = test[i];
		}
		for(i=0;i<globalN;i++)
		{
			source=Permutations[k][i];dest=Permutations[0][i];
			A[dest]=Temp[source];
		}
		return true;
	}

void printAll(IPD &IPD)
{
	IPD.printSolution();
	IPD.ML();
	IPD.printML();
	printSource();
}

void resetCounters()
{
	int i;
	for(i=0;i<globalN;i++)
		FractionalCounters[i]=0;
}

void updateFractionalCounters()
{
	int i,x;
	for(i=0;i<globalN;i++)
		if(fabs(solution[i])>EPS && fabs(solution[i]-1)>EPS) FractionalCounters[i]++;
}

int findMostFractional(IPD &IPD)
{
	int i,max,res;
	max=0;
	for(i=0;i<globalN;i++)
	{
		if(max<FractionalCounters[i] || max==FractionalCounters[i])
		{
			res=i;
			max=FractionalCounters[i];
		}
	}
	return res;
}

void insertConstraints(IPD &IPD,int depth)
{
	int i;
	/*
	if(depth==1)//2be deleted!!!
	{
		depth=2;
		depthConstraints[0][0]=depthConstraints[1][0]=1;
		depthB[0]=1;
	}*/
	for (i=0;i<depth;i++)
		IPD.Update3(depthConstraints[i],depthB[i],1); //adding the constraint: X[location]=val
	//IPD.adaptCostVector(depth);
	IPD.matrixAdaptation();
}

void set1Constraint(int depth, int location, int val)
{
	int i;
	for(i=0;i<globalN+globalM;i++)
		depthConstraints[depth][i]=0;
	depthConstraints[depth][location]=1;
	depthB[depth]=val;
	constraintsLocations[depth][0]=location;
	constraintsLocations[depth][1]=val;
}

int isFeasibleSolution(IPD &IPD,int depth)
{ // return -1 or 0=>fail, 1=>success, 2=>anomali, 3=>no feasible solution (constraints collide)
	int res=0;
	IPD.softReset();
	insertConstraints(IPD,depth);
	//IPD.zeroizeCV();
	IPD.Build(1);
	IPD.Solve();
	IPD.restoreCostVector();
	return IPD.noFeasibleSolution(depth);
}





bool Int(double y)
{
	int x;
	x=(int) y;
	if(fabs(y-x)>EPS)
	{
		return true; 	
	}
	return false;
}

int PT(IPD &IPD,double globalSNR) //CGA1Now
{ // return -1 or 0=>fail, 1=>success, 2=>anomali, 3=>no feasible solution (constraints collide)
	int x,i,solveAns,ind;
	int tries=0;
	//generateSignalBuffer(originalX,globalSNR,globalN,globalM);
	for(i=0;i<globalTimes;i++)
	{
		IPD.reset4PT();
	}
	complexity++;
	polarIterations=0;
	IPD.Build();	
	IPD.writeParity();
	while(1)
	{
		start=IPD.buildPolitope(globalSNR);
		//IPD.printB();
		complexity+=1;
		solveAns=IPD.Solve();//strange that the solver succeeds and yet this is not the optimal solution. contradicts the certificate prop.
		return true;
		polarIterations++;
		if(solveAns==1) return 1;
		if(IPD.sameSolution(SHIFT+globalN+globalZs))
		{
			return false;
		}
	}
	/*
	for (ind=1;ind<globalM;ind+=2)
	{
		if(!Int(solution[160+(ind-1)/2]))
		{
			IPD.polarCGA1(ind);
		}
		start=IPD.buildPolitope(globalSNR);	
		solveAns=IPD.Solve();
		if(integralPolar()) return 1;
	}
	return 0;
	*/
}


int dynamicPT(IPD &IPD) 
{ 
	int x,i,solveAns,ind;
	int tries=0;
	bool res;
	
	for(i=0;i<globalTimes;i++)
	{
		IPD.reset4PT();
	}
	complexity++;
	polarIterations=0;
	//IPD.Build();	
	IPD.writeParity();
	SHIFT=0;
	i=0;
	while(1)
	{
		IPD.buildDynamicPolitope(i);
		//IPD.printB();
		complexity+=1;
		solveAns=IPD.Solve();//strange that the solver succeeds and yet this is not the optimal solution. contradicts the certificate prop.
		res=IPD.integralPolar();
		globalLastRows=globalM-globalR;
		return res;
	}
}


void endNSA(int p, clock_t start)
{
	clock_t stop=clock();
	double diff=(double) (stop-start)/(CLOCKS_PER_SEC*3);
	if(p==1)
	{
		avgCGA1Time=(avgCGA1Time*globalCGA1Times+diff)/(globalCGA1Times+1);
		globalCGA1Times++;
	}
	else if(p==2)
	{
		avgCGA2Time=(avgCGA2Time*globalCGA2Times+diff)/(globalCGA2Times+1);
		globalCGA2Times++;
	}
	avgNSATime=(avgCGA1Time*globalCGA1Times+avgCGA2Time*globalCGA2Times)/(globalCGA1Times+globalCGA2Times);
}



int NSA(IPD &IPD,int depth=-1)
{ // return -1 or 0=>fail, 1=>success, 2=>anomali, 3=>no feasible solution (constraints collide)
	if (globalSolveADMM)
	{
		ADMM();
		return 1;
	}
	int x,i,solveAns;
	int tries=0;
	if(CODE_DIM<21) IPD.ML();
	if(mode==2) resetCounters();
	globalNSAfirst=true;
	while(1)
	{
		if(mode==1 ||mode==3 || (mode==2 || (code==7 || code==8 || code==9|| code==19|| code==20 || code==18||code==6)))
		{
			IPD.remConstraints(globalNSAfirst);
			globalRemCon=1;
		}
		//if(mode==3|| (mode==2 && (code!=7 && code!=7 && code!=9 && code!=19 && code!=20 && code!=18))) 
		if((mode==2 && (code!=7 && code!=7 && code!=9 && code!=19 && code!=20 && code!=18 &&code!=6))) 
		{
			IPD.matrixAdaptation();
			globalMA=1;
		}

		IPD.Build();
		complexity+=1;
		/*
		if(!globalFirst&&first)
		{
			IPD.updatePerms();
			first=false;
		}*/
		tries++;
		if(tries>100) 
		{
			anomalies++;
			complexity-=100;
			complexity+=complex;
			return 4;
		}
		solveAns=IPD.Solve();
		if(solveAns==2)
		{
			return 3;
		}
		globalNSAfirst=false;
		if(solveAns==1)
		{
			return 1;
			if(inFesibilityMode==1) return 1;
			return 1;
		}
		if(mode==2) updateFractionalCounters();
		if(IPD.sameSolution())
		{
			//cout<<"[1] Can not find optimal Solution\n";
			return 0;
			mistakesNSA++;
			IPD.printML();
			return 0;
		}
		if(IPD.integralX())
		{
			IPD.CGA1();
			continue;
		}
		else
		{
			IPD.ConstructH();
			if (IPD.CGA2())
			{
				//cout<<endl<<"**"<<endl;				
				continue;
			}
			else
			{
				//cout<<"[2] Can not find optimal Solution"<<"endl";
				mistakesNSA++;
				return 0;
			}
		}
		return -1;
	}
}





double round(double x)
{
	if(x<0.5 || x==0.5) return 0.0;
	else return 1.0;
}



int Alg1(IPD &IPD,ifstream &myfile3)
{
	int T[5];
	double Result[smallM][N];
	int Hist[smallM];
	int i,j,resNSA,res1,res2,res3;
	int pi;
	int anomali=-1;
	int output;
	ifstream PF(permFile.c_str());
	IPD.initPerm(PF);
	globalALG1Iteration=0;
	for(i=0;i<MaxDecodingAttempts;i++)
	{
		globalALG1Iteration++;
		IPD.softReset();
		if(modePT)	resNSA=dynamicPT(IPD);
		else resNSA=NSA(IPD);
		if(resNSA==1 && i==0) return 1;
		if(resNSA==2 && anomali!=-1)
		{
			cout<<"Huston we have a problem";
		}
		if(resNSA==2) anomali=i;
		for(j=0;j<IPD.CostVector.row;j++)
				Result[i][j]=solution[j];
		if(resNSA)
		{
			cout<<"success"<<endl;
			if(modePT) return true;
			corrected1++;
			for(j=i-1;j>-1;j--)
			{
				if(Hist[j]>62 || Hist[j]<0)
				{
				//out<<"bad";
				}
				Perm(Hist[j],solution+SHIFT);
		
				
			}
			return true;
		}
		while(1)
		{
			pi=rand()%permMod;
			if (pi>0&&pi<globalN) break; //only for cyclic codes
		}
		Hist[i]=pi;
		IPD.IPerm(pi,IPD.CostVector.A);
		IPD.IHPerm(pi);
	}	
	if(modePT==1) return false;
	res1=IPD.MinDistance(Result);
	for(i=0;i<MaxDecodingAttempts;i++)
	{
		for(j=i-1;j>-1;j--)
		{
			if(Hist[j]>62 || Hist[j]<0)
			{
				//cout<<"bad";
			}
			Perm(Hist[j],Result[i]);
		}
	}
	res2=IPD.MinDistance(Result);
	res3=IPD.MinDistance(Result);
	for(i=0;i<MaxDecodingAttempts;i++)
	{
		for(j=0;j<globalN;j++)
			Result[i][j]=round(Result[i][j]);
	}

	res3=IPD.MinDistance(Result);
	
	for(i=0;i<globalN;i++)
	{
		solution[i]=Result[res3][i];
	}
	return false;
}


void formatFile(ifstream &input)
{
	int j=0,k=0;
	ofstream outfile;
	string line;
	outfile.open("temp");
	while(getline(input,line))
	{
		while(line[k])
		{
			outfile<<line[k]<<" ";
			k++;
			j++;
			if(j==127)
			{
				outfile<<endl;
				j=0;
			}
		}
	}
}
/*
int diff(ifstream &f1)
{
	ifstream f2 (pwd+"output.txt");
	int c1,c2;
	while(!f1.eof())
	{
		if(f2.eof()) return 1;
		f1>>c1;f2>>c2;
		if(c1!=c2)return 1;
	}
	if(!f2.eof()) return 1;
	return 0;
}
*/

void buildSource(ifstream &f,int S[MAX_ITER][63],int len)
{
	int i=0,j;
	while(i<MAX_ITER)
	{
		for(j=0;j<len;j++)
		f>>S[i][j];
		i++;
	}
}



void printSource()
{
	int j;
	return;
	for(j=0;j<len;j++)
	{
		cout<<S[j]<<" ";
	}
}





int Alg3(IPD &IPD,Xopt &XO,int currentDepth)
{
	double Result[smallM][N];
	int Hist[smallM],pointer,adoption=0;
	int i,j,resNSA,location;
	int pi;
	bool better;
	int output;
	bool noFeasible=false;
	cycles++;
	if(currentDepth>MaxDepth) return XO.res();
	if(currentDepth==0)
	{
		IPD.softReset();
	}
	if(currentDepth!=0)
	{
		IPD.softReset();
		insertConstraints(IPD,currentDepth);
	}
	if(currentDepth>0)
	{
		cout<<"cd: "<<currentDepth<<endl;
		if(Ones)
		{
			if(!IPD.adaptCV(currentDepth)) 
			{
				return XO.res();
			}
			inFesibilityMode=1;
			resNSA=NSA(IPD);
			inFesibilityMode=0;
			IPD.restoreCostVector();
			if(resNSA!=1)
			{
				return XO.res();
			}
			IPD.softReset();
			insertConstraints(IPD,currentDepth);
		}
	}
	resNSA=NSA(IPD,currentDepth);
	if(resNSA!=1)
	{
		currentDepth=currentDepth;
	}
	//if(resNSA==3) return XO.res();
	better=XO.betterSolution();
	if(countNSA && resNSA) NSAright++;
	countNSA=false;
	if(currentDepth!=0)
	{
		//here in the condition check that the solution is not integral
		//if (!isFeasibleSolution(IPD,currentDepth)) return 0;  ///*****should get back to that*****
	}
	if(resNSA==1 && currentDepth!=0)
	{
		alg3Action=true;
		if(countErors)
		{
			globalNSAerrors++;
			countErors=false;
			if(globalNSAerrors>globalErrs+GC+1 ||globalNSAerrors<globalErrs+GC+1)
			{
				GC=GC;
			}
		}
 		resNSA=resNSA;
		if (better) 
			adoption=XO.adoptSolution();
		if(adoption>1)
		{
			adoption=adoption;
		}
		return XO.res();

	}
	if(resNSA==1)
	{
		if (better) 
			adoption=XO.adoptSolution();
		if(adoption>1)
		{
			adoption=adoption;
		}
		return XO.res();
	}
	if(better || currentDepth==0)
	{
		location=findMostFractional(IPD);
		set1Constraint(currentDepth,location,0);
		Alg3(IPD,XO,currentDepth+1);
		set1Constraint(currentDepth,location,1);
		Ones=1;
		Alg3(IPD,XO,currentDepth+1);
		return XO.res();
	}
}




void print(ofstream &f,int **A,int n,int m)
{
	int i,j;
	return;
	//f<<"[";
	for(i=0;i<n;i++)
	{
	//	f<<"[";
		for(j=0;j<m;j++)
			f<<A[i][j]<<" ";
		f<<endl;
	//	f<<"]"<<endl;
	}
	//f<<"]";
}
int shiftReduction()
{
	int i,res=0;
	for(i=0;i<globalX;i++)
		if(leafTranslation[i]!=-1) res++;
	return res;
}

double distance(double A[],double B[],int len)
{
	double res=0;
	int i;
	for(i=0;i<len;i++)
		res+=(1.0*A[i]-B[i])*(1.0*A[i]-B[i]);
	return res;
}
int run();
int checkRightSolution()
{
	int k;int res=-1,res1=-1;
	double d;
	double tmp[N];
	globalSH=SHIFT;
	if(!((modePT==1 && mode==1) || mode==5 || mode==7)) SHIFT=0;
	for(k=0;k<globalN;k++)
	{
		tmp[k]=solution[k+SHIFT];
	}
	for(k=0;k<globalN;k++)
	{
		solution[k]=tmp[k];
	}
	for(k=0;k<globalN;k++)
		if(abs(original[k]-solution[k])>EPS) res=k;
	if(res!=-1)
	{
		res=res;
	}
	globalGT=0;
	globalRD=0;
	for(k=0;k<globalN;k++)
	{
		globalGT+=(abs(originalX[k]-original[k])*abs(originalX[k]-original[k]));
		globalRD+=(abs(originalX[k]-solution[k])*abs(originalX[k]-solution[k]));
	}
	globalGT/=globalN;
	globalRD/=globalN;
	if(res==-1)
	{
		res=res;
	}
	return res;
}


void saveSol(int k)
{
	int j;
	for(j=0;j<globalN;j++)
	{
		globalSolutions[k][j]=solution[SHIFT+j];
	}
}

void printSol()
{
	int j,i;
	for(j=0;j<globalN;j++)
	{
		cout<<j<<": "<<solution[j]<<endl;
		/*
		cout<<j<<": ";
		for(i=0;i<globalALG1RowPermutations;i++)
		{
			cout<<globalSolutions[i][j]<<" ";
		}
		cout<<endl;
		*/
	}
	cout<<endl;
}

bool belongs2Weaks(int x,int A[N])
{
	int i;
	for(i=0;i<globalN;i++)
	{
		if(A[i]==x)return true;
		if(A[i]==-1) return false;
	}
	return false;
}

void assignAvg(int ind)
{
	int i,j,dim;
	double avg=0;
	bool flag1,flag0,master0=false,master1=false;
	avg=0;flag0=false;flag1=false;
	for(i=0;i<globalAvgIterations;i++)
	{
		//if(globalSolutions[i][j]<0.51) flag0=true;
		//if(globalSolutions[i][j]<0.1) master0=true;
		//if(globalSolutions[i][j]>0.49) flag1=true;
		//if(globalSolutions[i][j]>0.9) master1=true;
		avg+=globalSolutions[i][ind];
	}
	/*
	if(master0 && originalX[j]>0.55)
	{
		originalX[j]=0;
		return;
	}
	else if(master1 && originalX[j]<0.45)
	{
		originalX[j]=1;
		return;
	}
	if(flag1 && originalX[j]<0.25)
	{
		originalX[j]=1;
		return;
	}
	if(flag0 && originalX[j]>0.75)
	{
		originalX[j]=0;
		return;
	}
	if(flag1 && flag0)
	{
		globalLim++;
		return;
	}*/
	avg/=globalAvgIterations;
	//if(avg<0.4) avg=0;
	//else if(avg>0.6) avg=1;
	/*
	if((avg<originalCandidate && avg>0.45) ||(avg>originalCandidate && avg<0.55))
	{
		globalLim++;
		return;
	}*/
//		if(avg>0 || avg<1) return;

	/*
	if(avg>0.35 && originalX[j]>0.45) 
	{
		originalX[j]=1;
		return;
	}*/
	
	/*
	if(originalCandidate>0.5 && avg<0.3)
	{
		globalLim++;
		cout<<"skipped"<<originalCandidate<<" "<<avg<<endl;
		return;
	}*/
	/*
	if(abs(avg-originalCandidate)>0.3)
	{
		avg=originalCandidate+((avg-originalCandidate)/4);
		OX4print=avg;
		originalX[j]=avg;
		globalLim++;
		return;
	}*/
	OX4print=avg;
	originalX[ind]=avg;
	if(abs(originalCandidate-avg)<0.1)
	{
		globalLim++;
	}
}

void roundSol()
{
	int j;
	bool first=true;
	for(j=0;j>globalN;j++)
	{
		if(solution[SHIFT+j]<1 && solution[SHIFT+j]>0 && first) 
		{
			cout<<"was rounded"<<endl;
			first=false;
		}
		if(solution[SHIFT+j]>0.5) solution[SHIFT+j]=1;
		else solution[SHIFT+j]=0;
	}
}

void saveOriginalX()
{
	int j;
	for(j=0;j<globalN;j++)
		originalX[globalN+j]=originalX[j];
}

void restoreOriginalX()
{
	int j;
	for(j=0;j<globalN;j++)
		originalX[j]=originalX[j+globalN];
}


int integralLocals()
{
	int j,res;
	j=0;
	while(localWeaks[j]!=-1)
	{
		if(originalX[localWeaks[j]]>0 && originalX[localWeaks[j]]<1) return j;
		j++;
	}
	return -1;
}

void guess(bool Indicator[N])
{
	int i,x,j,cnt=0;
	for(i=0;i<N;i++)
		Indicator[i]=false;
	while(cnt<globalWeakerDepth)
	{
		x=rand()%globalWeakerRange;
		if(Indicator[x]) continue;
		cnt++;
		Indicator[x]=true;
	}
}

int run()
{
	int i,j,k,x,t,borns,result,mistakes1=0,mistakes2=0,len=1000000,cnt,period=1,res1,refines;
	bool flag=false,success=true,successb=true,refineInput,skip=false,Indicator[N];
	double snr=MAX_ITER,d,d1;
	int errors[MAX_Errors];
	int err=0,pointer,i1,moveOn=0;
	double ratio=0;
	int success3=-1;
	Code C;
	Xopt XO;
	srand(clock()); //start
	//ReadCode("6339alex.code",&C);
//	print(G,C.H,C.m,C.n);
	ifstream myfile1 (pcMatrixFile.c_str());
	if(!myfile1.is_open()) return 0;
	//globalFreeze=false;
	IPD IPD(myfile1);
	globalFreeze=true;
	//formatFile(myfile4);
	len=IPD.getLen();
	//buildSource(myfile7,S,len);
	iteration=0;
	while(iteration<period)
	{
		switch (mode)
		{
			case 0:
				{
					IPD.reset();
					int AA[N];
					for(j=0;j<globalN;j++)
						AA[j]=original[j];
					j=IPD.checkCodeWord(AA);
					success=NSA(IPD);
					cnt=checkRightSolution();	
					if(cnt==-1)
					{
						iteration++;
	  					return iteration;
	  					break;
					}
					break;
				}
			case 1:
				{
					if(modePT==0)
					{
						success=Alg1(IPD,myfile3);
						cnt=checkRightSolution();
						if(cnt!=-1)
						{
							return iteration;
						}
						iteration++;
						IPD.reset(1);
						break;
					}
					else
					{
						IPD.saveHmatrix();
						if(!IPD.originalCodeWord())
						{
							k=k;
						}
						for(k=0;k<globalALG1RowPermutations;k++)
						{
							if(k==0)
							{
								IPD.permuteH4DyanmicPolitope();
							}
							else
							{
								IPD.permuteH4DyanmicPolitope();
							}
							if(1)
							{
								cout<<k<<" chance"<<endl;
							}
							success=Alg1(IPD,myfile3);
							if(!success && k<globalALG1RowPermutations-1) continue;
							if(!IPD.codeWord() &&success)
							{
								success=success;
								cout<<"not a code word"<<endl;
								if(!IPD.originalCodeWord())
								{
									i=i;
								}
							}
							cnt=checkRightSolution();
							//start
							IPD.restoreHmatrix();
							d=distance(originalX,solution,globalN);
							globalf1<<d<<endl;
							if(cnt==-1)
								globalh1<<"True"<<endl;
							else 
							{
								globalh1<<"False"<<endl;
							}
							mode=0;
							run();
							iteration=0;
							res1=checkRightSolution();
							
							d1=distance(originalX,solution+SHIFT,globalN);
							globalf6<<d1<<endl;
							if(res1==-1)
								globalh6<<"True"<<endl;
							else
							{
								globalh6<<"False"<<endl;
							}
							if(cnt!=res1)
							{
								if(cnt==-1)
								{
									cout<<"1 is better"<<endl;
								}
								else if(res1==-1)
								{
									cout<<"6 is better"<<endl;
								}
								else cout<<"both decoders are wrong"<<endl;
							}
							mode=1;
							//end
							IPD.restoreHmatrix();
							cout<<"GT"<<globalTot<<endl;
							iteration=0;
							if(cnt==-1)
							{
								iteration++;
								break;
							}
							//IPD.reset(1);
							return iteration;
							break;
						} //end FOR??????
						break;
					}
				}
				case 2:
				{
					Ones=0;
					saveOriginalX();
					XO.reset();
					countErors=true;
					IPD.saveCostVector();
					alg3Action=false;
					countNSA=true;
					IPD.saveHmatrix();				
					cycles=0;
					success=Alg3(IPD,XO,0);
					XO.setSolution();
					cnt=checkRightSolution();
					if(cnt!=-1)
					{
						/*
						restoreOriginalX();
						NSA(IPD);
						cnt=checkRightSolution();	
						if(cnt==-1)
						{
							cout<<"The Error is Here!"<<endl;
						}*/
						return iteration;
					}
					iteration++;
					IPD.reset(1);
					break;
				}
				case 3:
				{
					countErors=true;
					IPD.saveCostVector();
					alg3Action=false;
					countNSA=true;
					IPD.saveHmatrix();				
					success=NSA(IPD);
					cnt=checkRightSolution();
					if(cnt!=-1) return iteration;
					iteration++;
					IPD.reset(1);
					break;

				}
				case 4:
					{
						countErors=true;
						start=PT(IPD,globalSNR);
						cnt=checkRightSolution();
						if(cnt!=-1) return iteration;// needs to check if indeed needs to include all the constraints
						iteration++;
						break;
					}
				case 5:
					{
						success=dynamicPT(IPD);
						cnt=checkRightSolution();
						if(cnt!=-1) return iteration;
						iteration++;
						IPD.reset(1);
						break;
					}
				case 6:
					{
						//IPD.printG();
						if(globalTot>0)IPD.restoreHmatrix();
						IPD.ML();
						cnt=checkRightSolution();
						if(cnt!=-1) return iteration;
						iteration++;
						IPD.reset(1);
						break;
					}
				case 7:
					{
						IPD.reset();
						//printOriginalX();
						double orgGlobalSafeZone=globalSafeZone;
						borns=0;
						globalLim=0;
						saveOriginalX();
						//modePT=0;
						IPD.saveHmatrix();
						globalFirst=true;
						refineInput=false;
						globalR=0;k=0;refines=0;
						globalLim=0;
						SHIFT=0;
						mode=3;
						NSA(IPD);
						success=IPD.integral();
						if(!success) refineInput=1;
						skip=false;
						while(refines<primeIterations && refineInput)
						{
							globalRefined=true;
							global1BitUse=true;
							refineInput=true;
							if(refines<primeIterations && refineInput)
							{
								globalRecycle=1;
								//if(borns>GAP) globalLim=globalReBorn-borns+GAP;
								saveOriginalX();
								//if(refines==0 || moveOn)IPD.
								IPD.minWeaks();
								originalCandidate=originalX[localWeaks[(0)]];
								guess(Indicator);
								IPD.minWeaks();
								for(j=1;j<globalWeakerRange;j++)
								{
									if(Indicator[j])originalX[localWeaks[j]]=0.5;
								}
								borns++;
								History[borns-1]=localWeaks[(0)];
								//if(borns<GAP+1)originalX[localWeaks[0]]=0.5;

								for(t=0;t<globalAvgIterations;t++)
								{
									//globalR=(rand()%(globalM/2))+(globalM/4);
									//if(t>globalAvgIterations/2)originalX[localWeaks[0]]=0.5;
									globalR=0;
									//globalR=0;
									IPD.matrixAdaptation();
									IPD.addAfterAdaptation();
									IPD.permute1bitDecoder();
									success=dynamicPT(IPD); //add row perm. and the solution of mode 3
									saveSol(t);
								}
								restoreOriginalX();
								if(globalRecycle) assignAvg(History[borns-1]);
								cout<<"original: "<<original[History[borns-1]]<<"  originalX: "<<
									originalCandidate<<"  corrected: "<<OX4print<<endl;
					
								//moveOn=(integralLocals()==-1);
								//if(!moveOn) globalSafeZone=globalSafeZone/2;
								refines++;
								//if(moveOn)globalSafeZone+=k*GSNStep;
								if(refines%globalRoundCheck==(globalRoundCheck-1))
								{
									NSA(IPD);
									success=IPD.integral();
									if(success) 
									{
										skip=true;
										break;
									}
								}
								continue;
							}
						}
							globalSafeZone=orgGlobalSafeZone;
							SHIFT=0;
							if(!skip)
							{
								mode=3;
								NSA(IPD);
								success=IPD.integral();
							}

							if(refineInput && success) 
							{
								mode=7;
								cnt=checkRightSolution();
								cout<<endl<<endl;
								cout<<"ALG3: "<<cnt<<"   ,GT: "<<globalGT<<"   RD: "<<globalRD;
								if(globalGT<globalRD)
								{
									cout<<"  ML is better";
									globalML4++;
									//cout<<"ML4: "<<globalML4<<endl;
								}
								cout<<endl<<endl;
								//printSol();
								if(cnt==-1)
								{
									global1bitCorrection++;
									if (globalRD<globalGT)
										globalSGD4++;
									if(globalRD==globalGT)
										globalEVEN++;
								}
							}
							
							globalSuccess7=success;
							mode=7;/*
							if(!success)
							{
								borns++;
								refineInput=1;
								refines=0;
								//cout<<"DPT Active"<<endl;
								//if(borns==globalReBorn/2) originalX[History[0]]=1-originalX[History[0]];
								//if(borns==(3*globalReBorn)/4) originalX[History[0]]=1-originalX[History[1]];
								//if((borns%2)==0)globalLim=0;
							}*/
						cnt=checkRightSolution();
						/*
						d=distance(originalX,solution+SHIFT,globalN);
						globalf1<<d<<endl;
						if(cnt!=-1) cout<<"failed"<<endl;
						if(cnt==-1)
							globalh1<<"1"<<endl;
						else 
						{
							globalh1<<"0"<<endl;
						}
						mode=6;

						if(code>33 ||mode<4)
						{
							IPD.restoreHmatrix();
							restoreOriginalX();
							run();
							res1=checkRightSolution();
							d=distance(originalX,solution,globalN);
							globalf6<<d<<endl;
							if(res1==-1)
								globalh6<<"1"<<endl;
							else 
							{
								globalh6<<"0"<<endl;
							}
							if(refineInput) 
							{
								cout<<"ML: "<<res1<<endl<<endl;
								//printSol();
							}
						}
						mode=7;
						//start
						
						IPD.restoreHmatrix();
						*/
						if(cnt==-1)
						{
							iteration++;
							break;
						}
						else
						{
							//printSol();
						}
						return iteration;
					}
				}
				break;
			}
	return iteration;
	cout<<endl<<"FER LP = "<<(float) mistakes1/numOfIterations<<endl;
	cout<<endl<<"FER ML = "<<(float) mistakes2/numOfIterations<<endl;
		for(i=0;i<err;i++)
		 cout<<errors[i]<< ", ";
	cout<<endl;
	cin>>x;
}

void initSumulationConstants()
{
	switch(code)
	{
		case -2:
			{
				pcMatrixFile=pwd+"BCH_63x45.txt";
				gMatrixFile=pwd+"G_63x45.txt";
				permFile=pwd+"perms63.txt";
				globalSnrL=2;globalSnrH=5;globalStep=0.5;
				globalN=63;
				permMod=63;
				globalK=45;
				globalM=18;
				globalR=0;
				globalDmin=3;
				return;

			}
		case -1:
			{
				pcMatrixFile=pwd+"BCH_31x21.txt";
				gMatrixFile=pwd+"G_31x21.txt";
				permFile=pwd+"perms63.txt";
				globalSnrL=0;globalSnrH=4;globalStep=0.5;
				globalN=31;
				permMod=31;
				globalK=21;
				globalM=10;
				globalR=0;
				return;

			}
		case 0:
			{
				pcMatrixFile=pwd+"BCH_63x39.txt";
				gMatrixFile=pwd+"G_63x39.txt";
				permFile=pwd+"perms63.txt";
				globalSnrL=2;globalSnrH=6;globalStep=0.5;
				globalN=63;
				permMod=63;
				globalK=39;
				globalM=24;
				globalR=0;
				globalDmin=4;
				return;

			}
		case 1:
			{
				pcMatrixFile=pwd+"H_129_100_MDPC.txt";
				gMatrixFile=pwd+"G_129_100_MDPC.txt";
				permFile=pwd+"perms129.txt";
				globalSnrL=5;globalSnrH=5.5;globalStep=0.5;
				globalN=129;
				permMod=129;
				globalK=100;
				globalM=29;
				globalR=15;
				return;
			}
			case 2:
			{
				pcMatrixFile=pwd+"H_127_92_MDPC.txt";
				gMatrixFile=pwd+"G_127_92_MDPC.txt";
				permFile=pwd+"perms127.txt";// TODO: NEED 2 update!!!!!!!!!!!!!@
				globalSnrL=5;globalSnrH=5.0;globalStep=0.5;
				globalN=127;
				permMod=127;
				globalK=92;
				globalM=35;
				globalR=globalM/2+1;
				return;
			}
			case 3: //still need to build G!!
			{
				pcMatrixFile=pwd+"H_127_106_MDPC.txt";
				gMatrixFile=pwd+"G_127_106_MDPC.txt";
				permFile=pwd+"perms127.txt";
				globalSnrL=3;globalSnrH=6;globalStep=0.5;
				globalN=127;
				permMod=127;
				globalK=106;
				globalM=21;
				
				return;
			}
			case 4:
			{
				pcMatrixFile=pwd+"H_127_106_BCH.txt";
				gMatrixFile=pwd+"G_127_106_BCH.txt";
				permFile=pwd+"perms127.txt";
				globalSnrL=2;globalSnrH=10;globalStep=0.5;
				globalN=127;
				permMod=127;
				globalK=106;
				globalM=21;
				
				return;
			}
			case 5:
			{
				pcMatrixFile=pwd+"H_127_99_BCH.txt";
				gMatrixFile=pwd+"G_127_99_BCH.txt";
				permFile=pwd+"perms127.txt";
				globalSnrL=2;globalSnrH=5.5;globalStep=0.5;
				globalN=127;
				permMod=127;
				globalK=99;
				globalM=28;
				
				return;
			}
			case 6:
			{
				pcMatrixFile=pwd+"H_127_92_BCH.txt";
				gMatrixFile=pwd+"G_127_92_BCH.txt";
				permFile=pwd+"perms127.txt";
				globalSnrL=3;globalSnrH=5;globalStep=0.5;
				globalN=127;
				permMod=127;
				globalK=92;
				globalM=35;
				return;
			}
			case 7:
			{
				frozenFile=pwd+"4frozen64.txt";
				politopeFile=pwd+"poly448.txt";
				pcMatrixFile=pwd+"4H_polar_64_32.txt";
				gMatrixFile=pwd+"G_polar_64_32.txt";
				permFile=pwd+"perms64.txt";
				globalSnrL=2.5;globalSnrH=2.5;globalStep=0.5;
				globalN=64;
				permMod=64;
				globalK=32;
				globalM=32;
				globalR=0;
				return;
			}
			case 8:
			{
				frozenFile=pwd+"6frozen128.txt";
				pcMatrixFile=pwd+"6H_polar_128_64.txt";
				gMatrixFile=pwd+"G_polar_128_64.txt";
				permFile=pwd+"perms128.txt";
				globalSnrL=3.5;globalSnrH=4;globalStep=0.5;
				globalN=128;
				permMod=128;
				globalK=64;
				globalM=64;
				globalR=0;
				return;
			}
			case 9:
			{
				frozenFile=pwd+"frozen64.txt";
				politopeFile=pwd+"poly448.txt";
				pcMatrixFile=pwd+"H_polar_64_32.txt";
				gMatrixFile=pwd+"G_polar_64_32.txt";
				permFile=pwd+"perms64.txt";
				globalSnrL=1;globalSnrH=4;globalStep=0.5;
				globalN=64;
				permMod=64;
				globalK=32;
				globalM=32;
				globalR=0;
				return;
			}
			case 10:
			{
				pcMatrixFile=pwd+"H15_11.txt";
				gMatrixFile=pwd+"G15_11.txt";
				permFile=pwd+"perms15.txt";
				globalSnrL=0;globalSnrH=5;globalStep=1;
				globalN=15;
				permMod=15;
				globalK=4;
				globalM=11;
				globalR=globalM/2+1;
				return;
			}
			case 11:
			{
				pcMatrixFile=pwd+"H1511.txt";
				gMatrixFile=pwd+"G1511.txt";
				permFile=pwd+"perms15.txt";
				globalSnrL=4;globalSnrH=5;globalStep=1;
				globalN=15;
				permMod=15;
				globalK=4;
				globalM=11;
				globalR=globalM/2+1;
				return;
			}
			case 12:
			{
				pcMatrixFile=pwd+"H7x4.txt";
				gMatrixFile=pwd+"G7x4.txt";
				permFile=pwd+"perms7.txt";
				globalSnrL=0;globalSnrH=5;globalStep=1;
				globalN=7;
				permMod=7;
				globalK=4;
				globalM=3;
				globalR=globalM/2+1;
				return;
			}
			case 13:
			{
				pcMatrixFile=pwd+"H15x4.txt";
				gMatrixFile=pwd+"G15x4.txt";
				permFile=pwd+"perms15.txt";
				globalSnrL=0;globalSnrH=5;globalStep=1;
				globalN=15;
				permMod=15;
				globalK=11;
				globalM=4;
				globalR=globalM/2+1;
				return;
			}
			case 14:
			{
				pcMatrixFile=pwd+"adaptedH.txt";
				gMatrixFile=pwd+"G_129_100_MDPC.txt";
				permFile=pwd+"perms129.txt";
				globalSnrL=4;globalSnrH=5.5;globalStep=0.5;
				globalN=129;
				permMod=129;
				globalK=100;
				globalM=29;
				globalR=globalM/2+1;
				return;
			}
			case 15:
			{
				pcMatrixFile=pwd+"H15Ham.txt";
				gMatrixFile=pwd+"G15Ham.txt";
				permFile=pwd+"perms15.txt";
				globalSnrL=2;globalSnrH=5;globalStep=1;
				globalN=15;
				permMod=15;
				globalK=11;
				globalM=4;
				globalR=globalM/2+1;
				return;
			}
			case 16:
			{
				pcMatrixFile=pwd+"RealH15.txt";
				gMatrixFile=pwd+"RealG15.txt";
				permFile=pwd+"perms15.txt";
				globalSnrL=0;globalSnrH=5;globalStep=1;
				globalN=15;
				permMod=15;
				globalK=11;
				globalM=4;
				globalR=globalM/2+1;
				return;
			}
			case 17:
			{
				pcMatrixFile=pwd+"H31x16.txt";
				gMatrixFile=pwd+"G31x16.txt";
				permFile=pwd+"perms31.txt";
				globalSnrL=2;globalSnrH=5;globalStep=1;
				globalN=31;
				permMod=31;
				globalK=16;
				globalM=15;
				globalR=globalM/2+1;
				return;
			}
			case 18:
			{
				pcMatrixFile=pwd+"H_127_85_BCH.txt";
				gMatrixFile=pwd+"G_127_85_BCH.txt";
				permFile=pwd+"perms127.txt";
				globalSnrL=2;globalSnrH=4;globalStep=0.5;
				globalN=127;
				permMod=127;
				globalK=85;
				globalM=42;
				//globalR=globalM/2+1;
				return;
			}
			case 19:
			{
				pcMatrixFile=pwd+"H_127_78_BCH.txt";
				gMatrixFile=pwd+"G_127_78_BCH.txt";
				permFile=pwd+"perms127.txt";
				globalSnrL=2;globalSnrH=5;globalStep=1;
				globalN=127;
				permMod=127;
				globalK=78;
				globalM=47;
				//globalR=globalM/2+1;
				return;
			}
			case 20:
			{
				pcMatrixFile=pwd+"H_127_71_BCH.txt";
				gMatrixFile=pwd+"G_127_71_BCH.txt";
				permFile=pwd+"perms127.txt";
				globalSnrL=2.5;globalSnrH=4;globalStep=0.5;
				globalN=127;
				permMod=127;
				globalK=71;
				globalM=56;
				//globalR=globalM/2+1;
				return;
			}
	}
}



void readExConstants()
{
	string str=pwd+"params.txt";cout<<str.c_str()<<endl;
	ifstream XXfile(str.c_str());
	XXfile>>modePT;cout<<modePT<<endl;
	XXfile>>mode;
	XXfile>>code;
	XXfile>>FACTOR;
	XXfile>>EnoughErrors;
	XXfile>>MaxGlobalTot;
	XXfile>>globalSnrL;
	cout<<"globalSNRL: "<<globalSnrL<<endl;
	XXfile>>globalSnrH;
	XXfile>>MaxDecodingAttempts;
	cout<<"MAX Decoding***: "<<MaxDecodingAttempts<<endl;
	XXfile>>MaxDepth;
}
		
void readFrozen(ifstream &Gfile)
{
	int i,j,dump=0;
	for(j=0;j<globalK;j++)
	{
	Gfile>>dump;
	globalFrozen[j]=dump-1;
	}
}

void writeReport(int simulations,double ratio[RA],double TC[RA])
{
	
	string str;
	int i,j;
	ostringstream convert;
	convert<<modePT<<"_"<<mode<<"_"<<code<<".txt";
	//cout<<"writing report"<<endl;
	str=pwd+"report"+convert.str();
	ofstream XXfile(str.c_str());
	XXfile<<"MaxDepth="<<MaxDepth<<"  ";
	XXfile<<"modePT="<<modePT<<"  ";
	XXfile<<"mode="<<mode<<"  ";
	XXfile<<"code="<<code<<"  ";
	XXfile<<"RemCon="<<globalRemCon<<"  ";
	XXfile<<"MA="<<globalMA<<"  ";
	XXfile<<"FACTOR="<<FACTOR<<endl;
	XXfile<<"PrimeIter="<<primeIterations<<endl;
	XXfile<<"WeakerDepth="<<globalWeakerDepth<<endl;
	XXfile<<"AvgIterations="<<globalAvgIterations<<endl;
	XXfile<<"globalRoundCheck="<<globalAvgIterations<<endl;
	XXfile<<"AvgIterations="<<globalTotTries<<endl;
	XXfile<<"globalRoundCheck="<<globalWeakerStep<<endl;
	for(i=0;i<simulations;i++)
	{
		XXfile<<globalSnrL+i*globalStep<<": ";
		if(globalSnrL+i*globalStep== (int) (globalSnrL+i*globalStep))
		{
			XXfile<<"  ";
		}
		XXfile<<ratio[i]<<"  	"<<"iterations: "<<LPIR[i]<<"       "<<"comp:"<<TC[i]<<"    1bit correction: "<<B1R[i]<<"  ML4: "<<globalML4R[i]<<
			"    SGD4 "<<globalSGD4R[i]<<"    EVEN: "<<globalEVENR[i]<<endl;
	}
}
int executeCommandLine(string cmdLine, DWORD & exitCode)
{
   PROCESS_INFORMATION processInformation = {0};
   STARTUPINFO startupInfo                = {0};
   startupInfo.cb                         = sizeof(startupInfo);
   int nStrBuffer                         = cmdLine.length() + 50;
 

   // Create the processz

   WaitForSingleObject(ghMutex, INFINITE);
   BOOL result = CreateProcess(NULL, (LPSTR) cmdLine.c_str(), 
                               NULL, NULL, FALSE, 
                               NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW, 
                               NULL, NULL, &startupInfo, &processInformation);

    ReleaseMutex(ghMutex);

   if (!result)
   {
	   cout<<"error cmdline"<<endl;
	   cout<<cmdLine.c_str()<<endl;
	   return -1;
   }
   else
   {
      // Successfully created the process.  Wait for it to finish.
      WaitForSingleObject( processInformation.hProcess, INFINITE );
      // Get the exit code.
      result = GetExitCodeProcess(processInformation.hProcess, &exitCode);
      // Close the handles.
      CloseHandle( processInformation.hProcess );
      CloseHandle( processInformation.hThread );
      // We succeeded.
      return (float) exitCode;
   }
}

void openSideProcess(void *v)
{
	int errors;
	DWORD EC;
	float res,tmpGR;
	ostringstream convert2run;
	WaitForSingleObject(ghMutex, INFINITE);
	convert2run<<globalThreadNum<<".exe";
	convert2run<<convert.str()<<" "<<globalThreadNum;
	globalThreadNum+=1;
	globalCMD=pwd+"AtomClP"+convert2run.str();	
	ReleaseMutex(ghMutex);
	res=executeCommandLine(globalCMD,EC);
    WaitForSingleObject(ghMutex, INFINITE);
	globalGR+=res;
    ReleaseMutex(ghMutex);
    _endthread();
}

bool sameSol(double X[N],double Y[N])
{
	int i;
	for (i=0;i<globalN;i++)
		if(X[i]!=Y[i]) return false;
	return true;
}

void main()
{
	
	LPDWORD exitCode=NULL;
	bool bw;
	double rML,r1,r2,r4,rNSA,rRML,rTot,NSAR,currentDist;
	int errs[1000];
	double li;
	float tmpGR;
	int backupDepth,backupStep,backupRange;
	int jj,kkk,i,k=0,tot=0,err=0,period=1,t,tries,kMin,winnerChanged;
	double ratio[RA],Complexity[RA],complexWE,snr,TC[RA],min,solCand[N][N],candiInd[N],winner[N],tt;
	int l,v,simulations,workIterations,kk,g;
	ifstream sf[smallM][smallM]; string filePath;ostringstream fp;
	logFile.open("C:\\MyProj\\glpkLog");
	ifstream FF;
	initSumulationConstants();
	simulations=(globalSnrH-globalSnrL)/globalStep+1;
	FF.open(frozenFile.c_str());
	glp_term_out(GLP_OFF);
	readFrozen(FF);
	backupDepth=globalWeakerDepth;	backupStep=globalWeakerStep;backupRange=globalWeakerRange;
	globalf1.open("C:\\MyProj\\dPT1.txt");
	globalh1.open("C:\\MyProj\\dPT1_true.txt");
	globalf6.open("C:\\MyProj\\dML6.txt");
	globalh6.open("C:\\MyProj\\dML6_true.txt");
	P1=glp_create_prob();
	
	/*
	for(li=0;li<500000000;li++)
	{	
	}
	cout<<"start"<<endl;

	for(li=0;li<5000000000;li++)
	{
		for(k=0;k<10;k++)
		{
			for(i=0;i<10;i++)
			{
				tt=3.43*2.34+k+i;
			}
		}
	}
	cout<<"stop"<<endl;
	return;*/
	for(l=0;l<simulations;l++)
	{
		snr=globalSnrL+l*globalStep;
		/*if(l>0)
		{
			globalWeakerDepth-=2;
			globalWeakerRange=backupRange;
			globalWeakerRange-=3;
			backupRange-=3;
		}*/
		
		globalLPI=(unsigned int) 0;
		LPIR[l]=0;
		globalML4=0;globalSGD4=0;globalEVEN=0;
		global1bitCorrection=0;
		globalSNR=snr;
		globalf1<<"SNR="<<globalSNR<<endl;globalh1<<"SNR="<<globalSNR<<endl;
		globalf6<<"SNR="<<globalSNR<<endl;globalh6<<"SNR="<<globalSNR<<endl;
		k=tot=err=0;
		complexity=0;
		MLError=0;
		clock_t begin=clock();
		while(1)
		{
			globalFreeze=false;
			tries=0;
			globalWeakerDepth=backupDepth;
			globalWeakerRange=backupRange;
			globalWeakerStep=backupStep;globalConverged=false;
			for(k=0;k<globalTotTries;k++)
				candiInd[k]=1000;
			winnerChanged=0;
			globalRefined=false;
			for(jj=0;jj<globalN;jj++)
				winner[-1]=0;;
			if (mode==7)
			{
				//globalTotTries=(int) (log10((double)globalWeakerStep)/log10(2.0));
				while(tries<globalTotTries)
				{
					//primeIterations=1+rand()%4;
					if(tries>0)
					{
						//globalWeakerDepth++;
						//globalWeakerRange=globalWeakerDepth+(rand()%(5*tries));
						cout<<globalWeakerDepth<<" "<<globalWeakerRange<<endl;
						for (kk=0;kk<globalN;kk++)
							originalX[kk]=globalSpecialBackup[kk];
					}
					//global1BitUse=false;
					k=run();
					globalWeakerRange=backupRange+rand()%20;
					if(!globalRefined) break;
					currentDist=distance(originalX,solution+SHIFT,globalN);
					if(!(currentDist>globalDistWinner && globalSuccess7))
					{
						if(distance(originalX,solution+SHIFT,globalN)==distance(originalX,winner,globalN))winnerChanged++;
						for(jj=0;jj<globalN;jj++)
							winner[jj]=solution[jj+SHIFT];
					}
					globalDistWinner=distance(originalX,winner,globalN);
					if(globalDistWinner<globalDmin ||(winnerChanged==1 && globalSuccess7)) break;
					//if(globalSuccess7) break;
					/*if(!global1BitUse) break;
					globalFreeze=true;
					if(globalSuccess7) 
						break;*/
						//candiInd[tries]=globalRD;
					//if(globalRD<0.14) break;
					/*bw=false;
					min=1000;
					kMin=0;
					for(kk=0;kk<tries;kk++)
					{
						if(candiInd[kk]<min)
						{
							kMin=kk;
							min=candiInd[kk];
						}
					}
					if(sameSol(solution,solCand[kMin]))
					{
						break;
					}
					if(globalSuccess7)
					{
						if(globalWeakerStep!=1)globalWeakerStep/=2;
						if(globalConverged)
							globalWeakerDepth=(globalWeakerDepth+globalWeakerStep)%45;
						else 
							globalWeakerDepth=(globalWeakerDepth-globalWeakerStep)%45;
						globalConverged=true;
						
					}
					else
					{
						if(globalConverged)
							if(globalWeakerStep!=1) globalWeakerStep/=2;
						globalWeakerDepth=(globalWeakerDepth+globalWeakerStep)%45;
					}
					for(kk=0;kk<globalN;kk++)
						solCand[tries][kk]=solution[kk];*/
					cout<<"iter: "<<tries+1<<"  current Ignoring Depth: "<<globalWeakerDepth<<endl;
					tries++;
				}/*
				min=1000;
				kMin=0;
				for(kk=0;kk<tries;kk++)
				{
					if(candiInd[kk]<min)
					{
						kMin=kk;
						min=candiInd[kk];
					}
				}*/
				/*
				if(global1BitUse)
				{
					for(kk=0;kk<globalN;kk++)
						solution[kk]=solCand[kMin][kk];
				}
*/				
				if(globalRefined)
				{
					for(jj=0;jj<globalN;jj++)
								solution[jj+SHIFT]=winner[jj];
				}
				k=(checkRightSolution()==-1);
			}

			
			else	
			{
				k=run();
			}
			tot+=k; 
			if(k<period)
			{
				tot++;
				errs[err]=tot;
				err++;
				globalErrs=err;
			}
			globalTot=tot;
			ratio[l]=(1.0*err)/(1.0*tot);
			r1=(1.0*corrected1)/(1.0*tot);
			r2=(1.0*corrected2)/(1.0*tot);
			r4=(1.0*corrected4)/(1.0*tot);
			GR=ratio[l];
			corrected3=corrected3;
			complex=complexity/tot;
			Complexity[l]=complex;
			logFile<<"iterations:  "<<tot<<endl;
			if(MLError!=err)
			{
				err=err;
			}
			B1R[l]=(1.0*global1bitCorrection)/(1.0*globalTot);
 			if(err>EnoughErrors) break;
			//if(tot>500) break;
			LPIR[l]=(1.0*globalLPI)/tot;
			globalML4R[l]=(1.0*globalML4)/tot;
			globalSGD4R[l]=(1.0*globalSGD4)/tot;
			globalEVENR[l]=(1.0*globalEVEN)/tot;
			if(tot%1==0)
			{
				cout<<GR<<" "<<tot<<" "<<LPIR[l]<<" SNR: "<<globalSNR<<" 1bit: "<<B1R[l]<<" "<<global1bitCorrection
					<<" ML4: "<<globalML4R[l]
					/*<<"  SHIFT: "<<globalSH<<" "
					<<globalN*log10(1.0*globalN)/log10(2.0)*/<<endl;
				//cout<<"MLR: "<<((1.0*MLError)/tot)<<endl;
			}
			//cout<<globalSolve<<endl;
			for(kkk=0;kkk<globalN;kkk++)
				winner[kkk]=-1;
			clock_t end=clock();
			TC[l]= double ((diffclock(end,begin)/tot));
			writeReport(simulations,ratio,TC);
		}
	}
	writeReport(simulations,ratio,TC);
	cin>>err;
	cout<<"end";
}//compress the pairs!!