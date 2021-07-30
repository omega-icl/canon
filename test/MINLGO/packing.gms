* Inspired by: https://doi.org/10.1007/s10898-012-9948-6
SETS
  N      Circles  / 1*10 /;
ALIAS (N2, N);

PARAMETERS
  R(N)   Radii
   / 1 6.0
     2 5.5
     3 5.0
     4 4.8
     5 4.3
     6 4.0
     7 3.8
     8 3.3
     9 2.9
    10 2.3 /;

SCALAR
  W      Width of arrangement / 30 /
  LMAX   Max length of arrangement;
LMAX = SUM(N,R(N));

VARIABLES
  X(N)   Centre X-coordinate
  Y(N)   Centre Y-coordinate
  L      Total lenfth of arrangement;

X.lo(N) = R(N);
Y.lo(N) = R(N);
X.up(N) = LMAX - R(N);
Y.up(N) = W - R(N);

EQUATIONS xposmax(N), distmin(N,N2), symbreakx, symbreaky;

*objmax .. L =L= LMAX;
xposmax(N) .. L - X(N) =G= R(N);
distmin(N,N2)$(ORD(N)<ORD(N2)) .. POWER(X(N)-X(N2),2) + POWER(Y(N)-Y(N2),2) =G= POWER(R(N)+R(N2),2);
symbreakx .. X('1') =L= X('2');
symbreaky .. Y('1') =L= Y('2');

*OPTION QCP = GUROBI;
*$onecho > gurobi.opt
*nonconvex 2
*$offecho

*OPTION QCP = ANTIGONE;
OPTION OPTCR = 1e-3;
OPTION OPTCA = 1e-5;
OPTION RESLIM = 86400;
*OPTION DECIMALS = 8;
*$onecho > snopt.opt
*Major feasibility tolerance 1.0e-10
*$offecho

MODEL TEST9 /ALL/;
TEST9.OPTFILE = 1;
SOLVE TEST9 USING QCP MINIMIZING L;

