VARIABLES x, y, z;
INTEGER VARIABLE y;
x.lo = 1;  x.up = 20;
y.lo = 1;  y.up = 20;

EQUATIONS f, g1, g2, g3;

f  .. z =E= -6*x - y;
g1 .. 0.3*POWER(x-8,2) + 0.04*POWER(y-6,4) + 0.1*EXP(2*x)/POWER(y,4) =L= 56;
g2 .. 1/x + 1/y - SQRT(x)*SQRT(y) =L= -4;
g3 .. 2*x - 5*y =L= -1;

OPTION MINLP = DICOPT;
OPTION OPTCR = 1e-7;
OPTION OPTCA = 1e-7;

MODEL TEST1 /ALL/;
SOLVE TEST1 USING MINLP MINIMIZING z;

