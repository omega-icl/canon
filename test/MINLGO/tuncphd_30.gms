POSITIVE VARIABLES x1,x2,x3,x4,x5,x6;
FREE VARIABLE obj;
EQUATIONS ob,c1,c2;

ob.. 0.0204*x1*x4*(x1+x2+x3) + 0.0187*x2*x3*(x1+1.57*x2+x4) + 0.0607*x1*x4*x5*x5*(x1+x2+x3) + 0.0437*x2*x3*x6*x6*(x1+1.57*x2+x4) =e= obj;
c1.. 0.001*x1*x2*x3*x4*x5*x6 - 2.07 =g= 0;
c2.. 0.00062*x1*x4*x5*x5*(x1+x2+x3)+0.00058*x2*x3*x6*x6*(x1+1.57*x2+x4) =l= 1;

Model test /all/;
x1.l = 5.332666;
x2.l = 4.656744;
x3.l = 10.43299;
x4.l = 120.08120;
x5.l = 0.7526074;
x6.l = 0.87865084;

x1.up = 1e2;
x2.up = 1e2;
x3.up = 1e2;
x4.up = 1e2;
x5.up = 1e2;
x6.up = 1e2;

*OPTION NLP = BARON;

Solve test using nlp minimizing obj;
display x1.l,x2.l,x3.l,x4.l,x5.l,x6.l;
