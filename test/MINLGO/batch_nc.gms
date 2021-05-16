$offlisting
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*         68        7       60        1        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*         35       23       12        0        0        0        0        0
*  FX      0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*        161       79       82        0
*
*  Solve m using MINLP minimizing objvar;


Variables  x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19
          ,x20,x21,x22,b23,b24,b25,b26,b27,b28,b29,b30,b31,b32,b33,b34,objvar;

Binary Variables  b23,b24,b25,b26,b27,b28,b29,b30,b31,b32,b33,b34;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
          ,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36
          ,e37,e38,e39,e40,e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,e51,e52,e53
          ,e54,e55,e56,e57,e58,e59,e60,e61,e62,e63,e64,e65,e66,e67,e68;


e1.. -(250*x7**0.6*x1 + 250*x8**0.6*x2 + 250*x9**0.6*x3 + 250*x10**0.6*x4 + 250
     *x11**0.6*x5 + 250*x12**0.6*x6) + objvar =E= 0;

e2..    x7 - 7.9*x13 =G= 0;

e3..    x8 - 2*x13 =G= 0;

e4..    x9 - 5.2*x13 =G= 0;

e5..    x10 - 4.9*x13 =G= 0;

e6..    x11 - 6.1*x13 =G= 0;

e7..    x12 - 4.2*x13 =G= 0;

e8..    x7 - 0.7*x14 =G= 0;

e9..    x8 - 0.8*x14 =G= 0;

e10..    x9 - 0.9*x14 =G= 0;

e11..    x10 - 3.4*x14 =G= 0;

e12..    x11 - 2.1*x14 =G= 0;

e13..    x12 - 2.5*x14 =G= 0;

e14..    x7 - 0.7*x15 =G= 0;

e15..    x8 - 2.6*x15 =G= 0;

e16..    x9 - 1.6*x15 =G= 0;

e17..    x10 - 3.6*x15 =G= 0;

e18..    x11 - 3.2*x15 =G= 0;

e19..    x12 - 2.9*x15 =G= 0;

e20..    x7 - 4.7*x16 =G= 0;

e21..    x8 - 2.3*x16 =G= 0;

e22..    x9 - 1.6*x16 =G= 0;

e23..    x10 - 2.7*x16 =G= 0;

e24..    x11 - 1.2*x16 =G= 0;

e25..    x12 - 2.5*x16 =G= 0;

e26..    x7 - 1.2*x17 =G= 0;

e27..    x8 - 3.6*x17 =G= 0;

e28..    x9 - 2.4*x17 =G= 0;

e29..    x10 - 4.5*x17 =G= 0;

e30..    x11 - 1.6*x17 =G= 0;

e31..    x12 - 2.1*x17 =G= 0;

e32.. x1*x18 =G= 6.4;

e33.. x2*x18 =G= 4.7;

e34.. x3*x18 =G= 8.3;

e35.. x4*x18 =G= 3.9;

e36.. x5*x18 =G= 2.1;

e37.. x6*x18 =G= 1.2;

e38.. x1*x19 =G= 6.8;

e39.. x2*x19 =G= 6.4;

e40.. x3*x19 =G= 6.5;

e41.. x4*x19 =G= 4.4;

e42.. x5*x19 =G= 2.3;

e43.. x6*x19 =G= 3.2;

e44.. x1*x20 =G= 1;

e45.. x2*x20 =G= 6.3;

e46.. x3*x20 =G= 5.4;

e47.. x4*x20 =G= 11.9;

e48.. x5*x20 =G= 5.7;

e49.. x6*x20 =G= 6.2;

e50.. x1*x21 =G= 3.2;

e51.. x2*x21 =G= 3;

e52.. x3*x21 =G= 3.5;

e53.. x4*x21 =G= 3.3;

e54.. x5*x21 =G= 2.8;

e55.. x6*x21 =G= 3.4;

e56.. x1*x22 =G= 2.1;

e57.. x2*x22 =G= 2.5;

e58.. x3*x22 =G= 4.2;

e59.. x4*x22 =G= 3.6;

e60.. x5*x22 =G= 3.7;

e61.. x6*x22 =G= 2.2;

e62.. 250000*x18/x13 + 150000*x19/x14 + 180000*x20/x15 + 160000*x21/x16 + 
      120000*x22/x17 =L= 6000;

e63..    x1 - b23 - 2*b29 =E= 1;

e64..    x2 - b24 - 2*b30 =E= 1;

e65..    x3 - b25 - 2*b31 =E= 1;

e66..    x4 - b26 - 2*b32 =E= 1;

e67..    x5 - b27 - 2*b33 =E= 1;

e68..    x6 - b28 - 2*b34 =E= 1;

* set non-default bounds
x1.lo = 1; x1.up = 4;
x2.lo = 1; x2.up = 4;
x3.lo = 1; x3.up = 4;
x4.lo = 1; x4.up = 4;
x5.lo = 1; x5.up = 4;
x6.lo = 1; x6.up = 4;
x7.lo = 300; x7.up = 3000;
x8.lo = 300; x8.up = 3000;
x9.lo = 300; x9.up = 3000;
x10.lo = 300; x10.up = 3000;
x11.lo = 300; x11.up = 3000;
x12.lo = 300; x12.up = 3000;
x13.lo = 86.4583333333333; x13.up = 379.746835443038;
x14.lo = 42.5; x14.up = 882.352941176471;
x15.lo = 89.25; x15.up = 833.333333333333;
x16.lo = 23.3333333333333; x16.up = 638.297872340426;
x17.lo = 21; x17.up = 666.666666666667;
x18.lo = 2.075; x18.up = 8.3;
x19.lo = 1.7; x19.up = 6.8;
x20.lo = 2.975; x20.up = 11.9;
x21.lo = 0.875; x21.up = 3.5;
x22.lo = 1.05; x22.up = 4.2;

Model m / all /;

m.limrow=0; m.limcol=0;
m.tolproj=0.0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set MINLP $set MINLP MINLP
Solve m using %MINLP% minimizing objvar;
