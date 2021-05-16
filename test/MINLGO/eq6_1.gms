$offlisting
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*         61        1        0       60        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*         17       17        0        0        0        0        0        0
*  FX      0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*        161       33      128        0
*
*  Solve m using NLP minimizing objvar;


Variables  x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,objvar;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
          ,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36
          ,e37,e38,e39,e40,e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,e51,e52,e53
          ,e54,e55,e56,e57,e58,e59,e60,e61;


e1.. -sqrt(sqr(x1 - x2) + sqr(x9 - x10)) =L= -2.995353;

e2.. -sqrt(sqr(x1 - x3) + sqr(x9 - x11)) =L= -2.532248;

e3.. -sqrt(sqr(x1 - x4) + sqr(x9 - x12)) =L= -2.638959;

e4.. -sqrt(sqr(x1 - x5) + sqr(x9 - x13)) =L= -2.638959;

e5.. -sqrt(sqr(x1 - x6) + sqr(x9 - x14)) =L= -2.121321;

e6.. -sqrt(sqr(x1 - x7) + sqr(x9 - x15)) =L= -1.914214;

e7.. -sqrt(sqr(x1 - x8) + sqr(x9 - x16)) =L= -2.828428;

e8.. -sqrt(sqr(x2 - x3) + sqr(x10 - x11)) =L= -2.699173;

e9.. -sqrt(sqr(x2 - x4) + sqr(x10 - x12)) =L= -2.805884;

e10.. -sqrt(sqr(x2 - x5) + sqr(x10 - x13)) =L= -2.805884;

e11.. -sqrt(sqr(x2 - x6) + sqr(x10 - x14)) =L= -2.288246;

e12.. -sqrt(sqr(x2 - x7) + sqr(x10 - x15)) =L= -2.081139;

e13.. -sqrt(sqr(x2 - x8) + sqr(x10 - x16)) =L= -2.995353;

e14.. -sqrt(sqr(x3 - x4) + sqr(x11 - x12)) =L= -2.342779;

e15.. -sqrt(sqr(x3 - x5) + sqr(x11 - x13)) =L= -2.342779;

e16.. -sqrt(sqr(x3 - x6) + sqr(x11 - x14)) =L= -1.825141;

e17.. -sqrt(sqr(x3 - x7) + sqr(x11 - x15)) =L= -1.618034;

e18.. -sqrt(sqr(x3 - x8) + sqr(x11 - x16)) =L= -2.532248;

e19.. -sqrt(sqr(x4 - x5) + sqr(x12 - x13)) =L= -2.44949;

e20.. -sqrt(sqr(x4 - x6) + sqr(x12 - x14)) =L= -1.931852;

e21.. -sqrt(sqr(x4 - x7) + sqr(x12 - x15)) =L= -1.724745;

e22.. -sqrt(sqr(x4 - x8) + sqr(x12 - x16)) =L= -2.638959;

e23.. -sqrt(sqr(x5 - x6) + sqr(x13 - x14)) =L= -1.931852;

e24.. -sqrt(sqr(x5 - x7) + sqr(x13 - x15)) =L= -1.724745;

e25.. -sqrt(sqr(x5 - x8) + sqr(x13 - x16)) =L= -2.638959;

e26.. -sqrt(sqr(x6 - x7) + sqr(x14 - x15)) =L= -1.207107;

e27.. -sqrt(sqr(x6 - x8) + sqr(x14 - x16)) =L= -2.121321;

e28.. -sqrt(sqr(x7 - x8) + sqr(x15 - x16)) =L= -1.914214;

e29..  - x1 =L= 1.210786;

e30..  - x2 =L= 1.043861;

e31..  - x3 =L= 1.506966;

e32..  - x4 =L= 1.400255;

e33..  - x5 =L= 1.400255;

e34..  - x6 =L= 1.917893;

e35..  - x7 =L= 2.125;

e36..  - x8 =L= 1.210786;

e37..    x1 =L= 1.210786;

e38..    x2 =L= 1.043861;

e39..    x3 =L= 1.506966;

e40..    x4 =L= 1.400255;

e41..    x5 =L= 1.400255;

e42..    x6 =L= 1.917893;

e43..    x7 =L= 2.125;

e44..    x8 =L= 1.210786;

e45..  - x9 =L= 3.710786;

e46..  - x10 =L= 3.543861;

e47..  - x11 =L= 4.006966;

e48..  - x12 =L= 3.900255;

e49..  - x13 =L= 3.900255;

e50..  - x14 =L= 4.417893;

e51..  - x15 =L= 4.625;

e52..  - x16 =L= 3.710786;

e53..    x9 =L= 3.710786;

e54..    x10 =L= 3.543861;

e55..    x11 =L= 4.006966;

e56..    x12 =L= 3.900255;

e57..    x13 =L= 3.900255;

e58..    x14 =L= 4.417893;

e59..    x15 =L= 4.625;

e60..    x16 =L= 3.710786;

e61.. 10*x2*x1 - 18*sqr(x1) - 14*sqr(x2) - 18*sqr(x9) + 10*x10*x9 - 14*sqr(x10)
       + 4*x3*x1 + 6*x3*x2 - 10*sqr(x3) + 4*x11*x9 + 6*x11*x10 - 10*sqr(x11) + 
      8*x4*x1 - 23*sqr(x4) + 8*x12*x9 - 23*sqr(x12) + 2*x5*x1 + 4*x5*x2 + 10*x5
      *x4 - 18*sqr(x5) + 2*x13*x9 + 4*x13*x10 + 10*x13*x12 - 18*sqr(x13) + 4*x6
      *x2 + 4*x6*x4 + 20*x6*x5 - 20*sqr(x6) + 4*x14*x10 + 4*x14*x12 + 20*x14*
      x13 - 20*sqr(x14) + 12*x8*x1 + 10*x8*x3 + 20*x8*x4 + 2*x8*x6 - 32*sqr(x8)
       + 12*x16*x9 + 10*x16*x11 + 20*x16*x12 + 2*x16*x14 - 32*sqr(x16) + 4*x7*
      x2 + 4*x7*x4 + 10*x7*x6 + 20*x7*x8 - 19*sqr(x7) + 4*x15*x10 + 4*x15*x12
       + 10*x15*x14 + 20*x15*x16 - 19*sqr(x15) + objvar =E= 0;

* set non-default bounds
x1.lo = -18.999999; x1.up = 18.999999;
x2.lo = -18.999999; x2.up = 18.999999;
x3.lo = -18.999999; x3.up = 18.999999;
x4.lo = -18.999999; x4.up = 18.999999;
x5.lo = -18.999999; x5.up = 18.999999;
x6.lo = -18.999999; x6.up = 18.999999;
x7.lo = -18.999999; x7.up = 18.999999;
x8.lo = -18.999999; x8.up = 18.999999;
x9.lo = -18.999999; x9.up = 18.999999;
x10.lo = -18.999999; x10.up = 18.999999;
x11.lo = -18.999999; x11.up = 18.999999;
x12.lo = -18.999999; x12.up = 18.999999;
x13.lo = -18.999999; x13.up = 18.999999;
x14.lo = -18.999999; x14.up = 18.999999;
x15.lo = -18.999999; x15.up = 18.999999;
x16.lo = -18.999999; x16.up = 18.999999;

Model m / all /;

m.limrow=0; m.limcol=0;
m.tolproj=0.0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set NLP $set NLP NLP
Solve m using %NLP% minimizing objvar;
