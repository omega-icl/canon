SETS

s 'solvents'
/methanol, ethanol, 2-propanol, acetone, mibk, ethylacetate,
chloroform, toluene, water/

i 'components in mixture'
/ibuprofen, s1, s2/

ii(i) 'selected solvent'
/s1, s2/

k 'groups'
/ch3, ch2, ch, ach, acch3, acch2, acch, oh, ch3oh, h2o, ch3co, ch3coo, cooh, chcl3/

;

* c 'integer cuts'
* /1*1/

* dyn(c) 'dynamic set of c'
;


alias(k,m);
alias(k,p);
alias(s,ss);

PARAMETERS
Hm 'enthalpy of fusion of ibuprofen' /25500/
Tm 'melting point of ibuprofen' /347.15/
T 'system temperature' /300/
Rg 'gas constant' /8.3144/
temp 'NN calibration parameter' /1/


R(k) 'Van der Waals volume of group k'
/ch3             0.9011
ch2              0.6744
ch               0.4469
ach              0.5313
acch3            1.2663
acch2            1.0396
acch             0.8121
oh               1.0000
ch3oh            1.4311
h2o              0.9200
ch3co            1.6724
ch3coo           1.9031
cooh             1.3013
chcl3            2.8700 /

Q(k) 'Van der Waals surface area of group k'
/ch3             0.848
ch2              0.540
ch               0.228
ach              0.400
acch3            0.968
acch2            0.660
acch             0.348
oh               1.200
ch3oh            1.432
h2o              1.400
ch3co            1.488
ch3coo           1.728
cooh             1.224
chcl3            2.410 /;

table v(s,k) 'number of groups in molecule i'
                 ch3     ch2     ch      ach     acch3   acch2   acch    oh      ch3oh   h2o     ch3co   ch3coo  cooh    chcl3

methanol                                                                         1
ethanol          1       1                                               1
2-propanol       2               1                                       1
acetone          1                                                                               1
mibk             2       1       1                                                               1
ethylacetate     1       1                                                                               1
chloroform                                                                                                               1
toluene                                  5       1
water                                                                                    1
;

table vib(i,k)'identity of ibuprofen'
             ch3     ch ach      acch2 acch   cooh
ibuprofen     3       1  4         1    1       1      ;


                                                                                                                          ;
*===============================================================================
*group interaction parameters
*===============================================================================
table a(k,m) 'group interaction parameters'
         ch3     ch2     ch      ach     acch3   acch2   acch    oh      ch3oh   h2o     ch3co   ch3coo  cooh    chcl3
ch3      0       0       0       61.13   76.5    76.5    76.5    986.5   697.2   1318    476.4   232.1   663.5   24.9
ch2      0       0       0       61.13   76.5    76.5    76.5    986.5   697.2   1318    476.4   232.1   663.5   24.9
ch       0       0       0       61.13   76.5    76.5    76.5    986.5   697.2   1318    476.4   232.1   663.5   24.9
ach      -11.12  -11.12  -11.12  0       167     167     167     636.1   637.4   903.8   25.77   5.994   537.4   -231.9
acch3    -69.7   -69.7   -69.7   -146.8  0       0       0       803.2   603.3   5695    -52.1   5688    872.3   -80.25
acch2    -69.7   -69.7   -69.7   -146.8  0       0       0       803.2   603.3   5695    -52.1   5688    872.3   -80.25
acch     -69.7   -69.7   -69.7   -146.8  0       0       0       803.2   603.3   5695    -52.1   5688    872.3   -80.25
oh       156.4   156.4   156.4   89.6    25.82   25.82   25.82   0       -137.1  353.5   84      101.1   199     -98.12
ch3oh    16.51   16.51   16.51   -50     -44.5   -44.5   -44.5   249.1   0       -181    23.39   -10.72  -202    -139.4
h2o      300     300     300     362.3   377.6   377.6   377.6   -229.1  289.6   0       -195.4  72.87   -14.09  353.7
ch3co    26.76   26.76   26.76   140.1   365.8   365.8   365.8   164.5   108.7   472.5   0       -213.7  669.4   -354.6
ch3coo   114.8   114.8   114.8   85.84   -170    -170    -170    245.4   249.6   200.8   372.2   0       660.2   -209.7
cooh     315.3   315.3   315.3   62.32   89.86   89.86   89.86   -151    339.8   -66.17  -297.8  -256.3  0       39.63
chcl3    36.7    36.7    36.7    288.5   69.9    69.9    69.9    742.1   649.1   826.8   552.1   176.5   504.2   0
                                                                                                                    ;

parameter ps(k,m);
ps(k,m)= exp(-a(k,m)/T);
parameter nib(i,k);
nib('ibuprofen',k)= vib('ibuprofen',k);
parameter qib;
qib = sum(k,nib('ibuprofen',k)*Q(k));
parameter rib;
rib = sum(k,nib('ibuprofen',k)*R(k));
parameter qs(s);
qs(s) = sum(k,v(s,k)*Q(k));
parameter rs(s);
rs(s) = sum(k,v(s,k)*R(k));
parameter eib(i,k);
eib('ibuprofen',k)=vib('ibuprofen',k)*Q(k)/qib;
parameter bib(i,k);
bib('ibuprofen',k)=sum(m,eib('ibuprofen',m)*ps(m,k));

* parameter yv(ii,s,c) 'store y values from previous iterations';
* parameter zv(c) 'store objective values from previous iterations';
* parameter xv(i,c) 'store mole fractions from previous iterations';



POSITIVE VARIABLES

x(i)             'liquid phase mole fraction of component i'
rc(i)            'van der waals volume of component i'
qc(i)            'van der waals area of component i'
J(i),L(i)       

th(k)
w(k)


;
FREE VARIABLES
z            'objective function'

lng(i)       'natural log of activity coefficient of ibuprofen'
lngc(i)      'natural log of combinatorial activity coefficient of ibuprofen'
lngr(i)      'natural log of residual activity coefficient of ibuprofen'

;


BINARY VARIABLES
y(ii,s)     'selected solvents'

INTEGER VARIABLE
n(i,k)    'number of groups k in component i' ;


*******************************BOUNDS*******************************************
*bounds on optmization variables
x.up(i)=1;
x.lo(i)=0.01;

n.up(ii,k)=6;

*variable bounds
rc.lo(i)= 0.1;
rc.up(i)= 10;
qc.lo(i)= 0.1;
qc.up(i)= 10;

J.lo('ibuprofen')=0.1;
J.up('ibuprofen')=10;
L.lo('ibuprofen')=0.1;
L.up('ibuprofen')=10;

th.up(k)=1;
th.lo(k)=0;
w.up(k)=3;
w.lo(k)=1e-100;

lngc.lo('ibuprofen')=-30;
lngc.up('ibuprofen')=30;
lngr.lo('ibuprofen')=-30;
lngr.up('ibuprofen')=30;
lng.lo('ibuprofen') = -30;
lng.up('ibuprofen') = 30;


*********************INITIAL POINTS*********************************************

*fixing of ibuprofen identity
qc.fx('ibuprofen')=qib;
rc.fx('ibuprofen')=rib;
n.fx('ibuprofen',k)=vib('ibuprofen',k);

*initilaization  of optimal solvent
x.l('ibuprofen')=0.31623;
x.l('s2')=0.01;
x.l('s1')=1-x.l('ibuprofen')-x.l('s2');

y.l('s1','methanol')=1;
y.l('s2','chloroform')=1;

z.l=x.l('ibuprofen');

n.l(ii,k)=sum(s,v(s,k)*y.l(ii,s));

rc.l(ii)= sum(s,rs(s)*y.l(ii,s));
qc.l(ii)= sum(s,qs(s)*y.l(ii,s));

J.l('ibuprofen')=rib/sum(i,x.l(i)*rc.l(i));
L.l('ibuprofen')=qib/sum(i,x.l(i)*rc.l(i));

lngc.l('ibuprofen')=1-J.l('ibuprofen')+log(J.l('ibuprofen'))-5*qib*(1-J.l('ibuprofen')/L.l('ibuprofen')+log(J.l('ibuprofen'))-log(L.l('ibuprofen')));

th.l(k)=sum(i,x.l(i)*Q(k)*n.l(i,k))/sum(i,x.l(i)*qc.l(i));
w.l(k)=sum(m,th.l(m)*ps(m,k));

lngr.l('ibuprofen')=qib*(1-sum(k,th.l(k)*bib('ibuprofen',k)/w.l(k)-eib('ibuprofen',k)*(log(bib('ibuprofen',k))-log(w.l(k)))));

lng.l('ibuprofen') = lngc.l('ibuprofen') + lngr.l('ibuprofen');






EQUATIONS
eq_obj
eq_logic1,eq_logic2,eq_logic3,eq_logic4
eq_unifac1,eq_unifac2,eq_unifac3,eq_unifac4,eq_unifac5
eq_unifac6,eq_unifac7,eq_unifac8,eq_unifac9,eq_unifac10,eq_unifac11

;
* nonlinear formulations in eq_unifac4,eq_unifac5,eq_unifac6,eq_unifac7,eq_unifac9,eq_unifac11


****** define objective by maximizing the solubility of ibuprofen ******
eq_obj.. 
z =e= x('ibuprofen');

****** define the logical constraints on the mixture system ******
eq_logic1(ii)..
sum(s,y(ii,s)) =e= 1;

eq_logic2(s)..
sum(ii,y(ii,s)) =l= 1;

eq_logic3(s,ss)$(ord(ss) <= (ord(s)))..
y('s1',s) + y('s2',ss) =l= 1;

eq_logic4..
sum(i,x(i)) =e= 1;

****** define the UNIFAC model for this mixture ******
eq_unifac1(ii,k)..
n(ii,k) =e= sum(s,v(s,k)*y(ii,s));

eq_unifac2(ii)..
qc(ii) =e= sum(s,qs(s)*y(ii,s));

eq_unifac3(ii)..
rc(ii) =e= sum(s,rs(s)*y(ii,s));

* combinatorial part of activity coefficient

* trilinear terms L*x*qc and L*x*rc in eq_unifac4 and eq_unifac5
eq_unifac4..
J('ibuprofen')*sum(i,x(i)*rc(i)) =e= rib;

eq_unifac5..
L('ibuprofen')*sum(i,x(i)*qc(i)) =e= qib;

* log function, bilinear terms J*log, division J/L in eq_unifac6
eq_unifac6..
lngc('ibuprofen') =e= 1-J('ibuprofen')+log(J('ibuprofen'))-5*qib*(1-J('ibuprofen')/L('ibuprofen')+log(J('ibuprofen'))-log(L('ibuprofen')));

*residual part of activity coefficient

* trilinear term th*x*qc, bilinear term continuous var * interger var: x*n
eq_unifac7(k)..
th(k)*sum(i,x(i)*qc(i)) =e= sum(i,x(i)*Q(k)*n(i,k));

eq_unifac8(k)..
w(k) =e= sum(m,th(m)*ps(m,k));

* division th/w, log function 
eq_unifac9..
lngr('ibuprofen') =e= qib*(1-sum(k,th(k)*bib('ibuprofen',k)/w(k)-eib('ibuprofen',k)*(log(bib('ibuprofen',k))-log(w(k)))));

*activity coefficient
eq_unifac10..
lng('ibuprofen') =e=  lngc('ibuprofen') + lngr('ibuprofen');

*ibuprofen solubility constraint

* division 1/T
eq_unifac11..
log(x('ibuprofen'))+ lng('ibuprofen') =e= Hm/Rg *(1/Tm-1/T);







SETS
nf 
* 'total feature set'
* /qa,qb,ra,rb,za,zb/

n_l1
* /l1n1*l1n20/

n_l2
* /l2n1*l2n20/

n_l3
* /l3n1*l3n20/

n_l4
* /l4n1/

;

PARAMETERS
scaler_mean(nf)
scaler_var(nf)

w_l1(nf,n_l1)
w_l2(n_l1,n_l2)
w_l3(n_l2,n_l3)
w_l4(n_l3,n_l4)

b_l1(n_l1)
b_l2(n_l2)
b_l3(n_l3)
b_l4(n_l4)
;

* PARAMETERS
* scaler_mean(nf)
* /
* qa  2.58602971
* qb  2.60220338
* ra  2.79284563
* rb  2.90664494
* za  0.33371024
* zb  0.33376051
* /

* scaler_var(nf)
* /
* qa  0.76838993
* qb  0.78661128
* ra  0.94966161
* rb  1.21327817
* za  0.17994648
* zb  0.18011281
* /



* ;

$GDXIN Relu_20_20_20.gdx
$load nf n_l1 n_l2 n_l3 n_l4 w_l1 w_l2 w_l3 w_l4 b_l1 b_l2 b_l3 b_l4 scaler_mean scaler_var
$GDXIN


VARIABLES
inps(nf)    'calculate the scaled input features'
nn_sum_l1(n_l1)
nn_sum_l2(n_l2)
nn_sum_l3(n_l3)
nn_sum_l4(n_l4)

nn_act_l1(n_l1)
nn_act_l2(n_l2)
nn_act_l3(n_l3)
nn_act_l4(n_l4)
;

inps.l('qa') = 1/scaler_var('qa')*(qc.l('s1') - scaler_mean('qa'));
inps.l('qb') = 1/scaler_var('qb')*(qc.l('s2') - scaler_mean('qb'));
inps.l('ra') = 1/scaler_var('ra')*(rc.l('s1') - scaler_mean('ra'));
inps.l('rb') = 1/scaler_var('rb')*(rc.l('s2') - scaler_mean('rb'));
inps.l('za') = 1/scaler_var('za')*(x.l('s1') - scaler_mean('za'));
inps.l('zb') = 1/scaler_var('zb')*(x.l('s2') - scaler_mean('zb'));


$macro relu(x) (x/2 + abs(x)/2)
EQUATIONS
eq_sur1,eq_sur2,eq_sur3,eq_sur4,eq_sur5
eq_sur6,eq_sur7,eq_sur8,eq_sur9,eq_sur10
eq_sur11,eq_sur12,eq_sur13,eq_sur14,eq_sur15
;

* ann max operator in eq_sur8 and sigmoid function in eq_sur11

****** surrogate model for stability check ******

* prepare the input feature 
eq_sur1..
inps('qa') =e= 1/scaler_var('qa')*(qc('s1') - scaler_mean('qa'));

eq_sur2..
inps('qb') =e= 1/scaler_var('qb')*(qc('s2') - scaler_mean('qb'));

eq_sur3..
inps('ra') =e= 1/scaler_var('ra')*(rc('s1') - scaler_mean('ra'));

eq_sur4..
inps('rb') =e= 1/scaler_var('rb')*(rc('s2') - scaler_mean('rb'));

eq_sur5..
inps('za') =e= 1/scaler_var('za')*(x('s1') - scaler_mean('za'));

eq_sur6..
inps('zb') =e= 1/scaler_var('zb')*(x('s2') - scaler_mean('zb'));

* relu ann calculation 
eq_sur7(n_l1)..
nn_sum_l1(n_l1) =e= sum(nf,w_l1(nf,n_l1)*inps(nf)) + b_l1(n_l1);

eq_sur8(n_l1)..
nn_act_l1(n_l1) =e= relu(nn_sum_l1(n_l1));

eq_sur9(n_l2)..
nn_sum_l2(n_l2) =e= sum(n_l1,w_l2(n_l1,n_l2)*nn_act_l1(n_l1)) + b_l2(n_l2);

eq_sur10(n_l2)..
nn_act_l2(n_l2) =e= relu(nn_sum_l2(n_l2));

eq_sur11(n_l3)..
nn_sum_l3(n_l3) =e= sum(n_l2,w_l3(n_l2,n_l3)*nn_act_l2(n_l2)) + b_l3(n_l3);

eq_sur12(n_l3)..
nn_act_l3(n_l3) =e= relu(nn_sum_l3(n_l3));

eq_sur13(n_l4)..
nn_sum_l4(n_l4) =e= sum(n_l3,w_l4(n_l3,n_l4)*nn_act_l3(n_l3)) + b_l4(n_l4);

eq_sur14(n_l4)..
nn_act_l4(n_l4) =e= 1/(1+exp(-nn_sum_l4(n_l4)));

eq_sur15(n_l4)..
nn_act_l4(n_l4) =g= 0.5;

$ontext
* relu formulation with max
eq_sur8(layer)..
nn_act(layer) =e= relu(nn_sum(layer));
* nn_act(layer) =e= max(0,nn_sum(layer));

eq_sur9..
res_sum =e= sum(layer,nn_weights2(layer)*nn_act(layer)) + nn_bias2;

* activation function sigmoid in the output layer
eq_sur10..
res_act =e= 1/(1+exp(-res_sum));

eq_sur11..
res_act =g= 0.5;
$offtext

model surrogate_model /all/;
*option nlp=conopt3;
*option mip=cplex;
*option rminlp=conopt3;
* option minlp=SCIP;
*option threads=4;

option decimals=5;
OPTION OPTCA = 1e-10;
*option iterlim = 1000;
option optcr  = 0.0001;
*OPTION reslim = 10800;
surrogate_model.OPTFILE = 1;

solve surrogate_model maximizing z using MINLP;

