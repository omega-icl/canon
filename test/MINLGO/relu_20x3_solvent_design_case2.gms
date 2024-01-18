SETS

s                   'candidate solvents'
/water, n-pentane, n-heptane, ethanol, 1-propanol, 1-butanol, 1-pentanol, acetone/
i                   'components in the mixture'
/ibuprofen, s1, s2/

ii(i)               'selected solvents in the mixture'
/s1,s2/

jj(i)               'solid solute in the mixture'
/ibuprofen/

k                   'groups'
/CH3, CH2, CH, aCH, aCCH2, aCCH, CH3CO, COOH, OH, H2O/

st                  'heated(initial) state or cooled(final) state'
/heated, cooled/

prp                 'melting/boiling T in K (p1, p2); molar mass in g/mol (p3); density in g/mL (p4)'
/p1*p4/

pr                  'UNIFAC group contributions'
/Q,R/

* ic                  'Integer cuts'
* /1*5/

;

alias(k,m);
alias(s,ss);
*alias(ic,ics);

*Parameter
* yv(ii,s,ic)
* yieldk(ic,jj)
* xi_sk(ic)
* v_sk(ic)

*Tk(ic,st)
*yk(ic,ii,s)
*xk(ic,i,st)
*nmk(ic,i,st)
*nmtk(ic,st)
*;

Table GS(k, pr)     'Group specifications (R = volume, Q = surface area)'
$onDelim
,Q,R
CH3,0.848,0.9011
CH2,0.54,0.6744
CH,0.228,0.4469
aCH,0.400,0.5313
aCCH2,0.660,1.0396
aCCH,0.348,0.8121
CH3CO,1.488,1.6724
COOH,1.224,1.3013
OH,1.2,1
H2O,1.4,0.92
$offDelim
;

Table v(s, k)       'number of groups in molecule i'
$ondelim
,CH3,CH2,CH,aCH,aCCH2,aCCH,CH3CO,COOH,OH,H2O
water,0,0,0,0,0,0,0,0,0,1
n-pentane,2,3,0,0,0,0,0,0,0,0
n-heptane,2,5,0,0,0,0,0,0,0,0
ethanol,1,1,0,0,0,0,0,0,1,0
1-propanol,1,2,0,0,0,0,0,0,1,0
1-butanol,1,3,0,0,0,0,0,0,1,0
1-pentanol,1,4,0,0,0,0,0,0,1,0
acetone,1,0,0,0,0,0,1,0,0,0
$offdelim
;

Table vib(jj,k)     'identity of ibuprofen'
$onDelim
,CH3,CH2,CH,aCH,aCCH2,aCCH,CH3CO,COOH,OH,H2O
ibuprofen,3,0,1,4,1,1,0,1,0,0
$offDelim
;

Table a(k,m)        'group interaction parameters'
$onDelim
,CH3,CH2,CH,aCH,aCCH2,aCCH,CH3CO,COOH,OH,H2O
CH3,0,0,0,61.13,76.5,76.5,476.4,663.5,986.5,1318
CH2,0,0,0,61.13,76.5,76.5,476.4,663.5,986.5,1318
CH,0,0,0,61.13,76.5,76.5,476.4,663.5,986.5,1318
aCH,-11.12,-11.12,-11.12,0,167,167,25.77,537.4,636.1,903.8
aCCH2,-69.7,-69.7,-69.7,-146.8,0,0,-52.1,872.3,803.2,5695
aCCH,-69.7,-69.7,-69.7,-146.8,0,0,-52.1,872.3,803.2,5695
CH3CO,26.76,26.76,26.76,140.1,365.8,365.8,0,669.4,164.5,472.5
COOH,315.3,315.3,315.3,62.32,89.86,89.86,-297.8,0,-151,-66.17
OH,156.4,156.4,156.4,89.6,25.82,25.82,84,199,0,353.5
H2O,300,300,300,362.3,377.6,377.6,-195.4,-14.09,-229.1,0
$offDelim
;

Table prop(s,prp)   'All relevant properties'
$onDelim
,p1,p2,p3,p4
water,373.15,273.15,18.015,0.998
n-pentane,309,143,72.151,0.626
n-heptane,371.53,182.6,100.205,0.684
ethanol,351.39,159.01,46.069,0.789
1-propanol,370,147,60.096,0.804
1-butanol,390.8,183.3,74.123,0.81
1-pentanol,411,194.65,88.1482,0.811
acetone,329.3,178.7,58.08,0.791
$offDelim
;

Parameters
Hm(jj)              'Enthalpy of fusion in J/mol'
/
ibuprofen  25500
/

Tm(jj)              'Melting point in K'
/
ibuprofen  347.15
/

Mwib(jj)              'Molar mass of solids in g/mol'
/
ibuprofen  206.28
/

Rg                  'gas constant'
/8.3144/ ;

Parameter nib(jj, k);
nib(jj,k) = vib(jj,k);

Parameter qib(jj);
qib(jj) = sum(k, nib(jj,k)*GS(k,'Q'));

Parameter rib(jj);
rib(jj) = sum(k, nib(jj,k)*GS(k,'R'));

Parameter qs(s);
qs(s) = sum(k, v(s,k)*GS(k,'Q'));

Parameter rs(s);
rs(s) = sum(k, v(s,k)*GS(k,'R'));

Parameter eib(jj,k);
eib(jj,k) = vib(jj,k)*GS(k,'Q')/qib(jj);

Scalar  Tos         'Minimum temperature offset'
/10/
;


Positive Variables

yield(jj)           'Crystal yield of ibuprofen'
Xmas                'Solvent use in terms of mass'
Vvol                 'Solvent use in terms of volume'
x(i,st)             'Liquid phase mole fraction of component i'
rc(i)               'van der Waals volume of component i'
qc(i)               'van der Waals area of component i'
J(i,st)             'UNIFAC intermediate variable to estimate solubility'
L(i,st)             'UNIFAC intermediate variable to estimate solubility'
T(st)               'Process temperature'
Mw(i)               'Molecular weight of component i'
den(ii)             'Density of selected solvents ii in g/mL'
vol(ii)             'Volume of selected solvents ii in mL at cooling state'
mas(i,st)          'Mass in g of each component'

th(k,st)            'UNIFAC intermediate to estimate solubility'
w(k,st)             'UNIFAC intermediate to estimate solubility'

molt(st)             'Total mass at state st'
mol(i,st)            'Mass of each component at each state'
;

Free Variables
z                   'Objective function'
ps(k,m,st)          'UNIFAC intermediate'
bib(i,k,st)         'UNIFAC intermediate'

lng(i,st)           'Natural log of activity coefficient of ibuprofen'
lngc(i,st)          'Natural log of combinatorial activity coefficient of ibuprofen'
lngr(i,st)          'Natural log of residual activity coefficient of ibuprofen'
;

Binary Variables
y(ii,s)             'Selected solvents'
;

integer Variables
n(i,k)
;

****** set bounds for all the variables ******
x.up(i,st)=1;
x.lo(jj,st)=1e-10;
x.lo('s1',st)=1e-8;
x.lo('s2',st)=0;

Xmas.lo=3.5;
Xmas.up = 200;

Vvol.lo = 4;
Vvol.up=500;

yield.lo(jj)=0.5;
yield.up(jj)=1;

T.up(st) =318.15;
T.lo(st) = 293.15;

mas.lo(jj,st)=1e-5;
mas.lo('s1',st)=1e-5;
mas.lo('s2',st)=0;
mas.up(i,st)=1000;

den.lo('s1')=0.5;
den.lo('s2')=0;
den.up(ii)=2;
vol.lo('s1')=1e-7;
vol.lo('s2')=0;
vol.up(ii)=2000;

*Bounds on other variables
*variable bounds
rc.lo(ii)= 0.000001;
rc.up(i)= 10;
qc.lo(ii)= 0.000001;
qc.up(i)= 10;

J.lo(jj,st)=0.000001;
J.up(jj,st)=10;
L.lo(jj,st)=0.000001;
L.up(jj,st)=10;

bib.up(i,k,st) = 5;
bib.lo(i,k,st) = -5;
th.up(k,st)=1;
th.lo(k,st)=0;
w.up(k,st)=3;
w.lo(k,st)=0;

lngc.lo(jj,st)=-30;
lngc.up(jj,st)=30;
lngr.lo(jj,st)=-30;
lngr.up(jj,st)=30;
lng.lo(jj,st) = -30;
lng.up(jj,st) = 30;

ps.lo(k,m,st) = 0;
ps.up(k,m,st) = 5;

mol.up(i,st) = 34;
molt.up(st) = 45;

Mw.up(ii) = 110;
Mw.up(jj) = Mwib(jj);

****** define the initial points for the problem ******
qc.fx(jj)=qib(jj);
rc.fx(jj)=rib(jj);


*fixing initial amount of ibuprofen to be crystallised
mol.fx('ibuprofen','heated')=1;

*initilaization  of mole fraction
x.l('ibuprofen','heated')=0.13892;
x.l('ibuprofen','cooled')=3.67181e-5;


x.l('s2','heated')=0.342767;
x.l('s2','cooled')=0.899967;

x.l('s1',st)=1-x.l('ibuprofen',st)-x.l('s2',st);

molt.l('heated')=7.19837;
molt.l('cooled')=37.3115;
mol.l(i,st)=x.l(i,st)*molt.l(st);

Yield.l(jj)=1-mol.l(jj,'cooled')/mol.l(jj,'heated');

y.fx('s1','ethanol')=1;
y.fx('s2','water')=1;

T.fx('heated')=312.9034;
T.fx('cooled')=293.15;
n.l(ii,k)=sum(s,v(s,k)*y.l(ii,s));
Mw.l(ii)=sum(s,prop(s,'p3')*y.l(ii,s));
mas.l(i,st)=mol.l(i,st)*Mw.l(i);
den.l(ii)=sum(s,prop(s,'p4')*y.l(ii,s));
vol.l(ii)= mas.l(ii,'cooled')/den.l(ii);

rc.l(ii)= sum(s,rs(s)*y.l(ii,s));
qc.l(ii)= sum(s,qs(s)*y.l(ii,s));

J.l(jj,st)=rib(jj)/sum(i,x.l(i,st)*rc.l(i));
L.l(jj,st)=qib(jj)/sum(i,x.l(i,st)*qc.l(i));

lngc.l(jj,st)=1-J.l(jj,st)+log(J.l(jj,st))-5*qib(jj)*(1-J.l(jj,st)/L.l(jj,st)+log(J.l(jj,st))-log(L.l(jj,st)));

ps.l(k,m,st)=exp(-a(k,m)/T.l(st));
bib.l(jj,k,st) = sum(m,eib(jj,m)*ps.l(m,k,st));
th.l(k,st)=sum(i,x.l(i,st)*GS(k,'Q')*n.l(i,k))/sum(i,x.l(i,st)*qc.l(i));
w.l(k,st)=sum(m,th.l(m,st)*ps.l(m,k,st));

lngr.l(jj,st)=qib(jj)*(1-sum(k,th.l(k,st)*bib.l(jj,k,st)/w.l(k,st)-eib(jj,k)*(log(bib.l(jj,k,st))-log(w.l(k,st)))));

lng.l(jj,st) = lngc.l(jj,st) + lngr.l(jj,st);


EQUATIONS
obj
eq_logic1,eq_logic2,eq_logic3
eq_unifac1,eq_unifac2,eq_unifac3,eq_unifac4,eq_unifac5
eq_unifac6,eq_unifac7,eq_unifac8,eq_unifac9,eq_unifac10
eq_unifac11,eq_unifac12,eq_unifac13,eq_unifac14,eq_unifac15,eq_unifac16
eq_pro1,eq_pro2,eq_pro3,eq_pro4,eq_pro5
eq_pro6,eq_pro7,eq_pro8,eq_pro9,eq_pro10
eq_pro11,eq_pro12,eq_pro13,eq_pro14,eq_pro15
eq_pro16,eq_pro17,eq_pro18
;

obj..
z =e= yield('ibuprofen');

****** solvent selection ******
eq_logic1(ii)..
sum(s,y(ii,s)) =e= 1;

eq_logic2(s)..
sum(ii,y(ii,s)) =l= 1;

eq_logic3(st)..
sum(i,x(i,st)) =e= 1;

****** UNIFAC model to calculate the solubility ******

*** identity of the selected solvents
eq_unifac1(ii)..
qc(ii) =e= sum(s,qs(s) * y(ii,s));

eq_unifac2(ii)..                            
rc(ii) =e= sum(s,rs(s) * y(ii,s));

eq_unifac3(jj)..                            
qc(jj) =e= qib(jj);

eq_unifac4(jj)..                            
rc(jj) =e= rib(jj);

****** activity coefficient calculation ******
eq_unifac5(jj,st)..                           
J(jj,st) * sum(i,x(i,st)*rc(i)) =e= rib(jj);

eq_unifac6(jj,st)..                           
L(jj,st) * sum(i,x(i,st)*qc(i)) =e= qib(jj);

eq_unifac7(jj,st)..                        
lngc(jj,st) =e= 1 - J(jj,st) + log(J(jj,st)) - 5*qib(jj)*(1 - J(jj,st)/L(jj,st) + log(J(jj,st)) - log(L(jj,st)));

eq_unifac8(k,m,st)..                         
ps(k,m,st) =e= exp(-a(k,m)/T(st));

eq_unifac9(jj,k,st)..                       
bib(jj,k,st) =e= sum(m,eib(jj,m)*ps(m,k,st));

eq_unifac10(k,st)..                           
th(k,st)*sum(i,x(i,st)*qc(i)) =e= sum(i,x(i,st)*GS(k,'Q')*n(i,k));

eq_unifac11(k,st)..                            
w(k,st) =e= sum(m,th(m,st)*ps(m,k,st));

eq_unifac12(jj,st)..                        
lngr(jj,st) =e= qib(jj)*(1-sum(k,th(k,st)*bib(jj,k,st)/w(k,st)-eib(jj,k)*(log(bib(jj,k,st)) - log(w(k,st)))));

* Overall activity coefficient
eq_unifac13(jj,st)..                         
lng(jj,st) =e= lngc(jj,st) + lngr(jj,st);

* Solubility oonstraint
eq_unifac14(st)..                         
log(x('ibuprofen',st)) + lng('ibuprofen',st) =e= Hm('ibuprofen')/Rg * (1/Tm('ibuprofen') - 1/T(st));

eq_unifac15(ii,k)..
n(ii,k) =e= sum(s, v(s,k)*y(ii,s));

eq_unifac16(jj,k)..
n(jj,k) =e= vib(jj,k);


****** Process operation constraints ******
eq_pro1(jj)..
yield(jj) =e= 1 - (mol(jj,'cooled') / mol(jj,'heated'));

* Constraints on the mol of each component 
eq_pro2(st)..                            
molt(st) =e= sum(i, mol(i,st));

eq_pro3(i,st)..                           
x(i,st) =e= mol(i,st)/molt(st);

eq_pro4..                                 
mol('s1','heated') =e= mol('s1','cooled');

eq_pro5..                                 
mol('s2','heated') =l= mol('s2','cooled');

eq_pro6(jj)..                            
mol(jj,'heated') =g= mol(jj,'cooled');

*  Tempeature constraints 
eq_pro7(ii,s)..                         
y(ii,s)*prop(s,'p2') =l= T('cooled') - Tos;

eq_pro8(ii,s)..                         
prop(s,'p1') =g= (y(ii,s) * (T('heated') + Tos));

eq_pro9(s)..                              
T('heated') =l= prop(s,'p1')-Tos + 600*(1 - sum(ii, y(ii,s)));

eq_pro10(s)..                              
T('cooled') =g= prop(s,'p2')+Tos - 600*(1 - sum(ii, y(ii,s)));

eq_pro11(s)..                               
T('heated') =g= T('cooled');

*  Mass and desentity calculation 
eq_pro12(ii)..                             
Mw(ii) =e= sum(s, prop(s, 'p3') * y(ii,s));

eq_pro13(jj)..                           
Mw(jj) =e= Mwib(jj);

eq_pro14(i,st)..                         
mas(i,st) =e= mol(i,st) * Mw(i);

eq_pro15(ii)..                            
den(ii) =e= sum(s, prop(s,'p4')*y(ii,s));

eq_pro16(ii)..                            
vol(ii) * den(ii) =e= mas(ii,'cooled');

* eq_pro17(ii,k)..                           
* n(ii,k) =e= sum(s, v(s,k)*y(ii,s));

* eq_pro18(jj,k)..                           
* n(jj,k) =e= vib(jj,k);

eq_pro17..                             
Xmas * (mas('ibuprofen','heated') - mas('ibuprofen','cooled')) =e= mas('s1','cooled') + mas('s2','cooled');

eq_pro18..                          
Vvol * (mas('ibuprofen','heated') - mas('ibuprofen','cooled')) =e= sum(ii,vol(ii));


$macro relu(x) (x/2 + abs(x)/2)

SETS
nf 'total feature set'
/qa,qb,ra,rb,za,zb,T/

lay1
/l1n1*l1n20/

lay2
/l2n1*l2n20/

lay3
/l3n1*l3n20/

lay4
/l4n1/

;

PARAMETERS
scaler_mean(nf)
/
qa  3.13001151   
qb  3.12929366   
ra  3.35533374   
rb  3.35424408
za  0.33362441   
zb  0.3337956   
T   305.6470905
/

scaler_var(nf)
/
qa  0.93590254 
qb  0.93614607 
ra  1.252925 
rb  1.25347394
za  0.17963123
zb  0.17959096 
T   7.21687419
/

w_l1(nf,lay1)
w_l2(lay1,lay2)
w_l3(lay2,lay3)
w_l4(lay3,lay4)

b_l1(lay1)
b_l2(lay2)
b_l3(lay3)
b_l4(lay4)

;

$GDXIN relu_20x3_solvent_design_case2.gdx
$load w_l1 w_l2 w_l3 w_l4 b_l1 b_l2 b_l3 b_l4
$GDXIN


VARIABLES
inps(nf,st)    'calculate the scaled input features'
nn_sum_l1(lay1,st)
nn_sum_l2(lay2,st)
nn_sum_l3(lay3,st)
nn_sum_l4(lay4,st)
nn_act_l1(lay1,st)
nn_act_l2(lay2,st)
nn_act_l3(lay3,st)
nn_act_l4(lay4,st)
;

****** define bounds for these variables ******
inps.lo(nf,st) = -5;
inps.up(nf,st) = 5;

nn_sum_l1.lo(lay1,st) = -200;
nn_sum_l2.lo(lay2,st) = -200;
nn_sum_l3.lo(lay3,st) = -200;
nn_sum_l4.lo(lay4,st) = -200;
nn_act_l1.lo(lay1,st) = -200;
nn_act_l2.lo(lay2,st) = -200;
nn_act_l3.lo(lay3,st) = -200;
nn_act_l4.lo(lay4,st) = -200;

nn_sum_l1.up(lay1,st) = 200;
nn_sum_l2.up(lay2,st) = 200;
nn_sum_l3.up(lay3,st) = 200;
nn_sum_l4.up(lay4,st) = 200;
nn_act_l1.up(lay1,st) = 200;
nn_act_l2.up(lay2,st) = 200;
nn_act_l3.up(lay3,st) = 200;
nn_act_l4.up(lay4,st) = 200;

****** define the initial points for these variables ******
inps.l('qa',st) = 1/scaler_var('qa')*(qc.l('s1') - scaler_mean('qa'));
inps.l('qb',st) = 1/scaler_var('qb')*(qc.l('s2') - scaler_mean('qb'));
inps.l('ra',st) = 1/scaler_var('ra')*(rc.l('s1') - scaler_mean('ra'));
inps.l('rb',st) = 1/scaler_var('rb')*(rc.l('s2') - scaler_mean('rb'));
inps.l('za',st) = 1/scaler_var('za')*(x.l('s1',st) - scaler_mean('za'));
inps.l('zb',st) = 1/scaler_var('zb')*(x.l('s2',st) - scaler_mean('zb'));
inps.l('T',st) = 1/scaler_var('T')*(T.l(st) - scaler_mean('T'));


nn_sum_l1.l(lay1,st) = sum(nf,w_l1(nf,lay1)*inps.l(nf,st)) + b_l1(lay1);
nn_act_l1.l(lay1,st) = relu(nn_sum_l1.l(lay1,st));
nn_sum_l2.l(lay2,st) = sum(lay1,w_l2(lay1,lay2)*nn_act_l1.l(lay1,st)) + b_l2(lay2);
nn_act_l2.l(lay2,st) = relu(nn_sum_l2.l(lay2,st));
nn_sum_l3.l(lay3,st) = sum(lay2,w_l3(lay2,lay3)*nn_act_l2.l(lay2,st)) + b_l3(lay3);
nn_act_l3.l(lay3,st) = relu(nn_sum_l3.l(lay3,st));
nn_sum_l4.l(lay4,st) = sum(lay3,w_l4(lay3,lay4)*nn_act_l3.l(lay3,st)) + b_l4(lay4);
nn_act_l4.l(lay4,st) = 1/(1+exp(-nn_sum_l4.l(lay4,st)));


EQUATIONS
eq_sur1,eq_sur2,eq_sur3,eq_sur4,eq_sur5
eq_sur6,eq_sur7,eq_sur8,eq_sur9,eq_sur10
eq_sur11,eq_sur12,eq_sur13,eq_sur14,eq_sur15
eq_sur16
;

* ann max operator in eq_sur8 and sigmoid function in eq_sur11

****** surrogate model for stability check ******

* prepare the input feature 
eq_sur1(st)..
inps('qa',st) =e= 1/scaler_var('qa')*(qc('s1') - scaler_mean('qa'));

eq_sur2(st)..
inps('qb',st) =e= 1/scaler_var('qb')*(qc('s2') - scaler_mean('qb'));

eq_sur3(st)..
inps('ra',st) =e= 1/scaler_var('ra')*(rc('s1') - scaler_mean('ra'));

eq_sur4(st)..
inps('rb',st) =e= 1/scaler_var('rb')*(rc('s2') - scaler_mean('rb'));

eq_sur5(st)..
inps('za',st) =e= 1/scaler_var('za')*(x('s1',st) - scaler_mean('za'));

eq_sur6(st)..
inps('zb',st) =e= 1/scaler_var('zb')*(x('s2',st) - scaler_mean('zb'));

eq_sur7(st)..
inps('T',st) =e= 1/scaler_var('T')*(T(st) - scaler_mean('T'));


* relu ann calculation 
eq_sur8(lay1,st)..
nn_sum_l1(lay1,st) =e= sum(nf,w_l1(nf,lay1)*inps(nf,st)) + b_l1(lay1);

eq_sur9(lay1,st)..
nn_act_l1(lay1,st) =e= relu(nn_sum_l1(lay1,st));

eq_sur10(lay2,st)..
nn_sum_l2(lay2,st) =e= sum(lay1,w_l2(lay1,lay2)*nn_act_l1(lay1,st)) + b_l2(lay2);

eq_sur11(lay2,st)..
nn_act_l2(lay2,st) =e= relu(nn_sum_l2(lay2,st));

eq_sur12(lay3,st)..
nn_sum_l3(lay3,st) =e= sum(lay2,w_l3(lay2,lay3)*nn_act_l2(lay2,st)) + b_l3(lay3);

eq_sur13(lay3,st)..
nn_act_l3(lay3,st) =e= relu(nn_sum_l3(lay3,st));

eq_sur14(lay4,st)..
nn_sum_l4(lay4,st) =e= sum(lay3,w_l4(lay3,lay4)*nn_act_l3(lay3,st)) + b_l4(lay4);

eq_sur15(lay4,st)..
nn_act_l4(lay4,st) =e= 1/(1+exp(-nn_sum_l4(lay4,st)));

eq_sur16(lay4,st)..
nn_act_l4(lay4,st) =g= 0.5;

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

model ANN_model /all/;
*option nlp=conopt3;
*option mip=cplex;
*option rminlp=conopt3;
*option minlp=SCIP;
*option threads=4;

option decimals=5;
OPTION OPTCA = 1e-10;
option iterlim = 1000000;
option optcr  = 0.0001;
OPTION reslim = 10800;

solve ANN_model maximizing z using MINLP;

