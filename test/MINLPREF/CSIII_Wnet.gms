********************************************************************************
*
* Problem definition for CSIII (Two-Pressure Cycle), max Wnet originally published in
*        D. Bongartz, A. Mitsos: "Deterministic global optimization of process
*             flowsheets in a reduced space using McCormick relaxations",
*             Journal of Global Optimization 69 (2017), 761-796.
*             https://link.springer.com/article/10.1007/s10898-017-0547-4
*        E-mail: amitsos@alum.mit.edu
*
* ==============================================================================
* © 2020, Process Systems Engineering (AVT.SVT), RWTH Aachen University
* ==============================================================================
*
* used in:
*    D. Bongartz: "Deterministic Global Flowsheet Optimization for the Design of Energy Conversion Processes",
*                   PhD Thesis, RWTH Aachen University, 2020.
*
*
********************************************************************************

option nlp = baron;
option optcr = 1e-3;
option reslim = 43200;


sets

                i               streams /1*18/

;


parameter

*Parameters for hot gas stream
                T_G_in                  gas inlet temperature [K]                                                                       /900/
                s_G_in                  heat capacity flow rate of gas [kW|K]                                   /200/


*Cycle parameters
                eta_st                  isentropic turbine efficiency [-]                                               /0.9/
                eta_sp                  isentropic pump efficiency [-]                                                  /0.8/
                T_max                           max. temperature [K]                                                                                    /873.0/
                x_min                           min. vapor quality in turbine [-]                                               /0.85/
                DeltaT_min              min. temperature difference in heat exchangers [K]      /15.0/
                DeltaT_Ap               subcooling at economizer outlet [K]                                             /10.0/

*Parameters for properties
                delta_h_v               enthalpy of vaporization  at T=T0 [kJ|kg]               /2480.0/
                p0                                      reference pressure [bar]                                                        /10E-3/
                T0                                      reference temperature [K]                                                       /313.8316/
                R                                       spec. gas constant  [kJ|kg*K]                                           /0.462/
                c_if                            spec. heat capacity (ideal liquid) [kJ|kg*K]    /4.18/
                cp_ig                           spec. heat capacity (ideal gas) [kJ|kg*K]               /2.08/
                v_if                            spec. volume of liquid water [m^3|kg]                   /0.001/
                A                                       par. Antoine                                                                                    /3.55959/
                B                                       par. Antoine                                                                                    /643.748/
                C                                       par. Antoine                                                                                    /-198.043/

;

free variables

                p_net           net Power output [kW] (objective)

;

positive variables

                T(i)            temperature [K]
                p(i)            pressure [bar]
                x(i)            vapor quality [-]
                h(i)            spec. enthalpy [kJ|kg]
                s(i)            spec. entropy [kJ|K]

                T_sat(i)                                                saturation tempearture
                enthalpie_sat_liq(i)            saturated liquid enthalpy [kJ|kg]
                enthalpie_sat_gas(i)            saturated vapor enthalpy [kJ|kg]
                entropie_sat_liq(i)             saturated liquid entropy [kJ|kg]
                entropie_sat_gas(i)             saturated vapor entrop [kJ|kg]

                m                               mass flow rate [kg|s]
                mBleed          mass flow rate of turbine bleed [kg|s]
                mMain                   mass flow rate of remaining turbine [kg|s]
                mHP                     mass flow rate through HP part [kg|s]
                mLP                     mass flow rate through LP part [kg|s]
                kBl                     fraction of mass flow rate extracted in the turbine [-]
                kLP                     fraction of mass flow rate going to LP part [-]
                TG2                     temperature of gas at G2 [K]
                TG3                     temperature of gas at G3 [K]
                TG4                     temperature of gas at G4 [K]
                TG5                     temperature of gas at G5 [K]
                TG6                     temperature of gas at G6 [K]
                TG7                     temperature of gas at G7 [K]
                Qzu                     heat transfer rate in boiler [kW]

                w_pump2                 spec. pumping work [kJ|kg]
                p_pump2                 pump Power consumption [kW]
                w_pump4                 spec. pumping work [kJ|kg]
                p_pump4                 pump Power consumption [kW]
                w_pump8                 spec. pumping work [kJ|kg]
                p_pump8                 pump Power consumption [kW]
                w_turbine12             spec. turbine work [kJ|kg]
                p_turbine12             Power output turbine [kW]
                w_turbine14             spec. turbine work [kJ|kg]
                p_turbine14             Power output turbine [kW]
                w_turbine15             spec. turbine work [kJ|kg]
                p_turbine15             Power output turbine [kW]

;


equations
*****Equalities
                eq_sat_temperature(i)
                eq_sat_enthalpie_liq(i)
                eq_sat_enthalpie_gas(i)
                eq_sat_entropie_liq(i)
                eq_sat_entropie_gas(i)

                eq_enthalpie_1iq(i)
                eq_enthalpie_gas(i)
                eq_entropie_gas(i)
                eq_enthalpie_2phase
                eq_entropie_2phase

                eq_mLP
                eq_mHP
                eq_mBleed
                eq_mMain

                eq_entropy_turbine12
                eq_pressure_turbine12
                eq_pressure_turbine122
                eq_work_turbine12
                eq_energy_balance_turbine12
                eq_p_turbine12
                eq_isobaric_mixer
                eq_energy_balance_mixer
                eq_entropy_turbine14
                eq_pressure_turbine14
                eq_pressure_turbine142
                eq_work_turbine14
                eq_energy_balance_turbine14
                eq_p_turbine14
                eq_entropy_turbine15
                eq_pressure_turbine15
                eq_pressure_turbine152
                eq_work_turbine15
                eq_energy_balance_turbine15
                eq_p_turbine15

                eq_enthalpy1
                eq_work_pump2
                eq_energy_balance_pump2
                eq_p_pump2

                eq_isobaric_deaerator
                eq_energy_balance_deaerator

                eq_work_pump4
                eq_energy_balance_pump4
                eq_p_pump4

                eq_work_pump8
                eq_energy_balance_pump8
                eq_p_pump8

                eq_Qzu
                eq_isobaric_LPEC
                eq_isobaric_LPEV
                eq_isobaric_LPSH
                eq_isobaric_HPEC
                eq_isobaric_HPEV
                eq_isobaric_HPSH
                eq_energy_balance_LPEC
                eq_energy_balance_LPEV
                eq_energy_balance_LPSH
                eq_energy_balance_HPEC
                eq_energy_balance_HPEV
                eq_energy_balance_HPSH
                eq_enthalpy_6
                eq_temperature_5
                eq_temperature_9
                eq_enthalpy_10

                eq_obj_p_net

                eq_constraint_deaerator

******Inequalities
                ineq_constraint_fully_evap7
                ineq_constraint_fully_evap11
                ineq_constraint_pinch
                ineq_constraint_pinch2
                ineq_constraint_pinch3
                ineq_constraint_pinch4
                ineq_constraint_pinch5
                ineq_constraint_h_16
                ineq_constraint_p42
                ineq_constraint_p84
;

*****Thermodynamics*****
eq_sat_temperature(i)..         T_sat(i) =e= B /(A - log10(p(i))) - C;
eq_sat_enthalpie_liq(i)$((ord(i) eq 1) or (ord(i) eq 3) or (ord(i) eq 14) or (ord(i) eq 15) or (ord(i) eq 17) or (ord(i) eq 18))..              enthalpie_sat_liq(i) =e= c_if * (T_sat(i) - T0) + v_if * 100 * (p(i) - p0);
eq_sat_enthalpie_gas(i)$((ord(i) eq 6) or (ord(i) eq 7) or (ord(i) eq 10) or (ord(i) eq 11) or (ord(i) eq 14) or (ord(i) eq 15) or (ord(i) eq 17) or (ord(i) eq 18))..          enthalpie_sat_gas(i) =e= delta_h_v + cp_ig * (T_sat(i) - T0);
eq_sat_entropie_liq(i)$((ord(i) eq 17) or (ord(i) eq 18))..             entropie_sat_liq(i) =e= c_if * log(T_sat(i) / T0);
eq_sat_entropie_gas(i)$((ord(i) eq 17) or (ord(i) eq 18))..             entropie_sat_gas(i) =e= delta_h_v / T0 + cp_ig * log(T_sat(i) / T0) - R * log(p(i) / p0);

eq_enthalpie_1iq(i)$((ord(i) eq 4) or (ord(i) eq 5) or (ord(i) eq 9))..         h(i) =e= c_if * (T(i) - T0) + v_if * 100 * (p(i) - p0);
eq_enthalpie_gas(i)$((ord(i) eq 7) or (ord(i) eq 11) or (ord(i) eq 13) or (ord(i) eq 16))..             h(i) =e= delta_h_v + cp_ig * (T(i) - T0);
eq_entropie_gas(i)$((ord(i) eq 11) or (ord(i) eq 13) or (ord(i) eq 16))..               s(i) =e= delta_h_v / T0 + cp_ig * log(T(i) / T0) - R * log(p(i) / p0);
eq_enthalpie_2phase(i)$((ord(i) eq 14) or (ord(i) eq 15) or (ord(i) eq 17) or (ord(i) eq 18))..         h(i) =e= enthalpie_sat_liq(i) + x(i) * (enthalpie_sat_gas(i) - enthalpie_sat_liq(i));
eq_entropie_2phase(i)$((ord(i) eq 17) or (ord(i) eq 18))..              s(i) =e= entropie_sat_liq(i) + x(i) * (entropie_sat_gas(i) - entropie_sat_liq(i));

*****Cycle*****
eq_mLP..                        mLP =e= m * kLP;
eq_mHP..                        mHP =e= m * (1 - kLP);
eq_mBleed..             mBleed =e= m * kBl;
eq_mMain..              mMain =e= m * (1 - kBl);

*HP-Turbine (11->12)
eq_entropy_turbine12..                  s('16') =e= s('11');
eq_pressure_turbine12..                 p('16') =e= p('4');
eq_pressure_turbine122..                p('12') =e= p('4');
eq_work_turbine12..                             w_turbine12 =e= eta_st * (h('11') - h('16'));
eq_energy_balance_turbine12..   h('12') =e= h('11') - w_turbine12;
eq_p_turbine12..                                        p_turbine12 =e= mHP * w_turbine12;

*LP-Turbine
*Inlet
eq_isobaric_mixer..                             p('13') =e= p('4');
eq_energy_balance_mixer..               h('13') =e= kLP * h('7') + (1 - kLP) * h('12');
*Bleed (13->14)
eq_entropy_turbine14..                  s('17') =e= s('13');
eq_pressure_turbine14..                 p('17') =e= p('2');
eq_pressure_turbine142..                p('14') =e= p('2');
eq_work_turbine14..                             w_turbine14 =e= eta_st * (h('13') - h('17'));
eq_energy_balance_turbine14..   h('14') =e= h('13') - w_turbine14;
eq_p_turbine14..                                        p_turbine14 =e= mBleed * w_turbine14;
*Main (13->15)
eq_entropy_turbine15..                  s('18') =e= s('13');
eq_pressure_turbine15..                 p('18') =e= p('1');
eq_pressure_turbine152..                p('15') =e= p('1');
eq_work_turbine15..                             w_turbine15 =e= eta_st * (h('13') - h('18'));
eq_energy_balance_turbine15..   h('15') =e= h('13') - w_turbine15;
eq_p_turbine15..                                        p_turbine15 =e= mMain * w_turbine15;

*Condensate pump
eq_enthalpy1..                                          h('1') =e= enthalpie_sat_liq('1');
eq_work_pump2..                                 w_pump2 =e= (v_if * 100 * (p('2') - p('1'))) / eta_sp;
eq_energy_balance_pump2..               h('2') =e= h('1') + w_pump2;
eq_p_pump2..                                            p_pump2 =e= mBleed * w_pump2;

*Deaerator
eq_isobaric_deaerator..                 p('3') =e= p('2');
eq_energy_balance_deaerator..   h('3') =e= kBl * h('14') + (1 - kBl) * h('2');
eq_constraint_deaerator..               h('3') =e= enthalpie_sat_liq('3');

*LP-Pump
eq_work_pump4..                                 w_pump4 =e= (v_if * 100 * (p('4') - p('3'))) / eta_sp;
eq_energy_balance_pump4..               h('4') =e= h('3') + w_pump4;
eq_p_pump4..                                            p_pump4 =e= m * w_pump4;

*HP-Pump
eq_work_pump8..                                 w_pump8 =e= (v_if * 100 * (p('8') - p('5'))) / eta_sp;
eq_energy_balance_pump8..               h('8') =e= h('5') + w_pump8;
eq_p_pump8..                                            p_pump8 =e= mHP * w_pump8;

*Boiler
*Overall
eq_Qzu..                                                        Qzu =e= s_G_in * (T_G_in - TG7);
*HP-Superheater
eq_isobaric_HPSH..                      p('11') =e= p('10');
eq_energy_balance_HPSH..        mHP * (h('11') - h('10')) =e= s_G_in * (T_G_in - TG2);
*HP-Evaporator
eq_isobaric_HPEV..                      p('10') =e= p('9');
eq_energy_balance_HPEV..        mHP * (h('10') - h('9')) =e= s_G_in * (TG2 - TG3);
eq_enthalpy_10..                                h('10') =e= enthalpie_sat_gas('10');
*LP-Superheater
eq_isobaric_LPSH..                      p('7') =e= p('6');
eq_energy_balance_LPSH..        mLP * (h('7') - h('6')) =e= s_G_in * (TG3 - TG4);
*HP-Economizer
eq_isobaric_HPEC..                      p('9') =e= p('8');
eq_energy_balance_HPEC..        mHP * (h('9') - h('8')) =e= s_G_in * (TG4 - TG5);
eq_temperature_9..                      T('9') =e= T_sat('9') - deltaT_ap;
*LP-Evaporator
eq_isobaric_LPEV..                      p('6') =e= p('5');
eq_energy_balance_LPEV..        mLP * (h('6') - h('5')) =e= s_G_in * (TG5 - TG6);
eq_enthalpy_6..                         h('6') =e= enthalpie_sat_gas('6');
*LP-Economizer
eq_isobaric_LPEC..                      p('5') =e= p('4');
eq_energy_balance_LPEC..        m * (h('5') - h('4')) =e= s_G_in * (TG6 - TG7);
eq_temperature_5..                      T('5') =e= T_sat('5') - deltaT_ap;

*Cycle
eq_obj_p_net..                                  p_net =e= p_turbine12 + p_turbine14 + p_turbine15 - (p_pump2 + p_pump4 + p_pump8);


*****Constraints*****
ineq_constraint_fully_evap7..           h('7') =g= enthalpie_sat_gas('7');
ineq_constraint_fully_evap11..  h('11') =g= enthalpie_sat_gas('11');
ineq_constraint_pinch..                         TG3 =g= T_sat('10') + deltaT_min;
ineq_constraint_pinch2..                        TG3 =g= T('7') + deltaT_min;
ineq_constraint_pinch3..                        TG4 =g= T('9') + deltaT_min;
ineq_constraint_pinch4..                        TG6 =g= T_sat('6') + deltaT_min;
ineq_constraint_pinch5..                        TG7 =g= T('4') + deltaT_min;
ineq_constraint_h_16..                          h('16') =g= enthalpie_sat_gas('16');
ineq_constraint_p42..                           p('4') =g= p('2');
ineq_constraint_p84..                           p('8') =g= p('4');


*****Bounds*****
p.lo(i) = 0.05;
p.fx('1') = 0.05;
p.lo('2') = 0.2;
p.up('2') = 5;
p.lo('4') = 3;
p.up('4') = 15;
p.lo('8') = 10;
p.up('8') = 100;
m.lo = 5;
m.up = 100;
h.lo('7') = 2480;
h.up('7') = 3750;
h.lo('11') = 2480;
h.up('11') = 3750;
kLP.lo = 0.05;
kLP.up = 0.5;
kBl.lo = 0.01;
kBl.up = 0.2;

T_sat.lo(i) = 300;
T_sat.up(i)=T_max;
T.lo(i) = 300;
T.up(i) = T_max;
mHP.lo = 1.25;
mHP.up = 95;
mLP.lo = 0.05;
mLP.up = 75;
mBleed.lo = 0.05;
mBleed.up = 20;
mMain.lo =              4;
mMain.up = 99;
p_net.lo = 200;
p_net.up = 375000;
p.up(i)=100;
p.up('14')=15;
p.up('15')=3;
p.up('16')=15;
p.up('17')=3;
x.lo('14')=x_min;
x.up('14')=1;
x.lo('15')=x_min;
x.up('15')=1;
x.lo('17')=0.5;
x.up('17')=1;
x.lo('18')=0.5;
x.up('18')=1;
h.lo('1')=0;
h.lo('2')=0.25;
h.lo('3')=0.25;
h.lo('4')=0.25;
h.lo('5')=0.25 ;
h.lo('6')=2480;
h.lo('7')=2480;
h.lo('8')=0.25;
h.lo('9')=0.25;
h.lo('10')=2480;
h.lo('11')=2480;
h.lo('12')=2480;
h.lo('13')=2480;
h.lo('14')=1240;
h.lo('15')=1240;
h.lo('16')=2480;
h.lo('17')=1240;
h.lo('18')=1240;
h.up('1')=300;
h.up('2')=325;
h.up('3')=1500;
h.up('4')=1500;
h.up('5')=1500;
h.up('6')=3200;
h.up('8')=1500;
h.up('9')=1500;
h.up('10')=3200;
h.up('12')=3750;
h.up('13')=3750;
h.up('14')=3200;
h.up('15')=3200;
h.up('15')=3750;
h.up('17')=3200;
h.up('18')=3200;
s.lo('11')=4.23;
s.lo('13')=4.23;
s.lo('16')=2.115;
s.lo('17')=2.115;
s.lo('18')=4.23;
s.up('11')=10;
s.up('13')=10;
s.up('16')=10;
s.up('17')=10;
s.up('18')=10;
enthalpie_sat_liq.lo(i)=0;
enthalpie_sat_liq.up(i)=1500;
enthalpie_sat_gas.lo(i)=2480;
enthalpie_sat_gas.up(i)=3200;
entropie_sat_liq.lo(i)=0;
entropie_sat_liq.up(i)=3;
entropie_sat_gas.lo(i)=4.23;
entropie_sat_gas.up(i)=10;
TG7.lo=300;
TG7.up=900;
TG6.lo=300;
TG6.up=900;
TG5.lo=300;
TG5.up=900;
TG4.lo=300;
TG4.up=900;
TG3.lo=300;
TG3.up=900;
TG2.lo=300;
TG2.up=900;
Qzu.lo=12400;
Qzu.up=375000;
w_pump2.lo=0;
w_pump2.up=12.5;
w_pump4.lo=0;
w_pump4.up=12.5;
w_pump8.lo=0;
w_pump8.up=12.5;
p_pump2.lo=0;
p_pump2.up=1250;
p_pump4.lo=0;
p_pump4.up=1250;
p_pump8.lo=0;
p_pump8.up=1250;
w_turbine12.lo=100;
w_turbine12.up=3750;
w_turbine14.lo=100;
w_turbine14.up=3750;
w_turbine15.lo=100;
w_turbine15.up=3750;
p_turbine12.lo=0;
p_turbine12.up=375000;
p_turbine14.lo=0;
p_turbine14.up=375000;
p_turbine15.lo=0;
p_turbine15.up=375000;

model CSIII_Wnet /all/ ;

CSIII_Wnet.optfile = 1;

solve CSIII_Wnet maximizing p_net using nlp;




