********************************************************************************
*
* Problem definition for CSI (Simple Rankine Cycle), max Wnet originally published in
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

*option nlp = baron;
option optcr = 1e-3;
option reslim = 21600;



sets

                i               streams /1*7/

;



parameter

*Parameters for hot gas stream
                T_G_in                  gas inlet temperature [K]                                               /900/
                T_G_out                 gas outlet temperature [K]                                              /448/
                mcp_G                           heat capacity flow rate of gas [kW|K]           /200/


*Cycle parameters
                eta_st                  isentropic turbine efficiency [-]                       /0.9/
                eta_sp                  isentropic pump efficiency [-]                          /0.8/
                T_max                           max. temperature [K]                                                            /873.0/
                x_min                           min. vapor quality in turbine [-]                       /0.85/
                DeltaT_min              min. temp. diff. in heat exchangers [K] /15.0/
                DeltaT_Ap               subcooling at economizer outlet [K]                     /10.0/

*Parameters for properties
                delta_h_v       enthalpy of vaporization at T=T0 [kJ|kg]                        /2480.0/
                p0                              reference pressure [bar]                                                                /10E-3/
                T0                              reference temperature [K]                                                               /313.8316/
                R                               spec. gas constant  [kJ|kg*K]                                                   /0.462/
                c_if                    spec. heat capacity  (ideal liquid) [kJ|kg*K]   /4.18/
                cp_ig                   spec. heat capacity  (ideal gas) [kJ|kg*K]              /2.08/
                v_if                    spec. volume of liquid water [m^3|kg]                           /0.001/
                A                               Antoine  par.           /3.55959/
                B                               Antoine  par.           /643.748/
                C                               Antoine  par.           /-198.043/

;



free variables

                Work_net                net Power output [kW] (objective)

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

                m                                               mass flow rate [kg|s]
                T_G2                                    temperature of gas at G2 [K]
                T_G3                                    temperature of gas at G3 [K]
                Q_zu                                    heat transfer rate in boiler [kW]
                w_pump                          spec. pumping work [kJ|kg]
                Work_pump                       pump Power consumption [kW]
                w_turbine                       spec. turbine work [kJ|kg]
                Work_turbine            Power output turbine [kW]


;



equations

*****equalities
                eq_sat_temperature(i)
                eq_sat_enthalpie_liq(i)
                eq_sat_enthalpie_gas(i)
                eq_sat_entropie_liq(i)
                eq_sat_entropie_gas(i)

                eq_enthalpie_1iq(i)
                eq_enthalpie_gas(i)
                eq_entropie_gas(i)
                eq_enthalpie_twophase(i)
                eq_entropie_twophase(i)

                eq_enthalpy_1
                eq_energy_balance_pump
                eq_w_pump
                eq_p_pump

                eq_Qzu
                eq_isobaric_economizer
                eq_isobaric_evaporator
                eq_isobaric_superheater
                eq_energy_balance_boiler
                eq_energy_balance_superheater
                eq_energy_balance_evaporator
                eq_temperature_3
                eq_enthalpie_4

                eq_pressure_turbine6
                eq_pressure_turbine7
                eq_entropy_balance_turb
                eq_energy_balance_turb
                eq_p_turb
                eq_w_turb

                eq_w_net

*****Inequalities
                ineq_fully_evap
                ineq_pinch
;


***** Thermodynamics *****
eq_sat_temperature(i)..         T_sat(i) =e= B /(A - log10(p(i))) - C;
eq_sat_enthalpie_liq(i)$((ord(i) eq 1) or (ord(i) eq 6) or (ord(i) eq 7))..             enthalpie_sat_liq(i) =e= c_if * (T_sat(i) - T0) + v_if * 100 * (p(i) - p0);
eq_sat_enthalpie_gas(i)$((ord(i) eq 4) or (ord(i) eq 5) or (ord(i) eq 6) or (ord(i) eq 7))..            enthalpie_sat_gas(i) =e= delta_h_v + cp_ig * (T_sat(i) - T0);
eq_sat_entropie_liq(i)$(ord(i) eq 7)..          entropie_sat_liq(i) =e= c_if * log(T_sat(i) / T0);
eq_sat_entropie_gas(i)$(ord(i) eq 7)..          entropie_sat_gas(i) =e= delta_h_v / T0 + cp_ig * log(T_sat(i) / T0) - R * log(p(i) / p0);

eq_enthalpie_1iq(i)$(ord(i) eq 3)..             h(i) =e= c_if * (T(i) - T0) + v_if * 100 * (p(i) - p0);
eq_enthalpie_gas(i)$(ord(i) eq 5)..             h(i) =e= delta_h_v + cp_ig * (T(i) - T0);
eq_entropie_gas(i)$(ord(i) eq 5)..              s(i) =e= delta_h_v / T0 + cp_ig * log(T(i) / T0) - R * log(p(i) / p0);
eq_enthalpie_twophase(i)$((ord(i) eq 6) or (ord(i) eq 7))..             h(i) =e= enthalpie_sat_liq(i) + x(i) * (enthalpie_sat_gas(i) - enthalpie_sat_liq(i));
eq_entropie_twophase(i)$(ord(i) eq 7)..         s(i) =e= entropie_sat_liq(i) + x(i) * (entropie_sat_gas(i) - entropie_sat_liq(i));


****** Cycle ******
* Condenser outlet
eq_enthalpy_1..                                 h('1') =e= enthalpie_sat_liq('1');

*Pump
eq_energy_balance_pump..                h('2') =e= h('1') + w_pump;
eq_w_pump..                                                     w_pump =e= (v_if * 100 * (p('2') - p('1'))) / eta_sp;
eq_p_pump..                                                     Work_pump =e= m * w_pump;

*Boiler
eq_Qzu..                                                                        Q_zu =e= mcp_G * (T_G_in - T_G_out);
eq_energy_balance_boiler..                      Q_zu =e= m * (h('5') - h('2'));
eq_isobaric_superheater..                       p('5') =e= p('4');
eq_energy_balance_superheater.. m * (h('5') - h('4')) =e= mcp_g * (T_G_in - T_G2);
eq_isobaric_evaporator..                        p('4') =e= p('3');
eq_enthalpie_4..                                                h('4') =e= enthalpie_sat_gas('4');
eq_energy_balance_evaporator..  m * (h('4') - h('3')) =e= mcp_g * (T_G2 - T_G3);
eq_isobaric_economizer..                        p('3') =e= p('2');
eq_temperature_3..                                      T('3') =e= T_sat('3') - deltaT_Ap;

*Turbine
eq_pressure_turbine7..                  p('7') =e= p('1');
eq_pressure_turbine6..                  p('6') =e= p('1');
eq_entropy_balance_turb..               s('7') =e= s('5');
eq_energy_balance_turb..                h('6') =e= h('5') - w_turbine;
eq_w_turb..             w_turbine =e= eta_st * (h('5') - h('7'));
eq_p_turb..             Work_turbine =e= m * w_turbine;

*cycle
eq_w_net..              Work_net =e= Work_turbine - Work_pump;

****** Constraints ******
ineq_pinch..                    T_G3 =g= T_sat('4') + DeltaT_min;
ineq_fully_evap..               h('5') =g= enthalpie_sat_gas('5');


****** Bounds ******
p.fx('1')=0.2;
p.lo(i)=0.2;
p.lo('2') = 3;
p.up('2') = 100;
m.lo = 5;
m.up = 100;
T_sat.lo(i) = 349;
T_sat.up(i)=T_max;
T.lo('3') = 349;
T.lo('5') = 349;
T.up('3') = T_max;
T.up('5') = T_max;

Work_net.lo = 200;
Work_net.up = 375000;
p.up(i)=100;
p.up('1')=0.2;
p.up('6')=3;
p.up('7')=3;
x.lo('6')=x_min;
x.up('6')=1;
x.lo('7')=0.5;
x.up('7')=1;
h.lo('1')=0;
h.lo('2')=0.25;
h.lo('3')=0.25;
h.lo('4')=2480;
h.lo('5')=2480;
h.lo('6')=1240;
h.lo('7')=1240;
h.up('1')=300;
h.up('2')=325;
h.up('3')=1500;
h.up('4')=3200;
h.up('5')=3750;
h.up('6')=3200;
h.up('7')=3200;
s.lo('5')=4.23;
s.lo('7')=2.115;
s.up('5')=10;
s.up('7')=10;
enthalpie_sat_liq.lo(i)=0;
enthalpie_sat_liq.up(i)=1500;
enthalpie_sat_gas.lo(i)=2480;
enthalpie_sat_gas.up(i)=3200;
entropie_sat_liq.lo(i)=0;
entropie_sat_liq.up(i)=3;
entropie_sat_gas.lo(i)=4.23;
entropie_sat_gas.up(i)=10;
T_G3.lo=423;
T_G3.up=900;
T_G2.lo=423;
T_G2.up=900;
Q_zu.lo=12400;
Q_zu.up=375000;
w_pump.lo=0.25;
w_pump.up=12.5;
Work_pump.lo=1.25;
Work_pump.up=1250;
w_turbine.lo=100;
w_turbine.up=3750;
Work_turbine.lo=500;
Work_turbine.up=375000;


model CSI_Wnet /all/;

CSI_Wnet.optfile = 1;

solve CSI_Wnet maximizing Work_net using nlp;


