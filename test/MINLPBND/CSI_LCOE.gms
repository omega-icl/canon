********************************************************************************
*
* Problem definition for CSI (Simple Rankine Cycle), min LCOE originally published in
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

                i               streams                                 /1*7/
                j               heat exchangers /1*4/

;


parameter

*Parameters for hot gas stream
                T_G_in          gas inlet temperature [K]                                       /900/
                T_G_out         gas outlet temperature [K]                                      /448/
                mcp_G                   heat capacity flow rate of gas [kW|K]   /200/


*Cycle parameters
                eta_st          isentropic turbine efficiency [-]               /0.9/
                eta_sp          isentropic pump efficiency [-]                  /0.8/
                T_max                   max. temperature [K]                                                    /873.0/
                x_min                   min. vapor quality in turbine [-]               /0.85/
                DeltaT_min      min. temnp. diff. in HX [K]                             /15.0/
                DeltaT_Ap       subcooling at economizer outlet [K]             /10.0/

*Parameters for properties
                delta_h_v               enthalpy of vaporization [kJ|kg]                                                /2480.0/
                p0                                      reference pressure [bar]                                                                /10E-3/
                T0                                      reference temperature [K]                                                               /313.8316/
                R                                       spec. gas constant  [kJ|kg*K]                                                   /0.462/
                c_if                            spec. heat capacity  (ideal liquid) [kJ|kg*K]   /4.18/
                cp_ig                           spec. heat capacity  (ideal gas) [kJ|kg*K]              /2.08/
                v_if                            spec. volume of liquid water [m^3|kg]                           /0.001/
                A               Antoine par.            /3.55959/
                B               Antoine par.            /643.748/
                C               Antoine par.            /-198.043/

*Heat Transfer
                k(j)            heat transfer coefficient [kJ|m^2*K]    /1 0.35, 2 0.06, 3 0.06, 4 0.03/
                Tcin            cooling water inlet temperature [K]             /293.15/
                Tcout           cooling water outlet temperature [K]    /298.15/

*Investment cost
                k1A             coefficient HX purchase cost correlation                /4.3247/
                k2A             coefficient HX purchase cost correlation                /-0.303/
                k3A             coefficient HX purchase cost correlation                /0.1634/
                c1A             coefficient HX pressure factor correlation      /0.03881/
                c2A             coefficient HX pressure factor correlation      /-0.11272/
                c3A             coefficient HX pressure factor correlation      /0.08183/
                FmA             material factor HX                                              /2.75/
                B1A             coefficient HX investment cost          /1.63/
                B2A             coefficient HX investment cost          /1.66/

*Gas turbine
                Work_GT         net Power output of gas turbine [kW]            /69676/
                Fuel_heat       fuel constumption of gas turbine [kW]           /182359/
                Inv_GT          investment cost of gas turbine [$]                      /22.7176e6/

*Economic data
                GasPrice                natural gas price [$|MWh_gas]                   /14/
                f_phi                   maintenance factor [-]                                  /1.06/
                f_annu          annuity factor [1|a]                                            /0.1875/
                Teq                     equivalent utilization time [h|a]       /4000/
                VarCost         var. operating cost [$|MWh]                     /4/

;



free variables

                LCOE            levelized cost of electricity [$|MWh_el] (objective)

;



positive variables

                T(i)            temperature[K]
                p(i)            pressure [bar]
                x(i)            vapor quality [-]
                h(i)            spec. enthalpy [kJ|kg]
                s(i)            spec. entropy [kJ|K]

                T_sat(i)                                                saturation tempearture
                enthalpie_sat_liq(i)            saturated liquid enthalpy [kJ|kg]
                enthalpie_sat_gas(i)            saturated vapor enthalpy [kJ|kg]
                entropie_sat_liq(i)             saturated liquid entropy [kJ|kg]
                entropie_sat_gas(i)             saturated vapor entrop [kJ|kg]


                m                                       mass flow rate [kg|s]
                T_G2                            temperature of gas at G2 [K]
                T_G3                            temperature of gas at G3 [K]
                Q_zu                            heat transfer rate in boiler [kW]
                w_pump                  spec. pumping work [kJ|kg]
                Work_pump               pump Power consumption [kW]
                w_turbine               spec. turbine work [kJ|kg]
                Work_turbine    Power output turbine [kW]
                Work_net                        net Power output of steam cycle [kW]


                Q(j)                    heat flow rate in HX [kW]
                dTa(j)          temperature difference at HX inlet [K]
                dTb(j)          temperature difference at HX outlet [K]
                LMTD(j)         logarithmic mean temperature difference in HX [K]
                Area(j)         area of HX [m2]
                Cp(j)                   purchase of HX in base state [$]
                Fp(j)                   pressure factor of HX [-]
                InvHX(j)                investment cost of HX [$]
                InvPump         investment cost of pump [$]
                InvTurb         investment cost of turbine [$]
                Inv                     investment cost of steam cycle [$]
                Work_CC         net Power output of the CCPP [kW]
                eta_CC          1st law efficiency of the CCPP [kW]
                FuelCost                fuel cost [$|MWh_el]
                CAPEX                   capital expenditure [$|MWh_el]

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
                eq_energy_balance_condenser
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
                eq_QSH
                eq_QEvap
                eq_QEco
                eq_temperature_3
                eq_enthalpie_4

                eq_pressure_turbine6
                eq_pressure_turbine7
                eq_p_turb
                eq_w_turb
                eq_entropy_balance_turb
                eq_energy_balance_turb
                eq_temperature_6

                eq_w_net

                eq_dT1a
                eq_dT1b
                eq_dT2a
                eq_dT2b
                eq_dT3a
                eq_dT3b
                eq_dT4a
                eq_dT4b
                eq_LMTD(j)
                eq_A(j)
                eq_Cp(j)
                eq_Fp(j)
                eq_Fp1
                eq_InvHX(j)
                eq_InvPump
                eq_InvTurb
                eq_Inv
                eq_WorkCC
                eq_etaCC
                eq_CAPEX
                eq_FuelCost
                eq_LCOE


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

eq_enthalpie_1iq(i)$((ord(i) eq 1) or (ord(i) eq 2) or (ord(i) eq 3))..         h(i) =e= c_if * (T(i) - T0) + v_if * 100 * (p(i) - p0);
eq_enthalpie_gas(i)$(ord(i) eq 5)..             h(i) =e= delta_h_v + cp_ig * (T(i) - T0);
eq_entropie_gas(i)$(ord(i) eq 5)..              s(i) =e= delta_h_v / T0 + cp_ig * log(T(i) / T0) - R * log(p(i) / p0);
eq_enthalpie_twophase(i)$((ord(i) eq 6) or (ord(i) eq 7))..             h(i) =e= enthalpie_sat_liq(i) + x(i) * (enthalpie_sat_gas(i) - enthalpie_sat_liq(i));
eq_entropie_twophase(i)$(ord(i) eq 7)..         s(i) =e= entropie_sat_liq(i) + x(i) * (entropie_sat_gas(i) - entropie_sat_liq(i));


****** Cycle ******
* Condenser
eq_enthalpy_1..                                         h('1') =e= enthalpie_sat_liq('1');
eq_energy_balance_condenser..           Q('1') =e= m * (h('6') - h('1'));

*Pump
eq_energy_balance_pump..                h('2') =e= h('1') + w_pump;
eq_w_pump..                                                     w_pump =e= (v_if * 100 * (p('2') - p('1'))) / eta_sp;
eq_p_pump..                                                     Work_pump =e= m * w_pump;

*Boiler
eq_Qzu..                                                                        Q_zu =e= mcp_G * (T_G_in - T_G_out);
eq_energy_balance_boiler..                      Q_zu =e= m * (h('5') - h('2'));
eq_isobaric_superheater..                       p('5') =e= p('4');
eq_energy_balance_superheater.. m * (h('5') - h('4')) =e= mcp_g * (T_G_in - T_G2);
eq_QSH..                                                        Q('4') =e= m * (h('5') - h('4'));
eq_isobaric_evaporator..        p('4') =e= p('3');
eq_enthalpie_4..                                h('4') =e= enthalpie_sat_gas('4');
eq_energy_balance_evaporator..          m * (h('4') - h('3')) =e= mcp_g * (T_G2 - T_G3);
eq_QEvap..                                                      Q('3') =e= m * (h('4') - h('3'));
eq_QEco..                                                       Q('2') =e= Q_zu - Q('3') - Q('4');
eq_isobaric_economizer..                p('3') =e= p('2');
eq_temperature_3..                              T('3') =e= T_sat('3') - deltaT_Ap;

*Turbine
eq_pressure_turbine7..                  p('7') =e= p('1');
eq_pressure_turbine6..                  p('6') =e= p('1');
eq_entropy_balance_turb..               s('7') =e= s('5');
eq_energy_balance_turb..                h('6') =e= h('5') - w_turbine;
eq_w_turb..                                                     w_turbine =e= eta_st * (h('5') - h('7'));
eq_p_turb..                                                     Work_turbine =e= m * w_turbine;
eq_temperature_6..                              T('6') =e= T_sat('6');

*cycle
eq_w_net..              Work_net =e= Work_turbine - Work_pump;

****** Economic analysis ******
*HX Areas
eq_dT1a..               dTa('1') =e= T('6') - Tcout;
eq_dT1b..               dTb('1') =e= T('1') - Tcin;
eq_dT2a..               dTa('2') =e= T_G_out - T('2');
eq_dT2b..               dTb('2') =e= T_G3 - T('3');
eq_dT3a..               dTa('3') =e= T_G3 - T_sat('4');
eq_dT3b..               dTb('3') =e= T_G2 - T_sat('4');
eq_dT4a..               dTa('4') =e= T_G2 - T_sat('4');
eq_dT4b..               dTb('4') =e= T_G_in - T('5');
eq_LMTD(j)..    LMTD(j) =e= ( dTa(j) * dTb(j) * (dTa(j)+dTb(j))/2 ) ** (1/3);
eq_A(j)..               Area(j) =e= Q(j) / (k(j) * LMTD(j));

*Investment
eq_Cp(j)..              Cp(j) =e= 10**(k1A + k2A*log10(Area(j)) + k3A*(log10(Area(j)))**2);
eq_Fp1..                        Fp('1') =e= 1;
eq_Fp(j)$((ord(j) eq 2) or (ord(j) eq 3) or (ord(j) eq 4))..            Fp(j) =e= 10**(c1A + c2A*log10(p('2')) + c3A*(log10(p('2')))**2);
eq_InvHX(j)..   InvHX(j) =e= 1.18 * (B1A+B2A*FmA*Fp(j)) * Cp(j);
eq_InvPump..    InvPump =e= 3540 * Work_pump**0.71;
eq_InvTurb..    InvTurb =e= 6000 * Work_turbine**0.7 + 60 * Work_turbine**0.95;
eq_Inv..                        Inv =e= InvPump + InvTurb + SUM(j,InvHX(j));

*CCPP
eq_WorkCC..             Work_CC =e= Work_net + Work_GT;
eq_CAPEX..              CAPEX =e= (Inv+Inv_GT)*f_phi*f_annu/((Work_CC/1000)*Teq);
eq_etaCC..              eta_CC =e= Work_CC / Fuel_heat;
eq_FuelCost..   FuelCost =e= GasPrice/eta_CC;
eq_LCOE..               LCOE =e= CAPEX + FuelCost + VarCost;


****** Constraints ******
ineq_pinch..            T_G3 =g= T_sat('4') + DeltaT_min;
ineq_fully_evap..       h('5') =g= enthalpie_sat_gas('5');



****** Bounds ******
p.fx('1')=0.2;
p.lo(i)=0.2;
p.lo('2') = 3;
p.up('2') = 100;
m.lo = 5;
m.up = 100;
T_sat.lo(i) = 349;
T_sat.up(i)=T_max;
T.lo(i) = 349;
T.lo(i) = 349;
T.up(i) = T_max;
T.up(i) = T_max;

Work_net.lo = 200;
Work_net.up = 375000;
p.up(i)=100;
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
Q.lo(j)=10;
Q.up(j)=100000;
dTa.lo(j)=DeltaT_min;
dTa.up(j)=570;
dTb.lo(j)=DeltaT_min;
dTb.up(j)=570;
LMTD.lo(j)=DeltaT_min;
LMTD.up(j)=570;
Area.lo(j)=10;
Area.up(j)=100000;
Cp.lo(j)=100;
Cp.up(j)=1e8;
Fp.lo(j)=1;
Fp.up(j)=100;
InvHX.lo(j)=100;
InvHX.up(j)=1e8;
InvPump.lo=100;
InvPump.up=1e7;
InvTurb.lo=100;
InvTurb.up=5e8;
Inv.lo=100;
Inv.up=5e8;
Work_CC.lo = 70200;
Work_CC.up = 445000;
eta_CC.lo = 0.001;
eta_CC.up = 1;
FuelCost.lo = 0;
FuelCost.up = 1000;
CAPEX.lo = 0;
CAPEX.up = 1000;
LCOE.lo = 0;
LCOE.up = 1000;



model CSI_LCOE /all/;

CSI_LCOE.optfile = 1;
CSI_LCOE.prioropt = 1;

solve CSI_LCOE minimizing LCOE using nlp;




