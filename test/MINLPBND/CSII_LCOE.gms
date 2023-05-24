********************************************************************************
*
* Problem definition for CSII (Regenerative Rankine Cycle), min LCOE originally published in
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

                i               streams /1*11/
                j               heat exchangers         /1*4/

;


parameter

*Parameters for hot gas stream
                T_G_in                  gas inlet temperature [K]                                               /900/
                mcpG                            heat capacity flow rate of gas [kW|K]           /200/


*Cycle parameters
                eta_st                  isentropic turbine efficiency [-]                                               /0.9/
                eta_sp                  isentropic pump efficiency [-]                                                  /0.8/
                T_max                           max. temperature [K]                                                                                    /873.0/
                x_min                           min. vapor quality in turbine [-]                                               /0.85/
                DeltaT_min              min. temperature difference in heat exchangers [K]      /15.0/
                DeltaT_Ap               subcooling at economizer outlet [K]                                             /10.0/

*Parameters for properties
                delta_h_v               enthalpy of vaporization at T=T0 [kJ|kg]                /2480.0/
                p0                                      reference pressure [bar]                                                        /10E-3/
                T0                                      reference temperature [K]                                                       /313.8316/
                R                                       spec. gas constant [kJ|kg*K]                                            /0.462/
                c_if                            spec. heat capacity (ideal liquid) [kJ|kg*K]    /4.18/
                cp_ig                           spec. heat capacity (ideal gas) [kJ|kg*K]               /2.08/
                v_if                            spec. volume of liquid water [m^3|kg]                   /0.001/
                A                                       Antoine par.                                                                                    /3.55959/
                B                                       Antoine par.                                                                                    /643.748/
                C                                       Antoine par.                                                                                    /-198.043/

*Heat Transfer
                k(j)            heat transfer coefficient [kJ|m^2*K]            /1 0.35, 2 0.06, 3 0.06, 4 0.03/
                Tcin            cooling water inlet temperature [K]                     /293.15/
                Tcout           cooling water outlet temperature [K]            /298.15/

*Investment cost
*HX
                k1A             coefficient HX purchase cost correlation                /4.3247/
                k2A             coefficient HX purchase cost correlation                /-0.303/
                k3A             coefficient HX purchase cost correlation                /0.1634/
                c1A             coefficient HX pressure factor correlation      /0.03881/
                c2A             coefficient HX pressure factor correlation      /-0.11272/
                c3A             coefficient HX pressure factor correlation      /0.08183/
                FmA             material factor fro HX                                                          /2.75/
                B1A             coefficient HX investment cost                                  /1.63/
                B2A             coefficient HX investment cost                                  /1.66/
*DAE
                k1B             coefficient deaerator purchase cost correlation /3.5565/
                k2B             coefficient deaerator purchase cost correlation /0.3776/
                k3B             coefficient deaerator purchase cost correlation /0.0905/
                FpB             pressure factor deaerator                                                               /1.25/
                FmB             material factor deaerator                                                               /1/
                B1B             coefficient deaerator investment cost                           /1.49/
                B2B             coefficient deaerator investment cost                           /1.52/

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
                kBleed          fraction of mass flow rate extracted in the turbine [-]
                TG2                     temperature of gas at G2 [K]
                TG3                     temperature of gas at G3 [K]
                TG4                     temperature of gas at G4 [K]
                Qzu                     heat transfer rate in boiler [kW]
                w_pump2         spec. pumping work [kJ|kg]
                p_pump2         pump Power consumption [kW]
                w_pump4         spec. pumping work [kJ|kg]
                p_pump4                 pump Power consumption [kW]
                w_turbine8              spec. turbine work [kJ|kg]
                p_turbine8              Power output turbine [kW]
                w_turbine9              spec. turbine work [kJ|kg]
                p_turbine9              Power output turbine [kW]
                p_net                           net Power output [kW]


                Q(j)                    heat flow rate in HX [kW]
                dTa(j)          temperature difference at HX inlet [K]
                dTb(j)          temperature difference at HX outlet [K]
                LMTD(j)         logarithmic mean temperature difference in HX [K]
                Area(j)         area of HX [m2]
                Cp(j)                   purchase of HX in base state [$]
                Fp(j)                   pressure factor of HX [-]
                InvHX(j)                investment cost of HX [$]
                V_DAE                   volume of deaerator [m^3]
                CpDAE                   purchase cost of deaerator in base state [$]
                InvDAE          investment cost of deaerator [$]
                InvPump2                investment cost of pump 2 [$]
                InvPump4                investment cost of pump 4 [$]
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
                eq_enthalpie_2phase(i)
                eq_entropie_2phase(i)

                eq_entropy_turbine8
                eq_pressure_turbine8
                eq_pressure_turbine82
                eq_work_turbine8
                eq_energy_balance_turbine8
                eq_p_turbine8
                eq_entropy_turbine9
                eq_pressure_turbine9
                eq_pressure_turbine92
                eq_work_turbine9
                eq_energy_balance_turbine9
                eq_p_turbine9
                eq_temperature9

                eq_enthalpie_1
                eq_work_pump2
                eq_energy_balance_pump2
                eq_p_pump2

                eq_isobaric_deaerator
                eq_energy_balance_deaerator

                eq_work_pump4
                eq_energy_balance_pump4
                eq_p_pump4

                eq_Qzu
                eq_isobaric_economizer
                eq_isobaric_evaporator
                eq_isobaric_superheater
                eq_energy_balance_boiler
                eq_energy_balance_superheater
                eq_energy_balance_evaporator
                eq_enthalpie_6
                eq_temperature_5

                eq_obj_p_net

                eq_constraint_deaerator

                eq_Qcond
                eq_Qeco
                eq_Qevap
                eq_QSH
                eq_VDAE
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
                eq_CpDAE
                eq_Fp(j)
                eq_Fp1
                eq_InvHX(j)
                eq_InvDAE
                eq_InvPump2
                eq_InvPump4
                eq_InvTurb
                eq_Inv
                eq_WorkCC
                eq_etaCC
                eq_CAPEX
                eq_FuelCost
                eq_LCOE


******Inequalities
                ineq_constraint_fully_evap
                ineq_constraint_pinch
                ineq_constraint_pinch2
                ineq_constraint_p42

;


*****Thermodynamics*****
eq_sat_temperature(i)..         T_sat(i) =e= B /(A - log10(p(i))) - C;
eq_sat_enthalpie_liq(i)$((ord(i) eq 1) or (ord(i) eq 3) or (ord(i) eq 8) or (ord(i) eq 9) or (ord(i) eq 10) or (ord(i) eq 11))..                enthalpie_sat_liq(i) =e= c_if * (T_sat(i) - T0) + v_if * 100 * (p(i) - p0);
eq_sat_enthalpie_gas(i)$((ord(i) eq 6) or (ord(i) eq 7) or (ord(i) eq 8) or (ord(i) eq 9) or (ord(i) eq 10) or (ord(i) eq 11))..                enthalpie_sat_gas(i) =e= delta_h_v + cp_ig * (T_sat(i) - T0);
eq_sat_entropie_liq(i)$((ord(i) eq 10) or (ord(i) eq 11))..             entropie_sat_liq(i) =e= c_if * log(T_sat(i) / T0);
eq_sat_entropie_gas(i)$((ord(i) eq 10) or (ord(i) eq 11))..             entropie_sat_gas(i) =e= delta_h_v / T0 + cp_ig * log(T_sat(i) / T0) - R * log(p(i) / p0);

eq_enthalpie_1iq(i)$((ord(i) eq 1) or (ord(i) eq 4) or (ord(i) eq 5))..         h(i) =e= c_if * (T(i) - T0) + v_if * 100 * (p(i) - p0);
eq_enthalpie_gas(i)$((ord(i) eq 6) or (ord(i) eq 7))..          h(i) =e= delta_h_v + cp_ig * (T(i) - T0);
eq_entropie_gas(i)$(ord(i) eq 7)..              s(i) =e= delta_h_v / T0 + cp_ig * log(T(i) / T0) - R * log(p(i) / p0);
eq_enthalpie_2phase(i)$((ord(i) eq 8) or (ord(i) eq 9) or (ord(i) eq 10) or (ord(i) eq 11))..           h(i) =e= enthalpie_sat_liq(i) + x(i) * (enthalpie_sat_gas(i) - enthalpie_sat_liq(i));
eq_entropie_2phase(i)$((ord(i) eq 10) or (ord(i) eq 11))..              s(i) =e= entropie_sat_liq(i) + x(i) * (entropie_sat_gas(i) - entropie_sat_liq(i));

*****Cycle*****
*Turbine
*Bleed (7->8)
eq_entropy_turbine8..                   s('10') =e= s('7');
eq_pressure_turbine8..                  p('10') =e= p('2');
eq_pressure_turbine82..                 p('8') =e= p('2');
eq_work_turbine8..                              w_turbine8 =e= eta_st * (h('7') - h('10'));
eq_energy_balance_turbine8..    h('8') =e= h('7')*(1-eta_st) + h('10')*eta_st;
eq_p_turbine8..                                 p_turbine8 =e= m * kBleed * w_turbine8;
*Main (7->9)
eq_entropy_turbine9..                   s('11') =e= s('7');
eq_pressure_turbine9..                  p('11') =e= p('1');
eq_pressure_turbine92..                 p('9') =e= p('1');
eq_work_turbine9..                              w_turbine9 =e= eta_st * (h('7') - h('11'));
eq_energy_balance_turbine9..    h('9') =e= h('7')*(1-eta_st) + h('11')*eta_st;
eq_p_turbine9..                                 p_turbine9 =e= m * (1-kBleed) * w_turbine9;

*Condenser
eq_temperature9..               T('9') =e= T_sat('9');
eq_enthalpie_1..                h('1') =e= enthalpie_sat_liq('1');
eq_Qcond..                              Q('1') =e= m * (1-kBleed) * (h('9') - h('1'));

*Condensate pump
eq_work_pump2..                         w_pump2 =e= (v_if * 100 * (p('2') - p('1'))) / eta_sp;
eq_energy_balance_pump2..       h('2') =e= h('1') + w_pump2;
eq_p_pump2..                                    p_pump2 =e= m * (1 - kBleed) * w_pump2;

*Deaerator
eq_isobaric_deaerator..                 p('3') =e= p('2');
eq_energy_balance_deaerator..   h('3') =e= kBleed * h('8') + (1 - kBleed) * h('2');
eq_constraint_deaerator..               h('3') =e= enthalpie_sat_liq('3');

*Feedwater pump
eq_work_pump4..                         w_pump4 =e= (v_if * 100 * (p('4') - p('3'))) / eta_sp;
eq_energy_balance_pump4..       h('4') =e= h('3') + w_pump4;
eq_p_pump4..                                    p_pump4 =e= m * w_pump4;

*Boiler
*Overall
eq_Qzu..                                                                        Qzu =e= mcpG * (T_G_in - TG4);
eq_energy_balance_boiler..                      m * (h('7') - h('4')) =e= mcpG * (T_G_in - TG4);
*Superheater
eq_isobaric_superheater..                       p('7') =e= p('6');
eq_energy_balance_superheater.. m * (h('7') - h('6')) =e= mcpG * (T_G_in - TG2);
eq_QSH..                                                                        Q('4') =e= m * (h('7') - h('6'));
eq_enthalpie_6..                                                h('6') =e= enthalpie_sat_gas('6');
*Evaporator
eq_isobaric_evaporator..                        p('6') =e= p('5');
eq_energy_balance_evaporator..  m * (h('6') - h('5')) =e= mcpG * (TG2 - TG3);
eq_Qevap..                                                              Q('3') =e= m * (h('6') - h('5'));
eq_temperature_5..                                      T('5') =e= T_sat('5') - deltaT_ap;
*Economizer
eq_isobaric_economizer..                        p('5') =e= p('4');
eq_Qeco..                                                               Q('2') =e= m * (h('5') - h('4'));

*Cycle
eq_obj_p_net..          p_net =e= m*(kBleed*w_turbine8 + (1-kBleed)*(w_turbine9 - w_pump2) - w_pump4);

****** Economic analysis ******
*HX Areas
eq_VDAE..               V_DAE =e= 1.5 * m * v_if * 600;
eq_dT1a..               dTa('1') =e= T('9') - Tcout;
eq_dT1b..               dTb('1') =e= T('1') - Tcin;
eq_dT2a..               dTa('2') =e= TG4 - T('4');
eq_dT2b..               dTb('2') =e= TG3 - T('5');
eq_dT3a..               dTa('3') =e= TG3 - T_sat('5');
eq_dT3b..               dTb('3') =e= TG2 - T_sat('5');
eq_dT4a..               dTa('4') =e= TG2 - T_sat('6');
eq_dT4b..               dTb('4') =e= T_G_in - T('7');
eq_LMTD(j)..    LMTD(j) =e= ( dTa(j) * dTb(j) * (dTa(j)+dTb(j))/2 ) ** (1/3);
eq_A(j)..               Area(j) =e= Q(j) / (k(j) * LMTD(j));

*Investment
eq_CpDAE..      CpDAE =e= 10**(k1B + k2B*log10(V_DAE) + k3B*(log10(V_DAE))**2);
eq_InvDAE..     InvDAE =e= 1.18 * (B1B+B2B*FmB*FpB) * CpDAE;
eq_Cp(j)..      Cp(j) =e= 10**(k1A + k2A*log10(Area(j)) + k3A*(log10(Area(j)))**2);
eq_Fp1..                Fp('1') =e= 1;
eq_Fp(j)$((ord(j) eq 2) or (ord(j) eq 3) or (ord(j) eq 4))..            Fp(j) =e= 10**(c1A + c2A*log10(p('4')) + c3A*(log10(p('4')))**2);
eq_InvHX(j)..   InvHX(j) =e= 1.18 * (B1A+B2A*FmA*Fp(j)) * Cp(j);
eq_InvPump2..   InvPump2 =e= 3540 * p_pump2**0.71;
eq_InvPump4..   InvPump4 =e= 3540 * p_pump4**0.71;
eq_InvTurb..    InvTurb =e= 6000 * (p_turbine8+p_turbine9)**0.7 + 60 * (p_turbine8+p_turbine9)**0.95;
eq_Inv..                        Inv =e= InvPump2 + InvPump4 + InvTurb + SUM(j,InvHX(j)) + InvDAE;

*CCPP
eq_WorkCC..             Work_CC =e= p_net + Work_GT;
eq_etaCC..              CAPEX =e= (Inv+Inv_GT)*f_phi*f_annu/((Work_CC/1000)*Teq);
eq_CAPEX..              eta_CC =e= Work_CC / Fuel_heat;
eq_FuelCost..   FuelCost =e= GasPrice/eta_CC;
eq_LCOE..               LCOE =e= ((Inv+Inv_GT)*f_phi*f_annu*1000/Teq+ GasPrice*Fuel_heat)/Work_CC  + VarCost;


*****Constraints*****
ineq_constraint_fully_evap..    h('7') =g= enthalpie_sat_gas('7');
ineq_constraint_pinch..                 TG3 =g= T_sat('5') + deltaT_min;
ineq_constraint_pinch2..                TG4 =g= T('4') + deltaT_min;
ineq_constraint_p42..                   p('4') =g= p('2');


******Bounds*****
p.lo(i) = 0.05;
p.fx('1') = 0.05;
p.lo('2') = 0.2;
p.up('2') = 5;
p.lo('4') = 3;
p.up('4') = 100;
m.lo = 5;
m.up = 100;
h.lo('7') = 2480;
h.up('7') = 3750;
kBleed.lo = 0.01;
kBleed.up = 0.2;

T_sat.lo(i) = 300;
T_sat.up(i)=T_max;
T.lo(i) = 300;
T.up(i) = T_max;
p_net.lo = 200;
p_net.up = 375000;
p.up(i)=100;
p.up('8')=5;
p.up('9')=3;
p.up('10')=5;
p.up('11')=3;
x.lo('8')=x_min;
x.up('8')=1;
x.lo('9')=x_min;
x.up('9')=1;
x.lo('10')=0.5;
x.up('10')=1;
x.lo('11')=0.5;
x.up('11')=1;
h.lo('1')=0;
h.lo('2')=0.25;
h.lo('3')=0.25;
h.lo('4')=0.25;
h.lo('5')=0.25 ;
h.lo('6')=2480;
h.lo('7')=2480;
h.lo('8')=1240;
h.lo('9')=1240;
h.lo('10')=1240;
h.lo('11')=1240;
h.up('1')=300;
h.up('2')=325;
h.up('3')=1500;
h.up('4')=1500;
h.up('5')=1500;
h.up('6')=3200;
h.up('7')=3750;
h.up('8')=3200;
h.up('9')=3200;
h.up('10')=3200;
h.up('11')=3200;
s.lo('7')=4.23;
s.lo('10')=2.115;
s.lo('11')=2.115;
s.up('7')=10;
s.up('10')=10;
s.up('11')=10;
enthalpie_sat_liq.lo(i)=0;
enthalpie_sat_liq.up(i)=1500;
enthalpie_sat_gas.lo(i)=2480;
enthalpie_sat_gas.up(i)=3200;
entropie_sat_liq.lo(i)=0;
entropie_sat_liq.up(i)=3;
entropie_sat_gas.lo(i)=4.23;
entropie_sat_gas.up(i)=10;
TG4.lo=423;
TG4.up=900;
TG3.lo=423;
TG3.up=900;
TG2.lo=423;
TG2.up=900;
Qzu.lo=12400;
Qzu.up=375000;
w_pump2.lo=0;
w_pump2.up=12.5;
w_pump4.lo=0;
w_pump4.up=12.5;
p_pump2.lo=0;
p_pump2.up=1250;
p_pump4.lo=0;
p_pump4.up=1250;
w_turbine8.lo=100;
w_turbine8.up=3750;
w_turbine9.lo=100;
w_turbine9.up=3750;
p_turbine8.lo=500;
p_turbine8.up=375000;
p_turbine9.lo=500;
p_turbine9.up=375000;
V_DAE.lo=1;
V_DAE.up=1000;
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
CpDAE.lo=100;
CpDAE.up=1e8;
Fp.lo(j)=1;
Fp.up(j)=100;
InvHX.lo(j)=100;
InvHX.up(j)=1e8;
InvPump2.lo=100;
InvPump2.up=1e7;
InvPump4.lo=100;
InvPump4.up=1e7;
InvDAE.lo=100;
InvDAE.up=1e7;
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

model CSII_LCOE /all/ ;

CSII_LCOE.optfile = 1;

solve CSII_LCOE minimizing LCOE using nlp;




