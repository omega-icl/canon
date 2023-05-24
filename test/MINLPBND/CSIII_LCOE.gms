********************************************************************************
*
* Problem definition for CSIII (Two-Pressure Cycle), min LCOE originally published in
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
option reslim = 43200;


sets

                i               streams                          /1*18/
                j               heat exchangers         /1*7/

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
                delta_h_v               enthalpy of vaporization  at T=T0 [kJ|kg]                       /2480.0/
                p0                                      reference pressure [bar]                                                                /10E-3/
                T0                                      reference temperature [K]                                                               /313.8336/
                R                                       spec. gas constant  [kJ|kg*K]                                                   /0.462/
                c_if                            spec. heat capacity  (ideal liquid) [kJ|kg*K]   /4.18/
                cp_ig                           spec. heat capacity  (ideal gas) [kJ|kg*K]              /2.08/
                v_if                            spec. volume of liquid water [m^3|kg]                           /0.001/
                A                                       Antoine par.                                                                                            /3.55959/
                B                                       Antoine par.                                                                                            /643.748/
                C                                       Antoine par.                                                                                            /-198.043/

*Heat Transfer
                k(j)            heat transfer coefficient [kJ|m^2*K]            /1 0.35, 2 0.06, 3 0.06, 4 0.06, 5 0.03, 6 0.06, 7 0.03/
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
                FmA             material factor HX                                                                      /2.75/
                B1A             coefficient HX investment cost                                  /1.63/
                B2A             coefficient HX investment cost                                  /1.66/
*deaerator
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
                mBleed          mass flow rate of turbine bleed [kg|s]
                mMain                   mass flow rate of remaining turbine [kg|s]
                mHP                     mass flow rate through HP part [kg|s]
                mLP                     mass flow rate through LP part [kg|s]
                kBl                     fraction of mass flow rate extracted in the turbine [-]
                kLP                     fraction of mass flow rate going to LP part [-]
                TG2                             temperature of gas at G2 [K]
                TG3                             temperature of gas at G3 [K]
                TG4                             temperature of gas at G4 [K]
                TG5                             temperature of gas at G5 [K]
                TG6                             temperature of gas at G6 [K]
                TG7                             temperature of gas at G7 [K]
                Qzu                             heat transfer rate in boiler [kW]

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
                p_net                           net Power output [kW]

                Q(j)                            heat flow rate in HX [kW]
                dTa(j)                  temperature difference at HX inlet [K]
                dTb(j)                  temperature difference at HX outlet [K]
                LMTD(j)                 logarithmic mean temperature difference in HX [K]
                Area(j)                 area of HX [m2]
                Cp(j)                           purchase of HX in base state [$]
                Fp(j)                           pressure factor of HX [-]
                InvHX(j)                        investment cost of HX [$]
                V_deaerator             volume of deaerator [m^3]
                Cpdeaerator             purchase cost of deaerator in base state [$]
                Invdeaerator    investment cost of deaerator [$]
                InvPump2                        investment cost of pump 2 [$]
                InvPump4                        investment cost of pump 4 [$]
                InvPump8                        investment cost of pump 8 [$]
                InvTurbHP               investment cost of HP turbine [$]
                InvTurbLP               investment cost of LP turbine [$]
                InvGen                  investment cost of generator [$]
                Inv                             investment cost of steam cycle [$]
                Work_CC                 net Power output of the CCPP [kW]
                eta_CC                  1st law efficiency of the CCPP [kW]
                FuelCost                        fuel cost [$|MWh_el]
                CAPEX                           capital expenditure [$|MWh_el]

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
                eq_temperature_14
                eq_temperature_15

                eq_Qcond
                eq_QecoLP
                eq_QevapLP
                eq_QSHLP
                eq_QecoHP
                eq_QevapHP
                eq_QSHHP
                eq_Vdeaerator
                eq_dT1a
                eq_dT1b
                eq_dT2a
                eq_dT2b
                eq_dT3a
                eq_dT3b
                eq_dT4a
                eq_dT4b
                eq_dT5a
                eq_dT5b
                eq_dT6a
                eq_dT6b
                eq_dT7a
                eq_dT7b
                eq_LMTD(j)
                eq_A(j)
                eq_Cp(j)
                eq_Cpdeaerator
                eq_FpHP(j)
                eq_FpLP(j)
                eq_Fp1
                eq_InvHX(j)
                eq_Invdeaerator
                eq_InvPump2
                eq_InvPump4
                eq_InvPump8
                eq_InvTurbHP
                eq_InvTurbLP
                eq_InvGen
                eq_Inv
                eq_WorkCC
                eq_etaCC
                eq_CAPEX
                eq_FuelCost
                eq_LCOE

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
eq_sat_enthalpie_gas(i)$((ord(i) eq 6) or (ord(i) eq 7) or (ord(i) eq 10) or (ord(i) eq 11) or (ord(i) eq 14) or (ord(i) eq 15) or (ord(i) eq 16) or (ord(i) eq 17) or (ord(i) eq 18))..                enthalpie_sat_gas(i) =e= delta_h_v + cp_ig * (T_sat(i) - T0);
eq_sat_entropie_liq(i)$((ord(i) eq 17) or (ord(i) eq 18))..             entropie_sat_liq(i) =e= c_if * log(T_sat(i) / T0);
eq_sat_entropie_gas(i)$((ord(i) eq 17) or (ord(i) eq 18))..             entropie_sat_gas(i) =e= delta_h_v / T0 + cp_ig * log(T_sat(i) / T0) - R * log(p(i) / p0);

eq_enthalpie_1iq(i)$((ord(i) eq 1) or (ord(i) eq 4) or (ord(i) eq 5) or (ord(i) eq 8) or (ord(i) eq 9))..               h(i) =e= c_if * (T(i) - T0) + v_if * 100 * (p(i) - p0);
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
eq_energy_balance_turbine12..   h('12') =e= h('11')*(1-eta_st) + eta_st*h('16');
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
eq_energy_balance_turbine14..   h('14') =e= h('13')*(1-eta_st) +eta_st*h('17');
eq_p_turbine14..                                        p_turbine14 =e= mBleed * w_turbine14;
eq_temperature_14..                             T('14') =e= T_sat('14');
*Main (13->15)
eq_entropy_turbine15..                  s('18') =e= s('13');
eq_pressure_turbine15..                 p('18') =e= p('1');
eq_pressure_turbine152..                p('15') =e= p('1');
eq_work_turbine15..                             w_turbine15 =e= eta_st * (h('13') - h('18'));
eq_energy_balance_turbine15..   h('15') =e= h('13')*(1-eta_st) + eta_st*h('18');
eq_p_turbine15..                                        p_turbine15 =e= mMain * w_turbine15;
eq_temperature_15..                             T('15') =e= T_sat('15');

*Condensate pump
eq_enthalpy1..                                          h('1') =e= enthalpie_sat_liq('1');
eq_work_pump2..                                 w_pump2 =e= (v_if * 100 * (p('2') - p('1'))) / eta_sp;
eq_energy_balance_pump2..               h('2') =e= h('1') + w_pump2;
eq_p_pump2..                                            p_pump2 =e= mMain * w_pump2;
eq_Qcond..                                                      Q('1') =e= mMain * (h('15') - h('1'));

*Deaerator
eq_isobaric_deaerator..                         p('3') =e= p('2');
eq_energy_balance_deaerator..           h('3') =e= kBl * h('14') + (1 - kBl) * h('2');
eq_constraint_deaerator..                       h('3') =e= enthalpie_sat_liq('3');

*LP-Pump
eq_work_pump4..                         w_pump4 =e= (v_if * 100 * (p('4') - p('3'))) / eta_sp;
eq_energy_balance_pump4..       h('4') =e= h('3') + w_pump4;
eq_p_pump4..                                    p_pump4 =e= m * w_pump4;

*HP-Pump
eq_work_pump8..                 w_pump8 =e= (v_if * 100 * (p('8') - p('5'))) / eta_sp;
eq_energy_balance_pump8..       h('8') =e= h('5') + w_pump8;
eq_p_pump8..                            p_pump8 =e= mHP * w_pump8;

*Boiler
*Overall
eq_Qzu..                                                        Qzu =e= s_G_in * (T_G_in - TG7);
*HP-Superheater
eq_isobaric_HPSH..                      p('11') =e= p('10');
eq_energy_balance_HPSH..        mHP * (h('11') - h('10')) =e= s_G_in * (T_G_in - TG2);
eq_QSHHP..                                              Q('7') =e= mHP * (h('11') - h('10'));
*HP-Evaporator
eq_isobaric_HPEV..                      p('10') =e= p('9');
eq_energy_balance_HPEV..        mHP * (h('10') - h('9')) =e= s_G_in * (TG2 - TG3);
eq_enthalpy_10..                                h('10') =e= enthalpie_sat_gas('10');
eq_QevapHP..                                    Q('6') =e= mHP * (h('10') - h('9'));
*LP-Superheater
eq_isobaric_LPSH..                      p('7') =e= p('6');
eq_energy_balance_LPSH..        mLP * (h('7') - h('6')) =e= s_G_in * (TG3 - TG4);
eq_QSHLP..                                              Q('5') =e= mLP * (h('7') - h('6'));
*HP-Economizer
eq_isobaric_HPEC..                      p('9') =e= p('8');
eq_energy_balance_HPEC..        mHP * (h('9') - h('8')) =e= s_G_in * (TG4 - TG5);
eq_temperature_9..                              T('9') =e= T_sat('9') - deltaT_ap;
eq_QecoHP..                                             Q('4') =e= mHP * (h('9') - h('8'));
*LP-Evaporator
eq_isobaric_LPEV..                      p('6') =e= p('5');
eq_energy_balance_LPEV..        mLP * (h('6') - h('5')) =e= s_G_in * (TG5 - TG6);
eq_enthalpy_6..                         h('6') =e= enthalpie_sat_gas('6');
eq_QevapLP..                                    Q('3') =e= mLP * (h('6') - h('5'));
*LP-Economizer
eq_isobaric_LPEC..                      p('5') =e= p('4');
eq_energy_balance_LPEC..        m * (h('5') - h('4')) =e= s_G_in * (TG6 - TG7);
eq_temperature_5..                      T('5') =e= T_sat('5') - deltaT_ap;
eq_QecoLP..                                             Q('2') =e= m * (h('5') - h('4'));

*Cycle
eq_obj_p_net..                                  p_net =e= m*((1-kLP)*(w_turbine12-w_pump8) + kBl*w_turbine14 + (1-kBl)*(w_turbine15-w_pump2) - w_pump4);

*****Economic evaluation*****
*HX Areas
eq_Vdeaerator..         V_deaerator =e= 1.5 * m * v_if * 600;
eq_dT1a..                               dTa('1') =e= T('15') - Tcout;
eq_dT1b..                               dTb('1') =e= T('1') - Tcin;
eq_dT2a..                               dTa('2') =e= TG7 - T('4');
eq_dT2b..                               dTb('2') =e= TG6 - T('5');
eq_dT3a..                               dTa('3') =e= TG6 - T_sat('5');
eq_dT3b..                               dTb('3') =e= TG5 - T_sat('5');
eq_dT4a..                               dTa('4') =e= TG5 - T('8');
eq_dT4b..                               dTb('4') =e= TG4 - T('9');
eq_dT5a..                               dTa('5') =e= TG4 - T_sat('5');
eq_dT5b..                               dTb('5') =e= TG3 - T('7');
eq_dT6a..                               dTa('6') =e= TG3 - T_sat('10');
eq_dT6b..                               dTb('6') =e= TG2 - T_sat('10');
eq_dT7a..                               dTa('7') =e= TG2 - T_sat('10');
eq_dT7b..                               dTb('7') =e= T_G_in - T('11');
eq_LMTD(j)..                    LMTD(j) =e= ( dTa(j) * dTb(j) * (dTa(j)+dTb(j))/2 ) ** (1/3);
eq_A(j)..                               Area(j) =e= Q(j) / (k(j) * LMTD(j));

*Investment
eq_Cpdeaerator..                Cpdeaerator =e= 10**(k1B + k2B*log10(V_deaerator) + k3B*(log10(V_deaerator))**2);
eq_Invdeaerator..               Invdeaerator =e= 1.18 * (B1B+B2B*FmB*FpB) * Cpdeaerator;
eq_Cp(j)..      Cp(j) =e= 10**(k1A + k2A*log10(Area(j)) + k3A*(log10(Area(j)))**2);
eq_Fp1..                Fp('1') =e= 1;
eq_FpLP(j)$((ord(j) eq 2) or (ord(j) eq 3) or (ord(j) eq 5))..          Fp(j) =e= 10**(c1A + c2A*log10(p('4')) + c3A*(log10(p('4')))**2);
eq_FpHP(j)$((ord(j) eq 4) or (ord(j) eq 6) or (ord(j) eq 7))..          Fp(j) =e= 10**(c1A + c2A*log10(p('8')) + c3A*(log10(p('8')))**2);
eq_InvHX(j)..           InvHX(j) =e= 1.18 * (B1A+B2A*FmA*Fp(j)) * Cp(j);
eq_InvPump2..           InvPump2 =e= 3540 * p_pump2**0.71;
eq_InvPump4..           InvPump4 =e= 3540 * p_pump4**0.71;
eq_InvPump8..           InvPump8 =e= 3540 * p_pump8**0.71;
eq_InvTurbLP..          InvTurbLP =e= 6000 * (p_turbine14+p_turbine15)**0.7;
eq_InvTurbHP..          InvTurbHP =e= 6000 * p_turbine12**0.7;
eq_InvGen..                     InvGen =e= 60 * (p_turbine14+p_turbine15 + p_turbine12)**0.95;
eq_Inv..                                Inv =e= InvPump2 + InvPump4 + InvPump8 + InvTurbLP + InvTurbHP + InvGen + SUM(j,InvHX(j)) + Invdeaerator;

*CCPP
eq_WorkCC..             Work_CC =e= p_net + Work_GT;
eq_etaCC..              CAPEX =e= (Inv+Inv_GT)*f_phi*f_annu/((Work_CC/1000)*Teq);
eq_CAPEX..              eta_CC =e= Work_CC / Fuel_heat;
eq_FuelCost..   FuelCost =e= GasPrice/eta_CC;
eq_LCOE..               LCOE =e= ((Inv+Inv_GT)*f_phi*f_annu*1000/Teq+GasPrice*Fuel_heat)/Work_CC + VarCost;



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


*****Bounds
p.lo(i) = 0.05;
p.up(i)=100;
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

T_sat.lo(i) = 293;
T_sat.up(i)=T_max;
T.lo(i) = 293;
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
h.up('16')=3750;
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
V_deaerator.lo=1;
V_deaerator.up=1000;
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
Cpdeaerator.lo=100;
Cpdeaerator.up=1e8;
Fp.lo(j)=1;
Fp.up(j)=100;
InvHX.lo(j)=100;
InvHX.up(j)=1e8;
InvPump2.lo=100;
InvPump2.up=1e7;
InvPump4.lo=100;
InvPump4.up=1e7;
InvPump8.lo=100;
InvPump8.up=1e7;
Invdeaerator.lo=100;
Invdeaerator.up=1e7;
InvTurbLP.lo=100;
InvTurbLP.up=5e8;
InvTurbHP.lo=100;
InvTurbHP.up=5e8;
InvGen.lo=100;
InvGen.up=5e7;
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


model CSIII_LCOE /all/ ;

CSIII_LCOE.optfile = 1;

solve CSIII_LCOE minimizing LCOE using nlp;




