#ifndef MC__RELUANN_HPP
#define MC__RELUANN_HPP
  
#ifdef MC__USE_PROFIL
 #include "mcprofil.hpp"
 typedef INTERVAL I;  
#else    
 #ifdef MC__USE_BOOST  
  #include "mcboost.hpp"
   typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
   typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
   typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
   typedef boost::numeric::interval<double,T_boost_policy> I;
 #else  
  #ifdef MC__USE_FILIB
   #include "mcfilib.hpp"
   typedef filib::interval<double> I;
  #else
   #include "interval.hpp"
   typedef mc::Interval I;
  #endif
 #endif
#endif

#if defined( USE_MC ) or defined( USE_MCISM )
 #include "mccormick.hpp"
#elif defined( USE_ISM ) or defined( USE_MCISM )
 #include "ismodel.hpp"
#elif defined( USE_ASM )
 #include "asmodel.hpp"
#endif

#include "ffunc.hpp"
#include "polimage.hpp"
#include "slift.hpp"

namespace mc
{
template <typename T>
T ReLU
( T const& x )
{
  return Op<T>::max( x, T(0.) );
}
#if defined( USE_ISM ) or defined( USE_MCISM )
template <typename T>
ISVar<T> ReLU
( ISVar<T> const& x )
{
  return relu( x );
}
#elif defined( USE_ASM )
template <typename T>
ASVar<T> ReLU
( ASVar<T> const& x )
{
  return relu( x );
}
#endif
template <typename T>
fadbad::F<T> ReLU
( fadbad::F<T> const& x )
{
  fadbad::F<T> z = ReLU( x.val() );
  z.setDepend( x );
  for( unsigned j=0; j<z.size(); ++j )
    z[j] = Op<T>::fstep( x.val() ) * x[j];
  return z;
}

class ReLUANNOp
: public FFOp
{
public:

#if defined( USE_MC )
  static std::vector<McCormick<I>> MCVar;
  static std::vector<McCormick<I>> MCRes;

#elif defined( USE_ISM )
  static ISModel<I>* ISMEnv;
  static std::vector<ISVar<I>> ISMVar;
  static std::vector<ISVar<I>> ISMRes;
  static std::vector<std::vector<PolVar<I>>> POLISMAux;
  static std::vector<double> DLISMAux;
  static std::vector<double> DUISMAux;

#elif defined( USE_MCISM )
  static ISModel<I>* ISMEnv;
  static std::vector<ISVar<I>> ISMVar;
  static std::vector<ISVar<I>> ISMRes;
  static std::vector<std::vector<PolVar<I>>> POLISMAux;
  static std::vector<double> DLISMAux;
  static std::vector<double> DUISMAux;
  static std::vector<McCormick<ISVar<I>>> MCISMVar;
  static std::vector<McCormick<ISVar<I>>> MCISMRes;

#elif defined( USE_ASM )  
  static ASModel<I>* ASMEnv;
  static std::vector<ASVar<I>>  ASMVar;
  static std::vector<ASVar<I>>  ASMRes;
  static std::vector<PolVar<I>> POLLASMAux;
  static std::vector<PolVar<I>> POLUASMAux;
  static std::vector<double> DXASMAux;
  static std::vector<double> DYASMAux;
#endif

  static std::vector<std::vector<std::vector<double>>> MLP;

  template <typename T>
  T MLPeval
    ( unsigned const nx, T const* x )
    const
    {
      assert( nx == options.ANNVAR );
      static std::vector<std::vector<T>> val;
      val.resize( MLP.size() );
      //std::cout << "Layers: " << val.size() << std::endl;

      for( unsigned l=0; l<MLP.size(); ++l ){
        assert( MLP[l].size() ); // number of neurons in layer l+1
        val[l].resize( MLP[l].size() );
        //std::cout << "Neurons in layer " << l << ": " << val[l].size() << std::endl;
        for( unsigned i=0; i<val[l].size(); ++i ){
          val[l][i] = MLP[l][i][0];
          //std::cout << "Inputs to neuron " << i << " in layer " << l << ": " << MLP[l][i].size()-1 << std::endl;
          for( unsigned j=0; j<MLP[l][i].size()-1; ++j ){
            //std::cout << "layer:" << l << " neuron:" << i << " input:" << j << std::endl;
            if( std::fabs(MLP[l][i][1+j]) < machprec() ) continue;
            val[l][i] += MLP[l][i][1+j] * (l? val[l-1][j]: x[j]);
          }
          if( l+1<MLP.size() ) val[l][i] = ReLU( val[l][i] );
        }
      }
      
      assert( MLP.back().size() == 1 );
      return val.back().back();
    }

  //! @brief ReLUANNOp options
  static struct Options
  {
    //! @brief Constructor
    Options():
      ANNVAR(0), ISMDIV(64), ASMBPS(8), ISMCONT(true), ISMSLOPE(true), ISMSHADOW(true), CUTSHADOW(false)
      {}

    //! @brief Number of variables in ANN
    unsigned ANNVAR;
    //! @brief Number of subdivisions in superposition model
    unsigned ISMDIV;
    //! @brief Number of ??? in superposition model   
    unsigned ASMBPS;
    //! @brief Whether to construct continuous or binary relaxation of superposition model
    bool     ISMCONT;
    //! @brief Whether to propagate slopes in superposition model
    bool     ISMSLOPE;
    //! @brief Whether to propagate shadow remainders in superposition model
    bool     ISMSHADOW;
    //! @brief Whether to append cuts from shadow remainders in superposition model relaxation
    bool     CUTSHADOW;
  } options;

  //! @brief Sizing
  void set_data
    ( std::vector<std::vector<std::vector<double>>> const& MLP, unsigned const NDIV )
    {
      assert( MLP.size() && MLP[0].size() && MLP[0][0].size() > 1 );
      unsigned const NVAR = MLP[0][0].size() - 1; 
      options.ANNVAR = NVAR;
      options.ISMDIV = NDIV;
      this->MLP = MLP;

#if defined( USE_MC )
      MCVar.resize( NVAR );
      MCRes.resize( 1 );

#elif defined( USE_ISM )
      if( ISMEnv ) delete ISMEnv;
      ISMEnv = new ISModel<I>( NVAR, NDIV );
      ISMVar.resize( NVAR );
      ISMRes.resize( 1 );
      POLISMAux.resize( NVAR );
      DLISMAux.resize( NDIV );
      DUISMAux.resize( NDIV );

#elif defined( USE_MCISM )
      if( ISMEnv ) delete ISMEnv;
      ISMEnv = new ISModel<I>( NVAR, NDIV );
      POLISMAux.resize( NVAR );
      DLISMAux.resize( NDIV );
      DUISMAux.resize( NDIV );
      MCISMVar.resize( NVAR );
      MCISMRes.resize( 1 );

#elif defined( USE_ASM )
      if( ASMEnv ) delete ASMEnv;
      ASMEnv = new ASModel<I>( NVAR, NDIV );
      ASMVar.resize( NVAR );
      ASMRes.resize( 1 );
      POLLASMAux.resize( NVAR );
      POLUASMAux.resize( NVAR );
#endif
    }

  //! @brief Constructors
  ReLUANNOp
    ()
    : FFOp( (int)EXTERN )
    {}

  // Functor
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      assert( nVar == options.ANNVAR );
      auto dep = FFDep();
      for( unsigned i=0; i<nVar; ++i ) dep += pVar[i].dep();
      dep.update( FFDep::TYPE::N );
      return *(insert_external_operation( *this, 1, dep, nVar, pVar )[0]);
    }

  // Evaluation overloads
  template <typename T>
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__RELUANNOP_TRACE
      std::cout << "ReLUANNOp::eval generic instantiation\n"; 
      std::cout << typeid( vRes[0] ).name() << std::endl;
#endif
      assert( nVar == options.ANNVAR && nRes == 1 );
      vRes[0] = MLPeval( nVar, vVar );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__RELUANNOP_TRACE
      std::cout << "ReLUANNOp::eval FFVar instantiation\n"; 
#endif
      assert( nVar == options.ANNVAR && nRes == 1 );
      vRes[0] = operator()( nVar, vVar );
    }
    
  void eval
    ( unsigned const nRes, SLiftVar* vRes, unsigned const nVar, SLiftVar const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__RELUANNOP_TRACE
      std::cout << "ReLUANNOp::eval SLiftVar instantiation\n"; 
#endif
      assert( nVar == options.ANNVAR && nRes == 1 );
      vVar->env()->lift( nRes, vRes, nVar, vVar );
    }

  void eval
    ( unsigned const nRes, PolVar<I>* vRes, unsigned const nVar, PolVar<I> const* vVar,
      unsigned const* mVar )
    const
    {
#ifdef MC__RELUANNOP_TRACE
      std::cout << "ReLUANNOp::eval PolVar<I> instantiation\n"; 
#endif
      assert( nVar == options.ANNVAR && nRes == 1 );
      PolBase<I>* img = vVar[0].image();
      FFBase* dag = vVar[0].var().dag();
      assert( img && dag );
      FFVar* pRes = dag->curOp()->varout[0];

#if defined( USE_MC )
      //for( unsigned i=0; i<nVar; ++i )
      //  IVar[i] = vVar[i].range();
      //IRes[0] = MLPeval( nVar, IVar.data() );
      //vRes[0].set( img, *pRes, IRes[0] );
      // compute McCormick relaxation at mid-point with subgradient in each direction
      for( unsigned i=0; i<nVar; ++i )
        MCVar[i] = McCormick<I>( vVar[i].range(), Op<I>::mid( vVar[i].range() ) ).sub( nVar, i );
      MCRes[0] = MLPeval( nVar, MCVar.data() );
      //std::cout << "MCRes[0] in " << MCRes[0] << std::endl;
      vRes[0].set( img, *pRes, MCRes[0].I() );
      //std::cout << "vRes[0] in " << vRes[0].range() << std::endl;

#elif defined( USE_ISM )  
      // evaluate interval superposition
      ISMEnv->options.SLOPE_USE  = options.ISMSLOPE;  
      ISMEnv->options.SHADOW_USE = options.ISMSHADOW;
      for( unsigned i=0; i<nVar; ++i )
        ISMVar[i].set( ISMEnv, i, vVar[i].range() );
      ISMRes[0] = MLPeval( nVar, ISMVar.data() );
      //std::cout << "MCRes[0] in " << ISMRes[0];
      vRes[0].set( img, *pRes, ISMRes[0].B() );
      //std::cout << "vRes[0] in " << vRes[0].range() << std::endl;
 
#elif defined( USE_MCISM )
      // compute McCormick relaxation with ISM bounds at mid-point with subgradient in each direction
      ISMEnv->options.SLOPE_USE  = options.ISMSLOPE;
      ISMEnv->options.SHADOW_USE = options.ISMSHADOW;
      for( unsigned i=0; i<nVar; ++i )
        MCISMVar[i] = McCormick<ISVar<I>>( ISVar<I>( ISMEnv, i, vVar[i].range() ), Op<I>::mid( vVar[i].range() ) ).sub( nVar, i );
      MCISMRes[0] = MLPeval( nVar, MCISMVar.data() );
      //std::cout << "MCISMRes[0] in " << MCISMRes[0] << std::endl;
      vRes[0].set( img, *pRes, MCISMRes[0].I().B() );
      //std::cout << "vRes[0] in " << vRes[0].range() << std::endl;

#elif defined( USE_ASM )
      // evaluate interval superposition
      UnivarPWLE<double>::nbpsMax = options.ASMBPS;       
      ASMEnv->options.SLOPE_USE    = options.ISMSLOPE;
      ASMEnv->options.SHADOW_USE   = options.ISMSHADOW;
      //std::cout << "  evaluating" << std::endl;
      for( unsigned i=0; i<nVar; ++i ){
//        std::cout << std::scientific << std::setprecision(12) << mc::Op<I>::l(vVar[i].range()) << std::endl;
//        std::cout << std::scientific << std::setprecision(12) << mc::Op<I>::u(vVar[i].range()) << std::endl;        
        ASMVar[i].set( ASMEnv, i, vVar[i].range() );
        //std::cout << "ASMVar[" << i << "] in " << ASMVar[i];
      }
      ASMRes[0] = MLPeval( nVar, ASMVar.data() );
//      std::cout << "ASMRes[0] in " << ASMRes[0];
      //{int dum; std::cout << "PAUSED, ENTER 1"; std::cin >> dum;}
      //std::cout << "Active shadow: " << ASMRes[0].get_shadow_info()[0] << ASMRes[0].get_shadow_info()[1] << std::endl;
      vRes[0].set( img, *pRes, ASMRes[0].B() );
      //std::cout << "  evaluated" << std::endl;
      //std::cout << "vRes[0] in " << vRes[0].range() << std::endl;
#endif
    }

  template< typename T >
  bool reval
    ( unsigned const nRes, T const* vRes, unsigned const nVar, T* vVar )
    const
    {
      throw std::runtime_error("Error: ReLUANNOp::eval no generic implementation\n");
    }

  bool reval
    ( unsigned const nRes, PolVar<I> const* vRes, unsigned const nVar, PolVar<I>* vVar )
    const
    {
#ifdef MC__RELUANNOP_TRACE
      std::cout << "ReLUANNOp::reval PolVar<T> instantiation\n"; 
#endif
      assert( nVar == options.ANNVAR && nRes == 1 );
      PolBase<I>* img = vVar[0].image();
      FFOp* pop = vVar[0].var().opdef().first;
      assert( img && pop );

#if defined( USE_MC )
      // polyhedral cut generation
      //std::cout << "MCRes[0] in " << MCRes[0] << std::endl;
      double rhs1 = -MCRes[0].cv(),
             rhs2 = -MCRes[0].cc();
      for( unsigned i=0; i<nVar; ++i ){
        rhs1 += MCRes[0].cvsub(i)*MCVar[i].cv();
        rhs2 += MCRes[0].ccsub(i)*MCVar[i].cc();
      }
      img->add_cut( pop, PolCut<I>::LE, rhs1, nVar, vVar, MCRes[0].cvsub(), vRes[0], -1. );
      img->add_cut( pop, PolCut<I>::GE, rhs2, nVar, vVar, MCRes[0].ccsub(), vRes[0], -1. );

#elif defined( USE_ISM )
      assert( ISMEnv->ndiv() == options.ISMDIV );
      // define auxiliary variables 
      for( unsigned i=0; i<nVar; ++i ){
        POLISMAux[i].resize( ISMEnv->ndiv() );
        for( unsigned k=0; k<ISMEnv->ndiv(); ++k )
          POLISMAux[i][k].set( img, Op<I>::zeroone(), options.ISMCONT );
      }

      // polyhedral cut generation
      //std::cout << ISMRes[0];
      auto cutF1 = *img->add_cut( pop, PolCut<I>::LE, 0., vRes[0], -1. );
      auto cutF2 = *img->add_cut( pop, PolCut<I>::GE, 0., vRes[0], -1. );
      for( unsigned i=0; i<nVar; ++i ){
        auto&& ISMi = ISMRes[0].C()[i];
        if( ISMi.empty() ) continue;
        for( unsigned k=0; k<ISMEnv->ndiv(); ++k ){
          DLISMAux[k] = Op<I>::l( ISMi[k] );
          DUISMAux[k] = Op<I>::u( ISMi[k] );
        }
        cutF1->append( ISMEnv->ndiv(), POLISMAux[i].data(), DLISMAux.data() );
        cutF2->append( ISMEnv->ndiv(), POLISMAux[i].data(), DUISMAux.data() );
      }

      // add polyhedral cuts for ISM-participating variables
      for( unsigned i=0; i<nVar; i++ ){
        if( POLISMAux[i].empty() ) continue;
        // auxiliaries add up to 1
        img->add_cut( pop, PolCut<I>::EQ, 1., ISMEnv->ndiv(), POLISMAux[i].data(), 1. );

        // link auxiliaries to model variables
        PolVar<I> POLvarL( 0. ), POLvarU( 0. );
        auto&& ISMi = ISMVar[i].C()[i];
        assert( !ISMi.empty() );
        for( unsigned k=0; k<ISMEnv->ndiv(); k++ ){
          DLISMAux[k] = Op<I>::l(ISMi[k]);
          DUISMAux[k] = Op<I>::u(ISMi[k]);
        } 
        img->add_cut( pop, PolCut<I>::LE, 0., ISMEnv->ndiv(), POLISMAux[i].data(), DLISMAux.data(), vVar[i], -1. );
        img->add_cut( pop, PolCut<I>::GE, 0., ISMEnv->ndiv(), POLISMAux[i].data(), DUISMAux.data(), vVar[i], -1. );
      }

#elif defined( USE_MCISM )
      assert( ISMEnv->ndiv() == options.ISMDIV );
      // define ISM auxiliary variables 
      for( unsigned i=0; i<nVar; ++i ){
        POLISMAux[i].resize( ISMEnv->ndiv() );
        for( unsigned k=0; k<ISMEnv->ndiv(); ++k )
          POLISMAux[i][k].set( img, Op<I>::zeroone(), options.ISMCONT );  
      }
  
      // polyhedral cut generation for ISM
      //std::cout << MCISMRes[0].I();  
      auto cutF1 = *img->add_cut( pop, PolCut<I>::LE, 0., vRes[0], -1. );
      auto cutF2 = *img->add_cut( pop, PolCut<I>::GE, 0., vRes[0], -1. );
      for( unsigned i=0; i<nVar; ++i ){
        auto&& ISMi = MCISMRes[0].I().C()[i];
        if( ISMi.empty() ) continue;
        for( unsigned k=0; k<ISMEnv->ndiv(); ++k ){
          DLISMAux[k] = Op<I>::l( ISMi[k] );
          DUISMAux[k] = Op<I>::u( ISMi[k] );
        }
        cutF1->append( ISMEnv->ndiv(), POLISMAux[i].data(), DLISMAux.data() );
        cutF2->append( ISMEnv->ndiv(), POLISMAux[i].data(), DUISMAux.data() );
      }

      // add polyhedral cuts for ISM-participating variables
      for( unsigned i=0; i<nVar; i++ ){
        if( POLISMAux[i].empty() ) continue;
        // auxiliaries add up to 1
        img->add_cut( pop, PolCut<I>::EQ, 1., ISMEnv->ndiv(), POLISMAux[i].data(), 1. );

        // link ISM auxiliaries to model variables
        PolVar<I> POLvarL( 0. ), POLvarU( 0. );
        auto&& ISMi = MCISMVar[i].I().C()[i];
        assert( !ISMi.empty() );
        for( unsigned k=0; k<ISMEnv->ndiv(); k++ ){
          DLISMAux[k] = Op<I>::l(ISMi[k]);
          DUISMAux[k] = Op<I>::u(ISMi[k]);
        }
        img->add_cut( pop, PolCut<I>::LE, 0., ISMEnv->ndiv(), POLISMAux[i].data(), DLISMAux.data(), vVar[i], -1. );
        img->add_cut( pop, PolCut<I>::GE, 0., ISMEnv->ndiv(), POLISMAux[i].data(), DUISMAux.data(), vVar[i], -1. );
      }

      // polyhedral cut generation for MC  
      //std::cout << "MCRes[0] in " << MCRes[0] << std::endl;
      double rhs1 = -MCISMRes[0].cv(),  
             rhs2 = -MCISMRes[0].cc();
      for( unsigned i=0; i<nVar; ++i ){ 
        rhs1 += MCISMRes[0].cvsub(i) * MCISMVar[i].cv();
        rhs2 += MCISMRes[0].ccsub(i) * MCISMVar[i].cc();
      }
      img->add_cut( pop, PolCut<I>::LE, rhs1, nVar, vVar, MCISMRes[0].cvsub(), vRes[0], -1. );
      img->add_cut( pop, PolCut<I>::GE, rhs2, nVar, vVar, MCISMRes[0].ccsub(), vRes[0], -1. );

#elif defined( USE_ASM )
      //img->options.BREAKPOINT_TYPE = mc::PolBase<I>::Options::BIN;//SOS2;
      //img->options.BREAKPOINT_RTOL = img->options.BREAKPOINT_ATOL = 0e0;
      // polyhedral cut generation for ASM
      switch(ASMRes[0].get_ASVar()){
        case 1:{
          // polyhedral cut generation
          //std::cout << "MCRes[0] in " << MCRes[0] << std::endl;
/*          double rhs1 =  ASMRes[0].get_cst();
                   rhs2 =  rhs1;
          double linearCoefUnd[nVar];
          double linearCoefOve[nVar];
          for( unsigned i=0; i<nVar; ++i ){
            linearCoefUnd[i] = 0.;
            linearCoefOve[i] = 0.;
          }          
          img->add_cut( pop, PolCut<I>::LE, rhs1, nVar, vVar, linearCoefUnd, vRes[0], -1. );
          img->add_cut( pop, PolCut<I>::GE, rhs2, nVar, vVar, linearCoefOve, vRes[0], -1. );*/


/*        double rhs1 =  ASMRes[0].get_cst();
          POLLASMAux[1].set( img, I(0.), true ); 
          POLUASMAux[1].set( img, I(0.), true );  
          POLLASMAux[0].set( img, I(rhs1,rhs1), true );
          POLUASMAux[0].set( img, I(rhs1,rhs1), true );  
          img->add_cut( pop, PolCut<I>::EQ, rhs1, POLUASMAux[0], 1. );
          img->add_cut( pop, PolCut<I>::EQ, rhs1, POLLASMAux[0], 1. );
          img->add_cut( pop, PolCut<I>::LE, 0., nVar, POLLASMAux.data(), 1., vRes[0], -1. );
          img->add_cut( pop, PolCut<I>::GE, 0., nVar, POLUASMAux.data(), 1., vRes[0], -1. );*/

          double rhs = ASMRes[0].get_cst() / nVar;
          for(unsigned i=0; i<nVar; ++i ){
             POLLASMAux[i].set( img, I(rhs), true );  
             POLUASMAux[i].set( img, I(rhs), true );  
             DXASMAux.assign( 2, 0. );
             DYASMAux.assign( 2, 0. );
	     DXASMAux[0] = mc::Op<I>::l(vVar[i].range());
	     DXASMAux[1] = mc::Op<I>::u(vVar[i].range());
	     DYASMAux[0] = rhs;
	     DYASMAux[1] = rhs;  
      	     img->add_semilinear_cuts( pop, 2, vVar[i], DXASMAux.data(), POLLASMAux[i], DYASMAux.data(), mc::PolCut<I>::EQ );
	     img->add_semilinear_cuts( pop, 2, vVar[i], DXASMAux.data(), POLUASMAux[i], DYASMAux.data(), mc::PolCut<I>::EQ ); 
          }

          img->add_cut( pop, PolCut<I>::LE, 0., nVar, POLLASMAux.data(), 1., vRes[0], -1. );
          img->add_cut( pop, PolCut<I>::GE, 0., nVar, POLUASMAux.data(), 1., vRes[0], -1. );


          break;    
          }
        case 2:{
          // polyhedral cut generation
          //std::cout << "MCRes[0] in " << MCRes[0] << std::endl;
/*          double rhs1 =  ASMRes[0].get_cst(),
                 rhs2 =  rhs1;
          double linearCoefUnd[nVar];
          double linearCoefOve[nVar];
          for( unsigned i=0; i<nVar; ++i ){
            linearCoefUnd[i] = ASMRes[0].get_lnr()[i];
            linearCoefOve[i] = ASMRes[0].get_lnr()[i];  
          }          
          img->add_cut( pop, PolCut<I>::LE, rhs1, nVar, vVar, linearCoefUnd, vRes[0], -1. );
          img->add_cut( pop, PolCut<I>::GE, rhs2, nVar, vVar, linearCoefOve, vRes[0], -1. );*/
          double rhs = ASMRes[0].get_cst() / nVar;
          for(unsigned i=0; i<nVar; ++i ){
             auto tmp_yrange = ( ASMRes[0].get_lnr()[i] * vVar[i].range() ) + rhs; 
             POLLASMAux[i].set( img, tmp_yrange, true );
             POLUASMAux[i].set( img, tmp_yrange, true );  
             DXASMAux.assign( 2, 0. );
             DYASMAux.assign( 2, 0. );
	     DXASMAux[0] = mc::Op<I>::l(vVar[i].range());
	     DXASMAux[1] = mc::Op<I>::u(vVar[i].range());
	     DYASMAux[0] = DXASMAux[0]*ASMRes[0].get_lnr()[i]+rhs;
	     DYASMAux[1] = DXASMAux[1]*ASMRes[0].get_lnr()[i]+rhs;
      	     img->add_semilinear_cuts( pop, 2, vVar[i], DXASMAux.data(), POLLASMAux[i], DYASMAux.data(), mc::PolCut<I>::EQ );
	     img->add_semilinear_cuts( pop, 2, vVar[i], DXASMAux.data(), POLUASMAux[i], DYASMAux.data(), mc::PolCut<I>::EQ ); 
          }

          img->add_cut( pop, PolCut<I>::LE, 0., nVar, POLLASMAux.data(), 1., vRes[0], -1. );
          img->add_cut( pop, PolCut<I>::GE, 0., nVar, POLUASMAux.data(), 1., vRes[0], -1. );
          
          break;
          }
        case 3:{
          append_ASMcuts( nRes, vRes, nVar, vVar, img, pop, ASMRes[0].get_lst() );          
          break;
          }
        case 4:{
          append_ASMcuts( nRes, vRes, nVar, vVar, img, pop, ASMRes[0].get_lst() );
          if( options.CUTSHADOW && options.ISMSHADOW )
            append_ASMcuts( nRes, vRes, nVar, vVar, img, pop, ASMRes[0].get_shadow() );
          break;
          }
        default:{
          std::cout << "ERROR in getting ASMRes[0].get_ASVar()" << std::endl;
          break;
          }
      }  

      //std::cout << *img;
      //{int dum; std::cout << "PAUSED, ENTER 1"; std::cin >> dum;}
#endif
      return true;
    }

#if defined( USE_ASM )
  void append_ASMcuts
    ( unsigned const nRes, PolVar<I> const* vRes, unsigned const nVar, PolVar<I>* vVar,
      PolBase<I>* img, FFOp* pop, std::vector<UnivarPWL<I>> const& pwlEst )
    const
    {
      for( unsigned i=0; i<nVar; ++i ){
        UnivarPWLE<double> const& uest = pwlEst[i].undEst;
        if( uest.empty() )
          POLLASMAux[i].set( img, I(0.), true );
        else{ 
          POLLASMAux[i].set( img, I(uest.get_lb(),uest.get_ub()), true );
          auto const [ucst,isuCst] = uest.get_cst();
          if( isuCst )
            img->add_cut( pop, PolCut<I>::EQ, ucst, POLLASMAux[i], 1. );
          else{
            unsigned NK = uest.first.size()-1;
            assert( uest.second.size() == uest.first.size() );
            if(NK==1){
              NK += 1;
              DXASMAux.assign( NK, 0. );
              DYASMAux.assign( NK, 0. );
	      for( unsigned j=0; j<NK; ++j ){
	        DXASMAux[j] = uest.first[j];
	        DYASMAux[j] = uest.second[j];
	      }            
            }
            else{
              DXASMAux.assign( NK, uest.first[0] );
              DYASMAux.assign( NK, uest.second[0] );
	      for( unsigned j=0; j<NK; ++j ){
	        DXASMAux[j] += uest.first[j+1];
	        DYASMAux[j] += uest.second[j+1];
	      }
	    }
	    img->add_semilinear_cuts( pop, NK, vVar[i], DXASMAux.data(), POLLASMAux[i], DYASMAux.data(), mc::PolCut<I>::EQ );
	  }
	}
        UnivarPWLE<double> const& oest = pwlEst[i].oveEst;
        if( oest.empty() )
          POLUASMAux[i].set( img, I(0.), true );
        else{
          POLUASMAux[i].set( img, I(oest.get_lb(),oest.get_ub()), true );
          auto const [ocst,isoCst] = oest.get_cst();
          if( isoCst )
            img->add_cut( pop, PolCut<I>::EQ, ocst, POLUASMAux[i], 1. );
          else{
            unsigned NK = oest.first.size()-1;
            assert( oest.second.size() == oest.first.size() );
            if(NK == 1){
              NK += 1;
              DXASMAux.assign( NK, 0. );
              DYASMAux.assign( NK, 0. );
	      for( unsigned j=0; j<NK; ++j ){
	        DXASMAux[j] = oest.first[j];
	        DYASMAux[j] = oest.second[j];
	      }
            }  
            else{  
              DXASMAux.assign( NK, oest.first[0] );
              DYASMAux.assign( NK, oest.second[0] );
	      for( unsigned j=0; j<NK; ++j ){
	        DXASMAux[j] += oest.first[j+1];
	        DYASMAux[j] += oest.second[j+1];
	      }
	    }	      
	    img->add_semilinear_cuts( pop, NK, vVar[i], DXASMAux.data(), POLUASMAux[i], DYASMAux.data(), mc::PolCut<I>::EQ );
	  }
	}
      }
      img->add_cut( pop, PolCut<I>::LE, 0., nVar, POLLASMAux.data(), 1., vRes[0], -1. );
      img->add_cut( pop, PolCut<I>::GE, 0., nVar, POLUASMAux.data(), 1., vRes[0], -1. );

    } 
#endif
  // Properties
  std::string name
    ()
    const
    { return "ReLUANN"; }
};

#if defined( USE_MC )
  inline std::vector<McCormick<I>> ReLUANNOp::MCVar = std::vector<McCormick<I>>();
  inline std::vector<McCormick<I>> ReLUANNOp::MCRes = std::vector<McCormick<I>>( 1 );

#elif defined( USE_ISM )
  inline ISModel<I>* ReLUANNOp::ISMEnv = nullptr;
  inline std::vector<ISVar<I>> ReLUANNOp::ISMVar = std::vector<ISVar<I>>();
  inline std::vector<ISVar<I>> ReLUANNOp::ISMRes = std::vector<ISVar<I>>( 1 );
  inline std::vector<std::vector<PolVar<I>>> ReLUANNOp::POLISMAux = std::vector<std::vector<PolVar<I>>>();
  inline std::vector<double> ReLUANNOp::DLISMAux = std::vector<double>();
  inline std::vector<double> ReLUANNOp::DUISMAux = std::vector<double>();

#elif defined( USE_MCISM )
  inline ISModel<I>* ReLUANNOp::ISMEnv = nullptr;
  inline std::vector<std::vector<PolVar<I>>> ReLUANNOp::POLISMAux = std::vector<std::vector<PolVar<I>>>();
  inline std::vector<double> ReLUANNOp::DLISMAux = std::vector<double>();
  inline std::vector<double> ReLUANNOp::DUISMAux = std::vector<double>();
  inline std::vector<McCormick<ISVar<I>>> ReLUANNOp::MCISMVar = std::vector<McCormick<ISVar<I>>>();
  inline std::vector<McCormick<ISVar<I>>> ReLUANNOp::MCISMRes = std::vector<McCormick<ISVar<I>>>( 1 );

#elif defined( USE_ASM )
  inline ASModel<I>* ReLUANNOp::ASMEnv = nullptr;
  inline std::vector<ASVar<I>> ReLUANNOp::ASMVar = std::vector<ASVar<I>>();
  inline std::vector<ASVar<I>> ReLUANNOp::ASMRes = std::vector<ASVar<I>>( 1 );
  inline std::vector<PolVar<I>> ReLUANNOp::POLLASMAux = std::vector<PolVar<I>>();
  inline std::vector<PolVar<I>> ReLUANNOp::POLUASMAux = std::vector<PolVar<I>>();
  inline std::vector<double> ReLUANNOp::DXASMAux = std::vector<double>();
  inline std::vector<double> ReLUANNOp::DYASMAux = std::vector<double>();
#endif

  inline std::vector<std::vector<std::vector<double>>> ReLUANNOp::MLP = std::vector<std::vector<std::vector<double>>>();
  inline ReLUANNOp::Options ReLUANNOp::options = ReLUANNOp::Options();

} // end namespace mc

#endif
