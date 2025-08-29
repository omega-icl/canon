#include <pybind11/pybind11.h>

namespace py = pybind11;

void mc_base  ( py::module_ & );
void mc_mipslv( py::module_ & );
void mc_nlpslv( py::module_ & );
void mc_minlpslv( py::module_ & );

PYBIND11_MODULE( canon, m )
{
  m.doc() = "Python interface of library CANON";

  mc_base( m );
  mc_mipslv( m );
  mc_nlpslv( m );
  mc_minlpslv( m );
}

