#include <pybind11/pybind11.h>

namespace py = pybind11;

void mc_nlpslv( py::module_ & );

PYBIND11_MODULE( canon, m )
{

  m.doc() = "Python interface of library CANON";

  mc_nlpslv( m );

}

