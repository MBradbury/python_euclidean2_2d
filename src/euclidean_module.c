#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>

// Starting point https://codereview.stackexchange.com/questions/52218/possible-optimizations-for-calculating-squared-euclidean-distance

/* Docstrings */
static const char module_docstring[] =
    "This module provides an interface for calculating the euclidean distance";
static const char euclidean_docstring[] =
    "Calculate the euclidean distance of two 1-dimensional vectors with 2 elements (2d points)";

static double euclidean2_2d(const double* left, const double* right)
{
    const double x1 = left[0];
    const double y1 = left[1];
    const double x2 = right[0];
    const double y2 = right[1];

    const double xdiff = x1 - x2;
    const double ydiff = y1 - y2;

    const double dist = sqrt(xdiff*xdiff + ydiff*ydiff);

    return dist;
}

static PyObject* euclidean_euclidean2_2d(PyObject* self, PyObject* args)
{
    PyObject *coord1, *coord2;

    /* Parse the input tuple manually */
    if (PyTuple_GET_SIZE(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "requires 2 arguments");
        return NULL;
    }

    coord1 = PyTuple_GET_ITEM(args, 0);
    coord2 = PyTuple_GET_ITEM(args, 1);

    if (!PyArray_Check(coord1) || !PyArray_Check(coord2)) {
        PyErr_SetString(PyExc_TypeError, "requires array arguments");
        return NULL;
    }

    if (PyArray_NDIM(coord1) != 1 || PyArray_NDIM(coord2) != 1) {
        PyErr_SetString(PyExc_TypeError, "requires 1d arrays");
        return NULL;
    }

    /* Get pointers to the data as C-types. */
    const double* x = (double*)PyArray_DATA(coord1);
    const double* y = (double*)PyArray_DATA(coord2);

    /* Call the external C function to compute the distance. */
    const double value = euclidean2_2d(x, y);

    if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError, "Euclidean returned an impossible value.");
        return NULL;
    }
 
    PyObject *ret = PyFloat_FromDouble(value);
    return ret;
}

// Methods in the module
static PyMethodDef module_methods[] = {
    {"euclidean2_2d", euclidean_euclidean2_2d, METH_VARARGS, euclidean_docstring},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "euclidean",     /* m_name */
    module_docstring,  /* m_doc */
    -1,                  /* m_size */
    module_methods,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};
#endif

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#   define PyMODINIT_FUNC void
#endif

static PyObject * euclidean_module_init(void)
{
    PyObject* m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("euclidean", module_methods, module_docstring);
#endif

    import_array();

    return m;
}


#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC initeuclidean(void)
    {
        euclidean_module_init();
    }
#else
    PyMODINIT_FUNC PyInit_euclidean(void)
    {
        return euclidean_module_init();
    }
#endif
