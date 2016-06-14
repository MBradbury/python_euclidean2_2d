from distutils.core import setup, Extension
setup(name="euclidean", version="1.0.2",
      description="Provides a fast implementation of euclidean distance",
      author="Matthew Bradbury",
      license="MIT",
      url="https://github.com/MBradbury/python_euclidean2_2d",
      ext_modules=[Extension("euclidean", ["src/euclidean_module.c"])],
      classifiers=["Programming Language :: Python :: 2",
                   "Programming Language :: Python :: 3",
                   "Programming Language :: Python :: Implementation :: CPython"]
)
