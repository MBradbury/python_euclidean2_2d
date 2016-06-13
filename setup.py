from distutils.core import setup, Extension
setup(name="euclidean", version="1.0.1",
	  description="Provides a fast implementation of euclidean distance",
	  author="Matthew Bradbury",
	  license="MIT",
      ext_modules=[Extension("euclidean", ["src/euclidean_module.c"])])
