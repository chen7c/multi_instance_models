from setuptools import setup

setup(name='multiinstance',
      version='0.1',
      description='Multi-instance logistic regression using Theano and Lasagne.',
      url='http://github.com/matted/multi_instance_models',
      author='Matt Edwards',
      author_email='matted@mit.edu',
      license='MIT',
      packages=['multiinstance'],
      zip_safe=True,
      install_requires=["numpy"], # TODO: we require dev versions of lasagne and theano, see requirements.txt
)
