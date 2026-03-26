from setuptools import setup, find_packages

setup(
    name='ia_futbol',
    version='0.1.0',
    description='Proyecto de IA aplicado al fútbol',
    # Busca todos los paquetes dentro de la carpeta 'src'
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)