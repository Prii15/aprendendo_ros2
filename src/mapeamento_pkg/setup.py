from setuptools import find_packages, setup

package_name = 'mapeamento_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_to_grid = mapeamento_pkg.lidar_to_grid:main',
            'lidar_to_grid_map = mapeamento_pkg.lidar_to_grid_map:main',
        ],
    },
)