import setuptools

setuptools.setup(
    name="indico_tf_ops",
    version="0.0.0",
    description="Some tensorflow ops, all kept in one convenient place",
    maintainer="Ben Townsend",
    install_requires=[
        'numpy',
        'scipy',
        # tensorflow-gpu or cpu=1.14.0
    ],
    packages=setuptools.find_packages(),
    package_data={"indico_tf_ops": ["libindico_kernels.so"]}
)
