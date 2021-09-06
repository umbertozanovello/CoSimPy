import setuptools

with open(".//docs//Documentation.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(name="cosimpy", version="1.2.0", licence="MIT", url='https://github.com/umbertozanovello/CoSimPy', packages=setuptools.find_packages(), author="Umberto Zanovello", description="Python electromagnetic co-simulation library", long_description=long_description, long_description_content_type="text/markdown", classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent",], python_requires='>=3.5',)
