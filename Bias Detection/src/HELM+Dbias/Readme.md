# HELM and Dbias Description
## _Implementation of bias detection from HELM and Dbias package_

This directory contains the implementation of HELM's bias evaluation method, including both categories: demographic and stereotype. The main.py file provides a demonstration of how to use these python files in a easy way. 


### Installation Note
Due to environmental complexity of Dbias, the required environment is more specific than the DistillRoBERTa bias detection method. Therefore use the provided requirements.txt in this folder in addition to the on in root for running Dbias.

```python
pip install requiremetns.txt
```

### Note
The Dbias wheel build raises many warnings during computation, but the bias detection is functional.

Python Notebook is recommended to test Dbias.

