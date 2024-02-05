# Change-Point-detection-using-Random-Forest

This project investigates the classifier-based non-parametric approach proposed by Malte Londschien, Peter Buhlmann, and Solt ¨
Kovacs (2023) [1]. The project will involve the use of random forest and k-nearest neighbor classifiers alongside a search ´
method based on binary segmentation to detect change points in both univariate and multivariate datasets. The experiments
will be conducted on datasets with single and multiple change points and then compared with benchmark results. Additionally, the project involves implementing a variation of the search algorithm paired with a neural network classifier and evaluating the results
to assess the algorithm’s feasibility with other classifiers.

## Datasets 
The following datasets are used : Iris, Abalone, Glass, Wine, a generated dataset for a change in mean drawn from a set of
normal distributions, a generated dataset for a change in variance drawn from a set of normal distributions, and a dataset drawn
from a set of Dirichlet distributions with varying parameters to replicate the results of the paper [1] used for investigation.
These datasets will be generated using the Python package NumPy. Additionally, The project will involve using five datasets from the Time
Series Segmentation Benchmark studied by Patrick Schafer, Arik Ermshaus, and Ulf Leser [2]

## References 
[1] M. Londschien, P. BA¼hlmann, and S. Kov ˜ A¡cs, “Random forests for change point detection,” ˜ Journal of Machine Learning Research, vol. 24, no. 216,
pp. 1–45, 2023.

[2] A. Ermshaus, P. Schafer, and U. Leser, “Clasp: parameter-free time series segmentation,” ¨ Data Mining and Knowledge Discovery, 2023