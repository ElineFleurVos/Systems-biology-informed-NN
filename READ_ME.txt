The code used in the paper is split in multiple files:
- analytical_solution.py --> contains a function that calculates the analytical solution using a set of model parameters
- data_generation.py --> generates the four different datasets and puts these datasets in the folder 'data_sets' !To mimic 
  the figures as shown in the paper, do not run this file. 
- SBINN_method_1.py --> implementation of method 1. Running this file is possible as new runs are saved as a test file with name 'TEST.npy'
- SBINN_method_2.py --> implementation of method 1. Running this file is possible as new runs are saved as a test file with name 'TEST.npy'
- sampling_strategy_plot.py --> plots Figure 1 from the paper with the different sampling strategies. 
- plotting_boxplots.py --> plots the Figures with boxplots found in the paper.
- statistical_analyis.py --> calculates all the p-values found across the paper and in appendix C and D. 

Finally, the folder 'results' contains the mean absolute percentage errors (MAPEs) for all the different tests done. 