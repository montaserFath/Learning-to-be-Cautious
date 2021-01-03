# Learning-to-be-Cautious
Learning to be Cautious


[Notebook](https://github.com/montaserFath/Learning-to-be-Cautious/blob/main/Learning_to_be_Cautious.ipynb) is a single notebook including:

    1- Loading datasets (MNIST, Fashion-MNIST, and E-MNIST) and converting it to a multi-armed bandit setting (convert labels to arms or actions).
   
    2- Train Deep Ensemble for be a reward distribution to caputure the epistemic uncertainty (train number of Neural Neworks with the same training data but with different initialization for the networks).

    3- Approximate Percentile Optimization with k-of-n game. to get a robust policy.
    
    4- Show different robust policies' behavior in training and out-of-distribution data and compare it with normal RL.
    
[Script](https://github.com/montaserFath/Learning-to-be-Cautious/blob/main/k_of_n.py) is a script for runing k-of-n game to get robust policies in a multi-armed bandit setting.
