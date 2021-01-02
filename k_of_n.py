import torch
import torchvision
import numpy as np
import torch.nn as nn
from torchvision import transforms as transforms
from IPython import display

relu = nn.ReLU()
def sort_and_k_least (a, k):
    '''
    sort a tensor from small value to big value and select lowest k values
    Inputs:
    a: (tesnor) tensor we want to sort it
    k: (int) size of lowest values
    return:
    numpy array (shape (k, ))
    '''
    _, index = torch.sort(a,  descending=False)
    return index[:k].numpy()
def sample_rewards_from_ensemble (method, n_samples, states, actions, models_numbers, models_dir, batch_size, device):
    '''
    Sample reward functions from deep ensamble and calculate the expected value of the policy given current given
    Args:
    method: (str) "test" if you want to play k-of-n a sample by sample in dataset, "TEST" play k-of-n for full datatset
    n_samples: (int) number of samples 
    states: (Tesnsor) states which the policy will evaluate on it
    actions:(Tesnsor) policy
    models_numbers:(list or 1d numpy array) models number you want to sample from it
    models_dir: (str) diroctory path where trained models saved
    batch_size: (int) batch size
    device: (str) device "cuda:0" or "cuda:1" or "cpu"
    return:
    expected_rewards:(Tensor shape (n_samples, ) ) expected value for each sample
    n_rs: (Tensor shape (n_samples, states_size, action_size) ) reward for each sample and each state
    ms: (1d numpy array) sampled models number
    '''
    
    ms = np.random.choice(models_numbers, n_samples)
    n_rs = torch.zeros((n_samples, states.shape[0], actions.shape[1]))

    if method == "TEST":
        estimated_rewards = torch.zeros((n_samples,))
    else:
        estimated_rewards = torch.zeros((n_samples, states.shape[0]))
    
    for s in range (n_samples):
        model = torch.load(models_dir+"/ensemble_model_{}".format(ms[s])).to(device)
        for batch in range (0, states.shape[0] , batch_size):
            n_rs[s, batch : batch+batch_size] = (model(states[batch : batch+batch_size].to(device))).detach().cpu()
        if method == "TEST":
            estimated_rewards[s]  = torch.sum(n_rs[s]*actions).item()
        else:
            estimated_rewards[s]  = torch.sum(n_rs[s]*actions, 1)

    return estimated_rewards, n_rs, ms, np.setdiff1d(models_numbers, ms)

    
def check_dataset (dataset):
    """
    Load dataset 
    dataset: (str) "E-MNIST"  or "E-MNIST", "MNIST-Fashion" 
    return:
    state: (tensor) dataset output
    """
    transform = transforms.ToTensor()
    
    if dataset == "MNIST":
        mnist_test = torchvision.datasets.MNIST('datasets', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(mnist_test,  shuffle=False)
        state = torch.zeros((len(testloader) , 1, 28, 28))
#         testing_set_y = torch.zeros(len(testloader))
        for i, data in enumerate (testloader):
            img, label = data
            state[i] = img.view(1,28,28)
#             testing_set_y[i] = label

    elif dataset == "E-MNIST":
        emnist_test  = torchvision.datasets.EMNIST(root="datasets", train=False, transform=transform, target_transform=None, download=True, split="letters")
        state = torch.zeros((len(emnist_test), 1, 28, 28))
        for i in  range (len(emnist_test)):
            state[i]  = torch.transpose(emnist_test[i][0], 1,-1).view(1, 28,28)

    elif dataset == "MNIST-Fashion":
        fashion_mnist_test = torchvision.datasets.FashionMNIST(root="datasets", train=False, transform=transform, target_transform=None, download=True)
        state = torch.zeros((len(fashion_mnist_test), 1, 28, 28), device="cpu")
        for i in  range (len(fashion_mnist_test)):
            state[i]  = fashion_mnist_test[i][0].view(1, 28,28)

    else:
        raise ValueError("dataset shoud be MNIST, E-MNIST or Fashion-MNIST")
        
    return state
def run_k_of_n (ks, ns, n_runs, n_itr, method, n_models,batch_size, models_dir, output_policies_dir, device, dataset, n_actions):
    '''
    Run k-of-n
    Inputs:
    ks: (list) k values
    ns: (list) n values
    n_runs: (int) how many times you want to repeat each k-of-n policy
    n_itr: (int) number of itration for k-of-n game
    method: (str) "test" if you want to play k-of-n a sample by sample in dataset, "TEST" play k-of-n for full datatset
    n_models: (int) number of models in Enamble
    batch_size: (int) batch size
    models_dir: (str) diroctory path where deep models have been saved
    output_policies_dir: (str) diroctory path where you want to save k-of-n policies
    device: (str) device "cuda:0" or "cuda:1" or "cpu"
    dataset: (str) "MNIST" or "E-MNIST"
    '''
    state = check_dataset (dataset)
    for i in range (len(ks)):
        k = ks[i]
        n = ns[i]
        for run in range (n_runs):
            expexted_value = np.zeros(n_itr)
            actions = torch.softmax(torch.ones((state.shape[0], n_actions), device="cpu"), dim=1 )
            total_regret = torch.zeros((actions.shape[0] , actions.shape[1]), device="cpu")
            models_numbers = np.arange(0, n_models, 1,dtype=np.int)
            for itr in range (n_itr):
                n_estimated_rewards, n_rs, mss , models_numbers = sample_rewards_from_ensemble (method, n,  state, actions, models_numbers, models_dir, batch_size, device)
                if method=="TEST":
                    k_index = sort_and_k_least(n_estimated_rewards, k)
                    expexted_value[itr] = n_estimated_rewards[k_index].sum()
                    mean_rs  = torch.mean(n_rs[k_index], 0)
                    P_t = mean_rs - torch.mm(torch.sum(actions*mean_rs, 1, dtype=torch.float).view(-1, 1), torch.ones((1, actions.shape[1]), dtype=torch.float))
                    total_regret += P_t

                else:
                    for s in range (state.shape[0]):
                        k_index = sort_and_k_least(n_estimated_rewards[:, s], k)
                        expexted_value[itr] += n_estimated_rewards[k_index, s].sum()
                        a = n_rs[:, s]
                        mean_rs  = torch.mean(a[k_index], 0)
                        P_t = mean_rs - torch.sum(mean_rs*actions[s])*torch.ones((1, actions.shape[1]))
                        total_regret[s] += P_t.view(-1)

                actions = relu(total_regret)/(torch.ones_like(total_regret)*torch.sum(relu(total_regret), 1).view(-1, 1))
                print("{}-of-{} run no {} itration number {}".format(k, n, run, itr))
                display.clear_output(wait=True)
                if dataset == "MNIST":
                    np.save(output_policies_dir+"/run_{}_mnist_actions_{}-of-{}_n_itr_{}".format(run, k,n, n_itr), actions.numpy())
                    np.save(output_policies_dir+"/run_{}_expected_value_mnist_{}-of-{}_n_itr_{}".format(run, k,n, n_itr), expexted_value)
                elif dataset == "E-MNIST":
                    np.save(output_policies_dir+"/run_{}_emnist_actions_{}-of-{}_n_itr_{}".format(run, k,n, n_itr), actions.numpy())
                    np.save(output_policies_dir+"/run_{}_expected_value_emnist_{}-of-{}_n_itr_{}".format(run, k,n, n_itr), expexted_value)
                elif dataset == "MNIST-Fashion":
                    np.save(output_policies_dir+"/run_{}_fashion_actions_{}-of-{}_n_itr_{}".format(run, k,n, n_itr), actions.numpy())
                    np.save(output_policies_dir+"/run_{}_expected_value_fashion_{}-of-{}_n_itr_{}".format(run, k,n, n_itr), expexted_value)
                else:
                    raise ValueError("dataset shoud be MNIST, E-MNIST or Fashion-MNIST")