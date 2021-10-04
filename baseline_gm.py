import torch
import torch.nn.functional as F
from model_gm import get_model_from_name
import routines_gm as routines

def get_avg_parameters(networks, weights=None):

    '''
        given a series of networks (of the same shape) stored in the argument
        <networks>, return a list containing all the parameters averaged over
        all the networks with the weights specified by <weights>

        "*" means to unpack. For example, 
        f( *[1, 2, 3] )                   = f( 1, 2, 3 ); 
        f( **{'a': 1, 'b': 2, 'c': 3} )   = f( 1, 2, 3 )

        zip function:
        zip( list_1, list_2, list_3 ) = 
        zip(
            (list_1[0], list_2[0], list_3[0]),
            ...
            (list_1[n], list_2[n], list_3[n])
        )
        where n is the least lenth of three lists

        <par_group> means a set of parameters at a specific position from all networks

        <avg_par> is the average parameter at that specific position
    '''

    avg_pars = []
    for par_group in zip(*[net.parameters() for net in networks]):  
        # Alexanderia
        # print([par.shape for par in par_group])
        if weights is not None:
            weighted_par_group = [par * weights[i] for i, par in enumerate(par_group)]
            avg_par = torch.sum(torch.stack(weighted_par_group), dim=0)
        else:
            # print("shape of stacked params is ", torch.stack(par_group).shape) # (2, 400, 784)
            avg_par = torch.mean(torch.stack(par_group), dim=0)
        # Alexanderia
        # print(avg_par.shape)
        avg_pars.append(avg_par)
    return avg_pars

def naive_ensembling(args, networks, test_loader):

    '''
        test the prediction accuracy of naive ensembling method

        naive ensembling method: combine two models to a single one by simply 
            averaging their parameters at the same position

        Net.state_dict(): 
            a dictionary that maps each layer to its parameters
        
        Net.parameters(): 
            an array-like data structure that stores the parameters of each layer
    '''

    # simply average the weights in networks
    if args.width_ratio != 1:
        print("Unfortunately naive ensembling can't work if models are not of same shape!")
        return -1, None
    weights = [(1-args.ensemble_step), args.ensemble_step]
    avg_pars = get_avg_parameters(networks, weights)
    ensemble_network = get_model_from_name(args)
    # put on GPU
    if args.gpu_id!=-1:
        ensemble_network = ensemble_network.cuda(args.gpu_id)

    # check the test performance of the method before
    log_dict = {}
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]
    # Alexanderia
    # routines.test(args, ensemble_network, test_loader, log_dict)

    # set the weights of the ensembled network
    for idx, (name, param) in enumerate(ensemble_network.state_dict().items()):
        ensemble_network.state_dict()[name].copy_(avg_pars[idx].data)

    # check the test performance of the method after ensembling
    log_dict = {}
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]
    return routines.test(args, ensemble_network, test_loader, log_dict), ensemble_network


def prediction_ensembling(args, networks, test_loader):

    '''
        calculate the prediction accuracy of the traditional ensembling methods

        traditional ensembling method: keep two models and directly average their 
            output results after inferences from each of them
    '''

    log_dict = {}
    log_dict['test_losses'] = []
    # test counter is not even used!
    # log_dict['test_counter'] = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]

    if args.dataset.lower() == 'cifar10':
        cifar_criterion = torch.nn.CrossEntropyLoss()

    # set all the networks in eval mode
    for net in networks:
        net.eval()
    test_loss = 0
    correct = 0

    #   with torch.no_grad():
    for data, target in test_loader:
        if args.gpu_id!=-1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)
        outputs = []
        # average the outputs of all nets
        assert len(networks) == 2
        if args.prediction_wts:
            wts = [(1 - args.ensemble_step), args.ensemble_step]
        else:
            wts = [0.5, 0.5]
        for idx, net in enumerate(networks):
            outputs.append(wts[idx]*net(data))
        #  print("number of outputs {} and each is of shape {}".format(len(outputs), outputs[-1].shape))
        #  number of outputs 2 and each is of shape torch.Size([1000, 10])
        output = torch.sum(torch.stack(outputs), dim=0) # sum because multiplied by wts above

        #  check loss of this ensembled prediction
        if args.dataset.lower() == 'cifar10':
            # mnist models return log_softmax outputs, while cifar ones return raw values!
            test_loss += cifar_criterion(output, target).item()
        elif args.dataset.lower() == 'mnist':
            test_loss += F.nll_loss(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    log_dict['test_losses'].append(test_loss)


    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


    return (float(correct) * 100.0)/len(test_loader.dataset)