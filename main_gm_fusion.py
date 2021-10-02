from cifar_gm import train as cifar_train
import utils_gm
import routines_gm as routines
import os
import fusion_gm

def run_fusion():
    '''
    first load pretrained models and then compare their accuracies with the fused model
    '''
    args = utils_gm.dotdict( {
        "model_name": "mlpnet",
        "dataset": "mnist",
        "gpu_id": -1,
        "num_models": 2,

        "weight": [0.5, 0.5],
        # "model_name": "naivenet",
        "dataset": "mnist",
        "disable_bias": True,
        "width_ratio": 1,
        "num_hidden_nodes1": 400,
        "num_hidden_nodes2": 200,
        "num_hidden_nodes3": 100
    } )
    '''
    get pre-trained models from "./mnist_models"
    '''
    models = []
    accuracies = []
    for idx in range( 2 ):
        model, accuracy = routines.get_pretrained_model(
            args, os.path.join( './mnist_models', 'model_{}/final.checkpoint'.format(idx)), idx = idx )
        models.append( model )
        accuracies.append( accuracy )
    
    print( f'* model 1 has accuracy {accuracies[0]}\n* model 2 has accuracy {accuracies[1]}' )
    '''
    do the graph-matching-based fusion
    '''
    fused_model = fusion_gm.get_fused_model( args, models )
    print( fused_model )
    '''
    test the accuracy for fused_model
    '''
    

if __name__ == '__main__':
    run_fusion()