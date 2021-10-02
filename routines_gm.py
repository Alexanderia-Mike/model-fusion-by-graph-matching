import torch
from model_gm import get_model_from_name

def get_pretrained_model(args, path, data_separated=False, idx=-1):
    model = get_model_from_name(args, idx=idx)

    if args.gpu_id != -1:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cuda:' + str(args.gpu_id))
            ),
        )
    else:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )


    model_state_dict = state['model_state_dict']

    if 'test_accuracy' not in state:
        state['test_accuracy'] = -1

    if 'epoch' not in state:
        state['epoch'] = -1

    if not data_separated:
        print("Loading model at path {} which had accuracy {} and at epoch {}".format(path, state['test_accuracy'],
                                                                                  state['epoch']))
    else:
        print("Loading model at path {} which had local accuracy {} and overall accuracy {} for choice {} at epoch {}".format(path,
            state['local_test_accuracy'], state['test_accuracy'], state['choice'], state['epoch']))

    model.load_state_dict(model_state_dict)

    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)

    if not data_separated:
        return model, state['test_accuracy']
    else:
        return model, state['test_accuracy'], state['local_test_accuracy']