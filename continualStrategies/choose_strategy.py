from StandardStrategies.XtoC import XtoC
from StandardStrategies.XtoY import XtoY
from StandardStrategies.CtoY import CtoY
from StandardStrategies.XtoCtoY import XtoCtoY

from ..det_VAEL.models.vael_networks import DetMNISTPairsEncoder


def choose_model(experiment, **kwargs):
    ## CONCET BOTTLENECK MODELS
    if experiment == "CtoY":
        model = CtoY(optimizer_name="Adam", model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                            optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, continual=True)
    elif experiment == "XtoY":
        model = XtoY(optimizer_name="Adam", model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, continual=True)
    elif experiment == "XtoCtoY":
        model = XtoCtoY(optimizer_name="Adam", model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, continual=True)
    elif experiment == "XtoC":
        model = XtoC(optimizer_name="Adam", model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                            optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, continual=True)
                            
    ## NEURO SYMBOLIC with DEEPPROBLOG
    elif experiment == '2MNIST':
        model = DetMNISTPairsEncoder(**kwargs)
    else:
        NotImplementedError("This model is not provided.")

    return model