from inspect import signature

from base_client import Client
import argparse
from anti_fraud_fl.client_methods import init_parameters, train_parameters, test_parameters, load_model, get_dataset, train, test

if __name__ == '__main__':
    client = Client(init_params=init_parameters,
                    train_params=train_parameters,
                    test_params=test_parameters,
                    load_model_fn=load_model,
                    get_dataset_fn=get_dataset,
                    train_fn=train,
                    test_fn=test)

    parser = argparse.ArgumentParser(description='Client Model Trainer')
    required_params = [*client.get_dataset_fn_required_parameters, *client.train_fn_required_parameters, *client.test_fn_required_parameters]

    for param in required_params:
        name = param['name']
        type = param['type']
        default = param['default']

        if type and default:
            parser.add_argument(name, type=param['type'], default=param['default'])
        elif type:
            parser.add_argument(name, type=param['type'])
        elif default:
            parser.add_argument(name, default=param['default'])
        else:
            parser.add_argument(name)

    args = vars(parser.parse_args())

    client.fit(**args)