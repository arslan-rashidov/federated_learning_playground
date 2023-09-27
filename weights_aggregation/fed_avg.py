from functools import reduce
from io import BytesIO
from typing import List, Tuple
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch

from weights_aggregation.weights_aggregation_strategy import WeightsAggregationStrategy


class FedAvgStrategy(WeightsAggregationStrategy):
    def aggregate_train(self, results: List[Tuple[BytesIO, BytesIO]]) -> OrderedDict:
        results_transformed = []

        layer_names = None

        for result in results:
            weights = torch.load(result[0])
            weights_layer_names = list(weights.keys())
            if layer_names is None:
                layer_names = weights_layer_names
            else:
                assert layer_names == weights_layer_names
            weights_layer_values = weights.values()
            num_examples = pd.read_csv(result[1])['train_num_examples'].iloc[0]
            print(num_examples)
            results_transformed.append((weights_layer_values, num_examples))

        del results

        num_examples_total = sum([result[1] for result in results_transformed])

        weighted_weights = [
            [layer * result[1] for layer in result[0]]
            for result in results_transformed
        ]
        weights_list = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        weights_prime = OrderedDict()
        for layer_name_i in range(len(layer_names)):
            layer_name = layer_names[layer_name_i]
            layer_values = weights_list[layer_name_i]
            weights_prime[layer_name] = layer_values

        return weights_prime