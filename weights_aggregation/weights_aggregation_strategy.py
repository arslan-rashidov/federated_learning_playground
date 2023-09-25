from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Tuple, OrderedDict


class WeightsAggregationStrategy(ABC):
    @abstractmethod
    def aggregate_train(self, results: List[Tuple[BytesIO, BytesIO]]) -> OrderedDict:
        pass