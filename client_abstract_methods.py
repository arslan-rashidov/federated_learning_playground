from abc import abstractmethod
from typing import Union, List

import torch

from metric import Metric


@abstractmethod
def load_model() -> torch.nn.Module:
    """
    Метод для генерации модели. На вход будут подаваться параметры, указанные на сервисе как Init Parameters.

    Обязательные параметры, которые необходимо учитывать разработчику ML:
    **init_parameters - параметры, которые разработчик ML указывает на сайте в разделе Init Parameters

    :return:
    Возвращает:
    (nn.Module) - модель.
    """

@abstractmethod
def get_dataset(dataset_path: str, with_split: bool) -> Union[(torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset), (torch.utils.data.Dataset, torch.utils.data.Dataset), (torch.utils.data.Dataset)]:
    """
    Метод для чтения, предобработки и разбития датасета(with_split=True). На вход будут подаваться dataset_path,
    with_split, а также параметры, которые разработчик ML укажет в коде как необходимые(специфические для каждого отдельного пользователя)

    Тут описывается вся логика предобработки датасета.

    Обязательные параметры, которые необходимо учитывать разработчику ML:
    dataset_path(str) - путь до csv-файла с датасетом.
    with_split(bool) - булева переменная, говорящая о необходимости разбития датасета на выборки(train, valid, test). Если
     True, тогда следует после предобработки данных разбить их на одну из следующих выборок - (train, test), (train, valid, test). Если
     False, тогда требуется лишь предобработка данных и возвращение лишь подготовленного
     датоасета(в дальнейшем будет использоваться для получения предсказаний модели на данных пользователя).


    :return:
    Возвращает один из следующих кортежей:
    (torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset) - возвращается при with_split=True. В будущем используется как train_set, valid_set и test_set
    (torch.utils.data.Dataset, torch.utils.data.Dataset) - возвращается при with_split=True. В будущем используется как train_set и test_set
    (torch.utils.data.Dataset) - возвращается при with_split=False. В будущем используется как test_set(выборка для тестировки модели на весах)
    """

@abstractmethod
def train(model: torch.nn.Module, train_set: torch.utils.data.Dataset, valid_set: torch.utils.data.Dataset = None) -> (List[Metric], torch.nn.Module):
    """
    Метод для тренировки модели, полученной из метода load_model. На вход будут подаваться: модель, сгенерированная
     методом load_model, train_set полученный из метода get_dataset, valid_set(опционально) полученный из
     метода get_dataset(если возврат выборки предусмотрен разрботчиком ML в методе get_dataset), а также
     параметры, которые разработчик ML укажет в коде как необходимые(специфические для каждого отдельного пользователя)

    Обязательные параметры, которые необходимо учитывать разработчику ML:
    model(nn.Module) - модель, полученный из метода load_model
    train_set(torch.utils.data.Dataset) - тренировочная выборка, полученная из метода get_dataset
    valid_set(torch.utils.data.Dataset) - валидационная выборка, полученная из метода get_dataset. Подается на вход только если в методе
     get_dataset предусмотрено получение валидационной выборки и ее возврата
    **train_parameters - параметры, которые разработчик ML указывает на сайте в разделе Train Parameters


    :return:
    (List[Metric], nn.Module) - кортеж состоящий из:
     1. Список метрик полученных в ходе обучения;
     2. Обученной модели;
    """

@abstractmethod
def test(model: torch.nn.Module,  test_set: torch.utils.data.Dataset, return_output: bool) -> Union[(List[Metric]), (List[Metric], list)]:
    """
    Метод для тестировки модели на данных. На вход подается model, полученная из load_model; return_output, булева
    переменная говорящая о необходимости возврата выхода из модели на данных; test_set - тестировочная выборка полученная из get_dataset, а также
     параметры, которые разработчик ML укажет в коде как необходимые(специфические для каждого отдельного пользователя)


    Обязательные параметры, которые необходимо учитывать разработчику ML:
    model(nn.Module) - модель, полученный из метода load_model
    test_set(torch.utils.data.Dataset) - тестировочная выборка, полученная из метода get_dataset
    return_output(bool) - булева переменная, говорящая о необходимости возвращать ответы модели
    **test_parameters - параметры, которые разработчик ML указывает на сайте в разделе Test Parameters

    :return:
    Возвращает один из следующих кортежей:
    (List[Metric]) - метрики полученные в ходе тестировки модели на данных
    (List[Metric], list) - метрики и ответы модели полученные в ходе тестировки модели на данных(только если return_output=True)
    """
