from util import imgload
from util.dataloader import BaseDataLoader


class SpectrogramLoader(BaseDataLoader):

    def __init__(self, paths, labels, pre_process_fn: callable=None,
                 shuffle: bool=True):
        super(SpectrogramLoader, self).__init__(paths=paths, labels=labels,
                                                pre_process_fn=pre_process_fn,
                                                shuffle=shuffle)

    def loader(self, source_path: str, *args, **kwargs):
        return imgload.img_load(source_path, args, kwargs)

    def get_train_data(self):
        raise NotImplementedError('Must implement a method to get trainable '
                                  'data')

    def get_test_data(self):
        raise NotImplementedError('Must implement a method to get testable '
                                  'data')
