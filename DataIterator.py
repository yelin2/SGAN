from torch.utils.data import DataLoader


class DataIterator:
    def __init__(self, dataset, minibatch=64, shuffle=True, num_workers=4):
        self.dataset = dataset
        self.loader = DataLoader(
            dataset, minibatch, num_workers=num_workers, shuffle=shuffle)
        self._iter = iter(self.loader)

    def __next__(self):
        try:
            data, target = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            data, target = next(self._iter)

        return data, target
