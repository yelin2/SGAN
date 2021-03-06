from torch.utils.data import DataLoader


class DataIterator:
    def __init__(self, dataset, minibatch=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.loader = DataLoader(
            dataset, minibatch, num_workers=num_workers, shuffle=shuffle)
        self._iter = iter(self.loader)

    def __next__(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            data = next(self._iter)

        return data
