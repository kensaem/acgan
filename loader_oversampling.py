from loader import *
import math
import random

class LoaderOversampling(Loader):
    RawDataTuple = collections.namedtuple("RawDataTuple", ['path', 'label'])

    data_each = {}
    size_each = {}

    def __init__(self, data_path, batch_size):
        super().__init__(data_path, batch_size)
        return

    def load_data(self):
        self.max_class_size = 0
        # Check the size of each class

        print("...Loading from %s" % self.data_path)
        dir_name_list = os.listdir(self.data_path)
        for dir_name in dir_name_list:
            dir_path = os.path.join(self.data_path, dir_name)
            file_name_list = os.listdir(dir_path)
            print("\tNumber of files in %s = %d" % (dir_name, len(file_name_list)))
            label = int(dir_name)
            class_data = []
            for file_name in file_name_list:
                file_path = os.path.join(dir_path, file_name)
                class_data.append(self.RawDataTuple(path=file_path, label=label))
            class_data.sort()
            self.data_each[label] = class_data
            self.size_each[label] = len(file_name_list)

        total_size = 0
        for label, size in self.size_each.items():
            total_size += size
        print("\tTotal number of data = %d" % total_size)

        print("...Loading done.")
        return

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.get_empty_batch(batch_size)
        for idx in range(batch_size):
            chosen_class = random.randrange(10)
            chosen_sample_idx = random.randrange(self.size_each[chosen_class])
            # print(chosen_class, chosen_sample_idx)

            single_data = self.data_each[chosen_class][chosen_sample_idx]
            image = cv2.imread(single_data.path, 1)

            batch.images[idx, :, :, :] = image
            batch.labels[idx] = single_data.label

            # Verifying batch
            # print(single_data.path)
            # print(batch.images[idx, 0, 0, 0])
            # print(batch.labels[idx])

        self.cur_idx += batch_size

        return batch


