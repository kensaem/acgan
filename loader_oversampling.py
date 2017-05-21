from loader import *
import math


class LoaderOversampling(Loader):
    RawDataTuple = collections.namedtuple("RawDataTuple", ['path', 'label'])

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
            self.max_class_size = max(self.max_class_size, len(file_name_list))
        print("Current maximum size for each class = %d" % self.max_class_size)

        # Load data from directory with up-sampling
        dir_name_list = os.listdir(self.data_path)
        for dir_name in dir_name_list:
            dir_path = os.path.join(self.data_path, dir_name)
            file_name_list = os.listdir(dir_path)
            current_class_size = 0
            while (current_class_size + len(file_name_list)) <= self.max_class_size:
                for file_name in file_name_list:
                    file_path = os.path.join(dir_path, file_name)
                    self.data.append(self.RawDataTuple(path=file_path, label=int(dir_name)))
                current_class_size += len(file_name_list)
            print("\tNumber of samples in %s = %d" % (dir_name, current_class_size))

        self.data.sort()
        print("\tTotal number of data = %d" % len(self.data))

        print("...Loading done.")
        return


