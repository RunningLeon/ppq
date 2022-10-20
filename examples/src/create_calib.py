import h5py
import tqdm
from torch.utils.data import DataLoader


def create_calib_input_data(calib_file: str, dataloader: DataLoader):

    with h5py.File(calib_file, mode='w') as file:
        calib_data_group = file.create_group('calib_data')
        input_data_group = calib_data_group.create_group('end2end')
        input_group = input_data_group.create_group('input')
        for data_id, input_data in enumerate(tqdm.tqdm(dataloader)):
            input_ndarray = input_data.detach().cpu().numpy()
            input_group.create_dataset(str(data_id),
                                       shape=input_ndarray.shape,
                                       compression='gzip',
                                       compression_opts=4,
                                       data=input_ndarray)
            file.flush()
