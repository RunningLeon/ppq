import mmcv
from mmcv.parallel import DataContainer
from mmcv.utils import print_log
from mmseg.datasets import (DATASETS, CityscapesDataset, build_dataloader,
                            build_dataset)
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class CustomCityscapesDataset(CityscapesDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 calib_txt=None,
                 **kwargs):
        super(CustomCityscapesDataset,
              self).__init__(img_suffix=img_suffix,
                             seg_map_suffix=seg_map_suffix,
                             **kwargs)
        self.calib_txt = calib_txt
        self.update_img_infos()

    def update_img_infos(self):
        if self.calib_txt is not None:
            calib_file_names = set()
            with open(self.calib_txt, 'r') as f:
                for line in f:
                    filename = line.strip().split('train/')[1]
                    calib_file_names.add(filename)
            self.img_infos = [
                info for info in self.img_infos
                if info['filename'] in calib_file_names
            ]
            print_log(f'Loaded {len(self.img_infos)} images for calibration',
                      logger=get_root_logger())

    def prepare_calib_img(self, idx):
        output = self.prepare_test_img(idx)
        img = output['img'][0]
        if isinstance(img, DataContainer):
            img = img.data[0]
        return img

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        if self.calib_txt is not None:
            return self.prepare_calib_img(idx)

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)


def build_mmseg_dataloader(model_cfg,
                           data_type,
                           calib_txt=None,
                           img_height=512,
                           img_width=1024):
    cfg = mmcv.Config.fromfile(model_cfg)
    # cfg.test_pipeline[1]['img_scale'] = (img_width, img_height)
    # cfg.test_pipeline[1]['transforms'][0]['keep_ratio'] = False
    # cfg.data.samples_per_gpu = 1
    # cfg.data.workers_per_gpu = 1
    dataset = cfg.data[data_type]
    dataset.pipeline = cfg.test_pipeline
    dataset.test_mode = True
    dataset.type = 'CustomCityscapesDataset'
    dataset['calib_txt'] = calib_txt
    dataset = build_dataset(dataset)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=1,
                                   workers_per_gpu=1,
                                   shuffle=False)
    return data_loader


def evaluate_model(model, data_loader, out_json):
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    results = []
    for batch_indices, data in zip(loader_indices, data_loader):
        img = data['img'][0]
        if isinstance(img, DataContainer):
            img = img.data[0]
        img = img.cuda()
        out = model.forward(img)[0].squeeze(1)
        out = list(out.cpu().numpy())
        out = dataset.pre_eval(out, indices=batch_indices)
        results.extend(out)
        batch_size = len(out)
        for _ in range(batch_size):
            prog_bar.update()
    metric_results = dataset.evaluate(results, metric='mIoU')
    mmcv.dump(metric_results, out_json, indent=4)
