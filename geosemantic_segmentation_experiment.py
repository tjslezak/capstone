import os
import rastervision as rv
from geotoolkit import build_class_color_dict

def build_scene(task, data_uri, id, channel_order=None):
    ## id = id.replace('-', '_')
    raster_source_uri = '{}/rasters/{}_raster.tif'.format(data_uri, id)
    label_source_uri = '{}/labels/{}_labels.tif'.format(data_uri, id)

    # Using with_rgb_class_map because input TIFFs have classes encoded as RGB colors.
    label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
        .with_rgb_class_map(task.class_map) \
        .with_raster_source(label_source_uri) \
        .build()

    # URI will be injected by scene config.
    # Using with_rgb(True) because we want prediction TIFFs to be in RGB format.
    label_store = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
        .with_rgb(True) \
        .build()

    # Must define raster_source separate from raster_source_uri so StatsTransformer
    # can convert uint16 images to uint8
    raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
        .with_uri(raster_source_uri) \
        .with_stats_transformer() \
        .build()

    scene = rv.SceneConfig.builder() \
                          .with_task(task) \
                          .with_id(id) \
                          .with_raster_source(raster_source, channel_order=channel_order) \
                          .with_label_source(label_source) \
                          .with_label_store(label_store) \
                          .build()

    return scene


class GeoSemanticSegmentation(rv.ExperimentSet):
    def exp_main(self, root_uri, data_uri, test_run=False):
        """Run an experiment on Sentinel-2 data of Arizona.
        Uses Tensorflow Deeplab backend with Mobilenet architecture.
        Args:
            root_uri: (str) root directory for experiment output
            data_uri: (str) root directory of data
            test_run: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        if test_run == 'True':
            test_run = True
        elif test_run == 'False':
            test_run = False

        train_ids = ['RVV', 'RWV', 'RXV', 'SQA', 'SQR', 'SQS',
                     'SQU', 'SQV', 'STA', 'STB', 'STC', 'STD',
                     'STE', 'STF', 'SUA', 'SUB', 'SUC', 'SUD',
                     'SUE', 'SUF', 'SVA', 'SVB', 'SVC', 'SVD',
                     'SVF', 'SWA', 'SWB', 'SWC', 'SWD', 'SWE',
                     'SWF', 'SXA', 'SXB', 'SXC', 'SXE', 'SXF']

        # 'SVE' = SP Crater
        val_ids = ['SVE', 'SQT', 'SXD']

        # blue, red, ir
        channel_order = [0, 1, 2]

        debug = False
        batch_size = 12
        chips_per_scene = 225
        chip_size = 366 
        num_steps = 150000
        model_type = rv.XCEPTION_65
        task_type = rv.SEMANTIC_SEGMENTATION
        ac_key = '{}_{}'.format('AZGEO', task_type.lower())

        if test_run:
            debug = True
            num_steps = 1
            batch_size = 1
            chips_per_scene = 225
            train_ids = train_ids[0:1]
            val_ids = val_ids[0:1]

        classes = build_class_color_dict()

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(chip_size) \
                            .with_classes(classes) \
                            .with_chip_options(
                                chips_per_scene=chips_per_scene,
                                debug_chip_probability=0.2,
                                negative_survival_probability=1.0) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(model_type) \
                                  .with_train_options(sync_interval=600) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(debug) \
                                  .build()

        train_scenes = [build_scene(task, data_uri, id, channel_order)
                      for id in train_ids]
        val_scenes = [build_scene(task, data_uri, id, channel_order)
                      for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()


        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('geoss-xception-batch12') \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .with_stats_analyzer() \
                                        .with_analyze_key(ac_key) \
                                        .with_chip_key(ac_key)
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
