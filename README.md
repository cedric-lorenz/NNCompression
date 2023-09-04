# Weather Data Compression with Neural Networks 
Welcome! This repository contains the code implementation of a neural network-based method for compressing weather data and is a fork of [this repo](https://github.com/spcl/NNCompression). The approach demonstrates the effectiveness of quantization techniques to achieve efficient data representation.

## Usage ⚙️
To use the code, follow these [setup steps](https://github.com/elenagensch/PADL23_weather_compression).

## Datasets 🗂️
The [ERA5 dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview), developed by the Copernicus Climate Change Service at the European Centre for Medium-Range Weather Forecasts (ECMWF), is an important resource for understanding global climate trends. It provides hourly estimates and spans a broad vertical coverage across 37 pressure levels from 1000 hPa to 1 hPa. Horizontally, the coverage is global with a resolution in the atmosphere of 0.25 × 0.25.

### Download data 

For our experiments, we used a certain subset as defined here:
```bash
python download.py --variable=geopotential --mode=single --level_type=pressure --years=2016 --resolution=0.5 --time=00:00 --pressure_level 10 50 100 200 300 400 500 700 850 925 1000 --custom_fn=dataset1.nc --output_dir=datasets
```
Remember that an API key available [here](https://cds.climate.copernicus.eu/api-how-to) is required.

## Experiments  🚀
If you followed these [setup steps](https://github.com/elenagensch/PADL23_weather_compression), follow the instructions in the [repository](https://github.com/elenagensch/PADL23_weather_compression) for running experiments. You can start training or test runs with the script `train.py`. See the next section for more information on the configurable parameters.
### Experiment example:
```shell
./scripts/quick-submit.sh -- python train.py --all --quantizing --testing --variable=z --dataloader_mode=sampling_nc --file_name=datasets/dataset.nc --width=512
```
### Mixed Precision:
Note: In the following, Weights & Biases is used as the logging engine. You can also use the code without it by removing all wandb related options.

#### 32-bit baseline
```shell
NUM_GPU=3 SERIES=mixed-precision NAME=baseline ./scripts/quick-submit.sh -- WANDB_CACHE_DIR=./wandb_cache WANDB_CONFIG_DIR=./wandb_config WANDB_API_KEY=<YOUR API KEY>  python train.py --testing --file_name=datasets/dataset.nc  --width=512 --model_precision=16 --all --quantizing --use_wandb
```

#### 16-bit mixed precision
```shell
NUM_GPU=3 SERIES=mixed-precision NAME=width768 ./scripts/quick-submit.sh -- WANDB_CACHE_DIR=./wandb_cache WANDB_CONFIG_DIR=./wandb_config WANDB_API_KEY=<YOUR API KEY> python train.py --testing --file_name=datasets/dataset.nc  --width=768 --model_precision=16 --all --quantizing --use_wandb
```
### Quantization with BiTorch

#### 32-bit baseline 
```shell
python train.py --variable=z --dataloader_mode=sampling_nc --testing --file_name=datasets/dataset.nc --use_wandb --all --width=256 --quantizing
```
#### 8-bit quantization
```shell
python train.py --variable=z --dataloader_mode=sampling_nc --testing --file_name=datasets/dataset.nc --use_wandb --all  --optimizer=radam --use_quantized_linear_layer --q_bits=8 --width=512
```
#### 4-bit quantization with 8bit pre-trained model
```shell
python train.py --variable=z --dataloader_mode=sampling_nc --testing --file_name=datasets/dataset.nc --use_wandb --all --use_quantized_linear_layer --q_bits=4 --width=768 --learning_rate=0.0003 --ckpt_path=8bit.ckpt --nepoches=30
```
### Fourier feature tuning
#### Trainable Fourier Features
```shell
python train.py --all --quantizing --testing --variable=z --model_precision=16  --dataloader_mode=sampling_nc --file_name=datasets/dataset.nc --trainable_fourierfeature=True
```
#### Number of fourier feature and sigma, the features are sampled from

```shell
python train.py --all --quantizing --testing --variable=z --model_precision=16 --nfeature=128 --dataloader_mode=sampling_nc --file_name=datasets/dataset.nc --wandb_sweep_config_name=sweep_config_fp16 --sigma=2
```
### Configuration Parameters  🛠
This section describes the various configuration parameters that can be used when running the script. These parameters can be specified via command-line arguments.

- `--num_gpu`: Number of GPUs to use for training. Default: -1 (all available GPUs)
- `--nepoches`: Number of epochs to train the model for. Default: 20
- `--batch_size`: Batch size for training. Default: 3
- `--num_workers`: Number of worker threads for data loading. Default: 1
- `--learning_rate`: Learning rate for optimization. Default: 3e-4
- `--accumulate_grad_batches`: Number of batches to accumulate gradients over before performing a backward pass. Default: 1
- `--sigma`: Sigma parameter for Fourier Features sampled from a Gaussian distribution. Default: 1.6
- `--ntfeature`: Number of temporal features. Default: 16
- `--width`: Width of the Fully Connected Neural Network. Default: 512
- `--depth`: Depth of the Fully Connected Neural Network. Default: 12
- `--tscale`: Time scale parameter. Default: 60.0
- `--zscale`: Z scale parameter (applied on pressure level). Default: 100.0
- `--variable`: Variable used for a certain purpose. Default: "z"
- `--dataloader_mode`: Mode for data loading from the dataloader depending on the dataset. Default: "sampling_nc" (for ERA5)
- `--data_path`: Path to the data directory. Default: "."
- `--file_name`: Name of the input file. (No default specified)
- `--ckpt_path`: Path to the model checkpoint directorys. Default: ""
- `--nfeature`: Number of fourier features. Default: 128
- `--use_fourierfeature`: Use Fourier features. (Flag, no value required)
- `--trainable_fourierfeature`: Train Fourier Features.  (Flag, no value required)
- `--use_batchnorm`: Use batch normalization. (Flag, no value required)
- `--use_skipconnect`: Use skip connections in the model. (Flag, no value required)
- `--use_invscale`: Use inverse scaling for a certain feature. (Flag, no value required)
- `--use_tembedding`: Use temporal embedding. (Flag, no value required)
- `--tembed_size`: Size of temporal embedding, corresponds to the number of time steps. Default: 400
- `--tresolution`: Temporal resolution parameter. Default: 24.0
- `--use_xyztransform`: Use XYZ transform for latitude and longitude. (Flag, no value required)
- `--use_stat`: Use statistics. (Flag, no value required)
- `--loss_type`: Type of loss function to use. Default: "scaled_mse"
- `--all`:  Use batch normalization, inverse scaling, skip connections, XYZ transformation and Fourier Features. 
- `--testing`: Run model evaluation on full dataset. (Flag, no value required)
- `--notraining`: Disable training, e.g. for test runs only. (Flag, no value required)
- `--generate_full_outputs`: Generate decompressed output data. (Flag, no value required)
- `--output_path`: Path for saving output files. Default: "."
- `--output_file`: Name of the output file. Default: "output.nc"
- `--quantizing`: Quantize Model to 16 bit. (Flag, no value required)
- `--use_wandb`: Use Weights & Biases for logging. (Flag, no value required)
- `--log_dir`: Directory for log files. Default: "../logs"
- `--seed`: Random seed for reproducibility. Default: 1111
- `--use_quantized_linear_layer`: Use BiTorch quantized linear layer. (Flag, no value required)
- `--q_bits`: Number of bits for quantization. Default: 2
