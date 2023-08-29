# Weather Data Compression with Neural Networks
Welcome! This repository contains the code implementation of a neural network-based method for compressing weather data. The approach demonstrates the effectiveness of quantization techniques to achieve efficient data representation.

## Usage 🛞
To use the code, follow these [setup steps](https://github.com/elenagensch/PADL23_weather_compression).

## Datasets 🗂️
The [ERA5 dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview), developed by the Copernicus Climate Change Service at the European Centre for Medium-Range Weather Forecasts (ECMWF), is an important resource for understanding global climate trends. It provides hourly estimates and spans a broad vertical coverage across 37 pressure levels from 1000 hPa to 1 hPa. Horizontally, the coverage is global with a resolution in the atmosphere of 0.25 × 0.25.

### Download data 

For our experiments, we used a certain subset as defined here:
```bash
python download.py --variable=geopotential --mode=single --level_type=pressure --years=2016 --resolution=0.5 --time=00:00 --pressure_level 10 50 100 200 300 400 500 700 850 925 1000 --custom_fn=dataset1.nc --output_dir=datasets
```
## Experiments  🚀
You can start training or test runs with the script `train.py`. See the next section for more information on configurable parameters
### Run experiments in Section 3.1
```bash
for W in 32 64 128 256 512
do
    python train.py --nepoches=20 --all  --quantizing --testing --variable=z  --dataloader_mode=sampling_nc --file_name=dataset1.nc --width=$W --output_file=dataset1_w${W}.nc
    python train.py --nepoches=20 --all  --quantizing --testing --variable=z  --dataloader_mode=sampling_nc --file_name=dataset2.nc --width=$W --output_file=dataset2_w${W}.nc
    python train.py --nepoches=20 --all  --quantizing --generate_full_outputs --variable=z --use_stat --tscale=360 --dataloader_mode=weatherbench  --file_name=dataset3_z_*.nc --width=$W --output_file=dataset3_z_w${W}.nc
    python train.py --nepoches=20 --all  --quantizing --generate_full_outputs --variable=z --use_stat --tscale=360 --dataloader_mode=weatherbench  --file_name=dataset4_z_*.nc --width=$W --output_file=dataset4_z_w${W}.nc
done
```

### Run experiments in Section 3.2 
```bash
python train.py --nepoches=20 --all  --quantizing --generate_full_outputs --variable=z --use_stat --tscale=360 --dataloader_mode=weatherbench  --file_name=dataset4_z_*.nc --width=512 --output_file=dataset4_z_w512.nc
python train.py --nepoches=20 --all  --quantizing --generate_full_outputs --variable=t --use_stat --tscale=360 --dataloader_mode=weatherbench  --file_name=dataset4_t_*.nc --width=512 --output_file=dataset4_t_w512.nc
cdo selyear,1979/2015 dataset4_z_w512.nc train_data_path/geopotential_500/geopotential_500_1979_2015.nc
cdo selyear,2016/2018 dataset4_z.nc train_data_path/geopotential_500/geopotential_500_2016_2018.nc
cdo selyear,1979/2015 dataset4_t_w512.nc train_data_path/temperature_850/temperature_850_1979_2015.nc
cdo selyear,2016/2018 dataset4_t.nc train_data_path/temperature_850/temperature_850_2016_2018.nc
cd WeatherBench
python -m src.train_nn -c config.yml --datadir=train_data_path
```

# Configuration Parameters  🛠
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
- `--ckpt_path`: Path to the model checkpoint directory. Default: ""
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
