import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import get_model_size_mb
import math
import numpy as np
import xarray as xr
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pyinterp
import pyinterp.backends.xarray
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange, tqdm
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torchmetrics import MeanSquaredError, MeanAbsoluteError
import wandb
import utils.metrics
import yaml
import random
import bitorchinfo
import bitorch
from bitorch.layers import QLinear
from bitorch.quantizations import InputDoReFa, WeightDoReFa


bitorch.mode = bitorch.RuntimeMode.RAW

YEAR = 2016


class ERA5stat:
    def __init__(self, file_name_mean, file_name_std, data_path, variable, grid_type):
        self.ds_mean = xr.load_dataset(f"{data_path}/{file_name_mean}")[variable]
        self.ds_std = xr.load_dataset(f"{data_path}/{file_name_std}")[variable]
        self.grid_type = grid_type
        if grid_type == "regular":
            self.interp_mean = pyinterp.backends.xarray.Grid3D(self.ds_mean)
            self.interp_std = pyinterp.backends.xarray.Grid3D(self.ds_std)
        elif grid_type == "sphere_grid":
            self.interp_mean = RegularGridInterpolator(
                (self.ds_mean.plev, self.ds_mean.y, self.ds_mean.x), self.ds_mean.data
            )
            self.interp_std = RegularGridInterpolator(
                (self.ds_std.plev, self.ds_std.y, self.ds_std.x), self.ds_std.data
            )

    def interp_regular(self, plev, lat, lon):
        mean = self.interp_mean.trivariate(
            dict(longitude=lon.ravel(), latitude=lat.ravel(), level=plev.ravel())
        ).reshape(lat.shape)
        std = self.interp_std.trivariate(
            dict(longitude=lon.ravel(), latitude=lat.ravel(), level=plev.ravel())
        ).reshape(lat.shape)
        return mean, std

    def interp_sphere_grid(self, plev, y, x):
        coord = torch.stack((plev, lat, lon))
        mean = self.interp_mean(coord).reshape(y.shape)
        std = self.interp_mean(coord).reshape(y.shape)
        return mean, std


class WeatherBenchDataset_sampling(Dataset):
    def __init__(self, file_name, data_path, nbatch, nsample, variable="z"):
        file_path = f"{data_path}/{file_name}"
        self.ds = xr.open_mfdataset(file_path).load()
        self.ds = self.ds.assign_coords(time=np.arange(len(self.ds.time)))
        self.interpolator = RegularGridInterpolator(
            (self.ds.time, self.ds.lat, self.ds.lon),
            self.ds[variable].data,
            bounds_error=False,
            fill_value=None,
        )
        self.variable = variable
        self.ntime = len(self.ds.time)
        self.nbatch = nbatch
        self.nsample = nsample
        self.rndeng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        self.mean = self.ds[variable].mean(dim=["time"]).to_numpy()
        self.std = (
            self.ds[variable].max(dim=["time"]) - self.ds[variable].min(dim=["time"])
        ).to_numpy()
        self.interp_mean = RegularGridInterpolator(
            (self.ds.lat, self.ds.lon), self.mean, bounds_error=False, fill_value=None
        )
        self.interp_std = RegularGridInterpolator(
            (self.ds.lat, self.ds.lon), self.std, bounds_error=False, fill_value=None
        )

    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        if isinstance(idx, int):
            rnds = self.rndeng.draw(self.nsample)
            time = rnds[:, 0] * (self.ntime - 1)
            pind = torch.zeros_like(time) + float(self.ds.level.mean())
            latind = 90 - 180 / math.pi * torch.acos(1 - 2 * rnds[:, 1])
            lonind = rnds[:, 2] * 360
            coord = torch.stack((time, pind, latind, lonind), dim=-1).to(torch.float32)
            coord_in = torch.stack((time, latind, lonind), dim=-1)
            var_sampled = self.interpolator(coord_in).reshape(latind.shape)
            var_sampled = torch.as_tensor(var_sampled).unsqueeze(-1).to(torch.float32)
            mean = torch.as_tensor(
                self.interp_mean(coord_in[..., 1:]).reshape(var_sampled.shape)
            ).to(torch.float32)
            std = torch.as_tensor(
                self.interp_std(coord_in[..., 1:]).reshape(var_sampled.shape)
            ).to(torch.float32)
            return coord, var_sampled, mean, std

    def getslice(self, tind, pind):
        lat_v = torch.as_tensor(self.ds.lat.to_numpy())
        lon_v = torch.as_tensor(self.ds.lon.to_numpy())
        lat, lon = torch.meshgrid((lat_v, lon_v), indexing="ij")
        p = torch.zeros_like(lat) + float(self.ds.level.mean())
        t = torch.zeros_like(lat) + float(tind)
        coord = torch.stack((t, p, lat, lon), dim=-1).unsqueeze(0).to(torch.float32)
        var = (
            torch.as_tensor(self.ds[self.variable].isel(time=tind).to_numpy())
            .unsqueeze(-1)
            .unsqueeze(0)
            .to(torch.float32)
        )
        mean = torch.as_tensor(self.mean).reshape(var.shape).to(torch.float32)
        std = torch.as_tensor(self.std).reshape(var.shape).to(torch.float32)
        return coord, var, mean, std


class ERA5Dataset_sampling(Dataset):
    def __init__(
        self, file_name, data_path, nbatch, nsample, variable="z", stat_config=None
    ):
        file_path = f"{data_path}/{file_name}"
        self.ds = xr.open_dataset(file_path)[variable].load()
        self.ds = self.ds.assign_coords(time=self.ds.time.dt.dayofyear - 1)
        self.interpolator = pyinterp.backends.xarray.Grid4D(self.ds)
        self.variable = variable
        self.ntime = len(self.ds.time)
        self.nbatch = nbatch
        self.nsample = nsample
        self.rndeng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        if stat_config is not None:
            self.stat = ERA5stat(**stat_config)
        else:
            self.stat = None

    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        if isinstance(idx, int):
            rnds = self.rndeng.draw(self.nsample)
            time = rnds[:, 0] * (self.ntime - 1)
            pind = torch.as_tensor(self.ds.level.to_numpy(), dtype=torch.float32)[
                torch.randperm(self.nsample) % len(self.ds.level)
            ]
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            latind = 90 - 180 / math.pi * torch.acos(1 - 2 * rnds[:, 1])
            lonind = rnds[:, 2] * 360
            coord = torch.stack((time, pind, latind, lonind), dim=-1)
            var_sampled = self.interpolator.quadrivariate(
                dict(
                    longitude=lonind.ravel(),
                    latitude=latind.ravel(),
                    time=time.ravel(),
                    level=pind.ravel(),
                )
            ).reshape(latind.shape)
            var_sampled = torch.as_tensor(var_sampled).unsqueeze(-1)
            if self.stat is None:
                return coord, var_sampled
            else:
                mean, std = self.stat.interp_regular(pind, latind, lonind)
                return coord, var_sampled, mean, std

    def getslice(self, tind, pind):
        lat_v = torch.as_tensor(self.ds.latitude.to_numpy())
        lon_v = torch.as_tensor(self.ds.longitude.to_numpy())
        lat, lon = torch.meshgrid((lat_v, lon_v), indexing="ij")
        p = torch.zeros_like(lat) + self.ds.level.to_numpy()[pind]
        t = torch.zeros_like(lat) + float(tind)
        coord = torch.stack((t, p, lat, lon), dim=-1).to(torch.float32)
        var = (
            torch.as_tensor(self.ds.isel(time=tind, level=pind).to_numpy())
            .to(torch.float32)
            .unsqueeze(-1)
        )
        return coord.unsqueeze(0), var.unsqueeze(0)


class FourierFeature(nn.Module):
    def __init__(self, sigma, infeature, outfeature, trainable):
        super(FourierFeature, self).__init__()
        self.feature_map = nn.Parameter(
            torch.normal(0.0, sigma, (outfeature, infeature)), requires_grad=trainable
        )

    def forward(self, x, cos_only: bool = False):
        # x shape: (..., infeature)
        x = 2 * math.pi * F.linear(x, self.feature_map)
        if cos_only:
            return torch.cos(x)
        else:
            return torch.cat((torch.sin(x), torch.cos(x)), dim=-1)


class LonLat2XYZ(nn.Module):
    def forward(self, x):
        # x shape: (..., (time, pressure, lat, lon))
        time = x[..., 0]
        p = x[..., 1]
        lat = x[..., 2]
        lon = x[..., 3]
        sinlat = torch.sin(lat)
        coslat = torch.cos(lat)
        sinlon = torch.sin(lon)
        coslon = torch.cos(lon)
        return torch.stack((time, p, sinlat, coslat * sinlon, coslat * coslon), dim=-1)


class NormalizeInput(nn.Module):
    def __init__(self, tscale, zscale):
        super(NormalizeInput, self).__init__()
        self.scale = nn.Parameter(
            torch.tensor(
                [1.0 / tscale, 1.0 / zscale, math.pi / 180.0, math.pi / 180.0]
            ),
            requires_grad=False,
        )

    def forward(self, x):
        return x * self.scale


class InvScale(nn.Module):
    def forward(self, coord, z_normalized):
        factor = 0.9
        p = coord[..., 1:2]
        std = 0.385e5 - 0.35e4 * torch.log(p)
        mean = 4.315e5 - 6.15e4 * torch.log(p)
        return (z_normalized / factor) * std + mean


class ResBlock(nn.Module):
    def __init__(
        self,
        width,
        use_batchnorm=True,
        use_skipconnect=True,
        use_quantized_linear_layer=False,
        q_bits=2,
    ):
        super(ResBlock, self).__init__()

        if use_quantized_linear_layer:
            self.fc1 = QLinear(
                width,
                width,
                bias=False,
                weight_quantization=WeightDoReFa(bits=q_bits),
                input_quantization=InputDoReFa(bits=q_bits),
            )
            self.fc2 = QLinear(
                width,
                width,
                bias=True,
                weight_quantization=WeightDoReFa(bits=q_bits),
                input_quantization=InputDoReFa(bits=q_bits),
            )
        else:
            self.fc1 = nn.Linear(width, width, bias=False)
            self.fc2 = nn.Linear(width, width, bias=True)

        self.use_batchnorm = use_batchnorm
        self.use_skipconnect = use_skipconnect
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(width)
            self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x_original):
        # x shape: (batch_size, width)
        x = x_original
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.gelu(x)
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.gelu(x)
        x = self.fc2(x)
        if self.use_skipconnect:
            return x + x_original
        else:
            return x


class FitNet(nn.Module):
    __constants__ = ["use_xyztransform", "use_fourierfeature", "use_invscale", "depth"]

    def __init__(self, args):
        super(FitNet, self).__init__()
        self.args = args
        if args.use_invscale:
            self.invscale = InvScale()
        if args.use_xyztransform:
            self.lonlat2xyz = LonLat2XYZ()
            ns = 3
        else:
            ns = 2
        if args.use_fourierfeature:
            self.fourierfeature_t = FourierFeature(
                args.sigma, 1, args.ntfeature, args.trainable_fourierfeature
            )
            self.fourierfeature_p = FourierFeature(
                args.sigma, 1, args.nfeature, args.trainable_fourierfeature
            )
            self.fourierfeature_s = FourierFeature(
                args.sigma, ns, args.nfeature, args.trainable_fourierfeature
            )
            nf = 2 * (2 * args.nfeature + args.ntfeature)
            if args.concat_input:
                nf += 5
        else:
            nf = 2 + ns

        self.normalize = NormalizeInput(args.tscale, args.zscale)
        self.depth = args.depth
        self.fci = nn.Linear(nf, args.width)
        self.fcs = nn.ModuleList(
            [
                ResBlock(
                    args.width,
                    args.use_batchnorm,
                    args.use_skipconnect,
                    args.use_quantized_linear_layer,
                    args.q_bits,
                )
                for i in range(args.depth)
            ]
        )
        self.fco = nn.Linear(args.width, 1)

        self.use_xyztransform = self.args.use_xyztransform
        self.use_fourierfeature = self.args.use_fourierfeature
        self.use_invscale = self.args.use_invscale
        self.concat_input = self.args.concat_input

    def forward(self, coord):
        batch_size = coord.shape[:-1]
        x = self.normalize(coord)
        if self.use_xyztransform:
            x = self.lonlat2xyz(x)
        if self.use_fourierfeature:
            t = x[..., 0:1]
            p = x[..., 1:2]
            s = x[..., 2:]
            if self.concat_input:
                x = torch.cat(
                    (
                        x,
                        self.fourierfeature_t(t),
                        self.fourierfeature_p(p),
                        self.fourierfeature_s(s),
                    ),
                    dim=-1,
                )
            else:
                x = torch.cat(
                    (
                        self.fourierfeature_t(t),
                        self.fourierfeature_p(p),
                        self.fourierfeature_s(s),
                    ),
                    dim=-1,
                )

        x = F.gelu(self.fci(x))
        x = x.flatten(end_dim=-2)  # batchnorm 1d only accepts (N, C) shape
        for fc in self.fcs:
            x = fc(x)
        x = F.gelu(x)
        x = self.fco(x)
        x = x.view(batch_size).unsqueeze(-1)
        if self.use_invscale or self.args.use_stat:
            x = torch.tanh(x)
        if self.use_invscale:
            x = self.invscale(coord, x)
        return x


class FitNetModule(pl.LightningModule):
    def __init__(self, args):
        super(FitNetModule, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = FitNet(args)
        self.input_type = torch.float32

    def train_dataloader(self):
        if self.args.dataloader_mode == "sampling_nc":
            dataset = ERA5Dataset_sampling(
                self.args.file_name,
                self.args.data_path,
                2677 * 9,
                361 * 120,
                variable=self.args.variable,
            )
        elif self.args.dataloader_mode == "weatherbench":
            dataset = WeatherBenchDataset_sampling(
                self.args.file_name,
                self.args.data_path,
                2677 * 9,
                361 * 120,
                variable=self.args.variable,
            )
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            prefetch_factor=8,
        )
        return dataloader

    def val_dataloader(self):
        it = 278
        ip = 6
        if self.args.dataloader_mode == "sampling_nc":
            data = ERA5Dataset_sampling(
                self.args.file_name,
                self.args.data_path,
                2677 * 9,
                361 * 120,
                variable=self.args.variable,
            ).getslice(it, ip)
        elif self.args.dataloader_mode == "weatherbench":
            data = WeatherBenchDataset_sampling(
                self.args.file_name,
                self.args.data_path,
                2677 * 9,
                361 * 120,
                variable=self.args.variable,
            ).getslice(it, ip)
        dataset = TensorDataset(*data)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers
        )
        return dataloader

    def configure_optimizers(self):
        optimizer = None
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == "radam":
            optimizer = torch.optim.RAdam(self.parameters(), lr=self.args.learning_rate)

        scheduler = None
        if self.args.scheduler == "reducelro":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", factor=0.25, patience=10000
            )
        elif self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        sched = {"scheduler": scheduler, "interval": "step", "monitor": "train_loss"}
        return [optimizer], [sched]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.args.use_stat:
            coord, var, mean, std = batch
            var_pred = self(coord) * 0.5 * 1.4 * std + mean
        else:
            coord, var = batch
            var_pred = self(coord)
        lat = coord[..., 2:3] / 180.0 * math.pi
        p = coord[..., 1:2]
        assert var.shape == var_pred.shape
        assert var.shape == lat.shape
        delta = var_pred - var

        (
            loss_linf,
            loss_mae,
            loss_mse,
            loss_scaled_mse,
            loss_logsumexp,
        ) = calculate_losses(delta, p)

        if self.args.loss_type == "scaled_mse":
            loss = loss_scaled_mse
        elif self.args.loss_type == "mse":
            loss = loss_mse
        elif self.args.loss_type == "logsumexp":
            loss = loss_logsumexp

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_loss_linf", loss_linf, sync_dist=True)
        self.log("train_loss_mae", loss_mae, sync_dist=True)
        self.log("train_loss_mse", loss_mse, sync_dist=True)
        self.log("train_loss_scaled_mse", loss_scaled_mse, sync_dist=True)
        self.log("train_loss_logsumexp", loss_logsumexp, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.args.use_stat:
                coord, var, mean, std = batch
                var_pred = self(coord) * 0.5 * 1.4 * std + mean
            else:
                coord, var = batch
                var_pred = self(coord)
            lat = coord[..., 2:3] / 180.0 * math.pi
            p = coord[..., 1:2]
            assert var.shape == var_pred.shape
            assert var.shape == lat.shape
            delta = var_pred - var

            (
                loss_linf,
                loss_mae,
                loss_mse,
                loss_scaled_mse,
                loss_logsumexp,
            ) = calculate_losses(delta, p)

            plt.figure(figsize=(10, 8))
            X, Y = (
                coord[..., 3].squeeze().detach().cpu(),
                coord[..., 2].squeeze().detach().cpu(),
            )
            plt.contour(X, Y, var.squeeze().detach().cpu(), colors="green")
            plt.contour(
                X, Y, var_pred.squeeze().detach().cpu(), colors="red"
            )  # cmap="Reds"
            plt.pcolormesh(
                X, Y, delta.squeeze().detach().cpu(), cmap="coolwarm", shading="nearest"
            )
            plt.axis("scaled")
            plt.title(f"p={torch.mean(coord[..., 1]).item()}")
            plt.colorbar(fraction=0.02, pad=0.04)
            plt.savefig(f"plots/validation_plot_{self.global_step}.png")
            plt.close()
            if self.trainer.is_global_zero:
                self.log(
                    "val_loss_linf", loss_linf, rank_zero_only=True, sync_dist=True
                )
                self.log("val_loss_mae", loss_mae, rank_zero_only=True, sync_dist=True)
                self.log("val_loss_mse", loss_mse, rank_zero_only=True, sync_dist=True)
                self.log(
                    "val_loss_scaled_mse",
                    loss_scaled_mse,
                    rank_zero_only=True,
                    sync_dist=True,
                )
                self.log(
                    "val_loss_logsumexp",
                    loss_logsumexp,
                    rank_zero_only=True,
                    sync_dist=True,
                )


def calculate_losses(delta, p):
    delta_abs = torch.abs(delta)
    loss_linf = delta_abs.max()
    loss_mae = delta_abs.mean()
    loss_mse = delta.pow(2).mean()
    loss_scaled_mse = (delta / (11 - torch.log(p))).pow(2).mean()
    loss_logsumexp = torch.logsumexp(input=torch.abs(delta), dim=(0, 1, 2))
    return loss_linf, loss_mae, loss_mse, loss_scaled_mse, loss_logsumexp


def test_on_wholedataset(
    file_name,
    data_path,
    output_path,
    output_file,
    logger,
    model,
    device="cuda",
    variable="z",
):
    ds = xr.open_dataset(f"{data_path}/{file_name}")
    ds_pred = xr.zeros_like(ds[variable]) - 9999
    ds = ds.assign_coords(time=ds.time.dt.dayofyear - 1)
    dtype = model.input_type
    lat = torch.tensor(ds.latitude.to_numpy(), dtype=dtype, device=device)
    lon = torch.tensor(ds.longitude.to_numpy(), dtype=dtype, device=device)
    ps = ds.level.to_numpy().astype(float)
    ts = ds.time.to_numpy().astype(float)
    model = model.to(device)
    metrics = list()
    for j in trange(ps.shape[0]):
        preds = list()
        targets = list()
        for i in range(ts.shape[0]):
            ti = float(ts[i])
            pj = float(ps[j])
            t = torch.tensor([ti], dtype=dtype, device=device)
            p = torch.tensor([pj], dtype=dtype, device=device)
            coord = (
                torch.stack(torch.meshgrid(t, p, lat, lon, indexing="ij"), dim=-1)
                .squeeze(0)
                .squeeze(0)
            )
            with torch.no_grad():
                var_pred = model(coord)
                ds_pred.data[i, j, :, :] = var_pred.cpu().numpy().squeeze(-1)
                pred = ds_pred.data[i, j, :, :]
                target = ds[variable][i, j, :, :]
                preds.append(pred)
                targets.append(target)
        pred_data = np.array(preds)
        target_data = xr.concat(targets, "time")
        p_metrics = utils.metrics.compute_evaluation_metrics(pred_data, target_data)
        logger.log_metrics({f"{key}_p{int(pj)}": val for key, val in p_metrics.items()})
        metrics.append(p_metrics)

    logger.log_metrics(utils.metrics.compute_mean_metrics(metrics))

    print(f"Saving dataset to {output_path}/{output_file}")
    ds_pred.to_netcdf(f"{output_path}/{output_file}")


def generate_outputs(model, output_path, output_file, device="cuda"):
    file_name = model.args.file_name
    data_path = model.args.data_path
    variable = model.args.variable  # "z"
    ds = xr.open_mfdataset(f"{data_path}/{file_name}").load()
    out_ds = xr.zeros_like(ds)

    mean = ds[variable].mean(dim=["time"]).to_numpy()
    std = (ds[variable].max(dim=["time"]) - ds[variable].min(dim=["time"])).to_numpy()
    assert len(ds[variable].shape) == 3
    lon_v = torch.as_tensor(ds.lon.to_numpy(), device=device, dtype=torch.float32)
    lat_v = torch.as_tensor(ds.lat.to_numpy(), device=device, dtype=torch.float32)
    lat, lon = torch.meshgrid((lat_v, lon_v), indexing="ij")
    p = torch.zeros_like(lat, device=device) + float(ds.level.mean())
    t = torch.zeros_like(lat, device=device)
    model = model.to(device)
    errors = np.zeros(len(ds.time))
    for it in tqdm(range(len(ds.time))):
        coord = torch.stack((t + it, p, lat, lon), dim=-1)
        with torch.no_grad():
            var_pred = model(coord).squeeze(-1).cpu().numpy() * 0.5 * 1.4 * std + mean
            out_ds[variable].data[it, :, :] = var_pred[:, :]
            var = ds[variable].isel(time=it).to_numpy()
            errors[it] = np.abs(var_pred - var).max()
    file_name = f"{output_path}/{output_file}"
    print(f"Saving to {file_name}")
    out_ds.to_netcdf(file_name)
    print(errors.max())


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    set_all_seeds(args.seed)

    model = FitNetModule(args)

    bitorchinfo.summary(model, input_size=(args.batch_size, 1, 4, 4))

    run_name = f"w{args.width}_b{args.batch_size}_fp{args.model_precision}_nf{args.nfeature}_sf{args.sigma}_tf{args.trainable_fourierfeature}_s{args.seed}"
    args.run_name = run_name
    args.output_file = f"dataset1_{run_name}.nc"

    if args.wandb_sweep:
        args.use_wandb = True
        with open(f"./{args.wandb_sweep_config_name}.yml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        run = wandb.init(config=config, name=args.run_name)

    if args.use_wandb:
        logger = WandbLogger(
            project=args.project_name, name=args.run_name, save_dir=args.log_dir
        )
    else:
        logger = CSVLogger(name=args.run_name, save_dir=args.log_dir)

    if args.ckpt_path != "":
        model_loaded = FitNetModule.load_from_checkpoint(args.ckpt_path)
        model.model.load_state_dict(model_loaded.model.state_dict())

    trainer = None
    if not args.notraining:
        strategy = pl.strategies.DataParallelStrategy()
        trainer = pl.Trainer(
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=1,
            accelerator="gpu",
            auto_select_gpus=True,
            devices=args.num_gpu,
            strategy=strategy,
            logger=logger,
            min_epochs=10,
            max_epochs=args.nepoches,
            gradient_clip_val=0.5,
            sync_batchnorm=True,
            precision=args.model_precision,
        )
        trainer.fit(model)

    model.eval()
    if (not trainer) or trainer.is_global_zero:
        print("Model size (MB):", get_model_size_mb(model))

    if args.quantizing:
        model.model.fcs = model.model.fcs.half()
        quantized_size = get_model_size_mb(model)
        model.model.fcs = model.model.fcs.float()
        print(f"Quantized (FP16) size (MB): {quantized_size}")

    if args.testing and ((not trainer) or trainer.is_global_zero):
        test_on_wholedataset(
            model.args.file_name,
            model.args.data_path,
            model.args.output_path,
            model.args.output_file,
            logger,
            model,
            variable=model.args.variable,
        )

    if args.generate_full_outputs and ((not trainer) or trainer.is_global_zero):
        generate_outputs(model, args.output_path, args.output_file)

    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project_name", default="padl23t2_experiments", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--num_gpu", default=-1, type=int)
    parser.add_argument("--nepoches", default=30, type=int)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--num_workers", default=20, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--model_precision", default=32, type=int)
    parser.add_argument("--sigma", default=1.6, type=float)
    parser.add_argument("--nfeature", default=128, type=int)
    parser.add_argument("--ntfeature", default=16, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--tscale", default=60.0, type=float)
    parser.add_argument("--zscale", default=100.0, type=float)
    parser.add_argument("--variable", default="z", type=str)
    parser.add_argument("--dataloader_mode", default="sampling_nc", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--scheduler", default="reducelro", type=str)
    parser.add_argument("--data_path", default=".", type=str)
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--ckpt_path", default="", type=str)
    parser.add_argument("--wandb_sweep", action="store_true")
    parser.add_argument("--wandb_sweep_config_name", default="sweep_config", type=str)
    parser.add_argument("--use_quantized_linear_layer", action="store_true")
    parser.add_argument("--q_bits", default=2, type=int)
    parser.add_argument("--use_batchnorm", action="store_true")
    parser.add_argument("--use_skipconnect", action="store_true")
    parser.add_argument("--use_invscale", action="store_true")
    parser.add_argument("--use_fourierfeature", action="store_true")
    parser.add_argument("--trainable_fourierfeature", default=False, type=bool)
    parser.add_argument("--concat_input", default=False, type=bool)
    parser.add_argument("--tresolution", default=24, type=float)
    parser.add_argument("--use_xyztransform", action="store_true")
    parser.add_argument("--use_stat", action="store_true")
    parser.add_argument("--loss_type", default="mse", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--generate_full_outputs", action="store_true")
    parser.add_argument("--output_path", default=".", type=str)
    parser.add_argument("--output_file", default="output.nc", type=str)
    parser.add_argument("--notraining", action="store_true")
    parser.add_argument("--quantizing", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--log_dir", default="../logs", type=str)
    parser.add_argument("--seed", default=1111, type=int)
    args = parser.parse_args()
    if args.all:
        args.use_batchnorm = True
        args.use_invscale = not args.use_stat
        args.use_skipconnect = True
        args.use_xyztransform = True
        args.use_fourierfeature = True
    if args.trainable_fourierfeature:
        args.concat_input = True
    main(args)
