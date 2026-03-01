import lightning.pytorch as L
import torch
import torch.nn as nn
import numpy as np
from math import floor, log10

from guess_llm.llm_utils.llm_no_scaling import (
    serialize,
    get_hidden_states_no_scaling,
    llama_generate_samples,
)
from guess_llm.utils.utils import build_mlp
import torch.nn.functional as F

EPS = 1e-8


class LLMPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.y_mean = config.y_mean
        self.y_std = config.y_std
        self.y_greedy_mean = config.y_greedy_mean
        self.y_greedy_std = config.y_greedy_std
        if config.log_scaling:
            self.y_min = getattr(config, "y_min", 0)
            self.y_greedy_min = config.y_greedy_min
        self.agg_type = getattr(config, "agg_type", "concat")

        if (
            hasattr(config, "model_name")
            and config.model_name == "Phi-3.5-mini-instruct"
        ):
            dim_model = 3072
        elif hasattr(config, "model_name") and config.model_name == "Llama-3.2-3B":
            dim_model = 3072
        else:
            dim_model = 4096

        if self.agg_type == "concat":
            self.input_dim = dim_model * config.n_hidden_states
        else:
            self.input_dim = dim_model

        self.feature_extractor = build_mlp(
            self.input_dim, config.hidden_layers, config.hidden_dim
        )

        if self.agg_type == "weighted_mean":
            self.agg_weights = nn.Parameter(torch.ones(config.n_hidden_states))
            self.agg_weights.data /= self.agg_weights.sum()

        self.hidden_states_list = config.hidden_states_list

    def get_features(self, x):
        # x.shape = [batch_size, n_hidden_states, 4096]
        if self.agg_type == "concat":
            x = x.reshape(x.shape[0], -1)
        elif self.agg_type == "weighted_mean":
            x = torch.einsum("ijk,j->ik", x, self.agg_weights)
        # x.shape = [batch_size, 4096]
        hidden_rep = self.feature_extractor(x)
        return hidden_rep

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method")

    def predict_from_raw(self, llm, tokenizer, train, precision, device="cuda:0"):
        input_str = serialize(train, precision=precision)
        hidden_states = get_hidden_states_no_scaling(
            model=llm, tokenizer=tokenizer, input_str=input_str
        )
        x = [
            torch.tensor(hidden_states[i], dtype=torch.float32)
            for i in self.hidden_states_list
        ]
        x = (
            torch.stack(x, dim=1).T.to(device).unsqueeze(0)
        )  # shape: [1, n_hidden_states, 4096]
        return self.predict(x)

    def predict_from_raw_with_llm(
        self, llm, tokenizer, train, n_samples, precision, temp=0.9, top_p=0.9
    ):
        input_str = serialize(train, precision=precision)
        return llama_generate_samples(
            model=llm,
            tokenizer=tokenizer,
            input_str=input_str,
            n_samples=n_samples,
            temp=temp,
            top_p=top_p,
        )


class LitWrapper(L.core.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class MagnitudeHead(nn.Module):
    def __init__(self, input_dim, min_mag, max_mag, use_arctan=False):
        super(MagnitudeHead, self).__init__()
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.register_buffer(
            "exponent_mapping", torch.arange(self.min_mag, self.max_mag + 1).float()
        )
        self.num_classes = len(self.exponent_mapping)
        self.class_head = nn.Linear(input_dim, self.num_classes)
        self.reg_head = nn.Linear(input_dim, 1)
        self.use_arctan = use_arctan

    def forward(self, x):
        # Classification logits
        pred_logits = self.class_head(x)  # Shape: [B, 7]

        pred_class = torch.argmax(pred_logits, dim=1)  # [B]

        # Map predicted class index to exponent
        exponents = self.exponent_mapping[pred_class]  # [B]
        scale = 10.0 ** exponents.unsqueeze(1)  # [B, 1]

        # Regression output
        if self.use_arctan:
            pred_reg = 10 * torch.arctan(0.5 * self.reg_head(x))  # [B, 1]
        else:
            pred_reg = self.reg_head(x)
        final_pred = pred_reg * scale  # Rescale to original y scale

        return {
            "final_pred": final_pred,
            "pred_reg": pred_reg,
            "pred_logits": pred_logits,
            "pred_class": pred_class,
        }


class SeparateMagRegHead(nn.Module):
    def __init__(
        self, input_dim, hidden_layers, hidden_dim, min_mag, max_mag, use_arctan=False
    ):
        super(SeparateMagRegHead, self).__init__()
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.register_buffer(
            "exponent_mapping", torch.arange(self.min_mag, self.max_mag + 1).float()
        )
        self.num_classes = len(self.exponent_mapping)
        self.class_head = build_mlp(
            input_dim, hidden_layers, hidden_dim, self.num_classes
        )
        self.reg_head = build_mlp((input_dim + 1), hidden_layers, hidden_dim, 1)
        self.use_arctan = use_arctan

    def forward(self, x):
        # Classification logits
        pred_logits = self.class_head(x)  # Shape: [B, 7]

        pred_class = torch.argmax(pred_logits, dim=1)  # [B]

        # Regression output
        b, input_dim = x.shape  # x: [B, input_dim]
        num_classes = self.exponent_mapping.shape[
            0
        ]  # typically 8 for exponents -3 to 4
        x_repeated = x.unsqueeze(1).repeat(1, num_classes, 1).view(-1, input_dim)
        exponents = self.exponent_mapping.view(1, -1).repeat(b, 1).view(-1, 1)
        powers = torch.pow(10, exponents)
        x_reg = torch.cat([x_repeated, powers], dim=1)

        if self.use_arctan:
            pred_reg = 10 * torch.arctan(
                0.5 * self.reg_head(x_reg)
            )  # [B*num_classes, 1]
        else:
            pred_reg = self.reg_head(x_reg)

        pred_reg = pred_reg.view(b, num_classes)  # [B, num_classes]
        final_pred = pred_reg * torch.pow(
            10.0, self.exponent_mapping
        )  # Rescale to correct y scale

        return {
            "final_pred": final_pred,
            "pred_reg": pred_reg,
            "pred_logits": pred_logits,
            "pred_class": pred_class,
        }


class ConditionalMagnitudeHead(nn.Module):
    def __init__(self, input_dim, min_mag, max_mag, use_arctan=False):
        super(ConditionalMagnitudeHead, self).__init__()
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.register_buffer(
            "exponent_mapping", torch.arange(self.min_mag, self.max_mag + 1).float()
        )
        self.num_classes = len(self.exponent_mapping)
        self.class_head = nn.Linear(input_dim, self.num_classes)
        self.reg_head = nn.Linear((input_dim + 1), 1)
        self.use_arctan = use_arctan

    def forward(self, x):
        # Classification logits
        pred_logits = self.class_head(x)  # Shape: [B, 7]

        pred_class = torch.argmax(pred_logits, dim=1)  # [B]

        # Regression output
        b, input_dim = x.shape  # x: [B, input_dim]
        num_classes = self.exponent_mapping.shape[
            0
        ]  # typically 8 for exponents -3 to 4
        x_repeated = x.unsqueeze(1).repeat(1, num_classes, 1).view(-1, input_dim)
        exponents = self.exponent_mapping.view(1, -1).repeat(b, 1).view(-1, 1)
        x_reg = torch.cat([x_repeated, exponents], dim=1)

        if self.use_arctan:
            pred_reg = 10 * torch.arctan(
                0.5 * self.reg_head(x_reg)
            )  # [B*num_classes, 1]
        else:
            pred_reg = self.reg_head(x_reg)

        pred_reg = pred_reg.view(b, num_classes)  # [B, num_classes]
        final_pred = pred_reg * torch.pow(
            10.0, self.exponent_mapping
        )  # Rescale to correct y scale

        return {
            "final_pred": final_pred,
            "pred_reg": pred_reg,
            "pred_logits": pred_logits,
            "pred_class": pred_class,
        }


class MagnitudeRegressionPredictor(LLMPredictor):
    def __init__(self, config):
        if hasattr(config, "type") and config.type == "separate":
            original_hidden_layers = config.hidden_layers
            config.hidden_layers = 0  # No hidden layers in the feature extractor model
        super().__init__(config)
        if hasattr(config, "type") and config.type == "separate":
            config.hidden_layers = original_hidden_layers

        self.target = config.target
        self.alpha = config.alpha
        self.beta = config.beta if hasattr(config, "beta") else 1.0
        self.min_mag = config.magnitudes[0]
        self.max_mag = config.magnitudes[1]
        self.use_arctan = config.use_arctan
        self.gate_annealing = True  # config.gate_annealing
        self.gate = 1.0  # config.gate
        self.T = 250  # config.T
        self.k = 1000  # config.k
        self.topk = config.topk if hasattr(config, "topk") else 0

        if hasattr(config, "type"):
            self.type = config.type
        elif hasattr(config, "conditional"):
            self.type = "conditional" if config.conditional else "standard"
        else:
            self.type = "standard"

        if config.hidden_layers == 0 or self.type == "separate":
            input_dim = self.input_dim
        else:
            input_dim = config.hidden_dim

        if self.type == "conditional":
            self.magnitude_reg_head = ConditionalMagnitudeHead(
                input_dim=input_dim,
                min_mag=self.min_mag,
                max_mag=self.max_mag,
                use_arctan=self.use_arctan,
            )
        elif self.type == "separate":
            self.magnitude_reg_head = SeparateMagRegHead(
                input_dim=input_dim,
                hidden_layers=config.hidden_layers,
                hidden_dim=config.hidden_dim,
                min_mag=self.min_mag,
                max_mag=self.max_mag,
                use_arctan=self.use_arctan,
            )
        else:
            self.magnitude_reg_head = MagnitudeHead(
                input_dim=input_dim,
                min_mag=self.min_mag,
                max_mag=self.max_mag,
                use_arctan=self.use_arctan,
            )

    def forward(self, x):
        hidden_rep = self.get_features(x)
        output = self.magnitude_reg_head(hidden_rep)
        if self.type == "conditional" or self.type == "separate":
            final_pred = output["final_pred"].gather(
                1, output["pred_class"].unsqueeze(1)
            )
        else:
            final_pred = output["final_pred"]
        return final_pred, output["pred_logits"], output["pred_reg"]

    def loss_function(
        self, pred_logits, pred_reg, y_greedy, y_pred, mean_reduce=True, step=0.0
    ):
        if self.target == "greedy":
            y = y_greedy
        elif self.target == "median":
            y = y_pred.median(axis=1, keepdim=True)[0]
        elif self.target == "mean":
            y = y_pred.mean(axis=1, keepdim=True)
        else:
            raise ValueError(f"Unknown target type: {self.target}")

        true_order = torch.floor(torch.log10(torch.abs(y))).clamp(
            min=self.min_mag, max=self.max_mag
        )
        class_indices = (true_order - self.min_mag).long().squeeze(-1)
        classification_loss = F.cross_entropy(
            pred_logits, class_indices, reduction="none"
        ).unsqueeze(1)

        pred_class = torch.argmax(pred_logits, dim=1)
        true_class = (true_order - self.min_mag).long().squeeze(-1)
        pred_order = self.magnitude_reg_head.exponent_mapping[pred_class].unsqueeze(1)
        pred_scale = torch.pow(10.0, pred_order)

        if self.type == "conditional" or self.type == "separate":
            pred_reg_top = pred_reg.gather(1, pred_class.unsqueeze(1))
        else:
            pred_reg_top = pred_reg

        # Loss of the top predicted class
        accuracy_loss = (pred_class == true_class).float()
        true_scale = torch.pow(10.0, true_order)
        regression_loss = F.mse_loss(pred_reg_top, y / true_scale, reduction="none")
        final_loss = F.mse_loss(pred_reg_top, y / pred_scale, reduction="none")

        # Weighted final_loss
        if self.type == "conditional":
            # Smooth gating mechanism
            if self.gate_annealing:
                gate = 1.0 if step < self.T else np.exp(-(step - self.T) / self.k)
            else:
                gate = self.gate

            topk_logits, topk_indices = torch.topk(
                pred_logits, k=self.topk, dim=1
            )  # both (batch_size, 3)
            topk_probs = F.softmax(topk_logits, dim=1)  # (batch_size, 3)
            topk_exponents = self.magnitude_reg_head.exponent_mapping[
                topk_indices
            ]  # (batch_size, 3)
            topk_scale = torch.pow(10.0, topk_exponents)
            topk_reg = pred_reg.gather(1, topk_indices)  # (batch_size, 3)

            topk_final_loss = F.mse_loss(topk_reg, y / topk_scale, reduction="none")
            topk_final_loss = (topk_final_loss * topk_probs).sum(dim=1)  # (batch_size,)

            total_loss = classification_loss + self.alpha * (1 - gate) * topk_final_loss

        elif self.type == "standard" or self.type == "separate":
            total_loss = self.alpha * classification_loss + self.beta * regression_loss

        if self.type == "conditional" or self.type == "separate":
            topk_logits, topk_indices = torch.topk(pred_logits, k=self.topk, dim=1)
            topk_probs = F.softmax(topk_logits, dim=1)  # (batch_size, 3)

            pred_scaled = pred_reg * torch.pow(
                10.0, self.magnitude_reg_head.exponent_mapping
            )
            topk_pred_scaled = pred_scaled.gather(1, topk_indices)  # (batch_size, 3)
            expected_pred = (topk_pred_scaled * topk_probs).sum(dim=1).unsqueeze(1)
            expected_loss = F.mse_loss(expected_pred, y, reduction="none")
        else:
            expected_loss = final_loss

        if mean_reduce:
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            final_loss = final_loss.mean()
            total_loss = total_loss.mean()
            expected_loss = expected_loss.mean()
            accuracy_loss = accuracy_loss.mean()
            if self.type == "conditional":
                topk_final_loss = topk_final_loss.mean()

        loss = {
            "classification_loss": classification_loss,
            "regression_loss": regression_loss,
            "final_loss": final_loss,
            "total_loss": total_loss,
            "expected_loss": expected_loss,
            "accuracy_loss": accuracy_loss,
        }

        if self.type == "conditional":
            loss["topk_final_loss"] = topk_final_loss

        return loss

    def predict(self, x):
        """Returns the greedy prediction"""
        final_pred, pred_logits, pred_reg = self.forward(x)

        return final_pred

    def predict_all(self, x, k=3):
        final_pred, pred_logits, pred_reg = self.forward(x)

        pred_class = torch.argmax(pred_logits, dim=1)
        pred_order = self.magnitude_reg_head.exponent_mapping[pred_class].unsqueeze(1)
        pred_scale = torch.pow(10.0, pred_order)

        # Get the top k predictions
        topk_logits, topk_indices = torch.topk(pred_logits, k=k, dim=1)
        topk_probs = F.softmax(topk_logits, dim=1)  # (batch_size, 3)

        if self.type == "conditional" or self.type == "separate":
            pred_scaled = pred_reg * torch.pow(
                10.0, self.magnitude_reg_head.exponent_mapping
            )
            topk_pred_scaled = pred_scaled.gather(1, topk_indices)  # (batch_size, 3)
            expected_pred = (topk_pred_scaled * topk_probs).sum(
                dim=1
            )  # (batch_size, 1)
        elif self.type == "standard":
            topk_exponents = self.magnitude_reg_head.exponent_mapping[
                topk_indices
            ]  # (batch_size, 3)
            expected_exponent = (topk_probs * topk_exponents).sum(
                dim=1, keepdim=True
            )  # (batch_size, 1)
            expected_scale = torch.pow(10.0, expected_exponent)
            expected_pred = pred_reg * expected_scale

        return {
            "final_pred": final_pred,
            "pred_logits": pred_logits,
            "pred_reg": pred_reg,
            "pred_order": pred_order,
            "pred_scale": pred_scale,
            "expected_pred": expected_pred,
        }

    def predict_expected(self, x):
        """Returns the expected prediction"""
        final_pred, pred_logits, pred_reg = self.forward(x)

        # Get the top k predictions
        topk_logits, topk_indices = torch.topk(pred_logits, k=self.topk, dim=1)
        topk_probs = F.softmax(topk_logits, dim=1)  # (batch_size, 3)

        if self.type == "conditional" or self.type == "separate":
            pred_scaled = pred_reg * torch.pow(
                10.0, self.magnitude_reg_head.exponent_mapping
            )
            topk_pred_scaled = pred_scaled.gather(1, topk_indices)  # (batch_size, 3)
            expected_pred = (topk_pred_scaled * topk_probs).sum(
                dim=1
            )  # (batch_size, 1)
        elif self.type == "standard":
            topk_exponents = self.magnitude_reg_head.exponent_mapping[
                topk_indices
            ]  # (batch_size, 3)
            expected_exponent = (topk_probs * topk_exponents).sum(
                dim=1, keepdim=True
            )  # (batch_size, 1)
            expected_scale = torch.pow(10.0, expected_exponent)
            expected_pred = pred_reg * expected_scale

        return expected_pred

    def predict_n_digits(self, y_greedy, n=2):
        """Returns the prediction obtained by taking the first n digits of the greedy outcome"""

        def truncate_to_first_n_digits(x, n):
            if x == 0:
                return 0  # Special case

            sign = -1 if x < 0 else 1
            x_abs = abs(x)

            if abs(x) < 1:
                truncated = int(x_abs * 10 ** (n - 1)) / 10 ** (n - 1)
                return sign * truncated

            # Scale the number to bring the first digit to the units place
            order = int(floor(log10(x_abs)))
            scale = 10 ** (order - n + 1)
            truncated = int(x_abs / scale) * scale

            return sign * truncated

        y_pred = (
            torch.tensor([truncate_to_first_n_digits(y, n) for y in y_greedy])
            .unsqueeze(-1)
            .to(y_greedy.device)
        )
        return y_pred

    def freeze_regression_head(self):
        # Freeze regression head
        for param in self.magnitude_reg_head.reg_head.parameters():
            param.requires_grad = False

    def unfreeze_regression_head(self):
        # Unfreeze regression head
        for param in self.magnitude_reg_head.reg_head.parameters():
            param.requires_grad = True

    def freeze_classification_head(self):
        # Freeze classification head
        for param in self.magnitude_reg_head.class_head.parameters():
            param.requires_grad = False

    def unfreeze_classification_head(self):
        # Unfreeze classification head
        for param in self.magnitude_reg_head.class_head.parameters():
            param.requires_grad = True


class LitMagnitudeRegressionPredictor(LitWrapper):
    def __init__(self, config):
        model = MagnitudeRegressionPredictor(config)
        super().__init__(config=config, model=model)
        self.config = config

    def training_step(self, batch, batch_idx):
        x, y_greedy, y_pred = batch["x"], batch["y_greedy"], batch["y"]
        final_pred, pred_logits, pred_reg = self.model(x)

        opt = self.optimizers(use_pl_optimizer=True)
        step_counts = [
            opt.state[p]["step"]
            for group in opt.param_groups
            for p in group["params"]
            if "step" in opt.state[p]
        ]
        current_step = max(step_counts) if step_counts else 0

        loss = self.model.loss_function(
            pred_logits, pred_reg, y_greedy, y_pred, step=current_step
        )

        self.log("train_loss", loss["total_loss"], prog_bar=True)

        # Calculate the original loss
        if self.model.target == "greedy":
            y = y_greedy
        elif self.model.target == "median":
            y = y_pred.median(axis=1, keepdim=True)[0]
        elif self.model.target == "mean":
            y = y_pred.mean(axis=1, keepdim=True)

        orig_loss = F.mse_loss(final_pred, y, reduction="mean")

        self.log("train_orig_loss", orig_loss, prog_bar=False)
        self.log("train_class_loss", loss["classification_loss"], prog_bar=False)
        self.log("current_step", current_step, prog_bar=True)
        # self.log("train_gate", loss['gate'], prog_bar=False)
        self.log("train_reg_loss", loss["regression_loss"], prog_bar=False)
        self.log("train_final_loss", loss["final_loss"], prog_bar=False)
        self.log("train_expected_loss", loss["expected_loss"], prog_bar=False)
        self.log("train_accuracy", loss["accuracy_loss"], prog_bar=False)

        return loss["total_loss"]

    def validation_step(self, batch, batch_idx):
        x, y_greedy, y_pred = batch["x"], batch["y_greedy"], batch["y"]
        final_pred, pred_logits, pred_reg = self.model(x)

        opt = self.optimizers(use_pl_optimizer=True)
        step_counts = [
            opt.state[p]["step"]
            for group in opt.param_groups
            for p in group["params"]
            if "step" in opt.state[p]
        ]
        current_step = max(step_counts) if step_counts else 0

        loss = self.model.loss_function(
            pred_logits, pred_reg, y_greedy, y_pred, step=current_step
        )

        # Calculate the original loss
        if self.model.target == "greedy":
            y = y_greedy
        elif self.model.target == "median":
            y = y_pred.median(axis=1, keepdim=True)[0]
        elif self.model.target == "mean":
            y = y_pred.mean(axis=1, keepdim=True)

        orig_loss = F.mse_loss(final_pred, y)

        self.log("val_loss", loss["total_loss"], prog_bar=True)
        self.log("val_orig_loss", orig_loss, prog_bar=False)
        self.log("val_class_loss", loss["classification_loss"], prog_bar=False)
        self.log("val_reg_loss", loss["regression_loss"], prog_bar=False)
        self.log("val_final_loss", loss["final_loss"], prog_bar=False)
        self.log("val_expected_loss", loss["expected_loss"], prog_bar=False)
        self.log("val_accuracy", loss["accuracy_loss"], prog_bar=False)

        return loss["total_loss"]

    def loss_function(self, pred_logits, pred_reg, y_greedy, y_pred):
        return self.model.loss_function(pred_logits, pred_reg, y_greedy, y_pred)

    def predict(self, x):
        """Returns the greedy prediction"""
        return self.model.predict(x)

    def predict_n_digits(self, y_greedy, n=2):
        """Returns the prediction obtained by taking the first n digits of the greedy outcome"""
        return self.model.predict_n_digits(y_greedy, n)


class QuantilePredictor(LLMPredictor):
    def __init__(self, config):
        super().__init__(config)
        self.quantiles = config.quantiles
        self.n_bootstrap = getattr(config, "n_bootstrap", 0)
        self.instance_normalization = getattr(config, "instance_normalization", False)
        self.log_scaling = getattr(config, "log_scaling", False)
        self.standard_scaling = getattr(config, "standard_scaling", True)
        self.median_mse_lambda = getattr(config, "median_mse_lambda", 0.0)

        self.ci_dict = {
            0.5: (0.25, 0.75),
            0.8: (0.1, 0.9),
            0.9: (0.05, 0.95),
            0.95: (0.025, 0.975),
        }

        self.median_idx = self.quantiles.index(0.5)
        self.lower_quantiles = [q for q in self.quantiles if q < 0.5]
        self.upper_quantiles = [q for q in self.quantiles if q > 0.5]

        if config.hidden_layers == 0:
            input_dim = self.input_dim
        else:
            input_dim = config.hidden_dim

        self.quantile_head = nn.Linear(input_dim, len(config.quantiles))

        # Quantile weights for weighted loss
        # Check if median weight exists in config
        if hasattr(config, "median_weight") and hasattr(config, "quantile_weights"):
            raise ValueError("Cannot set both median_weight and quantile_weights")
        if hasattr(config, "median_weight"):
            quantile_weights = torch.ones(len(config.quantiles))
            quantile_weights[self.median_idx] = config.median_weight
        elif hasattr(config, "quantile_weights"):
            quantile_weights = torch.tensor(
                config.quantile_weights, dtype=torch.float32
            )
            if len(quantile_weights) != len(config.quantiles):
                raise ValueError(
                    "Length of quantile_weights must match length of quantiles"
                )
        else:
            # Default to equal weights
            quantile_weights = torch.ones(len(config.quantiles))

        self.register_buffer(
            "quantile_weights", quantile_weights / quantile_weights.sum()
        )

    def transform(self, y):
        if self.log_scaling:
            y = torch.log(y - self.y_min + 1e-3)
        if self.standard_scaling:
            y = (y - self.y_mean) / self.y_std
        return y

    def inverse_transform(self, y):
        if self.standard_scaling:
            y = y * self.y_std + self.y_mean
        if self.log_scaling:
            y = torch.exp(y) - 1e-3 + self.y_min
        return y

    def forward(self, x):
        hidden = self.get_features(x)
        quantiles = self.quantile_head(hidden)
        return quantiles

    def loss_function(self, pred_quantiles, y):
        """Calculate the quantile loss"""
        y = self.transform(y)
        if self.n_bootstrap == 0:
            return self.pinball_loss(pred_quantiles, y)
        elif self.n_bootstrap > 0:
            losses = []
            for i in range(self.n_bootstrap):
                # Bootstrap sampling
                bootstrap_indices = np.random.choice(
                    y.shape[1], y.shape[1], replace=True
                )
                y_bootstrap = y[:, bootstrap_indices]
                losses.append(self.pinball_loss(pred_quantiles, y_bootstrap))
            loss = torch.stack(losses, dim=0).mean(dim=0)
        return loss

    def pinball_loss(self, pred_quantiles, y):
        if self.instance_normalization:
            # I didn't get to make this work, testing different ways of normalizing.
            # Keep it for now, but should be removed in the future if not used.
            center = torch.median(y, dim=1, keepdim=True)[
                0
            ]
            scale = y.std(dim=1, keepdim=True)
            y = (y - center) / (scale + 1e-1)
            pred_quantiles = (pred_quantiles - center) / (scale + 1e-1)
        loss = self.get_quantile_loss(pred_quantiles, y)
        if self.median_mse_lambda > 0:
            median_mse_loss = self.get_median_mse_loss(pred_quantiles, y)
            loss = loss + self.median_mse_lambda * median_mse_loss
        return loss

    def get_quantile_loss(self, pred_quantiles, y):
        """Calculate the quantile loss"""
        # Calculate the quantile loss
        quantile_loss = torch.zeros_like(pred_quantiles)
        for i, q in enumerate(self.quantiles):
            # entry-wise maximum
            loss = torch.max(
                (q - 1) * (y - pred_quantiles[:, [i]]), q * (y - pred_quantiles[:, [i]])
            )
            # average across all y samples
            loss = loss.mean(dim=1)
            quantile_loss[:, i] = loss
        # weighted sum across all quantiles
        quantile_loss = quantile_loss * self.quantile_weights
        quantile_loss = quantile_loss.sum(dim=1)
        return quantile_loss.mean()

    def get_median_mse_loss(self, pred_quantiles, y):
        """Calculate the MSE loss for the median quantile"""
        pred_median = pred_quantiles[:, self.quantiles.index(0.5)]
        sample_median = torch.median(y, dim=1)[0]
        mse_loss = (pred_median - sample_median) ** 2
        return (
            mse_loss.mean(),
            torch.median(mse_loss),
            torch.median(mse_loss / (sample_median + 1e-3)),
        )

    def predict(self, x):
        """Returns quantile predictions"""
        raw_quantiles = self.forward(x)
        quantiles = self.inverse_transform(raw_quantiles)
        return quantiles

    def get_ci_coverage(self, pred_quantiles, y, ci_level=0.95):
        """Calculate the coverage of the confidence interval"""
        y = self.transform(y)
        if ci_level in self.ci_dict:
            lower_bound = pred_quantiles[
                :, [self.quantiles.index(self.ci_dict[ci_level][0])]
            ]
            upper_bound = pred_quantiles[
                :, [self.quantiles.index(self.ci_dict[ci_level][1])]
            ]
            coverage = ((y >= lower_bound) & (y <= upper_bound)).float().mean(dim=1)
        else:
            raise ValueError(f"Unknown CI level: {ci_level}")
        return coverage


class LitQuantilePredictor(LitWrapper):
    def __init__(self, config):
        model = QuantilePredictor(config)
        super().__init__(config=config, model=model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        pred_quantiles = self.model(x)
        loss = self.model.loss_function(pred_quantiles, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        pred_quantiles = self.model(x)
        loss = self.model.loss_function(pred_quantiles, y)
        self.log("val_loss", loss, prog_bar=True)
        mean_median_mse_loss, median_median_mse_loss, median_rel_median_mse_loss = (
            self.model.get_median_mse_loss(pred_quantiles, self.model.transform(y))
        )
        self.log("val_median_mse_loss", mean_median_mse_loss, prog_bar=False)
        self.log("val_median_median_mse_loss", median_median_mse_loss, prog_bar=False)
        self.log(
            "val_median_rel_median_mse_loss", median_rel_median_mse_loss, prog_bar=False
        )
        (
            orig_mean_median_mse_loss,
            orig_median_median_mse_loss,
            orig_median_rel_median_mse_loss,
        ) = self.model.get_median_mse_loss(
            self.model.inverse_transform(pred_quantiles), y
        )
        self.log(
            "val_orig_mean_median_mse_loss", orig_mean_median_mse_loss, prog_bar=False
        )
        self.log(
            "val_orig_median_median_mse_loss",
            orig_median_median_mse_loss,
            prog_bar=False,
        )
        self.log(
            "val_orig_median_rel_median_mse_loss",
            orig_median_rel_median_mse_loss,
            prog_bar=False,
        )

        for ci_level in self.model.ci_dict.keys():
            lower, upper = self.model.ci_dict[ci_level]
            if lower in self.model.quantiles and upper in self.model.quantiles:
                coverage = self.model.get_ci_coverage(
                    pred_quantiles, y, ci_level=ci_level
                )
                self.log(f"val_{ci_level}_coverage", coverage.mean(), prog_bar=False)
                coverage_abs_error = torch.abs(coverage - ci_level).mean()
                self.log(
                    f"val_{ci_level}_coverage_abs_error",
                    coverage_abs_error,
                    prog_bar=False,
                )

        return loss


class QuantileMagnitudePredictor(QuantilePredictor):
    def __init__(self, config):
        super().__init__(config)
        if config.hidden_layers == 0:
            input_dim = self.input_dim
        else:
            input_dim = config.hidden_dim
        self.min_mag = config.magnitudes[0]
        self.max_mag = config.magnitudes[1]
        self.alpha = config.alpha
        self.beta = config.beta
        self.magnitude_heads = nn.ModuleList(
            [
                MagnitudeHead(
                    input_dim=input_dim,
                    min_mag=self.min_mag,
                    max_mag=self.max_mag,
                    use_arctan=config.use_arctan,
                )
                for _ in range(len(config.quantiles))
            ]
        )
        self.exponent_mapping = self.magnitude_heads[0].exponent_mapping

    def forward(self, x):
        hidden = self.get_features(x)
        quantiles_reg = []
        quantiles_order_logits = []
        quantiles_final = []
        for i, head in enumerate(self.magnitude_heads):
            pred = head(hidden)
            quantiles_final.append(pred["final_pred"])
            quantiles_reg.append(pred["pred_reg"])
            quantiles_order_logits.append(pred["pred_logits"])
        quantiles_final = torch.stack(quantiles_final, dim=1).squeeze(-1)
        quantiles_reg = torch.stack(quantiles_reg, dim=1).squeeze(-1)
        quantiles_order_logits = torch.stack(quantiles_order_logits, dim=1).squeeze(-1)
        return {
            "quantiles_final": quantiles_final,
            "quantiles_reg": quantiles_reg,
            "quantiles_order_logits": quantiles_order_logits,
        }

    def predict(self, x):
        """Returns quantile predictions"""
        out = self.forward(x)
        quantiles_final = out["quantiles_final"]
        return quantiles_final

    def predict_expected(self, x, k=3):
        out = self.forward(x)
        quantiles_reg = out["quantiles_reg"]  # [BS, num_quantiles]
        quantiles_order_logits = out[
            "quantiles_order_logits"
        ]  # [BS, num_quantiles, num_classes]
        topk_logits, topk_indices = torch.topk(quantiles_order_logits, k=k, dim=-1)
        topk_probs = torch.softmax(topk_logits, dim=-1)  # [BS, num_quantiles, 3]
        topk_exponents = self.exponent_mapping.to(topk_indices.device)[
            topk_indices
        ]  # [BS, num_quantiles, 3]
        expected_exponent = (topk_exponents * topk_probs).sum(
            dim=-1
        )  # [BS, num_quantiles]
        pred_scale = torch.pow(10.0, expected_exponent)  # [BS, num_quantiles]
        pred_quantiles = quantiles_reg * pred_scale  # [BS, num_quantiles]
        return pred_quantiles

    def loss_function(self, quantiles_final, quantiles_reg, quantiles_order_logits, y):
        """Calculate the quantile loss"""
        true_order = torch.floor(torch.log10(torch.abs(y))).clamp(
            min=self.min_mag, max=self.max_mag
        )
        true_scale = torch.pow(10.0, true_order)
        normalized_y = y / true_scale
        quantile_reg_loss = self.pinball_loss(quantiles_reg, normalized_y)

        y_quantiles = torch.quantile(
            y, torch.tensor(self.quantiles).to(y.device), dim=1
        ).T  # [BS, num_quantiles]
        true_order_q = torch.floor(torch.log10(torch.abs(y_quantiles))).clamp(
            min=self.min_mag, max=self.max_mag
        )
        class_indicies = (true_order_q - self.min_mag).long().squeeze(-1)
        # quantiles_order_logits.shape = [B, num_quantiles, num_classes]
        bs, num_quantiles, num_classes = quantiles_order_logits.shape
        quantiles_order_logits = quantiles_order_logits.view(
            bs * num_quantiles, num_classes
        )
        class_indicies = class_indicies.view(bs * num_quantiles)
        quantile_class_loss = F.cross_entropy(
            quantiles_order_logits, class_indicies, reduction="none"
        )
        quantile_class_loss = quantile_class_loss.view(bs, num_quantiles)
        quantile_class_loss = quantile_class_loss.mean(dim=0)
        quantile_class_loss = quantile_class_loss * self.quantile_weights
        quantile_class_loss = quantile_class_loss.sum()

        # Smooth gating mechanism
        combined_loss = self.alpha * quantile_class_loss + self.beta * quantile_reg_loss

        return {
            "class_loss": quantile_class_loss,
            "reg_loss": quantile_reg_loss,
            "combined_loss": combined_loss,
        }


class LitQuantileMagnitudePredictor(LitWrapper):
    def __init__(self, config):
        model = QuantileMagnitudePredictor(config)
        super().__init__(config=config, model=model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self.model(x)
        loss = self.model.loss_function(
            out["quantiles_final"],
            out["quantiles_reg"],
            out["quantiles_order_logits"],
            y,
        )
        self.log("train_loss", loss["combined_loss"], prog_bar=True)
        self.log("train_reg_loss", loss["reg_loss"], prog_bar=True)
        self.log("train_class_loss", loss["class_loss"], prog_bar=True)
        return loss["combined_loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self.model(x)
        loss = self.model.loss_function(
            out["quantiles_final"],
            out["quantiles_reg"],
            out["quantiles_order_logits"],
            y,
        )
        self.log("val_loss", loss["combined_loss"], prog_bar=True)
        self.log("val_reg_loss", loss["reg_loss"], prog_bar=True)
        self.log("val_class_loss", loss["class_loss"], prog_bar=True)
        pred_quantiles = self.model.predict_expected(x)
        (
            orig_mean_median_mse_loss,
            orig_median_median_mse_loss,
            orig_median_rel_median_mse_loss,
        ) = self.model.get_median_mse_loss(pred_quantiles, y)
        self.log(
            "val_orig_mean_median_mse_loss", orig_mean_median_mse_loss, prog_bar=False
        )
        self.log(
            "val_orig_median_median_mse_loss",
            orig_median_median_mse_loss,
            prog_bar=False,
        )
        self.log(
            "val_orig_median_rel_median_mse_loss",
            orig_median_rel_median_mse_loss,
            prog_bar=False,
        )

        for ci_level in self.model.ci_dict.keys():
            lower, upper = self.model.ci_dict[ci_level]
            if lower in self.model.quantiles and upper in self.model.quantiles:
                coverage = self.model.get_ci_coverage(
                    pred_quantiles, y, ci_level=ci_level
                )
                self.log(f"val_{ci_level}_coverage", coverage.mean(), prog_bar=False)
                coverage_abs_error = torch.abs(coverage - ci_level).mean()
                self.log(
                    f"val_{ci_level}_coverage_abs_error",
                    coverage_abs_error,
                    prog_bar=False,
                )

        return loss["combined_loss"]


class QuantileConditionalPredictor(QuantileMagnitudePredictor):
    def __init__(self, config):
        if hasattr(config, "type") and config.type == "separate":
            original_hidden_layers = config.hidden_layers
            config.hidden_layers = 0  # No hidden layers in the feature extractor model
        super().__init__(config)
        if hasattr(config, "type") and config.type == "separate":
            config.hidden_layers = original_hidden_layers

        if config.hidden_layers == 0 or config.type == "separate":
            input_dim = self.input_dim
        else:
            input_dim = config.hidden_dim

        if config.type == "conditional":
            self.magnitude_heads = nn.ModuleList(
                [
                    ConditionalMagnitudeHead(
                        input_dim=input_dim,
                        min_mag=self.min_mag,
                        max_mag=self.max_mag,
                        use_arctan=config.use_arctan,
                    )
                    for _ in range(len(config.quantiles))
                ]
            )
        elif config.type == "separate":
            self.magnitude_heads = nn.ModuleList(
                [
                    SeparateMagRegHead(
                        input_dim=input_dim,
                        hidden_layers=config.hidden_layers,
                        hidden_dim=config.hidden_dim,
                        min_mag=self.min_mag,
                        max_mag=self.max_mag,
                        use_arctan=config.use_arctan,
                    )
                    for _ in range(len(config.quantiles))
                ]
            )
        self.exponent_mapping = self.magnitude_heads[0].exponent_mapping
        self.topk = config.topk

    def predict_expected(self, x, k=None):
        if k is None:
            k = self.topk
        out = self.forward(x)
        quantiles_reg = out["quantiles_reg"]  # [BS, num_quantiles, num_classes]
        quantiles_order_logits = out[
            "quantiles_order_logits"
        ]  # [BS, num_quantiles, num_classes]
        # Get the top k predictions
        topk_logits, topk_indices = torch.topk(quantiles_order_logits, k=k, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)  # [bs, num_quantiles, k]
        pred_scaled = quantiles_reg * torch.pow(
            10.0, self.exponent_mapping.to(quantiles_reg.device)
        )  # [bs, num_quantiles, num_classes]
        topk_pred_scaled = pred_scaled.gather(
            -1, topk_indices
        )  # [bs, num_quantiles, k]
        pred_quantiles = (topk_pred_scaled * topk_probs).sum(
            dim=-1
        )  # [bs, num_quantiles]
        return pred_quantiles

    def pinball_loss(self, pred_quantiles, y, true_scale_q=None):
        if true_scale_q is None:
            return super().pinball_loss(pred_quantiles, y)
        else:
            return self.get_qnormalized_quantile_loss(
                pred_quantiles, y, true_scale_q=true_scale_q
            )

    def get_qnormalized_quantile_loss(self, pred_quantiles, y, true_scale_q):
        # Calculate the quantile loss
        quantile_loss = torch.zeros_like(pred_quantiles)
        for i, q in enumerate(self.quantiles):
            q_scale = true_scale_q[:, [i]]
            # entry-wise maximum
            loss = torch.max(
                (q - 1) * (y / q_scale - pred_quantiles[:, [i]]),
                q * (y / q_scale - pred_quantiles[:, [i]]),
            )
            # average across all y samples
            loss = loss.mean(dim=1)
            quantile_loss[:, i] = loss
        # weighted sum across all quantiles
        quantile_loss = quantile_loss * self.quantile_weights
        quantile_loss = quantile_loss.sum(dim=1)
        return quantile_loss.mean()

    def loss_function(self, quantiles_final, quantiles_reg, quantiles_order_logits, y):
        """Calculate the quantile loss"""
        # quantiles_reg [BS, num_quantiles, num_classes]
        quantiles_pred_class = torch.argmax(
            quantiles_order_logits, dim=-1, keepdim=True
        )  # [BS, num_quantiles, 1]
        quantiles_reg_top = torch.gather(
            quantiles_reg, dim=2, index=quantiles_pred_class
        ).squeeze(
            -1
        )  # [BS, num_quantiles]

        y_quantiles = torch.quantile(
            y, torch.tensor(self.quantiles).to(y.device), dim=1
        ).T  # [BS, num_quantiles]
        true_order_q = torch.floor(torch.log10(torch.abs(y_quantiles))).clamp(
            min=self.min_mag, max=self.max_mag
        )
        class_indicies = (true_order_q - self.min_mag).long().squeeze(-1)
        # quantiles_order_logits.shape = [B, num_quantiles, num_classes]
        bs, num_quantiles, num_classes = quantiles_order_logits.shape
        quantiles_order_logits = quantiles_order_logits.view(
            bs * num_quantiles, num_classes
        )
        class_indicies = class_indicies.view(bs * num_quantiles)

        true_scale_q = torch.pow(10.0, true_order_q)
        quantile_reg_loss = self.pinball_loss(
            quantiles_reg_top, y, true_scale_q=true_scale_q
        )

        quantile_class_loss = F.cross_entropy(
            quantiles_order_logits, class_indicies, reduction="none"
        )
        quantile_class_loss = quantile_class_loss.view(bs, num_quantiles)
        quantile_class_loss = quantile_class_loss.mean(dim=0)
        quantile_class_loss = quantile_class_loss * self.quantile_weights
        quantile_class_loss = quantile_class_loss.sum()

        # Smooth gating mechanism
        combined_loss = self.alpha * quantile_class_loss + self.beta * quantile_reg_loss

        return {
            "class_loss": quantile_class_loss,
            "reg_loss": quantile_reg_loss,
            "combined_loss": combined_loss,
        }

    def freeze_regression_head(self):
        # Freeze regression head
        for head in self.magnitude_heads:
            for param in head.reg_head.parameters():
                param.requires_grad = False

    def unfreeze_regression_head(self):
        # Unfreeze regression head
        for head in self.magnitude_heads:
            for param in head.reg_head.parameters():
                param.requires_grad = True

    def freeze_classification_head(self):
        # Freeze classification head
        for head in self.magnitude_heads:
            for param in head.class_head.parameters():
                param.requires_grad = False

    def unfreeze_classification_head(self):
        # Unfreeze classification head
        for head in self.magnitude_heads:
            for param in head.class_head.parameters():
                param.requires_grad = True


class LitQuantileConditionalPredictor(LitQuantileMagnitudePredictor):
    def __init__(self, config):
        super().__init__(config=config)
        self.model = QuantileConditionalPredictor(config)
