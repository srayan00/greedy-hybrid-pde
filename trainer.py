import torch
from ml_solver import MLSolver, DeepONet, FNOforPDE, PredictorRejector
from hybrid_solver import Router, ConstantRouter, HINTSRouter, LSTMGreedyRouter, HybridSolver

class MSEalphaepsilonLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-9):
        super(MSEalphaepsilonLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        mse_loss = torch.mean((predictions - targets) ** 2)
        target_norm = torch.sum(targets ** 2)
        adjusted_loss = mse_loss / (target_norm**self.alpha + self.epsilon)
        return adjusted_loss

class ApproxGreedyRouterLoss(torch.nn.Module):
    def __init__(self, cost = None, max_tokens = 20, is_score = True, flip = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cost = cost 
        self.max_tokens = max_tokens
        self.is_score = is_score
        self.flip = flip

    def forward(self, prediction, target, reduction = "mean"):
        probs = torch.nn.functional.softmax(prediction["routing_scores"], dim = -1)
        log_scores = torch.log(1/probs) 
        expert_predictions = prediction["complete_expert_predictions"] - torch.mean(prediction["complete_expert_predictions"], dim=-1, keepdim=True)
        errors_expert = torch.norm(expert_predictions - target.unsqueeze(0), dim=-1).transpose(2, 1)
        weights = torch.sum(errors_expert, dim = -1, keepdim=True) - errors_expert
        loss_vec = torch.sum(log_scores * weights, dim = -1).transpose(1, 0)
        loss_vec = torch.mean(loss_vec, dim = -1)
        if reduction == "none":
            return loss_vec
        elif reduction == "sum":
            return torch.sum(loss_vec)
        elif reduction == "mean":
            return torch.mean(loss_vec)

class EarlyStopping:
    def __init__(self, patience: int =7, verbose: bool=False, delta: float =0, warmup_epochs: int = 10):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.warmup_epochs = warmup_epochs
    
    def __call__(self, val_loss, epoch, loss=True):
        score = -val_loss if loss else val_loss
        if epoch < self.warmup_epochs:
            return True
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True

    def state_dict(self):
        return {"counter": self.counter, "best_score": self.best_score, "early_stop": self.early_stop, "delta": self.delta, "patience": self.patience, "verbose": self.verbose}
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.early_stop = state_dict["early_stop"]
        self.delta = state_dict["delta"]
        self.patience = state_dict["patience"]
        self.verbose = state_dict["verbose"]

class ScheduledSampler:
    def __init__(self, starting_teacher_forcing_prob: float = 1.0, decay: float = 0.95, 
                 slope: float = 0.05, ending_teacher_forcing_prob: float = 0.1, linear = True,
                 warmup_epochs: int = 10):
        self.starting_teacher_forcing_prob = starting_teacher_forcing_prob
        self.decay = decay
        self.slope = slope
        self.current_prob = starting_teacher_forcing_prob
        self.ending_teacher_forcing_prob = ending_teacher_forcing_prob
        self.linear = linear
        self.warmup_epochs = warmup_epochs
    
    # def __call__(self):
    #     self.current_prob = self.current_prob * self.decay
    #     return self.current_prob
    
    def __call__(self, epoch):
        if epoch >= self.warmup_epochs:
            if self.linear:
                self.current_prob = max(self.ending_teacher_forcing_prob, self.current_prob - self.slope)
            else:
                self.current_prob = max(self.current_prob * self.decay, self.ending_teacher_forcing_prob)
            print(f"Current Teacher Forcing Probability: {self.current_prob}")

    def state_dict(self):
        return {"starting_teacher_forcing_prob": self.starting_teacher_forcing_prob, "decay": self.decay, 
                "current_prob": self.current_prob, "ending_teacher_forcing_prob": self.ending_teacher_forcing_prob,
                "slope": self.slope, "linear": self.linear, "warmup_epochs": self.warmup_epochs}
    
    def load_state_dict(self, state_dict):
        self.starting_teacher_forcing_prob = state_dict["starting_teacher_forcing_prob"]
        self.decay = state_dict["decay"]
        self.current_prob = state_dict["current_prob"]

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_fn: torch.nn.Module,
        save_every: int, 
        save_path: str,
        parallel = False,
        use_amp: bool = False, 
        task: str = "xsum",
        scaler = None,
        max_norm: float = 1.0,
        early_stopper: EarlyStopping = None,
        lr_scheduler: list = None,
        scheduled_sampler: ScheduledSampler = None,
        warmup_epochs: int = 10
    ) -> None:
        self.gpu_id = device
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.parallel = parallel
        self.use_amp = use_amp
        self.task = task
        self.scaler = scaler
        self.early_stopper = early_stopper
        if lr_scheduler is not None:
            self.scheduler_wu = lr_scheduler[0]
            self.scheduler_re = lr_scheduler[1]
            self.scheduler_step = lr_scheduler[2]
        else:
            self.scheduler_wu = None
            self.scheduler_re = None
            self.scheduler_step = None
        self.warmup_epochs = warmup_epochs
        self.scheduled_sampler = scheduled_sampler
        self.max_norm = max_norm
        self.train_losses = None
        self.val_losses = None
    
    def _train_mode(self):
        self.model.train()
    
    def _run_batch(self, source, targets, epoch):
        # Zero gradient todo: make it none
        self.optimizer.zero_grad()
        # Automatic mixed precision 
        if epoch == 0:
            scaling = False
        else:
            scaling = True
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
        # Make Prediction and observe loss
            if isinstance(self.model, HybridSolver):
                channels = source.shape[1]
                f = source[:, 0, :]
                if channels > 1:
                    a = source[:, 1, :]
                else:
                    a = None
                if self.scheduled_sampler is not None:
                    print(f"Scheduled sampler prob {self.scheduled_sampler.current_prob}")
                    output = self.model(f, a, None, True, True, self.scheduled_sampler.current_prob, targets)
                else:
                    output = self.model(f, a, None, True, True, 1.0, targets)
                loss = self.loss_fn(output, targets)
            if isinstance(self.model, MLSolver):
                output = self.model(source)
                loss = self.loss_fn(output, targets)

        # Automatic mixed precision backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            print("Backward pass")
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
        else:
            print("Backward Pass")
            loss.backward()
        
        # Clip gradients
        if self.parallel:
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm = self.max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.max_norm)
        print(f"Gradient norms of all parameters{self._compute_gradient_norm()}")

        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            print("Optimizer Step")
            self.optimizer.step()
        if isinstance(self.model, DeepONet):
            print(self.model.branch_net.mlp.linear_layer_0.bias)
        elif isinstance(self.model, FNOforPDE):
            print(self.model.fno.fno_blocks.convs[0].weight)
        if torch.isnan(loss):
            print("Loss is NaN")
            raise ValueError("Loss is NaN")
        return loss.item()
    
    def _run_epoch(self, epoch):
        train_loss = 0
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        for batch_idx, info in enumerate(self.train_data):
            torch.autograd.set_detect_anomaly(True)
            if isinstance(self.model, MLSolver):
                source = info[0].to(self.gpu_id)
                targets = info[1].to(self.gpu_id)
            elif isinstance(self.model, HybridSolver):
                source = info[0].to(self.gpu_id)
                targets = info[1].to(self.gpu_id)
            loss = self._run_batch(source, targets, epoch)
            print("Batch: {}, Loss: {}".format(batch_idx, loss), flush=True)
            train_loss += loss
        return train_loss / len(self.train_data)
    
    def _save_checkpoint(self, epoch, is_best = False):
        if self.parallel:
            if isinstance(self.model, PredictorRejector):
                raise NotImplementedError
                ckp = self.model.module.rejector.state_dict()
            else:
                ckp = self.model.module.state_dict()
        else:
            if isinstance(self.model, PredictorRejector):
                raise NotImplementedError
                ckp = self.model.rejector.state_dict()
            else:
                ckp = self.model.state_dict()
        if self.scheduler_wu is not None:
            wu_ckp = self.scheduler_wu.state_dict()
        else:
            wu_ckp = None
        if self.scheduler_re is not None:
            re_ckp = self.scheduler_re.state_dict()
        else:
            re_ckp = None
        if self.scheduler_step is not None:
            scheduler_step_ckp = self.scheduler_step.state_dict()
        else:
            scheduler_step_ckp = None
        if self.early_stopper is not None:
            early_ckp = self.early_stopper.state_dict()
        else:
            early_ckp = None
        if self.scaler is not None:
            scaler_ckp = self.scaler.state_dict()
        else:
            scaler_ckp = None
        if self.scheduled_sampler is not None:
            scheduled_sampler_ckp = self.scheduled_sampler.state_dict()
        else:
            scheduled_sampler_ckp = None
        min_vals = None
        max_vals = None
    
        if is_best:
            full_path = self.save_path + "_best.pth"
        else:
            full_path = self.save_path + "_full.pth"
        torch.save({"model": ckp,
                    "optimizer": self.optimizer.state_dict(), 
                    "scheduler_wu": wu_ckp, 
                    "scheduler_re": re_ckp, 
                    "scheduler_step": scheduler_step_ckp,
                    "scaler": scaler_ckp,
                    "early_stopping": early_ckp,
                    "scheduled_sampler": scheduled_sampler_ckp,
                    "min_vals": min_vals,
                    "max_vals": max_vals,
                    "epoch": epoch,
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses}, full_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {full_path}", flush=True)
    
    def _run_validation(self):
        val_loss = 0
        self.model.train()
        if isinstance(self.model, PredictorRejector):
            raise NotImplementedError
            self.model.rejector.scaling = True
            self.model.rejector.train_min_max = False
        with torch.no_grad():
            for batch_idx, info in enumerate(self.val_data):
                if isinstance(self.model, HybridSolver):
                    source = info[0].to(self.gpu_id)
                    targets = info[1].to(self.gpu_id)
                    channels = source.shape[1]
                    f = source[:, 0, :]
                    if channels > 1:
                        a = source[:, 1, :]
                    else:
                        a = None
                    output = self.model(f, a, None, True, True, 0.0, targets)
                elif isinstance(self.model, MLSolver):
                    print(f"source is {info[0]}")
                    print(f"target is {info[1]}")
                    source = info[0].to(self.gpu_id)
                    targets = info[1].to(self.gpu_id)
                    output = self.model(source)
                    
                 # CHANGE THIS LATER
                loss = self.loss_fn(output, targets)
                val_loss += loss.item()
        print(f"number of validation batches is {len(self.val_data)}")
        return val_loss / len(self.val_data)
    
    def train(self, max_epochs: int, start_epoch = 0, train_losses = None, val_losses = None):
        self.train_losses = torch.zeros(max_epochs)
        self.val_losses = torch.zeros(max_epochs)
        if train_losses is not None:
            self.train_losses[:len(train_losses)] = train_losses
        if val_losses is not None:
            self.val_losses[:len(val_losses)] = val_losses
        for epoch in range(start_epoch, max_epochs):
            self._train_mode()
            train_loss = self._run_epoch(epoch)
            self.train_losses[epoch] = train_loss
            print("Epoch: {}, Train Loss: {}".format(epoch, train_loss), flush=True)
            print("Running Validation")
            if self.val_data is not None:
                val_loss = self._run_validation()
                self.val_losses[epoch] = val_loss
                print("Epoch: {}, Validation Loss: {}".format(epoch, val_loss), flush=True)
            if self.scheduler_wu is not None and epoch < self.warmup_epochs:
                self.scheduler_wu.step()
                print("Learning rate is now {}".format(self.scheduler_wu.get_last_lr()), flush=True)
            if self.scheduler_re is not None:
                self.scheduler_re.step(val_loss)
            if self.scheduler_step is not None:
                self.scheduler_step.step()
            if self.early_stopper is not None:
                is_best = self.early_stopper(val_loss, epoch)
                if is_best:
                    self._save_checkpoint(epoch, is_best)
            if self.scheduled_sampler is not None:
                self.scheduled_sampler(epoch)
            if self.early_stopper is not None:
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)
                if self.early_stopper.early_stop:
                    print("Early Stopping", flush=True)
                    break
            else:
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)

    def _compute_gradient_norm(self):
        norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                norms[name] = param.grad.norm(2).item()
        return norms
