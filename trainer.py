import torch
from ml_solver import MLSolver, DeepONet, FNOforPDE, PredictorRejector



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
        else:
            self.scheduler_wu = None
            self.scheduler_re = None
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
            if isinstance(self.model, PredictorRejector):
                raise NotImplementedError
                if self.scheduled_sampler is not None:
                    output = self.model(source, True, 20, self.scheduled_sampler.current_prob, targets, scaling=scaling)
                else:
                    output = self.model(source, True, 20, 1.0, targets, scaling=scaling)
                loss = self.loss_fn(output, targets, w1 = 0.0)
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
            print(self.model.fno.conv1.weight)
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
                source = info["input"].to(self.gpu_id)
                targets = info["solution"].to(self.gpu_id)
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
        self.model.eval()
        if isinstance(self.model, PredictorRejector):
            raise NotImplementedError
            self.model.rejector.scaling = True
            self.model.rejector.train_min_max = False
        with torch.no_grad():
            for batch_idx, info in enumerate(self.val_data):
                if isinstance(self.model, PredictorRejector):
                    raise NotImplementedError
                    source = info["input_ids"].to(self.gpu_id)
                    targets = info["output_ids"].to(self.gpu_id)
                    output = self.model(source, True, 20)
                elif isinstance(self.model, MLSolver):
                    source = info["input"].to(self.gpu_id)
                    targets = info["solution"].to(self.gpu_id)
                    output = self.model(source)
                    
                 # CHANGE THIS LATER
                if isinstance(self.model, PredictorRejector):
                    raise NotImplementedError
                    loss = self.loss_fn(output, targets, w1 = 0.0)
                elif isinstance(self.model, MLSolver):
                    loss = self.loss_fn(output, targets)
                val_loss += loss.item()
        print(f"number of validation batches is {len(self.val_data)}")
        return val_loss / len(self.val_data)
