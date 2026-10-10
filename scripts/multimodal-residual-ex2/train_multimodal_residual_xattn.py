#!/usr/bin/env python3
"""Train MACE gated cross-attention fusion with frozen pretrained encoders."""
from __future__ import annotations
import argparse, csv, json, math, os, random, sys, time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from dataset_multimodal_fusion import MACEMultimodalFusionDataset, LABEL_MAP, ID_TO_LABEL
from model_multimodal_residual_xattn import MACEResidualGatedCrossAttention

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def metrics(y,p):
    return {
        "accuracy": accuracy_score(y,p),
        "balanced_accuracy": balanced_accuracy_score(y,p),
        "macro_f1": f1_score(y,p,average="macro",zero_division=0),
        "weighted_f1": f1_score(y,p,average="weighted",zero_division=0),
    }

def save_cm(y,p,path,title):
    cm=confusion_matrix(y,p,labels=list(range(4)))
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(cm)
    fig.colorbar(im,ax=ax)
    names=[ID_TO_LABEL[i] for i in range(4)]
    ax.set_xticks(range(4),names,rotation=30,ha="right")
    ax.set_yticks(range(4),names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    pd.DataFrame(cm,index=names,columns=names).to_csv(path.with_suffix(".csv"))

def run_epoch(model,loader,criterion,device,optimizer=None,scaler=None,amp=True):
    training=optimizer is not None
    model.train(training)
    total_loss=0.0; ys=[]; ps=[]; rows=[]
    for batch in loader:
        video=batch["video"].to(device,non_blocking=True)
        tel=batch["telemetry"].to(device,non_blocking=True)
        y=batch["label"].to(device,non_blocking=True)

        if training: optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp and device.type=="cuda"):
            logits=model(video,tel)
            loss=criterion(logits,y)

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],1.0
            )
            scaler.step(optimizer); scaler.update()

        pred=logits.argmax(1)
        total_loss += loss.item()*y.size(0)
        ys.extend(y.detach().cpu().tolist()); ps.extend(pred.detach().cpu().tolist())

        if not training:
            probs=torch.softmax(logits,1).detach().cpu().numpy()
            for i in range(y.size(0)):
                row={
                    "sample_id":batch["sample_id"][i],
                    "flight_id":batch["flight_id"][i],
                    "segment_id":batch["segment_id"][i],
                    "true_id":int(y[i].item()),
                    "true_state":ID_TO_LABEL[int(y[i].item())],
                    "pred_id":int(pred[i].item()),
                    "pred_state":ID_TO_LABEL[int(pred[i].item())],
                    "video_start_sec":float(batch["video_start_sec"][i]),
                    "sync_method":batch["sync_method"][i],
                }
                for c in range(4): row[f"p_{ID_TO_LABEL[c]}"]=float(probs[i,c])
                rows.append(row)

    m=metrics(ys,ps); m["loss"]=total_loss/len(ys)
    return m,ys,ps,pd.DataFrame(rows)


@torch.no_grad()
def fusion_diagnostics(model, loader, device, amp=True):
    model.eval()
    gs=gss=rn=vn=dn=0.0; gn=n=0
    for batch in loader:
        video=batch["video"].to(device,non_blocking=True)
        tel=batch["telemetry"].to(device,non_blocking=True)
        with torch.autocast(device_type="cuda",dtype=torch.float16,
                            enabled=amp and device.type=="cuda"):
            o=model(video,tel,return_attention=True)
        g=o["gate"].float(); r=o["residual_features"].float()
        v=o["vision_features"].float(); f=o["fused_features"].float()
        gs+=g.sum().item(); gss+=(g*g).sum().item(); gn+=g.numel()
        rn+=r.norm(dim=1).sum().item(); vn+=v.norm(dim=1).sum().item()
        dn+=(f-v).norm(dim=1).sum().item(); n+=v.shape[0]
    gm=gs/max(gn,1); gv=max(gss/max(gn,1)-gm*gm,0.0)
    return {"alpha":float(model.residual_scale.detach().cpu()),
            "gate_mean":gm,"gate_std":gv**0.5,
            "residual_norm":rn/max(n,1),"vision_norm":vn/max(n,1),
            "applied_delta_norm":dn/max(n,1),
            "delta_to_vision_ratio":dn/max(vn,1e-12)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--vision-checkpoint",required=True)
    p.add_argument("--aircraft-checkpoint",required=True)
    p.add_argument("--normalization-json",default=None)
    p.add_argument("--annotation-file",default="data/annotations/MACE_Temporal_label_complete.xlsx")
    p.add_argument("--split-file",default="data/annotations/flight_splits.csv")
    p.add_argument("--telemetry-csv",default="results/flight_baseline/labelled_flight_telemetry.csv")
    p.add_argument("--flights-root",default="Flights")
    p.add_argument("--train-samples",type=int,default=6000)
    p.add_argument("--val-samples",type=int,default=1200)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--num-workers",type=int,default=8)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--patience",type=int,default=20)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--weight-decay",type=float,default=0.01)
    p.add_argument("--label-smoothing",type=float,default=0.05)
    p.add_argument("--attention-dim",type=int,default=256)
    p.add_argument("--attention-heads",type=int,default=4)
    p.add_argument("--dropout",type=float,default=0.30)
    p.add_argument("--residual-scale-init",type=float,default=0.05) #change back to original value 0.1
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--resume",default=None) 
    p.add_argument("--output-root",default="experiments")
    args=p.parse_args()

    seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this MViTv2 fusion experiment.")
    device=torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True

    job=os.environ.get("SLURM_JOB_NAME","MACE_Multimodal_Residual_XAttn_ex2")
    jid=os.environ.get("SLURM_JOB_ID","local")
    stamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out=Path(args.output_root)/f"{job}_{jid}_{stamp}"
    for d in ["checkpoints","tensorboard","confusion_matrices","predictions","reports"]:
        (out/d).mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(vars(args),indent=2))

    common=dict(
        annotation_file=args.annotation_file, flights_root=args.flights_root,
        split_file=args.split_file, telemetry_csv=args.telemetry_csv,
        aircraft_checkpoint=args.aircraft_checkpoint,
        normalization_json=args.normalization_json,
        window_seconds=5.0, frames_per_sample=16, telemetry_hz=10.0,
        image_size=224, base_seed=args.seed,
    )
    train_ds=MACEMultimodalFusionDataset(
        split="train",samples_per_epoch=args.train_samples,augment=True,**common
    )
    val_ds=MACEMultimodalFusionDataset(
        split="val",samples_per_epoch=args.val_samples,augment=False,**common
    )

    loader_kwargs=dict(
        batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True,
        persistent_workers=False,
    )
    if args.num_workers>0: loader_kwargs["prefetch_factor"]=2
    train_loader=DataLoader(train_ds,shuffle=False,**loader_kwargs)
    val_loader=DataLoader(val_ds,shuffle=False,**loader_kwargs)

    model=MACEResidualGatedCrossAttention(
        args.vision_checkpoint,args.aircraft_checkpoint,
        attention_dim=args.attention_dim,attention_heads=args.attention_heads,
        dropout=args.dropout,freeze_encoders=True,
        residual_scale_init=args.residual_scale_init,
    ).to(device)

    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Parameters: total={total:,} trainable={trainable:,}")
    print(f"Output: {out}")

    criterion=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer=AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,weight_decay=args.weight_decay
    )
    scheduler=CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=True)

    start_epoch=1; best=-math.inf; stale=0
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        if "scaler_state_dict" in ck: scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch=int(ck["epoch"])+1
        best=float(ck.get("best_val_macro_f1",-math.inf))
        stale=int(ck.get("epochs_without_improvement",0))

    writer=SummaryWriter(out/"tensorboard")
    history=[]
    for epoch in range(start_epoch,args.epochs+1):
        t0=time.time()
        # Match vision experiments: regenerate only training schedule each epoch.
        if epoch>start_epoch: train_ds.start_new_epoch(epoch)

        tr,_,_,_=run_epoch(model,train_loader,criterion,device,optimizer,scaler,True)
        va,y,pred,preds=run_epoch(model,val_loader,criterion,device,None,None,True)
        fusion_diag=fusion_diagnostics(model,val_loader,device,True)
        scheduler.step()

        row={"epoch":epoch,"lr":optimizer.param_groups[0]["lr"],
             **{f"train_{k}":v for k,v in tr.items()},
             **{f"val_{k}":v for k,v in va.items()},
             **{f"fusion_{k}":v for k,v in fusion_diag.items()},
             "epoch_seconds":time.time()-t0}
        history.append(row); pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)

        for k,v in tr.items(): writer.add_scalar(f"train/{k}",v,epoch)
        for k,v in va.items(): writer.add_scalar(f"val/{k}",v,epoch)
        for k,v in fusion_diag.items(): writer.add_scalar(f"fusion/{k}",v,epoch)
        writer.add_scalar("learning_rate",optimizer.param_groups[0]["lr"],epoch)

        preds.to_csv(out/"predictions"/f"epoch_{epoch:03d}_validation_predictions.csv",index=False)
        save_cm(y,pred,out/"confusion_matrices"/f"epoch_{epoch:03d}.png",f"Validation epoch {epoch}")

        improved=va["macro_f1"] > best + 1e-4
        if improved: best=va["macro_f1"]; stale=0
        else: stale+=1

        ck={
            "epoch":epoch,"model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "scheduler_state_dict":scheduler.state_dict(),
            "scaler_state_dict":scaler.state_dict(),
            "best_val_macro_f1":best,"epochs_without_improvement":stale,
            "args":vars(args),"label_map":LABEL_MAP,
        }
        torch.save(ck,out/"checkpoints"/"last.pt")
        if epoch%10==0: torch.save(ck,out/"checkpoints"/f"epoch_{epoch:03d}.pt")
        if improved:
            torch.save(ck,out/"checkpoints"/"best.pt")
            preds.to_csv(out/"predictions"/"best_validation_predictions.csv",index=False)
            save_cm(y,pred,out/"confusion_matrices"/"best.png","Best validation")
            (out/"reports"/"classification_report_best.txt").write_text(
                classification_report(y,pred,target_names=[ID_TO_LABEL[i] for i in range(4)],digits=4,zero_division=0)
            )
            (out/"reports"/"best_validation_metrics.json").write_text(json.dumps(va,indent=2))

        print(
            f"Epoch {epoch:03d} | train loss {tr['loss']:.4f} F1 {tr['macro_f1']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} "
            f"macro-F1 {va['macro_f1']:.4f} | best {best:.4f} | {row['epoch_seconds']:.1f}s"
        )
        writer.flush()
        if stale>=args.patience:
            print(f"Early stopping after {stale} epochs without improvement.")
            break

    writer.close()
    print(f"Finished. Best validation Macro-F1: {best:.4f}")
    print(f"Results: {out}")

if __name__=="__main__":
    main()

