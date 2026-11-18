import os, re, glob, json, warnings
warnings.filterwarnings("ignore")
import cv2, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from features import extract_features, FEATURE_NAMES

IMG_EXT=(".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff")

def infer_subject(path):
    name=os.path.basename(path)
    stem=os.path.splitext(name)[0]
    patterns=[r"^(?:patient|subject|participant|control|pd|hc)[_-]?([A-Za-z0-9]+)",
              r"^([A-Za-z]+[_-]?\d+)"]
    for p in patterns:
        m=re.search(p,stem,re.I)
        if m: return m.group(1)
    # Common spiral datasets encode participant and trial as e.g. 0068-3.
    m=re.match(r"^(\d+)[_-]\d+$",stem)
    if m: return m.group(1)
    # Conservative fallback: filename itself becomes the subject.
    return stem

def collect(root):
    rows=[]
    for folder,label in [("control",0),("parkinson",1)]:
        for p in glob.glob(os.path.join(root,folder,"**","*"),recursive=True):
            if p.lower().endswith(IMG_EXT):
                rows.append((p,label,infer_subject(p)))
    return rows

def make_table(root):
    out=[]; bad=[]
    for path,label,subject in collect(root):
        try:
            img=cv2.imread(path)
            f,_=extract_features(img)
            out.append([path,label,subject,*f])
        except Exception as e: bad.append((path,str(e)))
    return pd.DataFrame(out,columns=["path","label","subject",*FEATURE_NAMES]),bad

def metrics(y,pred,prob):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sensitivity=tp/(tp+fn) if tp+fn else 0
    specificity=tn/(tn+fp) if tn+fp else 0
    return {
        "accuracy":accuracy_score(y,pred),
        "balanced_accuracy":balanced_accuracy_score(y,pred),
        "sensitivity":sensitivity,
        "specificity":specificity,
        "precision":precision_score(y,pred,zero_division=0),
        "recall":recall_score(y,pred,zero_division=0),
        "f1":f1_score(y,pred,zero_division=0),
        "roc_auc":roc_auc_score(y,prob) if len(np.unique(y))==2 else np.nan,
        "pr_auc":average_precision_score(y,prob) if len(np.unique(y))==2 else np.nan,
        "brier":brier_score_loss(y,prob)
    }

def build_models():
    return {
        "SVM":Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("model",SVC(C=1.0,kernel="rbf",probability=True,class_weight="balanced",random_state=42))
        ]),
        "Random Forest":Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",RandomForestClassifier(n_estimators=500,max_features="sqrt",
                                            min_samples_leaf=2,class_weight="balanced",
                                            random_state=42,n_jobs=-1))
        ]),
        "XGBoost":Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",XGBClassifier(n_estimators=400,max_depth=3,learning_rate=.03,
                                   subsample=.85,colsample_bytree=.85,
                                   reg_lambda=2,eval_metric="logloss",
                                   random_state=42,n_jobs=4))
        ])
    }



def load_cnn_image(path, size=(128, 128)):
    """Load an image in grayscale and normalize pixels to [0, 1]."""
    image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    return (image.astype("float32")/255.0)[...,np.newaxis]


def select_balanced_accuracy_threshold(y_true, scores):
    """Choose a score cutoff using validation data, not the held-out test set."""
    y_true=np.asarray(y_true)
    scores=np.asarray(scores,dtype=float)
    candidates=np.unique(np.r_[0.0,scores,np.nextafter(scores.max(),np.inf)])
    balanced_accuracies=np.array([
        balanced_accuracy_score(y_true,scores>=threshold)
        for threshold in candidates
    ])
    best=np.flatnonzero(np.isclose(balanced_accuracies,balanced_accuracies.max()))
    selected=best[np.argmin(np.abs(candidates[best]-0.5))]
    return float(candidates[selected])


def feature_correlations(df):
    """Return feature-to-feature correlations, averaging repeated participant drawings."""
    data=df
    if "subject" in df.columns:
        data=df.groupby(["label","subject"],as_index=False)[FEATURE_NAMES].mean()
    return data[FEATURE_NAMES].corr()


def save_feature_correlation_heatmap(df,path):
    correlations=feature_correlations(df)
    values=correlations.to_numpy(dtype=float)
    mask=np.triu(np.ones(values.shape,dtype=bool),k=1)|~np.isfinite(values)
    cmap=plt.get_cmap("coolwarm").with_extremes(bad="white")

    fig,ax=plt.subplots(figsize=(13,11))
    heatmap=ax.imshow(
        np.ma.array(values,mask=mask),
        aspect="equal",cmap=cmap,vmin=-1,vmax=1
    )
    labels=[name.replace("_"," ") for name in correlations.index]
    ticks=np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels,rotation=90,ha="center",fontsize=8)
    ax.set_yticklabels(labels,fontsize=8)
    ax.set_title("Feature correlation comparison (lower triangle)")
    for row in range(len(labels)):
        for column in range(row+1):
            value=values[row,column]
            if np.isfinite(value):
                color="white" if abs(value)>=0.6 else "black"
                ax.text(
                    column,row,f"{value:.2f}",
                    ha="center",va="center",color=color,fontsize=7
                )
    colorbar=fig.colorbar(heatmap,ax=ax,pad=0.02,fraction=0.046)
    colorbar.set_label("Pearson correlation coefficient (r)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path),exist_ok=True)
    fig.savefig(path,dpi=300)
    plt.close(fig)
    return correlations


def build_cnn(input_shape=(128,128,1)):
    """Build a compact CNN for the two spiral-image classes."""
    import tensorflow as tf
    inputs=tf.keras.Input(shape=input_shape)
    x=tf.keras.layers.Conv2D(32,3,padding="same",activation="relu")(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(64,3,padding="same",activation="relu")(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(128,3,padding="same",activation="relu")(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(0.4)(x)
    x=tf.keras.layers.Dense(64,activation="relu")(x)
    x=tf.keras.layers.Dropout(0.3)(x)
    outputs=tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model=tf.keras.Model(inputs,outputs,name="spiral_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")]
    )
    return model


def train_cnn(train_df,test_df,outdir,seed=42):
    """Train CNN with participant-separated validation and held-out testing."""
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    tf.keras.utils.set_random_seed(seed)

    validation=None
    for split_seed in range(seed,seed+100):
        splitter=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=split_seed)
        fit_idx,val_idx=next(splitter.split(
            train_df,train_df["label"],groups=train_df["subject"]))
        fit_df=train_df.iloc[fit_idx]
        val_df=train_df.iloc[val_idx]
        if fit_df["label"].nunique()==2 and val_df["label"].nunique()==2:
            validation=(fit_df,val_df)
            break
    if validation is None:
        raise ValueError(
            "CNN validation split needs both classes in participant-separated "
            "groups. Check subject IDs and dataset size."
        )
    fit_df,val_df=validation
    X_fit=np.stack([load_cnn_image(p) for p in fit_df["path"]])
    X_val=np.stack([load_cnn_image(p) for p in val_df["path"]])
    X_test=np.stack([load_cnn_image(p) for p in test_df["path"]])
    y_fit=fit_df["label"].to_numpy(dtype=np.int32)
    y_val=val_df["label"].to_numpy(dtype=np.int32)

    classes=np.unique(y_fit)
    weights=compute_class_weight(class_weight="balanced",classes=classes,y=y_fit)
    class_weights={int(c):float(w) for c,w in zip(classes,weights)}

    model=build_cnn()
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",patience=10,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",factor=0.5,patience=4,min_lr=1e-6)
    ]
    model.fit(
        X_fit,y_fit,validation_data=(X_val,y_val),
        epochs=70,batch_size=16,class_weight=class_weights,
        callbacks=callbacks,verbose=0
    )
    validation_probabilities=model.predict(X_val,verbose=0).reshape(-1)
    threshold=select_balanced_accuracy_threshold(y_val,validation_probabilities)
    probabilities=model.predict(X_test,verbose=0).reshape(-1)
    predictions=(probabilities>=threshold).astype(int)
    model.save(os.path.join(outdir,"cnn.keras"))
    return model,predictions,probabilities,threshold

def save_research_figures(df, yte, pred, prob, model, model_name, outdir, results,
                          threshold=0.5):
    figdir=os.path.join(outdir,"figures"); os.makedirs(figdir,exist_ok=True)

    # 1. Class distribution
    counts=df["label"].map({0:"Control",1:"Parkinson"}).value_counts().reindex(["Control","Parkinson"])
    plt.figure(figsize=(6,4)); counts.plot(kind="bar")
    plt.ylabel("Number of images"); plt.xlabel("Class"); plt.title("Dataset class distribution")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(figdir,"class_distribution.png"),dpi=300); plt.close()

    # 2. Model comparison
    m=results.set_index("model")[["balanced_accuracy","sensitivity","specificity","f1","roc_auc"]]
    ax=m.plot(kind="bar",figsize=(10,5))
    ax.set_ylabel("Score"); ax.set_ylim(0,1.05); ax.set_title("Held-out subject-level model comparison")
    ax.legend(loc="lower right"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(figdir,"model_comparison.png"),dpi=300); plt.close()

    # 3. Confusion matrix
    cm=confusion_matrix(yte,pred,labels=[0,1])
    plt.figure(figsize=(5,4)); plt.imshow(cm,interpolation="nearest")
    plt.title(f"Confusion matrix — {model_name}")
    plt.colorbar(); plt.xticks([0,1],["Control","Parkinson"]); plt.yticks([0,1],["Control","Parkinson"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(os.path.join(figdir,"confusion_matrix.png"),dpi=300); plt.close()

    # 4. ROC
    fpr,tpr,_=roc_curve(yte,prob)
    auc=roc_auc_score(yte,prob)
    plt.figure(figsize=(6,5)); plt.plot(fpr,tpr,label=f"AUC = {auc:.3f}"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"ROC curve — {model_name}"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(figdir,"roc_curve.png"),dpi=300); plt.close()

    # 5. Precision-recall
    precision,recall,_=precision_recall_curve(yte,prob)
    ap=average_precision_score(yte,prob)
    plt.figure(figsize=(6,5)); plt.plot(recall,precision,label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision–Recall curve — {model_name}")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(os.path.join(figdir,"precision_recall_curve.png"),dpi=300); plt.close()

    # 6. Score distribution
    plt.figure(figsize=(7,4))
    plt.hist(prob[yte==0],bins=12,alpha=.7,label="Control")
    plt.hist(prob[yte==1],bins=12,alpha=.7,label="Parkinson")
    plt.axvline(threshold,linestyle="--",label=f"Decision threshold = {threshold:.3f}")
    plt.xlabel("Parkinson-class model score"); plt.ylabel("Number of test images")
    plt.title("Distribution of held-out model scores"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figdir,"score_distribution.png"),dpi=300); plt.close()

    # 7. Pairwise feature correlation heatmap.
    save_feature_correlation_heatmap(
        df,os.path.join(figdir,"feature_correlation_heatmap.png")
    )

    # 7. Feature importance / permutation importance on held-out data
    try:
        pi=permutation_importance(model, np.asarray(df.loc[df["path"].isin([]),FEATURE_NAMES]), np.array([]))
    except Exception:
        pi=None

    # Compute permutation importance on actual held-out feature matrix through caller-safe reload.
    # The model's pipeline accepts raw features.
    # Caller saves a dedicated feature importance figure below if possible.

def train(root,outdir,seed=42):
    os.makedirs(outdir,exist_ok=True)
    df,bad=make_table(root)
    if len(df)<20 or df.label.nunique()<2:
        raise ValueError("Need at least 20 valid images and both classes.")
    if df.subject.nunique()<4:
        raise ValueError("At least 4 unique subjects are recommended for subject-level validation.")

    df.to_csv(os.path.join(outdir,"features.csv"),index=False)

    # Hold out subjects, never individual images.
    gss=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed)
    train_idx,test_idx=next(gss.split(df[FEATURE_NAMES],df.label,groups=df.subject))
    tr=df.iloc[train_idx].copy(); te=df.iloc[test_idx].copy()

    # If the held-out group accidentally contains only one class, retry deterministically.
    if te.label.nunique()<2:
        raise ValueError("Held-out subject set contains one class. Increase dataset size or adjust subject IDs.")

    Xtr=tr[FEATURE_NAMES].values; ytr=tr.label.values
    Xte=te[FEATURE_NAMES].values; yte=te.label.values

    results=[]; artifacts={}; thresholds={}
    for name,base in build_models().items():
        base.fit(Xtr,ytr)
        p=base.predict(Xte); pr=base.predict_proba(Xte)[:,1]
        row=metrics(yte,p,pr); row["model"]=name
        row["decision_threshold"]=0.5
        results.append(row)
        joblib.dump(base,os.path.join(outdir,name.lower().replace(" ","_")+".joblib"))
        artifacts[name]=(base,p,pr)
        thresholds[name]=0.5

    # Train a CNN directly on spiral images, alongside feature-based ML models.
    cnn_model,cnn_pred,cnn_prob,cnn_threshold=train_cnn(tr,te,outdir,seed)
    cnn_row=metrics(yte,cnn_pred,cnn_prob)
    cnn_row["model"]="CNN"
    cnn_row["decision_threshold"]=cnn_threshold
    results.append(cnn_row)
    artifacts["CNN"]=(cnn_model,cnn_pred,cnn_prob)
    thresholds["CNN"]=cnn_threshold

    res=pd.DataFrame(results).sort_values("roc_auc",ascending=False)
    res.to_csv(os.path.join(outdir,"test_metrics.csv"),index=False)
    te[["path","subject","label"]].assign(
        pred=artifacts[res.iloc[0]["model"]][1],
        score=artifacts[res.iloc[0]["model"]][2]
    ).assign(decision_threshold=thresholds[res.iloc[0]["model"]]).to_csv(
        os.path.join(outdir,"heldout_predictions.csv"),index=False)

    # Bootstrap 95% CI for the primary ROC-AUC, when feasible.
    best=res.iloc[0]["model"]; prob=artifacts[best][2]
    rng=np.random.default_rng(seed)
    aucs=[]
    for _ in range(2000):
        idx=rng.integers(0,len(yte),len(yte))
        if len(np.unique(yte[idx]))==2:
            aucs.append(roc_auc_score(yte[idx],prob[idx]))
    ci=(float(np.percentile(aucs,2.5)),float(np.percentile(aucs,97.5))) if aucs else (np.nan,np.nan)

    # Generate figures for the selected primary model.
    best_model_obj, best_pred, best_prob = artifacts[best]
    save_research_figures(df, yte, best_pred, best_prob, best_model_obj, best,
                          outdir, res, thresholds[best])

    # Permutation importance for the primary model on held-out data.
    try:
        if best == "CNN":
            raise ValueError("Permutation importance for handcrafted features is not applicable to CNN.")
        pi=permutation_importance(best_model_obj, Xte, yte, n_repeats=30,
                                  random_state=seed, scoring="roc_auc")
        imp=pd.DataFrame({"feature":FEATURE_NAMES,
                          "importance_mean":pi.importances_mean,
                          "importance_std":pi.importances_std}).sort_values("importance_mean",ascending=False)
        imp.to_csv(os.path.join(outdir,"feature_importance.csv"),index=False)
        top=imp.head(15).sort_values("importance_mean")
        plt.figure(figsize=(8,6))
        plt.barh(top["feature"],top["importance_mean"],xerr=top["importance_std"])
        plt.xlabel("Permutation importance (ROC-AUC decrease)")
        plt.title(f"Top feature importance — {best}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,"figures","feature_importance.png"),dpi=300)
        plt.close()
    except Exception:
        pass

    meta={
        "best_model":best,
        "test_subjects":int(te.subject.nunique()),
        "train_subjects":int(tr.subject.nunique()),
        "test_images":int(len(te)),
        "train_images":int(len(tr)),
        "cnn_threshold":cnn_threshold,
        "roc_auc_95ci":[ci[0],ci[1]],
        "feature_names":FEATURE_NAMES
    }
    with open(os.path.join(outdir,"metadata.json"),"w") as f: json.dump(meta,f,indent=2)
    return df,res,meta,bad

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/raw")
    ap.add_argument("--out",default="models")
    a=ap.parse_args()
    df,res,meta,bad=train(a.data,a.out)
    print(res.to_string(index=False))
    print(meta)
