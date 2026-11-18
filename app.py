import os,sys,zipfile,tempfile,shutil,json
import streamlit as st
import pandas as pd, numpy as np, cv2, joblib
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"src"))
from research_train import train,save_feature_correlation_heatmap
from features import extract_features,FEATURE_NAMES

st.set_page_config(page_title="Research Spiral Analysis",page_icon="🌀",layout="wide")

MODEL_DIR="models"
st.title("🌀 Parkinson's Spiral Analysis")
st.caption("Research prototype • subject-level evaluation • calibrated interpretation framework")

tabs=st.tabs(["🚀 Train Model","🔍 Analyze Spiral","📊 Research Results"])

with tabs[0]:
    st.subheader("Train from your dataset")
    st.info("Expected ZIP structure: control/ and parkinson/ folders. For scientific validity, filenames should identify the participant/subject consistently.")
    z=st.file_uploader("Upload dataset ZIP",type=["zip"],key="dataset")
    if z and st.button("🚀 Train Model",type="primary"):
        with st.spinner("Extracting images, engineering features and training ML + CNN models..."):
            tmp=tempfile.mkdtemp()
            try:
                zp=os.path.join(tmp,"dataset.zip")
                open(zp,"wb").write(z.getbuffer())
                extract=os.path.join(tmp,"data")
                with zipfile.ZipFile(zp) as zz: zz.extractall(extract)
                # Locate folders if ZIP contains a top-level directory.
                candidates=[]
                for r,d,f in os.walk(extract):
                    if os.path.isdir(os.path.join(r,"control")) and os.path.isdir(os.path.join(r,"parkinson")):
                        candidates.append(r)
                if not candidates: raise ValueError("ZIP must contain control/ and parkinson/ folders.")
                data_root=candidates[0]
                df,res,meta,bad=train(data_root,MODEL_DIR)
                st.success("Training completed.")
                st.write(f"Images: {len(df)} • Subjects: {df.subject.nunique()}")
                st.dataframe(res,use_container_width=True)
                st.subheader("Held-out ROC-AUC 95% bootstrap interval")
                lo,hi=meta["roc_auc_95ci"]
                st.metric("Best model",meta["best_model"])
                st.write(f"ROC-AUC 95% CI: **{lo:.3f} – {hi:.3f}**")
                if bad: st.warning(f"{len(bad)} images were skipped.")
            finally:
                shutil.rmtree(tmp,ignore_errors=True)

with tabs[1]:
    available=[]
    for fn in ["svm.joblib","random_forest.joblib","xgboost.joblib","cnn.keras"]:
        if os.path.exists(os.path.join(MODEL_DIR,fn)): available.append(fn)
    if not available:
        st.warning("Train a model first.")
    else:
        model_file=st.selectbox("Model",available)
        up=st.file_uploader("Upload a spiral drawing",type=["png","jpg","jpeg","bmp","webp"],key="spiral")
        if up:
            arr=np.frombuffer(up.getvalue(),np.uint8)
            img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            try:
                feat,mask=extract_features(img)
                if model_file=="cnn.keras":
                    import tensorflow as tf
                    model=tf.keras.models.load_model(os.path.join(MODEL_DIR,model_file))
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
                    gray=cv2.resize(gray,(128,128),interpolation=cv2.INTER_AREA)
                    cnn_input=(gray.astype("float32")/255.0)[np.newaxis,...,np.newaxis]
                    score=float(model.predict(cnn_input,verbose=0).reshape(-1)[0])
                    metadata_path=os.path.join(MODEL_DIR,"metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path,"r") as f:
                            metadata=json.load(f)
                        threshold=float(metadata.get("cnn_threshold",0.5))
                    else:
                        threshold=0.5
                else:
                    model=joblib.load(os.path.join(MODEL_DIR,model_file))
                    score=float(model.predict_proba(feat.reshape(1,-1))[0,1])
                    threshold=0.5
                st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),caption="Input drawing",width=420)

                # Do NOT present this score as a medical probability.
                pred=int(score>=threshold)
                if abs(score-threshold)<=.10:
                    st.warning("🟠 **Indeterminate model output**")
                    st.write("The model score is close to its decision boundary. This drawing should not be interpreted as a clear classification.")
                    status="Indeterminate"
                elif pred==0:
                    st.success("🟢 **Pattern is more consistent with the Control class**")
                    status="Control-side"
                else:
                    st.warning("🟠 **Pattern is more consistent with the Parkinson class**")
                    status="Parkinson-side"

                st.metric("Model score for Parkinson class",f"{score*100:.1f}%")
                st.progress(score)
                if model_file=="cnn.keras":
                    st.caption(f"CNN decision cutoff: {threshold:.3f} (selected using validation data)")

                st.info("This is an ML score, not a clinical probability. It does not diagnose Parkinson's disease. Clinical assessment is required for diagnosis.")
                with st.expander("Technical features"):
                    st.dataframe(pd.DataFrame({"Feature":FEATURE_NAMES,"Value":feat}),use_container_width=True)
            except Exception as e: st.error(str(e))

with tabs[2]:
    p=os.path.join(MODEL_DIR,"test_metrics.csv")
    m=os.path.join(MODEL_DIR,"metadata.json")
    if os.path.exists(p):
        st.subheader("Held-out subject-level test performance")
        metrics_df=pd.read_csv(p)
        st.dataframe(metrics_df,use_container_width=True)
        st.caption("Metrics are calculated on subjects excluded from training. Performance estimates depend on dataset size and representativeness.")

        figdir=os.path.join(MODEL_DIR,"figures")
        correlation_figure=os.path.join(figdir,"feature_correlation_heatmap.png")
        feature_table=os.path.join(MODEL_DIR,"features.csv")
        if not os.path.exists(correlation_figure) and os.path.exists(feature_table):
            save_feature_correlation_heatmap(
                pd.read_csv(feature_table),correlation_figure
            )
        st.subheader("Research figures")
        figure_specs=[
            ("class_distribution.png","1. Dataset class distribution"),
            ("model_comparison.png","2. Model performance comparison"),
            ("confusion_matrix.png","3. Confusion matrix"),
            ("roc_curve.png","4. Receiver operating characteristic (ROC)"),
            ("precision_recall_curve.png","5. Precision–recall curve"),
            ("score_distribution.png","6. Held-out score distribution"),
            ("feature_importance.png","7. Feature importance"),
            ("feature_correlation_heatmap.png","8. Feature correlation heatmap")
        ]
        for fn,title in figure_specs:
            fp=os.path.join(figdir,fn)
            if os.path.exists(fp):
                st.markdown(f"#### {title}")
                st.image(fp,use_container_width=True)
                with open(fp,"rb") as f:
                    st.download_button(f"Download {fn}",f,file_name=fn,mime="image/png",key="dl_"+fn)
    else:
        st.info("Train a model to populate research metrics and figures.")
