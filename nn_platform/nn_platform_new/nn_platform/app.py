import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from nn_engine import (
    MLP, train, DATASETS,
    classification_report_dict, regression_metrics,
    bce_loss,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Networks Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dashboard CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: #1a2a4a !important;
    min-width: 230px !important; max-width: 230px !important;
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: #c8d8f0 !important; font-size: 13px !important; }
[data-testid="stSidebar"] h1 { color: #ffffff !important; font-size: 17px !important; font-weight: 700 !important; }
[data-testid="stSidebar"] h3 { color: #7aa3d4 !important; font-size: 11px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: .07em; margin-top: 14px !important; }
[data-testid="stSidebar"] hr { border-color: #2d4470 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #2563eb !important; color: #fff !important; border: none !important;
    border-radius: 7px !important; font-size: 13px !important; font-weight: 600 !important;
    width: 100% !important; padding: 10px !important; margin-top: 6px !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #1d4ed8 !important; }

.main .block-container {
    background: #f0f4fa !important;
    padding: 20px 28px 32px 28px !important;
    max-width: 100% !important;
}

.dash-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 14px; border-bottom: 1.5px solid #dde3ee; margin-bottom: 18px;
}
.dash-topbar-title { font-size: 17px; font-weight: 700; color: #1a2a4a; }
.dash-topbar-right { display: flex; gap: 8px; }
.btn-blue { background:#2563eb;color:#fff;border:none;border-radius:6px;padding:6px 18px;font-size:13px;font-weight:500;cursor:pointer; }
.btn-white { background:#fff;color:#374151;border:1.5px solid #d1d5db;border-radius:6px;padding:6px 14px;font-size:13px;cursor:pointer; }

.chart-card {
    background: #ffffff;
    border: 1px solid #dde3ee;
    border-radius: 10px;
    padding: 16px 18px 12px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 5px rgba(26,42,74,0.06);
}
.card-title   { font-size: 13px; font-weight: 600; color: #1a2a4a; margin-bottom: 2px; }
.card-caption { font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px; }
.card-footer  { text-align: center; margin-top: 10px; padding-top: 10px; border-top: 1px solid #f1f3f8; }
.btn-edit { background:#fff;color:#2563eb;border:1.5px solid #2563eb;border-radius:6px;padding:5px 20px;font-size:12px;font-weight:500;cursor:pointer; }

.metric-strip { display:flex; gap:10px; margin: 14px 0 6px 0; }
.mpill { flex:1; background:#fff; border:1px solid #dde3ee; border-radius:8px; padding:12px 10px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.mpill-val { font-size:1.3rem; font-weight:700; color:#2563eb; }
.mpill-lbl { font-size:10px; color:#9ca3af; margin-top:2px; }

.status-ok   { background:#f0fdf4;border:1px solid #86efac;border-radius:7px;padding:7px 12px;color:#166534;font-size:12px;margin-top:8px; }
.status-warn { background:#fffbeb;border:1px solid #fcd34d;border-radius:7px;padding:7px 12px;color:#92400e;font-size:12px;margin-top:8px; }
.status-info { background:#eff6ff;border:1px solid #bfdbfe;border-radius:7px;padding:7px 12px;color:#1e40af;font-size:12px;margin-top:8px; }

.stTabs [data-baseweb="tab-list"] {
    background:#fff; border-radius:8px; border:1px solid #dde3ee;
    padding:3px 5px; gap:3px; margin-bottom:14px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:6px !important; font-size:13px !important;
    font-weight:500 !important; color:#6b7280 !important; padding:5px 14px !important;
}
.stTabs [aria-selected="true"] { background:#2563eb !important; color:#fff !important; }

.empty-state {
    background:#fff; border:1px solid #dde3ee; border-radius:10px;
    padding:48px 0; text-align:center; color:#9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("history",None),("model",None),
             ("X_train",None),("y_train",None),
             ("X_test",None),("y_test",None),("trained",False)]:
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 NN Platform")
    st.markdown("---")
    st.markdown("### Dataset")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), label_visibility="collapsed")
    prob_type    = st.selectbox("Problem type",
        ["Binary classification","Multi-class classification","Regression"])
    n_samples  = st.slider("Samples", 100, 600, 300, 50)
    noise      = st.slider("Noise", 0.0, 0.4, 0.1, 0.05)
    test_split = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    uploaded   = st.file_uploader("Upload CSV (last col = label)", type="csv")

    st.markdown("### Architecture")
    model_type = st.selectbox("Model",[
        "Perceptron (historical)","Perceptron + activation","MLP (multi-layer)"])
    hidden_act = st.selectbox("Hidden activation",["relu","sigmoid","tanh"])
    n_hidden_layers, neurons_per_layer = 2, 8
    if model_type == "MLP (multi-layer)":
        n_hidden_layers   = st.slider("Hidden layers", 1, 5, 2)
        neurons_per_layer = st.slider("Neurons / layer", 2, 32, 8)

    st.markdown("### Training")
    lr         = st.select_slider("Learning rate",
                    options=[0.0001,0.001,0.005,0.01,0.05,0.1], value=0.01)
    epochs     = st.slider("Epochs", 50, 500, 200, 50)
    batch_size = st.select_slider("Batch size", options=[8,16,32,64,128], value=32)
    loss_options = {
        "Binary classification":["bce","mse"],
        "Multi-class classification":["cce"],
        "Regression":["mse"],
    }
    loss_fn = st.selectbox("Loss function", loss_options[prob_type])

    st.markdown("### Regularisation")
    reg_method   = st.selectbox("Method",["None","L2","Dropout","Early stopping"])
    l2_lambda    = st.slider("L2 λ",0.0,0.1,0.001,0.001) if reg_method=="L2" else 0.0
    dropout_rate = st.slider("Dropout",0.0,0.5,0.2,0.05)  if reg_method=="Dropout" else 0.0
    es_patience  = st.slider("Patience",5,30,10)            if reg_method=="Early stopping" else 0
    st.markdown("---")
    train_btn = st.button("🚀  Train model")

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
COLORS = ["#2563eb","#7c3aed","#059669","#d97706","#dc2626","#0891b2"]

def mfig(w=5, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#dde3ee")
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.grid(color="#eef0f5", linewidth=0.7, zorder=0)
    return fig, ax

def render_card(title, caption="CAPTION TEXT"):
    st.markdown(f"""<div class="chart-card">
    <div class="card-title">{title}</div>
    <div class="card-caption">{caption}</div>""", unsafe_allow_html=True)

def close_card():
    st.markdown("""<div class="card-footer">
    <button class="btn-edit">Edit report</button></div></div>""", unsafe_allow_html=True)

def load_data():
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        X  = df.iloc[:,:-1].values.astype(np.float32)
        y  = df.iloc[:,-1].values
        X  = (X-X.mean(0))/(X.std(0)+1e-9)
        if prob_type=="Regression":
            y=(y-y.min())/(y.max()-y.min()+1e-9)
            return X, y.reshape(-1,1).astype(np.float32)
        classes=np.unique(y)
        if prob_type=="Binary classification":
            return X,(y==classes[1]).astype(np.float32).reshape(-1,1)
        n_c=len(classes); oh=np.zeros((len(y),n_c),dtype=np.float32)
        for i,c in enumerate(classes): oh[y==c,i]=1
        return X,oh
    fn=DATASETS[dataset_name]
    try: return fn(n=n_samples,noise=noise)
    except TypeError: return fn(n=n_samples)

def build_arch(in_dim, out_dim):
    if "historical" in model_type: return [in_dim,out_dim],"linear"
    if "+" in model_type:          return [in_dim,out_dim],hidden_act
    return [in_dim]+[neurons_per_layer]*n_hidden_layers+[out_dim],hidden_act

def plot_loss(h):
    fig,ax=mfig(5,2.8); ep=range(1,len(h["train_loss"])+1)
    ax.plot(ep,h["train_loss"],color="#2563eb",lw=1.8,label="Train")
    ax.plot(ep,h["val_loss"],  color="#7c3aed",lw=1.8,ls="--",label="Val")
    ax.fill_between(ep,h["train_loss"],alpha=0.07,color="#2563eb")
    ax.set_xlabel("Epoch",fontsize=9,color="#6b7280"); ax.set_ylabel("Loss",fontsize=9,color="#6b7280")
    ax.legend(fontsize=8,framealpha=0); fig.tight_layout(pad=1.2); return fig

def plot_acc(h):
    if h["train_acc"][0] is None: return None
    fig,ax=mfig(5,2.8); ep=range(1,len(h["train_acc"])+1)
    ta=[v*100 for v in h["train_acc"]]; va=[v*100 for v in h["val_acc"]]
    ax.plot(ep,ta,color="#059669",lw=1.8,label="Train")
    ax.plot(ep,va,color="#d97706",lw=1.8,ls="--",label="Val")
    ax.fill_between(ep,ta,alpha=0.07,color="#059669")
    ax.set_ylim(0,105); ax.set_xlabel("Epoch",fontsize=9,color="#6b7280")
    ax.set_ylabel("Accuracy (%)",fontsize=9,color="#6b7280")
    ax.legend(fontsize=8,framealpha=0); fig.tight_layout(pad=1.2); return fig

def plot_boundary(model, X, y, title=""):
    if X.shape[1]!=2: return None
    fig,ax=mfig(4.5,3.2)
    x0mn,x0mx=X[:,0].min()-.35,X[:,0].max()+.35
    x1mn,x1mx=X[:,1].min()-.35,X[:,1].max()+.35
    xx,yy=np.meshgrid(np.linspace(x0mn,x0mx,120),np.linspace(x1mn,x1mx,120))
    grid=np.c_[xx.ravel(),yy.ravel()].astype(np.float32)
    Z=model.predict(grid)
    Z=Z.reshape(xx.shape) if Z.shape[1]==1 else np.argmax(Z,1).reshape(xx.shape)
    ax.contourf(xx,yy,Z,alpha=0.22,cmap=ListedColormap(["#dbeafe","#ede9fe","#d1fae5"]))
    ax.contour(xx,yy,Z,colors="#94a3b8",linewidths=0.8)
    yf=y.ravel() if y.shape[1]==1 else np.argmax(y,1)
    ax.scatter(X[:,0],X[:,1],c=yf,cmap=ListedColormap(["#2563eb","#7c3aed","#059669"]),
               s=16,edgecolors="white",linewidths=0.4,zorder=3)
    ax.set_title(title,fontsize=10,color="#1a2a4a",pad=5)
    fig.tight_layout(pad=1.2); return fig

def plot_cm(cm, n_cls):
    fig,ax=plt.subplots(figsize=(3.8,3.2)); fig.patch.set_facecolor("white")
    im=ax.imshow(cm,cmap="Blues",aspect="auto")
    plt.colorbar(im,ax=ax,fraction=0.04)
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xlabel("Predicted",fontsize=9,color="#6b7280")
    ax.set_ylabel("Actual",fontsize=9,color="#6b7280")
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=12,
                    color="white" if cm[i,j]>cm.max()/2 else "#1a2a4a")
    ax.spines[:].set_visible(False); fig.tight_layout(pad=1.2); return fig

def hbar(names, vals, xlabel):
    fig,ax=mfig(5,max(2.6,len(names)*0.55))
    bars=ax.barh(names,vals,color=COLORS[:len(names)],height=0.5,zorder=3)
    ax.set_xlim(0,max(vals)*1.18)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_width()+max(vals)*0.02,bar.get_y()+bar.get_height()/2,
                f"{v:.3f}",va="center",fontsize=8,color="#374151")
    ax.set_xlabel(xlabel,fontsize=9,color="#6b7280"); ax.invert_yaxis()
    fig.tight_layout(pad=1.2); return fig

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
if train_btn:
    X_all,y_all=load_data()
    n=len(X_all); n_test=max(10,int(n*test_split))
    idx=np.random.permutation(n)
    X_tr,y_tr=X_all[idx[n_test:]],y_all[idx[n_test:]]
    X_te,y_te=X_all[idx[:n_test]], y_all[idx[:n_test]]
    layers,act_used=build_arch(X_tr.shape[1],y_tr.shape[1])
    out_act="sigmoid" if loss_fn=="bce" else ("softmax" if loss_fn=="cce" else "linear")
    model=MLP(layers=layers,hidden_act=act_used,output_act=out_act,
              loss=loss_fn,l2_lambda=l2_lambda,dropout_rate=dropout_rate)
    st.session_state.update({"X_train":X_tr,"y_train":y_tr,"X_test":X_te,"y_test":y_te,
                              "model":model,"trained":False})
    with st.spinner("Training…"):
        prog=st.progress(0)
        live={"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
        def cb(epoch,total,m):
            live["train_loss"].append(m["t_loss"]); live["val_loss"].append(m["v_loss"])
            live["train_acc"].append(m["t_acc"]);   live["val_acc"].append(m["v_acc"])
            prog.progress(epoch/total)
        history=train(model,X_tr,y_tr,X_te,y_te,lr=lr,epochs=epochs,
                      batch_size=batch_size,early_stopping_patience=es_patience,
                      progress_callback=cb)
        st.session_state["history"]=history; st.session_state["trained"]=True
        prog.empty()
    st.success("✅ Training complete!")

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-topbar">
  <div class="dash-topbar-title">My Dashboard</div>
  <div class="dash-topbar-right">
    <button class="btn-blue">Share</button>
    <button class="btn-white">Export ▾</button>
    <button class="btn-white">···</button>
  </div>
</div>""", unsafe_allow_html=True)

tabs=st.tabs(["📈 Training","📋 Results","🔬 Boundaries","📚 Theory","⚗️ Experiments"])
tab_train,tab_results,tab_boundary,tab_theory,tab_exp=tabs

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    if not st.session_state["trained"]:
        st.markdown("""<div class="empty-state">
          <div style="font-size:2.2rem;margin-bottom:10px">🧠</div>
          <div style="font-size:15px;font-weight:600;color:#1a2a4a;margin-bottom:5px">No model trained yet</div>
          <div style="font-size:13px">Configure in the sidebar, then click <b>Train model</b>.</div>
        </div>""", unsafe_allow_html=True)
    else:
        h=st.session_state["history"]
        col1,col2=st.columns(2)
        with col1:
            render_card("Loss curve","TRAINING PROGRESS")
            st.pyplot(plot_loss(h))
            close_card()
        with col2:
            render_card("Accuracy curve","TRAINING PROGRESS")
            fig_acc=plot_acc(h)
            if fig_acc: st.pyplot(fig_acc)
            else: st.markdown("<div style='padding:30px;text-align:center;color:#9ca3af;font-size:13px'>N/A for regression tasks</div>",unsafe_allow_html=True)
            close_card()

        # metric strip
        ep_done=len(h["train_loss"]); gap=h["val_loss"][-1]-h["train_loss"][-1]
        ta=f"{h['train_acc'][-1]*100:.1f}%" if h["train_acc"][-1] is not None else "—"
        va=f"{h['val_acc'][-1]*100:.1f}%"   if h["val_acc"][-1]   is not None else "—"
        st.markdown(f"""<div class="metric-strip">
          <div class="mpill"><div class="mpill-val">{h['train_loss'][-1]:.4f}</div><div class="mpill-lbl">Train loss</div></div>
          <div class="mpill"><div class="mpill-val">{h['val_loss'][-1]:.4f}</div><div class="mpill-lbl">Val loss</div></div>
          <div class="mpill"><div class="mpill-val">{ta}</div><div class="mpill-lbl">Train acc</div></div>
          <div class="mpill"><div class="mpill-val">{va}</div><div class="mpill-lbl">Val acc</div></div>
          <div class="mpill"><div class="mpill-val">{ep_done}</div><div class="mpill-lbl">Epochs</div></div>
          <div class="mpill"><div class="mpill-val">{gap:.4f}</div><div class="mpill-lbl">Gen. gap</div></div>
        </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state["trained"]:
        st.info("Train a model first.")
    else:
        model=st.session_state["model"]; X_te=st.session_state["X_test"]; y_te=st.session_state["y_test"]
        y_pred=model.predict(X_te)
        if prob_type=="Regression":
            rm=regression_metrics(y_te,y_pred)
            col1,col2=st.columns([1,1])
            with col1:
                render_card("Regression metrics","MODEL PERFORMANCE")
                fig,ax=mfig(4.5,2.6)
                labels=["MSE","RMSE","R²"]; vals=[rm["mse"],rm["rmse"],max(0,rm["r2"])]
                bars=ax.bar(labels,vals,color=COLORS[:3],width=0.45,zorder=3)
                for bar,v in zip(bars,vals):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                            f"{v:.4f}",ha="center",fontsize=8,color="#374151")
                ax.set_ylabel("Value",fontsize=9,color="#6b7280"); fig.tight_layout(pad=1.2)
                st.pyplot(fig); close_card()
        else:
            y_pred_cls=model.predict_classes(X_te)
            n_cls=y_te.shape[1] if y_te.ndim>1 else 2
            y_true_cls=y_te.ravel() if n_cls==1 else np.argmax(y_te,1)
            report=classification_report_dict(y_true_cls,y_pred_cls,n_classes=max(n_cls,2))
            pc=report["per_class"]
            avg_f1=np.mean([pc[c]["f1"] for c in pc]); avg_prec=np.mean([pc[c]["precision"] for c in pc])
            avg_rec=np.mean([pc[c]["recall"] for c in pc])
            col1,col2,col3=st.columns(3)
            with col1:
                render_card("Classification metrics","MODEL PERFORMANCE")
                st.pyplot(hbar(["Accuracy","F1 score","Precision","Recall"],
                               [report["accuracy"],avg_f1,avg_prec,avg_rec],"Score"))
                close_card()
            with col2:
                render_card("Confusion matrix","ACTUAL VS PREDICTED")
                st.pyplot(plot_cm(report["confusion_matrix"],max(n_cls,2)))
                close_card()
            with col3:
                render_card("Per-class report","BREAKDOWN")
                rows=[{"Class":c,"Prec":f"{v['precision']:.3f}","Rec":f"{v['recall']:.3f}","F1":f"{v['f1']:.3f}"}
                      for c,v in pc.items()]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                gap=st.session_state["history"]["val_loss"][-1]-st.session_state["history"]["train_loss"][-1]
                if gap>0.05:
                    st.markdown(f'<div class="status-warn">⚠️ Possible overfitting (gap={gap:.3f})</div>',unsafe_allow_html=True)
                elif st.session_state["history"]["train_loss"][-1]>0.4:
                    st.markdown('<div class="status-info">ℹ️ High train loss — try more layers</div>',unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-ok">✅ Good fit — gap={gap:.3f}</div>',unsafe_allow_html=True)
                close_card()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: BOUNDARIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_boundary:
    if not st.session_state["trained"]:
        st.info("Train a model first.")
    else:
        model=st.session_state["model"]
        Xtr=st.session_state["X_train"]; ytr=st.session_state["y_train"]
        Xte=st.session_state["X_test"];  yte=st.session_state["y_test"]
        if Xtr.shape[1]==2:
            col1,col2=st.columns(2)
            with col1:
                render_card("Decision boundary — train set","2D VISUALISATION")
                fig=plot_boundary(model,Xtr,ytr,"Train set")
                if fig: st.pyplot(fig)
                close_card()
            with col2:
                render_card("Decision boundary — test set","2D VISUALISATION")
                fig=plot_boundary(model,Xte,yte,"Test set")
                if fig: st.pyplot(fig)
                close_card()
        else:
            st.markdown('<div class="chart-card" style="color:#9ca3af;font-size:13px;padding:32px;text-align:center">Decision boundary requires 2D input (2 features).</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: THEORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_theory:
    sections=[
        ("🔬 Perceptron",
         "The **perceptron** (Rosenblatt, 1958) is the simplest neural unit — a weighted sum followed by an activation function. The historical perceptron uses a step function; adding a differentiable activation enables gradient-based learning.",
         "ŷ = f(w·x + b)"),
        ("🔁 Backpropagation",
         "Backpropagation applies the **chain rule** to compute gradients layer by layer, propagating the error from the output backwards. Each weight is updated: `w ← w − η · ∂L/∂w`.",
         "∂L/∂W¹ = ∂L/∂ŷ · ∂ŷ/∂z² · ∂z²/∂h · ∂h/∂z¹ · x"),
        ("⚡ Activation functions",
         "| Function | Formula | Range | Use |\n|---|---|---|---|\n| Sigmoid | 1/(1+e⁻ˣ) | (0,1) | Output binary |\n| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | (−1,1) | Hidden |\n| ReLU | max(0,x) | [0,∞) | Hidden (default) |\n| Softmax | eˣⁱ/Σeˣʲ | (0,1) | Output multi-class |",
         None),
        ("📉 Loss functions",
         "**MSE:** `L = Σ(y−ŷ)²/n` — regression.\n\n**Binary CE:** `L = −[y·log(ŷ)+(1−y)·log(1−ŷ)]` — binary classification.\n\n**Categorical CE:** `L = −Σ yᵢ·log(ŷᵢ)` — multi-class.",
         None),
        ("🛡️ Regularisation",
         "| Technique | Mechanism |\n|---|---|\n| **L2** | Adds λΣw² to loss, penalises large weights |\n| **Dropout** | Randomly zeros neurons during training |\n| **Early stopping** | Halts when val loss stops improving |",
         "L_reg = L + λ·Σw²"),
        ("📊 Bias vs Variance",
         "**Bias** = too simple → underfitting. **Variance** = too complex → overfitting.\n\n`Error ≈ Bias² + Variance + Noise`\n\nDiverging train/val curves signal overfitting; both curves high signal underfitting.",
         None),
    ]
    for i in range(0,len(sections),2):
        cols=st.columns(2)
        for j,col in enumerate(cols):
            if i+j>=len(sections): break
            title,text,formula=sections[i+j]
            with col:
                render_card(title,"THEORY")
                st.markdown(text)
                if formula: st.code(formula,language="text")
                close_card()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_exp:
    render_card("Experiment runner","AUTOMATED COMPARISONS")
    exp_choice=st.radio("Experiment",["Exp 1 — Perceptron vs MLP",
        "Exp 2 — Effect of depth","Exp 3 — Overfitting vs regularisation"],
        horizontal=True,label_visibility="collapsed")
    run_exp=st.button("▶  Run experiment",key="run_exp")
    st.markdown("</div>",unsafe_allow_html=True)

    if run_exp:
        from nn_engine import make_circles
        Xa,ya=make_circles(n=300,noise=0.1)
        n=len(Xa); idx=np.random.permutation(n); sp=int(n*.8)
        Xtr,ytr=Xa[idx[:sp]],ya[idx[:sp]]; Xte,yte=Xa[idx[sp:]],ya[idx[sp:]]

        def qtrain(layers,act,l2=0,dr=0,ep=120):
            m=MLP(layers,hidden_act=act,output_act="sigmoid",loss="bce",l2_lambda=l2,dropout_rate=dr)
            train(m,Xtr,ytr,Xte,yte,lr=0.01,epochs=ep,batch_size=32)
            acc=np.mean(m.predict_classes(Xte)==yte.ravel())
            lv=bce_loss(m.predict(Xte),yte)
            return m,acc,lv

        if "Exp 1" in exp_choice:
            cfgs=[("Perceptron",[2,1],"linear",0,0),("Perceptron+σ",[2,1],"sigmoid",0,0),
                  ("MLP 1 hidden",[2,8,1],"relu",0,0),("MLP 2 hidden",[2,16,16,1],"relu",0,0)]
        elif "Exp 2" in exp_choice:
            cfgs=[(f"{d} layer{'s' if d>1 else ''}",[2]+[8]*d+[1],"relu",0,0) for d in range(1,6)]
        else:
            cfgs=[("No reg",[2,32,32,1],"relu",0,0),("L2 λ=0.001",[2,32,32,1],"relu",0.001,0),
                  ("Dropout 30%",[2,32,32,1],"relu",0,0.3)]

        results=[]; prog=st.progress(0)
        for ci,(name,layers,act,l2,dr) in enumerate(cfgs):
            m,acc,lv=qtrain(layers,act,l2=l2,dr=dr)
            results.append({"name":name,"acc":acc,"loss":lv,"model":m})
            prog.progress((ci+1)/len(cfgs))
        prog.empty()

        col1,col2=st.columns(2)
        with col1:
            render_card("Test accuracy comparison","EXPERIMENT RESULTS")
            st.pyplot(hbar([r["name"] for r in results],[r["acc"]*100 for r in results],"Test accuracy (%)"))
            close_card()
        with col2:
            render_card("Test loss comparison","EXPERIMENT RESULTS")
            st.pyplot(hbar([r["name"] for r in results],[r["loss"] for r in results],"Test loss"))
            close_card()

        if all(r["model"].layers[0]==2 for r in results):
            render_card("Decision boundaries","PER MODEL")
            n_cols=min(len(results),3); cols=st.columns(n_cols)
            for i,res in enumerate(results):
                fig=plot_boundary(res["model"],Xte,yte,res["name"])
                if fig: cols[i%n_cols].pyplot(fig)
            close_card()
