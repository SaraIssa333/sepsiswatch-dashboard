"""
SepsisWatch ICU Analytics — MSBA382 Healthcare Analytics
PhysioNet 2019 Challenge — Real ICU Data: BIDMC Boston + Emory Atlanta
40,336 patients · 41 clinical variables · Password: SEPSIS2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SepsisWatch ICU", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

# ── DESIGN TOKENS ─────────────────────────────────────────
DARK="#0A0F1E"; CARD="#111827"; BORDER="#1E2D45"
RED="#E84855"; ORANGE="#F5A623"; TEAL="#00BFA5"
BLUE="#2979FF"; MUTED="#6B7A8D"; WHITE="#EEF2FF"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'IBM Plex Sans',sans-serif;background-color:{DARK};color:{WHITE};}}
#MainMenu,footer,header{{visibility:hidden;}}
[data-testid="stSidebar"]{{background-color:{CARD}!important;border-right:1px solid {BORDER};}}
[data-testid="stSidebar"] *{{color:{WHITE}!important;}}
[data-testid="stSidebar"] .stSelectbox>div>div{{background-color:{DARK}!important;border:1px solid {BORDER}!important;}}
.main .block-container{{background-color:{DARK};padding-top:1.2rem;max-width:1440px;}}
.kpi-card{{background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:18px 20px;text-align:center;position:relative;overflow:hidden;margin-bottom:6px;}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.kpi-red::before{{background:{RED};}} .kpi-orange::before{{background:{ORANGE};}}
.kpi-teal::before{{background:{TEAL};}} .kpi-blue::before{{background:{BLUE};}}
.kpi-value{{font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:700;line-height:1.1;}}
.kpi-label{{font-size:0.7rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.12em;margin-top:4px;}}
.section-hdr{{font-size:1.05rem;font-weight:600;color:{WHITE};padding:10px 0 8px 0;border-bottom:1px solid {BORDER};margin-bottom:14px;}}
.eyebrow{{font-size:0.65rem;color:{TEAL};text-transform:uppercase;letter-spacing:0.15em;font-weight:600;}}
.title-band{{background:linear-gradient(135deg,{CARD} 55%,#0D1B30);border:1px solid {BORDER};border-radius:10px;padding:22px 28px;margin-bottom:20px;}}
.title-main{{font-size:1.8rem;font-weight:700;color:{WHITE};letter-spacing:-0.01em;}}
.title-sub{{font-size:0.85rem;color:{MUTED};margin-top:3px;}}
.badge{{display:inline-block;background:{RED};color:white;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;font-weight:700;padding:3px 10px;border-radius:20px;margin-top:8px;letter-spacing:0.06em;}}
@keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(232,72,85,0.4);}}70%{{box-shadow:0 0 0 10px rgba(232,72,85,0);}}100%{{box-shadow:0 0 0 0 rgba(232,72,85,0);}}}}
.kpi-alert{{animation:pulse 2.5s infinite;}}
[data-baseweb="tab-list"]{{background:{CARD}!important;border-radius:8px;border:1px solid {BORDER};padding:3px;}}
[data-baseweb="tab"]{{color:{MUTED}!important;font-size:0.82rem;}}
[aria-selected="true"]{{background:{RED}!important;color:white!important;border-radius:6px!important;font-weight:600!important;}}
[data-testid="metric-container"]{{background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:14px;}}
[data-testid="metric-container"] label{{color:{MUTED}!important;font-size:0.8rem!important;}}
[data-testid="metric-container"] [data-testid="metric-value"]{{color:{TEAL}!important;font-family:'IBM Plex Mono',monospace!important;font-size:1.6rem!important;}}
::-webkit-scrollbar{{width:5px;background:{DARK};}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:3px;}}
</style>
"""

CHART = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(family="IBM Plex Sans", color=WHITE),
             margin=dict(t=40,b=40,l=50,r=20),
             colorway=[RED,TEAL,ORANGE,BLUE,"#AB47BC","#26C6DA"],
             xaxis=dict(gridcolor=BORDER,linecolor=BORDER),
             yaxis=dict(gridcolor=BORDER,linecolor=BORDER),
             legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=WHITE)))

def chart(fig, title="", h=370):
    fig.update_layout(**CHART, title=dict(text=title, font=dict(size=13,color=WHITE)), height=h)
    fig.update_xaxes(showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1)
    return fig

def kpi(col, val, label, color="red", alert=False):
    cm = {"red":RED,"orange":ORANGE,"teal":TEAL,"blue":BLUE}
    cls = f"kpi-card kpi-{color}" + (" kpi-alert" if alert else "")
    col.markdown(f"""<div class="{cls}">
      <div class="kpi-value" style="color:{cm[color]};">{val}</div>
      <div class="kpi-label">{label}</div></div>""", unsafe_allow_html=True)

# ── AUTH ──────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;">
      <div style="text-align:center;margin-bottom:24px;">
        <div style="font-size:4rem;">🫀</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:2.2rem;font-weight:700;color:{WHITE};">
          Sepsis<span style="color:{RED};">Watch</span></div>
        <div style="font-size:0.9rem;color:{MUTED};margin-top:6px;">ICU Analytics Platform — MSBA382 Healthcare Analytics</div>
      </div>
      <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:36px 40px;text-align:center;width:380px;">
        <div style="font-size:0.8rem;color:{MUTED};margin-bottom:20px;line-height:1.6;">
          <b style="color:{WHITE};">Data Source:</b> PhysioNet 2019 Challenge<br>
          <b style="color:{WHITE};">Hospitals:</b> BIDMC Boston · Emory Atlanta<br>
          <b style="color:{WHITE};">Patients:</b> 40,336 real ICU records<br>
          <b style="color:{WHITE};">Variables:</b> 41 clinical features
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        pw = st.text_input("", type="password", placeholder="🔒 Enter dashboard password…")
        if st.button("Unlock Dashboard", use_container_width=True):
            if pw == "SEPSIS2026":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Incorrect password. Hint: SEPSIS2026")
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("sepsis_full.csv")
    # Add hospital & ICU unit labels
    df['hospital'] = df['dataset'].map({'A':'BIDMC - Boston, MA','B':'Emory - Atlanta, GA'})
    df['icu_unit'] = df['unit1'].map({1.0:'Medical ICU (MICU)',0.0:'Surgical ICU (SICU)',np.nan:'Unspecified'}).fillna('Unspecified')
    df['gender_label'] = df['gender'].map({1:'Male',0:'Female'})
    return df

df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style="text-align:center;padding:14px 0 18px;">
      <div style="font-size:2.2rem;">🫀</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:700;color:{WHITE};">
        Sepsis<span style="color:{RED};">Watch</span></div>
      <div style="font-size:0.68rem;color:{MUTED};">ICU Analytics · MSBA382</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="eyebrow">Patient Filters</div>', unsafe_allow_html=True)
    hospital_sel = st.multiselect("Hospital", ["BIDMC - Boston, MA","Emory - Atlanta, GA"],
                                  default=["BIDMC - Boston, MA","Emory - Atlanta, GA"])
    age_range = st.slider("Age Range", int(df.age.min()), int(df.age.max()),
                          (int(df.age.min()), int(df.age.max())))
    gender_filter = st.multiselect("Gender", ["Male","Female"], default=["Male","Female"])
    sepsis_filter = st.selectbox("Sepsis Status", ["All","Sepsis Only","Non-Sepsis Only"])
    icu_max = st.slider("Max ICU Stay (hours)", 12, 336, 336, step=12)
    unit_filter = st.multiselect("ICU Unit Type",
                                 ["Medical ICU (MICU)","Surgical ICU (SICU)","Unspecified"],
                                 default=["Medical ICU (MICU)","Surgical ICU (SICU)","Unspecified"])
    st.markdown("---")
    st.markdown(f"""<div style="font-size:0.7rem;color:{MUTED};line-height:1.8;">
    <b style="color:{WHITE};">Source:</b> PhysioNet 2019 Challenge<br>
    <b style="color:{WHITE};">Hospitals:</b> BIDMC + Emory<br>
    <b style="color:{WHITE};">Patients:</b> 40,336 ICU records<br>
    <b style="color:{WHITE};">Variables:</b> 41 clinical features<br>
    <b style="color:{WHITE};">Password:</b> <span style="font-family:monospace;color:{RED};">SEPSIS2026</span>
    </div>""", unsafe_allow_html=True)
    if st.button("🚪 Logout"):
        st.session_state.auth = False
        st.rerun()

# ── APPLY FILTERS ─────────────────────────────────────────
dff = df[df['hospital'].isin(hospital_sel)].copy()
dff = dff[dff['age'].between(*age_range)]
dff = dff[dff['gender_label'].isin(gender_filter)]
dff = dff[dff['icu_los_hours'] <= icu_max]
dff = dff[dff['icu_unit'].isin(unit_filter)]
if sepsis_filter == "Sepsis Only": dff = dff[dff['sepsis']==1]
elif sepsis_filter == "Non-Sepsis Only": dff = dff[dff['sepsis']==0]

# ── HEADER ────────────────────────────────────────────────
st.markdown(f"""<div class="title-band">
  <div style="display:flex;align-items:center;gap:18px;">
    <div style="font-size:3rem;">🫀</div>
    <div>
      <div class="title-main">SepsisWatch ICU Analytics</div>
      <div class="title-sub">Early Sepsis Detection & ICU Outcome Intelligence · PhysioNet 2019 Challenge · Real Patient Data</div>
      <span class="badge">BIDMC BOSTON · EMORY ATLANTA · 40,336 REAL ICU PATIENTS · 41 VARIABLES</span>
    </div>
    <div style="margin-left:auto;text-align:right;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:{MUTED};">
      Filtered: {len(dff):,} patients<br>
      Sepsis cases: {dff['sepsis'].sum():,}<br>
      Sepsis rate: {dff['sepsis'].mean():.1%}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
kpi(k1, f"{len(dff):,}", "Total ICU Patients", "blue")
kpi(k2, f"{dff['sepsis'].sum():,}", "Confirmed Sepsis", "red", alert=True)
kpi(k3, f"{dff['sepsis'].mean():.1%}", "Sepsis Rate", "orange")
kpi(k4, f"{dff['icu_los_hours'].mean():.1f}h", "Mean ICU Stay", "teal")
kpi(k5, f"{dff['age'].mean():.1f} yrs", "Mean Patient Age", "blue")
st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊 Demographics & Distribution",
    "🗺️ Geographic & Hospital Analysis",
    "🩺 Vital Signs",
    "🧪 Lab Biomarkers",
    "⚠️ Risk Stratification",
    "🤖 Predictive Model"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="section-hdr">Patient Demographics — Distribution by Age, Gender & Sepsis Status</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        counts = dff['sepsis'].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Non-Sepsis","Sepsis"], values=[counts.get(0,0),counts.get(1,0)],
            hole=0.58, marker_colors=[TEAL,RED],
            textinfo='percent+label', textfont=dict(color=WHITE,size=12)))
        fig.add_annotation(text=f"<b>{len(dff):,}</b><br>patients",
                           x=0.5,y=0.5,font=dict(size=14,color=WHITE),showarrow=False)
        chart(fig,"Overall Sepsis vs Non-Sepsis Distribution",340)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig = go.Figure()
        for label,color,val in [("Non-Sepsis",TEAL,0),("Sepsis",RED,1)]:
            fig.add_trace(go.Histogram(x=dff[dff['sepsis']==val]['age'],
                name=label,marker_color=color,opacity=0.75,nbinsx=20))
        chart(fig,"Age Distribution by Sepsis Status",340)
        fig.update_layout(barmode='overlay')
        fig.update_xaxes(title="Age (years)")
        fig.update_yaxes(title="Number of Patients")
        st.plotly_chart(fig,use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        # Gender distribution with sepsis breakdown
        gender_data = dff.groupby(['gender_label','sepsis'], observed=True).size().reset_index(name='count')
        gender_data['status'] = gender_data['sepsis'].map({0:'Non-Sepsis',1:'Sepsis'})
        # Also compute rates
        gender_rate = dff.groupby('gender_label', observed=True)['sepsis'].agg(['mean','count','sum']).reset_index()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        for status, color in [('Non-Sepsis',TEAL),('Sepsis',RED)]:
            sub = gender_data[gender_data['status']==status]
            fig.add_trace(go.Bar(x=sub['gender_label'],y=sub['count'],name=status,
                marker_color=color,hovertemplate=f"{status}: %{{y:,}}<extra></extra>"),secondary_y=False)
        fig.add_trace(go.Scatter(x=gender_rate['gender_label'],y=gender_rate['mean'],
            mode='markers+lines',name='Sepsis Rate',
            marker=dict(size=12,color=ORANGE),line=dict(color=ORANGE,width=2)),secondary_y=True)
        chart(fig,"Gender Distribution & Sepsis Rate",340)
        fig.update_layout(barmode='stack')
        fig.update_yaxes(title="Patient Count",secondary_y=False,gridcolor=BORDER)
        fig.update_yaxes(title="Sepsis Rate",tickformat='.0%',secondary_y=True,
                         gridcolor='rgba(0,0,0,0)',color=WHITE)
        st.plotly_chart(fig,use_container_width=True)

    with c4:
        # Age group sepsis rate
        ag = dff.groupby('age_group', observed=True)['sepsis'].agg(['mean','sum','count']).reset_index()
        ag.columns = ['age_group','sepsis_rate','sepsis_count','total']
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=ag['age_group'].astype(str),y=ag['total'],
            name='Total Patients',marker_color=BORDER,opacity=0.8),secondary_y=False)
        fig.add_trace(go.Bar(x=ag['age_group'].astype(str),y=ag['sepsis_count'],
            name='Sepsis Cases',marker_color=RED),secondary_y=False)
        fig.add_trace(go.Scatter(x=ag['age_group'].astype(str),y=ag['sepsis_rate'],
            mode='lines+markers',name='Sepsis Rate',
            line=dict(color=ORANGE,width=2.5),marker=dict(size=9)),secondary_y=True)
        chart(fig,"Sepsis Cases & Rate by Age Group",340)
        fig.update_layout(barmode='overlay')
        fig.update_yaxes(title="Patient Count",secondary_y=False,gridcolor=BORDER)
        fig.update_yaxes(title="Sepsis Rate",tickformat='.0%',secondary_y=True,
                         gridcolor='rgba(0,0,0,0)',color=WHITE)
        st.plotly_chart(fig,use_container_width=True)

    c5,c6 = st.columns(2)
    with c5:
        # ICU LOS by gender and sepsis
        fig = go.Figure()
        for label,color,val in [("Non-Sepsis",TEAL,0),("Sepsis",RED,1)]:
            fig.add_trace(go.Box(y=dff[dff['sepsis']==val]['icu_los_hours'].clip(0,200),
                name=label,marker_color=color,boxmean=True,line_color=color))
        chart(fig,"ICU Length of Stay — Sepsis vs Non-Sepsis",340)
        fig.update_yaxes(title="ICU Hours")
        st.plotly_chart(fig,use_container_width=True)
    with c6:
        # ICU unit type breakdown
        unit_data = dff.groupby(['icu_unit','sepsis'], observed=True).size().reset_index(name='count')
        unit_data['status'] = unit_data['sepsis'].map({0:'Non-Sepsis',1:'Sepsis'})
        fig = px.bar(unit_data,x='icu_unit',y='count',color='status',barmode='group',
            color_discrete_map={'Non-Sepsis':TEAL,'Sepsis':RED},
            labels={'icu_unit':'ICU Unit Type','count':'Patients','status':'Status'})
        chart(fig,"Patient Distribution by ICU Unit Type",340)
        fig.update_xaxes(title="")
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — GEOGRAPHIC & HOSPITAL ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-hdr">Geographic Distribution — Hospital Comparison: BIDMC Boston vs Emory Atlanta</div>', unsafe_allow_html=True)

    c1,c2 = st.columns([1,1])
    with c1:
        # US hospital map
        hospital_map = pd.DataFrame({
            'hospital': ['BIDMC - Boston, MA', 'Emory - Atlanta, GA'],
            'lat': [42.3389, 33.7949],
            'lon': [-71.1067, -84.3240],
            'city': ['Boston, Massachusetts', 'Atlanta, Georgia'],
            'patients': [
                len(dff[dff['hospital']=='BIDMC - Boston, MA']),
                len(dff[dff['hospital']=='Emory - Atlanta, GA'])
            ],
            'sepsis_cases': [
                dff[dff['hospital']=='BIDMC - Boston, MA']['sepsis'].sum(),
                dff[dff['hospital']=='Emory - Atlanta, GA']['sepsis'].sum()
            ],
            'sepsis_rate': [
                dff[dff['hospital']=='BIDMC - Boston, MA']['sepsis'].mean()*100,
                dff[dff['hospital']=='Emory - Atlanta, GA']['sepsis'].mean()*100
            ],
        })
        hospital_map['label'] = hospital_map.apply(
            lambda r: f"{r['hospital']}<br>Patients: {r['patients']:,}<br>Sepsis: {r['sepsis_cases']:,} ({r['sepsis_rate']:.1f}%)", axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=hospital_map['lon'],
            lat=hospital_map['lat'],
            mode='markers+text',
            marker=dict(
                size=[p/300 for p in hospital_map['patients']],
                color=[RED, TEAL],
                opacity=0.85,
                line=dict(width=2, color=WHITE)
            ),
            text=['BIDMC Boston', 'Emory Atlanta'],
            textposition=['top center', 'bottom center'],
            textfont=dict(color=WHITE, size=13, family='IBM Plex Sans'),
            hovertext=hospital_map['label'],
            hoverinfo='text',
            showlegend=False
        ))
        fig.update_layout(
            geo=dict(
                scope='usa',
                showland=True,
                landcolor='#1A2540',
                showocean=True,
                oceancolor='#0A0F1E',
                showlakes=True,
                lakecolor='#0A1520',
                showcoastlines=True,
                coastlinecolor=BORDER,
                showsubunits=True,
                subunitcolor=BORDER,
                showcountries=True,
                countrycolor=BORDER,
                bgcolor='rgba(0,0,0,0)',
                projection=dict(type='albers usa'),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=WHITE, family='IBM Plex Sans'),
            title=dict(text='ICU Data Collection Sites - United States',
                       font=dict(size=13, color=WHITE)),
            margin=dict(t=40, b=10, l=0, r=0),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Hospital KPI comparison
        h_stats = dff.groupby('hospital', observed=True).agg(
            patients=('patient_id','count'),
            sepsis_cases=('sepsis','sum'),
            sepsis_rate=('sepsis','mean'),
            mean_age=('age','mean'),
            mean_los=('icu_los_hours','mean'),
            pct_male=('gender','mean')
        ).reset_index()

        for _, row in h_stats.iterrows():
            color = RED if 'BIDMC' in row['hospital'] else TEAL
            short = 'BIDMC Boston' if 'BIDMC' in row['hospital'] else 'Emory Atlanta'
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {color};border-radius:8px;
                        padding:16px 20px;margin-bottom:12px;">
              <div style="font-size:0.75rem;color:{color};font-weight:700;text-transform:uppercase;
                          letter-spacing:0.1em;margin-bottom:8px;">🏥 {short}</div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">
                <div style="text-align:center;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{color};">{row['patients']:,}</div>
                  <div style="font-size:0.68rem;color:{MUTED};">PATIENTS</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{RED};">{row['sepsis_rate']:.1%}</div>
                  <div style="font-size:0.68rem;color:{MUTED};">SEPSIS RATE</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{ORANGE};">{row['mean_los']:.1f}h</div>
                  <div style="font-size:0.68rem;color:{MUTED};">MEAN ICU LOS</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:700;color:{WHITE};">{row['mean_age']:.1f} yrs</div>
                  <div style="font-size:0.68rem;color:{MUTED};">MEAN AGE</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:700;color:{WHITE};">{row['pct_male']:.0%}</div>
                  <div style="font-size:0.68rem;color:{MUTED};">% MALE</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:700;color:{WHITE};">{int(row['sepsis_cases']):,}</div>
                  <div style="font-size:0.68rem;color:{MUTED};">SEPSIS CASES</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        # Hospital sepsis rate comparison by age group
        hosp_age = dff.groupby(['hospital','age_group'], observed=True)['sepsis'].mean().reset_index()
        hosp_age['hospital_short'] = hosp_age['hospital'].str.split(' - ').str[0]
        fig = px.line(hosp_age, x='age_group', y='sepsis',
                      color='hospital_short',
                      color_discrete_map={'BIDMC':RED,'Emory':TEAL},
                      markers=True,
                      labels={'age_group':'Age Group','sepsis':'Sepsis Rate','hospital_short':'Hospital'})
        chart(fig,"Sepsis Rate by Age Group — Hospital Comparison",360)
        fig.update_yaxes(title="Sepsis Rate",tickformat='.1%')
        fig.update_xaxes(title="Age Group")
        st.plotly_chart(fig,use_container_width=True)

    with c4:
        # ICU unit type distribution by hospital
        unit_hosp = dff.groupby(['hospital','icu_unit'], observed=True).size().reset_index(name='count')
        unit_hosp['hospital_short'] = unit_hosp['hospital'].str.split(' - ').str[0]
        fig = px.bar(unit_hosp, x='hospital_short', y='count', color='icu_unit',
                     color_discrete_map={'Medical ICU (MICU)':RED,
                                         'Surgical ICU (SICU)':TEAL,'Unspecified':MUTED},
                     barmode='stack',
                     labels={'hospital_short':'Hospital','count':'Patients','icu_unit':'ICU Unit'})
        chart(fig,"ICU Unit Distribution by Hospital",360)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Number of Patients")
        st.plotly_chart(fig,use_container_width=True)

    c5,c6 = st.columns(2)
    with c5:
        # Gender distribution by hospital
        gen_hosp = dff.groupby(['hospital','gender_label'], observed=True).size().reset_index(name='count')
        gen_hosp['hospital_short'] = gen_hosp['hospital'].str.split(' - ').str[0]
        fig = px.bar(gen_hosp, x='hospital_short', y='count', color='gender_label',
                     color_discrete_map={'Male':BLUE,'Female':ORANGE},
                     barmode='group',
                     labels={'hospital_short':'Hospital','count':'Patients','gender_label':'Gender'})
        chart(fig,"Gender Distribution by Hospital",340)
        fig.update_xaxes(title="")
        st.plotly_chart(fig,use_container_width=True)

    with c6:
        # Mean vitals by hospital radar-style bar
        hosp_vitals = dff.groupby('hospital', observed=True).agg(
            hr=('hr_mean','mean'), sbp=('sbp_mean','mean'),
            o2=('o2sat_mean','mean'), resp=('resp_mean','mean')
        ).reset_index()
        hosp_vitals['hospital_short'] = hosp_vitals['hospital'].str.split(' - ').str[0]
        fig = go.Figure()
        vitals_labels = ['Mean HR (bpm)','Mean SBP (mmHg)','Mean O2Sat (%)','Mean Resp (br/min)']
        for _, row in hosp_vitals.iterrows():
            color = RED if 'BIDMC' in row['hospital'] else TEAL
            vals = [row['hr'], row['sbp'], row['o2'], row['resp']]
            fig.add_trace(go.Bar(name=row['hospital_short'],x=vitals_labels,y=vals,
                marker_color=color,
                hovertemplate='%{x}: %{y:.1f}<extra></extra>'))
        chart(fig,"Mean Vital Signs Comparison by Hospital",340)
        fig.update_layout(barmode='group')
        fig.update_yaxes(title="Mean Value")
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — VITAL SIGNS
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-hdr">Vital Signs Analysis — Trends & Patterns by Sepsis Status</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        vitals = {'Heart Rate (bpm)':'hr_mean','O2 Saturation (%)':'o2sat_mean',
                  'Temperature (°C)':'temp_mean','Respiratory Rate (br/min)':'resp_mean',
                  'Systolic BP (mmHg)':'sbp_mean','Mean Art. Pressure':'map_mean'}
        sv = st.selectbox("Select Vital Sign", list(vitals.keys()))
        col = vitals[sv]
        fig = go.Figure()
        for label,color,val in [("Non-Sepsis",TEAL,0),("Sepsis",RED,1)]:
            fig.add_trace(go.Violin(y=dff[dff['sepsis']==val][col].dropna(),
                name=label,box_visible=True,meanline_visible=True,
                line_color=color,fillcolor=color,opacity=0.55,points='outliers'))
        chart(fig,f"{sv} — Distribution by Sepsis Status",400)
        fig.update_yaxes(title=sv)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        vcols = ['hr_mean','o2sat_mean','sbp_mean','map_mean','resp_mean']
        vlabels = ['HR','O2Sat','SBP','MAP','Resp']
        sm = dff[dff['sepsis']==1][vcols].mean()
        nm = dff[dff['sepsis']==0][vcols].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Non-Sepsis',x=vlabels,y=[nm[c] for c in vcols],marker_color=TEAL))
        fig.add_trace(go.Bar(name='Sepsis',x=vlabels,y=[sm[c] for c in vcols],marker_color=RED))
        chart(fig,"Mean Vital Signs Comparison — Sepsis vs Non-Sepsis",400)
        fig.update_layout(barmode='group')
        fig.update_yaxes(title="Mean Value")
        st.plotly_chart(fig,use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        samp = dff.sample(min(2000,len(dff)),random_state=42)
        samp['status'] = samp['sepsis'].map({0:'Non-Sepsis',1:'Sepsis'})
        fig = px.scatter(samp.dropna(subset=['hr_mean','sbp_mean']),
                         x='hr_mean',y='sbp_mean',color='status',
                         color_discrete_map={'Non-Sepsis':TEAL,'Sepsis':RED},
                         opacity=0.5,
                         labels={'hr_mean':'Mean HR (bpm)','sbp_mean':'Mean SBP (mmHg)'},
                         hover_data=['age','gender_label','hospital'])
        fig.add_hline(y=90,line_dash='dot',line_color=ORANGE,opacity=0.7,
                      annotation_text="SBP < 90 — Shock Threshold",annotation_font_color=ORANGE)
        fig.add_vline(x=100,line_dash='dot',line_color=ORANGE,opacity=0.7,
                      annotation_text="HR > 100",annotation_font_color=ORANGE)
        chart(fig,"Heart Rate vs Systolic BP — Shock Detection Zone",380)
        st.plotly_chart(fig,use_container_width=True)
    with c4:
        worst = dff.groupby('age_group', observed=True).agg(
            max_hr=('hr_max','mean'),min_sbp=('sbp_min','mean'),
            max_resp=('resp_max','mean'),min_o2=('o2sat_min','mean')).round(1)
        fig = go.Figure(go.Heatmap(
            z=worst.values,x=['Max HR','Min SBP','Max Resp','Min O2Sat'],
            y=worst.index.astype(str),
            colorscale=[[0,TEAL],[0.5,ORANGE],[1,RED]],
            text=worst.values.round(1),texttemplate="%{text}",
            colorbar=dict(title="Value",tickfont=dict(color=WHITE))))
        chart(fig,"Worst-Case Vital Signs Heatmap by Age Group",380)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 4 — LAB BIOMARKERS
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-hdr">Laboratory Biomarkers — Clinical Thresholds & Organ Function</div>', unsafe_allow_html=True)
    sep = dff[dff['sepsis']==1]; non = dff[dff['sepsis']==0]
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Lactate — Sepsis",f"{sep['lactate_mean'].mean():.2f} mmol/L",
              delta=f"+{sep['lactate_mean'].mean()-non['lactate_mean'].mean():.2f} vs non-sepsis")
    m2.metric("Creatinine — Sepsis",f"{sep['creatinine_mean'].mean():.2f} mg/dL",
              delta=f"+{sep['creatinine_mean'].mean()-non['creatinine_mean'].mean():.2f} vs non-sepsis")
    m3.metric("WBC — Sepsis",f"{sep['wbc_mean'].mean():.1f} K/µL",
              delta=f"+{sep['wbc_mean'].mean()-non['wbc_mean'].mean():.1f} vs non-sepsis")
    m4.metric("Platelets — Sepsis",f"{sep['platelets_mean'].mean():.0f} K/µL",
              delta=f"{sep['platelets_mean'].mean()-non['platelets_mean'].mean():.0f} vs non-sepsis")
    st.markdown("<br>",unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        labs = {'Lactate':'lactate_mean','Creatinine':'creatinine_mean',
                'WBC':'wbc_mean','Glucose':'glucose_mean',
                'BUN':'bun_mean','Hemoglobin':'hgb_mean'}
        sl = st.selectbox("Select Lab Biomarker",list(labs.keys()))
        col = labs[sl]
        fig = go.Figure()
        for label,color,val in [("Non-Sepsis",TEAL,0),("Sepsis",RED,1)]:
            fig.add_trace(go.Box(y=dff[dff['sepsis']==val][col].dropna(),
                name=label,marker_color=color,boxmean=True,line_color=color,notched=True))
        thresh = {'Lactate':2.0,'Creatinine':1.2,'WBC':11.0,'Glucose':180,'BUN':20,'Hemoglobin':12}
        if sl in thresh:
            fig.add_hline(y=thresh[sl],line_dash='dot',line_color=ORANGE,
                annotation_text=f"Clinical threshold: {thresh[sl]}",annotation_font_color=ORANGE)
        chart(fig,f"{sl} — Sepsis vs Non-Sepsis (with Clinical Threshold)",400)
        fig.update_yaxes(title=sl)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        lcols = ['lactate_mean','creatinine_mean','wbc_mean','glucose_mean',
                 'bun_mean','hgb_mean','platelets_mean','potassium_mean']
        llabels = ['Lactate','Creatinine','WBC','Glucose','BUN','Hgb','Platelets','Potassium']
        corr = dff[lcols].dropna().corr()
        fig = go.Figure(go.Heatmap(z=corr.values,x=llabels,y=llabels,
            colorscale=[[0,BLUE],[0.5,CARD],[1,RED]],zmid=0,
            text=corr.values.round(2),texttemplate="%{text}",
            colorbar=dict(title="r",tickfont=dict(color=WHITE))))
        chart(fig,"Lab Value Correlation Matrix",400)
        st.plotly_chart(fig,use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        s2 = dff.dropna(subset=['lactate_mean','creatinine_mean']).sample(min(1500,len(dff)),random_state=7)
        s2['status'] = s2['sepsis'].map({0:'Non-Sepsis',1:'Sepsis'})
        fig = px.scatter(s2,x='lactate_mean',y='creatinine_mean',color='status',
            color_discrete_map={'Non-Sepsis':TEAL,'Sepsis':RED},opacity=0.55,
            labels={'lactate_mean':'Lactate (mmol/L)','creatinine_mean':'Creatinine (mg/dL)'},
            hover_data=['age','gender_label'])
        fig.add_hline(y=1.2,line_dash='dot',line_color=ORANGE,opacity=0.6)
        fig.add_vline(x=2.0,line_dash='dot',line_color=ORANGE,opacity=0.6,
                      annotation_text="High Risk Zone",annotation_font_color=ORANGE)
        chart(fig,"Lactate vs Creatinine — Organ Failure Detection",380)
        st.plotly_chart(fig,use_container_width=True)
    with c4:
        sc = ['sirs_temp','sirs_hr','sirs_resp','sirs_wbc']
        sl2 = ['Temp Abnormal','HR > 90','Resp > 20','WBC Abnormal']
        sr = dff[dff['sepsis']==1][sc].mean()
        nr = dff[dff['sepsis']==0][sc].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Non-Sepsis',x=sl2,y=nr,marker_color=TEAL))
        fig.add_trace(go.Bar(name='Sepsis',x=sl2,y=sr,marker_color=RED))
        chart(fig,"SIRS Criteria Prevalence — Sepsis vs Non-Sepsis",380)
        fig.update_layout(barmode='group')
        fig.update_yaxes(title="Proportion of Patients",tickformat='.0%')
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 5 — RISK STRATIFICATION
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="section-hdr">Sepsis Risk Stratification — Patterns, Trends & Clinical Risk Factors</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        ss = dff.groupby('sirs_score', observed=True)['sepsis'].agg(['mean','count']).reset_index()
        ss.columns = ['sirs_score','sepsis_rate','count']
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=ss['sirs_score'],y=ss['count'],name='Patient Count',
            marker_color=BORDER,hovertemplate="SIRS %{x}: %{y:,} patients<extra></extra>"),secondary_y=False)
        fig.add_trace(go.Scatter(x=ss['sirs_score'],y=ss['sepsis_rate'],mode='lines+markers',
            name='Sepsis Rate',line=dict(color=RED,width=2.5),marker=dict(size=10)),secondary_y=True)
        chart(fig,"SIRS Score vs Sepsis Rate — Clinical Threshold Analysis",370)
        fig.update_yaxes(title="Patient Count",secondary_y=False,gridcolor=BORDER)
        fig.update_yaxes(title="Sepsis Rate",tickformat='.0%',secondary_y=True,
                         gridcolor='rgba(0,0,0,0)',color=WHITE)
        fig.update_xaxes(title="SIRS Score (0-4)")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        od = dff[dff['sepsis']==1]['sepsis_onset_hour'].dropna()
        fig = go.Figure(go.Histogram(x=od,nbinsx=30,marker_color=RED,opacity=0.8))
        fig.add_vline(x=od.median(),line_dash='dash',line_color=ORANGE,
                      annotation_text=f"Median onset: {od.median():.0f}h",annotation_font_color=ORANGE)
        fig.add_vline(x=24,line_dash='dot',line_color=TEAL,opacity=0.7,
                      annotation_text="24h mark",annotation_font_color=TEAL)
        chart(fig,"Sepsis Onset Hour Distribution — ICU Admission Timeline",370)
        fig.update_xaxes(title="Hour After ICU Admission")
        fig.update_yaxes(title="Number of Sepsis Cases")
        st.plotly_chart(fig,use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        ig = dff.groupby('icu_stay_group', observed=True)['sepsis'].agg(['mean','sum','count']).reset_index()
        ig.columns = ['icu_stay_group','sepsis_rate','sepsis_count','total']
        fig = px.bar(ig,x='icu_stay_group',y='sepsis_rate',
                     color='sepsis_rate',
                     color_continuous_scale=[[0,TEAL],[0.5,ORANGE],[1,RED]],
                     text=[f"{v:.1%}" for v in ig['sepsis_rate']],
                     labels={'icu_stay_group':'ICU Stay Duration','sepsis_rate':'Sepsis Rate'})
        fig.update_traces(textposition='outside',textfont_color=WHITE)
        chart(fig,"Sepsis Rate by ICU Stay Duration",370)
        fig.update_yaxes(title="Sepsis Rate",tickformat='.0%')
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    with c4:
        rfd = {
            'High HR (>100 bpm)': dff[dff['hr_max']>100]['sepsis'].mean(),
            'Low SBP (<90 mmHg)': dff[dff['sbp_min']<90]['sepsis'].mean(),
            'High Lactate (>2)': dff[dff['lactate_max']>2]['sepsis'].mean(),
            'High Creatinine (>1.2)': dff[dff['creatinine_max']>1.2]['sepsis'].mean(),
            'Low O2Sat (<90%)': dff[dff['o2sat_min']<90]['sepsis'].mean(),
            'SIRS Score >= 3': dff[dff['sirs_score']>=3]['sepsis'].mean(),
            'Age >= 65 years': dff[dff['age']>=65]['sepsis'].mean(),
            'Medical ICU': dff[dff['icu_unit']=='Medical ICU (MICU)']['sepsis'].mean(),
        }
        rdf = pd.DataFrame(list(rfd.items()),columns=['factor','rate']).dropna().sort_values('rate')
        colors = [RED if v>0.15 else ORANGE if v>0.08 else TEAL for v in rdf['rate']]
        fig = go.Figure(go.Bar(y=rdf['factor'],x=rdf['rate'],orientation='h',
            marker_color=colors,
            text=[f"{v:.1%}" for v in rdf['rate']],
            textposition='outside',textfont=dict(color=WHITE)))
        chart(fig,"Sepsis Rate by Clinical Risk Factor",370)
        fig.update_xaxes(title="Sepsis Rate",tickformat='.0%')
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 6 — ML MODEL
# ═══════════════════════════════════════════════════════════
with tab6:
    st.markdown(f'<div class="section-hdr">Predictive Analytics — Early Sepsis Warning System (Bonus)</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.82rem;color:{MUTED};margin-bottom:14px;">Three ML classifiers trained on 14 clinical features available within the first hours of ICU admission. Predicts sepsis onset before clinical deterioration occurs.</div>', unsafe_allow_html=True)

    @st.cache_data
    def train_models(data):
        features = ['age','gender','hr_mean','hr_max','o2sat_mean','o2sat_min',
                    'temp_mean','sbp_mean','sbp_min','map_mean','resp_mean','resp_max',
                    'sirs_score','icu_los_hours']
        X = data[features]; y = data['sepsis']
        imp = SimpleImputer(strategy='median'); Xi = imp.fit_transform(X)
        Xtr,Xte,ytr,yte = train_test_split(Xi,y,test_size=0.25,random_state=42,stratify=y)
        sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr); Xtes = sc.transform(Xte)
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=150,max_depth=8,
                                                     random_state=42,n_jobs=-1,class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100,max_depth=4,random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced')
        }
        res = {}
        for name,model in models.items():
            Xtr2 = Xtrs if name=='Logistic Regression' else Xtr
            Xte2 = Xtes if name=='Logistic Regression' else Xte
            model.fit(Xtr2,ytr); yp = model.predict_proba(Xte2)[:,1]; ypred = model.predict(Xte2)
            cm = confusion_matrix(yte,ypred); tn,fp,fn,tp = cm.ravel()
            fpr,tpr,_ = roc_curve(yte,yp)
            fi = pd.DataFrame({'feature':features,'importance':model.feature_importances_}).sort_values('importance',ascending=False) if hasattr(model,'feature_importances_') else None
            res[name] = {'auc':roc_auc_score(yte,yp),'sensitivity':tp/(tp+fn),
                         'specificity':tn/(tn+fp),'accuracy':(tp+tn)/(tp+tn+fp+fn),
                         'cm':cm,'fpr':fpr,'tpr':tpr,'fi':fi,'model':model,'imp':imp,'features':features}
        return res

    with st.spinner("Training 3 ML models on 40,336 real ICU patients…"):
        results = train_models(df)

    # Model comparison cards
    mc = st.columns(3)
    colors_m = [RED,ORANGE,TEAL]
    for i,(name,res) in enumerate(results.items()):
        color_key = ['red','orange','teal'][i]
        mc[i].markdown(f"""<div class="kpi-card kpi-{color_key}">
          <div style="font-size:0.72rem;color:{MUTED};margin-bottom:5px;">{name}</div>
          <div class="kpi-value" style="color:{colors_m[i]};">AUC {res['auc']:.3f}</div>
          <div style="font-size:0.73rem;color:{MUTED};margin-top:5px;">
            Sensitivity: {res['sensitivity']:.1%} &nbsp;|&nbsp;
            Specificity: {res['specificity']:.1%} &nbsp;|&nbsp;
            Accuracy: {res['accuracy']:.1%}
          </div></div>""", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for (name,res),color in zip(results.items(),colors_m):
            fig.add_trace(go.Scatter(x=res['fpr'],y=res['tpr'],mode='lines',
                name=f"{name} (AUC={res['auc']:.3f})",line=dict(color=color,width=2.5),
                fill='tozeroy' if name=='Random Forest' else None,
                fillcolor='rgba(232,72,85,0.07)'))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random Chance',
            line=dict(dash='dot',color=MUTED,width=1)))
        chart(fig,"ROC Curves — All 3 Models Compared",400)
        fig.update_xaxes(title="False Positive Rate")
        fig.update_yaxes(title="True Positive Rate")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fi = results['Random Forest']['fi']
        fig = go.Figure(go.Bar(y=fi['feature'],x=fi['importance'],orientation='h',
            marker=dict(color=fi['importance'],colorscale=[[0,TEAL],[0.5,ORANGE],[1,RED]]),
            text=[f"{v:.3f}" for v in fi['importance']],
            textposition='outside',textfont=dict(color=WHITE)))
        chart(fig,"Feature Importance — Random Forest",400)
        fig.update_xaxes(title="Importance Score")
        st.plotly_chart(fig,use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        best = max(results,key=lambda k:results[k]['auc'])
        cm = results[best]['cm']
        fig = go.Figure(go.Heatmap(z=cm,
            x=['Predicted: No Sepsis','Predicted: Sepsis'],
            y=['Actual: No Sepsis','Actual: Sepsis'],
            text=cm,texttemplate="%{text}",
            colorscale=[[0,DARK],[0.5,ORANGE],[1,RED]],showscale=False))
        chart(fig,f"Confusion Matrix — {best} (Best Model)",340)
        st.plotly_chart(fig,use_container_width=True)
    with c4:
        st.markdown(f'<div class="eyebrow" style="margin-bottom:10px;">Live Patient Sepsis Risk Predictor</div>', unsafe_allow_html=True)
        pa = st.slider("Patient Age",18,100,65)
        pg = st.selectbox("Gender",["Male","Female"])
        ph = st.number_input("Mean Heart Rate (bpm)",40.0,200.0,88.0,step=1.0)
        ps = st.number_input("Mean Systolic BP (mmHg)",50.0,200.0,115.0,step=1.0)
        po = st.number_input("Mean O2 Saturation (%)",70.0,100.0,96.0,step=0.5)
        pr = st.number_input("Mean Respiratory Rate (br/min)",8.0,50.0,18.0,step=1.0)
        psi = st.selectbox("SIRS Score (0–4)",[0,1,2,3,4])
        pl = st.number_input("ICU Hours Elapsed",1.0,336.0,24.0,step=1.0)
        if st.button("🔍 Predict Sepsis Risk Now",use_container_width=True):
            res = results['Random Forest']
            pat = res['imp'].transform([[pa,1 if pg=="Male" else 0,
                                         ph,ph*1.1,po,po*0.95,37.0,
                                         ps,ps*0.85,ps*0.75,pr,pr*1.2,psi,pl]])
            prob = res['model'].predict_proba(pat)[0][1]
            rc = RED if prob>0.5 else ORANGE if prob>0.25 else TEAL
            rl = "⚠️ HIGH RISK — Immediate Review Recommended" if prob>0.5 else "⚡ MODERATE RISK — Close Monitoring" if prob>0.25 else "✅ LOW RISK — Routine Monitoring"
            st.markdown(f"""<div style="background:{CARD};border:2px solid {rc};border-radius:10px;
                padding:22px;text-align:center;margin-top:10px;">
              <div style="font-size:0.75rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.1em;">Predicted Sepsis Probability</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:2.6rem;
                          font-weight:700;color:{rc};margin:8px 0;">{prob:.1%}</div>
              <div style="font-size:0.95rem;font-weight:600;color:{rc};">{rl}</div>
            </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────
st.markdown(f"""
<div style="border-top:1px solid {BORDER};margin-top:28px;padding-top:14px;
     display:flex;justify-content:space-between;align-items:center;font-size:0.7rem;color:{MUTED};">
  <div>SepsisWatch Analytics · MSBA382 Healthcare Analytics</div>
  <div>PhysioNet 2019 Challenge · BIDMC Boston · Emory Atlanta</div>
  <div style="font-family:'IBM Plex Mono',monospace;color:{RED};">{len(df):,} patients · 41 variables · 6 tabs · Real ICU Data</div>
</div>""", unsafe_allow_html=True)
