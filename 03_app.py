import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import datetime
from io import BytesIO
import base64

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="CarbonBloom Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THEMED UI (CSS + ANIMATIONS) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Public+Sans:wght@300;400;600;700&display=swap');

        :root{
            --bg-start: #0f2027;
            --bg-mid: #2c5364;
            --bg-end: #203a43;
            --accent: #38ef7d;
            --muted: #9aa4a6;
            --card: rgba(255,255,255,0.06);
            --glass: rgba(255,255,255,0.06);
            --radius: 14px;
        }

        /* App background: animated gradient with subtle noise */
        .stApp {
            min-height: 100vh;
            background: linear-gradient(135deg,var(--bg-start), var(--bg-mid) 40%, var(--bg-end));
            background-size: 400% 400%;
            animation: gradientShift 18s ease infinite;
            font-family: 'Public Sans', Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
            color: #eef6f5;
            padding: 28px 36px;
        }

        @keyframes gradientShift {
            0%{background-position:0% 50%}
            50%{background-position:100% 50%}
            100%{background-position:0% 50%}
        }

        /* Subtle glass card */
        .card {
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border-radius: var(--radius);
            padding: 22px;
            box-shadow: 0 8px 30px rgba(2,6,23,0.45);
            border: 1px solid rgba(255,255,255,0.04);
        }

        /* Header / hero */
        .hero {
            display:flex; align-items:center; gap:22px; margin-bottom:18px;
        }
        .brand {
            display:flex; align-items:center; gap:14px;
        }
        .logo-badge{
            width:64px; height:64px; border-radius:14px; display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            box-shadow: 0 6px 18px rgba(0,0,0,0.35) inset;
            border: 1px solid rgba(255,255,255,0.06);
            font-size:28px;
        }
        .title {font-size:28px; font-weight:700; margin:0; color:#bfffdc}
        .subtitle {color:#bfffdc; margin:0; font-size:14px}

        /* Metrics */
        .metrics-row{display:flex; gap:18px; margin-top:10px}
        .metric {flex:1; padding:14px; border-radius:12px; background: rgba(255,255,255,0.02);}
        .metric h4{margin:0; font-size:14px; color:#dff6ee}
        .metric p{margin:6px 0 0 0; font-size:20px; font-weight:700}
        .metric-value{font-size:28px; font-weight:800; margin-top:6px; color:#fff}
        .metric-sub{color:var(--muted); font-size:12px; margin-top:6px}
        .chip{display:inline-block; padding:6px 10px; background:rgba(255,255,255,0.03); border-radius:999px; font-size:12px; color:#dff6ee}
        .top-nav{display:flex; gap:18px; align-items:center; margin-bottom:10px}

        /* Upload box styling */
        div[data-testid="stFileUploader"]{
            border-radius:12px; padding:18px; border: 1px dashed rgba(255,77,79,0.20);
            background: linear-gradient(180deg, rgba(255,77,79,0.02), rgba(0,0,0,0.02));
            color: #ff4d4f; text-align:center; font-weight:700;
        }
        /* Style the internal browse button and ensure it shows text */
        div[data-testid="stFileUploader"] button {
            background: linear-gradient(90deg,#ff4d4f,#ff7a7a);
            color: #ffffff; border: none; padding: 8px 14px; border-radius: 8px; font-weight:700;
        }
        div[data-testid="stFileUploader"] button:empty::after { content: "Browse"; color: #ffffff; }
        /* Make the drag/drop and limit text red */
        div[data-testid="stFileUploader"] .stMarkdown p, div[data-testid="stFileUploader"] span, div[data-testid="stFileUploader"] label { color: #ff4d4f !important }
        /* Force uploader filename and other uploader text to red but exclude buttons */
        div[data-testid="stFileUploader"] :not(button) { color: #ff4d4f !important }

        /* Sidebar tweaks */
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(0,0,0,0.18), rgba(255,255,255,0.02)); border-radius: 12px; padding: 12px; color: #eaf7f3 }

        /* Buttons */
        .stButton > button { border-radius: 999px; padding: 10px 20px; background: linear-gradient(90deg,#2dd4bf,#06b6d4); color: #042c2e; font-weight:700; }
        /* Style the specific download buttons to match card background */
        button[aria-label="Download Original Image"], button[aria-label="Download Detection Overlay"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            color: #eafff4; border: 1px solid rgba(255,255,255,0.04); padding: 8px 14px; border-radius: 10px; font-weight:700;
        }
        /* Custom download link styling to match site background with light-blue text */
        .download-link { display:inline-block; padding:10px 18px; border-radius:12px; text-decoration:none; background:transparent; color:#7fd3ff; border:1px solid rgba(255,255,255,0.03); font-weight:700 }

        /* Images */
        .img-card { border-radius: 12px; overflow: hidden; border:1px solid rgba(255,255,255,0.03) }

        /* Recommendation card */
        .rec { border-left: 4px solid var(--accent); padding:18px; border-radius:10px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); }

        /* Small animations */
        @keyframes floatUp { 0%{transform:translateY(6px)} 50%{transform:translateY(-6px)} 100%{transform:translateY(6px)} }
        .float { animation: floatUp 6s ease-in-out infinite; }

        /* Hide default header */
        header {visibility:hidden}
        /* Make tab labels red */
        [role="tab"] { color: #ff4d4f !important; font-weight:700 }
        /* Sidebar custom label class */
        .sidebar-label { color: #ff4d4f; font-weight:700; margin-bottom:6px }
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION HISTORY INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. LOGIC ENGINE ---
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    try:
        model.load_state_dict(torch.load("carbon_bloom_model.pth", map_location=device))
    except:
        return None, device
    model.to(device)
    model.eval()
    return model, device

def get_recommendation(density):
    if density < 15:
        return {
            "title": "Healthy Ecosystem 🌿",
            "bg": "#E8F5E9",
            "border": "#2E7D32",
            "body": "Vegetation density is optimal. The area actively reduces local temperatures.",
            "steps": ["Protect existing canopy.", "Monitor for invasive species."]
        }
    elif density < 40:
        return {
            "title": "Moderate Urbanization 🏗️",
            "bg": "#FFF3E0",
            "border": "#EF6C00",
            "body": "Balance is shifting. Concrete heat absorption is noticeable but manageable.",
            "steps": ["Implement roadside bioswales.", "Mandate 20% green cover for new builds."]
        }
    elif density < 70:
        return {
            "title": "High Heat Risk ⚠️",
            "bg": "#FFEBEE",
            "border": "#C62828",
            "body": "Critical lack of biomass. Significant Urban Heat Island (UHI) effect detected.",
            "steps": ["Deploy Cool Roofs (High Albedo).", "Install vertical gardens on south-facing walls."]
        }
    else:
        return {
            "title": "Critical Concrete Desert 🚨",
            "bg": "#FFEBEE",
            "border": "#B71C1C",
            "body": "Environment is hostile to biodiversity and retains dangerous heat levels at night.",
            "steps": ["Emergency Pocket Park creation.", "Replace asphalt with permeable pavers immediately."]
        }

# --- 5. MAIN UI LAYOUT ---
def main():
    model, device = load_model()
    
    # --- HERO SECTION (Custom HTML for Branding) ---
    # --- HERO SECTION (Polished) ---
    st.markdown('<div class="hero card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 5])
    with c1:
        st.markdown('<div class="brand">', unsafe_allow_html=True)
        st.markdown('<div class="logo-badge float">🌿</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="display:block">', unsafe_allow_html=True)
        st.markdown('<h1 class="title">CarbonBloom Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Precision urban heat and vegetative intelligence for resilient cities — visualized, prioritized, actionable.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### ⚙️ Control Center")
        st.markdown('<div class="sidebar-label">Upload Satellite Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"]) 
        if uploaded_file is not None:
            st.markdown(f'<div style="color:#ff4d4f; font-weight:700; margin-top:8px;">{uploaded_file.name}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎛️ Calibration")
        st.markdown('<div class="sidebar-label">🔴 AI Detection Power</div>', unsafe_allow_html=True)
        sensitivity = st.slider("", 0.0, 1.0, 0.30, 0.05, key='sensitivity')
        st.markdown('<div class="sidebar-label">🟢 Nature Protection</div>', unsafe_allow_html=True)
        green_threshold = st.slider("", 0, 100, 40, 5, key='green_threshold')
        
        st.markdown("---")
        st.markdown('<div style="background: rgba(56,239,125,0.06); padding:10px; border-radius:8px; color:#bfffdc; border:1px solid rgba(56,239,125,0.18);">💡 <strong>Pro Tip:</strong> <span style="color:#bfffdc">Use \"Nature Protection\" to shield dark gardens from being marked as concrete.</span></div>', unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2 = st.tabs(["🚀 Analysis Dashboard", "📂 Project History"])

    with tab1:
        if uploaded_file is not None:
            # PROCESS
            image = Image.open(uploaded_file).convert('RGB')
            orig_w, orig_h = image.size
            img_array = np.array(image)

            # INFERENCE
            input_img = cv2.resize(img_array, (512, 512))
            input_tensor = input_img.astype('float32') / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(input_tensor)
                pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

            # HYBRID LOGIC
            binary_mask = (pred > sensitivity).astype(np.uint8)
            full_size_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lower_green = np.array([25, green_threshold, green_threshold]) 
            upper_green = np.array([90, 255, 255])
            green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
            
            final_mask = full_size_mask.copy()
            final_mask[green_mask > 0] = 0 

            # METRICS
            concrete_pixels = np.sum(final_mask)
            total_pixels = final_mask.size
            density = (concrete_pixels / total_pixels) * 100
            rec = get_recommendation(density)

            # OVERLAY
            overlay = img_array.copy()
            if np.sum(final_mask) > 0:
                red_layer = np.zeros_like(img_array)
                red_layer[:] = [255, 0, 0] 
                mask_indices = (final_mask == 1)
                if mask_indices.any():
                    overlay[mask_indices] = cv2.addWeighted(
                        img_array[mask_indices], 0.6, red_layer[mask_indices], 0.4, 0
                    )

            # SAVE TO HISTORY
            current_id = f"{uploaded_file.name}-{density:.2f}"
            if not any(d['id'] == current_id for d in st.session_state.history):
                st.session_state.history.insert(0, {
                    "id": current_id, "time": datetime.datetime.now().strftime("%H:%M"),
                    "img": image, "overlay": overlay, "density": density, "rec": rec
                })

            # --- DISPLAY DASHBOARD ---
            st.markdown('<div class="animate">', unsafe_allow_html=True)

            # 1. Polished Metrics Row
            cols = st.columns(3)
            with cols[0]:
                st.markdown(f"""
                <div class="metric card">
                    <h4>🧱 Concrete Density</h4>
                    <div class="metric-value">{density:.1f}%</div>
                    <div class="metric-sub">Share of area flagged as hard surfaces</div>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                temp = 25 + (density * 0.15)
                st.markdown(f"""
                <div class="metric card">
                    <h4>🌡️ Surface Temp (Est.)</h4>
                    <div class="metric-value">{temp:.1f}°C</div>
                    <div class="metric-sub">Approximate daytime surface temperature</div>
                </div>
                """, unsafe_allow_html=True)
            with cols[2]:
                co2 = max(0, 100 - density)
                st.markdown(f"""
                <div class="metric card">
                    <h4>🌲 CO₂ Offset Potential</h4>
                    <div class="metric-value">{co2:.0f} pts</div>
                    <div class="metric-sub">Higher is better — greener areas reduce emissions</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 2. Visuals Row with download actions
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🛰️ Satellite Feed")
                st.image(image, use_container_width=True)
                buf_orig = BytesIO()
                image.save(buf_orig, format='PNG')
                buf_orig.seek(0)
                b64_orig = base64.b64encode(buf_orig.getvalue()).decode()
                href_orig = f"data:image/png;base64,{b64_orig}"
                st.markdown(f'<a class="download-link" href="{href_orig}" download="{uploaded_file.name}">Download Original Image</a>', unsafe_allow_html=True)
            with c2:
                st.markdown("#### 🔥 Heat Risk Detection")
                st.image(overlay, use_container_width=True)
                buf = BytesIO()
                Image.fromarray(overlay).save(buf, format='PNG')
                buf.seek(0)
                b64_overlay = base64.b64encode(buf.getvalue()).decode()
                href_overlay = f"data:image/png;base64,{b64_overlay}"
                st.markdown(f'<a class="download-link" href="{href_overlay}" download="{uploaded_file.name}_overlay.png">Download Detection Overlay</a>', unsafe_allow_html=True)

            # 3. Recommendation Card (Styled HTML)
            st.markdown(f"""
            <div class="rec" style="margin-top:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
                    <div>
                        <h3 style="margin:0; color:#e9fff6;">{rec['title']}</h3>
                        <p style="margin:6px 0 0 0; color:var(--muted)">{rec['body']}</p>
                    </div>
                    <div style="text-align:right; color:var(--muted); font-size:13px">Score: {density:.1f}%</div>
                </div>
                <div style="margin-top:12px;">
                    <strong style="color:#eafff4">Interventions</strong>
                    <ol style="margin:8px 0 0 16px; color:#d8f6ee">
                        <li>{rec['steps'][0]}</li>
                        <li>{rec['steps'][1]}</li>
                    </ol>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # EMPTY STATE (Welcome Screen)
            st.markdown("""
            <div class="card" style="text-align:center;">
                <h2 style="margin-bottom:6px;">Welcome to CarbonBloom Pro</h2>
                <p style="color: #cfeee6; font-size: 1.05rem; margin-top:6px;">Turn satellite data into prioritized, climate-smart interventions — fast.</p>
                <div style="margin-top:18px; display:flex; justify-content:center; gap:12px;">
                    <div style="max-width:520px; text-align:left;">
                        <ul style="color:#d8f6ee; line-height:1.5;">
                            <li>High-precision heat-risk mapping</li>
                            <li>Actionable engineering recommendations</li>
                            <li>Project history and easy export</li>
                            </ul>
                        </div>
                    </div>
                    <div style="margin-top:16px;">
                        <p style="color:var(--muted); font-size:0.95rem;">Get started by uploading a satellite image in the sidebar.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📜 Project Timeline")
        if len(st.session_state.history) == 0:
            st.markdown('<div style="background: rgba(173,216,230,0.06); padding:12px; border-radius:8px; color:#bfe8ff; border:1px solid rgba(173,216,230,0.18);">No analyses run in this session yet.</div>', unsafe_allow_html=True)
        else:
            for item in st.session_state.history:
                with st.expander(f"📍 Analysis at {item['time']} | Score: {item['density']:.1f}%"):
                    hc1, hc2 = st.columns([1, 2])
                    with hc1:
                        st.image(item['overlay'], caption="Detection Map")
                    with hc2:
                        st.write(f"**Diagnosis:** {item['rec']['title']}")
                        st.write(f"**Intervention:** {item['rec']['steps'][0]}")

if __name__ == "__main__":
    main()