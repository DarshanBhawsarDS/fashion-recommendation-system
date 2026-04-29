import streamlit as st
import numpy as np
import pickle
import sqlite3
import hashlib
import os
import io
import base64
import json
from datetime import datetime
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VOGUE.AI — Fashion Recommender",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
# INJECTED CSS  — editorial white fashion theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ════════════════════════════════════════
   SOFT LUXURY LIGHT THEME
════════════════════════════════════════ */

:root {

    --bg:             #f7f5f2;
    --bg-card:        #ffffff;
    --bg-subtle:      #f3ede7;

    --accent:         #c98f65;
    --accent-rose:    #d9a07f;
    --accent-warm:    #e7c9b0;
    --accent-light:   rgba(201,143,101,0.10);

    --text-primary:   #5f5248;
    --text-secondary: #7b6f66;
    --text-muted:     #a69b92;

    --border:         #e6ddd2;
    --border-strong:  #d8ccc0;

    --shadow-sm:      0 2px 8px rgba(201,143,101,0.05);
    --shadow-md:      0 8px 24px rgba(201,143,101,0.08);
    --shadow-lg:      0 20px 50px rgba(201,143,101,0.12);

    --success:        #7da988;
    --warn:           #d28b72;

    --radius-sm:      8px;
    --radius-md:      16px;
    --radius-lg:      24px;
    --radius-pill:    999px;
}

/* ════════════════════════════════════════
   FORCE LIGHT BACKGROUND
════════════════════════════════════════ */

html,
body,
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section[data-testid="stMain"],
section[data-testid="stMain"] > div,
.main,
.block-container,
[data-testid="stVerticalBlockBorderWrapper"] {

    background: #f7f5f2 !important;
    color: var(--text-primary) !important;
}

/* remove any dark streamlit layers */
div[data-testid="stDecoration"] {
    display: none !important;
}

/* ════════════════════════════════════════
   HIDE STREAMLIT UI
════════════════════════════════════════ */

#MainMenu,
footer,
header,
[data-testid="stToolbar"] {
    display: none !important;
}

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */

[data-testid="stSidebar"] {

    background: #fffaf6 !important;

    border-right: 1px solid var(--border) !important;

    box-shadow: 4px 0 24px rgba(201,143,101,0.05) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
}

/* ════════════════════════════════════════
   TYPOGRAPHY
════════════════════════════════════════ */

h1, h2, h3, h4 {

    font-family: 'Fraunces', serif !important;

    color: var(--text-primary) !important;

    font-weight: 700 !important;
}

p,
span,
div,
label,
li {

    font-family: 'Outfit', sans-serif !important;

    color: var(--text-secondary) !important;
}

/* markdown text */
[data-testid="stMarkdownContainer"] * {
    color: var(--text-secondary) !important;
}

/* ════════════════════════════════════════
   INPUTS
════════════════════════════════════════ */

input,
textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {

    background: #ffffff !important;

    border: 1.5px solid var(--border-strong) !important;

    border-radius: var(--radius-sm) !important;

    color: var(--text-primary) !important;

    box-shadow: var(--shadow-sm) !important;

    padding: 0.7rem 1rem !important;

    font-family: 'Outfit', sans-serif !important;
}

input:focus,
textarea:focus {

    border-color: var(--accent) !important;

    box-shadow:
        0 0 0 4px rgba(201,143,101,0.12) !important;

    outline: none !important;
}

/* ════════════════════════════════════════
   BUTTONS
════════════════════════════════════════ */

.stButton > button {

    background: linear-gradient(
        135deg,
        #d9a07f,
        #e7c9b0
    ) !important;

    border: none !important;

    color: white !important;

    border-radius: var(--radius-pill) !important;

    font-family: 'Outfit', sans-serif !important;

    font-weight: 600 !important;

    letter-spacing: 0.08em !important;

    padding: 0.7rem 1.5rem !important;

    box-shadow:
        0 8px 22px rgba(201,143,101,0.18) !important;

    transition: all 0.25s ease !important;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 12px 30px rgba(201,143,101,0.24) !important;
}

/* ════════════════════════════════════════
   FILE UPLOADER
════════════════════════════════════════ */

[data-testid="stFileUploader"] {

    background: #fffdfb !important;

    border: 2px dashed var(--border-strong) !important;

    border-radius: var(--radius-lg) !important;

    padding: 2rem !important;

    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stFileUploader"]:hover {

    border-color: var(--accent) !important;

    background: #fff8f3 !important;
}

/* ════════════════════════════════════════
   IMAGES
════════════════════════════════════════ */

[data-testid="stImage"] img {

    border-radius: var(--radius-md) !important;

    border: 1px solid var(--border) !important;

    box-shadow: var(--shadow-md) !important;

    transition: all 0.3s ease !important;
}

[data-testid="stImage"] img:hover {

    transform: translateY(-4px);

    box-shadow: var(--shadow-lg) !important;
}

/* ════════════════════════════════════════
   TABS
════════════════════════════════════════ */

[data-baseweb="tab-list"] {

    background: #ffffff !important;

    border-radius: var(--radius-pill) !important;

    border: 1px solid var(--border) !important;

    padding: 0.25rem !important;

    box-shadow: var(--shadow-sm) !important;
}

[data-baseweb="tab"] {

    border-radius: var(--radius-pill) !important;

    color: var(--text-muted) !important;

    font-weight: 600 !important;

    transition: all 0.2s ease !important;
}

[aria-selected="true"][data-baseweb="tab"] {

    background: linear-gradient(
        135deg,
        #d9a07f,
        #e7c9b0
    ) !important;

    color: white !important;

    box-shadow:
        0 6px 18px rgba(201,143,101,0.18) !important;
}

/* ════════════════════════════════════════
   ALERTS
════════════════════════════════════════ */

[data-testid="stAlert"] {

    background: #fffaf6 !important;

    border: 1px solid var(--border) !important;

    border-radius: var(--radius-md) !important;

    color: var(--text-secondary) !important;

    box-shadow: var(--shadow-sm) !important;
}

/* ════════════════════════════════════════
   METRICS
════════════════════════════════════════ */

[data-testid="stMetric"] {

    background: #ffffff !important;

    border-radius: var(--radius-md) !important;

    border: 1px solid var(--border) !important;

    box-shadow: var(--shadow-sm) !important;

    padding: 1rem !important;
}

[data-testid="stMetricLabel"] {

    color: var(--text-muted) !important;

    text-transform: uppercase !important;

    letter-spacing: 0.08em !important;

    font-size: 0.72rem !important;
}

[data-testid="stMetricValue"] {

    color: var(--text-primary) !important;

    font-family: 'Fraunces', serif !important;
}

/* ════════════════════════════════════════
   EXPANDERS
════════════════════════════════════════ */

[data-testid="stExpander"] {

    background: #ffffff !important;

    border: 1px solid var(--border) !important;

    border-radius: var(--radius-md) !important;

    box-shadow: var(--shadow-sm) !important;
}

/* ════════════════════════════════════════
   BRAND HEADER
════════════════════════════════════════ */

.brand-header {

    background:
        linear-gradient(
            180deg,
            #fffdfb 0%,
            #f7f5f2 100%
        );

    border-radius: 32px;

    padding: 4rem 2rem;

    text-align: center;

    border: 1px solid var(--border);

    box-shadow: var(--shadow-md);
}

.brand-title {

    font-family: 'Fraunces', serif;

    font-size: clamp(3rem, 6vw, 5.5rem);

    color: #b97f5c;

    letter-spacing: 0.12em;
}

.brand-subtitle {

    color: var(--text-muted);

    margin-top: 0.8rem;

    letter-spacing: 0.18em;

    font-style: italic;
}

.brand-line {

    width: 70px;
    height: 3px;

    border-radius: 999px;

    margin: 1.4rem auto 0;

    background: linear-gradient(
        90deg,
        #d9a07f,
        #e7c9b0
    );
}

/* ════════════════════════════════════════
   USER PILL
════════════════════════════════════════ */

.user-pill {

    background: #ffffff;

    border: 1px solid var(--border);

    border-radius: var(--radius-pill);

    padding: 0.45rem 1rem;

    box-shadow: var(--shadow-sm);

    color: var(--text-primary) !important;

    font-weight: 600;
}

/* ════════════════════════════════════════
   REVIEW / HISTORY CARDS
════════════════════════════════════════ */

.review-card,
.history-card,
.review-form-wrap {

    background: #ffffff !important;

    border: 1px solid var(--border) !important;

    border-radius: var(--radius-md) !important;

    box-shadow: var(--shadow-sm) !important;

    padding: 1rem !important;
}

/* ════════════════════════════════════════
   SCROLLBAR
════════════════════════════════════════ */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #f3ede7;
}

::-webkit-scrollbar-thumb {

    background: #d9c5b4;

    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {

    background: #cfa688;
}

/* ════════════════════════════════════════
   REMOVE ANY REMAINING BLACK
════════════════════════════════════════ */

* {

    caret-color: #c98f65 !important;
}

button,
input,
textarea,
select {

    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE  — SQLite
# ─────────────────────────────────────────────
DB_PATH = "fashion_users.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT UNIQUE NOT NULL,
                password  TEXT NOT NULL,
                created   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                timestamp     TEXT NOT NULL,
                uploaded_img  BLOB,
                rec_filenames TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS cart (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                filename    TEXT NOT NULL,
                added_at    TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS reviews (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                username    TEXT NOT NULL,
                filename    TEXT NOT NULL,
                rating      INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                body        TEXT,
                created_at  TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)


init_db()

# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_user(username: str, password: str):
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password, created) VALUES (?, ?, ?)",
                (username.strip(), hash_pw(password), datetime.now().isoformat())
            )
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Username already taken."


def login_user(username: str, password: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username.strip(), hash_pw(password))
        ).fetchone()
    return dict(row) if row else None


def get_user_stats(user_id: int) -> dict:
    with get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) as c FROM history WHERE user_id=?", (user_id,)
        ).fetchone()["c"]
        last = conn.execute(
            "SELECT timestamp FROM history WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)
        ).fetchone()
    return {"total": count, "last": last["timestamp"][:10] if last else "—"}


def save_history(user_id: int, img: Image.Image, rec_filenames: list):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    blob = buf.getvalue()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO history (user_id, timestamp, uploaded_img, rec_filenames) VALUES (?,?,?,?)",
            (user_id, datetime.now().isoformat(), blob, json.dumps(rec_filenames))
        )


def get_history(user_id: int, limit: int = 20) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]

# ─────────────────────────────────────────────
# CART HELPERS
# ─────────────────────────────────────────────

def cart_add(user_id: int, filename: str):
    """Add item to cart. Prevents duplicates per user."""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM cart WHERE user_id=? AND filename=?",
            (user_id, filename)
        ).fetchone()
        if existing:
            return False, "Already in cart."
        conn.execute(
            "INSERT INTO cart (user_id, filename, added_at) VALUES (?,?,?)",
            (user_id, filename, datetime.now().isoformat())
        )
    return True, "Added to cart."


def cart_remove(cart_id: int, user_id: int):
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM cart WHERE id=? AND user_id=?",
            (cart_id, user_id)
        )


def cart_clear(user_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM cart WHERE user_id=?", (user_id,))


def get_cart(user_id: int) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM cart WHERE user_id=? ORDER BY id DESC",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def cart_count(user_id: int) -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) as c FROM cart WHERE user_id=?", (user_id,)
        ).fetchone()["c"]

# ─────────────────────────────────────────────
# REVIEW HELPERS
# ─────────────────────────────────────────────

def add_review(user_id: int, username: str, filename: str, rating: int, body: str):
    """Post or update a review (one per user per item)."""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM reviews WHERE user_id=? AND filename=?",
            (user_id, filename)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE reviews SET rating=?, body=?, created_at=? WHERE user_id=? AND filename=?",
                (rating, body.strip(), datetime.now().isoformat(), user_id, filename)
            )
            return True, "Review updated."
        conn.execute(
            "INSERT INTO reviews (user_id, username, filename, rating, body, created_at) VALUES (?,?,?,?,?,?)",
            (user_id, username, filename, rating, body.strip(), datetime.now().isoformat())
        )
    return True, "Review posted."


def get_reviews_for_item(filename: str) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE filename=? ORDER BY id DESC",
            (filename,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_reviews_by_user(user_id: int) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE user_id=? ORDER BY id DESC",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_reviewed_items() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT filename FROM reviews ORDER BY filename"
        ).fetchall()
    return [r["filename"] for r in rows]


def avg_rating(filename: str):
    """Returns float average or None. Compatible with Python 3.9+."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT AVG(rating) as avg FROM reviews WHERE filename=?", (filename,)
        ).fetchone()
    return round(row["avg"], 1) if row and row["avg"] else None

# ─────────────────────────────────────────────
# MODEL  (cached)
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def extract_features(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t).squeeze().cpu().numpy()
    return out / (np.linalg.norm(out) + 1e-8)


def recommend(feat: np.ndarray, feature_list: np.ndarray) -> np.ndarray:
    n_neighbors = min(6, len(feature_list))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(feature_list)
    dists, indices = nn.kneighbors([feat])
    # skip index 0 (the query itself) if we have extras; otherwise return all
    result = indices[0]
    return result[1:] if len(result) > 1 else result


# ─────────────────────────────────────────────
# LOAD EMBEDDINGS
# ─────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    if not os.path.exists("embeddings.pkl") or not os.path.exists("filenames.pkl"):
        return None, None
    try:
        feats = np.array(pickle.load(open("embeddings.pkl", "rb")))
        fnames = pickle.load(open("filenames.pkl", "rb"))
        return feats, fnames
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        return None, None


features, filenames = load_embeddings()

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# last_recs: persisted across tab switches within the session
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []

# cart_msg: per-item feedback dict  { index: (ok, msg) }
if "cart_msgs" not in st.session_state:
    st.session_state.cart_msgs = {}

# track active review target (set when user clicks ✎ Review on recommend tab)
if "review_target" not in st.session_state:
    st.session_state.review_target = None

# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def _safe_toast(msg: str):
    """st.toast without the icon= kwarg for Streamlit < 1.31 compatibility."""
    try:
        st.toast(msg)
    except Exception:
        st.success(msg)


def stars_html(rating, size="0.9rem"):
    full = int(round(rating))
    s = "★" * full + "☆" * (5 - full)
    return f'<span style="font-size:{size};color:#7c6ff7;letter-spacing:0.05em;">{s}</span>'


def _seed_last_recs_from_cart(user_id: int):
    """
    Populate last_recs from the user's cart on first load so the Review tab
    always has items to show even before a fresh search.
    """
    if not st.session_state.last_recs:
        cart_items = get_cart(user_id)
        st.session_state.last_recs = [i["filename"] for i in cart_items]

# ─────────────────────────────────────────────
# BRAND HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div class="brand-title">VOGUE·AI</div>
    <div class="brand-subtitle">Visual fashion intelligence</div>
    <div class="brand-line"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTH SCREEN
# ─────────────────────────────────────────────

def auth_screen():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        tab_login, tab_reg = st.tabs(["Sign In", "Register"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user", placeholder="your username")
            password = st.text_input("Password", type="password", key="login_pw", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("ENTER →", key="btn_login", use_container_width=True):
                if not username or not password:
                    st.error("Fill in both fields.")
                else:
                    u = login_user(username, password)
                    if u:
                        st.session_state.user = u
                        st.session_state.last_recs = []
                        st.session_state.cart_msgs = {}
                        st.session_state.review_target = None
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

        with tab_reg:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user = st.text_input("Choose username", key="reg_user", placeholder="your username")
            new_pw   = st.text_input("Choose password", type="password", key="reg_pw", placeholder="min 6 chars")
            new_pw2  = st.text_input("Confirm password", type="password", key="reg_pw2", placeholder="repeat")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("CREATE ACCOUNT →", key="btn_reg", use_container_width=True):
                if len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pw != new_pw2:
                    st.error("Passwords don't match.")
                elif not new_user.strip():
                    st.error("Enter a username.")
                else:
                    ok, msg = register_user(new_user, new_pw)
                    if ok:
                        st.success(msg + " Sign in now.")
                    else:
                        st.error(msg)

# ─────────────────────────────────────────────
# CART TAB
# ─────────────────────────────────────────────

def render_cart_tab(user: dict):
    cart_items = get_cart(user["id"])
    total = len(cart_items)

    m1, m2, _ = st.columns([1, 1, 3])
    with m1:
        st.metric("Items in Cart", total)
    with m2:
        # placeholder for a future "total price" or "total pieces" metric
        st.metric("Total Pieces", total)

    st.markdown("---")

    if not cart_items:
        st.markdown(
            '<div class="cart-empty">✦ Your cart is empty<br>'
            '<span style="font-size:0.68rem;color:#2a2218;">'
            'Add pieces from your recommendations</span></div>',
            unsafe_allow_html=True
        )
        return

    _, act_r = st.columns([5, 1])
    with act_r:
        if st.button("CLEAR ALL", key="cart_clear_all"):
            cart_clear(user["id"])
            st.success("Cart cleared.")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    for item in cart_items:
        fname = item["filename"]
        added = item["added_at"][:16].replace("T", "  ")
        short_name = os.path.basename(fname)

        c_img, c_info, c_action = st.columns([1, 5, 1.5])

        with c_img:
            if os.path.exists(fname):
                st.image(fname, use_container_width=True)
            else:
                st.markdown(
                    '<div style="width:72px;height:72px;background:#1a1a1a;'
                    'border:1px solid #1e1a12;border-radius:2px;display:flex;'
                    'align-items:center;justify-content:center;'
                    'font-size:0.6rem;color:#3a3020;">NO IMG</div>',
                    unsafe_allow_html=True
                )

        with c_info:
            st.markdown(
                f'<div class="cart-item-name">{short_name}</div>'
                f'<div class="cart-item-added">Added  {added}</div>',
                unsafe_allow_html=True
            )
            avg = avg_rating(fname)
            if avg:
                st.markdown(
                    f'{stars_html(avg)} '
                    f'<span style="font-size:0.72rem;color:#b8924a;">{avg}</span>',
                    unsafe_allow_html=True
                )

        with c_action:
            if st.button("REMOVE", key=f"cart_rm_{item['id']}"):
                cart_remove(item["id"], user["id"])
                st.rerun()

        st.markdown('<hr style="margin:0.4rem 0;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# REVIEWS TAB
# ─────────────────────────────────────────────

def render_reviews_tab(user: dict):
    st.markdown("#### Write a Review")

    # Build the pool of reviewable items:
    # 1) anything from the most recent search
    # 2) items in the cart
    # 3) items the user has already reviewed (so they can update)
    cart_fnames   = [i["filename"] for i in get_cart(user["id"])]
    reviewed_fnames = [r["filename"] for r in get_reviews_by_user(user["id"])]
    recent_recs   = list(st.session_state.last_recs)  # copy to avoid mutation issues

    # Merge with order preserved, no duplicates
    seen = set()
    all_items = []
    for f in recent_recs + cart_fnames + reviewed_fnames:
        if f and f not in seen:
            seen.add(f)
            all_items.append(f)

    if not all_items:
        st.info("Upload an image to get recommendations, then come back here to review them.")
    else:
        st.markdown('<div class="review-form-wrap">', unsafe_allow_html=True)

        short_labels = {f: os.path.basename(f) for f in all_items}
        label_list   = list(short_labels.values())

        # If a review_target was set via the Recommend tab button, pre-select it
        default_idx = 0
        if st.session_state.review_target:
            target_label = short_labels.get(st.session_state.review_target)
            if target_label and target_label in label_list:
                default_idx = label_list.index(target_label)

        selected_label = st.selectbox(
            "Select item to review",
            options=label_list,
            index=default_idx,
            key="review_item_select"
        )
        # Map label back to full path
        selected_file = next((f for f, l in short_labels.items() if l == selected_label), None)

        if selected_file:
            rv_col_img, rv_col_form = st.columns([1, 3])
            with rv_col_img:
                if os.path.exists(selected_file):
                    st.image(selected_file, use_container_width=True)
                else:
                    st.caption("Image file not found on disk.")

            with rv_col_form:
                # Pre-fill if user already reviewed this item
                existing_rv = next(
                    (r for r in get_reviews_by_user(user["id"]) if r["filename"] == selected_file),
                    None
                )
                default_rating = existing_rv["rating"] if existing_rv else 5
                default_body   = existing_rv["body"]   if existing_rv else ""

                rating = st.slider(
                    "Rating",
                    min_value=1, max_value=5,
                    value=default_rating,
                    key="review_rating"
                )
                st.markdown(
                    f'{stars_html(rating, size="1.2rem")}',
                    unsafe_allow_html=True
                )
                review_text = st.text_area(
                    "Your review (optional)",
                    value=default_body,
                    placeholder="Silhouette, fabric, styling notes…",
                    max_chars=500,
                    key="review_body",
                    height=100
                )
                btn_label = "UPDATE REVIEW →" if existing_rv else "SUBMIT REVIEW →"
                if st.button(btn_label, key="btn_submit_review"):
                    ok, msg = add_review(
                        user["id"], user["username"],
                        selected_file, rating, review_text
                    )
                    if ok:
                        st.success(msg)
                        st.session_state.review_target = None
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── My Reviews
    with st.expander("My Reviews", expanded=True):
        my_review_list = get_reviews_by_user(user["id"])
        if not my_review_list:
            st.caption("You haven't reviewed anything yet.")
        else:
            for rv in my_review_list:
                short = os.path.basename(rv["filename"])
                ts    = rv["created_at"][:16].replace("T", "  ")
                body  = rv["body"] or ""

                rv_c1, rv_c2 = st.columns([1, 5])
                with rv_c1:
                    if os.path.exists(rv["filename"]):
                        st.image(rv["filename"], use_container_width=True)
                    else:
                        st.caption("—")
                with rv_c2:
                    st.markdown(
                        f'<div class="review-card">'
                        f'<div class="review-header">'
                        f'<span class="review-author">{rv["username"]}</span>'
                        f'{stars_html(rv["rating"])}'
                        f'</div>'
                        f'<div class="review-item-name">{short}</div>'
                        f'<div class="review-body">{body if body else "<em>No comment</em>"}</div>'
                        f'<div class="review-ts">{ts}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('<hr style="margin:0.3rem 0;">', unsafe_allow_html=True)

    st.markdown("---")

    # ── Community Reviews
    with st.expander("Community Reviews", expanded=False):
        reviewed_items = get_all_reviewed_items()
        if not reviewed_items:
            st.caption("No community reviews yet.")
        else:
            for fname in reviewed_items:
                short  = os.path.basename(fname)
                all_rv = get_reviews_for_item(fname)
                avg    = avg_rating(fname)

                st.markdown(
                    f'<div style="font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;'
                    f'color:#7a6e5e;margin-bottom:0.4rem;">'
                    f'{short} '
                    f'{stars_html(avg) if avg else ""}'
                    f'<span style="color:#b8924a;"> {avg or ""}</span>'
                    f' · {len(all_rv)} review{"s" if len(all_rv) != 1 else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                for rv in all_rv:
                    body = rv["body"] or ""
                    ts   = rv["created_at"][:16].replace("T", "  ")
                    st.markdown(
                        f'<div class="review-card">'
                        f'<div class="review-header">'
                        f'<span class="review-author">{rv["username"]}</span>'
                        f'{stars_html(rv["rating"])}'
                        f'</div>'
                        f'<div class="review-body">{body if body else "<em>No comment</em>"}</div>'
                        f'<div class="review-ts">{ts}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown("---")

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main_app():
    user = st.session_state.user

    # Seed last_recs from cart on first entry so Reviews tab has items available
    _seed_last_recs_from_cart(user["id"])

    # ── Top bar
    top_l, top_r = st.columns([7, 1])
    with top_l:
        n_cart = cart_count(user["id"])
        cart_badge = (
            f' <span class="cart-badge">{n_cart}</span>' if n_cart else ""
        )
        st.markdown(
            f'<span class="user-pill">⬡ {user["username"]}{cart_badge}</span>',
            unsafe_allow_html=True
        )
    with top_r:
        if st.button("Sign out", key="logout"):
            st.session_state.user      = None
            st.session_state.last_recs = []
            st.session_state.cart_msgs = {}
            st.session_state.review_target = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    tab_rec, tab_hist, tab_cart, tab_reviews = st.tabs([
        "Recommend", "My History", "Cart", "Reviews"
    ])

    # ─── TAB 1: Recommender ──────────────────
    with tab_rec:
        if features is None:
            st.error(
                "⚠️  embeddings.pkl / filenames.pkl not found. "
                "Run `build_torch.py` to generate embeddings first."
            )
            st.stop()

        st.markdown("#### Upload a garment image")
        uploaded = st.file_uploader(
            "PNG · JPG · JPEG",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded:
            try:
                img = Image.open(uploaded).convert("RGB")
            except Exception as e:
                st.error(f"Could not open image: {e}")
                st.stop()

            img_col, _, info_col = st.columns([1.2, 0.2, 2])
            with img_col:
                st.image(img, caption="Your upload", use_container_width=True)
            with info_col:
                st.markdown("#### Finding similar pieces…")
                with st.spinner("Extracting visual features"):
                    try:
                        feat    = extract_features(img)
                        indices = recommend(feat, features)
                    except Exception as e:
                        st.error(f"Feature extraction failed: {e}")
                        st.stop()

                rec_names = [filenames[i] for i in indices]
                # Persist for cart / review access across tab switches
                st.session_state.last_recs = rec_names

                st.success(f"Found {len(rec_names)} similar items")
                st.markdown(
                    "<span style='font-size:0.75rem;color:#5a5040;"
                    "letter-spacing:0.1em;text-transform:uppercase;'>"
                    "Results based on ResNet-50 visual embeddings</span>",
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown("#### Similar Items")

            cols = st.columns(len(rec_names)) if rec_names else []

            for i, col in enumerate(cols):
                with col:
                    fname = rec_names[i]

                    # ── Image (guard missing file)
                    if os.path.exists(fname):
                        st.image(fname, use_container_width=True)
                    else:
                        st.markdown(
                            f'<div style="background:#1a1a1a;border:1px solid #1e1a12;'
                            f'border-radius:2px;padding:1rem;text-align:center;'
                            f'font-size:0.65rem;color:#3a3020;">FILE<br>NOT FOUND<br>'
                            f'{os.path.basename(fname)}</div>',
                            unsafe_allow_html=True
                        )

                    # ── Average rating badge
                    avg = avg_rating(fname)
                    if avg:
                        st.markdown(
                            f'<div style="text-align:center;margin-bottom:0.3rem;">'
                            f'{stars_html(avg, "0.75rem")} '
                            f'<span style="font-size:0.68rem;color:#b8924a;">{avg}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    # ── Cart button
                    #    Use a unique key that includes the filename hash to survive reruns
                    fname_key = abs(hash(fname)) % 100000
                    if st.button("＋ Cart", key=f"cadd_{i}_{fname_key}"):
                        ok, msg = cart_add(user["id"], fname)
                        st.session_state.cart_msgs[i] = (ok, msg)

                    # Show inline cart feedback without a full rerun
                    if i in st.session_state.cart_msgs:
                        ok_flag, msg_txt = st.session_state.cart_msgs[i]
                        css_class = "inline-success" if ok_flag else "inline-warn"
                        st.markdown(
                            f'<div class="{css_class}">{msg_txt}</div>',
                            unsafe_allow_html=True
                        )

                    # ── Review button
                    if st.button("✎ Review", key=f"rev_{i}_{fname_key}"):
                        # Pre-select this item in the Review tab
                        st.session_state.review_target = fname
                        # Ensure item is at the front of last_recs so the selectbox finds it
                        if fname in st.session_state.last_recs:
                            st.session_state.last_recs.remove(fname)
                        st.session_state.last_recs.insert(0, fname)
                        st.info("Switch to the **Reviews** tab to write your review.")

            # Save to history (deduplicate: don't save same upload twice in one session)
            if "last_saved_upload" not in st.session_state or \
               st.session_state.last_saved_upload != uploaded.name:
                save_history(user["id"], img, rec_names)
                st.session_state.last_saved_upload = uploaded.name
                st.caption("✓ Saved to your history")

    # ─── TAB 2: History ─────────────────────
    with tab_hist:
        stats = get_user_stats(user["id"])

        m1, m2, _ = st.columns([1, 1, 3])
        with m1:
            st.metric("Total Searches", stats["total"])
        with m2:
            st.metric("Last Search", stats["last"])

        st.markdown("---")

        history = get_history(user["id"])
        if not history:
            st.info("No searches yet. Upload an image to get started.")
        else:
            for entry in history:
                ts   = entry["timestamp"][:19].replace("T", "  ")
                recs = json.loads(entry["rec_filenames"]) if entry["rec_filenames"] else []

                with st.expander(f"🕐  {ts}  —  {len(recs)} recommendations", expanded=False):
                    h_l, h_r = st.columns([1, 3])
                    with h_l:
                        if entry["uploaded_img"]:
                            try:
                                thumb = Image.open(io.BytesIO(entry["uploaded_img"]))
                                st.image(thumb, caption="Uploaded", use_container_width=True)
                            except Exception:
                                st.caption("Could not decode thumbnail.")
                    with h_r:
                        if recs:
                            rec_cols = st.columns(min(5, len(recs)))
                            for j, rc in enumerate(rec_cols):
                                with rc:
                                    if j < len(recs):
                                        r_fname = recs[j]
                                        if os.path.exists(r_fname):
                                            st.image(r_fname, use_container_width=True)
                                        else:
                                            st.caption("—")
                                        # Quick-add to cart from history
                                        if st.button("＋", key=f"hcart_{entry['id']}_{j}"):
                                            ok, msg = cart_add(user["id"], r_fname)
                                            _safe_toast("✓ Added" if ok else msg)
                                            st.rerun()
                        else:
                            st.caption("No recommendation data.")

    # ─── TAB 3: Cart ────────────────────────
    with tab_cart:
        render_cart_tab(user)

    # ─── TAB 4: Reviews ─────────────────────
    with tab_reviews:
        render_reviews_tab(user)


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if st.session_state.user is None:
    auth_screen()
else:
    main_app()