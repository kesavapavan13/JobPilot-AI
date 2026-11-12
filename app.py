"""JobPilot AI: Streamlit front end."""


import json
import logging
import streamlit as st
from pypdf import PdfReader
import db
from agent import MAX_ATTEMPTS, PipelineError, stream_pipeline, validate_inputs


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="JobPilot AI", page_icon="🧭", layout="wide")

STEP_LABELS = {
    "analyzer": "Analyzer: read the job description",
    "matcher": "Matcher: scored your resume",
    "tailor": "Tailor: rewrote bullets and cover letter",
    "reviewer": "Reviewer: fact-checked the draft",
    "save": "Tracker: saved the application",
}


def read_pdf(uploaded) -> str:
    reader = PdfReader(uploaded)
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def render_result(r: dict, key_prefix: str = "res"):
    match, tailored, review = r["match"], r["tailored"], r["review"]
    info = r["jd_info"]

    st.subheader(f"{info['role']} at {info['company']}")
    if review["approved"]:
        st.success(f"Reviewer approved the content after {r['attempts']} attempt(s). No unsupported claims found.")
    else:
        st.error(
            f"The reviewer still found unsupported claims after {r['attempts']} attempts. "
            "Do not send this content before you remove them yourself."
        )
        for claim in review["fabricated_claims"]:
            st.markdown(f"- {claim}")

    c1, c2, c3 = st.columns([1, 2, 2])
    c1.metric("Match score", f"{match['score']}/100")
    with c2:
        st.markdown("**Skills you have**")
        st.write(", ".join(match["matched_skills"]) or "None found")
    with c3:
        st.markdown("**Skills gap**")
        st.write(", ".join(match["missing_skills"]) or "None")

    tab_b, tab_c = st.tabs(["Resume bullets", "Cover letter"])
    with tab_b:
        bullets_md = "\n".join(f"- {b}" for b in tailored["bullets"])
        st.markdown(bullets_md)
        st.download_button("Download bullets", bullets_md, "bullets.md", key=f"{key_prefix}_b")
    with tab_c:
        st.write(tailored["cover_letter"])
        st.download_button("Download cover letter", tailored["cover_letter"], "cover_letter.txt", key=f"{key_prefix}_c")

    if r.get("saved_id"):
        st.caption(f"Saved to your tracker as application #{r['saved_id']}.")
    if r.get("save_error"):
        st.warning(f"Results were not saved to MySQL: {r['save_error']}")


def new_application_tab():
    left, right = st.columns(2)
    with left:
        st.markdown("#### Job description")
        jd = st.text_area("Paste the job description", height=320, label_visibility="collapsed",
                          placeholder="Paste the full job description here")
    with right:
        st.markdown("#### Your resume")
        pdf = st.file_uploader("Upload a PDF resume", type=["pdf"])
        resume_text = ""
        if pdf is not None:
            try:
                resume_text = read_pdf(pdf)
            except Exception as e:
                st.error(f"Could not read this PDF: {e}")
            if pdf is not None and not resume_text:
                st.warning("No text found. This PDF may be a scan. Paste your resume text below instead.")
        resume = st.text_area("Or paste your resume text", value=resume_text, height=200,
                              key=f"resume_{hash(resume_text)}")

    if st.button("Tailor my application", type="primary"):
        try:
            jd_clean, resume_clean = validate_inputs(jd, resume)
        except ValueError as e:
            st.error(str(e))
            return

        try:
            dup = db.find_duplicate(db.content_hash(jd_clean, resume_clean))
            if dup:
                st.info(f"You already ran this exact job and resume (#{dup['id']}, {dup['company']}). Running again anyway.")
        except Exception:
            pass  # tracker down; the pipeline reports it later

        st.session_state.pop("result", None)
        with st.status("Running agents...", expanded=True) as status:
            try:
                final = None
                for node, _, final in stream_pipeline(jd_clean, resume_clean):
                    label = STEP_LABELS[node]
                    if node == "reviewer":
                        rv = final["review"]
                        verdict = "approved" if rv["approved"] else f"rejected ({len(rv['fabricated_claims'])} unsupported claim(s))"
                        label += f": {verdict}, attempt {final['attempts']} of {MAX_ATTEMPTS}"
                    st.write(label)
                status.update(label="Finished", state="complete", expanded=False)
                st.session_state["result"] = final
            except (PipelineError, ValueError) as e:
                status.update(label="Failed", state="error")
                st.error(str(e))
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"Unexpected error: {e}")

    if st.session_state.get("result"):
        render_result(st.session_state["result"])


def history_tab():
    try:
        rows = db.list_applications()
    except Exception as e:
        st.error(f"Cannot reach MySQL. Check your .env settings and that the server is running.\n\n{e}")
        return
    if not rows:
        st.info("No applications yet. Run one from the New application tab.")
        return

    table = [
        {"ID": r["id"], "Company": r["company"], "Role": r["role"], "Score": r["match_score"],
         "Attempts": r["attempts"], "Verified": "Yes" if r["approved"] else "No",
         "Status": r["status"], "Date": r["created_at"]}
        for r in rows
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("#### Open an application")
    chosen = st.selectbox("Application", rows, format_func=lambda r: f"#{r['id']} {r['company']}: {r['role']}")
    if chosen:
        c1, c2, _ = st.columns([2, 1, 3])
        new_status = c1.selectbox("Status", db.STATUSES, index=db.STATUSES.index(chosen["status"])
                                  if chosen["status"] in db.STATUSES else 0, key=f"st_{chosen['id']}")
        if new_status != chosen["status"]:
            db.update_status(chosen["id"], new_status)
            st.rerun()
        if c2.button("Delete", key=f"del_{chosen['id']}"):
            db.delete_application(chosen["id"])
            st.rerun()
        if not chosen["approved"]:
            st.warning("This draft was not fully verified. Check it against your real experience.")
        st.markdown("**Bullets**")
        st.markdown("\n".join(f"- {b}" for b in chosen["bullets"]) or "_Not stored_")
        st.markdown("**Cover letter**")
        st.write(chosen["cover_letter"] or "_Not stored_")
        if chosen["missing_skills"]:
            st.markdown("**Skills gap:** " + ", ".join(chosen["missing_skills"]))


st.title("JobPilot AI")
st.caption("Paste a job description, add your resume, and get tailored bullets and a cover letter. "
           "A reviewer agent checks every claim against your original resume.")
tab_new, tab_hist = st.tabs(["New application", "History"])
with tab_new:
    new_application_tab()
with tab_hist:
    history_tab()
