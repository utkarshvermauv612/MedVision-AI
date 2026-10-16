import streamlit as st


def render_sidebar(model_available=True):
    """Render the MedVision AI sidebar."""

    with st.sidebar:
        st.markdown("## ✚ MedVision AI")

        st.caption("Medical Intelligence")

        st.divider()

        st.caption("WORKSPACE")

        selected_page = st.radio(
            "Navigation",
            options=[
                "Dashboard",
                "Analyze X-ray",
                "Model Performance",
                "About Project",
            ],
            key="app_navigation",
            label_visibility="collapsed",
        )

        st.divider()

        if model_available:
            st.success("Model files detected")
        else:
            st.warning(
                "Model files were not detected. "
                "Check your models directory."
            )

        st.divider()

        st.caption("MedVision AI")
        st.caption("Educational research prototype")

    return selected_page