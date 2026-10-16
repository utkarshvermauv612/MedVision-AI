import streamlit as st


def _change_theme():
    """Update the application theme when the toggle changes."""
    st.session_state.ui_theme = (
        "Dark"
        if st.session_state.header_theme_toggle
        else "Light"
    )


def render_top_header():
    """Render a compact glass header with a single theme toggle."""

    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "Dark"

    if "header_theme_toggle" not in st.session_state:
        st.session_state.header_theme_toggle = (
            st.session_state.ui_theme == "Dark"
        )

    # Synchronize the widget with the currently selected theme.
    st.session_state.header_theme_toggle = (
        st.session_state.ui_theme == "Dark"
    )

    brand_column, toggle_column = st.columns(
        [5, 1],
        vertical_alignment="center",
        gap="small",
    )

    with brand_column:
        # Single-line HTML prevents stray closing tags from appearing.
        brand_html = (
            '<div class="glass-header">'
            '<div class="header-brand">'
            '<div class="brand-icon">✚</div>'
            '</div>'
            '</div>'
        )

        st.markdown(
            brand_html,
            unsafe_allow_html=True,
        )

    with toggle_column:
        st.toggle(
            "Dark mode",
            key="header_theme_toggle",
            on_change=_change_theme,
            help="Switch between light and dark mode.",
        )


def page_header(title, subtitle, badge=None):
    """Render a page heading using native Streamlit components."""

    st.title(title)
    st.caption(subtitle)

    if badge:
        st.caption(f"✦ {badge}")


def section_header(title, description=None):
    """Render a section heading."""

    st.subheader(title)

    if description:
        st.caption(description)


def medical_disclaimer():
    """Display the medical research disclaimer."""

    st.warning(
        "Educational research prototype only. This application "
        "cannot diagnose pneumonia or replace professional medical "
        "evaluation."
    )